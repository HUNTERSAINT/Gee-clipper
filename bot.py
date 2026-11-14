"""Telegram front end for the YouTube clipper."""
import asyncio
import logging
import re
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import date

from telegram import BotCommand, InputMediaPhoto, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PicklePersistence,
    filters,
)

import config as C
import core
import yt_dlp

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
# httpx logs full request URLs, and Telegram's URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("clipper.bot")

URL_RE = re.compile(r"(https?://\S+)")
JOB_LOCK = asyncio.Semaphore(1)  # one video at a time (CPU + free-tier limits)
WAITING = 0
JOB_STATES: dict[int, dict] = {}
CANCEL_FLAGS: dict[int, asyncio.Event] = {}


class JobCancelled(Exception):
    pass

HELP = (
    "Send me a YouTube link and I'll cut it into short vertical clips with "
    "captions, a hook title, a thumbnail and ready-to-paste hashtags.\n\n"
    "Add a number to choose how many: https://youtu.be/xxxx 5\n\n"
    "Settings:\n"
    "/clips 3 - how many clips\n"
    "/layout auto|crop|blur - auto follows the speaker's face\n"
    "/captions on|off\n"
    "/hook on|off - title banner at the start\n"
    "/music on|off - background music matched to each clip\n"
    "/grade on|off - automatic colour grading\n"
    "/settings - show current settings\n"
    "/panel - button-based settings\n"
    "/preset viral|podcast|educational|cinematic|clean\n"
    "/queue, /cancel, /history\n"
    "/watermark YourBrand (or off)\n"
    "/formats all|vertical - 9:16, 1:1 and 16:9 exports\n"
    "/silence on|off - trim long pauses\n"
    "/translate Spanish - translated subtitles\n"
    "/clearlogo - remove your uploaded logo\n\n"
    f"Videos up to {C.MAX_VIDEO_MINUTES} min. Send the same link again and "
    "it's instant (the transcript is cached)."
)


def get_opts(ctx: ContextTypes.DEFAULT_TYPE) -> core.Opts:
    o = ctx.user_data.setdefault("opts", core.Opts())
    defaults = core.Opts()
    # Migrate settings saved by older bot versions when new options are added.
    for key, value in vars(defaults).items():
        if not hasattr(o, key):
            setattr(o, key, value)
    return o


def summary(o: core.Opts) -> str:
    return (
        f"Clips: {o.clips}\nLayout: {o.layout}\n"
        f"Captions: {'on' if o.captions else 'off'}\n"
        f"Hook title: {'on' if o.hook else 'off'}\n"
        f"Music: {'on' if o.music else 'off'}\n"
        f"Colour grade: {'on' if o.grade else 'off'}"
        + f"\nCaption theme: {o.caption_style}"
        + f"\nWatermark: {o.watermark or 'off'}"
        + f"\nExports: {o.export_formats}"
        + f"\nSilence removal: {'on' if o.remove_silence else 'off'}"
        + f"\nOutro: {o.outro_text or 'off'}"
    )


def allowed(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    blocked = ctx.application.bot_data.setdefault("blocked_users", set())
    return update.effective_user.id not in blocked and (not C.ALLOWED_USERS or update.effective_user.id in C.ALLOWED_USERS)


def use_quota(uid: int, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    if uid in C.ADMIN_USERS or C.DAILY_LIMIT <= 0:
        return True
    usage = ctx.application.bot_data.setdefault("daily_usage", {})
    today = date.today().isoformat()
    day, n = usage.get(str(uid), (today, 0))
    if day != today:
        n = 0
    if n >= C.DAILY_LIMIT:
        return False
    usage[str(uid)] = (today, n + 1)
    return True


def friendly_error(e: Exception) -> str:
    s = str(e)
    low = s.lower()
    if "not a bot" in low or "sign in to confirm" in low:
        return ("YouTube blocked this server. Set YT_COOKIES_FILE "
                "(see README) and try again.")
    if "private video" in low:
        return "That video is private."
    if "unavailable" in low:
        return "That video is unavailable."
    if "js runtime" in low or "javascript runtime" in low:
        return "yt-dlp needs Deno or Node.js installed (see README)."
    return s[:300]


def build_caption(clip: dict) -> str:
    tags = " ".join("#" + t for t in clip["hashtags"])
    parts = [
        f"🎬 {clip['title']}",
        f"⏱ {core.fmt_ts(clip['start'])}-{core.fmt_ts(clip['end'])} · "
        f"⭐ {clip['score']:.0f}/10",
        clip["description"],
        tags,
    ]
    return "\n".join(p for p in parts if p)[:1000]

def build_caption_pack(clip: dict) -> str:
    tags = " ".join("#" + t for t in clip.get("hashtags", []))
    title = clip.get("title", "New clip")
    desc = clip.get("description", "")
    return (
        "📲 READY-TO-POST CAPTIONS\n\n"
        f"TIKTOK\n{title}\n{desc}\n{tags}\n\n"
        f"INSTAGRAM REELS\n{desc}\n\n{tags}\n\n"
        f"YOUTUBE SHORTS\n{title}\n{desc}\n{tags}"
    )[:3900]



def settings_keyboard(o: core.Opts):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Captions {'ON' if o.captions else 'OFF'}", callback_data="toggle:captions"),
         InlineKeyboardButton(f"Hook {'ON' if o.hook else 'OFF'}", callback_data="toggle:hook")],
        [InlineKeyboardButton(f"Music {'ON' if o.music else 'OFF'}", callback_data="toggle:music"),
         InlineKeyboardButton(f"Grade {'ON' if o.grade else 'OFF'}", callback_data="toggle:grade")],
        [InlineKeyboardButton("Auto", callback_data="layout:auto"),
         InlineKeyboardButton("Face crop", callback_data="layout:crop"),
         InlineKeyboardButton("Blur", callback_data="layout:blur")],
        [InlineKeyboardButton("1 clip", callback_data="clips:1"),
         InlineKeyboardButton("3 clips", callback_data="clips:3"),
         InlineKeyboardButton("6 clips", callback_data="clips:6")],
        [InlineKeyboardButton("Bold", callback_data="style:bold"),
         InlineKeyboardButton("Classic", callback_data="style:classic"),
         InlineKeyboardButton("Neon", callback_data="style:neon"),
         InlineKeyboardButton("Minimal", callback_data="style:minimal")],
        [InlineKeyboardButton(f"Exports: {o.export_formats}", callback_data="formats:toggle"),
         InlineKeyboardButton(f"Silence trim {'ON' if o.remove_silence else 'OFF'}", callback_data="silence:toggle")],
    ])


async def panel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    o = get_opts(ctx)
    await update.message.reply_text("⚙️ Gee Clipper controls\n\n" + summary(o),
                                    reply_markup=settings_keyboard(o))


async def settings_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    o = get_opts(ctx)
    action, value = q.data.split(":", 1)
    if action == "toggle" and value in ("captions", "hook", "music", "grade"):
        setattr(o, value, not getattr(o, value))
    elif action == "layout" and value in ("auto", "crop", "blur"):
        o.layout = value
    elif action == "clips" and value.isdigit():
        o.clips = max(1, min(int(value), C.MAX_CLIPS))
    elif action == "style" and value in ("bold", "classic", "neon", "minimal"):
        o.caption_style = value
    elif action == "formats" and value == "toggle":
        o.export_formats = "vertical" if o.export_formats == "all" else "all"
    elif action == "silence" and value == "toggle":
        o.remove_silence = not o.remove_silence
    await q.edit_message_text("⚙️ Gee Clipper controls\n\n" + summary(o),
                              reply_markup=settings_keyboard(o))


async def preset_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = ctx.args[0].lower() if ctx.args else ""
    presets = {
        "viral": dict(clips=6, layout="auto", captions=True, hook=True, music=True, grade=True, caption_style="bold"),
        "podcast": dict(clips=4, layout="auto", captions=True, hook=False, music=False, grade=True, caption_style="classic"),
        "educational": dict(clips=4, layout="auto", captions=True, hook=True, music=False, grade=True, caption_style="minimal"),
        "cinematic": dict(clips=3, layout="blur", captions=True, hook=True, music=True, grade=True, caption_style="classic"),
        "clean": dict(clips=3, layout="auto", captions=False, hook=False, music=False, grade=True, caption_style="minimal"),
    }
    if name not in presets:
        await update.message.reply_text("Choose: viral, podcast, educational, cinematic, clean. Example: /preset viral")
        return
    o = get_opts(ctx)
    for key, value in presets[name].items():
        setattr(o, key, value)
    await update.message.reply_text(f"{name.title()} preset applied.\n\n" + summary(o),
                                    reply_markup=settings_keyboard(o))


async def queue_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    state = JOB_STATES.get(update.effective_user.id)
    active = sum(1 for x in JOB_STATES.values() if x.get("status") == "processing")
    queued = sum(1 for x in JOB_STATES.values() if x.get("status") == "queued")
    mine = f"Your job is {state.get('status')}." if state else "You have no active job."
    await update.message.reply_text(f"📋 Queue\n{mine}\nProcessing: {active}\nWaiting: {queued}")


async def cancel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    event = CANCEL_FLAGS.get(update.effective_user.id)
    if not event:
        await update.message.reply_text("You have no active job to cancel.")
        return
    event.set()
    await update.message.reply_text("Cancellation requested. Gee Clipper will stop after the current download/render step.")


async def outro_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args).strip()
    if not text:
        await update.message.reply_text("Usage: /outro Follow for more or /outro off")
        return
    o = get_opts(ctx)
    o.outro_text = "" if text.lower() == "off" else text[:80]
    await update.message.reply_text(f"Outro: {o.outro_text or 'off'}")


async def translate_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    language = " ".join(ctx.args).strip()[:40]
    if not language:
        await update.message.reply_text("Usage: /translate Spanish or /translate off")
        return
    get_opts(ctx).subtitle_language = "off" if language.lower() == "off" else language
    await update.message.reply_text(
        f"Translated subtitle language: {get_opts(ctx).subtitle_language}. "
        "Original subtitles are still included."
    )


async def captionstyle_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    value = ctx.args[0].lower() if ctx.args else ""
    if value not in ("bold", "classic", "neon", "minimal"):
        await update.message.reply_text("Choose a caption theme: bold, classic, neon or minimal. Example: /captionstyle neon")
        return
    o = get_opts(ctx)
    o.caption_style = value
    await update.message.reply_text(f"Caption theme set to {value}.", reply_markup=settings_keyboard(o))


async def watermark_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    o = get_opts(ctx)
    value = " ".join(ctx.args).strip()[:40]
    if not value:
        await update.message.reply_text("Usage: /watermark YourBrand or /watermark off")
        return
    o.watermark = "" if value.lower() == "off" else value
    await update.message.reply_text(f"Watermark: {o.watermark or 'off'}")


async def logo_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update, ctx):
        await update.message.reply_text("Sorry, this bot is private.")
        return
    if not update.message.photo:
        return
    photo = update.message.photo[-1]
    if photo.file_size and photo.file_size > 8 * 1024 * 1024:
        await update.message.reply_text("Please upload a smaller logo image (under 8 MB).")
        return
    folder = os.path.join(C.CACHE_DIR, "user-assets")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"logo_{update.effective_user.id}.png")
    try:
        remote = await photo.get_file()
        await remote.download_to_drive(custom_path=path)
        get_opts(ctx).logo_path = path
        await update.message.reply_text("✅ Logo saved. It will appear in the bottom-right corner of future clips. Use /clearlogo to remove it.")
    except Exception:
        log.exception("user logo upload failed")
        await update.message.reply_text("I couldn't save that logo. Please try uploading it again.")


async def music_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update, ctx):
        await update.message.reply_text("Sorry, this bot is private.")
        return
    media = update.message.audio or update.message.document
    if not media:
        return
    if media.file_size and media.file_size > 19 * 1024 * 1024:
        await update.message.reply_text("Please upload an audio file under 19 MB.")
        return
    folder = os.path.join(C.CACHE_DIR, "user-assets")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"music_{update.effective_user.id}.audio")
    try:
        remote = await media.get_file()
        await remote.download_to_drive(custom_path=path)
        get_opts(ctx).music_path = path
        await update.message.reply_text("🎵 Custom soundtrack saved. It will be used instead of the built-in music library. Use /clearmusic to return to automatic mood matching.")
    except Exception:
        log.exception("custom soundtrack upload failed")
        await update.message.reply_text("I couldn't save that audio. Please try again.")


async def clearmusic_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    o = get_opts(ctx)
    path = o.music_path
    o.music_path = ""
    if path:
        try:
            os.remove(path)
        except OSError:
            pass
    await update.message.reply_text("Custom soundtrack removed. Automatic mood-matched music is enabled.")


async def clearlogo_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    o = get_opts(ctx)
    path = o.logo_path
    o.logo_path = ""
    if path:
        try:
            os.remove(path)
        except OSError:
            pass
    await update.message.reply_text("Your logo overlay has been removed.")


async def formats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    value = ctx.args[0].lower() if ctx.args else ""
    if value not in ("all", "vertical"):
        await update.message.reply_text("Usage: /formats all or /formats vertical\nall creates 9:16, 1:1 and 16:9 versions.")
        return
    o = get_opts(ctx)
    o.export_formats = value
    await update.message.reply_text(f"Export formats set to {value}.", reply_markup=settings_keyboard(o))


async def silence_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    value = ctx.args[0].lower() if ctx.args else ""
    if value not in ("on", "off"):
        await update.message.reply_text("Usage: /silence on or /silence off")
        return
    o = get_opts(ctx)
    o.remove_silence = value == "on"
    await update.message.reply_text(f"Silence removal {value}.", reply_markup=settings_keyboard(o))


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in C.ADMIN_USERS:
        await update.message.reply_text("Admin only.")
        return
    data = ctx.application.bot_data
    usage = data.get("daily_usage", {})
    today = date.today().isoformat()
    today_count = sum(n for day, n in usage.values() if day == today)
    users = data.get("total_users", set())
    await update.message.reply_text(
        "📊 Gee Clipper admin dashboard\n"
        f"Total jobs: {data.get('total_jobs', 0)}\n"
        f"Unique users seen: {len(users)}\n"
        f"Today's accepted jobs: {today_count}\n"
        f"Active jobs: {sum(1 for j in JOB_STATES.values() if j.get('status') == 'processing')}\n"
        f"Queued jobs: {sum(1 for j in JOB_STATES.values() if j.get('status') == 'queued')}\n"
        f"Blocked users: {len(data.get('blocked_users', set()))}"
    )


async def block_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in C.ADMIN_USERS:
        await update.message.reply_text("Admin only.")
        return
    try:
        target = int(ctx.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /block TELEGRAM_USER_ID")
        return
    ctx.application.bot_data.setdefault("blocked_users", set()).add(target)
    await update.message.reply_text(f"User {target} blocked.")


async def unblock_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in C.ADMIN_USERS:
        await update.message.reply_text("Admin only.")
        return
    try:
        target = int(ctx.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /unblock TELEGRAM_USER_ID")
        return
    ctx.application.bot_data.setdefault("blocked_users", set()).discard(target)
    await update.message.reply_text(f"User {target} unblocked.")


async def history_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    history = ctx.user_data.get("history", [])
    if not history:
        await update.message.reply_text("No recent jobs yet.")
        return
    lines = ["🕘 Recent jobs"]
    for item in history[-10:][::-1]:
        lines.append(f"• {item.get('title', 'Video')[:55]} — {item.get('clips', 0)} clips ({item.get('status', 'done')})")
    await update.message.reply_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, reply_markup=settings_keyboard(get_opts(ctx)))


async def settings_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cmd = update.message.text.split()[0].lstrip("/").split("@")[0].lower()
    o = get_opts(ctx)
    arg = ctx.args[0].lower() if ctx.args else ""
    if cmd == "settings":
        pass
    elif cmd == "clips" and arg.isdigit():
        o.clips = max(1, min(int(arg), C.MAX_CLIPS))
    elif cmd == "layout" and arg in ("auto", "crop", "blur"):
        o.layout = arg
    elif cmd in ("captions", "hook", "music", "grade") and arg in ("on", "off"):
        setattr(o, cmd, arg == "on")
    else:
        usage = {
            "clips": f"/clips 1-{C.MAX_CLIPS}",
            "layout": "/layout auto|crop|blur",
            "captions": "/captions on|off",
            "hook": "/hook on|off",
            "music": "/music on|off",
            "grade": "/grade on|off",
        }[cmd]
        await update.message.reply_text(f"Usage: {usage}")
        return
    await update.message.reply_text(summary(o))


async def handle_link(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global WAITING
    if not allowed(update, ctx):
        await update.message.reply_text("Sorry, this bot is private.")
        return
    text = update.message.text or ""
    m = URL_RE.search(text)
    if not m:
        await update.message.reply_text("Send me a video link.\n\n" + HELP)
        return
    urls = [u.rstrip(".,)") for u in URL_RE.findall(text)][:3]
    url = urls[0]
    opts = core.Opts(**vars(get_opts(ctx)))
    tail = text.replace(url, "").strip()
    if tail.isdigit():
        opts.clips = max(1, min(int(tail), C.MAX_CLIPS))

    uid = update.effective_user.id
    allowed_urls = []
    for candidate in urls:
        if not use_quota(uid, ctx):
            break
        allowed_urls.append(candidate)
    if not allowed_urls:
        await update.message.reply_text(f"Daily limit reached ({C.DAILY_LIMIT} videos). Try again tomorrow.")
        return
    if len(allowed_urls) < len(urls):
        await update.message.reply_text(f"Daily limit reached; processing {len(allowed_urls)} of {len(urls)} submitted links.")
    event = asyncio.Event()
    CANCEL_FLAGS[uid] = event
    JOB_STATES[uid] = {"status": "queued", "url": allowed_urls}
    if JOB_LOCK.locked():
        await update.message.reply_text(f"Queued. {WAITING + 1} ahead. Use /queue or /cancel.")
    WAITING += 1
    try:
        async with JOB_LOCK:
            if event.is_set():
                await update.message.reply_text("Job cancelled before processing.")
                return
            JOB_STATES[uid]["status"] = "processing"
            for index, candidate in enumerate(allowed_urls, 1):
                if event.is_set():
                    break
                JOB_STATES[uid]["current"] = index
                await process(update, candidate, opts, ctx)
    finally:
        WAITING -= 1
        JOB_STATES.pop(uid, None)
        CANCEL_FLAGS.pop(uid, None)


async def process(update: Update, url: str, opts: core.Opts, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    stats = ctx.application.bot_data
    stats["total_jobs"] = int(stats.get("total_jobs", 0)) + 1
    stats["total_users"] = set(stats.get("total_users", [])) | {update.effective_user.id}
    status = await msg.reply_text("Checking video...")
    workdir = tempfile.mkdtemp(prefix="clipper_")

    async def step(text: str):
        event = CANCEL_FLAGS.get(update.effective_user.id)
        if event and event.is_set():
            raise JobCancelled()
        try:
            await status.edit_text(text)
        except Exception:
            log.exception("Telegram progress message update failed")

    delivery_files = []
    try:
        info = await asyncio.to_thread(core.get_info, url)
        duration = info.get("duration") or 0
        if info.get("is_live"):
            await step("Live streams aren't supported.")
            return
        if duration > C.MAX_VIDEO_MINUTES * 60:
            await step(
                f"Video is {int(duration // 60)} min. Max is {C.MAX_VIDEO_MINUTES} min."
            )
            return

        key = core.video_key(info)
        data = core.load_transcript(key)
        if data is None:
            await step(f"Downloading audio: {info.get('title', '')[:60]}")
            audio = await asyncio.to_thread(core.download_audio, url, workdir)
            chunks = await asyncio.to_thread(core.split_audio, audio, workdir)
            await step("Transcribing...")
            segments, words, lang = await asyncio.to_thread(core.transcribe, chunks)
            data = {"segments": segments, "words": words, "language": lang}
            if segments:
                core.save_transcript(key, data)
        else:
            await step("Using cached transcript...")
        segments, words = data["segments"], data["words"]
        if not segments:
            await step("Couldn't find any speech in this video.")
            return

        await step("Finding the best moments...")
        clips = await asyncio.to_thread(
            core.pick_clips, segments, opts.clips, info.get("heatmap")
        )
        if not clips:
            await step("Couldn't find good clip moments in this video.")
            return

        sent = 0
        used_tracks: dict = {}  # keeps each clip of this video on a different song
        for i, clip in enumerate(clips, 1):
            await step(f"Preparing clip {i}/{len(clips)}...")
            try:
                loop = asyncio.get_running_loop()

                def clip_progress(label):
                    asyncio.run_coroutine_threadsafe(step(label), loop)

                res = await asyncio.to_thread(
                    core.make_clip, url, words, clip, i, workdir, opts, clip_progress, used_tracks
                )
                with open(res["video"], "rb") as f:
                    thumb = open(res["cover_small"], "rb") if "cover_small" in res else None
                    try:
                        await msg.reply_video(
                            video=f,
                            caption=build_caption(clip),
                            thumbnail=thumb,
                            supports_streaming=True,
                            width=720,
                            height=1280,
                            read_timeout=180,
                            write_timeout=180,
                        )
                    finally:
                        if thumb:
                            thumb.close()
                sent += 1
                for key in ("video", "vtt", "subtitles", "translated_subtitles", "thumb", "cover"):
                    if res.get(key) and os.path.isfile(res[key]):
                        delivery_files.append((res[key], os.path.basename(res[key])))
                for fmt, variant_path in (res.get("variants") or {}).items():
                    if os.path.isfile(variant_path):
                        delivery_files.append((variant_path, os.path.basename(variant_path)))
                if res.get("vtt"):
                    try:
                        with open(res["vtt"], "rb") as vtt_file:
                            await msg.reply_document(vtt_file, filename=f"gee-clipper-{i}.vtt", caption="📝 WebVTT subtitles")
                    except Exception:
                        log.exception("WebVTT file send failed")
                for fmt, variant_path in (res.get("variants") or {}).items():
                    try:
                        if os.path.getsize(variant_path) <= C.TG_FILE_LIMIT:
                            with open(variant_path, "rb") as variant_file:
                                await msg.reply_video(variant_file, caption=f"📐 {fmt.title()} export", supports_streaming=True, read_timeout=180, write_timeout=180)
                        else:
                            await msg.reply_text(f"{fmt.title()} export is too large for Telegram's upload limit; the vertical clip was sent.")
                    except Exception:
                        log.exception("social aspect-ratio export failed format=%s", fmt)
                if res.get("translated_subtitles"):
                    try:
                        with open(res["translated_subtitles"], "rb") as translated_file:
                            await msg.reply_document(translated_file, filename=f"gee-clipper-{i}-translated.srt", caption=f"🌍 Translated subtitles ({opts.subtitle_language})")
                    except Exception:
                        log.exception("translated subtitle send failed")
                if res.get("subtitles"):
                    try:
                        with open(res["subtitles"], "rb") as subtitle_file:
                            await msg.reply_document(subtitle_file, filename=f"gee-clipper-{i}.srt", caption="📝 Subtitles (SRT)")
                    except Exception:
                        log.exception("subtitle file send failed")
                try:
                    await msg.reply_text(build_caption_pack(clip))
                except Exception:
                    log.exception("social caption pack send failed")
                if "thumb" in res:
                    try:
                        with open(res["thumb"], "rb") as a, open(res["cover"], "rb") as b:
                            await msg.reply_media_group(
                                media=[
                                    InputMediaPhoto(a, caption="🖼 Thumbnails: 16:9 + vertical cover"),
                                    InputMediaPhoto(b),
                                ],
                                write_timeout=120,
                            )
                    except Exception:
                        log.exception("thumbnail send failed")
            except Exception as e:
                log.exception("clip %s failed", i)
                await msg.reply_text(f"Clip {i} failed: {friendly_error(e)}")

        history = ctx.user_data.setdefault("history", [])
        history.append({"title": str(info.get("title") or "YouTube video")[:100],
                        "url": url, "clips": sent, "status": "completed" if sent else "no clips",
                        "date": date.today().isoformat()})
        del history[:-10]
        if delivery_files:
            archive = os.path.join(workdir, "gee-clipper-exports.zip")
            try:
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as pack:
                    for source, name in delivery_files:
                        pack.write(source, arcname=name)
                if os.path.getsize(archive) <= C.TG_FILE_LIMIT:
                    with open(archive, "rb") as pack:
                        await msg.reply_document(pack, filename="gee-clipper-exports.zip", caption="📦 Your clips, subtitles and thumbnails in one pack")
                else:
                    await msg.reply_text("📦 Export pack is larger than Telegram's upload limit. Individual files have been sent above.")
            except Exception:
                log.exception("export archive creation or delivery failed")
        await step(f"Done. {sent} clip(s) sent. Use /history for recent jobs.")
    except JobCancelled:
        try:
            await status.edit_text("Job cancelled.")
        except Exception:
            pass
        history = ctx.user_data.setdefault("history", [])
        history.append({"title": "Cancelled video", "clips": 0, "status": "cancelled"})
        del history[:-10]
    except Exception as e:
        log.exception("job failed")
        await step(f"Failed: {friendly_error(e)}")
        history = ctx.user_data.setdefault("history", [])
        history.append({"title": "Failed video", "clips": 0, "status": "failed"})
        del history[:-10]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def post_init(app: Application):
    await app.bot.set_my_commands(
        [
            BotCommand("start", "How to use"),
            BotCommand("settings", "Show settings"),
            BotCommand("clips", "Number of clips"),
            BotCommand("layout", "auto | crop | blur"),
            BotCommand("captions", "on | off"),
            BotCommand("hook", "on | off"),
            BotCommand("music", "on | off"),
            BotCommand("grade", "on | off"),
            BotCommand("panel", "Button-based settings"),
            BotCommand("preset", "Apply an editing preset"),
            BotCommand("queue", "Check queue"),
            BotCommand("cancel", "Cancel current job"),
            BotCommand("history", "Recent jobs"),
            BotCommand("watermark", "Set or remove text watermark"),
            BotCommand("clearlogo", "Remove uploaded brand logo"),
            BotCommand("clearmusic", "Remove uploaded soundtrack"),
            BotCommand("captionstyle", "Choose caption theme"),
            BotCommand("translate", "Translate subtitles"),
            BotCommand("outro", "Set a branded end card"),
            BotCommand("formats", "Choose video aspect-ratio exports"),
            BotCommand("silence", "Toggle silence removal"),
            BotCommand("stats", "Admin usage dashboard"),
            BotCommand("block", "Block a Telegram user"),
            BotCommand("unblock", "Unblock a Telegram user"),
        ]
    )


def main():
    if not C.TELEGRAM_TOKEN or not C.GROQ_API_KEY:
        sys.exit("Set TELEGRAM_BOT_TOKEN and GROQ_API_KEY in .env (see .env.example)")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not installed")
    core.prune_cache()
    cookie_path = C.COOKIES_FILE
    log.info(
        "Runtime diagnostics: yt_dlp_version=%s cookies_configured=%s "
        "cookie_file_exists=%s cookie_file_bytes=%s js_runtime=%s",
        yt_dlp.version.__version__,
        bool(cookie_path),
        bool(cookie_path and os.path.isfile(cookie_path)),
        os.path.getsize(cookie_path) if cookie_path and os.path.isfile(cookie_path) else 0,
        C.JS_RUNTIME or "yt-dlp default",
    )
    persistence = PicklePersistence(filepath=os.getenv("BOT_PERSISTENCE_FILE", "gee_clipper_data.pickle"))
    app = (
        Application.builder()
        .token(C.TELEGRAM_TOKEN)
        .persistence(persistence)
        .concurrent_updates(True)
        .read_timeout(30)
        .write_timeout(180)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^(toggle|layout|clips|style|formats|silence):"))
    app.add_handler(CommandHandler("panel", panel_cmd))
    app.add_handler(CommandHandler("preset", preset_cmd))
    app.add_handler(CommandHandler("queue", queue_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(CommandHandler("watermark", watermark_cmd))
    app.add_handler(CommandHandler("clearlogo", clearlogo_cmd))
    app.add_handler(CommandHandler("clearmusic", clearmusic_cmd))
    app.add_handler(MessageHandler(filters.AUDIO | filters.Document.AUDIO, music_upload))
    app.add_handler(MessageHandler(filters.PHOTO, logo_upload))
    app.add_handler(CommandHandler("captionstyle", captionstyle_cmd))
    app.add_handler(CommandHandler("translate", translate_cmd))
    app.add_handler(CommandHandler("outro", outro_cmd))
    app.add_handler(CommandHandler("formats", formats_cmd))
    app.add_handler(CommandHandler("silence", silence_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("block", block_cmd))
    app.add_handler(CommandHandler("unblock", unblock_cmd))
    app.add_handler(
        CommandHandler(["settings", "clips", "layout", "captions", "hook", "music", "grade"], settings_cmd)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_link))
    webhook_base = os.getenv("WEBHOOK_URL", "").strip().rstrip("/")
    if webhook_base:
        webhook_path = os.getenv("WEBHOOK_PATH", "").strip().strip("/")
        if not webhook_path:
            sys.exit("WEBHOOK_PATH must be set when WEBHOOK_URL is configured")
        port = int(os.getenv("PORT", os.getenv("WEBHOOK_PORT", "8080")))
        secret = os.getenv("WEBHOOK_SECRET", "").strip() or None
        log.info("Bot starting in webhook mode port=%s path_configured=%s", port, bool(webhook_path))
        app.run_webhook(
            listen="0.0.0.0", port=port, url_path=webhook_path,
            webhook_url=f"{webhook_base}/{webhook_path}", secret_token=secret,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        log.info("Bot running in polling mode")
        app.run_polling()


if __name__ == "__main__":
    main()
