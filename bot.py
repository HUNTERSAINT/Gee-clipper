"""Telegram front end for the YouTube clipper."""
import asyncio
import logging
import re
import os
import shutil
import sys
import tempfile
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
    "/watermark YourBrand (or off)\n\n"
    f"Videos up to {C.MAX_VIDEO_MINUTES} min. Send the same link again and "
    "it's instant (the transcript is cached)."
)


def get_opts(ctx: ContextTypes.DEFAULT_TYPE) -> core.Opts:
    return ctx.user_data.setdefault("opts", core.Opts())


def summary(o: core.Opts) -> str:
    return (
        f"Clips: {o.clips}\nLayout: {o.layout}\n"
        f"Captions: {'on' if o.captions else 'off'}\n"
        f"Hook title: {'on' if o.hook else 'off'}\n"
        f"Music: {'on' if o.music else 'off'}\n"
        f"Colour grade: {'on' if o.grade else 'off'}"
        + f"\nCaption theme: {o.caption_style}"
        + f"\nWatermark: {o.watermark or 'off'}"
    )


def allowed(update: Update) -> bool:
    return not C.ALLOWED_USERS or update.effective_user.id in C.ALLOWED_USERS


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
    if not allowed(update):
        await update.message.reply_text("Sorry, this bot is private.")
        return
    text = update.message.text or ""
    m = URL_RE.search(text)
    if not m:
        await update.message.reply_text("Send me a video link.\n\n" + HELP)
        return
    url = m.group(1)
    opts = core.Opts(**vars(get_opts(ctx)))
    tail = text.replace(url, "").strip()
    if tail.isdigit():
        opts.clips = max(1, min(int(tail), C.MAX_CLIPS))

    if not use_quota(update.effective_user.id, ctx):
        await update.message.reply_text(
            f"Daily limit reached ({C.DAILY_LIMIT} videos). Try again tomorrow."
        )
        return

    uid = update.effective_user.id
    event = asyncio.Event()
    CANCEL_FLAGS[uid] = event
    JOB_STATES[uid] = {"status": "queued", "url": url}
    if JOB_LOCK.locked():
        await update.message.reply_text(f"Queued. {WAITING + 1} ahead. Use /queue or /cancel.")
    WAITING += 1
    try:
        async with JOB_LOCK:
            if event.is_set():
                await update.message.reply_text("Job cancelled before processing.")
                return
            JOB_STATES[uid]["status"] = "processing"
            await process(update, url, opts, ctx)
    finally:
        WAITING -= 1
        JOB_STATES.pop(uid, None)
        CANCEL_FLAGS.pop(uid, None)


async def process(update: Update, url: str, opts: core.Opts, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
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
            BotCommand("captionstyle", "Choose caption theme"),
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
    app.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^(toggle|layout|clips|style):"))
    app.add_handler(CommandHandler("panel", panel_cmd))
    app.add_handler(CommandHandler("preset", preset_cmd))
    app.add_handler(CommandHandler("queue", queue_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(CommandHandler("watermark", watermark_cmd))
    app.add_handler(CommandHandler("captionstyle", captionstyle_cmd))
    app.add_handler(
        CommandHandler(["settings", "clips", "layout", "captions", "hook", "music", "grade"], settings_cmd)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_link))
    log.info("Bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
