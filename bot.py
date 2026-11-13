"""Telegram front end for the YouTube clipper."""
import asyncio
import logging
import re
import shutil
import sys
import tempfile
from datetime import date

from telegram import BotCommand, InputMediaPhoto, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config as C
import core

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
USAGE: dict[int, tuple[str, int]] = {}

HELP = (
    "Send me a YouTube link and I'll cut it into short vertical clips with "
    "captions, a hook title, a thumbnail and ready-to-paste hashtags.\n\n"
    "Add a number to choose how many: https://youtu.be/xxxx 5\n\n"
    "Settings:\n"
    "/clips 3 - how many clips\n"
    "/layout auto|crop|blur - auto follows the speaker's face\n"
    "/captions on|off\n"
    "/hook on|off - title banner at the start\n"
    "/settings - show current settings\n\n"
    f"Videos up to {C.MAX_VIDEO_MINUTES} min. Send the same link again and "
    "it's instant (the transcript is cached)."
)


def get_opts(ctx: ContextTypes.DEFAULT_TYPE) -> core.Opts:
    return ctx.user_data.setdefault("opts", core.Opts())


def summary(o: core.Opts) -> str:
    return (
        f"Clips: {o.clips}\nLayout: {o.layout}\n"
        f"Captions: {'on' if o.captions else 'off'}\n"
        f"Hook title: {'on' if o.hook else 'off'}"
    )


def allowed(update: Update) -> bool:
    return not C.ALLOWED_USERS or update.effective_user.id in C.ALLOWED_USERS


def use_quota(uid: int) -> bool:
    if uid in C.ADMIN_USERS or C.DAILY_LIMIT <= 0:
        return True
    today = date.today().isoformat()
    day, n = USAGE.get(uid, (today, 0))
    if day != today:
        n = 0
    if n >= C.DAILY_LIMIT:
        return False
    USAGE[uid] = (today, n + 1)
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


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)


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
    elif cmd in ("captions", "hook") and arg in ("on", "off"):
        setattr(o, cmd, arg == "on")
    else:
        usage = {
            "clips": f"/clips 1-{C.MAX_CLIPS}",
            "layout": "/layout auto|crop|blur",
            "captions": "/captions on|off",
            "hook": "/hook on|off",
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

    if not use_quota(update.effective_user.id):
        await update.message.reply_text(
            f"Daily limit reached ({C.DAILY_LIMIT} videos). Try again tomorrow."
        )
        return

    if JOB_LOCK.locked():
        await update.message.reply_text(
            f"Queued. {WAITING + 1} video(s) ahead of yours..."
        )
    WAITING += 1
    try:
        async with JOB_LOCK:
            await process(update, url, opts)
    finally:
        WAITING -= 1


async def process(update: Update, url: str, opts: core.Opts):
    msg = update.message
    status = await msg.reply_text("Checking video...")
    workdir = tempfile.mkdtemp(prefix="clipper_")

    async def step(text: str):
        try:
            await status.edit_text(text)
        except Exception:
            pass

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
        for i, clip in enumerate(clips, 1):
            await step(f"Rendering clip {i}/{len(clips)}...")
            try:
                res = await asyncio.to_thread(
                    core.make_clip, url, words, clip, i, workdir, opts
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

        await step(f"Done. {sent} clip(s) sent. Want more? /clips 6 then send the link again (instant).")
    except Exception as e:
        log.exception("job failed")
        await step(f"Failed: {friendly_error(e)}")
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
        ]
    )


def main():
    if not C.TELEGRAM_TOKEN or not C.GROQ_API_KEY:
        sys.exit("Set TELEGRAM_BOT_TOKEN and GROQ_API_KEY in .env (see .env.example)")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not installed")
    core.prune_cache()
    app = (
        Application.builder()
        .token(C.TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .read_timeout(30)
        .write_timeout(180)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(
        CommandHandler(["settings", "clips", "layout", "captions", "hook"], settings_cmd)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_link))
    log.info("Bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
