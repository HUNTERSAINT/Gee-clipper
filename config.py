"""All settings come from environment variables (see .env.example)."""
import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _ids(name: str) -> set[int]:
    return {int(x) for x in os.getenv(name, "").replace(" ", "").split(",") if x}


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")

MAX_VIDEO_MINUTES = _int("MAX_VIDEO_MINUTES", 90)
DEFAULT_CLIPS = _int("DEFAULT_CLIPS", 3)
MAX_CLIPS = _int("MAX_CLIPS", 6)
CLIP_MIN_SEC = _int("CLIP_MIN_SEC", 20)
CLIP_MAX_SEC = _int("CLIP_MAX_SEC", 75)
SOURCE_HEIGHT = _int("SOURCE_HEIGHT", 720)

# Videos per user per day (0 = unlimited). Protects your free Groq quota.
DAILY_LIMIT = _int("DAILY_LIMIT", 10)

COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "")
# yt-dlp needs a JavaScript runtime for YouTube. Deno is the default;
# set to "node" if you have Node.js installed instead.
JS_RUNTIME = os.getenv("YT_JS_RUNTIME", "")

ALLOWED_USERS = _ids("ALLOWED_USERS")  # empty = anyone can use the bot
ADMIN_USERS = _ids("ADMIN_USERS")  # exempt from the daily limit

CACHE_DIR = os.getenv("CACHE_DIR", "cache")
TG_FILE_LIMIT = 49 * 1024 * 1024  # Telegram bot upload cap is 50 MB
