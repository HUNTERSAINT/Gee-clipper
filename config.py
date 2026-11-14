"""All settings come from environment variables (see .env.example)."""
import base64
import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _ids(name: str) -> set[int]:
    return {int(x) for x in os.getenv(name, "").replace(" ", "").split(",") if x}


def _to_netscape(raw: str) -> str:
    """Accept Netscape cookies.txt text or a browser-extension JSON export."""
    if raw.startswith(("[", "{")):  # e.g. Cookie-Editor's JSON export
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("cookies", [])
        lines = ["# Netscape HTTP Cookie File"]
        for c in data:
            domain = c.get("domain", "")
            if not domain:
                continue
            host_only = c.get("hostOnly", not domain.startswith("."))
            expiry = int(c.get("expirationDate") or c.get("expires") or 0)
            lines.append(
                "\t".join(
                    [
                        ("#HttpOnly_" if c.get("httpOnly") else "") + domain,
                        "FALSE" if host_only else "TRUE",
                        c.get("path", "/"),
                        "TRUE" if c.get("secure") else "FALSE",
                        str(expiry),
                        c.get("name", ""),
                        str(c.get("value", "")),
                    ]
                )
            )
        return "\n".join(lines) + "\n"

    # Netscape text. Pasting into a web form can turn tabs into spaces: repair it.
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") and not s.startswith("#HttpOnly_"):
            out.append(s)
            continue
        out.append("\t".join(s.split("\t") if "\t" in s else s.split(None, 6)))
    if not out or not out[0].startswith("# Netscape"):
        out.insert(0, "# Netscape HTTP Cookie File")
    return "\n".join(out) + "\n"


def _cookies_from_env() -> str:
    """Write YT_COOKIES (text) or YT_COOKIES_B64 to a temp file; return its path.

    Lets hosts without file uploads (Railway, etc.) use cookies via a variable.
    Returns "" if nothing is set or the value can't be parsed.
    """
    raw = os.getenv("YT_COOKIES", "").strip()
    b64 = os.getenv("YT_COOKIES_B64", "").strip()
    try:
        if not raw and b64:
            raw = base64.b64decode(b64).decode("utf-8", "replace").strip()
        if not raw:
            return ""
        path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_to_netscape(raw))
        return path
    except Exception:
        return ""


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
# Leave language empty for auto-detection; set to an ISO-639-1 code such as "en" when known.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip().lower()
# Optional vocabulary/name hint, in the same language as the audio (Groq limit: 224 tokens).
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "").strip()
# llama-3.3-70b-versatile and llama-3.1-8b-instant were retired for
# developer-tier Groq accounts. Use currently supported replacements.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "qwen/qwen3.8-27b")

MAX_VIDEO_MINUTES = _int("MAX_VIDEO_MINUTES", 240)
DEFAULT_CLIPS = _int("DEFAULT_CLIPS", 3)
MAX_CLIPS = _int("MAX_CLIPS", 6)
CLIP_MIN_SEC = _int("CLIP_MIN_SEC", 20)
CLIP_MAX_SEC = _int("CLIP_MAX_SEC", 60)
SOURCE_HEIGHT = _int("SOURCE_HEIGHT", 720)

# Videos per user per day (0 = unlimited). Protects your free Groq quota.
DAILY_LIMIT = _int("DAILY_LIMIT", 10)

# YouTube login cookies: a file path, or the cookies themselves in a variable.
COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "") or _cookies_from_env()
# yt-dlp needs a JavaScript runtime for YouTube. Deno is the default;
# set to "node" if you have Node.js instead.
JS_RUNTIME = os.getenv("YT_JS_RUNTIME", "")

ALLOWED_USERS = _ids("ALLOWED_USERS")  # empty = anyone can use the bot
ADMIN_USERS = _ids("ADMIN_USERS")  # exempt from the daily limit

DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "data"))
DB_PATH = os.path.abspath(os.getenv("GEE_DB_PATH", os.path.join(DATA_DIR, "gee_clipper.sqlite3")))
CACHE_DIR = os.path.abspath(os.getenv("CACHE_DIR", os.path.join(DATA_DIR, "cache")))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
TG_FILE_LIMIT = 49 * 1024 * 1024  # Telegram bot upload cap is 50 MB
