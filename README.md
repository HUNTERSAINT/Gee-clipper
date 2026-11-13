# Gee Clipper

A Telegram bot that turns a YouTube link into short vertical clips ready for TikTok, Reels and Shorts. Built entirely on free tools.

## What you get for each clip

- **9:16 video** (720x1280), cut at sentence boundaries
- **Face-tracking crop** that follows the speaker with smooth pans. If no face is found, it uses a blurred-background layout so nothing is cut off
- **Karaoke captions**, three words at a time with the spoken word highlighted
- **Hook title banner** for the first 3 seconds
- **Loudness-normalised audio**
- **Thumbnails**: the best frame is picked automatically (sharp, well lit, big face, eyes open) and the hook text is added. You get a 16:9 thumbnail for YouTube and a 9:16 cover. The cover is also set as the video's Telegram preview
- **Title, description, hashtags and a 1-10 score**, ready to copy and paste

## How it works

1. `yt-dlp` downloads the audio only
2. Groq Whisper transcribes it with word-level timestamps
3. Groq Llama reads the transcript and picks the best 20-75 second moments. If the video has YouTube's "most replayed" data, it uses that as a hint too
4. `yt-dlp` downloads only those sections of the video
5. OpenCV tracks the speaker's face and `ffmpeg` crops, burns in captions and renders
6. Pillow builds the thumbnails
7. The bot sends everything back

The transcript is cached for 7 days, so asking for more clips from the same link is instant.

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Get a free API key at https://console.groq.com/keys
3. Install **ffmpeg** (`sudo apt install ffmpeg fonts-liberation`, or `brew install ffmpeg`).
4. Install **Deno** (https://deno.com). yt-dlp uses it for YouTube. If you have Node.js instead, set `YT_JS_RUNTIME=node`.
5. Then:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in the two keys
python bot.py
```

## Using the bot

Send a YouTube link. Add a number to choose how many clips: `https://youtu.be/xxxx 5`

| Command | What it does |
| --- | --- |
| `/clips 3` | Number of clips per video |
| `/layout auto\|crop\|blur` | `auto` follows the face and falls back to blur |
| `/captions on\|off` | Burned-in captions |
| `/hook on\|off` | Title banner at the start |
| `/settings` | Show current settings |

Settings are per user and reset when the bot restarts.

Set `ALLOWED_USERS` to your Telegram ID to make the bot private, and keep `DAILY_LIMIT` low so strangers can't use up your free Groq quota.

## Hosting for free

The bot uses polling, so it just needs a machine that stays on.

- **Your own PC or laptop**: free, works while it's on. Also the most reliable for YouTube downloads
- **Oracle Cloud Always Free VM**: best free option for 24/7 (needs a card for verification)
- **Railway / Render / Fly.io**: the included Dockerfile works, but free tiers are small. Rendering needs CPU and around 1 GB RAM

```bash
docker build -t gee-clipper .
docker run --env-file .env gee-clipper
```

## Limits and troubleshooting

- **YouTube blocks server IPs.** Downloads often work from a home connection but fail from datacenters ("Sign in to confirm you're not a bot"). Export `cookies.txt` from a logged-in browser (use a throwaway account) and set `YT_COOKIES_FILE`. Keep yt-dlp updated: `pip install -U yt-dlp`
- **Groq free-tier limits.** Whisper and Llama have per-minute and per-day limits (see console.groq.com/docs/rate-limits). The bot retries automatically and falls back to a smaller model for clip picking. Very long videos use more quota
- **Telegram's 50 MB upload cap** for bots. Clips are 720p and re-encoded smaller if needed
- **Face tracking** uses OpenCV's built-in detector. It works best on front-facing talking heads. With several people on screen it follows the largest face
- **Thumbnails** are the best real frame plus text, not AI-generated images
- **One video is processed at a time**, others queue
- Only clip content you own or have permission to use
