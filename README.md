# Gee Clipper

A Telegram bot that turns a YouTube link into short vertical clips ready for TikTok, Reels and Shorts. Built entirely on free tools.

## What you get for each clip

- **9:16 video** (720x1280), cut at sentence boundaries
- **Face-tracking crop** that follows the speaker with smooth pans. If no face is found, it uses a blurred-background layout so nothing is cut off
- **Karaoke captions**, three words at a time with the spoken word highlighted
- **Hook title banner** for the first 3 seconds
- **Adaptive colour grading**: every clip is measured (exposure, contrast, colour cast, saturation) and graded to fit its mood, then checked on real frames so highlights, blacks and skin tones stay natural
- **Clip-matched music**: royalty-free (CC0) tracks are analysed for tempo, energy and key, matched to each clip's mood and pacing, never repeated within one video, ducked under speech and loudness-normalised. If no track can be downloaded the clip is sent with its original audio
- **Loudness-normalised audio**
- **Thumbnails**: the best frame is picked automatically (sharp, well lit, big face, eyes open) and the hook text is added. You get a 16:9 thumbnail for YouTube and a 9:16 cover. The cover is also set as the video's Telegram preview
- **Title, description, hashtags and a 1-10 score**, ready to copy and paste
- **Platform caption packs** for TikTok, Instagram Reels and YouTube Shorts
- **Downloadable SRT subtitles** for each finished clip
- **Optional text watermark** that can be set per user
- **Four karaoke caption themes:** Bold, Classic, Neon and Minimal
- **Automatic silence trimming** that keeps natural short pauses and remaps caption timing
- **Three social aspect ratios:** 9:16, 1:1 and 16:9, with blurred-fill framing for landscape and square
- **SRT and WebVTT subtitle downloads**, with optional AI translation
- **Custom uploaded logo overlay** plus the text watermark option
- **Branded outro end card** with editable call-to-action text
- **Custom music upload** to replace the automatic soundtrack library
- **Batch processing** for up to three YouTube links in one message
- **ZIP export pack** containing clips, alternate formats, subtitles and thumbnails when it fits Telegram's upload limit
- **Admin tools** for usage stats and blocking/unblocking Telegram users

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
| `/music on\|off` | Background music matched to each clip |
| `/grade on\|off` | Automatic colour grading |
| `/settings` | Show current settings |
| `/panel` | Button-based settings panel |
| `/preset viral\|podcast\|educational\|cinematic\|clean` | Apply a ready-made editing preset |
| `/watermark YourBrand` | Add a text watermark; use `/watermark off` to remove it |
| `/captionstyle bold\|classic\|neon\|minimal` | Choose a caption theme |
| `/queue` | Check your current queue status |
| `/cancel` | Request cancellation of your current job |
| `/history` | Show your last 10 jobs |
| `/formats all\|vertical` | Export vertical only or all three aspect ratios |
| `/silence on\|off` | Enable or disable transcript-guided pause trimming |
| `/translate Spanish` | Add translated SRT subtitles; use `/translate off` to disable |
| `/outro Follow for more` | Set the branded end card; use `/outro off` to disable |
| Upload an audio file | Use it as your custom soundtrack; `/clearmusic` restores automatic music |
| Send a logo photo | Save a logo overlay for your clips; `/clearlogo` removes it |
| Send up to 3 YouTube links | Process them sequentially in one request |
| `/stats` | Admin-only usage dashboard |
| `/block 123456` / `/unblock 123456` | Admin user management |

Settings, recent job history, daily usage, logo path, custom music path and blocked-user list are stored with Telegram's local persistence file. The admin commands require your Telegram ID in `ADMIN_USERS`. Batch processing accepts up to three links in one message. Direct publishing to social platforms is not enabled: each platform requires its own approved app/OAuth credentials and user authorization. The optional AI subtitle translation consumes Groq quota.

Settings, recent job history and daily usage are stored with Telegram's local persistence file. They survive ordinary process restarts while that file remains on disk. Hosts with ephemeral filesystems can lose them when a container is replaced; attach persistent storage and set `BOT_PERSISTENCE_FILE` to a path on that storage if long-term persistence is required. The queue and cancellation controls are process-local. On Railway the bot uses a secure Telegram webhook (rather than long polling) to avoid duplicate getUpdates conflicts. Local development still uses polling unless WEBHOOK_URL, WEBHOOK_PATH and WEBHOOK_SECRET are configured.

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

## YouTube cookies

YouTube often blocks downloads from server IPs. Giving the bot the cookies of a logged-in **throwaway Google account** (never your main one) usually fixes it.

1. Log in to YouTube with the throwaway account in a browser (a private window is best).
2. Open `https://www.youtube.com/robots.txt` in that same tab, then export the cookies with a browser extension: **Cookie-Editor** (Firefox, including Android) or **Get cookies.txt LOCALLY** (Chrome). Cookie-Editor's JSON export works as is.
3. Don't log out and don't use YouTube in that browser again. YouTube rotates the cookies, and the exported copy stops working. Close the private window.
4. Give them to the bot:
   - **Railway or any host:** add a variable `YT_COOKIES` and paste the exported text as its value
   - **Your own machine:** save `cookies.txt` next to `bot.py` and set `YT_COOKIES_FILE=cookies.txt`
5. Redeploy or restart. If downloads start failing again, export fresh cookies.

Cookies are login credentials: never commit them or share them.

## Limits and troubleshooting

- **YouTube blocks server IPs.** Downloads often work from a home connection but fail from datacenters ("Sign in to confirm you're not a bot"). See "YouTube cookies" above. Keep yt-dlp updated: `pip install -U yt-dlp`
- **Groq free-tier limits.** Whisper and Llama have per-minute and per-day limits (see console.groq.com/docs/rate-limits). The bot retries automatically and falls back to a smaller model for clip picking. Very long videos use more quota
- **Telegram's 50 MB upload cap** for bots. Clips are 720p and re-encoded smaller if needed
- **Face tracking** uses OpenCV's built-in detector. It works best on front-facing talking heads. With several people on screen it follows the largest face
- **Thumbnails** are the best real frame plus text, not AI-generated images
- **One video is processed at a time**, others queue
- Only clip content you own or have permission to use
