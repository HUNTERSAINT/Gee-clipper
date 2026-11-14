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


## Persistent user data, saved clips and analytics

Gee Clipper stores user records, daily quota reservations, job history, generated clip metadata,
blocked accounts and saved favourites in a transactional SQLite database. Each generated clip
has a Save to favourites button. The /favorites command lists saved moments with direct YouTube
timestamp links; /unfavorite ID removes one. /analytics shows each user's job and clip totals
and remaining daily quota. Admin /stats reads database-backed global totals.

Set DATA_DIR to a persistent mounted directory to retain the database, transcript cache,
uploaded logos/music and Telegram settings across container replacements. GEE_DB_PATH,
CACHE_DIR and BOT_PERSISTENCE_FILE can override their individual paths. The default data/
directory is still ephemeral on hosts without a persistent volume.

## External publishing and paid plans

Direct TikTok, Instagram and YouTube publishing, OAuth connection flows, scheduled social
publishing, payment checkout and subscription activation require platform developer
credentials, user authorization, public callback URLs and payment-provider credentials.
They are not enabled by dummy tokens. No platform credentials or payment keys are stored
in the repository. Configure provider integrations only after obtaining approved developer
access; never commit API secrets.


## AI creator tools

After processing a video, /transcript downloads its timestamped transcript, /summary creates
AI show notes and chapter suggestions, and /titles generates alternative hooks and titles.
These are on-demand Groq requests and use your configured model quota.


## Creator tools: commentary, layouts, B-roll, metadata, timeline

These make a clip clearly your own edit rather than a straight repost. Every optional layer degrades
gracefully: a missing API key, failed download or LLM error just skips that layer and the clip still renders.

| Feature | How to use it |
|---|---|
| **Commentary + ducking** | `/commentary <script>` (free TTS) or send a voice message. It plays at the start of each clip and the source audio ducks to 15%, rising smoothly afterwards. `/clearcommentary` removes it. |
| **Commentary frame** | `/layout frame`: blurred backdrop, framed clip and a persistent title card with the clip's hook. |
| **Split screen** | Send a video with the caption `facecam`, then `/layout split`: source on the top 65%, you on the bottom 35%. The face-cam starts with each clip and loops if shorter. `/clearfacecam` removes it. |
| **Active speaker** | `/layout speaker`: cuts to whoever is talking (mouth movement). |
| **Two-speaker stack** | `/layout auto` (or `stack`) stacks both people when two faces are on screen for a good part of the clip; otherwise it follows one face. |
| **B-roll** | Set `PEXELS_API_KEY` (free) or `BROLL_DIR`. The LLM picks keywords with timestamps; 2.5-3.5 s cutaways fade in over the upper frame, at least 4 s apart. `/broll on\|off`. |
| **Metadata** | `/metadata on\|off` (default on): `metadata.json` with 3 YouTube titles under 50 characters, per-platform descriptions and hashtags, and a 2-sentence hook script. |
| **Timeline export** | `/timeline on`: a zip with an FCPXML project (source cut, B-roll lane), the captions as `.srt` and the rendered preview, for DaVinci Resolve or Final Cut. |

Captions, title cards and the logo stay out of the platform UI zones (bottom ~250 px and right ~140 px of a
1080x1920 frame, scaled for the 720x1280 output).

Notes: output is 720x1280 to keep renders fast and files under Telegram's 50 MB limit. Speaker framing follows
faces, not voices: the bot does not tell who is talking. Only use footage you have the rights to.

### Caption accuracy

Captions are timed against the clip itself, not the full-video transcript: each clip is normalised so audio and
video both start at 0 (cuts from YouTube can start mid-container, which shifts every caption), then its own audio is
transcribed again for word-level timing and text. Silence and intro trimming use those accurate words, so speech is
never cut mid-word. If the re-transcription fails, the full-video timings are used as a fallback.

### More creator tools

| Feature | Command | What it does |
|---|---|---|
| Active-speaker framing | `/layout speaker` | Watches mouth movement on two people and cuts the 9:16 crop to whoever is talking (with hysteresis, so it does not flicker). Falls back to normal framing if two faces are not found. |
| Intro trim | `/intro on\|off` (default on) | Opens each clip on its first full sentence, skipping filler and half sentences. |
| A/B hooks | `/ab on\|off` | Also renders Variant B with a different hook; both are sent so you can compare. |
| Push-in zoom | `/zoom on\|off` | Slow zoom on questions, exclamations and numbers. |
| Keyword highlight | `/accent <hex>\|off` | Current word in your colour; numbers, money and percentages always green. |
| Sound effects | `/sfx on\|off` | Soft whoosh on B-roll and pop on zooms, generated locally (no downloads, no licences). |
| Chapters and summary | `/chapters on\|off` | YouTube-ready chapters plus a 5-point summary for videos of 3+ minutes. |
| Brand kits | `/brand save\|use\|list\|delete <name>` | Saves layout, caption theme, accent, watermark, logo, outro and effects per channel. |
| Schedule CSV | automatic | `gee-clipper-schedule.csv` with title, description, hashtags and suggested UTC time per platform. |
| Quality report | `/quality on\|off` | Per clip: length, hook, speech pace, silence, loudness, peaks, exposure and sharpness, with a score. |

Emoji in captions are not supported: the bundled caption font has no emoji glyphs and subtitle rendering does not
handle colour emoji reliably.

## Translated voiceovers

Use /voiceover Spanish (or French, German, Portuguese, Italian, Hindi, Arabic, Japanese,
Korean or Chinese) to translate the spoken dialogue and generate a synthetic narration.
The narration replaces the original speech in the finished clip and is mixed with the
selected background music. Use /voiceover off to preserve the original speaker. Voiceover
is optional and uses the Edge TTS service plus Groq translation; service availability and
network access are required.


## Premium plans and referral rewards

The bot supports 30-day Premium subscriptions through Paystack checkout and server-side
verification. Set PAYSTACK_SECRET_KEY and PAYSTACK_PREMIUM_PRICE_KOBO in Railway Variables
before enabling checkout; the price is deliberately not hard-coded. Users run /upgrade email,
pay on Paystack, then /verify REFERENCE. Admins can use /grantpremium USER_ID DAYS and
/revokepremium USER_ID. /invite creates a Telegram deep link; each genuinely new referred
user adds one daily free job to the inviter, up to 10. /plan shows the current plan.
