"""The brain: download, transcribe, pick the best moments, build each clip."""
import glob
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from groq import Groq, RateLimitError

import config as C
import grade as G
import music as M
import video as V
import voiceover as VO

log = logging.getLogger("clipper.core")


def _safe_ydl_message(message) -> str:
    """Remove URL query strings and credential-like headers before logging."""
    text = str(message)
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[query redacted]", text)
    text = re.sub(r"(?i)(cookie|authorization):\s*[^\r\n]+", r"\1: [redacted]", text)
    text = re.sub(r"(?i)(sid|sapisid|token)=([^&\s]+)", r"\1=[redacted]", text)
    return text[:1200]


class YtDlpLogBridge:
    """Forward useful yt-dlp diagnostics to Railway without verbose secrets."""
    def debug(self, message):
        # yt-dlp debug output can contain request details; keep it disabled.
        pass

    def warning(self, message):
        log.warning("yt-dlp: %s", _safe_ydl_message(message))

    def error(self, message):
        log.error("yt-dlp: %s", _safe_ydl_message(message))


groq = Groq(api_key=C.GROQ_API_KEY or "missing")


@dataclass
class Opts:
    clips: int = C.DEFAULT_CLIPS
    layout: str = "auto"  # auto | crop | blur
    captions: bool = True
    hook: bool = True
    music: bool = True
    grade: bool = True
    watermark: str = ""
    caption_style: str = "bold"
    export_formats: str = "all"  # vertical only, or all social aspect ratios
    remove_silence: bool = True
    logo_path: str = ""
    music_path: str = ""
    voiceover_language: str = "off"
    subtitle_language: str = "off"
    outro_text: str = "Follow for more"
    music_path: str = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get(obj, key, default=None):
    """Groq SDK returns objects or dicts depending on version; handle both."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def with_retry(fn, tries: int = 6):
    """Groq's free tier rate-limits; wait (honouring retry-after) and retry."""
    for i in range(tries):
        try:
            return fn()
        except RateLimitError as e:
            wait = 15 * (i + 1)
            try:
                ra = e.response.headers.get("retry-after")
                if ra:
                    wait = min(max(float(ra), 1) + 1, 120)
            except Exception:
                pass
            log.warning("Rate limited, sleeping %ss", wait)
            time.sleep(wait)
    raise RuntimeError("Groq rate limit kept failing. Try again in a few minutes.")


def ydl_base() -> dict:
    opts = {
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "logger": YtDlpLogBridge(),
        # Avoid the web_embedded client path implicated in recent YouTube
        # GVS PO-token 403s; let yt-dlp select its remaining default clients.
        "extractor_args": {"youtube": {"player_client": ["default", "-web_embedded"]}},
        "sleep_interval_requests": 1,
        # Bound stalled media requests and avoid effectively endless retries.
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "file_access_retries": 2,
    }
    if C.COOKIES_FILE and Path(C.COOKIES_FILE).exists():
        opts["cookiefile"] = C.COOKIES_FILE
    if C.JS_RUNTIME:
        opts["js_runtimes"] = {C.JS_RUNTIME.lower(): {}}
    return opts


# --------------------------------------------------------------------------- #
# video info, audio, transcript (+ cache)
# --------------------------------------------------------------------------- #
def get_info(url: str) -> dict:
    log.info("YouTube stage=get_info starting")
    try:
        with yt_dlp.YoutubeDL(ydl_base()) as ydl:
            info = ydl.extract_info(url, download=False)
        log.info(
            "YouTube stage=get_info success video_id=%s extractor=%s duration=%s",
            info.get("id"), info.get("extractor_key"), info.get("duration")
        )
        return info
    except Exception:
        log.exception("YouTube stage=get_info failed")
        raise


def video_key(info: dict) -> str:
    raw = f"{info.get('extractor_key', 'x')}_{info.get('id', 'x')}"
    return re.sub(r"[^\w\-]", "_", raw)


def download_audio(url: str, workdir: str) -> str:
    opts = ydl_base() | {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(workdir, "audio_src.%(ext)s"),
    }
    log.info("YouTube stage=download_audio starting")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            result = ydl.prepare_filename(info)
        log.info("YouTube stage=download_audio success video_id=%s", info.get("id"))
        return result
    except Exception:
        log.exception("YouTube stage=download_audio failed")
        raise


def split_audio(src: str, workdir: str, chunk_sec: int = 900) -> list[Path]:
    """16 kHz mono 32 kbps mp3 chunks (~3.6 MB each, under Groq's 25 MB cap)."""
    V.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1",
            "-ar", "16000", "-b:a", "32k", "-f", "segment",
            "-segment_time", str(chunk_sec), "-reset_timestamps", "1", "seg_%03d.mp3",
        ],
        cwd=workdir,
    )
    return sorted(Path(workdir).glob("seg_*.mp3"))


def transcribe(chunks: list[Path], chunk_sec: int = 900):
    segments, words, lang = [], [], None
    for idx, chunk in enumerate(chunks):
        offset = idx * chunk_sec
        data = chunk.read_bytes()
        resp = with_retry(
            lambda: groq.audio.transcriptions.create(
                file=(chunk.name, data),
                model=C.WHISPER_MODEL,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        )
        lang = lang or _get(resp, "language")
        for s in _get(resp, "segments", []) or []:
            segments.append(
                {
                    "start": _get(s, "start") + offset,
                    "end": _get(s, "end") + offset,
                    "text": _get(s, "text", "").strip(),
                }
            )
        for w in _get(resp, "words", []) or []:
            words.append(
                {
                    "word": _get(w, "word", "").strip(),
                    "start": _get(w, "start") + offset,
                    "end": _get(w, "end") + offset,
                }
            )
    return segments, words, lang


def load_transcript(key: str):
    p = Path(C.CACHE_DIR) / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_transcript(key: str, data: dict) -> None:
    Path(C.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    (Path(C.CACHE_DIR) / f"{key}.json").write_text(json.dumps(data))


def prune_cache(days: int = 7) -> None:
    cutoff = time.time() - days * 86400
    for p in Path(C.CACHE_DIR).glob("*.json"):
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# LLM picks the best moments (uses YouTube "most replayed" data when present)
# --------------------------------------------------------------------------- #
def _prompt(has_heat: bool, n_clips: int = C.MAX_CLIPS) -> str:
    heat = (
        "\n- Each line starts with [seconds|H#]. H# (0-9) is how often viewers "
        "replayed that part; higher means a better moment. Use it as a strong hint."
        if has_heat
        else ""
    )
    return f"""You are an expert short-form video editor (TikTok, Reels, YouTube Shorts).
You get a timestamped transcript excerpt from a longer video. Pick the moments most likely to perform as standalone clips.

Rules:
- Each clip is {C.CLIP_MIN_SEC}-{C.CLIP_MAX_SEC} seconds long.
- Start on a strong hook (bold claim, question, story start, surprise). Never start mid-thought.
- End on a complete thought or punchline.
- Prefer emotional peaks, funny moments, surprising facts, strong opinions, useful tips.
- Skip intros, sponsor reads, small talk, rambling.
- Use the exact seconds shown in the brackets.
- Return at most {n_clips} clips; fewer or none if nothing is good.{heat}
- "hook" is 2-5 punchy words for the thumbnail (no emoji, no hashtags).
- "description" is a 1-2 sentence caption for social media.
- "hashtags" is 3-6 relevant words without the # sign.

Respond with JSON only:
{{"clips":[{{"start":0,"end":0,"score":1,"title":"max 8 words","hook":"...","description":"...","hashtags":["..."]}}]}}"""


def heat_for(heatmap):
    if not heatmap:
        return None

    def f(a: float, b: float) -> int:
        vals = [
            h.get("value", 0)
            for h in heatmap
            if h.get("end_time", 0) > a and h.get("start_time", 0) < b
        ]
        return round(max(vals) * 9) if vals else 0

    return f


def build_windows(segments, heat_fn=None, window_sec: int = 600):
    windows, cur, win_start = [], [], None
    for s in segments:
        if win_start is None:
            win_start = s["start"]
        tag = f"{s['start']:.0f}s"
        if heat_fn:
            tag += f"|H{heat_fn(s['start'], s['end'])}"
        cur.append(f"[{tag}] {s['text']}")
        if s["end"] - win_start >= window_sec:
            windows.append("\n".join(cur))
            cur, win_start = [], None
    if cur:
        windows.append("\n".join(cur))
    return windows


def _ask(model: str, prompt: str, text: str):
    return with_retry(
        lambda: groq.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        ),
        tries=4,
    )


def pick_clips(segments, n_clips: int, heatmap=None):
    heat_fn = heat_for(heatmap)
    prompt = _prompt(heat_fn is not None, n_clips)
    candidates = []
    for text in build_windows(segments, heat_fn):
        try:
            resp = _ask(C.LLM_MODEL, prompt, text)
        except RuntimeError:
            if not C.LLM_FALLBACK_MODEL:
                raise
            log.warning("Primary model rate-limited, using fallback")
            resp = _ask(C.LLM_FALLBACK_MODEL, prompt, text)
        try:
            data = json.loads(resp.choices[0].message.content)
            candidates.extend(data.get("clips", []))
        except Exception:
            log.warning("Bad JSON from LLM, skipping a window")

    clean = []
    for c in candidates:
        try:
            start, end = float(c["start"]), float(c["end"])
        except Exception:
            continue
        # snap to sentence boundaries
        starts = [s["start"] for s in segments if s["start"] <= start + 0.5]
        ends = [s["end"] for s in segments if s["end"] >= end - 0.5]
        if not starts or not ends:
            continue
        start, end = max(starts), min(ends)
        dur = end - start
        if dur < C.CLIP_MIN_SEC * 0.8 or dur > C.CLIP_MAX_SEC * 1.3:
            continue
        tags = [re.sub(r"\W", "", str(t)) for t in (c.get("hashtags") or [])]
        clean.append(
            {
                "start": start,
                "end": end,
                "score": float(c.get("score", 5)),
                "title": str(c.get("title", "Clip"))[:80],
                "hook": str(c.get("hook") or "")[:40],
                "description": str(c.get("description") or "")[:300],
                "hashtags": [t for t in tags if t][:6],
            }
        )

    clean.sort(key=lambda c: c["score"], reverse=True)
    chosen = []
    for c in clean:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in chosen):
            chosen.append(c)
        if len(chosen) >= n_clips:
            break
    return sorted(chosen, key=lambda c: c["start"])


# --------------------------------------------------------------------------- #
# build one finished clip (video + thumbnails)
# --------------------------------------------------------------------------- #
def download_section(url: str, start: float, end: float, workdir: str, base: str) -> str:
    opts = ydl_base() | {
        "format": (
            f"bv*[height<={C.SOURCE_HEIGHT}]+ba/b[height<={C.SOURCE_HEIGHT}]/b"
        ),
        "outtmpl": os.path.join(workdir, base + ".%(ext)s"),
        "merge_output_format": "mp4",
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
    }
    last_progress_log = [0.0]

    def report_download(progress):
        now = time.monotonic()
        status = progress.get("status")
        if status in ("finished", "error") or now - last_progress_log[0] >= 10:
            total = progress.get("total_bytes") or progress.get("total_bytes_estimate")
            log.info(
                "YouTube segment download status=%s bytes=%s total=%s eta=%s",
                status, progress.get("downloaded_bytes"), total, progress.get("eta")
            )
            last_progress_log[0] = now

    opts["progress_hooks"] = [report_download]
    log.info("YouTube stage=download_section starting start=%s end=%s", start, end)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:
        log.exception("YouTube stage=download_section failed start=%s end=%s", start, end)
        raise
    files = [
        p
        for p in glob.glob(os.path.join(workdir, base + ".*"))
        if not p.endswith((".part", ".ytdl"))
    ]
    if not files:
        raise RuntimeError("Section download produced no file")
    log.info(
        "YouTube stage=download_section success file_bytes=%s",
        os.path.getsize(files[0])
    )
    return files[0]



def classify_music_mood(text: str) -> tuple[str, float]:
    """Use the language model to understand the clip and choose a soundtrack mood."""
    allowed = {"uplifting", "dramatic", "emotional", "energetic", "calm", "funny"}
    prompt = (
        "Classify the emotional tone and pacing of this short video transcript. "
        "Choose the best background music mood from: uplifting, dramatic, emotional, "
        "energetic, calm, funny. Judge the meaning and delivery implied by the words, "
        "not isolated keyword matches. Also rate the pacing/intensity from 1 (very slow, quiet) "
        "to 5 (fast, intense). Return JSON only: {\"mood\":\"one category\",\"energy\":3}."
    )
    try:
        try:
            response = _ask(C.LLM_MODEL, prompt, (text or "")[:5000])
        except Exception:
            if not C.LLM_FALLBACK_MODEL:
                raise
            response = _ask(C.LLM_FALLBACK_MODEL, prompt, (text or "")[:5000])
        result = json.loads(response.choices[0].message.content)
        mood = str(result.get("mood", "")).strip().lower()
        if mood not in allowed:
            raise ValueError("LLM returned an unsupported soundtrack mood")
        try:
            energy = min(5.0, max(1.0, float(result.get("energy", 3))))
        except (TypeError, ValueError):
            energy = 3.0
        return mood, energy
    except Exception:
        log.exception("Soundtrack mood classification failed; using calm instrumental")
        return "calm", 2.5


def translate_subtitle_file(source: str, target_language: str, output_path: str):
    """Translate subtitle lines in bounded batches while preserving exact timecodes."""
    raw = Path(source).read_text(encoding="utf-8")
    blocks = [b for b in re.split(r"\n\s*\n", raw.strip()) if b.strip()]
    parsed = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            parsed.append((lines[1], " ".join(lines[2:])))
    translated = []
    for offset in range(0, len(parsed), 35):
        batch = parsed[offset:offset + 35]
        payload = json.dumps([text for _, text in batch], ensure_ascii=False)
        prompt = (
            f"Translate each subtitle string into {target_language}. Preserve meaning, tone, "
            'names and short-form speech. Return JSON only as {"translations":[...]} '
            "with exactly the same number and order of strings."
        )
        try:
            response = _ask(C.LLM_MODEL, prompt, payload)
        except Exception:
            response = _ask(C.LLM_FALLBACK_MODEL, prompt, payload)
        values = json.loads(response.choices[0].message.content).get("translations", [])
        if len(values) != len(batch):
            raise RuntimeError("Translation response did not match subtitle count")
        translated.extend(str(x).replace("\n", " ").strip() for x in values)
    is_vtt = output_path.lower().endswith(".vtt")
    output = ["WEBVTT"] if is_vtt else []
    for i, ((timecode, _), text) in enumerate(zip(parsed, translated), 1):
        output.append((f"{timecode}\n{text}" if is_vtt else f"{i}\n{timecode}\n{text}"))
    Path(output_path).write_text("\n\n".join(output) + "\n", encoding="utf-8")


def make_clip(
    url: str, words, clip: dict, idx: int, workdir: str, opts: Opts, progress=None,
    used_tracks: dict | None = None,
) -> dict:
    def report(stage):
        if progress:
            progress(stage)

    report(f"Downloading video segment for clip {idx}...")
    raw = download_section(url, clip["start"], clip["end"], workdir, f"raw_{idx}")
    clip = dict(clip)
    clip_words = words
    if opts.remove_silence:
        try:
            raw, clip_words, compact_duration = V.remove_silence(
                raw, words, clip["start"], clip["end"]
            )
            clip["start"], clip["end"] = 0.0, compact_duration
        except Exception:
            log.exception("Silence trimming failed; keeping original clip timing")
            clip_words = words
    report(f"Analysing video framing for clip {idx}...")
    log.info("Clip %s stage=probe starting", idx)
    meta = V.probe(raw)
    hook_text = clip.get("hook") or clip["title"]

    ass_name = None
    if opts.captions or opts.hook or opts.watermark:
        ass_name = f"cap_{idx}.ass"
        V.build_ass(
            clip_words, clip["start"], clip["end"], os.path.join(workdir, ass_name),
            hook=hook_text if opts.hook else None, captions=opts.captions,
            watermark=opts.watermark, caption_style=opts.caption_style,
        )

    log.info("Clip %s stage=layout starting", idx)
    plan = V.plan_layout(raw, meta, opts.layout)
    log.info("Clip %s stage=layout success kind=%s", idx, plan["kind"])
    out = os.path.join(workdir, f"clip_{idx}.mp4")
    report(f"Rendering vertical video for clip {idx}...")
    log.info("Clip %s stage=render starting", idx)
    spoken_text = " ".join(
        str(w.get("word", "")) for w in clip_words
        if w.get("start", 0) >= clip["start"] and w.get("end", 0) <= clip["end"]
    )
    mood_text = " ".join((clip.get("title", ""), clip.get("description", ""), spoken_text))
    music_mood, energy = classify_music_mood(mood_text) if (opts.music or opts.grade) else ("calm", 3.0)
    log.info("Clip %s mood=%s energy=%.1f", idx, music_mood, energy)
    voiceover_path = None
    if opts.voiceover_language and opts.voiceover_language.lower() != "off":
        try:
            report(f"Translating and generating {opts.voiceover_language} voiceover for clip {idx}...")
            translated_speech = translate_text(spoken_text, opts.voiceover_language)
            voiceover_path = VO.generate(
                translated_speech, opts.voiceover_language, workdir, idx, V.probe(raw)["dur"]
            )
            log.info("Clip %s translated voiceover ready language=%s", idx, opts.voiceover_language)
        except Exception:
            log.exception("Voiceover generation failed; preserving original spoken audio")
            voiceover_path = None
    grade_vf = None
    if opts.grade:
        try:
            g = G.plan(raw, music_mood, energy)
            grade_vf = g["vf"] if g else None
        except Exception:
            log.exception("Grading failed; rendering ungraded")
    track = None
    if opts.music:
        try:
            if opts.music_path and os.path.isfile(opts.music_path):
                custom_features = M.analyse_audio(M._decode(opts.music_path))
                track = (opts.music_path, "user-upload", custom_features)
                log.info("Using user-uploaded soundtrack")
            else:
                track = M.pick(music_mood, energy, used_tracks)
        except Exception:
            log.exception("Custom soundtrack failed; falling back to the free music library")
            try:
                track = M.pick(music_mood, energy, used_tracks)
            except Exception:
                log.exception("Built-in music selection failed; continuing without music")
    try:
        V.render(raw, plan, ass_name, out, workdir, grade_vf=grade_vf, music=track, logo_path=opts.logo_path, voiceover_path=voiceover_path)
    except RuntimeError:
        if not track:
            raise
        log.exception("Render with music failed; retrying without music")
        track = None
        try:
            V.render(raw, plan, ass_name, out, workdir, grade_vf=grade_vf, music=None, logo_path=opts.logo_path, voiceover_path=voiceover_path)
        except RuntimeError:
            log.exception("Render with grade failed; retrying plain")
            grade_vf = None
            V.render(raw, plan, ass_name, out, workdir, grade_vf=None, music=None, logo_path=opts.logo_path, voiceover_path=voiceover_path)
    log.info("Clip %s stage=render success bytes=%s", idx, os.path.getsize(out))
    if os.path.getsize(out) > C.TG_FILE_LIMIT:
        V.render(raw, plan, ass_name, out, workdir, crf=32, grade_vf=grade_vf, music=track, logo_path=opts.logo_path)
    if opts.outro_text:
        try:
            out = V.append_outro(out, workdir, idx, opts.outro_text)
        except Exception:
            log.exception("Outro rendering failed; keeping clip without outro")

    result = {"video": out, "layout": plan["kind"], "track": track[1] if track else None}
    try:
        vtt_path = os.path.join(workdir, f"subtitles_{idx}.vtt")
        V.build_vtt(clip_words, clip["start"], clip["end"], vtt_path)
        result["vtt"] = vtt_path
    except Exception:
        log.exception("WebVTT export failed")
    if opts.export_formats == "all":
        try:
            result["variants"] = V.render_social_variants(out, workdir, idx)
        except Exception:
            log.exception("Square/landscape export failed; keeping vertical version")
    try:
        srt_path = os.path.join(workdir, f"subtitles_{idx}.srt")
        V.build_srt(clip_words, clip["start"], clip["end"], srt_path)
        result["subtitles"] = srt_path
        if opts.subtitle_language and opts.subtitle_language.lower() != "off":
            translated_srt = os.path.join(workdir, f"subtitles_{idx}_{re.sub(r'[^a-z0-9]+', '_', opts.subtitle_language.lower())}.srt")
            translate_subtitle_file(srt_path, opts.subtitle_language, translated_srt)
            result["translated_subtitles"] = translated_srt
    except Exception:
        log.exception("subtitle export failed")
    try:
        report(f"Creating thumbnails for clip {idx}...")
        log.info("Clip %s stage=thumbnails starting", idx)
        result |= V.make_thumbnails(raw, hook_text, workdir, idx)
        log.info("Clip %s stage=thumbnails success", idx)
    except Exception:
        log.exception("thumbnail generation failed")
    log.info("Clip %s stage=complete", idx)
    return result




def translate_text(text: str, target_language: str) -> str:
    """Translate spoken dialogue for optional voiceover, retaining names and meaning."""
    prompt = (
        f"Translate the following spoken dialogue into {target_language}. "
        "Preserve meaning, tone and names. Do not add explanations or speaker labels. "
        "Return JSON only: {\\"translation\\": \\"translated dialogue\\"}."
    )
    payload = (text or "")[:5500]
    try:
        response = _ask(C.LLM_MODEL, prompt, payload)
    except Exception:
        response = _ask(C.LLM_FALLBACK_MODEL, prompt, payload)
    result = json.loads(response.choices[0].message.content)
    translated = str(result.get("translation", "")).strip()
    if not translated:
        raise RuntimeError("The translation model returned empty voiceover text")
    return translated


def summarize_transcript(segments) -> str:
    """Produce chaptered show notes from transcript windows while preserving timestamps."""
    windows = build_windows(segments, window_sec=1800)
    if not windows:
        return "No transcript text was available."
    reports = []
    prompt = (
        "You are an editor creating useful show notes from a transcript. "
        "Summarise this section in 2-4 concise bullets, identify up to 3 chapter headings "
        "with approximate timestamps copied from the transcript, and list key takeaways. "
        "Do not invent facts. Return JSON only: "
        '{"summary":"...","chapters":[{"time":"00:00","title":"..."}],"takeaways":["..."]}.'
    )
    for index, window in enumerate(windows[:8], 1):
        try:
            response = _ask(C.LLM_MODEL, prompt, window[:18000])
        except Exception:
            response = _ask(C.LLM_FALLBACK_MODEL, prompt, window[:18000])
        data = json.loads(response.choices[0].message.content)
        section = [f"PART {index}", str(data.get("summary", "")).strip()]
        chapters = data.get("chapters") or []
        if chapters:
            section.append("Chapters: " + "; ".join(
                f"{x.get('time','')} {x.get('title','')}" for x in chapters[:3]
            ))
        takeaways = data.get("takeaways") or []
        if takeaways:
            section.append("Key takeaways: " + " • ".join(str(x) for x in takeaways[:5]))
        reports.append("\n".join(x for x in section if x))
    return "\n\n".join(reports)


def generate_title_ideas(segments) -> list[str]:
    """Generate accurate alternate titles and hooks from transcript evidence."""
    transcript = "\n".join(
        f"[{s['start']:.0f}s] {s.get('text','')}" for s in segments
    )[:14000]
    prompt = (
        "You are a short-form video headline editor. Generate 10 distinct, punchy, "
        "accurate clip titles/hooks grounded only in this transcript. Mix curiosity, "
        "specific benefit, surprising claim and story-led styles. Avoid fake promises, "
        "fabricated quotes and generic clickbait. Return JSON only: "
        '{"titles":["..."]}. Keep each under 9 words.'
    )
    try:
        response = _ask(C.LLM_MODEL, prompt, transcript)
    except Exception:
        response = _ask(C.LLM_FALLBACK_MODEL, prompt, transcript)
    values = json.loads(response.choices[0].message.content).get("titles", [])
    return [str(x).strip()[:100] for x in values[:10] if str(x).strip()]
