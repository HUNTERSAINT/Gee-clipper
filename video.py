"""Video side: face tracking, 9:16 layouts, captions, ffmpeg render, thumbnails."""
import json
import logging
import math
import os
import re
import statistics
import subprocess
import textwrap
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

log = logging.getLogger("clipper.video")

OUT_W, OUT_H = 720, 1280

_HC = cv2.data.haarcascades
_FRONTAL = cv2.CascadeClassifier(_HC + "haarcascade_frontalface_default.xml")
_PROFILE = cv2.CascadeClassifier(_HC + "haarcascade_profileface.xml")
_EYES = cv2.CascadeClassifier(_HC + "haarcascade_eye.xml")


def run(cmd: list[str], cwd: str | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-600:]}")
    return p.stdout


def probe(path: str) -> dict:
    out = run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", path,
        ]
    )
    j = json.loads(out)
    s = j["streams"][0]
    num, _, den = s["r_frame_rate"].partition("/")
    d = float(den) if den else 1.0
    fps = float(num) / d if d else 25.0
    return {
        "w": int(s["width"]),
        "h": int(s["height"]),
        "fps": fps or 25.0,
        "dur": float(j["format"]["duration"]),
    }


# --------------------------------------------------------------------------- #
# face tracking -> smooth 9:16 crop
# --------------------------------------------------------------------------- #
def _small_gray(frame, width: int = 640):
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _detect(gray):
    """Return all usable frontal/profile face boxes, not just the largest face."""
    if _FRONTAL.empty():
        return []
    side = max(24, gray.shape[1] // 30)
    found = []
    for img in (gray, cv2.equalizeHist(gray)):
        for classifier in (_FRONTAL, _PROFILE):
            if classifier.empty():
                continue
            boxes = classifier.detectMultiScale(
                img, scaleFactor=1.08, minNeighbors=4, minSize=(side, side)
            )
            for box in boxes:
                x, y, w, h = map(int, box)
                # De-duplicate near-identical boxes from equalised/original passes.
                if not any(
                    abs((x + w / 2) - (a + c / 2)) < max(w, c) * 0.35
                    and abs((y + h / 2) - (b + d / 2)) < max(h, d) * 0.35
                    for a, b, c, d in found
                ):
                    found.append((x, y, w, h))
    return found


def _choose_face(faces, previous=None):
    """Prefer the same face over time; use size only to choose the first face."""
    if not faces:
        return None
    if previous is None:
        return max(faces, key=lambda f: f[2] * f[3])
    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2, py + ph / 2

    def score(face):
        x, y, w, h = face
        cx, cy = x + w / 2, y + h / 2
        distance = math.hypot((cx - pcx) / max(1, pw), (cy - pcy) / max(1, ph))
        size_change = abs(math.log(max(1, w * h) / max(1, pw * ph)))
        return distance + 0.22 * size_change

    candidate = min(faces, key=score)
    # If the closest detection is implausibly far away, a new shot/speaker likely began.
    if score(candidate) > 2.4:
        return max(faces, key=lambda f: f[2] * f[3])
    return candidate


def sample_faces(path: str, every: float = 0.35):
    """Sample faces and maintain identity continuity to avoid jumping to the largest face."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every)))
    samples, previous, i = [], None, 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    g = _small_gray(frame)
                    faces = _detect(g)
                    chosen = _choose_face(faces, previous)
                    if chosen is not None:
                        x, y, w, h = chosen
                        previous = chosen
                        samples.append(
                            (i / fps, (x + w / 2) / g.shape[1], h / g.shape[0])
                        )
                    else:
                        samples.append((i / fps, None, None))
            i += 1
    finally:
        cap.release()
    return samples


def plan_crop(samples, src_w, src_h, hold=1.6, thresh=0.10, min_rate=0.25):
    """Pan only when the speaker moves (or changes); hold still otherwise."""
    if not samples:
        return None
    detected = [s for s in samples if s[1] is not None]
    if len(detected) / len(samples) < min_rate:
        return None
    cw = int(src_h * 9 / 16) // 2 * 2
    if cw >= src_w:
        return None

    xs, last = [], None
    for _, cx, _ in samples:
        if cx is not None:
            last = cx
        xs.append(last)
    first = next(x for x in xs if x is not None)
    xs = [first if x is None else x for x in xs]
    sm = [statistics.median(xs[max(0, i - 3): i + 4]) for i in range(len(xs))]

    def left(cx):
        return min(max(cx * src_w - cw / 2, 0), src_w - cw)

    cur = left(sm[0])
    keys, last_t = [(0.0, cur)], 0.0
    for (t, _, _), cx in zip(samples, sm):
        target = left(cx)
        if abs(target - cur) > thresh * cw and t - last_t >= hold:
            keys.append((t, target))
            cur, last_t = target, t
    return {"kind": "crop", "cw": cw, "keys": keys}


def x_expr(keys, ramp: float = 0.45) -> str:
    """ffmpeg expression: eased pans between keyframes."""
    expr = f"{keys[0][1]:.1f}"
    for (_, prev), (t, x) in zip(keys, keys[1:]):
        expr += f"+({x - prev:.1f})*clip((t-{t:.2f})/{ramp},0,1)"
    return expr


def plan_layout(raw: str, meta: dict, layout: str = "auto") -> dict:
    w, h = meta["w"], meta["h"]
    if w / h <= 9 / 16 + 0.02:
        return {"kind": "fill"}  # already vertical
    if layout == "blur":
        return {"kind": "blur"}
    plan = plan_crop(sample_faces(raw), w, h)
    if plan:
        return plan
    return {"kind": "blur" if layout == "auto" else "center"}


# --------------------------------------------------------------------------- #
# captions (karaoke highlight) + hook title
# --------------------------------------------------------------------------- #
def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _group_words(ws, max_words: int = 5, max_chars: int = 28):
    groups, cur = [], []
    for w in ws:
        txt = w["word"]
        used = sum(len(x["word"]) for x in cur) + len(cur)
        if cur and (len(cur) >= max_words or used + len(txt) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if txt.rstrip().endswith((".", "?", "!", ";", ":")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _bgr(rgb: str) -> str:
    rgb = re.sub(r"[^0-9A-Fa-f]", "", rgb or "")[:6].rjust(6, "0")
    return rgb[4:6] + rgb[2:4] + rgb[0:2]


def _num_style(word: str) -> str:
    """Numbers, money and percentages stay green so the key figure is always easy to spot."""
    t = _ass_text(word)
    if any(c.isdigit() for c in word) or "$" in word or "%" in word:
        return r"{\c&H7CE86A&}" + t + r"{\c&HFFFFFF&}"
    return t


def _ass_text(s: str) -> str:
    return re.sub(r"[{}\\]", "", s).upper()


def _srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def build_srt(words, clip_start: float, clip_end: float, path: str):
    selected = [w for w in words if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.3]
    groups, current = [], []
    for word in selected:
        if current and (len(current) >= 8 or word["start"] - current[0]["start"] >= 4.0):
            groups.append(current)
            current = []
        current.append(word)
        if word["word"].rstrip().endswith((".", "?", "!")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    blocks = []
    for index, group in enumerate(groups, 1):
        start = max(0.0, group[0]["start"] - clip_start)
        end = max(start + 0.12, group[-1]["end"] - clip_start)
        text = " ".join(w["word"].strip() for w in group).strip()
        blocks.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(blocks) + ("\n" if blocks else ""))


_FILLERS = {"um", "uh", "uhh", "umm", "erm", "so", "and", "but", "like", "okay", "ok", "well", "yeah", "right", "i mean"}


def intro_start(words, max_search: float = 6.0) -> float:
    """Start time of the first real sentence, so a clip opens on a strong line instead of
    a half sentence or filler. Returns 0 when the clip already opens well."""
    for i, w in enumerate(words):
        if w["start"] > max_search:
            break
        token = w["word"].strip()
        prev = words[i - 1]["word"].rstrip() if i else ""
        starts_sentence = token[:1].isupper() and (i == 0 or prev.endswith((".", "?", "!")))
        if starts_sentence and token.lower().strip(",.") not in _FILLERS:
            return 0.0 if i == 0 else float(w["start"])
    return 0.0


def zoom_moments(words, dur: float, avoid=(), max_n: int = 4, gap: float = 3.5):
    """Times of emphasised lines (questions, exclamations, numbers) for a short push-in."""
    out, last = [], -99.0
    sentence_start = 0
    for i, w in enumerate(words):
        t = w["word"].rstrip()
        strong = t.endswith(("!", "?")) or any(c.isdigit() for c in t) or "%" in t or "$" in t
        if strong:
            ts = float(words[sentence_start]["start"]) if t.endswith(("!", "?")) else float(w["start"])
            if (1.0 <= ts <= dur - 2.5 and ts - last >= gap
                    and not any(a <= ts <= b for a, b in avoid)):
                out.append(round(ts, 2))
                last = ts
        if t.endswith((".", "?", "!")):
            sentence_start = min(i + 1, len(words) - 1)
        if len(out) >= max_n:
            break
    return out


def _stream_start(path: str, kind: str) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=start_time",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(p.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0


def normalize_timeline(path: str) -> str:
    """Make audio and video both start at exactly 0. Cuts from yt-dlp can start a fraction of a
    second (or more) into the container, which shifts every caption against the speech."""
    vs, a_start = _stream_start(path, "v:0"), _stream_start(path, "a:0")
    if abs(vs) < 0.02 and abs(a_start) < 0.02:
        return path
    out = os.path.splitext(path)[0] + "_t0.mp4"
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip() != ""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", "setpts=PTS-STARTPTS"]
    if has_audio:
        cmd += ["-af", "asetpts=PTS-STARTPTS"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", out]
    run(cmd)
    log.info("Timeline normalised video_start=%.3f audio_start=%.3f", vs, a_start)
    return out


def remove_silence(path: str, words: list, clip_start: float, clip_end: float,
                  max_gap: float = 1.15, padding: float = 0.18):
    """Trim long transcript gaps while preserving natural breaths and remap word timestamps."""
    relevant = [w for w in words if w.get("end", 0) > clip_start and w.get("start", 0) < clip_end]
    if len(relevant) < 3:
        return path, words, clip_end - clip_start
    spans = []
    for word in relevant:
        start = max(0.0, float(word["start"]) - clip_start - padding)
        end = min(clip_end - clip_start, float(word["end"]) - clip_start + padding)
        if spans and start - spans[-1][1] <= max_gap:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    original = clip_end - clip_start
    kept = sum(max(0.0, b - a) for a, b in spans)
    if len(spans) < 1 or original - kept < 0.8:
        return path, words, original
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    has_audio = has_audio.returncode == 0 and bool(has_audio.stdout.strip())
    parts, concat_inputs, remapped, offset = [], [], [], 0.0
    for index, (start, end) in enumerate(spans):
        parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]")
        concat_inputs.append(f"[v{index}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]")
            concat_inputs.append(f"[a{index}]")
        for word in relevant:
            ws, we = float(word["start"]) - clip_start, float(word["end"]) - clip_start
            if ws >= start - 0.02 and we <= end + 0.02:
                remapped.append({
                    "word": word.get("word", ""),
                    "start": offset + max(0.0, ws - start),
                    "end": offset + max(0.0, we - start),
                })
        offset += end - start
    n = len(spans)
    if has_audio:
        parts.append("".join(concat_inputs) + f"concat=n={n}:v=1:a=1[vout][aout]")
    else:
        parts.append("".join(concat_inputs) + f"concat=n={n}:v=1:a=0[vout]")
    out = os.path.splitext(path)[0] + "_tight.mp4"
    maps = ["-map", "[vout]"] + (["-map", "[aout]"] if has_audio else [])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-filter_complex", ";".join(parts), *maps,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out])
    log.info("Silence trim original=%.2fs kept=%.2fs removed=%.2fs", original, offset, original-offset)
    return out, remapped, offset


def build_vtt(words, clip_start: float, clip_end: float, path: str):
    """Write WebVTT captions with clip-relative timestamps."""
    selected = [w for w in words if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.3]
    groups, current = [], []
    for word in selected:
        if current and (len(current) >= 8 or word["start"] - current[0]["start"] >= 4.0):
            groups.append(current)
            current = []
        current.append(word)
        if word["word"].rstrip().endswith((".", "?", "!")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    def stamp(seconds):
        ms = max(0, int(round(seconds * 1000)))
        hh, ms = divmod(ms, 3600000)
        mm, ms = divmod(ms, 60000)
        ss, ms = divmod(ms, 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
    blocks = ["WEBVTT"]
    for group in groups:
        start = max(0.0, group[0]["start"] - clip_start)
        end = max(start + 0.12, group[-1]["end"] - clip_start)
        caption = " ".join(w["word"].strip() for w in group).strip()
        blocks.append(f"{stamp(start)} --> {stamp(end)}\n{caption}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(blocks) + "\n")


def render_social_variants(source: str, workdir: str, idx: int, formats=("square", "landscape")) -> dict:
    """Create 1:1 and 16:9 versions with blurred fill and the finished vertical clip centered."""
    outputs = {}
    dimensions = {"square": (1080, 1080), "landscape": (1280, 720)}
    for fmt in formats:
        if fmt not in dimensions:
            continue
        width, height = dimensions[fmt]
        target = os.path.join(workdir, f"clip_{idx}_{fmt}.mp4")
        graph = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=28:8[back];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[front];"
            f"[back][front]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
        )
        run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", source,
            "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart", target,
        ])
        outputs[fmt] = target
    return outputs


def build_ass(words, clip_start, clip_end, path, hook=None, hook_secs=3.0,
              captions=True, watermark=None, caption_style="bold", banner=None, accent=None):
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {OUT_W}\nPlayResY: {OUT_H}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,Liberation Sans,{({'bold': 64, 'classic': 54, 'neon': 60, 'minimal': 44}.get(caption_style, 64))},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"1,0,0,0,100,100,0,0,1,{({'bold': 6, 'classic': 4, 'neon': 5, 'minimal': 2}.get(caption_style, 6))},1,2,93,93,340,1\n"
        "Style: Hook,Liberation Sans,50,&H00FFFFFF,&H000000FF,&H40000000,&H64000000,"
        "1,0,0,0,100,100,0,0,3,14,0,8,93,93,150,1\n"
        "Style: Header,Liberation Sans,38,&H00FFFFFF,&H000000FF,&HB0000000,&H64000000,"
        "1,0,0,0,100,100,0,0,3,16,0,8,150,93,100,1\n"
        "Style: Watermark,Liberation Sans,24,&H80FFFFFF,&H000000FF,&H80000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,2,1,7,35,35,35,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    if watermark:
        wm = _ass_text(str(watermark)[:40])
        lines.append(f"Dialogue: 2,{ass_time(0)},{ass_time(clip_end - clip_start)},Watermark,,0,0,0,,{wm}")
    if banner:
        head = r"\N".join(textwrap.wrap(_ass_text(banner), 22)[:2])
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(clip_end - clip_start)},Header,,0,0,0,,{head}")
    if hook:
        wrapped = r"\N".join(textwrap.wrap(_ass_text(hook), 20)[:3])
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(hook_secs)},Hook,,0,0,0,,{wrapped}"
        )
    if captions:
        ws = [
            w for w in words
            if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.3
        ]
        groups = _group_words(ws)
        for gi, group in enumerate(groups):
            nxt_group = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
            for j, w in enumerate(group):
                start = w["start"] - clip_start
                if j + 1 < len(group):
                    end = group[j + 1]["start"] - clip_start
                else:
                    end = w["end"] - clip_start
                    if nxt_group is not None and nxt_group - clip_start - end < 0.4:
                        end = nxt_group - clip_start
                end = max(end, start + 0.05)
                hi = _bgr(accent) if accent else {"bold": "00FFFF", "classic": "0000FF", "neon": "00FFFF", "minimal": "00D7FF"}.get(caption_style, "00FFFF")
                text = " ".join(
                    (r"{\c&H" + hi + "&}" + _ass_text(x["word"]) + r"{\c&HFFFFFF&}")
                    if k == j else _num_style(x["word"])
                    for k, x in enumerate(group)
                )
                lines.append(
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}"
                )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# render the 9:16 clip
# --------------------------------------------------------------------------- #
def render(raw: str, plan: dict, ass_name: str | None, out: str, workdir: str,
           crf: int = 24, grade_vf: str | None = None, music=None,
           logo_path: str | None = None, voiceover_path: str | None = None,
           broll: list | None = None, commentary_path: str | None = None,
           facecam_path: str | None = None,
           zoom: list | None = None, sfx_path: str | None = None) -> None:
    grade = ("," + grade_vf) if grade_vf else ""
    kind = plan["kind"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", raw]
    n_in = 1

    def add_input(*args):
        nonlocal n_in
        cmd.extend(args)
        n_in += 1
        return n_in - 1

    music_input = voiceover_input = commentary_input = facecam_input = None
    if music:
        import music as M
        music_path, music_slug, mfeat = music
        music_input = add_input("-stream_loop", "-1", "-i", music_path)  # always input 1
        log.info("Clip soundtrack track=%s", music_slug)
    if voiceover_path and os.path.isfile(voiceover_path):
        voiceover_input = add_input("-i", voiceover_path)
    if commentary_path and os.path.isfile(commentary_path):
        commentary_input = add_input("-i", commentary_path)
    if kind == "split" and facecam_path and os.path.isfile(facecam_path):
        facecam_input = add_input("-stream_loop", "-1", "-i", facecam_path)
    elif kind == "split":
        kind = plan["kind"] = "frame"  # no face-cam supplied: fall back to the framed preset

    sfx_input = add_input("-i", sfx_path) if sfx_path and os.path.isfile(sfx_path) else None

    # ---- picture: base layout -> [vbase] ------------------------------------
    if kind == "crop":
        video_graph = (
            f"[0:v]crop=w={plan['cw']}:h=ih:x='{x_expr(plan['keys'])}':y=0,"
            f"scale={OUT_W}:{OUT_H},setsar=1{grade}[vbase]"
        )
    elif kind == "center":
        video_graph = (
            "[0:v]crop=w='trunc(ih*9/16/2)*2':h=ih,"
            f"scale={OUT_W}:{OUT_H},setsar=1{grade}[vbase]"
        )
    elif kind == "fill":
        video_graph = (
            f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},setsar=1{grade}[vbase]"
        )
    elif kind in ("frame", "split", "stack"):
        import layouts
        video_graph = layouts.graph(kind, plan, grade, facecam_input)
    else:
        video_graph = (
            "[0:v]split[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=25:6,eq=brightness=-0.12[bg];"
            f"[b]scale={OUT_W}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{grade}[vbase]"
        )

    # ---- subtle push-in on emphasised moments ---------------------------------
    label = "[vbase]"
    if zoom and kind not in ("stack", "split"):
        z = "+".join(f"max(0,min(min((t-{a:.2f})/0.5,({a:.2f}+1.3-t)/0.8),1))" for a in zoom)
        video_graph += (
            f";[vbase]scale=w='trunc({OUT_W}*(1+0.07*min(1,{z}))/2)*2':"
            f"h='trunc({OUT_H}*(1+0.07*min(1,{z}))/2)*2':eval=frame:flags=bilinear,"
            f"crop={OUT_W}:{OUT_H},setsar=1[vzoom]"
        )
        label = "[vzoom]"

    # ---- B-roll cutaways above the speaker, under captions ------------------
    for i, b in enumerate([b for b in (broll or []) if os.path.isfile(b.get("path", ""))]):
        bi = add_input("-i", b["path"])
        s0, e0 = b["t"], b["t"] + b["d"]
        video_graph += (
            f";[{bi}:v]scale={OUT_W - 120}:{int((OUT_W - 120) * 0.62) // 2 * 2}:"
            "force_original_aspect_ratio=increase,"
            f"crop={OUT_W - 120}:{int((OUT_W - 120) * 0.62) // 2 * 2},setsar=1,fps=30,"
            f"trim=duration={b['d']:.2f},setpts=PTS-STARTPTS+{s0:.2f}/TB,format=yuva420p,"
            f"fade=t=in:st={s0:.2f}:d=0.3:alpha=1,fade=t=out:st={e0 - 0.3:.2f}:d=0.3:alpha=1[br{i}];"
            f"{label}[br{i}]overlay=x=(W-w)/2:y={270 if kind == "frame" else 190}:eof_action=pass:"
            f"enable='between(t,{s0:.2f},{e0:.2f})'[vb{i}]"
        )
        label = f"[vb{i}]"

    # ---- captions / header ---------------------------------------------------
    if ass_name:
        video_graph += f";{label}ass={ass_name}[vcap]"
        label = "[vcap]"

    # ---- logo (top-left: bottom and right edges are covered by platform UI) --
    if logo_path and os.path.isfile(logo_path):
        logo_input = add_input("-i", logo_path)
        video_graph += (
            f";[{logo_input}:v]scale=110:-1,format=rgba[brandlogo];"
            f"{label}[brandlogo]overlay=30:100[vbranded]"
        )
        label = "[vbranded]"
    video_label = label

    # ---- audio ---------------------------------------------------------------
    audio_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", raw],
        capture_output=True, text=True,
    )
    has_audio = audio_probe.returncode == 0 and bool(audio_probe.stdout.strip())
    dur = probe(raw)["dur"]
    base = f"[{voiceover_input}:a:0]" if voiceover_input is not None else ("[0:a:0]" if has_audio else None)
    pre, voice_ref = "", base
    if commentary_input is not None:
        cm = (f"[{commentary_input}:a:0]aresample=48000,highpass=f=90,afftdn=nf=-30,"
              "loudnorm=I=-16:TP=-2:LRA=9")
        if base:
            # While the creator speaks the source drops to ~15% and rises again smoothly.
            pre = (f"{base}aresample=48000[src0];{cm},asplit=2[cm][cmsc0];"
                   "[cmsc0]apad[cmsc];"
                   "[src0][cmsc]sidechaincompress=threshold=0.038:ratio=20:attack=20:release=450:"
                   "makeup=1[srcduck];"
                   "[srcduck][cm]amix=inputs=2:duration=first:dropout_transition=0:weights='1 1',"
                   "volume=2[voicebus]")
        else:
            pre = f"{cm}[voicebus]"
        voice_ref = "[voicebus]"
    if music:
        audio_graph = M.mix_filter(bool(voice_ref), mfeat, dur)
        if voice_ref:
            audio_graph = audio_graph.replace("[0:a:0]", voice_ref, 1)
        graph = video_graph + ";" + (pre + ";" if pre else "") + audio_graph
        maps = ["-map", video_label, "-map", "[aout]", "-t", f"{dur:.2f}"]
        log.info("Clip soundtrack track=%s voice_present=%s voiceover=%s commentary=%s",
                 music_slug, bool(voice_ref), voiceover_input is not None, commentary_input is not None)
    elif voice_ref and (voiceover_input is not None or commentary_input is not None):
        graph = (
            video_graph + ";" + (pre + ";" if pre else "")
            + f"{voice_ref}aresample=48000,loudnorm=I=-16:TP=-2:LRA=9,"
            f"apad,atrim=duration={dur:.2f},alimiter=limit=0.95[aout]"
        )
        maps = ["-map", video_label, "-map", "[aout]", "-t", f"{dur:.2f}"]
    else:
        graph = video_graph
        maps = ["-map", video_label] + (["-map", "0:a:0"] if has_audio else [])
    if sfx_input is not None:
        if "[aout]" in graph:
            graph = graph.rsplit("[aout]", 1)[0] + "[amain]"
            tail = (f";[amain][{sfx_input}:a:0]amix=inputs=2:duration=first:dropout_transition=0:"
                    "weights='1 1',volume=2,alimiter=limit=0.95[aout]")
        elif has_audio:
            tail = (f";[0:a:0]aresample=48000[amain];[amain][{sfx_input}:a:0]amix=inputs=2:duration=first:"
                    "dropout_transition=0:weights='1 1',volume=2,alimiter=limit=0.95[aout]")
        else:
            tail = f";[{sfx_input}:a:0]anull[aout]"
        graph += tail
        maps = ["-map", video_label, "-map", "[aout]", "-t", f"{dur:.2f}"]
    if "-t" not in maps:
        maps += ["-t", f"{dur:.2f}"]  # looped inputs (face-cam, music) must not run forever
    cmd += ["-filter_complex", graph] + maps + [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out,
    ]
    run(cmd, cwd=workdir)


def append_outro(source: str, workdir: str, idx: int, text: str = "Follow for more") -> str:
    """Append a short branded end card while preserving source audio and adding a silent tail."""
    card = os.path.join(workdir, f"outro_{idx}.png")
    output = os.path.join(workdir, f"clip_{idx}_outro.mp4")
    image = Image.new("RGB", (OUT_W, OUT_H), (8, 28, 34))
    draw = ImageDraw.Draw(image)
    for y in range(OUT_H):
        blend = y / max(1, OUT_H - 1)
        color = (int(7 + 8 * blend), int(35 + 30 * blend), int(42 + 38 * blend))
        draw.line((0, y, OUT_W, y), fill=color)
    draw.rounded_rectangle((48, 420, OUT_W - 48, 860), radius=36, outline=(40, 190, 175), width=4)
    font, lines, stroke, line_h = _fit(draw, _clean(text)[:80], OUT_W - 120, 260, max_lines=3, start=76, stop=38)
    y0 = (OUT_H - line_h * len(lines)) // 2
    _draw_lines(draw, lines, font, stroke, line_h, 60, y0, OUT_W - 120, True)
    small = _font(22)
    brand = "MADE WITH GEE CLIPPER"
    bw = draw.textlength(brand, font=small)
    draw.text(((OUT_W - bw) / 2, 910), brand, font=small, fill=(160, 205, 205))
    image.save(card)
    probe_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", source],
        capture_output=True, text=True,
    )
    has_audio = probe_audio.returncode == 0 and bool(probe_audio.stdout.strip())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", source,
           "-loop", "1", "-framerate", "30", "-t", "1.8", "-i", card]
    if has_audio:
        cmd += ["-f", "lavfi", "-t", "1.8", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        graph = (
            "[0:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[v0];"
            "[0:a]aresample=48000,asetpts=PTS-STARTPTS[a0];"
            "[1:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[v1];"
            "[2:a]aresample=48000,asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        graph = (
            "[0:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[v0];"
            "[1:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[v1];"
            "[v0][v1]concat=n=2:v=1:a=0[v]"
        )
        maps = ["-map", "[v]"]
    run(cmd + ["-filter_complex", graph, *maps, "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output],
        cwd=workdir)
    return output


# --------------------------------------------------------------------------- #
# thumbnails: best frame + hook text (16:9 and 9:16 cover)
# --------------------------------------------------------------------------- #
_FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font(size: int):
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default(size)


def _clean(text: str) -> str:
    text = re.sub(r"[^\w\s'!?.,:%$&+\-]", "", text or "")
    return re.sub(r"\s+", " ", text).strip().upper()


def best_frame(path: str, n: int = 14):
    """Pick the sharpest, best-lit frame, preferring big faces with open eyes."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    best = None
    for k in range(n):
        t = dur * (0.06 + 0.8 * k / max(1, n - 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = math.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) / 6
        score = sharp - abs(float(gray.mean()) - 115) / 115 * 0.6
        face = None
        g = _small_gray(frame)
        faces = _detect(g)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            fh = h / g.shape[0]
            score += min(fh, 0.5) * 3
            if (not _EYES.empty()
                    and len(_EYES.detectMultiScale(g[y:y + h, x:x + w], 1.1, 6)) >= 2):
                score += 0.6
            face = ((x + w / 2) / g.shape[1], (y + h / 2) / g.shape[0], fh)
        if best is None or score > best[0]:
            best = (score, frame, face)
    if best is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Could not read any frame for the thumbnail")
        best = (0, frame, None)
    cap.release()
    img = Image.fromarray(cv2.cvtColor(best[1], cv2.COLOR_BGR2RGB))
    return img, best[2]


def _cover(img: Image.Image, W: int, H: int, cx: float = 0.5) -> Image.Image:
    s = max(W / img.width, H / img.height)
    nw, nh = int(img.width * s + 0.5), int(img.height * s + 0.5)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = int(min(max(cx * nw - W / 2, 0), nw - W))
    top = (nh - H) // 2
    return img.crop((left, top, left + W, top + H))


def _enhance(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Color(img).enhance(1.2)
    return ImageEnhance.Sharpness(img).enhance(1.25)


def _gradient(size, side: str, strength: int = 200, span: float = 0.62):
    W, H = size
    a = np.zeros((H, W), dtype=np.uint8)
    if side in ("left", "right"):
        n = int(W * span)
        ramp = np.linspace(strength, 0, n).astype(np.uint8)
        if side == "left":
            a[:, :n] = ramp[None, :]
        else:
            a[:, W - n:] = ramp[::-1][None, :]
    elif side == "bottom":
        n = int(H * span)
        a[H - n:, :] = np.linspace(0, strength, n).astype(np.uint8)[:, None]
    else:  # top
        n = int(H * span)
        a[:n, :] = np.linspace(strength, 0, n).astype(np.uint8)[:, None]
    layer = np.zeros((H, W, 4), dtype=np.uint8)
    layer[..., 3] = a
    return Image.fromarray(layer)


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _fit(draw, text, max_w, max_h, max_lines=3, start=150, stop=34):
    for size in range(start, stop - 1, -6):
        font = _font(size)
        stroke = max(3, size // 12)
        lines = _wrap(draw, text, font, max_w - 2 * stroke)
        if not lines or len(lines) > max_lines:
            continue
        line_h = int(size * 1.08)
        widest = max(draw.textlength(l, font=font) for l in lines)
        if widest <= max_w - 2 * stroke and line_h * len(lines) <= max_h:
            return font, lines, stroke, line_h
    font = _font(stop)
    stroke = max(3, stop // 12)
    return font, _wrap(draw, text, font, max_w)[:max_lines], stroke, int(stop * 1.08)


def _draw_lines(draw, lines, font, stroke, line_h, x0, y0, box_w, center):
    for i, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        x = x0 + (box_w - w) / 2 if center else x0
        color = (255, 221, 0) if i == len(lines) - 1 else (255, 255, 255)
        draw.text(
            (x, y0 + i * line_h), line, font=font, fill=color,
            stroke_width=stroke, stroke_fill=(0, 0, 0),
        )


def thumb_16x9(img, face, text: str, out: str) -> Image.Image:
    W, H = 1280, 720
    cx = face[0] if face else 0.5
    base = _enhance(_cover(img, W, H, cx)).convert("RGBA")
    side = "right" if (face and face[0] < 0.5) else "left"
    base = Image.alpha_composite(base, _gradient((W, H), side))
    if text:
        d = ImageDraw.Draw(base)
        box_w, margin = int(W * 0.52), int(W * 0.04)
        x0 = margin if side == "left" else W - margin - box_w
        font, lines, stroke, lh = _fit(d, text, box_w, int(H * 0.8))
        _draw_lines(d, lines, font, stroke, lh, x0, (H - lh * len(lines)) // 2,
                    box_w, center=False)
    base.convert("RGB").save(out, quality=90)
    return base


def cover_9x16(img, face, text: str, out: str) -> Image.Image:
    W, H = OUT_W, OUT_H
    cx = face[0] if face else 0.5
    base = _enhance(_cover(img, W, H, cx)).convert("RGBA")
    # keep the text off the speaker: face high in frame -> text goes at the bottom
    text_top = not face or face[1] >= 0.42
    base = Image.alpha_composite(
        base, _gradient((W, H), "top" if text_top else "bottom", span=0.45)
    )
    if text:
        d = ImageDraw.Draw(base)
        box_w = int(W * 0.88)
        font, lines, stroke, lh = _fit(d, text, box_w, int(H * 0.26), start=120)
        y0 = int(H * 0.07) if text_top else int(H * 0.93) - lh * len(lines)
        _draw_lines(d, lines, font, stroke, lh, (W - box_w) // 2, y0,
                    box_w, center=True)
    base.convert("RGB").save(out, quality=90)
    return base


def make_thumbnails(raw: str, hook: str, workdir: str, idx: int) -> dict:
    img, face = best_frame(raw)
    text = _clean(hook)
    p16 = os.path.join(workdir, f"thumb_{idx}.jpg")
    p9 = os.path.join(workdir, f"cover_{idx}.jpg")
    ps = os.path.join(workdir, f"cover_small_{idx}.jpg")
    thumb_16x9(img, face, text, p16)
    cover = cover_9x16(img, face, text, p9)
    cover.convert("RGB").resize((180, 320), Image.LANCZOS).save(ps, quality=85)
    return {"thumb": p16, "cover": p9, "cover_small": ps}
