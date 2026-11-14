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
    """Plain grayscale first; histogram-equalised only if nothing is found
    (equalising helps dark shots but can wash out flat, bright ones)."""
    if _FRONTAL.empty():  # cascade data missing (e.g. OpenCV 5): no face features
        return []
    side = max(24, gray.shape[1] // 25)
    for img in (gray, cv2.equalizeHist(gray)):
        faces = _FRONTAL.detectMultiScale(img, 1.1, 5, minSize=(side, side))
        if len(faces) == 0 and not _PROFILE.empty():
            faces = _PROFILE.detectMultiScale(img, 1.1, 5, minSize=(side, side))
        if len(faces):
            return faces
    return []


def sample_faces(path: str, every: float = 0.5):
    """Returns [(time, face_center_x 0-1 or None, face_height 0-1 or None)]."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every)))
    samples, i = [], 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                g = _small_gray(frame)
                faces = _detect(g)
                if len(faces):
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    samples.append(
                        (i / fps, (x + w / 2) / g.shape[1], h / g.shape[0])
                    )
                else:
                    samples.append((i / fps, None, None))
        i += 1
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


def _group_words(ws, max_words: int = 3, max_chars: int = 17):
    groups, cur = [], []
    for w in ws:
        txt = w["word"]
        used = sum(len(x["word"]) for x in cur) + len(cur)
        if cur and (len(cur) >= max_words or used + len(txt) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if txt.endswith((".", "?", "!")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


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
    if len(spans) < 2 or original - kept < 1.5:
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
              captions=True, watermark=None, caption_style="bold"):
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {OUT_W}\nPlayResY: {OUT_H}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,Liberation Sans,{({'bold': 64, 'classic': 54, 'neon': 60, 'minimal': 44}.get(caption_style, 64))},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"1,0,0,0,100,100,0,0,1,{({'bold': 6, 'classic': 4, 'neon': 5, 'minimal': 2}.get(caption_style, 6))},1,2,40,40,340,1\n"
        "Style: Hook,Liberation Sans,50,&H00FFFFFF,&H000000FF,&H40000000,&H64000000,"
        "1,0,0,0,100,100,0,0,3,14,0,8,50,50,150,1\n"
        "Style: Watermark,Liberation Sans,24,&H80FFFFFF,&H000000FF,&H80000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,2,1,9,35,35,35,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    if watermark:
        wm = _ass_text(str(watermark)[:40])
        lines.append(f"Dialogue: 2,{ass_time(0)},{ass_time(clip_end - clip_start)},Watermark,,0,0,0,,{wm}")
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
                text = " ".join(
                    (r"{\c&H" + {"bold": "00FFFF", "classic": "0000FF", "neon": "00FFFF", "minimal": "00D7FF"}.get(caption_style, "00FFFF") + "&}" + _ass_text(x["word"]) + r"{\c&HFFFFFF&}")
                    if k == j else _ass_text(x["word"])
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
           crf: int = 24, grade_vf: str | None = None, music=None, logo_path: str | None = None) -> None:
    grade = ("," + grade_vf) if grade_vf else ""
    ass = f",ass={ass_name}" if ass_name else ""
    kind = plan["kind"]
    if kind == "crop":
        vf = (
            f"crop=w={plan['cw']}:h=ih:x='{x_expr(plan['keys'])}':y=0,"
            f"scale={OUT_W}:{OUT_H},setsar=1{grade}{ass}"
        )
        video_graph = f"[0:v]{vf}[vout]"
    elif kind == "center":
        vf = (
            "crop=w='trunc(ih*9/16/2)*2':h=ih,"
            f"scale={OUT_W}:{OUT_H},setsar=1{grade}{ass}"
        )
        video_graph = f"[0:v]{vf}[vout]"
    elif kind == "fill":
        vf = (
            f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},setsar=1{grade}{ass}"
        )
        video_graph = f"[0:v]{vf}[vout]"
    else:
        video_graph = (
            "[0:v]split[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=25:6,eq=brightness=-0.12[bg];"
            f"[b]scale={OUT_W}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{grade}"
            f"{ass}[vout]"
        )

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", raw]
    if music:
        import music as M
        music_path, music_slug, mfeat = music
        cmd += ["-stream_loop", "-1", "-i", music_path]
        log.info("Clip soundtrack track=%s", music_slug)
    logo_input = None
    if logo_path and os.path.isfile(logo_path):
        logo_input = 2 if music else 1
        cmd += ["-i", logo_path]
        video_graph += (
            f";[{logo_input}:v]scale=150:-1,format=rgba[brandlogo];"
            "[vout][brandlogo]overlay=W-w-24:H-h-24[vbranded]"
        )
        video_label = "[vbranded]"
    else:
        video_label = "[vout]"

    audio_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", raw],
        capture_output=True, text=True,
    )
    has_audio = audio_probe.returncode == 0 and bool(audio_probe.stdout.strip())
    dur = probe(raw)["dur"]
    if music:
        graph = video_graph + ";" + M.mix_filter(has_audio, mfeat, dur)
        maps = ["-map", video_label, "-map", "[aout]", "-t", f"{dur:.2f}"]
        log.info("Clip soundtrack track=%s voice_present=%s", music_slug, has_audio)
    else:
        graph = video_graph
        maps = ["-map", video_label] + (["-map", "0:a:0"] if has_audio else [])
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
