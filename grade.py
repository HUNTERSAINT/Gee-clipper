"""Adaptive per-clip colour grading.

Every clip is measured (exposure, contrast, colour cast, sharpness, faces) and gets
its own correction, then a "look" picked from the clip's mood. The result is one
ffmpeg filter string. Before it is used, it is run through ffmpeg on real frames
from the clip and checked (blown highlights, crushed blacks, skin tones, oversaturation);
if a check fails the look is dialled back until it passes.
"""
import logging
import math
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

log = logging.getLogger("clipper.grade")

# Curve control points: denser near the ends where grading matters most.
_XS = np.array([0.0, 0.06, 0.14, 0.25, 0.40, 0.55, 0.70, 0.84, 0.94, 1.0])

# contrast : blend towards an S-curve (0 = none)    lift : raised blacks (faded film)
# roll     : highlight roll-off                      exp  : shift of the target brightness
# sh/hi    : RGB offsets added to shadows/highlights  mid : RGB gain in the mid-tones
# sat/vib  : saturation multiplier / vibrance        vig  : vignette angle in rad, ~0.6 is ffmpeg default (None = off)
LOOKS = {
    "uplifting": dict(exp=0.02, contrast=0.28, lift=0.010, roll=0.03,
                      sh=(0.004, 0.000, -0.004), hi=(0.024, 0.008, -0.020),
                      mid=(1.030, 1.010, 0.960), sat=1.05, vib=0.20, vig=None, sharp=0.30),
    "dramatic": dict(exp=-0.03, contrast=0.50, lift=0.000, roll=0.02,
                     sh=(-0.018, 0.004, 0.026), hi=(0.022, 0.006, -0.016),
                     mid=(1.020, 1.000, 0.980), sat=0.96, vib=0.12, vig=0.58, sharp=0.35),
    "emotional": dict(exp=0.01, contrast=0.20, lift=0.040, roll=0.04,
                      sh=(0.000, 0.006, 0.010), hi=(0.018, 0.006, -0.010),
                      mid=(1.020, 1.000, 0.985), sat=0.93, vib=0.10, vig=0.50, sharp=0.0),
    "energetic": dict(exp=0.00, contrast=0.45, lift=0.000, roll=0.02,
                      sh=(0.000, 0.000, 0.008), hi=(0.010, 0.000, -0.006),
                      mid=(1.010, 1.000, 0.990), sat=1.10, vib=0.30, vig=None, sharp=0.45),
    "calm": dict(exp=0.00, contrast=0.18, lift=0.015, roll=0.03,
                 sh=(-0.004, 0.002, 0.008), hi=(0.000, 0.000, 0.000),
                 mid=(0.995, 1.000, 1.010), sat=1.00, vib=0.15, vig=None, sharp=0.25),
    "funny": dict(exp=0.04, contrast=0.28, lift=0.005, roll=0.02,
                  sh=(0.000, 0.000, 0.000), hi=(0.010, 0.006, 0.000),
                  mid=(1.015, 1.005, 0.985), sat=1.08, vib=0.25, vig=None, sharp=0.30),
}


# --------------------------------------------------------------------------- #
# measuring the clip
# --------------------------------------------------------------------------- #
def _sample_frames(path: str, n: int = 9, width: int = 480):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for k in range(n):
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (0.05 + 0.9 * k / max(1, n - 1))))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        if w > width:
            f = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames


def _luma(img_bgr):
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), rgb


def _faces(img_bgr):
    try:  # imported lazily: video.py imports nothing from here
        from video import _detect
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return [tuple(int(v) for v in f) for f in _detect(gray)]
    except Exception:
        return []


def measure(frames):
    lum, cast_px, sat, lap = [], [], [], []
    for f in frames:
        y, rgb = _luma(f)
        lum.append(y[::2, ::2].ravel())
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        mask = (y > 0.2) & (y < 0.8) & ((mx - mn) < 0.18)
        if mask.mean() > 0.01:
            cast_px.append(rgb[mask][::4])
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2] > 40
        if v.any():
            sat.append(float(hsv[..., 1][v].mean()) / 255.0)
        lap.append(float(cv2.Laplacian((y * 255).astype(np.uint8), cv2.CV_64F).var()))
    lum = np.concatenate(lum)
    p1, p50, p995 = np.percentile(lum, [1, 50, 99.5])

    wb = None
    if cast_px:
        px = np.concatenate(cast_px)
        if len(px) > 500:
            m = px.mean(axis=0)  # R, G, B
            gray = float(m.mean())
            gains = np.array([gray / max(m[0], 1e-3), gray / max(m[1], 1e-3), gray / max(m[2], 1e-3)])
            gains = 1.0 + (gains - 1.0) * 0.5  # only half the correction: warm scenes stay warm
            gains = np.clip(gains, 0.94, 1.06)
            if np.abs(gains - 1.0).max() >= 0.012:
                wb = tuple(float(g) for g in gains / gains.mean())
    return {
        "p1": float(p1), "p50": float(p50), "p995": float(p995),
        "sat": float(np.mean(sat)) if sat else 0.3,
        "sharp": float(np.median(lap)) if lap else 120.0,
        "wb": wb,
    }


# --------------------------------------------------------------------------- #
# building the filter
# --------------------------------------------------------------------------- #
def _fmt(vals) -> str:
    return " ".join(f"{x:.4f}/{y:.4f}" for x, y in zip(_XS, vals))


def _master(x, lo, hi, gamma, contrast, lift, roll):
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0) ** gamma
    y = (1 - contrast) * y + contrast * (3 * y * y - 2 * y ** 3)  # S-curve blend
    y = y - roll * y ** 5
    return lift + y * (1 - lift)


def _channel(y, c, sh, hi, mid):
    out = y + sh[c] * (1 - y) ** 3 + hi[c] * y ** 3
    out = out * (1 + (mid[c] - 1) * 4 * y * (1 - y))
    return np.maximum.accumulate(np.clip(out, 0.0, 1.0))


def build_vf(stats: dict, look: str, s: float = 1.0) -> str:
    L = LOOKS.get(look, LOOKS["calm"])
    lo = float(np.clip(stats["p1"], 0.0, 0.10)) * 0.85
    hi_ = float(np.clip(stats["p995"], 0.88, 1.0))
    hi_ = max(hi_, lo + 0.6)

    m1 = float(np.clip((stats["p50"] - lo) / (hi_ - lo), 0.05, 0.95))
    target = float(np.clip(0.46 + L["exp"] * s, 0.30, 0.56))
    mt = m1 + float(np.clip((target - m1) * 0.65, -0.10, 0.14))
    gamma = float(np.clip(math.log(mt) / math.log(m1), 0.7, 1.45))

    y = _master(_XS, lo, hi_, gamma, L["contrast"] * s, L["lift"] * s, L["roll"] * s)
    sh = tuple(v * s for v in L["sh"])
    hi = tuple(v * s for v in L["hi"])
    mid = tuple(1 + (v - 1) * s for v in L["mid"])

    parts = []
    if stats.get("wb"):
        r, g, b = stats["wb"]
        parts.append(f"colorchannelmixer=rr={r:.4f}:gg={g:.4f}:bb={b:.4f}")
    r_, g_, b_ = (_fmt(_channel(y, c, sh, hi, mid)) for c in range(3))
    parts.append(f"curves=r='{r_}':g='{g_}':b='{b_}':interp=pchip")

    vib = L["vib"] * s + float(np.clip((0.22 - stats["sat"]) * 1.5, 0.0, 0.25))  # flat footage gets a bit more
    parts.append(f"vibrance=intensity={vib:.3f}")
    sat = 1 + (L["sat"] - 1) * s
    if abs(sat - 1) > 0.005:
        parts.append(f"eq=saturation={sat:.3f}")
    if L["vig"]:
        angle = L["vig"] * s  # larger angle = stronger vignette in ffmpeg
        parts.append(f"vignette=angle={angle:.3f}")
    soft = stats["sharp"] < 60
    crisp = stats["sharp"] > 250
    amount = float(np.clip(L["sharp"] * (1.4 if soft else 0.7 if crisp else 1.0), 0.0, 0.7)) * s
    if amount > 0.05:
        parts.append(f"unsharp=5:5:{amount:.2f}:3:3:0")
    return ",".join(parts)


# --------------------------------------------------------------------------- #
# checking the result on real frames, through ffmpeg itself
# --------------------------------------------------------------------------- #
def _chroma(img):
    f = img.astype(np.float32) / 255.0
    return f.max(axis=2) - f.min(axis=2)


def _frame_metrics(img_bgr, faces):
    """Chroma (max-min) is used instead of HSV saturation: it does not jump when
    a grade merely darkens the picture."""
    y, _ = _luma(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ch = _chroma(img_bgr)
    out = {
        "clip_hi": float((y >= 0.985).mean()),
        "crush": float((y <= 0.015).mean()),
        "chroma": float(ch[y > 0.08].mean()) if (y > 0.08).any() else 0.0,
        "skin": [],
    }
    for (x, yy, w, h) in faces:
        x0, x1 = x + w // 4, x + 3 * w // 4
        y0, y1 = yy + h // 4, yy + 3 * h // 4
        patch = hsv[y0:y1, x0:x1, 0]
        if patch.size:
            out["skin"].append((float(patch.mean()), float(ch[y0:y1, x0:x1].mean())))
    return out


def _run_vf(frames, vf):
    td = tempfile.mkdtemp(prefix="grade_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(td, f"s{i}.png"), f)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-start_number", "0",
             "-i", os.path.join(td, "s%d.png"), "-vf", vf + ",format=rgb24",
             "-start_number", "0", os.path.join(td, "o%d.png")],
            check=True, capture_output=True, timeout=40,
        )
        return [cv2.imread(os.path.join(td, f"o{i}.png")) for i in range(len(frames))]
    finally:
        shutil.rmtree(td, ignore_errors=True)


def check(frames, vf):
    """Returns (ok, reason). Grades up to three real frames and inspects them."""
    pick = [frames[i] for i in sorted({int(i) for i in np.linspace(0, len(frames) - 1, min(3, len(frames)))})]
    outs = _run_vf(pick, vf)
    for src, dst in zip(pick, outs):
        if dst is None:
            return False, "ffmpeg produced no frame"
        faces = _faces(src)
        a, b = _frame_metrics(src, faces), _frame_metrics(dst, faces)
        if b["clip_hi"] > max(0.03, a["clip_hi"] + 0.015):
            return False, f"highlights clipping ({b['clip_hi']:.1%})"
        if b["crush"] > max(0.05, a["crush"] + 0.06):
            return False, f"blacks crushed ({b['crush']:.1%})"
        if b["chroma"] > max(a["chroma"] * 1.45 + 0.03, 0.17):
            return False, f"oversaturated (chroma {a['chroma']:.2f}->{b['chroma']:.2f})"
        for (h0, s0), (h1, s1) in zip(a["skin"], b["skin"]):
            if abs(h1 - h0) > 5.0 or s1 > s0 * 1.4 + 0.03:
                return False, f"skin tone shifted (hue {h0:.1f}->{h1:.1f}, chroma {s0:.2f}->{s1:.2f})"
    return True, ""


def plan(raw: str, mood: str = "calm", energy: float = 3.0) -> dict | None:
    """Grade for one clip, or None if the clip could not be analysed."""
    frames = _sample_frames(raw)
    if not frames:
        return None
    stats = measure(frames)
    look = mood if mood in LOOKS else "calm"
    s = 1.0
    vf, reason = "", ""
    for _ in range(4):
        vf = build_vf(stats, look, s)
        try:
            ok, reason = check(frames, vf)
        except Exception as exc:  # never let grading break a clip
            log.warning("Grade check failed (%s); using strength %.2f unchecked", type(exc).__name__, s)
            ok, reason = True, ""
        if ok:
            break
        log.info("Grade look=%s strength=%.2f rejected: %s", look, s, reason)
        s *= 0.7
    else:
        vf = build_vf(stats, look, 0.25)
        s = 0.25
    log.info(
        "Grade look=%s strength=%.2f median=%.2f p1=%.2f p99.5=%.2f sat=%.2f wb=%s",
        look, s, stats["p50"], stats["p1"], stats["p995"], stats["sat"], stats["wb"],
    )
    return {"vf": vf, "look": look, "strength": s, "stats": stats}
