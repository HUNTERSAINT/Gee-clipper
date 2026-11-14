"""Pre-flight quality report for a rendered clip: flags problems before you post it."""
import re
import subprocess

import cv2

import grade


def _loudness(path: str):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=90).stderr
    i = re.findall(r"I:\s+(-?[\d.]+) LUFS", r)
    tp = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r)
    return (float(i[-1]) if i else None), (float(tp[-1]) if tp else None)


def report(video: str, words: list, dur: float, hook: str, score: float | None,
           min_sec: float = 15, max_sec: float = 65) -> dict:
    flags, points = [], 10.0

    def flag(text, cost):
        nonlocal points
        flags.append(text)
        points -= cost

    if dur < min_sec:
        flag(f"Short ({dur:.0f}s): under {min_sec:.0f}s gives little to hold viewers", 1.0)
    if dur > max_sec:
        flag(f"Long ({dur:.0f}s) for a Short or Reel", 1.0)

    h = (hook or "").split()
    if not h:
        flag("No hook text on screen", 1.5)
    elif len(h) > 8:
        flag(f"Hook is {len(h)} words; 2-6 lands faster", 1.0)
    if score is not None and score < 6:
        flag(f"Moment rated {score:.0f}/10 by the AI: consider a stronger section", 1.5)

    if words and dur > 0:
        wps = len(words) / dur
        if wps > 4.3:
            flag(f"Very fast speech ({wps:.1f} words/s): captions will be hard to read", 1.0)
        elif wps < 1.2:
            flag(f"Sparse speech ({wps:.1f} words/s): may feel slow", 1.0)
        gaps = sum(max(0.0, b["start"] - a["end"]) for a, b in zip(words, words[1:]) if b["start"] - a["end"] > 1.0)
        if gaps / dur > 0.15:
            flag(f"{gaps / dur:.0%} of the clip is silence", 1.0)
    else:
        flag("No speech detected", 2.0)

    try:
        lufs, tp = _loudness(video)
        if lufs is not None and (lufs < -19 or lufs > -11):
            flag(f"Loudness {lufs:.0f} LUFS (aim for about -14)", 1.0)
        if tp is not None and tp > -0.5:
            flag(f"Audio peaks at {tp:.1f} dBFS and may clip", 1.0)
    except Exception:
        lufs = tp = None

    try:
        frames = grade._sample_frames(video, n=6, width=360)
        st = grade.measure(frames) if frames else None
        if st:
            if st["p50"] < 0.22:
                flag("Picture is dark", 1.0)
            elif st["p50"] > 0.75:
                flag("Picture is overexposed", 1.0)
            if st["sharp"] < 35:
                flag("Picture looks soft or blurry", 1.0)
    except Exception:
        pass

    return {"score": max(0.0, round(points, 1)), "flags": flags, "lufs": lufs, "true_peak": tp}


def text(rep: dict) -> str:
    head = f"Quality {rep['score']:.0f}/10"
    if not rep["flags"]:
        return f"✅ {head}: no issues found"
    return f"{'⚠️' if rep['score'] < 8 else '✅'} {head}\n" + "\n".join(f"• {f}" for f in rep["flags"])
