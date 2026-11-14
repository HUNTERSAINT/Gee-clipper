"""Layout presets beyond the single-speaker crop: two-speaker stack, split screen, framed commentary.

Output is 720x1280. Platform UI overlays (Shorts / TikTok / Reels) cover roughly the bottom
250px and right 140px of a 1080x1920 frame, which is 167px and 93px at 720x1280.
"""
import cv2
import numpy as np

import video as V

W, H = V.OUT_W, V.OUT_H
SAFE_BOTTOM = 167
SAFE_RIGHT = 93
SPLIT_TOP = int(H * 0.65) // 2 * 2       # 832: source
SPLIT_BOTTOM = H - SPLIT_TOP             # 448: creator face-cam
STACK_H = H // 2                         # two speakers, 720x640 each


def two_faces(path: str, every: float = 0.6, min_rate: float = 0.35):
    """Left/right face centres (0-1) if two separate people are on screen often enough."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every)))
    lefts, rights, n, hits, i = [], [], 0, 0, 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    g = V._small_gray(frame)
                    faces = sorted(V._detect(g), key=lambda f: -f[2] * f[3])[:2]
                    n += 1
                    if len(faces) == 2:
                        a, b = sorted(((x + w / 2) / g.shape[1] for x, y, w, h in faces))
                        if b - a > 0.22:
                            lefts.append(a)
                            rights.append(b)
                            hits += 1
            i += 1
    finally:
        cap.release()
    if n == 0 or hits / n < min_rate:
        return None
    return float(np.median(lefts)), float(np.median(rights))


def face_center(path: str):
    """Median horizontal position (0-1) of the main face, or 0.5."""
    xs = [cx for _, cx, _ in V.sample_faces(path, every=0.8) if cx is not None]
    return float(np.median(xs)) if xs else 0.5


def graph(kind: str, plan: dict, grade: str, facecam_input: int | None = None) -> str:
    """Video filtergraph ending in [vbase]; captions, B-roll and logo are added after."""
    if kind == "frame":
        return (
            "[0:v]split[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            "boxblur=25:6,eq=brightness=-0.15[bg];"
            f"[b]scale={W - 80}:-2,pad=iw+8:ih+8:4:4:color=white@0.92[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2+50,setsar=1{grade}[vbase]"
        )
    if kind == "split" and facecam_input is not None:
        cw = f"min(iw,ih*{W}/{SPLIT_TOP})"
        cx = plan.get("cx", 0.5)
        return (
            f"[0:v]crop=w='{cw}':h=ih:x='max(0,min(iw-({cw}),{cx:.4f}*iw-({cw})/2))',"
            f"scale={W}:{SPLIT_TOP},setsar=1{grade}[top];"
            f"[{facecam_input}:v]scale={W}:{SPLIT_BOTTOM}:force_original_aspect_ratio=increase,"
            f"crop={W}:{SPLIT_BOTTOM},setsar=1{grade}[bot];"
            "[top][bot]vstack=inputs=2[vbase]"
        )
    if kind == "stack":
        l, r = plan["left"], plan["right"]
        cw = f"min(iw,ih*{W}/{STACK_H})"
        def crop(cx):
            return (f"crop=w='{cw}':h=ih:x='max(0,min(iw-({cw}),{cx:.4f}*iw-({cw})/2))',"
                    f"scale={W}:{STACK_H},setsar=1")
        return (
            "[0:v]split[s1][s2];"
            f"[s1]{crop(l)}[t];[s2]{crop(r)}[b];"
            f"[t][b]vstack=inputs=2,setsar=1{grade}[vbase]"
        )
    raise ValueError(kind)


def _face_boxes(path: str, every: float = 0.6, min_rate: float = 0.35):
    """Median (cx, cy, w, h) of the left and right person, all 0-1, or None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every)))
    L_, R_, n, i = [], [], 0, 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    g = V._small_gray(frame)
                    gh, gw = g.shape
                    faces = sorted(V._detect(g), key=lambda f: -f[2] * f[3])[:2]
                    n += 1
                    if len(faces) == 2:
                        a, b = sorted(faces, key=lambda f: f[0])
                        if (b[0] + b[2] / 2 - a[0] - a[2] / 2) / gw > 0.22:
                            L_.append(((a[0] + a[2] / 2) / gw, (a[1] + a[3] / 2) / gh, a[2] / gw, a[3] / gh))
                            R_.append(((b[0] + b[2] / 2) / gw, (b[1] + b[3] / 2) / gh, b[2] / gw, b[3] / gh))
            i += 1
    finally:
        cap.release()
    if n == 0 or len(L_) / n < min_rate:
        return None
    return tuple(float(x) for x in np.median(L_, axis=0)), tuple(float(x) for x in np.median(R_, axis=0))


def mouth_motion(path: str, left, right, every: float = 0.2):
    """Frame-to-frame movement in the lower half of each face: who is talking, over time."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every)))
    ts, ml, mr, prev, i = [], [], [], None, 0

    def roi(img, box):
        h, w = img.shape
        cx, cy, fw, fh = box
        x0, x1 = int(max(0, (cx - fw / 2) * w)), int(min(w, (cx + fw / 2) * w))
        y0, y1 = int(max(0, (cy + 0.05 * fh) * h)), int(min(h, (cy + fh / 2 + 0.1 * fh) * h))
        return cv2.resize(img[y0:y1, x0:x1], (48, 24)) if x1 - x0 > 4 and y1 - y0 > 4 else np.zeros((24, 48), np.uint8)

    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    g = cv2.GaussianBlur(V._small_gray(frame, 480), (3, 3), 0)
                    cur = (roi(g, left).astype(np.float32), roi(g, right).astype(np.float32))
                    if prev is not None:
                        ts.append(i / fps)
                        ml.append(float(np.abs(cur[0] - prev[0]).mean()))
                        mr.append(float(np.abs(cur[1] - prev[1]).mean()))
                    prev = cur
            i += 1
    finally:
        cap.release()
    return ts, ml, mr


def choose_speakers(ts, ml, mr, hold: float = 1.6, margin: float = 1.25, win: int = 5):
    """[(time, 0|1)] switch points (0 = left person). Smoothed, with hysteresis so it does not flicker."""
    if not ts:
        return [(0.0, 0)]
    k = np.ones(win) / win
    sl, sr = np.convolve(ml, k, mode="same"), np.convolve(mr, k, mode="same")
    cur, last, out = 0 if sl[0] >= sr[0] else 1, 0.0, [(0.0, 0 if sl[0] >= sr[0] else 1)]
    for t, a, b in zip(ts, sl, sr):
        want = cur
        if cur == 0 and b > a * margin and b > 0.4:
            want = 1
        elif cur == 1 and a > b * margin and a > 0.4:
            want = 0
        if want != cur and t - last >= hold:
            cur, last = want, t
            out.append((t, cur))
    return out


def speaker_plan(raw: str, meta: dict):
    """9:16 crop that cuts to whoever's mouth is moving. None if two people aren't on screen."""
    boxes = _face_boxes(raw)
    if not boxes:
        return None
    left, right = boxes
    cw = int(meta["h"] * 9 / 16) // 2 * 2
    if cw >= meta["w"]:
        return None

    def x_for(box):
        return min(max(box[0] * meta["w"] - cw / 2, 0), meta["w"] - cw)

    ts, ml, mr = mouth_motion(raw, left, right)
    switches = choose_speakers(ts, ml, mr)
    keys = [(0.0, x_for(left if switches[0][1] == 0 else right))]
    for t, who in switches[1:]:
        keys.append((t, x_for(left if who == 0 else right)))
    return {"kind": "crop", "cw": cw, "keys": keys, "speaker_cuts": len(keys) - 1}
