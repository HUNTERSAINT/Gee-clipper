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
