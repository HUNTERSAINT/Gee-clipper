"""Keyword-triggered B-roll. Every step is optional: no key, no match or any error -> no B-roll."""
import json
import logging
import os
import random
import re
import urllib.parse
import urllib.request

log = logging.getLogger("clipper.broll")

PEXELS_KEY = os.getenv("PEXELS_API_KEY", "").strip()
LOCAL_DIR = os.getenv("BROLL_DIR", "").strip()
MIN_GAP = 4.0       # seconds between B-roll cutaways
MAX_ITEMS = 4

PROMPT = (
    "You pick moments in a short talking-head video where a stock-footage cutaway would help. "
    "From the timed transcript choose key concrete nouns, brands, places, numbers or concepts "
    "that can be shown visually. Return JSON only: "
    '{"items":[{"t":seconds_from_clip_start,"query":"1-3 word stock footage search"}]}. '
    f"At most {MAX_ITEMS} items, at least {MIN_GAP:.0f} seconds apart, none in the first 2 seconds "
    "or last 2 seconds. Queries must be visual and generic (no people's names)."
)


def available() -> bool:
    return bool(PEXELS_KEY or (LOCAL_DIR and os.path.isdir(LOCAL_DIR)))


def _timed_text(words, dur):
    out, line, t0 = [], [], None
    for w in words:
        if t0 is None:
            t0 = w["start"]
        line.append(w["word"].strip())
        if w["end"] - t0 >= 3:
            out.append(f"[{t0:.1f}] {' '.join(line)}")
            line, t0 = [], None
    if line:
        out.append(f"[{t0:.1f}] {' '.join(line)}")
    return "\n".join(out)[:4000]


def plan_moments(words, dur: float, llm_json) -> list[dict]:
    """words: clip-relative word timings. llm_json(prompt, text) -> dict."""
    if not words or dur < 8:
        return []
    data = llm_json(PROMPT, _timed_text(words, dur))
    items, last = [], -99.0
    for it in sorted(data.get("items", []), key=lambda x: float(x.get("t", 0))):
        try:
            t, q = float(it["t"]), re.sub(r"[^\w\s-]", "", str(it["query"])).strip()[:40]
        except (KeyError, TypeError, ValueError):
            continue
        d = round(random.uniform(2.5, 3.5), 1)
        if q and 2.0 <= t <= dur - d - 1.0 and t - last >= MIN_GAP:
            items.append({"t": round(t, 1), "d": d, "query": q})
            last = t
        if len(items) >= MAX_ITEMS:
            break
    return items


def _pexels(query: str, workdir: str, name: str) -> str | None:
    if not PEXELS_KEY:
        return None
    url = ("https://api.pexels.com/videos/search?" +
           urllib.parse.urlencode({"query": query, "orientation": "portrait", "per_page": 6}))
    req = urllib.request.Request(url, headers={"Authorization": PEXELS_KEY, "User-Agent": "GeeClipper/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        vids = json.load(r).get("videos", [])
    random.shuffle(vids)
    for v in vids:
        files = [f for f in v.get("video_files", [])
                 if f.get("file_type") == "video/mp4" and 540 <= (f.get("width") or 0) <= 1080 and f.get("link")]
        if not files:
            continue
        f = min(files, key=lambda x: x["width"])
        path = os.path.join(workdir, name + ".mp4")
        with urllib.request.urlopen(urllib.request.Request(f["link"], headers={"User-Agent": "GeeClipper/1.0"}),
                                    timeout=30) as r, open(path, "wb") as out:
            size = 0
            while chunk := r.read(1 << 18):
                size += len(chunk)
                if size > 15 * 1024 * 1024:
                    raise RuntimeError("b-roll too large")
                out.write(chunk)
        return path
    return None


def _local(query: str) -> str | None:
    if not (LOCAL_DIR and os.path.isdir(LOCAL_DIR)):
        return None
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    files = [f for f in os.listdir(LOCAL_DIR) if f.lower().endswith((".mp4", ".mov", ".webm"))]
    hits = [f for f in files if any(t in f.lower() for t in terms)]
    return os.path.join(LOCAL_DIR, random.choice(hits)) if hits else None


def fetch(items: list[dict], workdir: str, idx: int) -> list[dict]:
    """Attach a local file to each moment; moments without footage are dropped."""
    out = []
    for n, it in enumerate(items):
        try:
            path = _local(it["query"]) or _pexels(it["query"], workdir, f"broll_{idx}_{n}")
        except Exception as exc:
            log.warning("B-roll fetch failed query=%s reason=%s", it["query"], type(exc).__name__)
            path = None
        if path:
            out.append({**it, "path": path})
    log.info("B-roll ready %d/%d", len(out), len(items))
    return out


def build(words, dur, workdir, idx, llm_json) -> list[dict]:
    if not available():
        return []
    try:
        return fetch(plan_moments(words, dur, llm_json), workdir, idx)
    except Exception:
        log.exception("B-roll planning failed; continuing without B-roll")
        return []
