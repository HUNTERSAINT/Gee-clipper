"""Multi-platform metadata.json for each clip. Falls back to the clip's own fields if the LLM is unavailable."""
import json
import logging
import re

log = logging.getLogger("clipper.metadata")

PROMPT = (
    "You write social media metadata for a short vertical clip made by a creator who adds their own "
    "commentary on top of it. Frame everything around the commentary angle (the creator's take), not "
    "as a repost. Return JSON only with keys: "
    '"youtube_titles" (exactly 3 curiosity-driven titles, each under 50 characters), '
    '"youtube" {"description","hashtags"}, "facebook" {"description","hashtags"}, '
    '"instagram" {"description","hashtags"}, "tiktok" {"description","hashtags"}, '
    '"hook_script" (exactly 2 spoken sentences the creator can record as an opening). '
    "hashtags are lists of 4-8 words without the # sign. Platform tone: YouTube = search-friendly, "
    "Facebook = conversational, Instagram = punchy with line breaks, TikTok = casual and short."
)


def _tags(v, limit=8):
    return [re.sub(r"\W", "", str(t)) for t in (v or []) if re.sub(r"\W", "", str(t))][:limit]


def fallback(clip: dict) -> dict:
    title = (clip.get("title") or "Watch this")[:49]
    desc = clip.get("description") or title
    tags = _tags(clip.get("hashtags")) or ["shorts", "viral"]
    plat = {"description": desc, "hashtags": tags}
    return {
        "youtube_titles": [title, f"Why {title}"[:49], f"{title}?"[:49]],
        "youtube": dict(plat, hashtags=tags + ["shorts"]),
        "facebook": dict(plat), "instagram": dict(plat), "tiktok": dict(plat),
        "hook_script": f"Here's my take on this. {desc}",
        "source": "fallback",
    }


def generate(clip: dict, spoken_text: str, llm_json) -> dict:
    base = fallback(clip)
    try:
        data = llm_json(PROMPT, f"Clip title: {clip.get('title','')}\nTranscript:\n{spoken_text[:4000]}")
        titles = [str(t).strip()[:49] for t in data.get("youtube_titles", []) if str(t).strip()][:3]
        out = {"youtube_titles": titles or base["youtube_titles"]}
        for p in ("youtube", "facebook", "instagram", "tiktok"):
            d = data.get(p) or {}
            out[p] = {"description": str(d.get("description") or base[p]["description"])[:2200],
                      "hashtags": _tags(d.get("hashtags")) or base[p]["hashtags"]}
        out["hook_script"] = str(data.get("hook_script") or base["hook_script"])[:400]
        out["source"] = "llm"
        return out
    except Exception:
        log.exception("Metadata generation failed; using fallback")
        return base


def write(meta: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path
