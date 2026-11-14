"""CSV of ready-to-post rows (title, description, hashtags per platform, suggested time) for a job."""
import csv
import datetime as dt

SLOTS = {"youtube": 12, "tiktok": 18, "instagram": 19, "facebook": 20}  # suggested UTC hours


def build_csv(path: str, items: list[dict], start: dt.date | None = None) -> str:
    """items: [{"file": name, "metadata": dict}] in clip order."""
    start = start or (dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Platform", "Clip file", "Title", "Description", "Hashtags", "Suggested time (UTC)"])
        for n, it in enumerate(items):
            m = it["metadata"]
            day = start + dt.timedelta(days=n)
            for plat, hour in SLOTS.items():
                p = m.get(plat) or {}
                title = (m.get("youtube_titles") or [""])[0] if plat == "youtube" else ""
                tags = " ".join("#" + t for t in p.get("hashtags", []))
                when = dt.datetime.combine(day, dt.time(hour, 0)).strftime("%Y-%m-%d %H:%M")
                w.writerow([plat.title(), it["file"], title, p.get("description", ""), tags, when])
    return path
