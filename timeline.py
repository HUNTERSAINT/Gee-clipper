"""Editable timeline package: FCPXML + the cut source, B-roll, captions and the finished render, zipped."""
import os
import shutil
import xml.sax.saxutils as X
import zipfile

FPS = 30


def _t(sec: float) -> str:
    return f"{int(round(sec * FPS))}/{FPS}s"


def fcpxml(name: str, dur: float, src_name: str, w: int, h: int, broll: list[dict]) -> str:
    assets = [
        f'<asset id="r2" name="Source cut" start="0s" duration="{_t(dur)}" hasVideo="1" hasAudio="1" '
        f'format="r1" src="media/{X.escape(src_name)}"/>'
    ]
    lanes = []
    for i, b in enumerate(broll):
        rid = f"b{i}"
        assets.append(
            f'<asset id="{rid}" name="B-roll {i + 1}: {X.escape(b["query"])}" start="0s" '
            f'duration="{_t(b["d"] + 1)}" hasVideo="1" format="r1" src="media/{X.escape(b["file"])}"/>'
        )
        lanes.append(
            f'<asset-clip ref="{rid}" lane="1" offset="{_t(b["t"])}" start="0s" '
            f'duration="{_t(b["d"])}" name="B-roll {i + 1}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
    <format id="r1" name="FFVideoFormat{h}p{FPS}" frameDuration="1/{FPS}s" width="{w}" height="{h}"/>
    {chr(10).join("    " + a for a in assets).strip()}
  </resources>
  <library>
    <event name="Gee Clipper">
      <project name="{X.escape(name)}">
        <sequence format="r1" duration="{_t(dur)}" tcStart="0s" tcFormat="NDF">
          <spine>
            <asset-clip ref="r2" offset="0s" start="0s" duration="{_t(dur)}" name="Source cut">
              {chr(10).join("              " + l for l in lanes).strip()}
            </asset-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""


def build(workdir: str, idx: int, raw: str, final: str | None, srt: str | None,
          broll: list[dict], dur: float, w: int, h: int, limit: int) -> str | None:
    """Returns the zip path, or None if it would exceed `limit` bytes."""
    root = os.path.join(workdir, f"timeline_{idx}")
    media = os.path.join(root, "media")
    os.makedirs(media, exist_ok=True)
    src_name = "source_cut.mp4"
    shutil.copy(raw, os.path.join(media, src_name))
    items = []
    for i, b in enumerate(broll):
        fn = f"broll_{i + 1}.mp4"
        shutil.copy(b["path"], os.path.join(media, fn))
        items.append({**b, "file": fn})
    with open(os.path.join(root, "project.fcpxml"), "w", encoding="utf-8") as f:
        f.write(fcpxml(f"Gee Clipper clip {idx}", dur, src_name, w, h, items))
    if srt and os.path.isfile(srt):
        shutil.copy(srt, os.path.join(root, "captions.srt"))
    if final and os.path.isfile(final):
        shutil.copy(final, os.path.join(root, "rendered_preview.mp4"))
    with open(os.path.join(root, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Open project.fcpxml in DaVinci Resolve (File > Import > Timeline) or Final Cut Pro, and\n"
            "Premiere Pro via an FCPXML importer. If media shows offline, relink to the media/ folder.\n"
            "captions.srt can be imported as a subtitle track. rendered_preview.mp4 is the finished clip.\n"
        )
    path = os.path.join(workdir, f"clip_{idx}_timeline.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for base, _, files in os.walk(root):
            for fn in files:
                full = os.path.join(base, fn)
                z.write(full, os.path.relpath(full, root))
    shutil.rmtree(root, ignore_errors=True)
    if os.path.getsize(path) > limit:
        os.remove(path)
        return None
    return path
