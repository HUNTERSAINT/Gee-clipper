"""Whoosh and pop sound effects, synthesised with ffmpeg (no downloads, no licences)."""
import os
import subprocess
import tempfile

_DEF = {
    "whoosh": ("anoisesrc=d=0.55:c=pink:a=0.6:r=48000,highpass=f=500,lowpass=f=7000,"
               "afade=t=in:d=0.22,afade=t=out:st=0.22:d=0.33,volume=0.9"),
    "pop": ("sine=f=640:d=0.10:r=48000,lowpass=f=2500,afade=t=out:st=0.01:d=0.09,volume=1.0"),
}


def _sample(kind: str, cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{kind}.wav")
    if not os.path.isfile(path):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", _DEF[kind], "-ac", "2", path],
                       check=True, timeout=30)
    return path


def track(events: list[tuple[float, str]], dur: float, workdir: str, idx: int,
          level: float = 0.22) -> str | None:
    """One wav, `dur` seconds long, with each (time, kind) effect placed on it. None if no events."""
    events = [(t, k) for t, k in events if k in _DEF and 0 <= t < dur - 0.1][:12]
    if not events:
        return None
    cache = os.path.join(tempfile.gettempdir(), "gee-clipper-sfx")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for _, kind in events:
        cmd += ["-i", _sample(kind, cache)]
    parts = [f"[{i}:a]aresample=48000,adelay={int(t * 1000)}|{int(t * 1000)}[e{i}]" for i, (t, _) in enumerate(events)]
    n = len(events)
    mix = "".join(f"[e{i}]" for i in range(n))
    # amix divides by the input count; multiply back so each effect keeps its own level
    parts.append(f"{mix}amix=inputs={n}:duration=longest:dropout_transition=0,volume={n * level:.3f},"
                 f"apad=whole_dur={dur:.2f},atrim=duration={dur:.2f}[sfx]")
    out = os.path.join(workdir, f"sfx_{idx}.wav")
    subprocess.run(cmd + ["-filter_complex", ";".join(parts), "-map", "[sfx]", out], check=True, timeout=60)
    return out
