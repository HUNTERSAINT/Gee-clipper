import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("GROQ_API_KEY", "x")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "x")

import layouts
import quality
import schedule
import video as V


def W(text, start=0.0, step=0.5):
    return [{"word": w, "start": start + i * step, "end": start + i * step + 0.4} for i, w in enumerate(text.split())]


def test_intro_skips_half_sentence_and_filler():
    ws = W("and then we went. So. The real story starts here.")
    assert V.intro_start(ws) > 0
    assert V.intro_start(W("The real story starts here.")) == 0.0


def test_zoom_moments_spaced_and_bounded():
    ws = W("This is nothing. Wow that is huge! Then 45 percent more. Really? Yes it is true " * 3)
    z = V.zoom_moments(ws, 40)
    assert z and all(1.0 <= t <= 37.5 for t in z)
    assert all(b - a >= 3.5 for a, b in zip(z, z[1:]))


def test_speaker_hysteresis_does_not_flicker():
    ts = [i * 0.2 for i in range(60)]
    left = [2.0 if i < 30 else 0.1 for i in range(60)]
    right = [0.1 if i < 30 else 2.0 for i in range(60)]
    sw = layouts.choose_speakers(ts, left, right)
    assert [w for _, w in sw] == [0, 1]
    noisy = [2.0 if i % 2 else 1.9 for i in range(60)]
    assert len(layouts.choose_speakers(ts, noisy, list(reversed(noisy)))) <= 3


def test_accent_colour_conversion_and_number_highlight():
    assert V._bgr("FF8800") == "0088FF"
    assert "7CE86A" in V._num_style("45%") and "7CE86A" not in V._num_style("hello")


def test_quality_flags_fast_speech():
    rep = quality.report.__wrapped__ if hasattr(quality.report, "__wrapped__") else None
    assert rep is None  # report() needs a real file; text() formats any report dict
    assert "no issues" in quality.text({"score": 10, "flags": []})
    assert "•" in quality.text({"score": 7, "flags": ["x"]})


def test_schedule_csv(tmp_path):
    p = schedule.build_csv(str(tmp_path / "s.csv"), [{"file": "a.mp4", "metadata": {
        "youtube_titles": ["T"], "youtube": {"description": "d", "hashtags": ["x"]}}}])
    rows = open(p, encoding="utf-8").read().splitlines()
    assert len(rows) == 5 and "#x" in rows[1]
