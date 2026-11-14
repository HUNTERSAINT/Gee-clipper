"""Clip-matched background music from a CC0 library.

Tracks are downloaded on demand, measured once (tempo, energy, brightness, major/minor)
and cached. For each clip the best-fitting track is chosen from the clip's mood and
energy, never repeating a track inside one job while an unused one is available.
If nothing can be fetched the clip simply goes out with its original audio.
"""
import json
import logging
import os
import random
import subprocess
import time
import urllib.request

import numpy as np

log = logging.getLogger("clipper.music")

BASE = (
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/"
    "John_Bartmann/Public_Domain_Soundtrack_Music_Album_One/"
)
CACHE = os.getenv("MUSIC_CACHE_DIR", os.path.join(os.getenv("DATA_DIR", "data"), "music"))
SR = 11025

# slug: (file, {mood: weight}, rough energy 1-5 used until real analysis exists)
_T = lambda n, t: f"John_Bartmann_-_{n:02d}_-_{t}.mp3"
TRACKS = {
    "mad-hatter-tea-party": (_T(1, "Mad_Hatter_Tea_Party"), {"funny": 1.0, "energetic": 0.4}, 4),
    "happy-african-village": (_T(2, "Happy_African_Village"), {"uplifting": 1.0, "funny": 0.4}, 3),
    "bouncy-gypsy-beats": (_T(3, "Bouncy_Gypsy_Beats"), {"energetic": 1.0, "funny": 0.5}, 4),
    "earning-happiness": (_T(4, "Earning_Happiness"), {"uplifting": 1.0, "calm": 0.3}, 3),
    "home-at-last": (_T(5, "Home_At_Last"), {"calm": 1.0, "emotional": 0.6}, 2),
    "balkan-gypsy-party": (_T(6, "Balkan_Gypsy_Party"), {"energetic": 1.0, "funny": 0.6}, 5),
    "african-moon": (_T(7, "African_Moon"), {"dramatic": 0.8, "calm": 0.5, "emotional": 0.4}, 2),
    "gameshow-brazz": (_T(8, "Gameshow_Brazz"), {"energetic": 0.8, "funny": 0.8}, 4),
    "happy-clappy": (_T(9, "Happy_Clappy"), {"funny": 0.8, "uplifting": 0.7}, 3),
    "heart-of-acceptance": (_T(10, "Heart_of_Acceptance"), {"emotional": 1.0, "calm": 0.5}, 2),
    "iyanetha": (_T(11, "Iyanetha"), {"emotional": 0.7, "calm": 0.6, "dramatic": 0.4}, 2),
    "interstellar-space": (_T(12, "Interstellar_Space"), {"dramatic": 1.0, "calm": 0.4}, 2),
    "robot-gypsy-jazz": (_T(13, "Robot_Gypsy_Jazz"), {"funny": 0.8, "energetic": 0.6}, 4),
    "west-in-africa": (_T(15, "West_in_Africa"), {"uplifting": 0.8, "energetic": 0.5}, 3),
    "somewhere-nice": (_T(16, "Somewhere_Nice"), {"calm": 1.0, "uplifting": 0.4}, 1),
    "secret-agent-rock": (_T(17, "Secret_Agent_Rock"), {"dramatic": 1.0, "energetic": 0.6}, 4),
    "tender-moment": (_T(19, "Tender_Moment"), {"emotional": 1.0, "calm": 0.6}, 1),
    "umlungu": (_T(20, "Umlungu"), {"uplifting": 0.9, "energetic": 0.4}, 3),
}

FAIL_TTL = 15 * 60  # seconds before a failed download is retried
_failed: dict[str, float] = {}


# --------------------------------------------------------------------------- #
# download + analysis
# --------------------------------------------------------------------------- #
def _path(slug):
    return os.path.join(CACHE, slug + ".mp3")


def _fetch(slug) -> str | None:
    """Local path of a validated track, downloading it if needed."""
    os.makedirs(CACHE, exist_ok=True)
    target = _path(slug)
    if os.path.isfile(target) and os.path.getsize(target) > 50000:
        return target
    if time.time() - _failed.get(slug, 0) < FAIL_TTL:
        return None
    tmp = target + ".part"
    try:
        req = urllib.request.Request(BASE + TRACKS[slug][0],
                                     headers={"User-Agent": "GeeClipper/1.0 (CC0 soundtrack cache)"})
        with urllib.request.urlopen(req, timeout=35) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 18):
                f.write(chunk)
        if os.path.getsize(tmp) < 50000:
            raise RuntimeError("file too small")
        ok = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                             "stream=codec_type", "-of", "csv=p=0", tmp],
                            capture_output=True, text=True, timeout=15)
        if ok.returncode != 0 or not ok.stdout.strip():
            raise RuntimeError("not audio")
        os.replace(tmp, target)
        log.info("Music downloaded track=%s", slug)
        return target
    except Exception as exc:
        _failed[slug] = time.time()
        log.warning("Music download failed track=%s reason=%s", slug, type(exc).__name__)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None


def _decode(path, seconds=150):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds), "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, timeout=60, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def analyse_audio(x: np.ndarray) -> dict:
    """Tempo (BPM), energy (0-1), brightness (0-1), mode (+1 major / -1 minor), start offset."""
    if len(x) < SR * 3:
        raise ValueError("too short")
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft).astype(np.float32)
    nfr = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)

    rms = np.sqrt((x[idx] ** 2).mean(axis=1))
    energy = float(np.clip((20 * np.log10(np.mean(rms) + 1e-6) + 40) / 30, 0, 1))
    cent = (S * freqs).sum(1) / (S.sum(1) + 1e-9)
    brightness = float(np.clip((np.median(cent) - 500) / 2500, 0, 1))

    # tempo: autocorrelation of spectral flux
    flux = np.maximum(0, np.diff(np.log1p(S * 10), axis=0)).sum(1)
    flux = flux - flux.mean()
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = SR / hop
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)
    seg = ac[lo:hi]
    lag = lo + int(np.argmax(seg))
    bpm = 60 * fps / lag
    while bpm < 70:
        bpm *= 2
    while bpm > 170:
        bpm /= 2

    # mode: correlate chroma with Krumhansl major/minor profiles
    chroma = np.zeros(12)
    band = (freqs > 65) & (freqs < 2000)
    midi = 69 + 12 * np.log2(freqs[band] / 440.0)
    pc = np.round(midi).astype(int) % 12
    power = (S[:, band] ** 2).mean(axis=0)
    for k in range(12):
        chroma[k] = power[pc == k].sum()
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = lambda prof: max(np.corrcoef(np.roll(prof, k), chroma)[0, 1] for k in range(12))
    mode = 1.0 if best(maj) >= best(mnr) else -1.0

    # first moment the music is really playing (skip silent lead-in)
    loud = np.where(rms > rms.max() * 0.25)[0]
    start = float(loud[0] * hop / SR) if len(loud) else 0.0
    return {"bpm": round(float(bpm), 1), "energy": round(energy, 3),
            "brightness": round(brightness, 3), "mode": mode, "start": round(min(start, 6.0), 2)}


def features(slug) -> dict | None:
    cache = os.path.join(CACHE, slug + ".json")
    try:
        with open(cache) as f:
            return json.load(f)
    except Exception:
        pass
    path = _fetch(slug)
    if not path:
        return None
    try:
        feat = analyse_audio(_decode(path))
        with open(cache, "w") as f:
            json.dump(feat, f)
        log.info("Music analysed track=%s %s", slug, feat)
        return feat
    except Exception as exc:
        log.warning("Music analysis failed track=%s reason=%s", slug, type(exc).__name__)
        return {"bpm": 100.0, "energy": TRACKS[slug][2] / 5, "brightness": 0.5, "mode": 0.0, "start": 0.0}


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
_MOOD_TARGET = {  # desired (energy, brightness, mode, bpm)
    "uplifting": (0.6, 0.6, 1.0, 110),
    "dramatic": (0.65, 0.35, -1.0, 100),
    "emotional": (0.3, 0.35, -0.3, 80),
    "energetic": (0.8, 0.65, 0.3, 130),
    "calm": (0.25, 0.4, 0.3, 80),
    "funny": (0.65, 0.7, 1.0, 120),
}


def _score(slug, mood, energy, feat, used):
    w = TRACKS[slug][1].get(mood, 0.0)
    te, tb, tm, tbpm = _MOOD_TARGET.get(mood, _MOOD_TARGET["calm"])
    e = (energy - 1) / 4 if energy else te  # clip energy 1-5 -> 0-1
    te = 0.5 * te + 0.5 * e
    fit = 1 - (abs(feat["energy"] - te) * 1.2 + abs(feat["brightness"] - tb) * 0.5
               + (abs(feat["mode"] - tm) * 0.15 if feat["mode"] else 0)
               + abs(np.log2(feat["bpm"] / tbpm)) * 0.25)
    s = w * 2.0 + fit
    if slug in used:
        s -= 1.5 + 0.5 * used[slug]
    return s + random.random() * 0.08


def pick(mood: str, energy: float = 3.0, used: dict | None = None):
    """Best available track for this clip: (path, slug, features) or None."""
    used = used if used is not None else {}
    mood = mood if mood in _MOOD_TARGET else "calm"
    ranked = sorted(TRACKS, key=lambda s: -(TRACKS[s][1].get(mood, 0.0) * 2.0 + random.random() * 0.3))
    cands = []
    for slug in ranked:  # only analyse what is plausible; keeps the first clip fast
        if TRACKS[slug][1].get(mood, 0.0) < 0.3 and len(cands) >= 3:
            continue
        feat = features(slug)
        if feat:
            cands.append((_score(slug, mood, energy, feat, used), slug, feat))
        if len(cands) >= 6:
            break
    if not cands:
        return None
    cands.sort(reverse=True)
    _, slug, feat = cands[0]
    path = _fetch(slug)
    if not path:
        return None
    used[slug] = used.get(slug, 0) + 1
    log.info("Music chosen track=%s mood=%s energy=%.1f bpm=%s", slug, mood, energy, feat["bpm"])
    return path, slug, feat


# --------------------------------------------------------------------------- #
# mixing
# --------------------------------------------------------------------------- #
def mix_filter(has_voice: bool, feat: dict, duration: float, level_db: float = -27.0) -> str:
    """Audio part of the filter graph. Input 0 = clip, input 1 = music (looped)."""
    st = max(0.0, feat.get("start", 0.0))
    fade_out = max(0.0, duration - 1.6)
    music = (
        f"[1:a:0]aresample=48000,atrim=start={st:.2f},asetpts=PTS-STARTPTS,"
        f"highpass=f=60,loudnorm=I={level_db:.0f}:TP=-3:LRA=11,"
        "equalizer=f=1200:t=q:w=0.9:g=-3,"  # leave room for the voice
        f"afade=t=in:st=0:d=1.2,afade=t=out:st={fade_out:.2f}:d=1.6[music]"
    )
    if not has_voice:
        return music.replace("[music]", "[m0];[m0]loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.95[aout]")
    return (
        "[0:a:0]aresample=48000,highpass=f=80,afftdn=nr=8:nf=-25,acompressor=threshold=0.1:ratio=2.5:attack=8:release=160,loudnorm=I=-16:TP=-2:LRA=9[voice0];"
        "[voice0]asplit=2[voice][vsc];"
        + music + ";"
        "[music][vsc]sidechaincompress=threshold=0.015:ratio=5:attack=15:release=450:makeup=1[ducked];"
        "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0:weights='1 1',volume=2,"
        "loudnorm=I=-14:TP=-1.5:LRA=9,alimiter=limit=0.95[aout]"
    )
