"""Optional translated voiceovers using the free Edge TTS endpoint.

This is opt-in. It requires outbound internet access and may be rate-limited by the
speech service. No credentials are required or stored.
"""
import asyncio
import os
import subprocess

VOICE_BY_LANGUAGE = {
    "english": "en-US-AriaNeural", "en": "en-US-AriaNeural",
    "spanish": "es-ES-ElviraNeural", "es": "es-ES-ElviraNeural",
    "french": "fr-FR-DeniseNeural", "fr": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural", "de": "de-DE-KatjaNeural",
    "portuguese": "pt-BR-FranciscaNeural", "pt": "pt-BR-FranciscaNeural",
    "italian": "it-IT-ElsaNeural", "it": "it-IT-ElsaNeural",
    "hindi": "hi-IN-SwaraNeural", "hi": "hi-IN-SwaraNeural",
    "arabic": "ar-SA-ZariyahNeural", "ar": "ar-SA-ZariyahNeural",
    "japanese": "ja-JP-NanamiNeural", "ja": "ja-JP-NanamiNeural",
    "korean": "ko-KR-SunHiNeural", "ko": "ko-KR-SunHiNeural",
    "chinese": "zh-CN-XiaoxiaoNeural", "zh": "zh-CN-XiaoxiaoNeural",
}


def voice_for_language(language: str) -> str:
    key = (language or "english").strip().lower()
    return VOICE_BY_LANGUAGE.get(key, VOICE_BY_LANGUAGE.get(key[:2], "en-US-AriaNeural"))


async def _synthesize(text: str, voice: str, path: str):
    import edge_tts
    await edge_tts.Communicate(text, voice=voice, rate="+5%").save(path)


def generate(text: str, language: str, workdir: str, index: int, duration: float) -> str:
    """Synthesize translated narration and fit it to the clip's total duration."""
    text = " ".join((text or "").split()).strip()
    if not text:
        raise ValueError("No translated speech was supplied for voiceover")
    source = os.path.join(workdir, f"voiceover_{index}_source.mp3")
    target = os.path.join(workdir, f"voiceover_{index}.m4a")
    asyncio.run(_synthesize(text[:6000], voice_for_language(language), source))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", source],
        capture_output=True, text=True, check=True, timeout=20,
    )
    spoken_duration = max(0.1, float(probe.stdout.strip()))
    ratio = spoken_duration / max(0.1, float(duration))
    # Keep speech intelligible: adjust modestly, then pad or trim to the clip.
    tempo = min(1.5, max(0.75, ratio))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", source,
         "-af", f"atempo={tempo:.4f},apad,atrim=duration={float(duration):.3f},"
                "loudnorm=I=-16:TP=-2:LRA=9",
         "-c:a", "aac", "-b:a", "128k", target],
        capture_output=True, text=True, check=True, timeout=90,
    )
    return target
