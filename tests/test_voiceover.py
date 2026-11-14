import unittest
import voiceover


class VoiceoverTests(unittest.TestCase):
    def test_supported_language_voice_selection(self):
        self.assertEqual(voiceover.voice_for_language("Spanish"), "es-ES-ElviraNeural")
        self.assertEqual(voiceover.voice_for_language("fr"), "fr-FR-DeniseNeural")
        self.assertEqual(voiceover.voice_for_language("Hindi"), "hi-IN-SwaraNeural")

    def test_unknown_language_uses_safe_default(self):
        self.assertEqual(voiceover.voice_for_language("Yoruba"), "en-US-AriaNeural")


if __name__ == "__main__":
    unittest.main()
