import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core


class TranscriptionTests(unittest.TestCase):
    def test_fallback_word_timestamps_tokenizes_normal_text(self):
        words = core._fallback_word_timestamps({
            "text": "Hello there, how are you?",
            "start": 2.0,
            "end": 4.0,
        })
        self.assertEqual(
            [item["word"] for item in words],
            ["Hello", "there,", "how", "are", "you?"],
        )
        self.assertAlmostEqual(words[0]["start"], 2.0)
        self.assertAlmostEqual(words[-1]["end"], 4.0)
        self.assertTrue(all(item["end"] > item["start"] for item in words))

    def test_fallback_returns_empty_for_invalid_segment(self):
        self.assertEqual(
            core._fallback_word_timestamps({"text": "No duration", "start": 3, "end": 3}),
            [],
        )

    def test_transcript_cache_isolated_by_model_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core.C, "CACHE_DIR", directory):
                with patch.object(core.C, "WHISPER_MODEL", "whisper-large-v3"):
                    core.save_transcript("youtube_test", {
                        "segments": [{"start": 0, "end": 1, "text": "hello"}],
                        "words": [{"word": "hello", "start": 0, "end": 1}],
                        "language": "en",
                    })
                    self.assertIsNotNone(core.load_transcript("youtube_test"))
                    saved = list(Path(directory).glob("*.json"))
                    self.assertEqual(len(saved), 1)
                with patch.object(core.C, "WHISPER_MODEL", "whisper-large-v3-turbo"):
                    self.assertIsNone(core.load_transcript("youtube_test"))

    def test_old_unversioned_transcript_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core.C, "CACHE_DIR", directory):
                Path(directory, "youtube_test.json").write_text(
                    '{"segments":[{"text":"old incorrect transcript"}],"words":[]}',
                    encoding="utf-8",
                )
                self.assertIsNone(core.load_transcript("youtube_test"))


if __name__ == "__main__":
    unittest.main()
