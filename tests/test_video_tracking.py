import unittest

import core
import video


class CaptionFallbackTests(unittest.TestCase):
    def test_word_alignment_preserves_tokens_and_segment_bounds(self):
        words = core._fallback_word_timestamps({
            "text": "Hello there, world!",
            "start": 2.0,
            "end": 4.0,
        })
        self.assertEqual([item["word"] for item in words], ["Hello", "there,", "world!"])
        self.assertAlmostEqual(words[0]["start"], 2.0)
        self.assertAlmostEqual(words[-1]["end"], 4.0)
        self.assertTrue(all(item["end"] > item["start"] for item in words))


class FaceTrackingTests(unittest.TestCase):
    def test_tracker_prefers_continuous_face_over_larger_distractor(self):
        previous = (90, 40, 40, 40)
        continuing = (96, 42, 40, 40)
        larger_distractor = (260, 30, 100, 100)
        self.assertEqual(video._choose_face([larger_distractor, continuing], previous), continuing)

    def test_first_face_selection_uses_largest_confident_candidate(self):
        small = (10, 10, 30, 30)
        large = (100, 40, 80, 80)
        self.assertEqual(video._choose_face([small, large]), large)


if __name__ == "__main__":
    unittest.main()
