import os
import tempfile
import unittest
from unittest.mock import patch
import config
import storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(config, "DB_PATH", os.path.join(self.temp.name, "gee.db"))
        self.patch.start()
        storage.init()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_quota_limit_and_day_rollover(self):
        self.assertTrue(storage.consume_quota(11, 2, "2026-11-14"))
        self.assertTrue(storage.consume_quota(11, 2, "2026-11-14"))
        self.assertFalse(storage.consume_quota(11, 2, "2026-11-14"))
        self.assertTrue(storage.consume_quota(11, 2, "2026-11-15"))

    def test_jobs_clips_and_user_scoped_favorites(self):
        storage.record_user(11)
        job = storage.create_job(11, "https://youtu.be/example")
        storage.update_job(job, title="Example")
        clip = storage.record_clip(11, job, "https://youtu.be/example",
            {"title":"Opening", "start":12, "end":42, "score":9, "hashtags":["demo"]})
        self.assertTrue(storage.favorite(11, clip))
        self.assertFalse(storage.favorite(22, clip))
        self.assertEqual(storage.list_favorites(11)[0]["title"], "Opening")
        self.assertTrue(storage.unfavorite(11, clip))
        self.assertEqual(storage.list_favorites(11), [])
        storage.update_job(job, status="completed", clips=1)
        self.assertEqual(storage.history(11)[0]["status"], "completed")

    def test_block_list(self):
        storage.set_blocked(11, True)
        self.assertTrue(storage.is_blocked(11))
        storage.set_blocked(11, False)
        self.assertFalse(storage.is_blocked(11))


if __name__ == "__main__":
    unittest.main()
