import os
import tempfile
import unittest
from unittest.mock import patch
import storage


class MonetizationStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(storage, "DB_PATH", os.path.join(self.temp.name, "gee.db"))
        self.patch.start()
        storage.init()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_referrals_are_unique_and_capped(self):
        self.assertTrue(storage.record_referral(1, 2))
        self.assertFalse(storage.record_referral(1, 2))
        self.assertFalse(storage.record_referral(2, 2))
        self.assertEqual(storage.referral_count(1), 1)

    def test_subscription_activation_and_revocation(self):
        expiry = storage.activate_subscription(1, 30)
        self.assertTrue(storage.is_premium(1))
        self.assertEqual(storage.subscription(1)["expires_at"], expiry)
        storage.revoke_subscription(1)
        self.assertFalse(storage.is_premium(1))

    def test_payment_records_are_user_scoped(self):
        storage.record_payment("ref123", 1, 500000)
        self.assertIsNotNone(storage.payment_for_user("ref123", 1))
        self.assertIsNone(storage.payment_for_user("ref123", 2))


if __name__ == "__main__":
    unittest.main()
