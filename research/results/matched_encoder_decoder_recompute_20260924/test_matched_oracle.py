import unittest
from .matched_oracle import MatchedPathOracle,FRONTIERS
class MatchedPathTests(unittest.TestCase):
    def test_frontier_order_is_complete(self):
        self.assertEqual(list(FRONTIERS),['decoder_only','down3_pair','down2_pair','down1_pair','full_unet'])
    def test_frontier_starts_at_exact_key_and_stays_active(self):
        key=FRONTIERS['down2_pair']
        self.assertFalse(MatchedPathOracle.starts_frontier(FRONTIERS['down3_pair'],key))
        self.assertTrue(MatchedPathOracle.starts_frontier(key,key))
        self.assertTrue(MatchedPathOracle.starts_frontier(FRONTIERS['down1_pair'],key,already_started=True))
if __name__=='__main__': unittest.main()
