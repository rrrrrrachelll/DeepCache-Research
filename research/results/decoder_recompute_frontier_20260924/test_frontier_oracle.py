import unittest
from .frontier_oracle import DecoderFrontierOracle,FRONTIERS
class FrontierTests(unittest.TestCase):
    def test_frontier_order_is_complete(self):
        self.assertEqual(list(FRONTIERS),['up_block_0','up_block_1','up_block_2','up_block_3','mid_decoder','full_unet'])
    def test_frontier_starts_at_exact_key_and_stays_active(self):
        key=FRONTIERS['up_block_2']
        self.assertFalse(DecoderFrontierOracle.starts_frontier(FRONTIERS['up_block_3'],key))
        self.assertTrue(DecoderFrontierOracle.starts_frontier(key,key))
        self.assertTrue(DecoderFrontierOracle.starts_frontier(FRONTIERS['up_block_1'],key,already_started=True))
if __name__=='__main__': unittest.main()
