import unittest
from .symmetric_oracle import PATHS, SymmetricMatchedOracle

class SymmetricPathTests(unittest.TestCase):
    def test_paths_are_progressive(self):
        self.assertEqual(PATHS, {"matched_0": 0, "matched_01": 1, "matched_012": 2, "matched_0123": 3, "full_unet": 4})

    def test_symmetric_membership(self):
        f = SymmetricMatchedOracle.belongs_to_path
        self.assertTrue(f(("down", "block", 0, 0), 0))
        self.assertTrue(f(("up", "block", 0, 0), 0))
        self.assertFalse(f(("down", "block", 1, 0), 0))
        self.assertFalse(f(("mid", "mid_block", 0, 0), 3))
        self.assertTrue(f(("mid", "mid_block", 0, 0), 4))

if __name__ == '__main__': unittest.main()
