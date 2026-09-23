import unittest
import torch
from .spatial_residual import feature_tensor, fit_lowrank, predict_lowrank

class SpatialResidualTests(unittest.TestCase):
    def test_feature_groups(self):
        cached=torch.zeros(2,4,3,3); conv=torch.ones(2,2,3,3); shallow=torch.ones(2,3,3,3)*2
        self.assertEqual(feature_tensor('cached_lowrank',cached,conv,shallow).shape[1],4)
        self.assertEqual(feature_tensor('conv_delta',cached,conv,shallow).shape[1],6)
        self.assertEqual(feature_tensor('shallow_delta',cached,conv,shallow).shape[1],7)
        self.assertEqual(feature_tensor('combined',cached,conv,shallow).shape[1],9)
    def test_lowrank_fits_linear_signal(self):
        torch.manual_seed(1); x=torch.randn(200,3); w=torch.randn(3,2); y=x@w+0.2
        params=fit_lowrank({'4':{'x':x,'y':y}},rank=2,ridge=1e-6)
        image=x[:8].T.reshape(1,3,2,4)
        predicted=predict_lowrank(params,4,image).permute(0,2,3,1).reshape(-1,2)
        self.assertLess(torch.mean(torch.abs(predicted-y[:8])).item(),0.08)

if __name__=='__main__': unittest.main()
