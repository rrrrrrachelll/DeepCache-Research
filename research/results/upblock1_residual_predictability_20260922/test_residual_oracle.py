import unittest,torch
from .residual_oracle import channel_stats,combine_stats,fit_channel_models

class StatsTests(unittest.TestCase):
    def test_affine_residual_is_recovered(self):
        x=torch.arange(24,dtype=torch.float32).reshape(2,3,2,2)/10
        slope=torch.tensor([.5,-.25,1.0])[None,:,None,None]
        intercept=torch.tensor([.1,.2,-.3])[None,:,None,None]
        teacher=x+slope*x+intercept
        fitted=fit_channel_models(channel_stats(x,teacher))
        self.assertTrue(torch.allclose(fitted['slope'],slope.flatten(),atol=1e-5))
        self.assertTrue(torch.allclose(fitted['intercept'],intercept.flatten(),atol=1e-5))
    def test_combination_matches_concatenation(self):
        x1=torch.randn(1,2,2,2); x2=torch.randn(1,2,2,2)
        y1=x1+.2*x1; y2=x2+.2*x2
        combined=combine_stats([channel_stats(x1,y1),channel_stats(x2,y2)])
        direct=channel_stats(torch.cat([x1,x2]),torch.cat([y1,y2]))
        self.assertEqual(combined['n'],direct['n'])
        for k in ('sum_x','sum_y','sum_x2','sum_xy'): self.assertTrue(torch.allclose(combined[k],direct[k]))
if __name__=='__main__': unittest.main()
