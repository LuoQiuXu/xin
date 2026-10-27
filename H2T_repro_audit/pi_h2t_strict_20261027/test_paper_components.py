"""CPU tests for PI-H2T v2 literal equations. No dataset/GPU required."""
import unittest
import torch
from torch import nn

from paper_components import PaperPIF, PIExtractor, pool_feature, receiver_ratio, h2tf, classify_vector, group_stats


class FormulaTests(unittest.TestCase):
    def test_pif_is_two_weight_no_bias(self):
        p=PaperPIF()
        self.assertEqual(sum(x.numel() for x in p.parameters()),2)
        self.assertIsNone(p.fuse.bias)

    def test_pif_formula_and_channel_permutation(self):
        f=torch.tensor([[[[1.,3.]],[[5.,7.]],[[9.,11.]]]])
        p=PaperPIF()
        with torch.no_grad():
            p.fuse.weight.copy_(torch.tensor([[[0.3],[0.7]]]))
        wanted=.3*(f-f.mean(dim=1,keepdim=True))+.7*f
        self.assertTrue(torch.allclose(p(f),wanted,atol=1e-6))
        idx=torch.tensor([2,0,1])
        self.assertTrue(torch.allclose(p(f[:,idx]),p(f)[:,idx],atol=1e-6),
                        "PIF is permutation-EQUIVARIANT across channels, not invariant")

    def test_pif_gradients(self):
        p=PaperPIF()
        x=torch.randn(2,64,8,8,requires_grad=True)
        p(x).square().mean().backward()
        self.assertIsNotNone(p.fuse.weight.grad)
        self.assertTrue(torch.isfinite(p.fuse.weight.grad).all())
        self.assertIsNotNone(x.grad)

    def test_pif_attaches_after_extractor(self):
        class Dummy(nn.Module):
            def forward(self,x):return x*2
        p=PIExtractor(Dummy())
        x=torch.randn(2,64,3,3)
        with torch.no_grad():
            p.pif.fuse.weight.copy_(torch.tensor([[[0.]],[[1.]]]))
        self.assertTrue(torch.allclose(p(x),x*2,atol=1e-6))

    def test_ratio_limits_and_polarity(self):
        f=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
        w=torch.tensor([[1.,0.],[0.,1.]])
        y=torch.tensor([0,0,0],dtype=torch.long)
        r=receiver_ratio(f,y,w)
        self.assertTrue(torch.allclose(r,torch.tensor([0.,1.,.5]),atol=1e-6))
        donor=torch.tensor([[6.,6.],[7.,7.],[8.,8.]])
        z=h2tf(f,donor,r)
        self.assertTrue(torch.allclose(z[0],donor[0],atol=1e-6))
        self.assertTrue(torch.allclose(z[1],f[1],atol=1e-6))
        self.assertTrue(torch.allclose(z[2],.5*(f[2]+donor[2]),atol=1e-6))

    def test_frozen_ratio_stop_grad_compatible(self):
        features=torch.randn(8,64,requires_grad=True)
        weights=torch.randn(100,64,requires_grad=True)
        y=torch.arange(8)
        with torch.no_grad():
            r=receiver_ratio(features,y,weights)
        self.assertFalse(r.requires_grad)
        self.assertEqual(r.shape,(8,))
        self.assertTrue((r>=0).all() and (r<=1).all())

    def test_pool_and_classifier(self):
        x=torch.randn(3,64,8,8)
        a=pool_feature(x)
        self.assertTrue(torch.allclose(a,x.mean((2,3)),atol=1e-6))
        class TinyHead(nn.Module):
            def forward(self,z):
                if z.ndim!=4:raise AssertionError("head expects feature maps")
                return z.mean((2,3))[:,:4]
        self.assertEqual(classify_vector(TinyHead(),a).shape,(3,4))

    def test_group_boundaries(self):
        labels=torch.tensor([0,35,36,70,71,99])
        predictions=labels.clone()
        predictions[1]=0
        p=group_stats(predictions,labels)
        self.assertEqual(p["Many"],50.)
        self.assertEqual(p["Medium"],100.)
        self.assertEqual(p["Few"],100.)
        self.assertAlmostEqual(p["Overall"],100.*5/6)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            PaperPIF()(torch.rand(2,64))
        with self.assertRaises(ValueError):
            h2tf(torch.rand(2,64),torch.rand(3,64),torch.rand(2))
        with self.assertRaises(ValueError):
            receiver_ratio(torch.rand(2,64),torch.tensor([2,3]),torch.rand(2,64))


if __name__=="__main__":
    unittest.main()
