"""CPU regression tests for only the Eq8 classifier-W reference semantics."""
import unittest
import torch
from ratio_reference import cosine_receiver_ratio,select_weights,fuse_features

class ReferenceTests(unittest.TestCase):
    def test_current_same_as_frozen_at_initialization(self):
        a=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        y=torch.tensor([0,1,0])
        w=torch.eye(2)
        original=w.clone()
        r0=cosine_receiver_ratio(a,y,select_weights(original,w,"frozen_stage1"))
        r1=cosine_receiver_ratio(a,y,select_weights(original,w,"current_stage2"))
        self.assertTrue(torch.equal(r0,r1))
        self.assertTrue(torch.allclose(r0,torch.tensor([0.,0.,1.])))

    def test_live_W_changes_ratio_but_frozen_W_stays_constant(self):
        x=torch.tensor([[1.,0.],[0.,1.]])
        y=torch.tensor([0,1])
        initial=torch.eye(2)
        live=torch.eye(2)
        r_before=cosine_receiver_ratio(x,y,select_weights(initial,live,"current_stage2"))
        with torch.no_grad():
            live[0]=torch.tensor([-1.,0.])
        r_after=cosine_receiver_ratio(x,y,select_weights(initial,live,"current_stage2"))
        r_frozen=cosine_receiver_ratio(x,y,select_weights(initial,live,"frozen_stage1"))
        self.assertEqual(r_before[0].item(),0.)
        self.assertEqual(r_after[0].item(),1.)
        self.assertEqual(r_frozen[0].item(),0.)
        self.assertEqual(r_after[1].item(),0.)

    def test_no_autograd_through_live_reference(self):
        x=torch.randn(5,64,requires_grad=True)
        w=torch.randn(100,64,requires_grad=True)
        labels=torch.arange(5)
        r=cosine_receiver_ratio(x,labels,w)
        self.assertFalse(r.requires_grad)
        self.assertEqual(r.shape,(5,))
        self.assertTrue(torch.isfinite(r).all())

    def test_ratio_polarity_and_exact_mixing(self):
        x=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
        y=torch.zeros(3,dtype=torch.long)
        weights=torch.tensor([[1.,0.],[0.,1.]])
        r=cosine_receiver_ratio(x,y,weights)
        self.assertTrue(torch.allclose(r,torch.tensor([0.,1.,.5])))
        donor=torch.tensor([[8.,9.],[9.,9.],[10.,10.]])
        mixed=fuse_features(x,donor,r)
        self.assertTrue(torch.allclose(mixed[0],donor[0]))
        self.assertTrue(torch.allclose(mixed[1],x[1]))
        self.assertTrue(torch.allclose(mixed[2],(x[2]+donor[2])/2))

    def test_frozen_reference_is_independent_copy(self):
        live=torch.eye(2)
        old=live.detach().clone()
        with torch.no_grad():live[0,0]=-3.
        self.assertEqual(old[0,0].item(),1.)
        self.assertEqual(select_weights(old,live,"frozen_stage1")[0,0].item(),1.)
        self.assertEqual(select_weights(old,live,"current_stage2")[0,0].item(),-3.)

    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError):
            select_weights(torch.eye(2),torch.eye(2),"reverse_ratio")
        with self.assertRaises(ValueError):
            fuse_features(torch.randn(4,64),torch.randn(5,64),torch.randn(4))
        with self.assertRaises(ValueError):
            cosine_receiver_ratio(torch.randn(4,64),torch.arange(4),torch.randn(2,64))

if __name__=="__main__":
    unittest.main()
