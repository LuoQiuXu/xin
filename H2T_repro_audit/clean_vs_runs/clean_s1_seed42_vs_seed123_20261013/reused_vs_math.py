"""Verbatim VS audit math. No audit runner or no-training guards are imported."""
import numpy as np
PRIOR = 5.
def regularize_variance(var0,counts,target):
    lam = PRIOR/(np.asarray(counts)+PRIOR)
    return (1-lam)[...,None]*var0+lam[...,None]*target

def scores(features,donor_classes,mean,var,eps):
    bias = np.square(mean[donor_classes]-features)
    return var[donor_classes]/(var[donor_classes]+bias+eps)

def top_indices(s,number,high=True):
    return np.argsort(-s if high else s,axis=-1,kind='stable')[...,:number]

def selected_bool(indices):
    return np.eye(64,dtype=bool)[indices].any(-2)
