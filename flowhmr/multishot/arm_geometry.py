"""Projection helpers used only by the bounded upper-limb fit."""
import numpy as np

SM=np.array([16,17,18,19,20,21,1,2,4,5,7,8])
MH=np.array([5,6,7,8,62,41,9,10,11,12,13,14])

def project(points,K):
    q=np.einsum('tij,tkj->tki',K,points)
    return q[...,:2]/np.where(abs(q[...,2:])>1e-8,q[...,2:],np.nan)


def polar(h):
    u,_,vt=np.linalg.svd(h)
    return vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
