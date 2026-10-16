"""Multi-intent learned-evidence adaptation of window-limited simplified GLR.

For the declared source c, the null permits one S(c)->E(c) nuisance onset.
For destination j!=c, the alternative is all S(j) or all E(j) in a window.
This implements the log statistic of Lau & Tay (2019), equations 10-14,
on calibrated neural pseudo-likelihoods. Their IID/asymptotic guarantees do
not transfer to correlated TCN evidence or this restarted multi-intent rule.
Source: https://arxiv.org/html/1902.03460v3
"""
import numpy as np
from .common import T0


def run(e_s,e_exc,c0,window,tau,t0=T0):
    if e_s.shape!=e_exc.shape or e_s.ndim!=3 or window<1 or tau<=0:
        raise ValueError('Invalid window-GLR emissions or parameters')
    B,T,K=e_s.shape
    cs=np.concatenate([np.zeros((B,1,K)),e_s.cumsum(1)],axis=1)
    ce=np.concatenate([np.zeros((B,1,K)),e_exc.astype(np.float64).cumsum(1)],axis=1)
    idx=np.arange(B); c=c0.copy(); last=np.full(B,t0)
    C=np.full((B,T),-1,np.int64); alarm=np.zeros((B,T),bool); C[:,t0]=c
    for t in range(t0+1,T):
        left=max(t0+1,t-window+1)
        starts=np.arange(left,t+1); boundaries=np.arange(left,t+2)
        delta=cs[idx[:,None],boundaries[None],c[:,None]]-ce[idx[:,None],boundaries[None],c[:,None]]
        suffix=np.maximum.accumulate(delta[:,::-1],axis=1)[:,::-1]
        denominator=ce[idx,t+1,c][:,None]-cs[idx[:,None],starts[None],c[:,None]]+suffix[:,:-1]
        numerator=np.maximum(cs[:,t+1,None,:]-cs[:,starts,:],ce[:,t+1,None,:]-ce[:,starts,:])
        score=numerator-denominator[...,None]
        score[np.arange(B)[:,None],np.arange(len(starts))[None],c[:,None]]=-np.inf
        score=np.where((starts[None]>last[:,None])[...,None],score,-np.inf)
        alternatives=score.max(1)
        new=alternatives.argmax(-1); fire=alternatives[idx,new]>tau
        c=np.where(fire,new,c); last=np.where(fire,t,last)
        C[:,t],alarm[:,t]=c,fire
    return C,alarm
