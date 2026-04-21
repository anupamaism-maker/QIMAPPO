import numpy as np


class Buffer:
    def __init__(self, cfg):
        self.cfg = cfg; self._r()

    def _r(self):
        self.ls=[]; self.gs=[]; self.acts=[]; self.rews=[]
        self.lps=[]; self.vals=[]; self.dns=[]; self.zus=[]

    def store(self, ls, gs, acts, rews, lps, val, done, zus):
        self.ls.append([s.copy() for s in ls])
        self.gs.append(gs.copy()); self.acts.append(acts)
        self.rews.append(list(rews)); self.lps.append(list(lps))
        self.vals.append(float(val)); self.dns.append(bool(done))
        self.zus.append([z.copy() for z in zus])

    def gae(self, last_V):
        cfg=self.cfg; T=len(self.rews); U=cfg.N_UAVS
        A=np.zeros((T,U)); R=np.zeros((T,U))
        for u in range(U):
            g=0.0; nV=last_V
            for t in reversed(range(T)):
                m=0 if self.dns[t] else 1
                δ=self.rews[t][u]+cfg.GAMMA*nV*m-self.vals[t]
                g=δ+cfg.GAMMA*cfg.GAE_LAMBDA*m*g
                A[t,u]=g; R[t,u]=g+self.vals[t]; nV=self.vals[t]
        μ=A.mean(); σ=A.std()+1e-8; A=(A-μ)/σ
        return A, R

    def batches(self, B):
        T=len(self.rews); idx=np.random.permutation(T)
        return [idx[i:i+B] for i in range(0,T,B)]

    def clear(self): self._r()
    def __len__(self): return len(self.rews)
