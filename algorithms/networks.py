import numpy as np


relu  = lambda x: np.maximum(0,x)
tanh  = lambda x: np.tanh(x)
relu_d= lambda x: (x>0).astype(float)
tanh_d= lambda x: 1-np.tanh(x)**2
def softmax(x):
    e = np.exp(x - np.max(x)); return e/(e.sum()+1e-8)


class Dense:
    def __init__(self, ind, outd, lr=1e-4, seed=0):
        r = np.random.RandomState(seed); b = np.sqrt(6/ind)
        self.W = r.uniform(-b,b,(outd,ind)); self.b = np.zeros(outd)
        self.lr = lr
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
        self.t=0; self._x=None; self._gW=None; self._gb=None

    def fwd(self, x):
        self._x = x.copy()
        return self.W@x + self.b

    def bwd(self, g):
        self._gW = np.outer(g, self._x); self._gb = g.copy()
        return self.W.T@g

    def step(self):
        self.t += 1; b1,b2,eps=0.9,0.999,1e-8
        for p,g,m,v in [(self.W,self._gW,self.mW,self.vW),
                         (self.b,self._gb,self.mb,self.vb)]:
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
            p -= self.lr*(m/(1-b1**self.t))/(np.sqrt(v/(1-b2**self.t))+eps)

    def copy(self):
        n=Dense(self.W.shape[1],self.W.shape[0],self.lr)
        n.W=self.W.copy(); n.b=self.b.copy(); return n


class ActorNet:

    def __init__(self, cfg, pqc=None):
        self.cfg = cfg; self.pqc = pqc
        D   = cfg.N_QUBITS if pqc else cfg.OBS_DIM
        H   = cfg.HIDDEN_DIM; lr = cfg.LR_ACTOR
        nc  = cfg.ACT_CONT;   K  = cfg.ACT_DISC_N
        self.fc1  = Dense(D, H, lr, 1); self.fc2  = Dense(H, H, lr, 2)
        self.mu   = Dense(H, nc, lr, 3); self.lsig = Dense(H, nc, lr, 4)
        self.disc = Dense(H, K,  lr, 5)
        self._cache = {}

    def fwd(self, s_u):
        z = self.pqc.forward(s_u) if self.pqc else s_u.astype(float)
        pre1=self.fc1.fwd(z);    h1=relu(pre1)
        pre2=self.fc2.fwd(h1);   h2=relu(pre2)
        mu       = tanh(self.mu.fwd(h2))
        log_sigma= np.clip(self.lsig.fwd(h2), -2, 0.5)
        logits   = self.disc.fwd(h2)
        probs    = softmax(logits)
        self._cache = dict(s_u=s_u,z=z,pre1=pre1,h1=h1,pre2=pre2,h2=h2)
        return dict(z_u=z,h2=h2,mu=mu,log_sigma=log_sigma,
                    logits=logits,probs=probs)

    def sample(self, s_u):
        cfg = self.cfg; out = self.fwd(s_u)
        sig = np.exp(out['log_sigma']); eps = np.random.randn(cfg.ACT_CONT)
        ar  = out['mu'] + sig*eps
        vxy = float(np.clip((ar[0]+1)/2*cfg.V_H, 0, cfg.V_H))
        phi = float(np.clip((ar[1]+1)*np.pi, 0, 2*np.pi))
        vz  = float(np.clip(ar[2]*cfg.V_V, -cfg.V_V, cfg.V_V))
        ad  = int(np.random.choice(len(out['probs']), p=out['probs']))
        lpc = float(np.sum(-0.5*eps**2 - out['log_sigma'] - 0.5*np.log(2*np.pi)))
        lpd = float(np.log(out['probs'][ad]+1e-8))
        return dict(v_xy=vxy,phi=phi,v_z=vz,a_disc=ad,
                    log_prob=lpc+lpd,eps_n=eps,out=out)


class CriticNet:
    def __init__(self, cfg):
        G=cfg.GLOBAL_STATE_DIM; H=cfg.HIDDEN_DIM; lr=cfg.LR_CRITIC
        self.fc1=Dense(G,H,lr,10); self.fc2=Dense(H,H,lr,11); self.fc3=Dense(H,1,lr,12)
        self._c={}

    def fwd(self, S):
        pre1=self.fc1.fwd(S.astype(float)); h1=relu(pre1)
        pre2=self.fc2.fwd(h1);              h2=relu(pre2)
        V=float(self.fc3.fwd(h2)[0])
        self._c=dict(S=S,pre1=pre1,h1=h1,pre2=pre2,h2=h2); return V

    def update(self, target):
        h2=self._c['h2']; pre2=self._c['pre2']; pre1=self._c['pre1']
        V=float(self.fc3.fwd(h2)[0]); err=V-target; loss=err**2
        g3=np.array([2*err])
        g2=self.fc3.bwd(g3)*relu_d(pre2)
        g1=self.fc2.bwd(g2)*relu_d(pre1)
        self.fc1.bwd(g1)
        self.fc3.step(); self.fc2.step(); self.fc1.step()
        return float(loss)


class DQNet:
    def __init__(self, obs_dim, n_actions, lr=1e-4):
        H=256
        self.fc1=Dense(obs_dim, H, lr, 20)
        self.fc2=Dense(H, H, lr, 21)
        self.fc3=Dense(H, n_actions, lr, 22)
        self._c={}

    def fwd(self, s):
        h1=relu(self.fc1.fwd(s.astype(float)))
        h2=relu(self.fc2.fwd(h1))
        q=self.fc3.fwd(h2)
        self._c=dict(h1=h1,h2=h2); return q

    def update(self, td_err):
        h2=self._c['h2']
        g3=np.array([td_err]*self.fc3.W.shape[0])
        g2=self.fc3.bwd(g3); g1=self.fc2.bwd(g2); self.fc1.bwd(g1)
        self.fc3.step(); self.fc2.step(); self.fc1.step()

    def copy(self):
        import copy; return copy.deepcopy(self)