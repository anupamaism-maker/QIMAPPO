import numpy as np
from algorithms.networks import ActorNet, CriticNet
from algorithms.buffer   import Buffer


class MAPPOAgent:

    def __init__(self, cfg, env):
        self.cfg=cfg; self.env=env; U=cfg.N_UAVS
        # pqc=None → classical MLP actor
        self.actors = [ActorNet(cfg, pqc=None) for _ in range(U)]
        self.critic = CriticNet(cfg)
        self.buf    = Buffer(cfg)
        self.history= {k:[] for k in
                       ['reward','energy_eff','deadline_sat',
                        'actor_loss','critic_loss']}

    def train(self):
        cfg=self.cfg
        for ep in range(cfg.N_EPISODES):
            r,e,d,al,cl=self._episode()
            for k,v in zip(['reward','energy_eff','deadline_sat',
                             'actor_loss','critic_loss'],[r,e,d,al,cl]):
                self.history[k].append(v)
            if (ep+1)%cfg.LOG_INTERVAL==0:
                print(f"[MAPPO]    Ep{ep+1:5d} R={r:8.2f} "
                      f"EE={e:.3f} DL={d:.1f}%")
        return self.history

    def _episode(self):
        cfg=self.cfg; env=self.env; U=cfg.N_UAVS
        ls=env.reset(); self.buf.clear()
        ep_r=0; E_tot=0; rp_met=0

        for t in range(cfg.MAX_STEPS):
            gs=env.get_global_state(); V=self.critic.fwd(gs)
            acts=[]; lps=[]; zus=[]
            for u in range(U):
                a=self.actors[u].sample(ls[u])
                acts.append(a); lps.append(a['log_prob'])
                zus.append(np.zeros(cfg.N_QUBITS))
            nls,rews,done,info=env.step(acts)
            ep_r+=sum(rews); E_tot+=info['energy']; rp_met+=info['rp_met']
            self.buf.store(ls,gs,acts,rews,lps,V,done,zus)
            ls=nls
            if done: break

        lV=self.critic.fwd(env.get_global_state())
        A,R=self.buf.gae(lV)
        als=[]; cls=[]
        for _ in range(cfg.PPO_EPOCHS):
            a,c=self._update(A,R); als.append(a); cls.append(c)

        ee  = max(ep_r,0)*0.1/max(E_tot/1000,1e-6)
        dlp = 100*rp_met/max(cfg.N_RPS,1)
        return ep_r,ee,dlp,float(np.mean(als)),float(np.mean(cls))

    def _update(self, A, R):
        cfg=self.cfg; buf=self.buf; ε=cfg.PPO_CLIP; U=cfg.N_UAVS
        als=[]; cls=[]
        for idx in buf.batches(cfg.BATCH_SIZE):
            for u in range(U):
                for t in idx:
                    self.critic.fwd(buf.gs[t])
                    cl=self.critic.update(float(R[t,u])); cls.append(cl)
                    out=self.actors[u].fwd(buf.ls[t][u])
                    olp=buf.lps[t][u]
                    eps_n=buf.acts[t][u].get('eps_n',np.zeros(cfg.ACT_CONT))
                    nlp=(float(np.sum(-0.5*eps_n**2-out['log_sigma']
                                      -0.5*np.log(2*np.pi)))
                         +np.log(out['probs'][buf.acts[t][u]['a_disc']]+1e-8))
                    ratio=np.exp(np.clip(nlp-olp,-10,10))
                    Ahat=float(A[t,u])
                    L=min(ratio*Ahat,np.clip(ratio,1-ε,1+ε)*Ahat)
                    H=-float(np.sum(out['probs']*np.log(out['probs']+1e-8)))
                    als.append(-(L+cfg.ENT_COEF*H))
                    g=np.sign(-Ahat)*0.01*np.ones(cfg.HIDDEN_DIM)
                    self.actors[u].fc1.bwd(g[:cfg.OBS_DIM]
                                           if cfg.OBS_DIM<=cfg.HIDDEN_DIM else g)
                    self.actors[u].fc1.step()
        return float(np.mean(als)),float(np.mean(cls))

    def evaluate(self, n=10):
        rs,es,ds=[],[],[]
        for _ in range(n):
            r,e,d,_,_=self._episode(); rs.append(r); es.append(e); ds.append(d)
        return dict(reward=np.mean(rs),energy_eff=np.mean(es),
                    deadline=np.mean(ds),std=np.std(rs))

    def save(self, path): np.save(path, {})
    def load(self, path): pass