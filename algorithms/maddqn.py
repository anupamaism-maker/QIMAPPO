import numpy as np
from collections import deque
from algorithms.networks import DQNet


class MADDQNAgent:

    def __init__(self, cfg, env):
        self.cfg = cfg; self.env = env; U = cfg.N_UAVS
        obs = cfg.OBS_DIM; na = cfg.MADDQN_N_ACTS

        self.qnets    = [DQNet(obs, na, cfg.LR_ACTOR) for _ in range(U)]
        self.t_nets   = [DQNet(obs, na, cfg.LR_ACTOR) for _ in range(U)]
        for u in range(U): self.t_nets[u]=self.qnets[u].copy()

        self.replays  = [deque(maxlen=cfg.REPLAY_SIZE) for _ in range(U)]
        self.eps      = cfg.EPS_START
        self.step_cnt = 0

        self._build_action_table()

        self.history = {k:[] for k in
                        ['reward','energy_eff','deadline_sat','td_loss']}

    def _build_action_table(self):
        cfg = self.cfg
        v_levels   = np.linspace(0, cfg.V_H,    cfg.N_DISC_V)
        phi_levels = np.linspace(0, 2*np.pi,    cfg.N_DISC_PHI, endpoint=False)
        vz_levels  = np.linspace(-cfg.V_V, cfg.V_V, cfg.N_DISC_VZ)
        sched_acts = list(range(cfg.ACT_DISC_N))
        self.action_table = []
        for v in v_levels:
            for p in phi_levels:
                for vz in vz_levels:
                    for s in sched_acts:
                        self.action_table.append(
                            dict(v_xy=v, phi=p, v_z=vz, a_disc=s))

    def train(self):
        cfg = self.cfg
        for ep in range(cfg.N_EPISODES):
            r,e,d,loss = self._episode()
            for k,v in zip(['reward','energy_eff','deadline_sat','td_loss'],
                           [r,e,d,loss]):
                self.history[k].append(v)
            if (ep+1)%cfg.LOG_INTERVAL==0:
                print(f"[MADDQN]   Ep{ep+1:5d} R={r:8.2f} "
                      f"EE={e:.3f} DL={d:.1f}% ε={self.eps:.3f}")
            self.eps = max(cfg.EPS_END, self.eps*cfg.EPS_DECAY)
        return self.history

    def _episode(self):
        cfg=self.cfg; env=self.env; U=cfg.N_UAVS
        ls=env.reset(); ep_r=0; E_tot=0; rp_met=0; losses=[]

        for t in range(cfg.MAX_STEPS):
            acts=[]; act_idxs=[]
            for u in range(U):
                if np.random.rand() < self.eps:
                    ai = np.random.randint(len(self.action_table))
                else:
                    q  = self.qnets[u].fwd(ls[u])
                    ai = int(np.argmax(q))
                acts.append(self.action_table[ai]); act_idxs.append(ai)

            nls,rews,done,info = env.step(acts)
            ep_r+=sum(rews); E_tot+=info['energy']; rp_met+=info['rp_met']

            for u in range(U):
                self.replays[u].append(
                    (ls[u].copy(), act_idxs[u], rews[u],
                     nls[u].copy(), float(done)))

            ls=nls; self.step_cnt+=1

            if len(self.replays[0]) >= cfg.BATCH_SIZE:
                l=self._learn(); losses.append(l)

            if self.step_cnt % cfg.TARGET_UPDATE == 0:
                for u in range(U): self.t_nets[u]=self.qnets[u].copy()

            if done: break

        ee  = max(ep_r,0)*0.1/max(E_tot/1000,1e-6)
        dlp = 100*rp_met/max(cfg.N_RPS,1)
        return ep_r, ee, dlp, float(np.mean(losses)) if losses else 0.0

    def _learn(self):
        cfg=self.cfg; U=cfg.N_UAVS; B=cfg.BATCH_SIZE; γ=cfg.GAMMA; loss=0
        for u in range(U):
            rep = list(self.replays[u])
            idx = np.random.choice(len(rep), B, replace=False)
            for i in idx:
                s,a,r,ns,d=rep[i]
                q   = self.qnets[u].fwd(s)
                qt  = self.t_nets[u].fwd(ns)
                a_n = int(np.argmax(self.qnets[u].fwd(ns)))
                target = r + γ*(1-d)*qt[a_n]
                td_err = target - q[a]; loss += td_err**2
                self.qnets[u].update(float(td_err))
        return loss/(U*B)

    def evaluate(self, n=10):
        saved_eps = self.eps; self.eps = 0.0
        rs,es,ds=[],[],[]
        for _ in range(n):
            r,e,d,_=self._episode(); rs.append(r); es.append(e); ds.append(d)
        self.eps = saved_eps
        return dict(reward=np.mean(rs),energy_eff=np.mean(es),
                    deadline=np.mean(ds),std=np.std(rs))

    def save(self, path): np.save(path, {})
    def load(self, path): pass