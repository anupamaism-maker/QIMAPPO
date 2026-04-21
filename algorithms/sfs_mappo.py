import numpy as np
from algorithms.mappo       import MAPPOAgent
from sfs.sfs_rp_selection   import SFSRPSelector


class SFSMAPPOAgent:

    def __init__(self, cfg, env):
        self.cfg = cfg; self.env = env
        self.history = {k:[] for k in
                        ['reward','energy_eff','deadline_sat',
                         'actor_loss','critic_loss']}

    def train(self):
        cfg = self.cfg; env = self.env

        print("[SFS-MAPPO] Running SFS RP selection...")
        node_energies = np.random.uniform(0.5, 1.0, cfg.N_NODES)
        sfs  = SFSRPSelector(env.node_pos, node_energies, cfg)
        rp_pos, rp_idx, fit = sfs.select()
        print(f"[SFS-MAPPO] RPs selected: {len(rp_pos)}, "
              f"fitness={fit:.4f}")

        env.rp_pos = rp_pos
        env.rp_positions = rp_pos

        mappo = MAPPOAgent(cfg, env)

        for ep in range(cfg.N_EPISODES):
            r,e,d,al,cl = mappo._episode()
            for k,v in zip(['reward','energy_eff','deadline_sat',
                             'actor_loss','critic_loss'],[r,e,d,al,cl]):
                self.history[k].append(v)
            if (ep+1)%cfg.LOG_INTERVAL==0:
                print(f"[SFS-MAPPO] Ep{ep+1:5d} R={r:8.2f} "
                      f"EE={e:.3f} DL={d:.1f}%")

        self._mappo = mappo
        return self.history

    def evaluate(self, n=10):
        if hasattr(self, '_mappo'):
            return self._mappo.evaluate(n)
        return {}

    def save(self, path): np.save(path, {})
    def load(self, path): pass