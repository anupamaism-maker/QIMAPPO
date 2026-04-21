import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                  import Config
from environment.uav_env     import UAVIoTEnv
from algorithms.mappo        import MAPPOAgent
from algorithms.sfs_mappo    import SFSMAPPOAgent
from algorithms.maddqn       import MADDQNAgent
from sfs.sfs_rp_selection    import SFSRPSelector


def train_one(name, AgentClass, cfg, env):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")
    agent = AgentClass(cfg, env)
    t0    = time.time()
    h     = agent.train()
    print(f"\n[{name}] Done in {time.time()-t0:.1f}s")
    print(f"  Final reward:     {h['reward'][-1]:.2f}")
    print(f"  Final energy eff: {h['energy_eff'][-1]:.3f} Mbit/kJ")
    print(f"  Final deadline:   {h['deadline_sat'][-1]:.1f}%")
    np.save(f"{cfg.CSV_DIR}/{name.lower().replace('-','_')}_history.npy", h)
    agent.save(f"{cfg.CHECKPOINT_DIR}/{name.lower().replace('-','_')}_final.npy")
    return h


def main():
    cfg = Config()

    env = UAVIoTEnv(cfg)
    node_E = np.random.uniform(0.5, 1.0, cfg.N_NODES)
    sfs    = SFSRPSelector(env.node_pos, node_E, cfg)
    rp_pos, _, fit = sfs.select()
    print(f"[SFS] RPs selected, fitness={fit:.4f}")

    histories = {}

    env_m = UAVIoTEnv(cfg); env_m.rp_pos = rp_pos
    histories['MAPPO'] = train_one('MAPPO', MAPPOAgent, cfg, env_m)

    env_s = UAVIoTEnv(cfg); env_s.rp_pos = rp_pos
    histories['SFS-MAPPO'] = train_one('SFS-MAPPO', SFSMAPPOAgent, cfg, env_s)

    env_d = UAVIoTEnv(cfg); env_d.rp_pos = rp_pos
    histories['MADDQN'] = train_one('MADDQN', MADDQNAgent, cfg, env_d)

    np.save(f"{cfg.CSV_DIR}/all_baselines.npy", histories)
    print("\n[Done] All baselines trained. Run compare.py to see results.")
    return histories


if __name__ == "__main__":
    main()