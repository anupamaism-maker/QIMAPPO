import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                  import Config
from environment.uav_env     import UAVIoTEnv
from algorithms.pqc_policy   import QISKIT_AVAILABLE
from algorithms.qi_mappo     import QIMAPPOAgent
from sfs.sfs_rp_selection    import SFSRPSelector
from utils.logger            import Logger


def main():
    print("="*65)
    print("  QI-MAPPO: Quantum-Inspired Multi-Agent PPO")
    print("  Multi-UAV IoT Data Collection")
    print("="*65)
    if QISKIT_AVAILABLE:
        print("[✓] Qiskit Aer quantum circuit simulation")
    else:
        print("[!] Qiskit not found — numpy fallback (install: pip install qiskit qiskit-aer)")
    print()

    cfg = Config()
    env = UAVIoTEnv(cfg)

    
    print("[Step 1] SFS RP selection (Algorithm 1)...")
    node_E = np.random.uniform(0.5, 1.0, cfg.N_NODES)
    sfs    = SFSRPSelector(env.node_pos, node_E, cfg)
    rp_pos, rp_idx, fit = sfs.select()
    env.rp_pos = rp_pos
    print(f"  RPs: {len(rp_pos)}, fitness={fit:.4f}\n")

    from algorithms.pqc_policy import PQCPolicyLayer
    pqc_demo = PQCPolicyLayer(cfg.N_QUBITS, cfg.N_LAYERS,
                               cfg.QUANTUM_LR, cfg.OBS_DIM)
    print("[Step 2] PQC circuit (D=%d qubits, L=%d layers):" %
          (cfg.N_QUBITS, cfg.N_LAYERS))
    print(pqc_demo.diagram(np.random.randn(cfg.OBS_DIM)))
    print()

    agent  = QIMAPPOAgent(cfg, env)
    logger = Logger(f"{cfg.CSV_DIR}/qi_mappo_train.csv",
                    ['episode','reward','energy_eff','deadline_sat',
                     'actor_loss','critic_loss'])

    print("[Step 3] Training QI-MAPPO (%d episodes)..." % cfg.N_EPISODES)
    t0 = time.time()
    history = agent.train()
    elapsed = time.time() - t0

    print("\n" + "="*65)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final reward:     {history['reward'][-1]:.2f}")
    print(f"Final energy eff: {history['energy_eff'][-1]:.3f} Mbit/kJ")
    print(f"Final deadline:   {history['deadline_sat'][-1]:.1f}%")

    np.save(f"{cfg.CSV_DIR}/qi_mappo_history.npy", history)
    agent.save(f"{cfg.CHECKPOINT_DIR}/qi_mappo_final.npy")

    print("\n[Step 4] Evaluating (10 episodes)...")
    m = agent.evaluate(10)
    print(f"  Mean reward:    {m['reward']:.2f} ± {m['std']:.2f}")
    print(f"  Energy eff:     {m['energy_eff']:.3f} Mbit/kJ")
    print(f"  Deadline sat:   {m['deadline']:.1f}%")
    return history


if __name__ == "__main__":
    main()