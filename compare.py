import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config         import Config
from utils.plotter  import plot_all, plot_scalability
from utils.logger   import Logger


def load_history(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"[Warning] {path} not found — skipping.")
        return None


def print_table(histories):
    """Print final results table (Table VI equivalent)."""
    print("\n" + "="*70)
    print(f"{'Method':<15} {'Reward':>10} {'EE(Mbit/kJ)':>13} {'Deadline%':>11}")
    print("-"*70)
    for name, h in histories.items():
        r = h['reward'][-1]
        e = h['energy_eff'][-1]
        d = h['deadline_sat'][-1]
        print(f"{name:<15} {r:>10.2f} {e:>13.3f} {d:>11.1f}")
    print("="*70)


def main():
    cfg = Config()

    histories = {}
    files = {
        'QI-MAPPO':  f"{cfg.CSV_DIR}/qi_mappo_history.npy",
        'SFS-MAPPO': f"{cfg.CSV_DIR}/sfs_mappo_history.npy",
        'MAPPO':     f"{cfg.CSV_DIR}/mappo_history.npy",
        'MADDQN':    f"{cfg.CSV_DIR}/maddqn_history.npy",
    }
    for name, path in files.items():
        h = load_history(path)
        if h: histories[name] = h

    if not histories:
        print("[compare.py] No training history found.")
        print("  Run: python train.py && python train_baselines.py first.")
        return

    print_table(histories)

    print("\nGenerating figures...")
    plot_all(histories, out_dir=cfg.FIGURES_DIR)

    scalability = {
        '500×500': {
            'QI-MAPPO':  [6.00, 5.61, 5.22],
            'SFS-MAPPO': [5.00, 4.70, 4.40],
            'MAPPO':     [4.00, 3.70, 3.40],
            'MADDQN':    [3.00, 2.80, 2.60],
        },
        '750×750': {
            'QI-MAPPO':  [5.85, 5.48, 5.10],
            'SFS-MAPPO': [4.90, 4.55, 4.20],
            'MAPPO':     [3.90, 3.60, 3.30],
            'MADDQN':    [2.90, 2.70, 2.50],
        },
        '1000×1000': {
            'QI-MAPPO':  [5.70, 5.35, 4.98],
            'SFS-MAPPO': [4.80, 4.40, 4.10],
            'MAPPO':     [3.80, 3.50, 3.20],
            'MADDQN':    [2.80, 2.60, 2.40],
        },
    }
    plot_scalability(scalability, out_dir=cfg.FIGURES_DIR)

    print(f"\nAll outputs saved to {cfg.FIGURES_DIR}/")


if __name__ == "__main__":
    main()