import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False
    print("[Plotter] matplotlib not found. Install: pip install matplotlib")


COLORS = {
    'QI-MAPPO':  '#2563EB',
    'SFS-MAPPO': '#059669',
    'MAPPO':     '#DC2626',
    'MADDQN':    '#D97706',
}
MARKERS = {'QI-MAPPO': 'o', 'SFS-MAPPO': 's', 'MAPPO': '^', 'MADDQN': 'D'}
WINDOW  = 50   # sliding window for variance


def _smooth(arr, w=20):
    arr = np.asarray(arr, dtype=float)
    w = min(w, len(arr))
    if w < 2:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def _window_variance(arr, w=WINDOW):
    arr = np.asarray(arr, dtype=float)
    out = []
    for i in range(len(arr)):
        s = max(0, i - w)
        out.append(float(np.var(arr[s:i + 1])))
    return np.array(out)


def _ylim(arrays, pad_frac=0.08, bottom_zero=False):
    combined = np.concatenate([np.asarray(a, float) for a in arrays])
    lo = float(combined.min())
    hi = float(combined.max())
    margin = (hi - lo) * pad_frac
    lo = max(0.0, lo - margin) if bottom_zero else lo - margin
    return lo, hi + margin


def _markevery(n, target=10):
    return max(1, n // target)


def plot_all(histories, out_dir='results/figures'):

    if not MPL:
        print("[Plotter] Cannot plot — matplotlib missing.")
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    smoothed = {}
    for name, h in histories.items():
        r = _smooth(h['reward'])
        smoothed[name] = r
        ax.plot(r, label=name, color=COLORS.get(name, 'k'),
                marker=MARKERS.get(name, 'o'),
                markevery=_markevery(len(r)), linewidth=1.5)
    ax.set_xlabel('Episodes'); ax.set_ylabel('Average Reward')
    ax.set_title('Reward Convergence')
    ax.set_ylim(*_ylim(list(smoothed.values())))
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/reward.pdf', dpi=150)
    plt.close(fig); print("[Plotter] Saved reward.pdf")

    fig, ax = plt.subplots(figsize=(7, 4))
    variances = {}
    for name, h in histories.items():
        v = _window_variance(h['reward'])
        variances[name] = v
        ax.plot(v, label=name, color=COLORS.get(name, 'k'),
                marker=MARKERS.get(name, 'o'),
                markevery=_markevery(len(v)), linewidth=1.5)
    ax.set_xlabel('Episodes'); ax.set_ylabel('Variance of Return (Sliding)')
    ax.set_title('Training Stability')
    ax.set_ylim(*_ylim(list(variances.values()), bottom_zero=True))
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/stability.pdf', dpi=150)
    plt.close(fig); print("[Plotter] Saved stability.pdf")

    fig, ax = plt.subplots(figsize=(7, 4))
    ee_curves = {}
    for name, h in histories.items():
        e = _smooth(h['energy_eff'])
        ee_curves[name] = e
        ax.plot(e, label=name, color=COLORS.get(name, 'k'),
                marker=MARKERS.get(name, 'o'),
                markevery=_markevery(len(e)), linewidth=1.5)
    ax.set_xlabel('Episodes'); ax.set_ylabel('Energy Efficiency (Mbit/kJ)')
    ax.set_title('Energy Efficiency')
    ax.set_ylim(*_ylim(list(ee_curves.values()), bottom_zero=True))
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/energy_eff.pdf', dpi=150)
    plt.close(fig); print("[Plotter] Saved energy_eff.pdf")

    fig, ax = plt.subplots(figsize=(7, 4))
    dl_curves = {}
    for name, h in histories.items():
        d = _smooth(h['deadline_sat'])
        dl_curves[name] = d
        ax.plot(d, label=name, color=COLORS.get(name, 'k'),
                marker=MARKERS.get(name, 'o'),
                markevery=_markevery(len(d)), linewidth=1.5)
    ax.set_xlabel('Episodes'); ax.set_ylabel('Deadline Satisfaction (%)')
    ax.set_title('Deadline Satisfaction')
    ax.set_ylim(0, 105)
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/deadline.pdf', dpi=150)
    plt.close(fig); print("[Plotter] Saved deadline.pdf")

    for loss_key, title, fname in [
        ('actor_loss',  'Actor Loss',  'actor_loss.pdf'),
        ('critic_loss', 'Critic Loss', 'critic_loss.pdf'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        loss_curves = {}
        for name, h in histories.items():
            if loss_key in h and name != 'MADDQN':
                lc = _smooth(h[loss_key])
                loss_curves[name] = lc
                ax.plot(lc, label=name, color=COLORS.get(name, 'k'), linewidth=1.5)
        ax.set_xlabel('Episodes'); ax.set_ylabel('Loss')
        ax.set_title(title)
        if loss_curves:
            ax.set_ylim(*_ylim(list(loss_curves.values())))
        ax.legend(); ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(f'{out_dir}/{fname}', dpi=150)
        plt.close(fig); print(f"[Plotter] Saved {fname}")

    print(f"\n[Plotter] All figures saved to {out_dir}/")


def plot_scalability(results_by_area, out_dir='results/figures'):
    
    if not MPL:
        return
    os.makedirs(out_dir, exist_ok=True)

    areas = list(results_by_area.keys())
    n_groups = max(len(v) for data in results_by_area.values()
                   for v in data.values())
    x   = np.arange(n_groups)
    w   = min(0.20, 0.80 / max(len(results_by_area.get(areas[0], {'_': []})), 1))
    all_vals = [v for data in results_by_area.values()
                  for vals in data.values() for v in vals]
    global_ymax = max(all_vals) * 1.20 if all_vals else 1.0

    fig, axes = plt.subplots(1, len(areas), figsize=(4.5 * len(areas), 4))
    if len(areas) == 1:
        axes = [axes]

    for ax, area in zip(axes, areas):
        data = results_by_area[area]
        algo_names = list(data.keys())
        n_algos = len(algo_names)
        offsets = np.linspace(-(n_algos - 1) / 2, (n_algos - 1) / 2, n_algos)
        for off, (name, vals) in zip(offsets, data.items()):
            ax.bar(x[:len(vals)] + off * w, vals, width=w, label=name,
                   color=COLORS.get(name, 'k'), alpha=0.85,
                   edgecolor='k', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(n_groups)])
        ax.set_xlabel('Configuration Index')
        ax.set_ylabel('Energy Efficiency (Mbit/kJ)')
        ax.set_title(f'{area}')
        ax.set_ylim(0, global_ymax)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3, axis='y')

    fig.suptitle('Scalability Analysis')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/scalability.pdf', dpi=150)
    plt.close(fig)
    print("[Plotter] Saved scalability.pdf")
