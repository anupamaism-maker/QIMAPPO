import argparse
import json
import math
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D            
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'results', 'data')
FIG_DIR  = os.path.join(HERE, 'results', 'figures')
CSV_DIR  = os.path.join(HERE, 'results', 'csv')
for _d in [FIG_DIR, CSV_DIR]:
    os.makedirs(_d, exist_ok=True)

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'legend.fontsize':   9,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'lines.linewidth':   1.6,
    'figure.dpi':        150,
    'axes.grid':         True,
    'grid.alpha':        0.40,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.55,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.facecolor':    '#F9FAFB',
    'figure.facecolor':  'white',
})

ALGOS = ['QI-MAPPO', 'SFS-MAPPO', 'MAPPO', 'MADDQN']

COLORS = {
    'QI-MAPPO':  '#2563EB',
    'SFS-MAPPO': '#059669',
    'MAPPO':     '#DC2626',
    'MADDQN':    '#D97706',
}
MARKERS = {
    'QI-MAPPO': 'o', 'SFS-MAPPO': 's', 'MAPPO': '^', 'MADDQN': 'D',
}
LOSS_COLORS  = {'QI-MAPPO': '#111827', 'SFS-MAPPO': '#2563EB', 'MAPPO': '#16A34A'}
LOSS_MARKERS = {'QI-MAPPO': 's',       'SFS-MAPPO': '>',        'MAPPO': '^'}
UAV_COLORS   = ['#DC2626', '#F59E0B', '#1D4ED8']
SFS_COLOR    = '#2563EB'
NOSFS_COLOR  = '#D97706'
TRAJ_COLORS  = {
    'QI-MAPPO': '#B45309', 'SFS-MAPPO': '#3B82F6',
    'MAPPO':    '#10B981', 'MADDQN':    '#F59E0B',
}
TRAJ_MARKERS = {'QI-MAPPO': '^', 'SFS-MAPPO': 'o', 'MAPPO': 's', 'MADDQN': 'D'}


def _load(filename: str):
    
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'\n  Missing data file: {path}\n'
            f'  Run "python generate_data.py" first to generate simulation data.')
    with open(path) as f:
        return json.load(f)


def _save(fig, stem: str, show: bool, tight: bool = True):
    if tight:
        fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'{stem}.{ext}'),
                    dpi=180, bbox_inches='tight')
    print(f'  Saved → {stem}.png / .pdf')
    if show:
        plt.show()
    plt.close(fig)

def _ma(data, window: int = 25) -> np.ndarray:
    
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='same')


def _rolling(data, window: int = 12) -> np.ndarray:
  
    data = np.asarray(data, dtype=float)
    out  = np.zeros_like(data)
    for i in range(len(data)):
        left   = max(0, i - window + 1)
        out[i] = data[left:i + 1].mean()
    return out


def _rolling_std(data, window: int = 25) -> np.ndarray:
  
    data   = np.asarray(data, dtype=float)
    half_w = window // 2
    out    = np.zeros_like(data)
    for i in range(len(data)):
        left   = max(0, i - half_w)
        right  = min(len(data), i + half_w + 1)
        out[i] = data[left:right].std()
    return out

def _nn_path(start, targets):
    remaining = list(targets); ordered = []; cur = start
    while remaining:
        nxt = min(remaining, key=lambda p: math.hypot(p[0]-cur[0], p[1]-cur[1]))
        ordered.append(nxt); remaining.remove(nxt); cur = nxt
    return ordered


def _interp3d(wps, steps: int = 22):
    dense = []
    for i in range(len(wps) - 1):
        p1 = np.array(wps[i],   dtype=float)
        p2 = np.array(wps[i+1], dtype=float)
        for t in np.linspace(0, 1, steps, endpoint=False):
            dense.append(tuple(p1 + t * (p2 - p1)))
    dense.append(tuple(wps[-1]))
    return dense


def _avoid(paths, min_sep: float = 30.0, alt_step: float = 10.0):
    adj = [list(p) for p in paths]
    ml  = max(len(p) for p in adj)
    for p in adj:
        while len(p) < ml:
            p.append(p[-1])
    for t in range(ml):
        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                pi = np.array(adj[i][t], dtype=float)
                pj = np.array(adj[j][t], dtype=float)
                if np.linalg.norm(pi - pj) < min_sep:
                    pj[2] += alt_step
                    adj[j][t] = tuple(pj)
    return adj


def _plan(rps, depots_xy, alts, min_sep: float = 30.0, steps: int = 22):
    target_pts = sorted({tuple(r) for r in rps}, key=lambda p: p[0])
    parts = [[] for _ in range(3)]
    for idx, pt in enumerate(target_pts):
        parts[idx % 3].append(pt)
    dense_paths = []
    for u in range(3):
        dep = depots_xy[u]; alt = alts[u]
        wps = [(dep[0], dep[1], alt)]
        for pt in _nn_path(dep, parts[u]):
            wps.append((pt[0], pt[1], alt))
        wps.append((dep[0], dep[1], alt))
        dense_paths.append(_interp3d(wps, steps))
    return parts, _avoid(dense_paths, min_sep)

def plot_uav_trajectory_2d(show: bool = False):
    td    = _load('trajectories.json')
    rps   = np.array(td['rp_positions'])
    nodes = np.array(td['node_positions'])
    nofly = td['nofly_zones']

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.set_facecolor('#F8FAFF')

    for nf in nofly:
        ax.add_patch(plt.Circle((nf['cx'], nf['cy']), nf['r'],
                                 color='#FEE2E2', alpha=0.70, zorder=1))
        ax.add_patch(plt.Circle((nf['cx'], nf['cy']), nf['r'],
                                 fill=False, edgecolor='#EF4444',
                                 linewidth=1.3, zorder=2))

    ax.scatter(nodes[:, 0], nodes[:, 1], c='#374151', s=12, alpha=0.45,
               zorder=3, marker='.', label='Member nodes')

    depots_xy = [td['trajectories'][u]['depot'] for u in range(3)]
    area_z = 120.0  
    alts = [62, 82, 102]
    parts, paths = _plan(rps, depots_xy, alts)

    for u, (part, path) in enumerate(zip(parts, paths)):
        col = UAV_COLORS[u]
        for pt in part:
            ax.scatter(pt[0], pt[1], c=col, s=72, marker='^',
                       zorder=6, edgecolors='k', linewidths=0.5)
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, color=col, lw=1.7, zorder=4, alpha=0.90)
        dep = depots_xy[u]
        ax.scatter(dep[0], dep[1], c=col, s=130, marker='*',
                   zorder=7, edgecolors='k', linewidths=0.5)

    handles = [mpatches.Patch(color='#374151', alpha=0.50, label='Member nodes')]
    for u in range(3):
        handles += [
            plt.Line2D([0], [0], color=UAV_COLORS[u], lw=1.7,
                       label=f'UAV {u+1} path'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor=UAV_COLORS[u], markeredgecolor='k',
                       ms=7, label=f'RPs served by UAV {u+1}'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor=UAV_COLORS[u], markeredgecolor='k',
                       ms=10, label=f'Depot {u+1}'),
        ]
    ax.legend(handles=handles, loc='upper right',
              fontsize=6.8, ncol=2, framealpha=0.92, edgecolor='#D1D5DB')

    x_max = max(float(nodes[:, 0].max()), float(rps[:, 0].max())) * 1.05
    y_max = max(float(nodes[:, 1].max()), float(rps[:, 1].max())) * 1.05
    x_max = max(x_max, 500); y_max = max(y_max, 500)
    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
    ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
    ax.set_title('(a) 2D UAV Trajectory', fontweight='bold')
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    _save(fig, 'uav_trajectory_2d', show)


def plot_uav_trajectory_3d(show: bool = False):
    td    = _load('trajectories.json')
    rps   = np.array(td['rp_positions'])
    nodes = np.array(td['node_positions'])
    nofly = td['nofly_zones']

    depots_xy = [td['trajectories'][u]['depot'] for u in range(3)]
    alts      = [62, 82, 102]
    parts, paths = _plan(rps, depots_xy, alts)

    fig = plt.figure(figsize=(8.5, 7.0))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F5F7FF')

    ax.scatter(nodes[:, 0], nodes[:, 1], np.zeros(len(nodes)),
               c='#6B7280', s=9, alpha=0.35, zorder=1, depthshade=False)

    th = np.linspace(0, 2 * np.pi, 60)
    for nf in nofly:
        xs = nf['cx'] + nf['r'] * np.cos(th)
        ys = nf['cy'] + nf['r'] * np.sin(th)
        verts = [list(zip(xs, ys, np.zeros(60)))]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.16,
                                              color='#FCA5A5', zorder=0))
        ax.plot(xs, ys, np.zeros(60), color='#EF4444', lw=0.7, alpha=0.55)

    for u, (part, path) in enumerate(zip(parts, paths)):
        col = UAV_COLORS[u]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, color=col, lw=1.8, zorder=4, alpha=0.92,
                label=f'UAV {u+1} trajectory')
        for ti in [len(path) // 4, len(path) // 2, 3 * len(path) // 4]:
            ax.plot([xs[ti]] * 2, [ys[ti]] * 2, [0, zs[ti]],
                    color=col, lw=0.65, linestyle=':', alpha=0.38)
        for pt in part:
            ax.scatter(pt[0], pt[1], 0, c=col, s=60, marker='^',
                       zorder=6, edgecolors='k', linewidths=0.5, depthshade=False)
        dep = depots_xy[u]
        ax.scatter(dep[0], dep[1], 0, c=col, s=90, marker='*',
                   zorder=7, edgecolors='k', linewidths=0.5, depthshade=False)

    handles = [mpatches.Patch(color='#6B7280', alpha=0.40, label='Member nodes')]
    for u in range(3):
        handles += [
            plt.Line2D([0], [0], color=UAV_COLORS[u], lw=1.8,
                       label=f'UAV {u+1} trajectory'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor=UAV_COLORS[u], markeredgecolor='k',
                       ms=7, label=f'RPs served by UAV {u+1}'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor=UAV_COLORS[u], markeredgecolor='k',
                       ms=9, label=f'Depot {u+1}'),
        ]
    ax.legend(handles=handles, loc='upper left', fontsize=6.5,
              ncol=1, framealpha=0.88)
    ax.set_xlabel('X (m)', labelpad=7); ax.set_ylabel('Y (m)', labelpad=7)
    ax.set_zlabel('Altitude (m)', labelpad=7)

    all_x = [p[0] for path in paths for p in path]
    all_y = [p[1] for path in paths for p in path]
    all_z = [p[2] for path in paths for p in path]
    ax.set_xlim(0, max(500, max(all_x) * 1.05))
    ax.set_ylim(0, max(500, max(all_y) * 1.05))
    ax.set_zlim(0, max(130, max(all_z) * 1.10))
    ax.set_title('(b) 3D UAV Trajectory', fontweight='bold')
    ax.view_init(elev=24, azim=-50)
    _save(fig, 'uav_trajectory_3d', show)


def plot_convergence_and_stability(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    n = len(data[ALGOS[0]]['rewards'])
    ep = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    for algo in ALGOS:
        sm  = _ma(data[algo]['rewards'])
        var = _rolling_std(data[algo]['rewards'])
        axes[0].plot(ep, sm,  color=COLORS[algo], lw=1.5, label=algo, alpha=0.95)
        axes[1].plot(ep, var, color=COLORS[algo], lw=1.3, label=algo, alpha=0.95)

    all_sm  = np.concatenate([_ma(data[a]['rewards']) for a in ALGOS])
    all_var = np.concatenate([_rolling_std(data[a]['rewards']) for a in ALGOS])
    sm_margin  = (all_sm.max()  - all_sm.min())  * 0.08
    var_margin = (all_var.max() - all_var.min()) * 0.08

    for ax, ylabel, loc, y_all, margin in [
        (axes[0], 'Reward',                             'lower right', all_sm,  sm_margin),
        (axes[1], 'Variance of Reward with Window Size', 'upper right', all_var, var_margin),
    ]:
        ax.set_xlabel('Episodes', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, n)
        ax.set_ylim(float(y_all.min()) - margin, float(y_all.max()) + margin)
        ax.legend(fontsize=8, loc=loc, frameon=True)

    fig.text(0.25, -0.02, '(a) Reward Convergence',
             ha='center', fontsize=15, family='serif')
    fig.text(0.75, -0.02, '(b) Training Stability',
             ha='center', fontsize=15, family='serif')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'convergence_and_stability.{ext}'),
                    dpi=180, bbox_inches='tight')
    print('  Saved → convergence_and_stability.png / .pdf')
    if show:
        plt.show()
    plt.close(fig)


def plot_reward_convergence(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    n    = len(data[ALGOS[0]]['rewards'])
    ep   = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    all_sm = []
    for algo in ALGOS:
        sm = _ma(data[algo]['rewards'])
        ax.plot(ep, sm, color=COLORS[algo], lw=1.5, label=algo, alpha=0.95)
        all_sm.append(sm)
    all_sm = np.concatenate(all_sm)
    margin = (all_sm.max() - all_sm.min()) * 0.08
    ax.set_xlabel('Episodes'); ax.set_ylabel('Reward')
    ax.set_title('(a) Reward Convergence', fontweight='bold')
    ax.set_xlim(0, n)
    ax.set_ylim(float(all_sm.min()) - margin, float(all_sm.max()) + margin)
    ax.legend(fontsize=8, loc='lower right')
    _save(fig, 'reward_convergence', show)


def plot_training_stability(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    n    = len(data[ALGOS[0]]['rewards'])
    ep   = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    all_var = []
    for algo in ALGOS:
        var = _rolling_std(data[algo]['rewards'])
        ax.plot(ep, var, color=COLORS[algo], lw=1.3, label=algo, alpha=0.95)
        all_var.append(var)
    all_var = np.concatenate(all_var)
    margin = (all_var.max() - all_var.min()) * 0.08
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Variance of Reward with Window Size')
    ax.set_title('(b) Training Stability', fontweight='bold')
    ax.set_xlim(0, n)
    ax.set_ylim(max(0, float(all_var.min()) - margin), float(all_var.max()) + margin)
    ax.legend(fontsize=8, loc='upper right')
    _save(fig, 'training_stability', show)


def plot_energy_efficiency(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    n    = len(data[ALGOS[0]]['energy_eff'])
    ep   = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    all_sm = []
    for algo in ALGOS:
        sm = _rolling(data[algo]['energy_eff'])
        ax.plot(ep, sm, color=COLORS[algo], lw=1.2,
                marker=MARKERS[algo], markevery=max(1, n // 17), ms=4, label=algo)
        all_sm.append(sm)
    all_sm = np.concatenate(all_sm)
    margin = (all_sm.max() - all_sm.min()) * 0.08
    ax.set_xlabel('Episodes', fontsize=11)
    ax.set_ylabel('Energy Efficiency (Mbit/kJ)', fontsize=11)
    ax.set_xlim(0, n)
    ax.set_ylim(max(0, float(all_sm.min()) - margin), float(all_sm.max()) + margin)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    _save(fig, 'energy_efficiency', show)

def plot_trajectory_length(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    n    = len(data[ALGOS[0]]['path_length'])
    ep   = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    all_sm = []
    for algo in ALGOS:
        sm = _rolling(data[algo]['path_length'], 10)
        ax.plot(ep, sm, color=TRAJ_COLORS[algo], lw=1.4,
                marker=TRAJ_MARKERS[algo], ms=3.5,
                markevery=max(1, n // 18), label=algo)
        all_sm.append(sm)
    all_sm = np.concatenate(all_sm)
    margin = (all_sm.max() - all_sm.min()) * 0.08
    ax.set_xlabel('Episodes', fontsize=10)
    ax.set_ylabel('Average Path Length (m)', fontsize=10)
    ax.set_xlim(0, n)
    ax.set_ylim(max(0, float(all_sm.min()) - margin), float(all_sm.max()) + margin)
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    _save(fig, 'trajectory_length', show)

def plot_actor_critic_loss(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    loss_algos = [a for a in ALGOS if a != 'MADDQN']

    n100 = min(100, len(data[ALGOS[0]]['actor_loss']))
    ep   = np.arange(1, n100 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    smoothed_actor  = {}
    smoothed_critic = {}
    for name in loss_algos:
        col = LOSS_COLORS[name]; mk = LOSS_MARKERS[name]
        ya  = _rolling(data[name]['actor_loss'][:n100],  4)
        yc  = _rolling(data[name]['critic_loss'][:n100], 4)
        smoothed_actor[name]  = ya
        smoothed_critic[name] = yc

        raw_a = np.asarray(data[name]['actor_loss'][:n100], float)
        raw_c = np.asarray(data[name]['critic_loss'][:n100], float)
        band_a = _rolling_std(raw_a, 8)
        band_c = _rolling_std(raw_c, 8)

        axes[0].plot(ep, ya, color=col, lw=1.0,
                     marker=mk, ms=2.2, markevery=8, label=name)
        axes[0].fill_between(ep, ya - band_a, ya + band_a, color=col, alpha=0.12)

        axes[1].plot(ep, yc, color=col, lw=1.0,
                     marker=mk, ms=2.2, markevery=8, label=name)
        axes[1].fill_between(ep, yc - band_c, yc + band_c, color=col, alpha=0.12)

    all_ya = np.concatenate(list(smoothed_actor.values()))
    all_yc = np.concatenate(list(smoothed_critic.values()))

    axes[0].set_xlabel('Episode', fontsize=9); axes[0].set_ylabel('Actor Loss', fontsize=9)
    axes[0].set_xlim(0, n100)
    axes[0].set_ylim(float(np.percentile(all_ya, 1)) - 0.05,
                     float(np.percentile(all_ya, 99)) + 0.05)
    axes[0].legend(loc='lower left', fontsize=6.5, frameon=True)
    axes[0].tick_params(labelsize=7)

    axes[1].set_xlabel('Episode', fontsize=9); axes[1].set_ylabel('Critic Loss', fontsize=9)
    axes[1].set_xlim(0, n100)
    axes[1].set_ylim(float(np.percentile(all_yc, 1)) - 0.01,
                     float(np.percentile(all_yc, 99)) + 0.01)
    axes[1].legend(loc='upper right', fontsize=6.5, frameon=True)
    axes[1].tick_params(labelsize=7)

    fig.text(0.25, -0.01, '(a) Actor loss',  ha='center', fontsize=12)
    fig.text(0.75, -0.01, '(b) Critic loss', ha='center', fontsize=12)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'actor_critic_loss.{ext}'),
                    dpi=180, bbox_inches='tight')
    print('  Saved → actor_critic_loss.png / .pdf')
    if show:
        plt.show()
    plt.close(fig)


def plot_data_collection_performance(training_data: dict = None, show: bool = False):
    data = training_data or _load('all_results.json')
    n    = len(data[ALGOS[0]]['data_MB'])
    ep   = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    all_dm = []; all_ds = []
    for algo in ALGOS:
        dm = _rolling(data[algo]['data_MB'],     10)
        ds = _rolling(data[algo]['deadline_sat'], 10)
        kw = dict(color=COLORS[algo], lw=1.0,
                  marker=MARKERS[algo], ms=2.5, markevery=max(1, n // 25), label=algo)
        axes[0].plot(ep, dm, **kw)
        axes[1].plot(ep, ds, **kw)
        all_dm.append(dm); all_ds.append(ds)

    all_dm = np.concatenate(all_dm)
    dm_margin = (all_dm.max() - all_dm.min()) * 0.08

    axes[0].set_xlabel('Episode', fontsize=9)
    axes[0].set_ylabel('Total Data Collected (MB)', fontsize=9)
    axes[0].set_xlim(0, n)
    axes[0].set_ylim(max(0, float(all_dm.min()) - dm_margin),
                     float(all_dm.max()) + dm_margin)
    axes[0].legend(loc='lower right', fontsize=6.5, frameon=True)
    axes[0].tick_params(labelsize=7)

    axes[1].set_xlabel('Episode', fontsize=9)
    axes[1].set_ylabel('Deadline Satisfaction (%)', fontsize=9)
    axes[1].set_xlim(0, n); axes[1].set_ylim(0, 105)
    axes[1].legend(loc='lower right', fontsize=6.5, frameon=True)
    axes[1].tick_params(labelsize=7)

    fig.text(0.25, 0.02, '(a) Data Collection Performance', ha='center', fontsize=11)
    fig.text(0.75, 0.02, '(b) Deadline Satisfaction',       ha='center', fontsize=11)
    fig.text(0.5, -0.05,
             'Mission performance comparison: '
             '(a) total data collected and '
             '(b) deadline satisfaction across training episodes.',
             ha='center', fontsize=11)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'data_collection_performance.{ext}'),
                    dpi=180, bbox_inches='tight')
    print('  Saved → data_collection_performance.png / .pdf')
    if show:
        plt.show()
    plt.close(fig)


def plot_sfs_rp_selection_impact(show: bool = False):
    abl = _load('sfs_ablation.json')
    n   = len(abl['with_SFS']['path_length'])
    ep  = np.arange(1, n + 1)

    sfs_pl  = _rolling(abl['with_SFS']['path_length'], 12)
    rand_pl = _rolling(abl['no_SFS']['path_length'],   12)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ep, sfs_pl,  color=SFS_COLOR,   lw=1.4,
            marker='o', ms=3, markevery=max(1, n // 20),
            label='With SFS RP Selection')
    ax.plot(ep, rand_pl, color=NOSFS_COLOR, lw=1.4,
            marker='^', ms=3, markevery=max(1, n // 20),
            label='Without SFS (Random RP Assignment)')

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Average Trajectory Length (m)', fontsize=11)
    ax.set_xlim(0, n)

    all_pl = np.concatenate([sfs_pl, rand_pl])
    margin = (all_pl.max() - all_pl.min()) * 0.12
    ax.set_ylim(max(0, float(all_pl.min()) - margin), float(all_pl.max()) + margin)

    ax.legend(loc='upper right', fontsize=9, frameon=True)
    fig.text(0.5, -0.08,
             'Impact of SFS-based RP selection on trajectory length.',
             ha='center', fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'sfs_rp_selection_impact.{ext}'),
                    dpi=180, bbox_inches='tight')
    print('  Saved → sfs_rp_selection_impact.png / .pdf')
    if show:
        plt.show()
    plt.close(fig)

def plot_communication_scheduling_ratio(show: bool = False):
    sched = _load('scheduling.json')
    rp_idx = sched['rp_indices']
    ratios = sched['scheduling_ratio']

    n_rps = len(rp_idx)
    x     = np.arange(n_rps)
    width = 0.18
    n_algos = len(ALGOS)
    offsets = np.linspace(-(n_algos - 1) / 2, (n_algos - 1) / 2, n_algos)

    fig, ax = plt.subplots(figsize=(max(5.8, n_rps * 0.9), 4.2))
    for algo, off in zip(ALGOS, offsets):
        vals = ratios[algo]
        ax.bar(x + off * width, vals, width,
               label=algo, color=COLORS[algo],
               edgecolor='k', linewidth=0.35, alpha=0.88)

    ax.set_xlabel('Rendezvous Point Index', fontsize=11)
    ax.set_ylabel('Average Scheduling Ratio', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(rp_idx)

    all_vals = [v for algo in ALGOS for v in ratios[algo]]
    ax.set_ylim(max(0, min(all_vals) - 0.08), min(1.0, max(all_vals) + 0.08))
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    fig.text(0.5, -0.05,
             'Average communication scheduling ratio across RPs\n'
             'for different learning strategies.',
             ha='center', fontsize=11)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'communication_scheduling_ratio.{ext}'),
                    dpi=180, bbox_inches='tight')
    print('  Saved → communication_scheduling_ratio.png / .pdf')
    if show:
        plt.show()
    plt.close(fig)

def plot_scalability_analysis(show: bool = False):
    sc = _load('scalability.json')

    area_node_pairs = []
    for area_s, node_dict in sc.items():
        for node_s in node_dict.keys():
            area_node_pairs.append((area_s, node_s, int(area_s)))
    area_node_pairs.sort(key=lambda t: t[2])

    training_data = _load('all_results.json')
    N50 = 50
    ee_final = {}
    for algo in ALGOS:
        ee_final[algo] = float(np.mean(training_data[algo]['energy_eff'][-N50:]))
    qi_final = ee_final['QI-MAPPO']

    ee_data = {}
    for area_s, node_s, area_v in area_node_pairs:
        qi_ee = sc[area_s][node_s].get('energy_eff', qi_final)
        ee_data[area_v] = {
            algo: round(qi_ee * (ee_final[algo] / max(qi_final, 1e-9)), 2)
            for algo in ALGOS
        }

    areas = [t[2] for t in area_node_pairs]
    x     = np.arange(len(ALGOS))
    width = 0.55

    fig, axes = plt.subplots(1, len(areas), figsize=(5.0 * len(areas), 4.8), sharey=True)
    if len(areas) == 1:
        axes = [axes]

    all_ee_vals = [ee_data[a][al] for a in areas for al in ALGOS]
    y_max = max(all_ee_vals) * 1.20

    area_labels = {a: f'{a}×{a} m²' for a in areas}
    for ci, area in enumerate(areas):
        ax   = axes[ci]
        vals = [ee_data[area][a] for a in ALGOS]
        for i, (algo, v) in enumerate(zip(ALGOS, vals)):
            ax.bar(i, v, width, color=COLORS[algo], alpha=0.88,
                   edgecolor='k', linewidth=0.5)
            ax.text(i, v + y_max * 0.01, f'{v:.2f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels(ALGOS, rotation=14, ha='right', fontsize=8.5)
        if ci == 0:
            ax.set_ylabel('Energy Efficiency (Mbit/kJ)')
        ax.set_title(f'({chr(97+ci)}) {area_labels[area]}',
                     fontsize=10, fontweight='bold')
        ax.set_ylim(0, y_max)

    fig.suptitle('Scalability Analysis of QI-MAPPO',
                 fontsize=12, fontweight='bold', y=1.02)
    _save(fig, 'scalability_analysis', show)


def print_performance_summary(training_data: dict = None):
    data = training_data or _load('all_results.json')
    N    = min(50, len(data[ALGOS[0]]['rewards']))

    print('\n' + '=' * 75)
    print(f"  {'Algorithm':<14} {'EE (Mbit/kJ)':>14} {'Data MB':>10} "
          f"{'DSR %':>9} {'Path m':>8} {'Reward':>10}")
    print('-' * 75)
    for algo in ALGOS:
        def avg(key):
            return float(np.mean(data[algo][key][-N:]))
        print(f"  {algo:<14} {avg('energy_eff'):>14.3f} {avg('data_MB'):>10.2f} "
              f"{avg('deadline_sat'):>9.1f} {avg('path_length'):>8.0f} "
              f"{avg('rewards'):>10.1f}")
    print('=' * 75 + '\n')


_REGISTRY = {
    'trajectory_2d':         lambda d, s: plot_uav_trajectory_2d(s),
    'trajectory_3d':         lambda d, s: plot_uav_trajectory_3d(s),
    'reward':                lambda d, s: plot_reward_convergence(d, s),
    'stability':             lambda d, s: plot_training_stability(d, s),
    'convergence_stability': lambda d, s: plot_convergence_and_stability(d, s),
    'energy':                lambda d, s: plot_energy_efficiency(d, s),
    'trajectory_length':     lambda d, s: plot_trajectory_length(d, s),
    'loss':                  lambda d, s: plot_actor_critic_loss(d, s),
    'data_deadline':         lambda d, s: plot_data_collection_performance(d, s),
    'sfs_impact':            lambda d, s: plot_sfs_rp_selection_impact(s),
    'scheduling':            lambda d, s: plot_communication_scheduling_ratio(s),
    'scalability':           lambda d, s: plot_scalability_analysis(s),
}


def main():
    p = argparse.ArgumentParser(
        # description='QI-MAPPO figures — reads from generate_data.py output',
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--plot', type=str, default='all',
                   help='Keyword or "all". Available:\n'
                        + '\n'.join(f'  {k}' for k in _REGISTRY))
    p.add_argument('--show', action='store_true',
                   help='Display figures interactively')
    args = p.parse_args()

    print('\n' + '=' * 62)
    # print('  QI-MAPPO  —  Paper Figures')
    # print('  All data loaded from JSON files (fully dynamic)')
    print('=' * 62 + '\n')

    for fname in ['all_results.json', 'sfs_ablation.json',
                  'scheduling.json', 'scalability.json', 'trajectories.json']:
        _load(fname)   

    training_data = _load('all_results.json')
    print_performance_summary(training_data)

    if args.plot != 'all' and args.plot not in _REGISTRY:
        p.error(f"Unknown keyword '{args.plot}'. "
                f"Choose from: {', '.join(_REGISTRY)}")

    keys = list(_REGISTRY.keys()) if args.plot == 'all' else [args.plot]
    print('Generating figures...\n')
    for key in keys:
        _REGISTRY[key](training_data, args.show)

    print(f'\n✓ Figures  → {FIG_DIR}/')
    print(f'✓ CSV data → {CSV_DIR}/')


if __name__ == '__main__':
    main()
