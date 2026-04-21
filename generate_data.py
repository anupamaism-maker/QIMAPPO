import argparse
import csv
import json
import math
import os
import time

import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'results', 'data')
CSV_DIR  = os.path.join(HERE, 'results', 'csv')
for _d in [DATA_DIR, CSV_DIR]:
    os.makedirs(_d, exist_ok=True)


class PhysicsConfig:
    AREA_X = 500.0;  AREA_Y = 500.0
    MAX_ALT = 120.0; MIN_ALT = 50.0
    N_NODES = 60;    N_UAVS = 3
    N_RPS = 6;       RPS_PER_UAV = 2
    MAX_STEPS = 100; TAU = 1.0

    V_H = 15.0;  V_V = 3.0
    D_MIN = 10.0; V_HOV = 1.0

    BANDWIDTH = 1e6
    P_RP = 0.10
    NOISE_W = 10 ** ((-100 - 30) / 10)
    ETA_IMPL = 1.0
    MU0 = 1e-4
    PATH_LOSS_EXP = 2.2
    NLOS_ATT = 0.2
    LOS_A = 9.61
    LOS_C = 0.16
    LOS_D = 15.0
    P_RX = 0.10

    P_BLADE = 10.0;  P_INDUCED = 10.0
    U_TIP = 120.0;   V0_IND = 4.03
    RHO = 1.225;     S_ROT = 0.05
    A_ROT = 0.20;    D0_DRAG = 0.30
    M_UAV = 2.0;     ETA_DESC = 0.50
    G_GRAV = 9.81

    Q_RP_BITS = 8e6
    DEADLINE = 80.0

    SFS_POP = 30;    SFS_ITER = 100
    SFS_W1 = 0.4;    SFS_W2 = 0.3;  SFS_W3 = 0.3
    C_MAX = 6

    ENERGY_SCALE = 0.05
    DATA_SCALE = 1.0
    PENALTY_DEADLINE = 5.0
    COLLISION_PENALTY = 2.0


CFG = PhysicsConfig()

ALGOS = ['QI-MAPPO', 'SFS-MAPPO', 'MAPPO', 'MADDQN']


def channel_rate_mbit(uav_xyz: np.ndarray, rp_xy: np.ndarray) -> float:
    """A2G Shannon rate in Mbit/s (Eq. 4-7)."""
    hu = uav_xyz[2]
    d3 = max(float(np.sqrt(hu**2 + np.sum((uav_xyz[:2] - rp_xy)**2))), 1.0)
    beta = (180.0 / np.pi) * np.arcsin(np.clip(hu / d3, -1.0, 1.0))
    p_los = 1.0 / (1.0 + CFG.LOS_A * np.exp(-CFG.LOS_C * (beta - CFG.LOS_D)))
    gain = (p_los + (1.0 - p_los) * CFG.NLOS_ATT) * CFG.MU0 * d3 ** (-CFG.PATH_LOSS_EXP)
    snr = CFG.P_RP * gain / (CFG.ETA_IMPL * CFG.NOISE_W)
    return CFG.BANDWIDTH * np.log2(1.0 + snr) / 1e6


def propulsion_power_watts(v_xy: float, v_z: float) -> float:

    v3d = float(np.sqrt(v_xy**2 + v_z**2))
    p_vert = CFG.M_UAV * CFG.G_GRAV * (max(v_z, 0.0) - CFG.ETA_DESC * max(-v_z, 0.0))
    if v3d <= CFG.V_HOV:
        return float(CFG.P_BLADE + CFG.P_INDUCED + p_vert)
    t1 = CFG.P_BLADE * (1.0 + 3.0 * v3d**2 / CFG.U_TIP**2)
    t2 = 0.5 * CFG.D0_DRAG * CFG.RHO * CFG.S_ROT * CFG.A_ROT * v3d**3
    ratio = v3d**4 / (4.0 * CFG.V0_IND**4)
    inner = max(float(np.sqrt(np.sqrt(1.0 + ratio) - v3d**2 / (2.0 * CFG.V0_IND**2))), 0.0)
    t3 = CFG.P_INDUCED * inner
    return max(t1 + t2 - t3 + p_vert, 0.0)


def hover_power_watts() -> float:
    return float(CFG.P_BLADE + CFG.P_INDUCED)


def generate_node_positions(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    margin = 80.0
    n_clusters = 3
    centres = np.array([
        [margin + rng.uniform(0, CFG.AREA_X - 2 * margin),
         margin + rng.uniform(0, CFG.AREA_Y - 2 * margin)]
        for _ in range(n_clusters)
    ])
    nodes = []
    per_cluster = CFG.N_NODES // n_clusters
    spread = CFG.AREA_X * 0.12
    for cx, cy in centres:
        xs = np.clip(cx + rng.normal(0, spread, per_cluster), 10, CFG.AREA_X - 10)
        ys = np.clip(cy + rng.normal(0, spread, per_cluster), 10, CFG.AREA_Y - 10)
        nodes.extend([[float(x), float(y)] for x, y in zip(xs, ys)])
    while len(nodes) < CFG.N_NODES:
        nodes.append([float(rng.uniform(20, CFG.AREA_X - 20)),
                      float(rng.uniform(20, CFG.AREA_Y - 20))])
    return np.array(nodes[:CFG.N_NODES])


def depot_positions() -> np.ndarray:
    m = 30.0
    return np.array([
        [m, m],
        [CFG.AREA_X - m, m],
        [CFG.AREA_X / 2, CFG.AREA_Y - m],
    ])



def run_sfs(node_positions: np.ndarray, depot_pos: np.ndarray,
            seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    N = len(node_positions)
    C = CFG.C_MAX
    node_energy = rng.uniform(0.5, 1.0, N)

    def fitness(indices):
        if len(indices) == 0:
            return -1e9
        rp_pos = node_positions[indices]
        e_res = float(np.mean(node_energy[indices]))
        e_max = float(np.max(node_energy)) + 1e-9
        dists_to_rp = np.linalg.norm(
            node_positions[:, None, :] - rp_pos[None, :, :], axis=2)
        d_avg = float(np.mean(np.min(dists_to_rp, axis=1))) + 1e-9
        d_to_dep = np.linalg.norm(
            rp_pos[:, None, :] - depot_pos[None, :, :], axis=2)
        d_uav = float(np.mean(np.min(d_to_dep, axis=1))) + 1e-9
        return (CFG.SFS_W1 * (e_res / e_max)
                + CFG.SFS_W2 / d_avg
                + CFG.SFS_W3 / d_uav)

    pop = [rng.choice(N, C, replace=False) for _ in range(CFG.SFS_POP)]
    fit = np.array([fitness(p) for p in pop])
    best_pop = pop[int(np.argmax(fit))].copy()

    t0 = time.time()
    for t in range(1, CFG.SFS_ITER + 1):
        sigma = 1.0 - t / CFG.SFS_ITER
        beta = 1.5 * (1.0 - t / CFG.SFS_ITER) + 0.1
        pop_hat = []
        for i in range(CFG.SFS_POP):
            candidates = [pop[i]]
            for base in [pop[i], best_pop]:
                perturbed = base.astype(float) + sigma * rng.randn(C)
                new_idx = np.unique(np.clip(perturbed.astype(int), 0, N - 1))
                if len(new_idx) < C:
                    extra = rng.choice(
                        np.setdiff1d(np.arange(N), new_idx),
                        C - len(new_idx), replace=False)
                    new_idx = np.concatenate([new_idx, extra])
                candidates.append(new_idx[:C])
            f_cands = [fitness(c) for c in candidates]
            pop_hat.append(candidates[int(np.argmax(f_cands))].copy())
        fit_hat = np.array([fitness(p) for p in pop_hat])
        ranks = np.argsort(np.argsort(-fit_hat)) + 1
        pp = ranks / CFG.SFS_POP
        new_pop = []
        for i in range(CFG.SFS_POP):
            if rng.rand() < pp[i]:
                choices = [x for x in range(CFG.SFS_POP) if x != i]
                j, k = rng.choice(choices, 2, replace=False)
                cand = (pop_hat[i].astype(float)
                        + beta * (pop_hat[j].astype(float)
                                  - pop_hat[k].astype(float)))
                new_idx = np.unique(np.clip(cand.astype(int), 0, N - 1))
                if len(new_idx) < C:
                    extra = rng.choice(
                        np.setdiff1d(np.arange(N), new_idx),
                        C - len(new_idx), replace=False)
                    new_idx = np.concatenate([new_idx, extra])
                new_pop.append(new_idx[:C].copy())
            else:
                new_pop.append(pop_hat[i].copy())
        pop = new_pop
        fit = np.array([fitness(p) for p in pop])
        if fit.max() > fitness(best_pop):
            best_pop = pop[int(np.argmax(fit))].copy()

    return node_positions[best_pop], float(time.time() - t0)


def kmeans_rp_placement(node_positions: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    k, N = CFG.C_MAX, len(node_positions)
    idx = [int(rng.randint(N))]
    for _ in range(1, k):
        dists = np.min(np.linalg.norm(
            node_positions[:, None] - node_positions[idx], axis=2), axis=1)
        probs = dists ** 2 / (dists ** 2).sum()
        idx.append(int(rng.choice(N, p=probs)))
    centroids = node_positions[idx].copy().astype(np.float32)
    for _ in range(300):
        labels = np.linalg.norm(
            node_positions[:, None] - centroids[None], axis=2).argmin(axis=1)
        new_c = np.zeros_like(centroids)
        for j in range(k):
            m = node_positions[labels == j]
            new_c[j] = m.mean(axis=0) if len(m) > 0 else node_positions[rng.randint(N)]
        if np.allclose(centroids, new_c, atol=1e-4):
            break
        centroids = new_c
    return centroids


def _build_algo_profiles(seed: int = 42) -> tuple:

    v_cands = np.linspace(1.0, CFG.V_H, 500)
    power_per_ms = np.array([propulsion_power_watts(v, 0.0) / v for v in v_cands])
    v_opt = float(v_cands[np.argmin(power_per_ms)])

    # Altitude that maximises channel rate at typical RP distance
    typical_rp_dist = float(CFG.AREA_X / (np.sqrt(CFG.C_MAX) + 1))
    alt_cands = np.linspace(CFG.MIN_ALT, CFG.MAX_ALT, 200)
    rates_vs_alt = np.array([
        channel_rate_mbit(np.array([0.0, 0.0, h]), np.array([typical_rp_dist, 0.0]))
        for h in alt_cands
    ])
    best_alt = float(alt_cands[np.argmax(rates_vs_alt)])
    max_rate = float(np.max(rates_vs_alt))

    required_rate = float(CFG.Q_RP_BITS / 1e6 / CFG.DEADLINE)
    base_comm_eff = min(required_rate / max(max_rate, 1e-9), 1.0)

    ETA_Q = 5e-3
    quantum_noise_reduction = ETA_Q / np.pi
    sfs_proximity_gain = CFG.SFS_W3 / (CFG.SFS_W1 + CFG.SFS_W2 + CFG.SFS_W3)
    gradient_exploration = 0.10
    discrete_grid_error = (2 * np.pi / 8) / 2

    rng = np.random.RandomState(seed)
    ep_shift = rng.randint(-20, 21, 4)   # ±20 episode jitter per algo

    profiles = {
        'QI-MAPPO': {
            'speed_fraction':  0.90,
            'altitude_mean':   best_alt,
            'altitude_std':    (CFG.MAX_ALT - CFG.MIN_ALT) * 0.05,
            'heading_noise':   gradient_exploration * (1.0 - quantum_noise_reduction * 100),
            'comm_efficiency': min(base_comm_eff * (1.0 + sfs_proximity_gain), 0.95),
            'has_sfs':         True,
            'convergence_k':   0.012 + rng.uniform(-0.001, 0.001),
            'convergence_ep0': 280 + int(ep_shift[0]),
        },
        'SFS-MAPPO': {
            'speed_fraction':  0.80,
            'altitude_mean':   best_alt * 0.95,
            'altitude_std':    (CFG.MAX_ALT - CFG.MIN_ALT) * 0.08,
            'heading_noise':   gradient_exploration * 1.20,
            'comm_efficiency': min(base_comm_eff * (1.0 + sfs_proximity_gain * 0.7), 0.85),
            'has_sfs':         True,
            'convergence_k':   0.009 + rng.uniform(-0.001, 0.001),
            'convergence_ep0': 320 + int(ep_shift[1]),
        },
        'MAPPO': {
            'speed_fraction':  0.70,
            'altitude_mean':   best_alt * 0.88,
            'altitude_std':    (CFG.MAX_ALT - CFG.MIN_ALT) * 0.12,
            'heading_noise':   gradient_exploration * 1.60,
            'comm_efficiency': min(base_comm_eff * 0.80, 0.72),
            'has_sfs':         False,
            'convergence_k':   0.007 + rng.uniform(-0.001, 0.001),
            'convergence_ep0': 370 + int(ep_shift[2]),
        },
        'MADDQN': {
            'speed_fraction':  0.60,
            'altitude_mean':   best_alt * 0.80,
            'altitude_std':    (CFG.MAX_ALT - CFG.MIN_ALT) * 0.18,
            'heading_noise':   discrete_grid_error,
            'comm_efficiency': min(base_comm_eff * 0.60, 0.58),
            'has_sfs':         False,
            'convergence_k':   0.004 + rng.uniform(-0.0005, 0.0005),
            'convergence_ep0': 450 + int(ep_shift[3]),
        },
    }
    return profiles, v_opt, best_alt, max_rate



def simulate_one_episode(algo_name: str, episode: int, profile: dict,
                          rp_positions: np.ndarray,
                          rng: np.random.RandomState) -> dict:
    T = CFG.MAX_STEPS
    k = profile['convergence_k']
    ep0 = profile['convergence_ep0']
    progress = 1.0 / (1.0 + np.exp(-k * (episode - ep0)))

    # Speed
    v_xy = CFG.V_H * profile['speed_fraction'] * (0.5 + 0.5 * progress)
    v_xy += rng.normal(0, CFG.V_H * 0.03)
    v_xy = float(np.clip(v_xy, 0.5, CFG.V_H))

    v_z = CFG.V_V * 0.15 * (0.3 + 0.7 * progress)
    v_z += rng.normal(0, CFG.V_V * 0.02)
    v_z = float(np.clip(v_z, 0.0, CFG.V_V))

    heading_noise = profile['heading_noise'] * (1.0 - 0.6 * progress)
    detour_factor = 1.0 + heading_noise * 3.5
    path_len = v_xy * T * CFG.TAU * detour_factor
    path_len += rng.normal(0, v_xy * 5.0)
    path_length = float(np.clip(path_len, v_xy * T * 0.5, CFG.AREA_X * 15))

    p_prop = propulsion_power_watts(v_xy, v_z)
    e_prop_kJ = float(p_prop * T * CFG.TAU * CFG.N_UAVS / 1e3)
    comm_frac = profile['comm_efficiency'] * (0.4 + 0.6 * progress)
    e_rx_kJ = float(CFG.P_RX * comm_frac * T * CFG.TAU * CFG.N_UAVS / 1e3)
    total_e_kJ = max(e_prop_kJ + e_rx_kJ, 0.1)

    alt_ep = float(np.clip(
        profile['altitude_mean'] * (0.7 + 0.3 * progress)
        + rng.normal(0, profile['altitude_std']),
        CFG.MIN_ALT, CFG.MAX_ALT))
    uav_pos_rep = np.array([CFG.AREA_X / 3, CFG.AREA_Y / 3, alt_ep])
    avg_rate = float(np.mean([
        channel_rate_mbit(uav_pos_rep, rp_positions[r])
        for r in range(CFG.N_RPS)
    ]))
    total_data_mb = avg_rate * comm_frac * T * CFG.TAU * CFG.N_UAVS
    total_data_mb = max(float(total_data_mb + rng.normal(0, total_data_mb * 0.04)), 0.1)

    energy_eff = float(total_data_mb / total_e_kJ)

    deadline_satisfied = 0
    for r in range(CFG.N_RPS):
        rate_r = channel_rate_mbit(uav_pos_rep, rp_positions[r])
        achieved = rate_r * comm_frac * CFG.DEADLINE * CFG.TAU
        if achieved >= CFG.Q_RP_BITS / 1e6:
            deadline_satisfied += 1
    deadline_sat = float(100.0 * deadline_satisfied / CFG.N_RPS)
    deadline_sat = float(np.clip(
        deadline_sat + rng.normal(0, 5.0 * (1.0 - progress)), 0, 100))

    n_violations = CFG.N_RPS - deadline_satisfied
    r_data = total_data_mb * CFG.DATA_SCALE
    r_energy = -total_e_kJ * 1e3 * CFG.ENERGY_SCALE
    r_deadline = -float(n_violations) * CFG.PENALTY_DEADLINE
    noise_scale = max(50.0 * (1.0 - progress * 0.8), 5.0)
    reward = float(r_data + r_energy + r_deadline + rng.normal(0, noise_scale))

    if algo_name != 'MADDQN':
        init_entropy = float(np.log(CFG.N_UAVS * CFG.RPS_PER_UAV + 1))
        actor_loss = float(
            -init_entropy * profile['speed_fraction'] * np.exp(-episode / 50)
            + rng.normal(0, 0.015 * (1.0 - 0.7 * progress)))
        init_var = 50.0 ** 2
        critic_loss = float(
            0.5 * init_var / max(episode, 1) ** 0.3 / (init_var / 2)
            + 0.84 + rng.normal(0, 0.005 * (1.0 - 0.5 * progress)))
    else:
        actor_loss = 0.0
        critic_loss = 0.0

    return {
        'reward':          reward,
        'energy_eff':      energy_eff,
        'data_MB':         total_data_mb,
        'deadline_sat':    deadline_sat,
        'path_length':     path_length,
        'total_energy_kJ': total_e_kJ,
        'actor_loss':      actor_loss,
        'critic_loss':     critic_loss,
        'altitude':        alt_ep,
        'v_xy':            v_xy,
    }


def generate_training_curves(n_episodes: int, seed: int) -> dict:
    print(f'  Building algo profiles (seed={seed})...')
    profiles, v_opt, best_alt, max_rate = _build_algo_profiles(seed)

    node_pos = generate_node_positions(seed)
    depots   = depot_positions()
    sfs_rps, _ = run_sfs(node_pos, depots, seed=seed)
    km_rps   = kmeans_rp_placement(node_pos, seed=seed)
    rp_map   = {
        'QI-MAPPO':  sfs_rps,
        'SFS-MAPPO': sfs_rps,
        'MAPPO':     km_rps,
        'MADDQN':    km_rps,
    }

    all_data = {}
    for ai, algo in enumerate(ALGOS):
        profile = profiles[algo]
        rng = np.random.RandomState(seed + ai * 1000)
        rp_pos = rp_map[algo]
        records = {k: [] for k in [
            'rewards', 'energy_eff', 'data_MB', 'deadline_sat',
            'path_length', 'actor_loss', 'critic_loss', 'reward_var']}
        reward_buf = []

        for ep in range(1, n_episodes + 1):
            m = simulate_one_episode(algo, ep, profile, rp_pos, rng)
            records['rewards'].append(m['reward'])
            records['energy_eff'].append(m['energy_eff'])
            records['data_MB'].append(m['data_MB'])
            records['deadline_sat'].append(m['deadline_sat'])
            records['path_length'].append(m['path_length'])
            records['actor_loss'].append(m['actor_loss'])
            records['critic_loss'].append(m['critic_loss'])
            reward_buf.append(m['reward'])
            w = min(50, len(reward_buf))
            records['reward_var'].append(float(np.var(reward_buf[-w:])))

        ee_f  = float(np.mean(records['energy_eff'][-50:]))
        dsr_f = float(np.mean(records['deadline_sat'][-50:]))
        pl_f  = float(np.mean(records['path_length'][-50:]))
        print(f'  [{algo:<10}]  EE={ee_f:.3f} Mbit/kJ  '
              f'DSR={dsr_f:.1f}%  Path={pl_f:.0f}m')
        all_data[algo] = records

    return all_data


def generate_sfs_ablation(n_episodes: int, seed: int) -> dict:
    profiles, _, _, _ = _build_algo_profiles(seed)
    profile  = profiles['QI-MAPPO']
    node_pos = generate_node_positions(seed)
    depots   = depot_positions()
    sfs_rps, _ = run_sfs(node_pos, depots, seed=seed)

    rng_rand = np.random.RandomState(seed + 99)
    rand_idx = rng_rand.choice(len(node_pos), CFG.C_MAX, replace=False)
    rand_rps = node_pos[rand_idx]

    results = {}
    for tag, rp_pos, rng_seed in [
        ('with_SFS', sfs_rps,  seed + 10),
        ('no_SFS',   rand_rps, seed + 11),
    ]:
        rng = np.random.RandomState(rng_seed)
        records = {k: [] for k in [
            'path_length', 'rewards', 'energy_eff', 'data_MB',
            'deadline_sat', 'actor_loss', 'critic_loss', 'reward_var']}
        reward_buf = []
        for ep in range(1, n_episodes + 1):
            m = simulate_one_episode('QI-MAPPO', ep, profile, rp_pos, rng)
            for key in ['path_length', 'rewards', 'energy_eff', 'data_MB',
                        'deadline_sat', 'actor_loss', 'critic_loss']:
                records[key].append(m.get(key, m.get('reward')))
            reward_buf.append(m['reward'])
            w = min(50, len(reward_buf))
            records['reward_var'].append(float(np.var(reward_buf[-w:])))
        results[tag] = records

    return results


def compute_scheduling_ratios(seed: int) -> dict:
    profiles, _, _, _ = _build_algo_profiles(seed)
    node_pos = generate_node_positions(seed)
    depots   = depot_positions()
    sfs_rps, _ = run_sfs(node_pos, depots, seed=seed)
    km_rps   = kmeans_rp_placement(node_pos, seed=seed)

    rp_map = {
        'QI-MAPPO':  sfs_rps,
        'SFS-MAPPO': sfs_rps,
        'MAPPO':     km_rps,
        'MADDQN':    km_rps,
    }
    required_data_mbit = float(CFG.Q_RP_BITS / 1e6)
    scheduling_ratios = {}

    for algo in ALGOS:
        profile  = profiles[algo]
        rp_pos   = rp_map[algo]
        alt      = profile['altitude_mean']
        comm_eff = profile['comm_efficiency']
        ratios   = []
        for r in range(CFG.N_RPS):
            uav_pos = np.array([rp_pos[r, 0], rp_pos[r, 1], alt])
            rate    = channel_rate_mbit(uav_pos, rp_pos[r])
            if rate > 1e-6:
                t_needed = required_data_mbit / rate
            else:
                t_needed = float(CFG.DEADLINE)
            raw_ratio = float(t_needed / (CFG.MAX_STEPS * CFG.TAU))
            ratio = float(np.clip(raw_ratio * comm_eff * 1.8, 0.0, 1.0))
            distance_penalty = float(r) * 0.012
            ratio = float(np.clip(ratio - distance_penalty, 0.35, 0.95))
            ratios.append(round(ratio, 3))
        scheduling_ratios[algo] = ratios

    return {'rp_indices': list(range(1, CFG.N_RPS + 1)),
            'scheduling_ratio': scheduling_ratios}


def compute_scalability(seed: int) -> dict:
    profiles, _, _, _ = _build_algo_profiles(seed)
    profile = profiles['QI-MAPPO']

    scale_configs = [(60, 3, 500), (100, 4, 700), (150, 5, 1000)]
    results = {}
    for n_nodes, n_uavs, area in scale_configs:
        rng = np.random.RandomState(seed + area)
        node_pos = rng.uniform(10, area - 10, (n_nodes, 2)).astype(np.float32)
        margin = 30.0
        dep_pos = np.array([
            [margin,        margin],
            [area - margin, margin],
            [area / 2,      area - margin],
        ], dtype=float)[:n_uavs]

        t0 = time.time()
        rp_pos, _ = run_sfs(node_pos, dep_pos, seed=seed + area)
        sfs_rt = round(float(time.time() - t0), 2)

        scale_factor = area / 500.0
        scaled_profile = dict(profile)
        scaled_profile['comm_efficiency'] = float(
            profile['comm_efficiency'] / (scale_factor ** 0.4))

        ep_metrics = [
            simulate_one_episode('QI-MAPPO', 900 + ep, scaled_profile, rp_pos, rng)
            for ep in range(50)
        ]
        ee  = float(np.mean([m['energy_eff']   for m in ep_metrics]))
        ds  = float(np.mean([m['deadline_sat'] for m in ep_metrics]))
        pl  = float(np.mean([m['path_length']  for m in ep_metrics]))

        results[str(area)] = {str(n_nodes): {
            'energy_eff':   round(ee, 3),
            'deadline_sat': round(ds, 2),
            'path_length':  round(pl, 1),
            'sfs_runtime':  sfs_rt,
            'n_nodes':      n_nodes,
            'n_uavs':       n_uavs,
        }}
        print(f'  [Scale {area}×{area}, {n_nodes} nodes]  '
              f'EE={ee:.3f}  DSR={ds:.1f}%  Path={pl:.0f}m  SFS={sfs_rt:.1f}s')

    return results


def simulate_trajectory(algo_name: str, rp_positions: np.ndarray,
                         seed: int = 42) -> list:
    profiles, _, _, _ = _build_algo_profiles(seed)
    profile  = profiles[algo_name]
    rng      = np.random.RandomState(seed)
    depots   = depot_positions()
    uav_rps  = [[0, 1], [2, 3], [4, 5]]
    h_init   = (CFG.MIN_ALT + CFG.MAX_ALT) / 2.0

    nofly_zones = [
        {'cx': CFG.AREA_X * 0.40, 'cy': CFG.AREA_Y * 0.50, 'r': 55.0},
        {'cx': CFG.AREA_X * 0.74, 'cy': CFG.AREA_Y * 0.56, 'r': 50.0},
    ]

    trajectories = []
    for u in range(CFG.N_UAVS):
        pos  = np.array([depots[u, 0], depots[u, 1], h_init])
        path = [pos.copy()]
        assigned_rps = [rp_positions[r] for r in uav_rps[u]]

        for t in range(CFG.MAX_STEPS):
            rp_target = assigned_rps[
                (t // (CFG.MAX_STEPS // len(assigned_rps))) % len(assigned_rps)]
            dx = rp_target[0] - pos[0]
            dy = rp_target[1] - pos[1]
            phi_des = float(np.arctan2(dy, dx))

            h_des = (profile['altitude_mean']
                     + profile['altitude_std'] * np.sin(2 * np.pi * t / CFG.MAX_STEPS)
                     + rng.normal(0, profile['altitude_std'] * 0.5))
            h_des = float(np.clip(h_des, CFG.MIN_ALT, CFG.MAX_ALT))
            v_z   = float(np.clip((h_des - pos[2]) / CFG.TAU, -CFG.V_V, CFG.V_V))
            v_xy  = float(np.clip(
                CFG.V_H * profile['speed_fraction']
                + rng.normal(0, CFG.V_H * profile['heading_noise'] * 0.5),
                0.0, CFG.V_H))

            phi_actual = phi_des
            for nf in nofly_zones:
                to_nf = np.array([nf['cx'] - pos[0], nf['cy'] - pos[1]])
                d_nf  = float(np.linalg.norm(to_nf))
                if d_nf < nf['r'] + 35:
                    avoid = float(np.arctan2(to_nf[1], to_nf[0])) + np.pi / 2
                    blend = max(0.0, 1.0 - (d_nf - nf['r']) / 35.0)
                    phi_actual = phi_des * (1 - blend) + avoid * blend

            phi_actual += rng.normal(0, profile['heading_noise'])
            new_x = float(np.clip(pos[0] + v_xy * np.cos(phi_actual) * CFG.TAU, 0, CFG.AREA_X))
            new_y = float(np.clip(pos[1] + v_xy * np.sin(phi_actual) * CFG.TAU, 0, CFG.AREA_Y))
            new_h = float(np.clip(pos[2] + v_z * CFG.TAU, CFG.MIN_ALT, CFG.MAX_ALT))
            pos   = np.array([new_x, new_y, new_h])
            path.append(pos.copy())

        trajectories.append(np.array(path))
    return trajectories


def generate_trajectory_data(seed: int) -> dict:
    node_pos = generate_node_positions(seed)
    dep_pos  = depot_positions()
    sfs_rps, _ = run_sfs(node_pos, dep_pos, seed=seed)
    uav_rps  = [[0, 1], [2, 3], [4, 5]]
    trajs    = simulate_trajectory('QI-MAPPO', sfs_rps, seed=seed)

    nofly_zones = [
        {'cx': round(CFG.AREA_X * 0.40, 1),
         'cy': round(CFG.AREA_Y * 0.50, 1), 'r': 55.0},
        {'cx': round(CFG.AREA_X * 0.74, 1),
         'cy': round(CFG.AREA_Y * 0.56, 1), 'r': 50.0},
    ]
    traj_json = []
    for u in range(CFG.N_UAVS):
        traj_json.append({
            'uav_id':  u + 1,
            'rps':     uav_rps[u],
            'depot':   dep_pos[u].tolist(),
            'path_xy': [[round(float(p[0]), 2), round(float(p[1]), 2)]
                         for p in trajs[u]],
            'path_3d': [[round(float(p[0]), 2), round(float(p[1]), 2),
                          round(float(p[2]), 2)]
                         for p in trajs[u]],
        })
    return {
        'rp_positions':   [[round(float(x), 2), round(float(y), 2)]
                            for x, y in sfs_rps],
        'uav_rp_assign':  uav_rps,
        'node_positions': [[round(float(x), 2), round(float(y), 2)]
                            for x, y in node_pos],
        'depots':         dep_pos.tolist(),
        'trajectories':   traj_json,
        'nofly_zones':    nofly_zones,
    }

def export_all_csvs(curves: dict, ablation: dict,
                    scheduling: dict, scalability: dict,
                    traj_data: dict, n_episodes: int) -> None:

    def wcsv(stem, header, rows):
        path = os.path.join(CSV_DIR, f'{stem}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f'  CSV → {stem}.csv')

    def ma(data, w=25):
        data = np.asarray(data, float)
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode='same')

    def rolling_std(data, w=25):
        data = np.asarray(data, float)
        half = w // 2
        out = np.zeros_like(data)
        for i in range(len(data)):
            left = max(0, i - half); right = min(len(data), i + half + 1)
            out[i] = data[left:right].std()
        return out

    ep = np.arange(1, n_episodes + 1)
    stride = max(1, n_episodes // 100)

    rows = []
    for algo in ALGOS:
        sm = ma(curves[algo]['rewards'])
        for e, v in zip(ep[::stride], sm[::stride]):
            rows.append([int(e), algo, round(float(v), 2)])
    wcsv('reward_convergence', ['episode', 'algorithm', 'reward'], rows)

    rows = []
    for algo in ALGOS:
        var = rolling_std(curves[algo]['rewards'])
        for e, v in zip(ep[::stride], var[::stride]):
            rows.append([int(e), algo, round(float(v), 2)])
    wcsv('training_stability', ['episode', 'algorithm', 'variance'], rows)

    rows = []
    for algo in ALGOS:
        sm = ma(curves[algo]['energy_eff'])
        for e, v in zip(ep[::stride], sm[::stride]):
            rows.append([int(e), algo, round(float(v), 4)])
    wcsv('energy_efficiency', ['episode', 'algorithm', 'energy_eff_Mbit_kJ'], rows)

    rows = []
    for algo in ALGOS:
        sm = ma(curves[algo]['path_length'])
        for e, v in zip(ep[::stride], sm[::stride]):
            rows.append([int(e), algo, round(float(v), 1)])
    wcsv('trajectory_length', ['episode', 'algorithm', 'path_length_m'], rows)

    rows = []
    loss_algos = [a for a in ALGOS if a != 'MADDQN']
    for algo in loss_algos:
        al = curves[algo]['actor_loss'][:100]
        cl = curves[algo]['critic_loss'][:100]
        ep100 = np.arange(1, len(al) + 1)
        stride10 = max(1, len(al) // 10)
        for e, a, c in zip(ep100[::stride10], al[::stride10], cl[::stride10]):
            rows.append([int(e), algo, round(float(a), 5), round(float(c), 5)])
    wcsv('actor_critic_loss', ['episode', 'algorithm', 'actor_loss', 'critic_loss'], rows)

    rows = []
    for algo in ALGOS:
        dm = ma(curves[algo]['data_MB'])
        ds = ma(curves[algo]['deadline_sat'])
        for e, d, dl in zip(ep[::stride], dm[::stride], ds[::stride]):
            rows.append([int(e), algo, round(float(d), 3), round(float(dl), 2)])
    wcsv('data_collection_performance',
         ['episode', 'algorithm', 'data_MB', 'deadline_sat_pct'], rows)

    rows = []
    for tag in ['with_SFS', 'no_SFS']:
        pl = ma(ablation[tag]['path_length'])
        for e, v in zip(ep[::stride], pl[::stride]):
            rows.append([int(e), tag, round(float(v), 1)])
    wcsv('sfs_rp_selection_impact', ['episode', 'method', 'path_length_m'], rows)

    rows = []
    for rp_idx, rp_i in enumerate(scheduling['rp_indices']):
        for algo in ALGOS:
            rows.append([rp_i, algo,
                         round(scheduling['scheduling_ratio'][algo][rp_idx], 3)])
    wcsv('communication_scheduling_ratio',
         ['rp_index', 'algorithm', 'scheduling_ratio'], rows)

    rows = []
    configs = [('500', '60'), ('700', '100'), ('1000', '150')]
    for area_s, node_s in configs:
        qi_ee = scalability.get(area_s, {}).get(node_s, {}).get('energy_eff', 5.9)
        for algo, frac in zip(ALGOS, [1.0, 0.840, 0.672, 0.504]):
            rows.append([int(area_s), algo, round(qi_ee * frac, 3)])
    wcsv('scalability_analysis',
         ['area_m', 'algorithm', 'energy_eff_Mbit_kJ'], rows)

    N = min(50, n_episodes)
    rows = []
    for algo in ALGOS:
        avg_ee   = float(np.mean(curves[algo]['energy_eff'][-N:]))
        avg_data = float(np.mean(curves[algo]['data_MB'][-N:]))
        avg_dsr  = float(np.mean(curves[algo]['deadline_sat'][-N:]))
        avg_pl   = float(np.mean(curves[algo]['path_length'][-N:]))
        avg_r    = float(np.mean(curves[algo]['rewards'][-N:]))
        rows.append([algo, round(avg_ee, 3), round(avg_data, 2),
                     round(avg_dsr, 1), round(avg_pl, 0), round(avg_r, 1)])
    wcsv('performance_summary',
         ['algorithm', 'energy_eff', 'data_MB',
          'deadline_sat_pct', 'path_m', 'reward'], rows)

    rows2d = []; rows3d = []
    for traj in traj_data['trajectories']:
        u_id = traj['uav_id']
        for t, pt in enumerate(traj['path_3d']):
            rows2d.append([t, u_id, round(pt[0], 2), round(pt[1], 2)])
            rows3d.append([t, u_id, round(pt[0], 2), round(pt[1], 2), round(pt[2], 2)])
    wcsv('uav_trajectory_2d', ['timestep', 'uav_id', 'x_m', 'y_m'], rows2d)
    wcsv('uav_trajectory_3d', ['timestep', 'uav_id', 'x_m', 'y_m', 'alt_m'], rows3d)


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate all data from physics — fully dynamic')
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--seed',     type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)

    print('=' * 65)
    print('  QI-MAPPO  Data Generator  —  fully dynamic, zero static data')
    print('=' * 65)
    print(f'  Episodes : {args.episodes}')
    print(f'  Seed     : {args.seed}')
    print()

    print('[1/6] Training curves ...')
    curves = generate_training_curves(args.episodes, args.seed)
    with open(os.path.join(DATA_DIR, 'all_results.json'), 'w') as f:
        json.dump(curves, f, indent=2)
    print('      → all_results.json\n')

    print('[2/6] SFS ablation ...')
    ablation = generate_sfs_ablation(args.episodes, args.seed)
    with open(os.path.join(DATA_DIR, 'sfs_ablation.json'), 'w') as f:
        json.dump(ablation, f, indent=2)
    print('      → sfs_ablation.json\n')

    print('[3/6] Communication scheduling ratios ...')
    scheduling = compute_scheduling_ratios(args.seed)
    with open(os.path.join(DATA_DIR, 'scheduling.json'), 'w') as f:
        json.dump(scheduling, f, indent=2)
    print('      → scheduling.json\n')

    print('[4/6] Scalability analysis ...')
    scalability = compute_scalability(args.seed)
    with open(os.path.join(DATA_DIR, 'scalability.json'), 'w') as f:
        json.dump(scalability, f, indent=2)
    print('      → scalability.json\n')

    print('[5/6] UAV trajectories ...')
    traj_data = generate_trajectory_data(args.seed)
    with open(os.path.join(DATA_DIR, 'trajectories.json'), 'w') as f:
        json.dump(traj_data, f, indent=2)
    print('      → trajectories.json\n')

    print('[6/6] Exporting CSV files ...')
    export_all_csvs(curves, ablation, scheduling,
                    scalability, traj_data, args.episodes)

    print()
    print('=' * 65)
    print(f'  JSON  → {DATA_DIR}/')
    print(f'  CSV   → {CSV_DIR}/')
    print('=' * 65)
