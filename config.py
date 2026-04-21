import os
import numpy as np


class Config:

    AREA_X          = 500.0;   AREA_Y     = 500.0       # m
    MAX_ALT         = 120.0;   MIN_ALT    = 50.0         # m
    N_NODES         = 60;      N_UAVS     = 3
    N_RPS           = 6;       RPS_PER_UAV= 2
    MAX_STEPS       = 100;     TAU        = 1.0          # s

    # ── UAV Kinematics ────────────────────────────────────────────────
    V_H = 15.0; V_V = 3.0; D_MIN = 10.0; V_HOV = 1.0

    # ── A2G Channel ───────────────────────────────────────────────────
    BANDWIDTH     = 1e6;   P_RP       = 0.10
    NOISE_DBM     = -100.0
    NOISE_W       = 10 ** ((-100.0 - 30) / 10)
    ETA_IMPL      = 1.0;   MU0        = 1e-4
    PATH_LOSS_EXP = 2.2;   NLOS_ATT   = 0.2
    LOS_A = 9.61;  LOS_C = 0.16;  LOS_D = 15.0
    P_RX  = 0.10

    # ── Rotary-Wing Energy ────────────────────────────────────────────
    P_BLADE  = 10.0;  P_INDUCED = 10.0;  U_TIP   = 120.0
    V0_IND   = 4.03;  RHO       = 1.225; S_ROT   = 0.05
    A_ROT    = 0.20;  D0_DRAG   = 0.30;  M_UAV   = 2.0
    KAPPA_U  = 0.01;  ETA_DESC  = 0.50;  G_GRAV  = 9.81

    # ── Data & Deadlines ──────────────────────────────────────────────
    Q_RP_BITS        = 8e6;   DEADLINE         = 80.0
    PENALTY_DEADLINE = 5.0;   ENERGY_SCALE     = 0.05
    DATA_SCALE       = 1.0;   COLLISION_PENALTY= 2.0

    # ── SFS ───────────────────────────────────────────────────────────
    SFS_POP  = 30;   SFS_ITER = 100
    SFS_W1   = 0.4;  SFS_W2   = 0.3;  SFS_W3 = 0.3
    R_COMM   = 150.0; C_MAX   = 6

    # ── PPO / Training ────────────────────────────────────────────────
    N_EPISODES    = 1000;  BATCH_SIZE   = 256
    LR_ACTOR      = 1e-4;  LR_CRITIC    = 1e-4
    GAMMA         = 0.99;  GAE_LAMBDA   = 0.95
    PPO_CLIP      = 0.2;   ENT_COEF     = 0.01
    VF_COEF       = 0.5;   MAX_GRAD_NORM= 0.5
    PPO_EPOCHS    = 10;    REPLAY_SIZE  = int(1e6)
    HIDDEN_DIM    = 256;   SEED         = 42

    # ── PQC Parameters (Table) ─────────────────────────────
    N_QUBITS   = 8      # D — qubits (4 or 8)
    N_LAYERS   = 2      # L — variational layers
    QUANTUM_LR = 1e-3   # quantum parameter LR

    # ── MADDQN ───────────────────────────────────────────────────────
    N_DISC_V   = 5;   N_DISC_PHI = 8;  N_DISC_VZ = 3
    EPS_START  = 1.0; EPS_END    = 0.05; EPS_DECAY = 0.995
    TARGET_UPDATE = 100

    # ── Derived ───────────────────────────────────────────────────────
    OBS_DIM          = 4 + RPS_PER_UAV * 5
    GLOBAL_STATE_DIM = N_UAVS * OBS_DIM
    ACT_CONT         = 3
    ACT_DISC_N       = RPS_PER_UAV + 1
    MADDQN_N_ACTS    = N_DISC_V * N_DISC_PHI * N_DISC_VZ * ACT_DISC_N

    # ── I/O ───────────────────────────────────────────────────────────
    RESULTS_DIR    = "results"
    FIGURES_DIR    = "results/figures"
    CSV_DIR        = "results/csv"
    CHECKPOINT_DIR = "results/checkpoints"
    LOG_INTERVAL   = 10
    SAVE_INTERVAL  = 100

    def __init__(self):
        for d in [self.RESULTS_DIR, self.FIGURES_DIR,
                  self.CSV_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)
        np.random.seed(self.SEED)