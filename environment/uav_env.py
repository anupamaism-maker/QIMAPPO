import numpy as np


class UAVIoTEnv:
    def __init__(self, cfg, rp_positions=None):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.SEED)
        self.rp_positions = rp_positions   
        self._build_topology()
        # state holders
        self.uav_pos = self.uav_energy = None
        self.rp_data = self.rp_deadline = self.rp_sched = None
        self.t = 0; self.done = False

    def _build_topology(self):
        cfg = self.cfg
        self.node_pos = self.rng.uniform(0, cfg.AREA_X, (cfg.N_NODES, 2))
        if self.rp_positions is not None:
            self.rp_pos = np.array(self.rp_positions)
        else:
            idx = self.rng.choice(cfg.N_NODES, cfg.N_RPS, replace=False)
            self.rp_pos = self.node_pos[idx]
        self.uav_rp_map = {u: list(range(u*cfg.RPS_PER_UAV,
                                          (u+1)*cfg.RPS_PER_UAV))
                           for u in range(cfg.N_UAVS)}
        fracs = [[.1,.1],[.9,.1],[.5,.9],[.1,.9],[.9,.9],[.5,.5]]
        self.depots = np.array([[f[0]*cfg.AREA_X, f[1]*cfg.AREA_Y]
                                 for f in fracs[:cfg.N_UAVS]])

    
    def reset(self):
        cfg = self.cfg
        h0  = (cfg.MAX_ALT + cfg.MIN_ALT) / 2
        self.uav_pos    = np.column_stack([self.depots.copy(),
                                           np.full(cfg.N_UAVS, h0)]).astype(float)
        self.uav_energy = np.full(cfg.N_UAVS, 5000.0)
        self.rp_data    = np.full(cfg.N_RPS, cfg.Q_RP_BITS)
        self.rp_deadline= np.full(cfg.N_RPS, float(cfg.DEADLINE))
        self.rp_sched   = np.zeros((cfg.N_UAVS, cfg.N_RPS), int)
        self.t = 0; self.done = False
        return self._local_states()

    
    def step(self, actions):
        cfg = self.cfg; τ = cfg.TAU
        total_energy = 0.0; rp_met = 0; rewards = []
        self.rp_sched[:] = 0
        for u in range(cfg.N_UAVS):
            a = actions[u]
            vxy = float(a['v_xy']); phi = float(a['phi']); vz = float(a['v_z'])
            ad  = int(a['a_disc'])
    
            self.uav_pos[u,0] = np.clip(self.uav_pos[u,0]+vxy*np.cos(phi)*τ, 0, cfg.AREA_X)
            self.uav_pos[u,1] = np.clip(self.uav_pos[u,1]+vxy*np.sin(phi)*τ, 0, cfg.AREA_Y)
            self.uav_pos[u,2] = np.clip(self.uav_pos[u,2]+vz*τ, cfg.MIN_ALT, cfg.MAX_ALT)
    
            rp_idx = -1; data_recv = 0.0
            if ad < cfg.RPS_PER_UAV:
                r = self.uav_rp_map[u][ad]
                if self.rp_data[r] > 1e-3:
                    rp_idx = r; self.rp_sched[u,r] = 1
                    rate = self._rate(u, r)
                    recv = min(rate*τ, self.rp_data[r])
                    self.rp_data[r] -= recv; data_recv = recv
    
            spd  = np.sqrt(vxy**2 + vz**2)
            Efly = self._prop_power(spd, vz)*τ
            Ecom = cfg.P_RX*τ if rp_idx >= 0 else 0.0
            Estep= Efly + Ecom
            self.uav_energy[u] = max(0, self.uav_energy[u]-Estep)
            total_energy += Estep
    
            for r in self.uav_rp_map[u]: self.rp_deadline[r] -= 1.0
    
            col = sum(cfg.COLLISION_PENALTY
                      for u2 in range(cfg.N_UAVS) if u2!=u and
                      np.linalg.norm(self.uav_pos[u]-self.uav_pos[u2]) < cfg.D_MIN)
    
            nviol = sum(1 for r in self.uav_rp_map[u] if self.rp_deadline[r]<0)
            rew   = (-Estep*cfg.ENERGY_SCALE
                     - cfg.PENALTY_DEADLINE*nviol
                     + data_recv/cfg.Q_RP_BITS*cfg.DATA_SCALE*10
                     - col)
            rewards.append(float(rew))
            rp_met += sum(1 for r in self.uav_rp_map[u] if self.rp_data[r]<=1)
        self.t += 1; self.done = self.t >= cfg.MAX_STEPS
        return self._local_states(), rewards, self.done, \
               {'energy': total_energy, 'rp_met': min(rp_met, cfg.N_RPS)}

    
    def _local_states(self):
        cfg = self.cfg; states = []
        for u in range(cfg.N_UAVS):
            p = self.uav_pos[u]
            pn= np.array([p[0]/cfg.AREA_X, p[1]/cfg.AREA_Y, p[2]/cfg.MAX_ALT])
            En= self.uav_energy[u]/5000.0
            rf= []
            for r in self.uav_rp_map[u]:
                rf += [self.rp_pos[r,0]/cfg.AREA_X, self.rp_pos[r,1]/cfg.AREA_Y,
                       np.clip(self.rp_data[r]/cfg.Q_RP_BITS,0,1),
                       np.clip(self.rp_deadline[r]/cfg.DEADLINE,0,1),
                       float(self.rp_sched[u,r])]
            states.append(np.array([*pn, En, *rf], dtype=np.float32))
        return states

    def get_global_state(self):
        return np.concatenate(self._local_states()).astype(np.float32)

    
    def _rate(self, u, r):
        cfg = self.cfg
        h = self.uav_pos[u,2]; w = self.uav_pos[u,:2]; pr = self.rp_pos[r]
        d = np.sqrt(h**2 + np.sum((w-pr)**2) + 1e-9)
        β = (180/np.pi)*np.arcsin(np.clip(h/d,-1,1))
        PL= 1/(1+cfg.LOS_A*np.exp(-cfg.LOS_C*(β-cfg.LOS_D)))
        µL= cfg.MU0*d**(-cfg.PATH_LOSS_EXP)
        µN= cfg.NLOS_ATT*µL
        Eh= PL*µL + (1-PL)*µN
        SNR = cfg.P_RP*Eh/(cfg.ETA_IMPL*cfg.NOISE_W+1e-30)
        return float(cfg.BANDWIDTH*np.log2(1+SNR))

    
    def _prop_power(self, spd, vz):
        cfg = self.cfg
        Ph  = cfg.P_BLADE + cfg.P_INDUCED
        if spd <= cfg.V_HOV:
            Pfly = Ph
        else:
            v2  = spd**2; v4 = spd**4
            Pb  = cfg.P_BLADE*(1+3*v2/cfg.U_TIP**2)
            Pd  = 0.5*cfg.D0_DRAG*cfg.RHO*cfg.S_ROT*cfg.A_ROT*spd**3
            Pi  = cfg.P_INDUCED*np.sqrt(max(np.sqrt(1+v4/(4*cfg.V0_IND**4))
                                            - v2/(2*cfg.V0_IND**2), 1e-9))
            Pfly= Pb+Pd+Pi
        Pv = (cfg.M_UAV*cfg.G_GRAV*vz if vz>=0
              else cfg.ETA_DESC*cfg.M_UAV*cfg.G_GRAV*abs(vz))
        return float(Pfly+Pv)