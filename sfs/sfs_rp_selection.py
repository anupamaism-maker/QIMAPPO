import numpy as np


class SFSRPSelector:
    

    def __init__(self, node_pos: np.ndarray,
                 node_energy: np.ndarray,
                 cfg):
        self.node_pos    = np.asarray(node_pos,    dtype=float)
        self.node_energy = np.asarray(node_energy, dtype=float)
        self.cfg         = cfg
        self.rng         = np.random.RandomState(cfg.SEED)

        self._diag = np.sqrt(cfg.AREA_X**2 + cfg.AREA_Y**2)
        self._r_comm = self._diag / 5.0

    
    def select(self):
    
        cfg = self.cfg
        pop_size = cfg.SFS_POP
        n_rps    = cfg.N_RPS
        n_iter   = cfg.SFS_ITER

        population = np.array([
            self.rng.choice(cfg.N_NODES, n_rps, replace=False)
            for _ in range(pop_size)
        ])

        fitnesses = np.array([self._fitness(ind) for ind in population])
        best_idx  = int(np.argmax(fitnesses))
        best_ind  = population[best_idx].copy()
        best_fit  = fitnesses[best_idx]

        for iteration in range(n_iter):
    
            alpha = 1.0 - iteration / n_iter         

            for i in range(pop_size):
                
                candidate = self._move(population[i], best_ind, alpha)
                f_cand    = self._fitness(candidate)

                if f_cand > fitnesses[i]:
                    population[i] = candidate
                    fitnesses[i]  = f_cand

                    if f_cand > best_fit:
                        best_ind = candidate.copy()
                        best_fit = f_cand

        rp_idx = best_ind
        rp_pos = self.node_pos[rp_idx]
        return rp_pos, rp_idx, best_fit

    def _fitness(self, indices: np.ndarray) -> float:
        """Compute the weighted fitness of a candidate RP set."""
        cfg    = self.cfg
        rp_pos = self.node_pos[indices]

        diffs    = self.node_pos[:, None, :] - rp_pos[None, :, :]   
        dists    = np.linalg.norm(diffs, axis=-1)                    
        covered  = np.any(dists <= self._r_comm, axis=1)             
        coverage = covered.mean()

        energy = self.node_energy[indices].mean()

        if len(indices) > 1:
            rp_diffs = rp_pos[:, None, :] - rp_pos[None, :, :]      
            rp_dists = np.linalg.norm(rp_diffs, axis=-1)            
            iu       = np.triu_indices(len(indices), k=1)
            mean_sep = rp_dists[iu].mean() / self._diag             
        else:
            mean_sep = 0.0

        fitness = (cfg.SFS_W1 * coverage
                   + cfg.SFS_W2 * energy
                   + cfg.SFS_W3 * mean_sep)
        return float(fitness)

    def _move(self, current: np.ndarray,
              best: np.ndarray,
              alpha: float) -> np.ndarray:
        
        n_rps      = self.cfg.N_RPS
        candidate  = current.copy()
        n_changes  = max(1, int(alpha * n_rps))   

        positions_to_change = self.rng.choice(n_rps, n_changes, replace=False)
        for pos in positions_to_change:
            if self.rng.random() < alpha:
               
                new_gene = best[pos]
            else:
                
                new_gene = self.rng.randint(0, self.cfg.N_NODES)

            if new_gene not in candidate:
                candidate[pos] = new_gene
            else:
                available = list(set(range(self.cfg.N_NODES)) - set(candidate))
                if available:
                    candidate[pos] = self.rng.choice(available)

        return candidate