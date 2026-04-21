import numpy as np, warnings

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    try:
        from qiskit import QuantumCircuit
        from qiskit.providers.aer import AerSimulator
        QISKIT_AVAILABLE = True
    except ImportError:
        QISKIT_AVAILABLE = False
        warnings.warn("[PQC] Qiskit not found — numpy fallback active.\n"
                      "Install: pip install qiskit qiskit-aer", RuntimeWarning)


class PQCPolicyLayer:
    def __init__(self, n_qubits=8, n_layers=2, quantum_lr=1e-3, obs_dim=14):
        self.D  = n_qubits; self.L = n_layers
        self.lr = quantum_lr; self.p = obs_dim
        rng = np.random.RandomState(42)
        self.theta = rng.uniform(0, np.pi/2, (n_layers, n_qubits))
        bound = np.sqrt(6/(self.p+self.D))
        self.W_enc = rng.uniform(-bound, bound, (self.D, self.p))
        if QISKIT_AVAILABLE:
            self._sim = AerSimulator(method='statevector')
        self._m = np.zeros_like(self.theta)
        self._v = np.zeros_like(self.theta)
        self._t = 0

    def forward(self, s_u):
        x = self._encode(s_u)
        return self._qiskit_fwd(x) if QISKIT_AVAILABLE else self._np_fwd(x)

    def update(self, s_u, delta_J):
        x = self._encode(s_u)
        grad = np.zeros_like(self.theta)
        for l in range(self.L):
            for n in range(self.D):
                orig = self.theta[l, n]
                self.theta[l, n] = orig + np.pi/2
                zp = self._qiskit_fwd(x) if QISKIT_AVAILABLE else self._np_fwd(x)
                self.theta[l, n] = orig - np.pi/2
                zm = self._qiskit_fwd(x) if QISKIT_AVAILABLE else self._np_fwd(x)
                self.theta[l, n] = orig
                grad[l, n] = (zp[n]-zm[n])/2 * delta_J[n]

        self._t += 1; b1,b2,eps = 0.9,0.999,1e-8
        self._m = b1*self._m + (1-b1)*grad
        self._v = b2*self._v + (1-b2)*grad**2
        mh = self._m/(1-b1**self._t); vh = self._v/(1-b2**self._t)
        self.theta += self.lr * mh/(np.sqrt(vh)+eps)
        self.theta  = np.clip(self.theta, 0, np.pi/2)

    def _build_circuit(self, x):
        qc  = QuantumCircuit(self.D)
        nx  = len(x)
        for n in range(self.D):                         
            qc.ry(float(x[(2*n)   % nx]), n)
            qc.rz(float(x[(2*n+1) % nx]), n)
        for l in range(self.L):                         
            for n in range(self.D): qc.ry(float(self.theta[l,n]), n)
            for n in range(self.D-1): qc.cx(n, n+1)    
        return qc

    def _qiskit_fwd(self, x):
        qc = self._build_circuit(x)
        qc.save_statevector()
        sv  = np.array(self._sim.run(qc).result().get_statevector(), dtype=complex)
        return self._pauli_z(sv)

    def _pauli_z(self, sv):
        probs = np.abs(sv)**2; z = np.zeros(self.D)
        for i in range(len(probs)):
            for n in range(self.D):
                z[n] += probs[i] * (1 - 2*((i>>n)&1))
        return z

    def _np_fwd(self, x):
        ang = np.zeros(self.D); nx = len(x)
        for n in range(self.D):
            ang[n] = x[(2*n)%nx] + 0.5*x[(2*n+1)%nx]
        for l in range(self.L):
            ang += self.theta[l]
            for n in range(self.D-1):
                d = 0.15*np.sin(ang[n]-ang[n+1])
                ang[n] -= d; ang[n+1] += d
        return np.clip(np.cos(ang), -1, 1)

    def _encode(self, s_u):
        return np.arctan(self.W_enc @ s_u.astype(float))

    def get_params(self):  return self.theta.flatten().copy()
    def set_params(self, f): self.theta = np.clip(f.reshape(self.L,self.D), 0, np.pi/2)
    def copy(self):
        n = PQCPolicyLayer(self.D, self.L, self.lr, self.p)
        n.theta = self.theta.copy(); n.W_enc = self.W_enc.copy()
        n._m = self._m.copy(); n._v = self._v.copy(); n._t = self._t
        return n
    def diagram(self, s_u):
        if not QISKIT_AVAILABLE: return "[PQC] Install qiskit to draw."
        return str(self._build_circuit(self._encode(s_u)).draw(output='text', fold=-1))