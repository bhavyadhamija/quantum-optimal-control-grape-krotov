"""
control.py — piecewise-constant propagation, GRAPE and Krotov for state-to-state transfer.

Hamiltonian:  H(t) = H0 + u(t) H1,   u(t) constant on each of K time steps of length dt.
Fidelity:     F = |<target|psi(T)>|^2   (global phase does not matter).

GRAPE (Khaneja et al. 2005) uses the exact gradient of F with respect to every step amplitude,
obtained from the eigen-decomposition of each step Hamiltonian, and L-BFGS-B so that the amplitude
can be bounded, |u_k| <= u_max.

Krotov (first-order, sequential update; Reich, Ndong & Koch 2012) changes the field step by step
while the forward state is propagated with the already-updated field:
    du_k = S_k / lambda_a * Im < chi_old(t_k) | H1 | psi_new(t_k) >,
which makes the fidelity increase monotonically from one iteration to the next.
"""
import numpy as np
from scipy.optimize import minimize


class StepSystem:
    """Bundles H0, H1 and the time grid; builds and caches the step propagators for a given control."""

    def __init__(self, H0, H1, T, K):
        self.H0 = np.asarray(H0, dtype=complex)
        self.H1 = np.asarray(H1, dtype=complex)
        self.N = self.H0.shape[0]
        self.T, self.K = T, K
        self.dt = T / K
        self.t_mid = (np.arange(K) + 0.5) * self.dt

    def eig(self, u):
        """Eigen-decomposition of H0 + u_k H1 for all steps at once: returns E (K,N), V (K,N,N)."""
        H = self.H0[None] + np.asarray(u)[:, None, None] * self.H1[None]
        return np.linalg.eigh(H)

    def propagate(self, u, psi0, keep=False):
        """Forward propagation. With keep=True returns all K+1 states."""
        E, V = self.eig(u)
        ph = np.exp(-1j * self.dt * E)
        psi = np.asarray(psi0, dtype=complex)
        out = [psi] if keep else None
        for k in range(self.K):
            psi = V[k] @ (ph[k] * (V[k].conj().T @ psi))
            if keep:
                out.append(psi)
        return np.array(out) if keep else psi

    def fidelity(self, u, psi0, target):
        return abs(np.vdot(target, self.propagate(u, psi0))) ** 2


def grape_value_and_grad(sys, u, psi0, target):
    """F and the exact dF/du_k for piecewise-constant controls."""
    E, V = sys.eig(u)
    dt, K = sys.dt, sys.K
    ph = np.exp(-1j * dt * E)                               # (K,N)
    Vh = np.conj(np.transpose(V, (0, 2, 1)))
    # forward states psi_{k-1} (state before step k) in each step's eigenbasis
    psi = np.asarray(psi0, dtype=complex)
    b = np.empty((K, sys.N), complex)
    for k in range(K):
        b[k] = Vh[k] @ psi
        psi = V[k] @ (ph[k] * b[k])
    o = np.vdot(target, psi)
    F = abs(o) ** 2
    # backward states lambda_k = U_{k+1}^+ ... U_K^+ target
    lam = np.asarray(target, dtype=complex)
    a = np.empty((K, sys.N), complex)
    for k in range(K - 1, -1, -1):
        a[k] = Vh[k] @ lam
        lam = V[k] @ (np.conj(ph[k]) * a[k])
    # dU_k/du = V (M o G) V^+,  M = V^+ H1 V,  G_ij = (e^{-i dt E_i} - e^{-i dt E_j}) / (E_i - E_j)
    M = Vh @ sys.H1[None] @ V
    dE = E[:, :, None] - E[:, None, :]
    dph = ph[:, :, None] - ph[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(np.abs(dE) > 1e-12, dph / np.where(np.abs(dE) > 1e-12, dE, 1.0),
                     -1j * dt * ph[:, :, None])
    g = np.einsum("ki,kij,kj->k", np.conj(a), M * G, b)
    grad = 2 * np.real(np.conj(o) * g)
    return F, grad


def grape(sys, u0, psi0, target, u_max=None, maxiter=2000, ftol=1e-14, gtol=1e-10, history=None):
    """Maximise F with L-BFGS-B. u_max (scalar or per-step array) bounds |u_k|. Returns (u_opt, F_opt, n_iter)."""
    hist = [] if history is None else history

    def fun(x):
        F, g = grape_value_and_grad(sys, x, psi0, target)
        return 1.0 - F, -g

    def cb(xk):
        hist.append(1.0 - sys.fidelity(xk, psi0, target))

    if u_max is None:
        bounds = None
    else:
        ub = np.broadcast_to(np.asarray(u_max, float), (sys.K,))
        bounds = list(zip(-ub, ub))
        u0 = np.clip(u0, -ub, ub)
    r = minimize(fun, np.asarray(u0, float), jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                 options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=ftol, gtol=gtol))
    return r.x, 1.0 - r.fun, r.nit


def krotov(sys, u0, psi0, target, lambda_a, shape=None, iters=200, tol=1e-6, history=None):
    """First-order Krotov with sequential update. shape S_k (default 1) switches the update off at the edges.
    Stops when 1-F < tol or after `iters` iterations. Returns (u, F, n_iter)."""
    u = np.array(u0, float)
    S = np.ones(sys.K) if shape is None else np.asarray(shape)
    hist = [] if history is None else history
    dt, K = sys.dt, sys.K
    psiT = sys.propagate(u, psi0)
    F = abs(np.vdot(target, psiT)) ** 2
    hist.append(1 - F)
    it = 0
    for it in range(1, iters + 1):
        if 1 - F < tol:
            it -= 1
            break
        # backward propagation of chi with the old field; chi(T) = <target|psi(T)> target
        E, V = sys.eig(u)
        ph = np.exp(-1j * dt * E)
        chi = np.vdot(target, psiT) * np.asarray(target, complex)
        chis = np.empty((K, sys.N), complex)
        for k in range(K - 1, -1, -1):
            chi = V[k] @ (np.conj(ph[k]) * (V[k].conj().T @ chi))   # chi at the start of step k
            chis[k] = chi
        # forward sweep with sequential update
        psi = np.asarray(psi0, complex)
        for k in range(K):
            u[k] += S[k] / lambda_a * np.imag(np.vdot(chis[k], sys.H1 @ psi))
            Hk = sys.H0 + u[k] * sys.H1
            e, W = np.linalg.eigh(Hk)
            psi = W @ (np.exp(-1j * dt * e) * (W.conj().T @ psi))
        psiT = psi
        F = abs(np.vdot(target, psiT)) ** 2
        hist.append(1 - F)
    return u, F, it


def check_gradient(sys, u, psi0, target, n=5, h=1e-6, seed=0):
    """Compare the analytic GRAPE gradient with central finite differences at n random steps."""
    _, g = grape_value_and_grad(sys, u, psi0, target)
    rng = np.random.default_rng(seed)
    out = []
    for k in rng.choice(sys.K, size=n, replace=False):
        up, um = u.copy(), u.copy()
        up[k] += h; um[k] -= h
        fd = (sys.fidelity(up, psi0, target) - sys.fidelity(um, psi0, target)) / (2 * h)
        out.append((int(k), g[k], fd))
    return out
