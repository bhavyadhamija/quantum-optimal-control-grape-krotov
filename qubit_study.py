"""
qubit_study.py — time-optimal driving of a two-level system (an electronic transition in the rotating frame).

    H(t) = (Delta/2) sigma_z + (u(t)/2) sigma_x,     |u(t)| <= Omega_max = 1,    |0> -> |1>

Time is in units of 1/Omega_max. For each pulse duration T, GRAPE finds the best bounded pulse; the
best fidelity F*(T) is compared with
  * Delta = 0: the exact bound F*(T) = sin^2(T/2) for T < pi (a rotation about x by at most T),
  * Delta = 1: the exact minimum time sqrt(2) pi (two pi rotations about tilted axes), and an independent
    optimisation over bang/off sequences u in {+1, 0, -1} with free switching times (Pontryagin: time-optimal
    controls are made of such arcs; Boscain & Mason 2006).
Krotov is run at one duration for a convergence comparison.
"""
import json
import os
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from control import StepSystem, grape, krotov, check_gradient

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
PSI0, TARGET = np.array([1, 0], complex), np.array([0, 1], complex)
K = 120


def best_grape(Delta, T, starts=6, seed=0):
    sys = StepSystem(Delta / 2 * SZ, SX / 2, T, K)
    rng = np.random.default_rng(seed)
    best = (None, -1.0)
    for s in range(starts):
        u0 = np.full(K, 0.5) if s == 0 else rng.uniform(-1, 1, K)
        u, F, _ = grape(sys, u0, PSI0, TARGET, u_max=1.0, maxiter=500)
        if F > best[1]:
            best = (u, F)
        if best[1] > 1 - 1e-10:
            break
    return best


def _su2(Delta, v, tau):
    """exp(-i tau (Delta sz + v sx)/2) in closed form."""
    w = np.hypot(Delta, v)
    if w == 0:
        return np.eye(2, dtype=complex)
    c, s = np.cos(w * tau / 2), np.sin(w * tau / 2) / w
    return np.array([[c - 1j * s * Delta, -1j * s * v], [-1j * s * v, c + 1j * s * Delta]])


def bangbang_best(Delta, T, max_arcs=4, restarts=12, seed=1):
    """Best fidelity over piecewise-constant controls with values in {+1, 0, -1}, free arc lengths."""
    rng = np.random.default_rng(seed)
    best = 0.0
    for n in range(1, max_arcs + 1):
        for _ in range(restarts):
            vals = rng.choice([1, 0, -1], size=n)

            def negF(x):
                w = np.exp(x - x.max()); w = w / w.sum() * T
                psi = PSI0
                for v, tau in zip(vals, w):
                    psi = _su2(Delta, v, tau) @ psi
                return -abs(psi[1]) ** 2

            r = minimize(negF, rng.normal(size=n), method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-15, maxiter=3000))
            best = max(best, -r.fun)
            if best > 1 - 1e-12:
                return 1.0
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    # gradient check
    sys = StepSystem(0.5 * SZ, SX / 2, 3.0, K)
    gc = check_gradient(sys, np.random.default_rng(3).uniform(-1, 1, K), PSI0, TARGET)
    res["gradient_check"] = [dict(step=k, analytic=a, finite_difference=f) for k, a, f in gc]
    print("gradient check:", gc)

    Ts = np.round(np.arange(0.5, 5.01, 0.25), 3)
    for Delta in (0.0, 1.0):
        rows = []
        for T in Ts:
            u, F = best_grape(Delta, T)
            ref = np.sin(T / 2) ** 2 if (Delta == 0 and T < np.pi) else (1.0 if Delta == 0 else bangbang_best(Delta, T))
            rows.append(dict(T=float(T), F_grape=F, F_reference=ref))
            print(f"Delta={Delta} T={T:5.2f}  F_grape={F:.8f}  F_ref={ref:.8f}")
        res[f"Delta_{Delta:g}"] = rows
    # minimum time: smallest T with 1 - F <= 1e-6, by bisection, for GRAPE and for the reference
    def bisect(fid, lo, hi, eps=1e-6, tol=1e-4):
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if fid(mid) >= 1 - eps else (mid, hi)
        return 0.5 * (lo + hi)

    for Delta in (0.0, 1.0):
        ref = (lambda T: np.sin(T / 2) ** 2 if T < np.pi else 1.0) if Delta == 0 else (lambda T: bangbang_best(1.0, T))
        tg = bisect(lambda T: best_grape(Delta, T)[1], 2.0, 6.0)
        tr = bisect(ref, 2.0, 6.0)
        # exact time for F = 1: pi (one pi rotation about x) for Delta = 0; sqrt(2) pi for Delta = Omega_max
        # (two pi rotations about the tilted axes (-1,0,1)/sqrt2 then (1,0,1)/sqrt2, each at speed sqrt2)
        res[f"Delta_{Delta:g}_Tmin"] = dict(grape=tg, reference=tr, threshold=1e-6,
                                            exact_F1=(np.pi if Delta == 0 else np.sqrt(2) * np.pi))
        print(f"Delta={Delta}: T_min (1-F<=1e-6) GRAPE {tg:.4f}  reference {tr:.4f}")

    # optimal pulses just above the speed limit, for plotting
    pulses = {}
    for Delta in (0.0, 1.0):
        Tm = res[f"Delta_{Delta:g}_Tmin"]["reference"]
        T = float(Tm + 0.002)
        u, F = best_grape(Delta, T)
        sys = StepSystem(Delta / 2 * SZ, SX / 2, T, K)
        states = sys.propagate(u, PSI0, keep=True)
        bloch = np.array([[2 * np.real(np.conj(s[0]) * s[1]), 2 * np.imag(np.conj(s[0]) * s[1]),
                           abs(s[0]) ** 2 - abs(s[1]) ** 2] for s in states])
        pulses[f"{Delta:g}"] = dict(T=T, F=F, t=sys.t_mid.tolist(), u=u.tolist(), bloch=bloch.tolist())
    res["pulses"] = pulses

    # Krotov vs GRAPE convergence, Delta = 1, T = 4 (no amplitude bound for either)
    T = 4.0
    sys = StepSystem(0.5 * SZ, SX / 2, T, K)
    u0 = 0.3 * np.sin(np.pi * sys.t_mid / T) ** 2
    hk, hg = [], []
    shape = np.sin(np.pi * sys.t_mid / T) ** 2
    uk, Fk, nk = krotov(sys, u0, PSI0, TARGET, lambda_a=1.0, shape=shape, iters=300, tol=1e-10, history=hk)
    hg.append(1 - sys.fidelity(u0, PSI0, TARGET))
    ug, Fg, ng = grape(sys, u0, PSI0, TARGET, maxiter=300, history=hg)
    res["convergence"] = dict(T=T, krotov=hk, grape=hg, F_krotov=Fk, F_grape=Fg,
                              u_krotov=uk.tolist(), u_grape=ug.tolist(), t=sys.t_mid.tolist(),
                              krotov_monotonic=bool(np.all(np.diff(hk) <= 1e-15)))
    print(f"convergence: Krotov F={Fk:.10f} in {nk} it (monotonic {res['convergence']['krotov_monotonic']}); "
          f"GRAPE F={Fg:.10f} in {ng} it")
    json.dump(res, open(os.path.join(OUT, "qubit_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
