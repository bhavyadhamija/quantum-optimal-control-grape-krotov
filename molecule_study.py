"""
molecule_study.py — infrared-laser control of OH vibration (nuclear motion), atomic units throughout.

    H(t) = H0 - mu E(t)    in the lowest 15 vibrational states of the Morse OH model (morse.py)
    targets: v = 0 -> 1 (fundamental) and v = 0 -> 2 (two-quantum ladder climb)
    constraint: |E(t)| <= E_max s(t), s(t) = flat top with 400 a.u. sin^2 ramps, so the pulse switches on and off smoothly

For each duration T, GRAPE (from several starting guesses) gives the best achievable fidelity F*(T).
Compared with:
  * a conventional resonant sin^2 pi pulse of the same duration (amplitude capped at E_max),
  * the two-level rotating-wave estimate of the minimum time, T_RWA = pi / (|mu_01| E_max).
Krotov is run without an amplitude bound at one duration per target, and every optimised pulse is
re-propagated in a 60-state basis (bound states plus discretised continuum) to check truncation.
"""
import json
import os
import time
import numpy as np

import morse
from control import StepSystem, grape, krotov, check_gradient

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NL = 15                 # vibrational levels kept
DT = 10.0               # control time step (a.u.); each step is propagated exactly
RAMP = 400.0            # sin^2 switch-on and switch-off time of the field envelope (a.u., about 10 fs)
H0, H1, EN, MU = morse.model(NL)
W01, W12 = EN[1] - EN[0], EN[2] - EN[1]
PSI0 = np.eye(NL)[0]


def shape(t, T):
    ramp = RAMP
    s = np.ones_like(t)
    a, b = t < ramp, t > T - ramp
    s[a] = np.sin(np.pi * t[a] / (2 * ramp)) ** 2
    s[b] = np.sin(np.pi * (T - t[b]) / (2 * ramp)) ** 2
    return s


def guess(t, T, target, Emax, phase=0.0):
    s = shape(t, T)
    if target == 1:
        return 0.5 * Emax * s * np.cos(W01 * t + phase)
    return 0.5 * Emax * s * (np.cos(W01 * t + phase) + np.cos(W12 * t + 2 * phase)) / 2


def best_grape(T, target, Emax, starts=3, maxiter=1500, seed=0, warm=None):
    """Best of several GRAPE runs. `warm` = optimum from a shorter duration; it is padded with zero field
    (which leaves the populations unchanged), so the result can never be worse than at the shorter T."""
    K = int(round(T / DT))
    sys = StepSystem(H0, H1, T, K)
    tgt = np.eye(NL)[target]
    rng = np.random.default_rng(seed)
    best = (None, -1.0, 0)
    inits = [] if warm is None else [np.concatenate([warm, np.zeros(K - len(warm))])[:K]]
    inits += [guess(sys.t_mid, T, target, Emax, phase=0 if s == 0 else rng.uniform(0, 2 * np.pi))
              for s in range(starts)]
    for u0 in inits:
        u, F, n = grape(sys, u0, PSI0, tgt, u_max=Emax * shape(sys.t_mid, T), maxiter=maxiter)
        if F > best[1]:
            best = (u, F, n)
        if best[1] > 1 - 1e-6:
            break
    return best[0], best[1], sys


def pi_pulse_baseline(T, target, Emax):
    """Resonant sin^2 pi pulse(s): for v=2, a pi pulse on 0-1 followed by one on 1-2, each T/2 long."""
    K = int(round(T / DT))
    sys = StepSystem(H0, H1, T, K)
    t = sys.t_mid
    if target == 1:
        E0 = min(Emax, 2 * np.pi / (abs(MU[0, 1]) * T))
        u = E0 * np.sin(np.pi * t / T) ** 2 * np.cos(W01 * t)
    else:
        h = T / 2
        E1 = min(Emax, 2 * np.pi / (abs(MU[0, 1]) * h))
        E2 = min(Emax, 2 * np.pi / (abs(MU[1, 2]) * h))
        u = np.where(t < h, E1 * np.sin(np.pi * t / h) ** 2 * np.cos(W01 * t),
                     E2 * np.sin(np.pi * (t - h) / h) ** 2 * np.cos(W12 * t))
    return sys.fidelity(u, PSI0, np.eye(NL)[target])


def big_basis_fidelity(u, T, target, n=60):
    H0b, H1b, _, _ = morse.model(n, r_min=0.8, r_max=14.0, n=1000)
    sys = StepSystem(H0b, H1b, T, len(u))
    psi = sys.propagate(u, np.eye(n)[0])
    return abs(psi[target]) ** 2, float(np.sum(np.abs(psi[NL:]) ** 2))


def populations(u, T):
    sys = StepSystem(H0, H1, T, len(u))
    states = sys.propagate(u, PSI0, keep=True)
    return np.arange(len(u) + 1) * sys.dt, np.abs(states) ** 2


def spectrum(u):
    n = 16 * len(u)
    amp = np.abs(np.fft.rfft(u, n=n))
    w = 2 * np.pi * np.fft.rfftfreq(n, d=DT)
    return w, amp / amp.max()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = dict(model=dict(levels=NL, n_bound=morse.n_bound(), mass=morse.MASS, dt=DT,
                          E=EN.tolist(), mu01=MU[0, 1], mu12=MU[1, 2], mu02=MU[0, 2],
                          w01=W01, w12=W12, period01=2 * np.pi / W01,
                          levels_vs_analytic=float(np.abs(EN - morse.analytic_levels(NL)).max())))
    sys = StepSystem(H0, H1, 3000.0, 300)
    gc = check_gradient(sys, 0.01 * np.random.default_rng(0).normal(size=300), PSI0, np.eye(NL)[1], h=1e-7)
    res["gradient_check"] = [dict(step=k, analytic=a, finite_difference=f) for k, a, f in gc]

    scans = {(1, 0.04): np.arange(1000, 6001, 500), (1, 0.02): np.arange(1000, 10001, 1000),
             (1, 0.01): np.arange(2000, 20001, 2000), (2, 0.02): np.arange(2000, 16001, 1000)}
    res["scans"] = {}
    best_pulses = {}
    for (target, Emax), Ts in scans.items():
        rows = []
        u = None
        for T in Ts:
            u, F, sys = best_grape(float(T), target, Emax, warm=u)
            Fb = pi_pulse_baseline(float(T), target, Emax)
            rows.append(dict(T=float(T), F_grape=F, F_pi_pulse=Fb))
            print(f"v=0->{target} Emax={Emax} T={T:6.0f}  F_grape={F:.6f}  F_pi={Fb:.4f}   [{time.time()-t0:.0f}s]", flush=True)
            if Emax == 0.02 and F > 0.9999 and (target, Emax) not in best_pulses:
                best_pulses[(target, Emax)] = (float(T), u, F)
        res["scans"][f"v{target}_E{Emax}"] = dict(target=target, Emax=Emax, T_RWA=np.pi / (abs(MU[0, target if target == 1 else 1]) * Emax), rows=rows)

    # analysis of the shortest high-fidelity pulse for each target at E_max = 0.02
    res["pulses"] = {}
    for (target, Emax), (T, u, F) in best_pulses.items():
        tp, pop = populations(u, T)
        w, amp = spectrum(u)
        Fbig, leak = big_basis_fidelity(u, T, target)
        res["pulses"][f"v{target}"] = dict(T=T, Emax=Emax, F=F, F_60_states=Fbig, population_above_v14=leak,
                                         t=(np.arange(len(u)) + 0.5).tolist(), E=u.tolist(),
                                         pop_t=tp[::5].tolist(), pop=pop[::5, :5].tolist(),
                                         w=w[w < 0.06].tolist(), spec=amp[w < 0.06].tolist(),
                                         pulse_area=float(np.sum(u) * DT), fluence=float(np.sum(u ** 2) * DT))
        print(f"pulse v{target}: T={T} F={F:.8f}  60-state F={Fbig:.8f}  leak={leak:.1e}")

    # Krotov (no amplitude bound) vs GRAPE (bound 0.02) at one duration per target
    res["krotov"] = {}
    for target, T, lam in ((1, 8000.0, 2.0), (2, 12000.0, 2.0)):
        K = int(round(T / DT))
        sys = StepSystem(H0, H1, T, K)
        tgt = np.eye(NL)[target]
        s = shape(sys.t_mid, T)
        u0 = guess(sys.t_mid, T, target, 0.01)
        hk, hg = [], [1 - sys.fidelity(u0, PSI0, tgt)]
        uk, Fk, nk = krotov(sys, u0, PSI0, tgt, lambda_a=lam, shape=s, iters=400, tol=1e-6, history=hk)
        ug, Fg, ng = grape(sys, u0, PSI0, tgt, u_max=0.02 * s, maxiter=1500, history=hg)
        Fbig, leak = big_basis_fidelity(uk, T, target)
        res["krotov"][f"v{target}"] = dict(T=T, lambda_a=lam, F_krotov=Fk, iters_krotov=nk, F_grape=Fg, iters_grape=ng,
                                           krotov_history=hk, grape_history=hg,
                                           krotov_monotonic=bool(np.all(np.diff(hk) <= 1e-14)),
                                           peak_krotov=float(np.abs(uk).max()), peak_grape=float(np.abs(ug).max()),
                                           F_krotov_60_states=Fbig, t=sys.t_mid.tolist(),
                                           E_krotov=uk.tolist(), E_grape=ug.tolist())
        print(f"Krotov v{target} T={T}: F={Fk:.7f} in {nk} it, peak {np.abs(uk).max():.4f}, monotonic "
              f"{res['krotov'][f'v{target}']['krotov_monotonic']};  GRAPE F={Fg:.7f} in {ng} it")
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "molecule_results.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
