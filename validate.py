"""
validate.py — independent checks of every building block (runs in under a minute).

1. Morse levels from the DVR vs the exact Morse formula; number of bound states.
2. GRAPE gradient vs central finite differences (two-level system and OH).
3. Propagator vs an independent ODE integrator (scipy solve_ivp, RK 8th order) for a smooth OH pulse.
4. Two-level sanity check: a resonant square pulse of area pi transfers |0> -> |1> exactly.
5. Krotov: fidelity never decreases from one iteration to the next.
6. Basis truncation: the same OH pulse propagated with 15 and with 60 vibrational states.
"""
import numpy as np
from scipy.integrate import solve_ivp

import morse
from control import StepSystem, krotov, check_gradient

SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)


def main():
    ok = True
    # 1
    H0, H1, En, mu = morse.model(15)
    dE = np.abs(En - morse.analytic_levels(15)).max()
    print(f"[1] Morse levels v=0..14: max |DVR - exact| = {dE:.1e} Eh; bound states = {morse.n_bound()}")
    ok &= dE < 1e-10 and morse.n_bound() == 22

    # 2
    q = StepSystem(0.5 * SZ, SX / 2, 3.0, 120)
    gq = check_gradient(q, np.random.default_rng(3).uniform(-1, 1, 120), np.array([1, 0], complex), np.array([0, 1], complex))
    m = StepSystem(H0, H1, 3000.0, 300)
    gm = check_gradient(m, 0.01 * np.random.default_rng(0).normal(size=300), np.eye(15)[0], np.eye(15)[1], h=1e-7)
    for name, g in (("two-level", gq), ("OH", gm)):
        rel = max(abs(a - f) / abs(f) for _, a, f in g)
        print(f"[2] GRAPE gradient, {name}: max relative deviation from finite differences = {rel:.1e}")
        ok &= rel < 1e-5

    # 3
    T, K = 2000.0, 400
    sysm = StepSystem(H0, H1, T, K)
    t = sysm.t_mid
    u = 0.02 * np.sin(np.pi * t / T) ** 2 * np.cos(En[1] * t - En[0] * t)
    psi_step = sysm.propagate(u, np.eye(15, dtype=complex)[0])
    dt = T / K
    rhs = lambda tt, y: -1j * (H0 + u[min(int(tt // dt), K - 1)] * H1) @ y
    psi_ode = np.eye(15, dtype=complex)[0]
    for k in range(K):    # integrate each constant-field step with an independent ODE solver
        s = solve_ivp(rhs, (k * dt, (k + 1) * dt), psi_ode, method="DOP853", rtol=1e-12, atol=1e-13,
                      t_eval=[(k + 1) * dt])
        psi_ode = s.y[:, -1]
    d3 = np.abs(psi_step - psi_ode).max()
    print(f"[3] step propagator vs DOP853 ODE solver: max |difference| in final state = {d3:.1e}")
    ok &= d3 < 1e-8

    # 4
    qs = StepSystem(0 * SZ, SX / 2, np.pi, 50)
    F4 = qs.fidelity(np.ones(50), np.array([1, 0], complex), np.array([0, 1], complex))
    print(f"[4] resonant pi pulse in a two-level system: 1 - F = {1 - F4:.1e}")
    ok &= 1 - F4 < 1e-12

    # 5
    sysk = StepSystem(H0, H1, 4000.0, 400)
    hist = []
    s = np.sin(np.pi * sysk.t_mid / 4000.0) ** 2
    krotov(sysk, 0.005 * s * np.cos((En[1] - En[0]) * sysk.t_mid), np.eye(15)[0], np.eye(15)[1],
           lambda_a=2.0, shape=s, iters=40, history=hist)
    mono = np.all(np.diff(hist) <= 1e-14)
    print(f"[5] Krotov, 40 iterations on OH: 1-F from {hist[0]:.3f} to {hist[-1]:.2e}, monotonic = {mono}")
    ok &= bool(mono)

    # 6
    H0b, H1b, _, _ = morse.model(60, r_min=0.8, r_max=14.0, n=1000)
    big = StepSystem(H0b, H1b, T, K)
    psi_big = big.propagate(u, np.eye(60, dtype=complex)[0])
    d6 = abs(abs(psi_big[1]) ** 2 - abs(psi_step[1]) ** 2)
    print(f"[6] 15 vs 60 states, population of v=1 differs by {d6:.1e}; population above v=14: "
          f"{np.sum(np.abs(psi_big[15:]) ** 2):.1e}")
    ok &= d6 < 1e-6
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
