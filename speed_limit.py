"""
speed_limit.py — minimum pulse duration for OH v = 0 -> 1 as a function of the peak-field bound.

For each E_max the duration is increased step by step, each GRAPE run warm-started from the previous
optimum padded with zero field (so the best fidelity can only grow with T), until F >= 0.99; the crossing
is then refined by bisection. The result is compared with the two-level rotating-wave estimate
T_RWA = pi / (|mu_01| E_max): if OH were a two-level system, T_min * E_max would be constant.

Usage:  python speed_limit.py [--two-level] [E_max ...]       (default: 0.005 0.01 0.02 0.04 0.06)
With --two-level only v = 0 and v = 1 are kept: the same calculation for a molecule without a ladder,
which isolates the effect of the higher vibrational levels.
"""
import json
import os
import sys
import time
import numpy as np

import molecule_study as ms
from molecule_study import best_grape, MU, OUT

THRESH = 0.99


def t_min(Emax, step=0.12, rel_tol=0.01):
    T_rwa = np.pi / (abs(MU[0, 1]) * Emax)
    T = max(1000.0, round(0.5 * T_rwa / 10) * 10)
    evals, u, F = [], None, 0.0
    while True:
        u, F, _ = best_grape(T, 1, Emax, starts=2, warm=u)
        evals.append(dict(T=T, F=F)); print(f"  Emax={Emax} T={T:.0f} F={F:.5f}", flush=True)
        if F >= THRESH:
            break
        lo_T, lo_u = T, u
        T = round((T + step * T_rwa) / 10) * 10
    lo, hi = lo_T, T
    while (hi - lo) / hi > rel_tol:
        mid = round(0.5 * (lo + hi) / 10) * 10
        um, Fm, _ = best_grape(float(mid), 1, Emax, starts=2, warm=lo_u)
        evals.append(dict(T=float(mid), F=Fm)); print(f"  Emax={Emax} T={mid:.0f} F={Fm:.5f}", flush=True)
        if Fm >= THRESH:
            hi = mid
        else:
            lo, lo_u = mid, um
    return hi, T_rwa, evals


def main():
    args = sys.argv[1:]
    tag = ""
    if "--two-level" in args:
        args.remove("--two-level")
        ms.H0, ms.H1, ms.NL, ms.PSI0 = ms.H0[:2, :2], ms.H1[:2, :2], 2, ms.PSI0[:2]
        tag = "_2level"
    fields = [float(x) for x in args] or [0.005, 0.01, 0.02, 0.04, 0.06]
    os.makedirs(OUT, exist_ok=True)
    for Emax in fields:
        t0 = time.time()
        T, T_rwa, ev = t_min(Emax)
        out = dict(Emax=Emax, threshold=THRESH, T_min=T, T_RWA=T_rwa, evaluations=ev, runtime_s=time.time() - t0)
        json.dump(out, open(os.path.join(OUT, f"speed_limit{tag}_E{Emax}.json"), "w"), indent=1)
        print(f"{tag} Emax={Emax}: T_min={T:.0f} a.u.  T_RWA={T_rwa:.0f}  ratio={T / T_rwa:.3f}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
