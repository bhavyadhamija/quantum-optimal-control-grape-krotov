"""Figures from results/qubit_results.json and results/molecule_results.json."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NAVY, PURPLE, BLUE, TEAL, GREY, RED, ORANGE = "#1E2A4A", "#5B4BB7", "#2F6FDE", "#1A9E8F", "#6B7280", "#B03A2E", "#D9822B"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
FS = 0.02418884326          # fs per atomic unit of time


def qubit():
    q = json.load(open(os.path.join(OUT, "qubit_results.json")))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    a = ax[0]
    Tf = np.linspace(0.3, 5.2, 400)
    a.plot(Tf, np.where(Tf < np.pi, np.sin(Tf / 2) ** 2, 1), color=NAVY, lw=1.5, label="exact bound, Δ = 0")
    for key, c, lab in (("Delta_0", PURPLE, "GRAPE, Δ = 0"), ("Delta_1", ORANGE, "GRAPE, Δ = Ω$_{max}$")):
        r = q[key]
        a.plot([x["T"] for x in r], [x["F_grape"] for x in r], "o", color=c, ms=6, mfc="none", mew=1.6, label=lab)
    r = q["Delta_1"]
    a.plot([x["T"] for x in r], [x["F_reference"] for x in r], "-", color=ORANGE, lw=1, alpha=0.7,
           label="bang-bang optimum, Δ = Ω$_{max}$")
    for key, c in (("Delta_0_Tmin", PURPLE), ("Delta_1_Tmin", ORANGE)):
        a.axvline(q[key]["grape"], color=c, ls=":", lw=1)
    a.set_xlabel("pulse duration T  (units of 1/Ω$_{max}$)"); a.set_ylabel(r"best fidelity  $|\langle 1|\psi(T)\rangle|^2$")
    a.set_title("Two-level system: fidelity vs duration"); a.legend(fontsize=8.5, loc="lower right", frameon=False)

    a = ax[1]
    for key, c, lab in (("0", PURPLE, "Δ = 0"), ("1", ORANGE, "Δ = Ω$_{max}$")):
        p = q["pulses"][key]
        t = np.array(p["t"]); u = np.array(p["u"])
        a.step(t, u, where="mid", color=c, lw=1.8, label=f"{lab}, T = {p['T']:.2f}")
    a.axhline(1, color=GREY, ls="--", lw=0.8); a.axhline(-1, color=GREY, ls="--", lw=0.8)
    a.set_ylim(-1.25, 1.25); a.set_xlabel("time  (1/Ω$_{max}$)"); a.set_ylabel("control u(t) / Ω$_{max}$")
    a.set_title("Optimal pulses just above T$_{min}$ (bang-bang)"); a.legend(fontsize=9, frameon=False, loc="lower left")

    a = ax[2]
    c = q["convergence"]
    a.semilogy(np.arange(len(c["krotov"])), np.maximum(c["krotov"], 1e-16), color=TEAL, lw=1.8, label="Krotov (λ$_a$ = 1)")
    a.semilogy(np.arange(len(c["grape"])), np.maximum(c["grape"], 1e-16), "o-", color=PURPLE, ms=4, label="GRAPE (L-BFGS)")
    a.set_xlabel("iteration"); a.set_ylabel("1 − F"); a.set_title(f"Convergence, Δ = Ω$_{{max}}$, T = {c['T']:g}")
    a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "qubit_control.png"), dpi=200)

    # Bloch-sphere projections of the time-optimal trajectories
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
    for key, c, lab in (("0", PURPLE, "Δ = 0"), ("1", ORANGE, "Δ = Ω$_{max}$")):
        b = np.array(q["pulses"][key]["bloch"])
        ax[0].plot(b[:, 1], b[:, 2], color=c, lw=2, label=lab)
        ax[1].plot(b[:, 0], b[:, 2], color=c, lw=2, label=lab)
    for a, xl in zip(ax, (r"$\langle\sigma_y\rangle$", r"$\langle\sigma_x\rangle$")):
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot(np.cos(th), np.sin(th), color=GREY, lw=0.8)
        a.plot([0], [1], "o", color=NAVY); a.plot([0], [-1], "*", color=RED, ms=12)
        a.set_aspect("equal"); a.set_xlabel(xl); a.set_ylabel(r"$\langle\sigma_z\rangle$"); a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15)
    ax[0].legend(frameon=False, loc="center"); ax[0].set_title("Bloch vector, y–z projection")
    ax[1].set_title("Bloch vector, x–z projection")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "qubit_bloch.png"), dpi=200)


def molecule():
    m = json.load(open(os.path.join(OUT, "molecule_results.json")))
    sc = m["scans"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    a = ax[0]
    for key, c, lab in (("v1_E0.01", BLUE, "0.01"), ("v1_E0.02", PURPLE, "0.02"), ("v1_E0.04", TEAL, "0.04")):
        r = sc[key]["rows"]
        T = np.array([x["T"] for x in r]) * FS
        a.plot(T, [x["F_grape"] for x in r], "o-", color=c, lw=1.6, ms=5, label=f"GRAPE, E$_{{max}}$ = {lab} a.u.")
        a.plot(T, [x["F_pi_pulse"] for x in r], "--", color=c, lw=1, alpha=0.8)
        a.axvline(sc[key]["T_RWA"] * FS, color=c, ls=":", lw=1)
    a.plot([], [], "--", color=GREY, lw=1, label="resonant sin² π pulse, same bound")
    a.plot([], [], ":", color=GREY, lw=1, label="two-level RWA estimate π/(|μ$_{01}$|E$_{max}$)")
    a.set_xlabel("pulse duration T (fs)"); a.set_ylabel("population of v = 1 at T")
    a.set_title("OH, v = 0 → 1: fidelity vs pulse duration"); a.legend(fontsize=8.5, frameon=False, loc="lower right")
    a.set_ylim(-0.02, 1.03)

    a = ax[1]
    r = sc["v2_E0.02"]["rows"]
    T = np.array([x["T"] for x in r]) * FS
    a.plot(T, [x["F_grape"] for x in r], "o-", color=PURPLE, lw=1.6, ms=5, label="GRAPE, E$_{max}$ = 0.02 a.u.")
    a.plot(T, [x["F_pi_pulse"] for x in r], "--", color=PURPLE, lw=1, label="two sequential sin² π pulses (0→1, 1→2)")
    r1 = sc["v1_E0.02"]["rows"]
    a.plot(np.array([x["T"] for x in r1]) * FS, [x["F_grape"] for x in r1], "o-", color=GREY, lw=1, ms=3, alpha=0.6,
           label="v = 0 → 1 for comparison")
    a.set_xlabel("pulse duration T (fs)"); a.set_ylabel("population of target level at T")
    a.set_title("OH, v = 0 → 2 (ladder climbing)"); a.legend(fontsize=8.5, frameon=False, loc="lower right")
    a.set_ylim(-0.02, 1.03)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "molecule_fidelity_vs_T.png"), dpi=200)

    # optimal pulses, spectra and populations
    keys = [k for k in ("v1", "v2") if k in m["pulses"]]
    fig, ax = plt.subplots(3, len(keys), figsize=(6.5 * len(keys), 9.5), squeeze=False)
    cols = [NAVY, PURPLE, ORANGE, TEAL, GREY]
    for j, k in enumerate(keys):
        p = m["pulses"][k]
        t = np.array(p["t"]) * p["T"] / len(p["t"]) * FS
        a = ax[0, j]
        a.plot(t, np.array(p["E"]) * 514.22, color=PURPLE, lw=0.6)
        a.set_ylabel("E(t)  (10$^9$ V/m)"); a.set_xlabel("time (fs)")
        a.set_title(f"Optimal pulse, v = 0 → {k[1]}, T = {p['T'] * FS:.0f} fs, F = {p['F']:.5f}")
        a = ax[1, j]
        w = np.array(p["w"]) * 219474.63
        a.plot(w, p["spec"], color=PURPLE, lw=1.2)
        w01 = m["model"]["w01"] * 219474.63; w12 = m["model"]["w12"] * 219474.63
        a.axvline(w01, color=RED, ls=":", lw=1); a.axvline(w12, color=ORANGE, ls=":", lw=1)
        a.text(w01, 1.02, " ω$_{01}$", color=RED, fontsize=9); a.text(w12, 0.9, "ω$_{12}$ ", color=ORANGE, fontsize=9, ha="right")
        a.set_xlim(2500, 5000); a.set_xlabel("frequency (cm$^{-1}$)"); a.set_ylabel("|spectrum| (normalised)")
        a = ax[2, j]
        pt = np.array(p["pop_t"]) * FS; pop = np.array(p["pop"])
        for v in range(4):
            a.plot(pt, pop[:, v], color=cols[v], lw=1.6, label=f"v = {v}")
        a.set_xlabel("time (fs)"); a.set_ylabel("population")
        a.legend(frameon=False, fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "molecule_optimal_pulses.png"), dpi=200)

    # Krotov vs GRAPE
    kk = [k for k in ("v1", "v2") if k in m["krotov"]]
    fig, ax = plt.subplots(2, len(kk), figsize=(6.5 * len(kk), 7.5), squeeze=False)
    for j, k in enumerate(kk):
        r = m["krotov"][k]
        a = ax[0, j]
        a.semilogy(np.maximum(r["krotov_history"], 1e-16), color=TEAL, lw=1.8, label="Krotov (no amplitude bound)")
        a.semilogy(np.maximum(r["grape_history"], 1e-16), color=PURPLE, lw=1.8, label="GRAPE (|E| ≤ 0.02 a.u.)")
        a.set_xlabel("iteration"); a.set_ylabel("1 − F"); a.legend(frameon=False, fontsize=9)
        a.set_title(f"v = 0 → {k[1]}, T = {r['T'] * FS:.0f} fs")
        a = ax[1, j]
        t = np.array(r["t"]) * FS
        a.plot(t, np.array(r["E_grape"]) * 514.22, color=PURPLE, lw=0.5, alpha=0.8, label="GRAPE")
        a.plot(t, np.array(r["E_krotov"]) * 514.22, color=TEAL, lw=0.5, alpha=0.9, label="Krotov")
        a.set_xlabel("time (fs)"); a.set_ylabel("E(t)  (10$^9$ V/m)"); a.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "molecule_krotov_vs_grape.png"), dpi=200)


def speed_limit():
    import glob
    rows = {}
    for tag in ("", "_2level"):
        d = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"speed_limit{tag}_E*.json"))]
        rows[tag] = sorted(d, key=lambda x: x["Emax"])
    if not rows[""]:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    a = ax[0]
    E = np.array([r["Emax"] for r in rows[""]]); Tm = np.array([r["T_min"] for r in rows[""]])
    a.loglog(E, Tm * FS, "o-", color=PURPLE, lw=1.8, ms=7, label="OH, 15 vibrational levels (GRAPE)")
    if rows["_2level"]:
        E2 = np.array([r["Emax"] for r in rows["_2level"]]); T2 = np.array([r["T_min"] for r in rows["_2level"]])
        a.loglog(E2, T2 * FS, "s--", color=GREY, lw=1.4, ms=6, label="same, only v = 0, 1 kept (GRAPE)")
    Ef = np.geomspace(E.min() * 0.8, E.max() * 1.25, 50)
    a.loglog(Ef, np.array([rows[""][0]["T_RWA"] * rows[""][0]["Emax"]]) / Ef * FS, ":", color=RED, lw=1.3,
             label="two-level RWA estimate π/(|μ$_{01}$|E$_{max}$)")
    a.set_xlabel("peak field bound E$_{max}$ (a.u.)"); a.set_ylabel("minimum duration for F ≥ 0.99 (fs)")
    a.set_title("OH v = 0 → 1: quantum speed limit vs field strength"); a.legend(frameon=False, fontsize=8.5)
    sec = a.secondary_xaxis("top", functions=(lambda x: x * 514.22, lambda x: x / 514.22))
    sec.set_xlabel("E$_{max}$ (10$^9$ V/m)", fontsize=9)
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis, ticks in ((a.xaxis, [0.005, 0.01, 0.02, 0.04, 0.06]), (a.yaxis, [50, 100, 200, 300, 500]),
                        (sec.xaxis, [2.5, 5, 10, 20, 30])):
        axis.set_major_locator(FixedLocator(ticks)); axis.set_minor_locator(NullLocator()); axis.set_major_formatter(plain)
    a = ax[1]
    if rows["_2level"]:
        common = sorted(set(E) & set(E2))
        ratio = [Tm[list(E).index(e)] / T2[list(E2).index(e)] for e in common]
        a.semilogx(common, ratio, "o-", color=PURPLE, lw=1.8, ms=7)
        a.axhline(1, color=GREY, lw=0.8, ls="--")
        for e, r in zip(common, ratio):
            a.annotate(f"{r:.2f}", (e, r), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        a.set_xlabel("peak field bound E$_{max}$ (a.u.)"); a.set_ylabel("T$_{min}$(full OH) / T$_{min}$(two levels)")
        a.set_title("Cost of the vibrational ladder"); a.set_ylim(0.8, max(ratio) * 1.15)
        a.xaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.04, 0.06])); a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_major_formatter(plain)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "molecule_speed_limit.png"), dpi=200)


if __name__ == "__main__":
    qubit()
    if os.path.exists(os.path.join(OUT, "molecule_results.json")):
        molecule()
    speed_limit()
    print("plots written")
