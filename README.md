# Quantum Optimal Control with GRAPE and Krotov: a Qubit and a Vibrating OH Bond

Laser pulses are designed by optimal control to drive a quantum system into a target state, and the best achievable fidelity is mapped against pulse duration (the quantum speed limit). Two systems are studied:

- **Electronic motion:** a two-level system (an electronic transition in the rotating frame) with a bounded control.
- **Nuclear motion:** the OH bond as a Morse oscillator driven by an infrared laser. The targets are v = 0 → 1 and v = 0 → 2.

GRAPE (exact gradients, L-BFGS-B with amplitude bounds) and first-order Krotov (sequential, monotonic update) are written from scratch in NumPy/SciPy.

## Key results

**Two-level system** (time in units of 1/Ω_max, |u| ≤ Ω_max)

| | Minimum time, exact | GRAPE |
|---|---|---|
| Resonant (Δ = 0) | π (one π-rotation) | reproduces F*(T) = sin²(T/2) to 8 digits |
| Detuned (Δ = Ω_max) | √2·π = 4.443 (two π-rotations about tilted axes) | optimal pulse is bang-bang with one switch; F = 1 − 10⁻¹⁴ at T_min |

- **Independent reference:** an optimisation over switching times agrees with GRAPE at every duration, to within 5×10⁻⁵.
- **Krotov:** converges monotonically, reaching 1 − F = 10⁻¹⁰ in 183 iterations.

![qubit](results/qubit_control.png)

**OH vibrational excitation** (ω₀₁ = 3785 cm⁻¹, ω₁₂ = 3605 cm⁻¹, |μ₀₁| = 0.0257 a.u.)

| Field bound E_max | 0.005 a.u. (2.6 GV/m) | 0.01 a.u. (5.1 GV/m) | 0.02 a.u. (10.3 GV/m) | 0.04 a.u. | 0.06 a.u. |
|---|---|---|---|---|---|
| Minimum duration for F ≥ 0.99, full OH | 18,800 a.u. (455 fs) | 10,030 a.u. (243 fs) | 6,110 a.u. (148 fs) | 4,770 a.u. (115 fs) | 4,200 a.u. (102 fs) |
| Same, with only v = 0, 1 kept | 18,440 a.u. | 9,470 a.u. | 4,920 a.u. | 2,660 a.u. | 1,900 a.u. |
| Cost of the vibrational ladder | ×1.02 | ×1.06 | ×1.24 | ×1.79 | ×2.21 |

- **The ladder sets the speed limit.** A pure two-level system gets faster as 1/E_max. Real OH saturates at strong fields, because the higher levels (v = 2, 3) are driven too and the pulse must undo that leakage.
- **The optimal pulses use the ladder.** During the optimal 0 → 1 pulse, up to 27% of the population passes through v = 2. The 0 → 2 pulse climbs through v = 3. The pulse spectra show peaks at ω₀₁ and ω₁₂.
- **Optimal control vs a simple pulse.** At E_max = 0.02 a.u. GRAPE reaches F = 1.0000 at 169 fs (v = 1) and 242 fs (v = 2). A resonant sin² π pulse under the same bound reaches only 0.45 (v = 1) and 0.16 (v = 2) at those durations.
- **Krotov vs GRAPE.** Krotov (no amplitude bound) gives smooth, nearly sinusoidal pulses with peaks of 0.021 and 0.028 a.u. and converges monotonically. GRAPE converges in far fewer iterations and produces near-square-wave carriers that sit on the amplitude bound.

![fidelity](results/molecule_fidelity_vs_T.png)
![speed limit](results/molecule_speed_limit.png)
![pulses](results/molecule_optimal_pulses.png)

## Method

**Hamiltonian and fidelity**
- H(t) = H₀ + u(t)H₁, with the control u(t) constant on each time step. Each step is propagated exactly by diagonalisation.
- The fidelity is F = |⟨target|ψ(T)⟩|².

**GRAPE**
- The gradient dF/du_k is exact for every step, computed from the eigendecomposition of each step Hamiltonian.
- Optimiser: L-BFGS-B with |u_k| ≤ u_max·s(t). The envelope s(t) has 400 a.u. sin² ramps, so the field switches on and off smoothly.
- Several starting guesses are tried. Each duration is warm-started from the optimum at a shorter duration, padded with zero field, so the best fidelity can only grow with T.

**Krotov**
- Update rule: Δu(t) = (S(t)/λ_a)·Im⟨χ(t)|H₁|ψ(t)⟩, applied sequentially along the pulse while the state is propagated with the already-updated field. This guarantees monotonic convergence.

**OH model**
- Morse potential: D₀ = 0.1994 Eh, β = 1.189 a₀⁻¹, r₀ = 1.821 a₀. It has 22 bound states.
- Dipole function: μ(r) = 1.634·r·e^(−r/1.134) a.u.
- Eigenstates are computed with a sinc-DVR; the optimisation uses the lowest 15 states.

A full explanation of the theory, every derivation and every result is in [docs/WRITEUP.md](docs/WRITEUP.md).

## Validation (`validate.py`, all pass)

- Morse levels agree with the exact formula to 3×10⁻¹⁵ Eh.
- GRAPE gradients agree with finite differences to within 3×10⁻⁸ relative, for both systems.
- The step propagator agrees with an independent 8th-order ODE solver (DOP853) to 7×10⁻¹¹.
- A resonant π pulse in the two-level system gives 1 − F = 10⁻¹⁴.
- Krotov is monotonic.
- 15 vs 60 vibrational states (bound plus discretised continuum) agree to 10⁻¹⁴. The optimal pulses leave less than 10⁻⁹ population above v = 14.

## Files

| File | Purpose |
|---|---|
| `control.py` | propagation, GRAPE (exact gradient), Krotov, gradient check |
| `morse.py` | OH Morse oscillator, DVR eigenstates, dipole matrix |
| `qubit_study.py` | two-level fidelity vs T, minimum time, bang-bang reference, Krotov vs GRAPE (about 1 min) |
| `molecule_study.py` | OH fidelity vs T at three field bounds, v = 0 → 2, pulses, spectra, Krotov (about 20 min) |
| `speed_limit.py` | minimum duration vs field strength; `--two-level` for the comparison (several minutes per field) |
| `make_plots.py` | all figures into `results/` |
| `validate.py` | independent checks (seconds) |
| `docs/WRITEUP.md` | detailed write-up: theory, derivations, results, likely viva questions |

```bash
pip install -r requirements.txt
python validate.py
python qubit_study.py
python molecule_study.py
python speed_limit.py 0.005 0.01 0.02 0.04 0.06
python speed_limit.py --two-level 0.005 0.01 0.02 0.04 0.06
python make_plots.py
```

## Limitations

- **Bound states only.** The OH model is one-dimensional and non-rotating, with no continuum beyond the 60-state check. Fields of 0.04–0.06 a.u. (above 5×10¹³ W/cm²) would in reality also cause ionisation and dissociation.
- **Square-wave fields.** GRAPE's bounded optima are near-square-wave fields on a 10 a.u. (0.24 fs) grid. A real pulse shaper would need a bandwidth limit or a smoothness penalty.
- **Zero net area not enforced.** The net pulse area (DC component) is not constrained.

## References

- N. Khaneja et al., *J. Magn. Reson.* 172, 296 (2005): GRAPE.
- D. M. Reich, M. Ndong, C. P. Koch, *J. Chem. Phys.* 136, 104103 (2012): Krotov's method.
- M. H. Goerz et al., *SciPost Phys.* 7, 080 (2019): Krotov package and update formula.
- A. P. Peirce, M. A. Dahleh, H. Rabitz, *Phys. Rev. A* 37, 4950 (1988): optimal control of molecular motion.
- W. Zhu, J. Botina, H. Rabitz, *J. Chem. Phys.* 108, 1953 (1998): OH Morse control model.
- W. Jakubetz, J. Manz, V. Mohan, *J. Chem. Phys.* 90, 3686 (1989): OH dipole function parameters.
- T. Caneva et al., *Phys. Rev. Lett.* 103, 240501 (2009): optimal control at the quantum speed limit.
- G. C. Hegerfeldt, *Phys. Rev. Lett.* 111, 260501 (2013): time-optimal driving of a two-level system.
- U. Boscain, P. Mason, *J. Math. Phys.* 47, 062101 (2006): time-minimal control of a spin-½.
- D. T. Colbert, W. H. Miller, *J. Chem. Phys.* 96, 1982 (1992): sinc-DVR.

## Author

Bhavya Dhamija, BITS Pilani
