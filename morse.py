"""
morse.py — the OH bond as a Morse oscillator with a dipole function, in atomic units.

    V(r)  = D0 [exp(-beta (r - r0)) - 1]^2 - D0        D0 = 0.1994 Eh, beta = 1.189 /a0, r0 = 1.821 a0
    mu(r) = mu0 r exp(-r / r*)                          mu0 = 1.634 e, r* = 1.134 a0 (0.6 Angstrom)
    m     = reduced mass of 16O-1H

Vibrational eigenstates come from a sinc-DVR (Colbert & Miller 1992) on a uniform grid; the control
problem is then written in the basis of the lowest n_levels eigenstates:
    H0 = diag(E_v),   H1 = -mu_{vw}   (field E(t) couples through -mu E(t)).
"""
import numpy as np

AMU = 1822.888486
D0, BETA, R0 = 0.1994, 1.189, 1.821
MU0, RSTAR = 1.634, 1.134
MASS = 15.9949146 * 1.00782503 / (15.9949146 + 1.00782503) * AMU

AU_TIME_FS = 0.02418884326
AU_FIELD_V_PER_M = 5.14220674763e11
AU_ENERGY_CM = 219474.6313


def potential(r):
    return D0 * (np.exp(-BETA * (r - R0)) - 1.0) ** 2 - D0


def dipole(r):
    return MU0 * r * np.exp(-r / RSTAR)


def dvr(r_min=0.8, r_max=9.0, n=600):
    """Sinc-DVR eigenstates. Returns grid r, energies E (all), eigenvectors C (columns, normalised on grid)."""
    r = np.linspace(r_min, r_max, n)
    dx = r[1] - r[0]
    i = np.arange(n)
    d = i[:, None] - i[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(d == 0, np.pi ** 2 / 3.0, 2.0 / np.where(d == 0, 1, d) ** 2) * (-1.0) ** d
    T = T / (2 * MASS * dx ** 2)
    E, C = np.linalg.eigh(T + np.diag(potential(r)))
    return r, E, C


def analytic_levels(nmax):
    """Exact Morse energies E_v = w (v+1/2) - [w (v+1/2)]^2 / (4 D0) - D0."""
    w = BETA * np.sqrt(2 * D0 / MASS)
    v = np.arange(nmax) + 0.5
    return w * v - (w * v) ** 2 / (4 * D0) - D0


def n_bound():
    lam = np.sqrt(2 * MASS * D0) / BETA
    return int(np.floor(lam - 0.5)) + 1


def model(n_levels=15, **grid):
    """H0 (relative to the ground state), H1 = -mu, energies, dipole matrix in the lowest n_levels states."""
    r, E, C = dvr(**grid)
    C = C[:, :n_levels]
    C = C * np.sign(C[np.argmax(np.abs(C), axis=0), np.arange(n_levels)])   # fix eigenvector signs
    mu = C.T @ (dipole(r)[:, None] * C)
    En = E[:n_levels]
    H0 = np.diag(En - En[0])
    return H0, -mu, En, mu
