""Detailed-balance efficiency models used in this work_written by Seungwoo Lee_Feb 2026

This module implements the four device limits used throughout the manuscript:

1) SQ (Shockley--Queisser) detailed balance
2) CM (Carrier multiplication) detailed balance with the radiative penalty
3) HC (Hot-carrier) endoreversible detailed balance with cooling coefficient kappa_C
4) CM--HC (Combined carrier multiplication + hot-carrier) endoreversible detailed balance

The implementation is intentionally self-contained and uses only numpy/scipy.
It can be used either with optically thick absorption (step-function) or with an
arbitrary absorptance spectrum a(E) supplied as an array.

Spectrum note
-------------
For fully offline reproducibility, we approximate 1-sun illumination with a
scaled 5778 K blackbody whose total incident power equals 1000 W/m^2.
If you want to use the ASTM G-173 AM1.5G spectrum instead, replace
`phi_sun(E_J)` accordingly.

Units
-----
- Energies are in eV in the public API unless otherwise noted.
- Internally we convert to Joules for Planck distributions.
- Currents are in A/m^2, powers in W/m^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

# --- Fundamental constants (SI) ---
q = 1.602176634e-19  # C
h = 6.62607015e-34  # J s
c = 299792458.0  # m/s
kB = 1.380649e-23  # J/K

# Sun geometry (used only for absolute scaling; we renormalize to 1000 W/m^2)
R_sun = 6.9634e8  # m
AU = 1.495978707e11  # m
f_sun = (R_sun / AU) ** 2
Omega_sun = np.pi * f_sun

# Lambertian emission into a hemisphere
Omega_emit = np.pi


def planck_photon_radiance_E(E_J: np.ndarray, T: float, mu_J: np.ndarray | float = 0.0) -> np.ndarray:
    """Photon spectral radiance per unit energy.

    Parameters
    ----------
    E_J:
        Photon energy grid (J).
    T:
        Temperature (K).
    mu_J:
        Photon chemical potential (J). Must satisfy mu_J < E_J elementwise.

    Returns
    -------
    radiance:
        Photon radiance per (J * m^2 * s * sr). (Absolute units are consistent
        up to the usual Planck-law prefactors.)
    """
    x = (E_J - mu_J) / (kB * T)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denom = np.expm1(x)
        out = (2.0 / (h ** 3 * c ** 2)) * (E_J ** 2) / denom
    # Numerical safety: for x<=0, denom<=0 -> unphysical divergence.
    # In our detailed-balance integrals the absorptance is zero below the band edge,
    # so returning 0 (instead of inf) prevents 0*inf -> NaN without affecting results.
    out = np.where(x > 1e-12, out, 0.0)
    return out


def phi_sun_blackbody(E_J: np.ndarray, Ts: float = 5778.0) -> np.ndarray:
    """Unscaled blackbody solar photon flux (per J per m^2 per s)."""
    return Omega_sun * planck_photon_radiance_E(E_J, Ts, 0.0)


def phi_cell(E_J: np.ndarray, T: float, mu_J: np.ndarray | float) -> np.ndarray:
    """Lambertian photon flux emitted by the cell (per J per m^2 per s)."""
    return Omega_emit * planck_photon_radiance_E(E_J, T, mu_J)


@dataclass(frozen=True)
class Spectrum:
    """Convenience wrapper for the offline 1-sun spectrum used in the code."""

    E_eV: np.ndarray
    Ts: float = 5778.0

    def __post_init__(self):
        if self.E_eV.ndim != 1 or np.any(self.E_eV <= 0):
            raise ValueError("E_eV must be a 1D array of positive energies.")

    @property
    def E_J(self) -> np.ndarray:
        return self.E_eV * q

    @property
    def phi_unscaled(self) -> np.ndarray:
        return phi_sun_blackbody(self.E_J, self.Ts)

    @property
    def P_in_unscaled(self) -> float:
        return float(np.trapz(self.E_J * self.phi_unscaled, self.E_J))

    @property
    def scale_to_1sun(self) -> float:
        # Scale AM0 solar constant down to 1000 W/m^2.
        return 1000.0 / self.P_in_unscaled

    @property
    def phi(self) -> np.ndarray:
        return self.scale_to_1sun * self.phi_unscaled

    @property
    def P_in(self) -> float:
        return 1000.0


def m_eta_piecewise(E_eV: np.ndarray, Eg_eV: float, eta_cm: float) -> np.ndarray:
    """Piecewise-linear carrier-multiplication yield m(E).

    This implements Eq. (S7) of the SI, i.e. a linear ramp within each
    interval [n Eg, (n+1) Eg). For eta_cm=1, this reduces to the ideal
    integer yield m(E)=floor(E/Eg).

    Notes
    -----
    - E_eV can be an array.
    - eta_cm should be in [0, 1].
    """
    x = E_eV / Eg_eV
    m = np.ones_like(x)
    mask = x >= 2.0
    n = np.floor(x[mask]).astype(int)
    m[mask] = (n - 1) + eta_cm * (x[mask] - n)
    return m


def absorptance_step(E_eV: np.ndarray, Eg_eV: float) -> np.ndarray:
    """Optically thick step-function absorptance."""
    return (E_eV >= Eg_eV).astype(float)


def absorptance_yablonovitch(alpha_m: np.ndarray, n: float, d_m: float) -> np.ndarray:
    """Yablonovitch 4n^2 light-trapping absorptance model.

    A(E, d) = (alpha * 4 n^2 d) / (1 + alpha * 4 n^2 d)

    Parameters
    ----------
    alpha_m:
        Absorption coefficient alpha(E) (1/m).
    n:
        Refractive index (assumed energy-independent here).
    d_m:
        Thickness (m).
    """
    L = 4.0 * (n ** 2) * d_m
    x = alpha_m * L
    return x / (1.0 + x)


# --- Detailed-balance solvers ---

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def sq_current_density(
    spectrum: Spectrum,
    a_E: np.ndarray,
    V: float,
    Tc: float = 300.0,
) -> float:
    """SQ net current density J(V) for arbitrary absorptance a(E)."""
    mu = q * V
    integrand = a_E * (spectrum.phi - phi_cell(spectrum.E_J, Tc, mu))
    return q * _trapz(integrand, spectrum.E_J)


def cm_current_density(
    spectrum: Spectrum,
    a_E: np.ndarray,
    Eg_eV: float,
    eta_cm: float,
    V: float,
    Tc: float = 300.0,
) -> float:
    """CM net current density J(V) with radiative penalty."""
    m = m_eta_piecewise(spectrum.E_eV, Eg_eV, eta_cm)
    mu = q * V * m
    integrand = a_E * (m * spectrum.phi - phi_cell(spectrum.E_J, Tc, mu))
    return q * _trapz(integrand, spectrum.E_J)


def maximize_power_over_V(
    J_of_V: Callable[[float], float],
    V_max: float,
    n_V: int = 600,
) -> Tuple[float, float, float]:
    """Grid-search maximum of P(V)=J(V)V.

    Returns
    -------
    V_opt, J_opt, P_opt
    """
    Vs = np.linspace(0.0, V_max, n_V)
    Js = np.array([J_of_V(float(V)) for V in Vs])
    Ps = Js * Vs
    i = int(np.nanargmax(Ps))
    return float(Vs[i]), float(Js[i]), float(Ps[i])


def sq_max_efficiency(
    spectrum: Spectrum,
    Eg_eV: float,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
) -> Dict[str, float]:
    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)
    V_opt, J_opt, P_opt = maximize_power_over_V(
        lambda V: sq_current_density(spectrum, a_E, V, Tc=Tc),
        V_max=Eg_eV,
    )
    return {
        "eta": P_opt / spectrum.P_in,
        "V": V_opt,
        "J": J_opt,
        "P": P_opt,
    }


def cm_max_efficiency(
    spectrum: Spectrum,
    Eg_eV: float,
    eta_cm: float,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
) -> Dict[str, float]:
    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)
    V_opt, J_opt, P_opt = maximize_power_over_V(
        lambda V: cm_current_density(spectrum, a_E, Eg_eV, eta_cm, V, Tc=Tc),
        V_max=Eg_eV,
    )
    return {
        "eta": P_opt / spectrum.P_in,
        "V": V_opt,
        "J": J_opt,
        "P": P_opt,
    }


def hc_power_density(
    spectrum: Spectrum,
    a_E: np.ndarray,
    mu_eh_eV: float,
    Th: float,
    kappa_C: float,
    Tc: float = 300.0,
) -> Tuple[float, float, float, float]:
    """Return (P, J, V, DeltaE_eV) for HC at (mu_eh, Th)."""
    mu = mu_eh_eV * q
    # Net pair flux
    J = q * _trapz(a_E * (spectrum.phi - phi_cell(spectrum.E_J, Th, mu)), spectrum.E_J)
    if not np.isfinite(J) or J <= 0:
        return -np.inf, J, np.nan, np.nan

    # Net energy flux into carriers (radiative balance minus cooling)
    Q_rad = _trapz(a_E * (spectrum.E_J * (spectrum.phi - phi_cell(spectrum.E_J, Th, mu))), spectrum.E_J)
    Q_net = Q_rad - kappa_C * (Th - Tc)

    # Average extracted energy per pair
    DeltaE_J = q * Q_net / J
    DeltaE_eV = DeltaE_J / q

    # De Vos voltage relation
    qV_J = (1.0 - Tc / Th) * DeltaE_J + (Tc / Th) * mu
    V = qV_J / q

    if not np.isfinite(V) or V < 0:
        return -np.inf, J, V, DeltaE_eV

    P = J * V
    return P, J, V, DeltaE_eV


def cmhc_power_density(
    spectrum: Spectrum,
    a_E: np.ndarray,
    Eg_eV: float,
    eta_cm: float,
    mu_eh_eV: float,
    Th: float,
    kappa_C: float,
    Tc: float = 300.0,
) -> Tuple[float, float, float, float]:
    """Return (P, J, V, DeltaE_eV) for CM--HC at (mu_eh, Th).

    CM enters (i) the extracted pair flux via m(E) multiplying absorption and
    (ii) the emission term through an energy-dependent photon chemical potential
    mu_gamma(E)=m(E) * mu_eh.

    The absorbed energy flux is still set by photons (no m factor).
    """
    m = m_eta_piecewise(spectrum.E_eV, Eg_eV, eta_cm)
    mu_gamma = (m * mu_eh_eV) * q

    # Net pair flux
    J = q * _trapz(a_E * (m * spectrum.phi - phi_cell(spectrum.E_J, Th, mu_gamma)), spectrum.E_J)
    if not np.isfinite(J) or J <= 0:
        return -np.inf, J, np.nan, np.nan

    # Net energy flux (no m on absorption energy)
    Q_rad = _trapz(a_E * (spectrum.E_J * (spectrum.phi - phi_cell(spectrum.E_J, Th, mu_gamma))), spectrum.E_J)
    Q_net = Q_rad - kappa_C * (Th - Tc)

    DeltaE_J = q * Q_net / J
    DeltaE_eV = DeltaE_J / q

    mu = mu_eh_eV * q
    qV_J = (1.0 - Tc / Th) * DeltaE_J + (Tc / Th) * mu
    V = qV_J / q

    if not np.isfinite(V) or V < 0:
        return -np.inf, J, V, DeltaE_eV

    P = J * V
    return P, J, V, DeltaE_eV


def maximize_hc(
    power_fn: Callable[[float, float], Tuple[float, float, float, float]],
    Eg_eV: float,
    Th_range: Tuple[float, float] = (301.0, 2200.0),
    n_Th: int = 260,
    mu_range: Tuple[float, float] | None = None,
    n_mu: int = 220,
) -> Dict[str, float]:
    """Coarse but robust grid search for (mu_eh, Th) maximizing power."""
    if mu_range is None:
        mu_range = (0.0, Eg_eV)

    Ths = np.linspace(Th_range[0], Th_range[1], n_Th)
    mus = np.linspace(mu_range[0], mu_range[1], n_mu)

    best = {
        "P": -np.inf,
        "eta": -np.inf,
        "J": np.nan,
        "V": np.nan,
        "Th": np.nan,
        "mu": np.nan,
        "DeltaE": np.nan,
    }

    for Th in Ths:
        for mu in mus:
            P, J, V, DeltaE = power_fn(float(mu), float(Th))
            if P > best["P"]:
                best.update({"P": float(P), "J": float(J), "V": float(V), "Th": float(Th), "mu": float(mu), "DeltaE": float(DeltaE)})

    return best


def hc_max_efficiency(
    spectrum: Spectrum,
    Eg_eV: float,
    kappa_C: float,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
    Th_range: Tuple[float, float] = (301.0, 2200.0),
    n_Th: int = 160,
    mu_range: Tuple[float, float] | None = None,
    n_mu: int = 140,
) -> Dict[str, float]:
    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)

    best = maximize_hc(
        lambda mu, Th: hc_power_density(spectrum, a_E, mu, Th, kappa_C, Tc=Tc),
        Eg_eV=Eg_eV,
        Th_range=Th_range,
        n_Th=n_Th,
        mu_range=mu_range,
        n_mu=n_mu,
    )
    best["eta"] = best["P"] / spectrum.P_in
    return best


def cmhc_max_efficiency(
    spectrum: Spectrum,
    Eg_eV: float,
    eta_cm: float,
    kappa_C: float,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
    Th_range: Tuple[float, float] = (301.0, 2200.0),
    n_Th: int = 160,
    mu_range: Tuple[float, float] | None = None,
    n_mu: int = 140,
) -> Dict[str, float]:
    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)

    best = maximize_hc(
        lambda mu, Th: cmhc_power_density(spectrum, a_E, Eg_eV, eta_cm, mu, Th, kappa_C, Tc=Tc),
        Eg_eV=Eg_eV,
        Th_range=Th_range,
        n_Th=n_Th,
        mu_range=mu_range,
        n_mu=n_mu,
    )
    best["eta"] = best["P"] / spectrum.P_in
    return best

# -----------------------------------------------------------------------------
# Fast helpers for κ=0 figure generation
# -----------------------------------------------------------------------------


def hc_max_efficiency_mu0(
    spectrum: Spectrum,
    Eg_eV: float,
    kappa_C: float = 0.0,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
    Th_grid: np.ndarray | None = None,
) -> Dict[str, float]:
    """Fast HC maximum-efficiency helper for κ=0 with μ fixed to 0.

    Rationale
    ---------
    For the reversible limit (κ=0) considered in many of the main-text plots,
    the power-optimal photon chemical potential in this endoreversible hot-carrier
    (HC) model tends to μ→0.  Fixing μ=0 reduces the optimization to a 1D sweep
    over Th, making figure generation orders of magnitude faster than the full
    2D (Th, μ) grid search used by hc_max_efficiency().

    Use hc_max_efficiency() for κ>0.
    """

    if abs(kappa_C) > 1e-15:
        raise ValueError("hc_max_efficiency_mu0 is intended for kappa_C=0 only.")

    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)

    if Th_grid is None:
        # Moderate resolution for plots (can be increased if needed)
        Th_grid = np.linspace(Tc, 6000.0, 180)

    best = {"P": -np.inf, "J": 0.0, "V": 0.0, "Th": float(Tc), "mu": 0.0, "DeltaE": 0.0}

    for Th in Th_grid:
        P, J, V, DeltaE = hc_power_density(spectrum, a_E, 0.0, float(Th), kappa_C, Tc=Tc)
        if P > best["P"]:
            best.update({
                "P": float(P),
                "J": float(J),
                "V": float(V),
                "Th": float(Th),
                "mu": 0.0,
                "DeltaE": float(DeltaE),
            })

    best["eta"] = best["P"] / spectrum.P_in
    return best


def cmhc_max_efficiency_mu0(
    spectrum: Spectrum,
    Eg_eV: float,
    eta_cm: float,
    kappa_C: float = 0.0,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
    Th_grid: np.ndarray | None = None,
) -> Dict[str, float]:
    """Fast CM–HC maximum-efficiency helper for κ=0 with μ fixed to 0."""

    if abs(kappa_C) > 1e-15:
        raise ValueError("cmhc_max_efficiency_mu0 is intended for kappa_C=0 only.")

    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)

    if Th_grid is None:
        Th_grid = np.linspace(Tc, 6000.0, 180)

    best = {"P": -np.inf, "J": 0.0, "V": 0.0, "Th": float(Tc), "mu": 0.0, "DeltaE": 0.0}

    for Th in Th_grid:
        P, J, V, DeltaE = cmhc_power_density(spectrum, a_E, Eg_eV, eta_cm, 0.0, float(Th), kappa_C, Tc=Tc)
        if P > best["P"]:
            best.update({
                "P": float(P),
                "J": float(J),
                "V": float(V),
                "Th": float(Th),
                "mu": 0.0,
                "DeltaE": float(DeltaE),
            })

    best["eta"] = best["P"] / spectrum.P_in
    return best

# =============================================================================
# Vectorized HC / CM--HC grid maximizers (speed path for kappa>0 sweeps)
# =============================================================================


def hc_power_density_muvec(
    spectrum: Spectrum,
    a_E: np.ndarray,
    mu_eh_eV_vec: np.ndarray,
    Th: float,
    kappa_C: float,
    Tc: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HC power density for a vector of mu values at fixed Th."""

    mu_J = mu_eh_eV_vec * q  # (n_mu,)

    # Cell emission for each mu: (n_mu, n_E)
    phi_c = phi_cell(spectrum.E_J, Th, mu_J[:, None])

    # Net pair flux
    J = q * np.trapz(a_E[None, :] * (spectrum.phi[None, :] - phi_c), spectrum.E_J, axis=1)

    # Net energy flux into carriers (radiative balance minus cooling)
    Q_rad = np.trapz(a_E[None, :] * (spectrum.E_J[None, :] * (spectrum.phi[None, :] - phi_c)), spectrum.E_J, axis=1)
    Q_net = Q_rad - kappa_C * (Th - Tc)

    # Average extracted energy per pair
    DeltaE_J = q * Q_net / J
    DeltaE_eV = DeltaE_J / q

    # De Vos voltage relation
    qV_J = (1.0 - Tc / Th) * DeltaE_J + (Tc / Th) * mu_J
    V = qV_J / q

    P = J * V

    # Physical masks
    bad = (~np.isfinite(P)) | (~np.isfinite(J)) | (~np.isfinite(V)) | (J <= 0) | (V < 0)
    P = np.where(bad, -np.inf, P)

    return P, J, V, DeltaE_eV


def cmhc_power_density_muvec(
    spectrum: Spectrum,
    a_E: np.ndarray,
    Eg_eV: float,
    eta_cm: float,
    mu_eh_eV_vec: np.ndarray,
    Th: float,
    kappa_C: float,
    Tc: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized CM--HC power density for a vector of mu values at fixed Th."""

    m = m_eta_piecewise(spectrum.E_eV, Eg_eV, eta_cm)  # (n_E,)
    mu_gamma_J = (mu_eh_eV_vec[:, None] * m[None, :]) * q  # (n_mu, n_E)

    phi_c = phi_cell(spectrum.E_J, Th, mu_gamma_J)

    # Net pair flux
    J = q * np.trapz(a_E[None, :] * (m[None, :] * spectrum.phi[None, :] - phi_c), spectrum.E_J, axis=1)

    # Net energy flux (no m factor on absorbed photon energy)
    Q_rad = np.trapz(a_E[None, :] * (spectrum.E_J[None, :] * (spectrum.phi[None, :] - phi_c)), spectrum.E_J, axis=1)
    Q_net = Q_rad - kappa_C * (Th - Tc)

    DeltaE_J = q * Q_net / J
    DeltaE_eV = DeltaE_J / q

    mu_J = mu_eh_eV_vec * q
    qV_J = (1.0 - Tc / Th) * DeltaE_J + (Tc / Th) * mu_J
    V = qV_J / q

    P = J * V

    bad = (~np.isfinite(P)) | (~np.isfinite(J)) | (~np.isfinite(V)) | (J <= 0) | (V < 0)
    P = np.where(bad, -np.inf, P)

    return P, J, V, DeltaE_eV


def hc_max_efficiency_vectorized(
    spectrum: Spectrum,
    Eg_eV: float,
    kappa_C: float,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
    Th_range: Tuple[float, float] = (301.0, 2200.0),
    n_Th: int = 160,
    mu_range: Tuple[float, float] | None = None,
    n_mu: int = 160,
) -> Dict[str, float]:
    """Vectorized grid maximization for HC (faster than nested Python loops)."""

    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)

    if mu_range is None:
        mu_range = (0.0, Eg_eV)

    Ths = np.linspace(Th_range[0], Th_range[1], n_Th)
    mus = np.linspace(mu_range[0], mu_range[1], n_mu)

    best = {"P": -np.inf, "eta": -np.inf, "J": np.nan, "V": np.nan, "Th": np.nan, "mu": np.nan, "DeltaE": np.nan}

    for Th in Ths:
        P, J, V, DeltaE = hc_power_density_muvec(spectrum, a_E, mus, float(Th), kappa_C, Tc=Tc)
        idx = int(np.argmax(P))
        if P[idx] > best["P"]:
            best.update({"P": float(P[idx]), "J": float(J[idx]), "V": float(V[idx]), "Th": float(Th), "mu": float(mus[idx]), "DeltaE": float(DeltaE[idx])})

    best["eta"] = best["P"] / spectrum.P_in
    return best


def cmhc_max_efficiency_vectorized(
    spectrum: Spectrum,
    Eg_eV: float,
    eta_cm: float,
    kappa_C: float,
    a_E: np.ndarray | None = None,
    Tc: float = 300.0,
    Th_range: Tuple[float, float] = (301.0, 2200.0),
    n_Th: int = 160,
    mu_range: Tuple[float, float] | None = None,
    n_mu: int = 160,
) -> Dict[str, float]:
    """Vectorized grid maximization for CM--HC."""

    if a_E is None:
        a_E = absorptance_step(spectrum.E_eV, Eg_eV)

    if mu_range is None:
        mu_range = (0.0, Eg_eV)

    Ths = np.linspace(Th_range[0], Th_range[1], n_Th)
    mus = np.linspace(mu_range[0], mu_range[1], n_mu)

    best = {"P": -np.inf, "eta": -np.inf, "J": np.nan, "V": np.nan, "Th": np.nan, "mu": np.nan, "DeltaE": np.nan}

    for Th in Ths:
        P, J, V, DeltaE = cmhc_power_density_muvec(spectrum, a_E, Eg_eV, eta_cm, mus, float(Th), kappa_C, Tc=Tc)
        idx = int(np.argmax(P))
        if P[idx] > best["P"]:
            best.update({"P": float(P[idx]), "J": float(J[idx]), "V": float(V[idx]), "Th": float(Th), "mu": float(mus[idx]), "DeltaE": float(DeltaE[idx])})

    best["eta"] = best["P"] / spectrum.P_in
    return best
