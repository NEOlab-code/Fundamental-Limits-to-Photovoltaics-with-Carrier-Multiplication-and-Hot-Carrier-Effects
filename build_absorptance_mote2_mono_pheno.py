"""Build a phenomenological MoTe2 monolayer absorptance spectrum.

Motivation
----------
A fully digitized exciton-resolved broadband absorptance dataset for monolayer MoTe2
is not always available in a single machine-readable format. For GitHub-level
reproducibility, we provide a minimal phenomenological model that captures the
key qualitative features needed in Fig. 4 of the manuscript:
  - A/B exciton resonances near the optical gap
  - A weak higher-energy resonance / continuum shoulder
  - A smooth background + excitonic resonances (absolute amplitude anchored downstream)

Model
-----
Absorptance is modeled as a smooth background step plus Lorentzian exciton peaks:
    A(E) = A_bg(E) + sum_i A_i * L(E; E_i, gamma_i)
where L is a unit-peak Lorentzian.

The parameter set below is intentionally compact and editable.

Run:
  python scripts/build_absorptance_mote2_mono_pheno.py

Outputs:
  data/raw/absorptance_mote2_mono_pheno_shape.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def lorentz(E: np.ndarray, E0: float, gamma: float) -> np.ndarray:
    return (gamma**2) / ((E - E0) ** 2 + gamma**2)


def smooth_step(E: np.ndarray, E0: float, w: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((E - E0) / w))


def main() -> None:
    # Use the same energy grid as the other monolayer datasets for easy overlay
    ref = pd.read_csv(DATA / "absorptance_wse2_mono.csv")
    E = ref["energy_eV"].to_numpy()

    # --- Compact phenomenological parameter set ---
    Eg = 1.10  # eV (monolayer optical gap used in main text)

    # Background: weak rise above the gap (continuum / higher-energy transitions)
    A_bg_sat = 0.030
    A_bg = A_bg_sat * smooth_step(E, Eg + 0.18, 0.35)

    # Excitonic resonances (A/B) + a weak higher-energy shoulder
    peaks = [
        # (center_eV, gamma_eV, amplitude)
        (1.10, 0.030, 0.060),  # A exciton
        (1.30, 0.060, 0.030),  # B exciton
        (1.90, 0.200, 0.020),  # higher-energy shoulder
    ]

    A = A_bg.copy()
    for E0, g, amp in peaks:
        A += amp * lorentz(E, E0, g)
    # Final clip (numerical safety)
    A = np.clip(A, 0.0, 0.12)

    # Hard cutoff below Eg to avoid unphysical Bose-Einstein divergence when μ exceeds E.
    # This enforces absorptance support only for E ≥ Eg in the detailed-balance integrals.
    A[E < Eg] = 0.0

    out = pd.DataFrame({"energy_eV": E, "absorptance": A})
    out_path = DATA / "raw" / "absorptance_mote2_mono_pheno_shape.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (max (unanchored)={A.max():.3f})")


if __name__ == "__main__":
    main()
