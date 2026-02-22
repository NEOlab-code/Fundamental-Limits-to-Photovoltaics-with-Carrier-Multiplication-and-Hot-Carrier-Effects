"""Calibrate monolayer absorptance spectra to literature anchor points.

Motivation
----------
Earlier drafts normalized monolayer absorptance curves using an ad-hoc global cap
(e.g., setting the spectrum maximum to 0.10). For publication-quality transparency,
we instead anchor the absolute amplitude by matching the absorptance at a specific
energy (typically the A-exciton peak) to a value reported in the literature.

Inputs
------
- data/raw/absorptance_*_mono_*shape.csv : shape-only spectra (arbitrary scaling)
- data/monolayer_abs_anchor.csv         : anchor energies and target absorptance

Outputs
-------
- data/absorptance_wse2_mono.csv
- data/absorptance_mos2_mono.csv
- data/absorptance_mote2_mono.csv

All outputs share the same CSV schema:
    energy_eV, absorptance
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RAW  = os.path.join(DATA, 'raw')

ANCHOR_CSV = os.path.join(DATA, 'monolayer_abs_anchor.csv')

SHAPE_MAP = {
    'WSe2_mono': os.path.join(RAW, 'absorptance_wse2_mono_shape.csv'),
    'MoS2_mono': os.path.join(RAW, 'absorptance_mos2_mono_shape.csv'),
    'MoTe2_mono': os.path.join(RAW, 'absorptance_mote2_mono_pheno_shape.csv'),
}

OUT_MAP = {
    'WSe2_mono': os.path.join(DATA, 'absorptance_wse2_mono.csv'),
    'MoS2_mono': os.path.join(DATA, 'absorptance_mos2_mono.csv'),
    'MoTe2_mono': os.path.join(DATA, 'absorptance_mote2_mono.csv'),
}


def _load_shape(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept a few legacy schemas
    if 'energy_eV' not in df.columns:
        for c in df.columns:
            if 'energy' in c.lower() and 'ev' in c.lower():
                df = df.rename(columns={c: 'energy_eV'})
                break
    if 'absorptance' not in df.columns:
        for c in df.columns:
            if 'abs' in c.lower():
                df = df.rename(columns={c: 'absorptance'})
                break
    assert 'energy_eV' in df.columns and 'absorptance' in df.columns, f'Unexpected columns in {path}: {df.columns}'
    df = df[['energy_eV', 'absorptance']].copy()
    df = df.sort_values('energy_eV')
    return df


def _scale_to_anchor(df: pd.DataFrame, E_anchor: float, A_target: float) -> tuple[pd.DataFrame, float, float]:
    E = df['energy_eV'].to_numpy(dtype=float)
    a = df['absorptance'].to_numpy(dtype=float)

    # Interpolate current absorptance at the anchor energy
    A_here = float(np.interp(E_anchor, E, a))
    if A_here <= 0:
        raise ValueError(f'Anchor absorptance is non-positive at E={E_anchor} eV (A={A_here}).')

    scale = A_target / A_here

    a_scaled = a * scale
    # Physical bounds
    a_scaled = np.clip(a_scaled, 0.0, 1.0)

    out = pd.DataFrame({'energy_eV': E, 'absorptance': a_scaled})
    return out, A_here, scale



def main() -> None:
    anchors = pd.read_csv(ANCHOR_CSV)
    print('Loaded anchors:')
    print(anchors.to_string(index=False))

    for _, row in anchors.iterrows():
        mat = row['material']
        E_anchor = float(row['E_anchor_eV'])
        A_target = float(row['A_anchor'])

        if mat not in SHAPE_MAP:
            raise KeyError(f'No shape file mapping for material={mat}')
        in_path = SHAPE_MAP[mat]
        out_path = OUT_MAP[mat]

        df_shape = _load_shape(in_path)
        df_out, A_here, scale = _scale_to_anchor(df_shape, E_anchor, A_target)
        df_out.to_csv(out_path, index=False)

        print(f'[{mat}] {os.path.basename(in_path)} -> {os.path.basename(out_path)}')
        print(f'    anchor: E={E_anchor:.3f} eV, A_here={A_here:.5f} -> A_target={A_target:.5f}')
        print(f'    scale factor: {scale:.5f}\n')


if __name__ == '__main__':
    main()
