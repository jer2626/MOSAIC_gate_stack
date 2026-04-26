"""
MOSAIC Task D: Simulation Design of Experiments for Gate Stack Database
Sweeps the M-O-S design space, evaluates all combinations via C_device_modeling,
screens infeasible designs, and saves the full input-output database.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import qmc

from C_device_modeling import load_materials_db, evaluate_gate_stack_v2, get_il_props

# ── Design space definition ─────────────────────────────────────────────────

GATE_MATERIALS = [
    'TiN', 'TaN', 'W (Tungsten)', 'Mo (Molybdenum)',
    'Ru (Ruthenium)', 'Pt (Platinum)', 'Ni (Nickel)',
    'Poly-Si (n+)', 'Poly-Si (p+)', 'TiAlN', 'TaCN',
]

DIELECTRICS = [
    'SiO\u2082', 'SiON', 'HfO\u2082', 'HfSiON',
    'ZrO\u2082', 'Al\u2082O\u2083', 'La\u2082O\u2083',
    'HfAlO', 'Y\u2082O\u2083',
]

SUBSTRATES = [
    'Si', 'Strained Si', 'Ge', 'SiGe',
    'GaAs', 'InGaAs', 'GaN', 'SiC (4H)', 'InP',
]

# Structural parameter ranges
STRUCT_PARAMS = {
    't_HK_nm': (1.0, 10.0),
    't_IL_nm': (0.0,  1.0),   # enforced 0 for substrates with no IL
    'Nsub':    (1e15, 1e18),   # log-uniform
    'L_nm':    (20.0, 100.0),
    'VDD':     (0.7,  1.8),
    'T_K':     (300.0, 400.0),
}

# Fixed for this DoE (width normalisation)
W_UM     = 1.0
CARRIER  = 'n'   # NMOS

# Number of LHS structural samples per M-O-S combination
N_STRUCT_SAMPLES = 5   # → total ≈ 11×9×9×5 = 4455 rows

# ── Latin Hypercube Sampler ─────────────────────────────────────────────────

def lhs_struct_samples(n: int, substrate: str, seed: int = 42) -> list:
    """
    Generate n LHS samples of structural parameters, respecting IL constraints.
    Returns list of dicts.
    """
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    raw = sampler.random(n)   # shape (n, 6): [t_HK, t_IL, log10_Nsub, L, VDD, T]

    lo = [1.0,  0.0,  15.0,  20.0, 0.7, 300.0]
    hi = [10.0, 1.0,  18.0, 100.0, 1.8, 400.0]
    scaled = qmc.scale(raw, lo, hi)

    il_name, _ = get_il_props(substrate)
    samples = []
    for row in scaled:
        t_HK = float(np.clip(row[0], 1.0, 10.0))
        t_IL = float(np.clip(row[1], 0.0, 1.0)) if il_name is not None else 0.0
        t_IL = round(t_IL, 2)
        Nsub = float(10 ** np.clip(row[2], 15, 18))
        L    = float(np.clip(row[3], 20, 100))
        VDD  = float(np.clip(row[4], 0.7, 1.8))
        T    = float(np.clip(row[5], 300, 400))
        samples.append(dict(t_HK_nm=round(t_HK, 2),
                            t_IL_nm=t_IL,
                            Nsub=round(Nsub, 3),
                            L_nm=round(L, 1),
                            VDD=round(VDD, 3),
                            T_K=round(T, 1)))
    return samples


# ── Additional targeted grid for key structural combinations ────────────────

def targeted_grid() -> list:
    """
    Add a coarse structured grid of important structural parameters so that
    the target regions are well-sampled.
    """
    t_HKs  = [1.5, 2.0, 3.0, 5.0, 7.0]
    t_ILs  = [0.0, 0.3, 0.5, 1.0]
    Nsubs  = [1e15, 1e16, 5e16, 1e17, 5e17]
    Ls     = [20, 30, 50, 100]
    VDDs   = [0.7, 1.0, 1.5]
    T_Ks   = [300.0]
    return [dict(t_HK_nm=t, t_IL_nm=il, Nsub=ns, L_nm=L, VDD=vd, T_K=300)
            for t, il, ns, L, vd in itertools.product(t_HKs, t_ILs, Nsubs, Ls, VDDs)]


# ── Feasibility screening constraints ──────────────────────────────────────

CONSTRAINTS = {
    'EOT_nm':    (0.3, 10.0),
    'VTH_V':     (0.1, 1.2),
    'IG_Aum':    (0,   1e-3),    # max 1 mA/μm (very loose; tightened via TARGET_REQUIREMENTS in E)
    'SS_mVdec':  (60,  200),
    'Eox_MVcm':  (0,   None),    # must be below EBD (checked in evaluate)
    'VBD_V':     (0.3, None),
    'COX_Fm2':   (1e-3, None),
    't_HK_nm':   (0.5, 10.0),
    'CBO_min':   1.0,            # eV – checked pre-simulation
}


def screen_row(row: dict, db: dict) -> bool:
    """Return True if the row passes basic feasibility checks."""
    dp = db['diels'].get(row.get('Dielectric', ''), {})
    if dp.get('CBO', 0) < CONSTRAINTS['CBO_min']:
        return False
    if not row.get('feasible', False):
        return False
    eot = row.get('EOT_nm', 999)
    if not (CONSTRAINTS['EOT_nm'][0] <= eot <= CONSTRAINTS['EOT_nm'][1]):
        return False
    vbd = row.get('VBD_V', 0)
    if vbd < CONSTRAINTS['VBD_V'][0]:
        return False
    return True



# ── Main DoE runner ─────────────────────────────────────────────────────────

def run_doe(db: dict,
            use_targeted_grid: bool = False,
            n_lhs_per_combo:   int  = N_STRUCT_SAMPLES,
            verbose:           bool = True) -> pd.DataFrame:
    """
    Run the full design of experiments.

    Parameters
    ----------
    db               : materials database from load_materials_db()
    use_targeted_grid: also add the coarse structured grid (adds ~3000 rows)
    n_lhs_per_combo  : LHS samples per M-O-S combination
    verbose          : print progress

    Returns
    -------
    DataFrame with all evaluated (and screened) gate stack configurations.
    """
    all_results   = []
    total_combos  = len(GATE_MATERIALS) * len(DIELECTRICS) * len(SUBSTRATES)
    done = 0

    for gate, diel, sub in itertools.product(GATE_MATERIALS, DIELECTRICS, SUBSTRATES):
        # Skip if materials not in database
        if gate not in db['gates'] or diel not in db['diels'] or sub not in db['subs']:
            continue

        # Pre-filter: skip dielectrics with CBO < 1 eV (guaranteed high leakage)
        dp = db['diels'][diel]
        if dp.get('CBO', 0) < 1.0:
            continue

        # Generate structural samples for this M-O-S combination
        structs = lhs_struct_samples(n_lhs_per_combo, sub, seed=done)

        for struct in structs:
            try:
                row = evaluate_gate_stack_v2(
                    gate_name=gate, diel_name=diel, sub_name=sub,
                    t_HK_nm=struct['t_HK_nm'],
                    t_IL_nm=struct['t_IL_nm'],
                    Nsub=struct['Nsub'],
                    L_nm=struct['L_nm'],
                    VDD=struct['VDD'],
                    T=struct['T_K'],
                    carrier=CARRIER,
                    db=db,
                )
                # Store dielectric properties for downstream use
                row['EBD_HK'] = dp.get('EBD', 8.0)
                row['CBO_HK'] = dp.get('CBO', 1.5)
                row['kappa']  = dp.get('kappa', 10.0)

                all_results.append(row)
            except Exception as e:
                if verbose:
                    print(f"  Error [{gate}/{diel}/{sub}]: {e}")

        done += 1
        if verbose and done % 100 == 0:
            print(f"  Progress: {done}/{total_combos} M-O-S combos done, "
                  f"{len(all_results)} rows collected")

    # Optional targeted grid (adds coverage of critical parameter ranges)
    if use_targeted_grid:
        if verbose:
            print("\nAdding targeted structural grid...")
        grid = targeted_grid()
        g_done = 0
        for gate, diel, sub in itertools.product(GATE_MATERIALS[:5],
                                                  DIELECTRICS[:4],
                                                  SUBSTRATES[:3]):
            if gate not in db['gates'] or diel not in db['diels'] or sub not in db['subs']:
                continue
            dp = db['diels'][diel]
            for struct in grid[:12]:   # sample first 12 grid points only
                il_name, _ = get_il_props(sub)
                t_IL = struct['t_IL_nm'] if il_name is not None else 0.0
                try:
                    row = evaluate_gate_stack_v2(
                        gate, diel, sub,
                        struct['t_HK_nm'], t_IL,
                        struct['Nsub'], struct['L_nm'], struct['VDD'],
                        struct['T_K'], CARRIER, db=db,
                    )
                    row['EBD_HK'] = dp.get('EBD', 8.0)
                    row['CBO_HK'] = dp.get('CBO', 1.5)
                    row['kappa']  = dp.get('kappa', 10.0)
                    all_results.append(row)
                except Exception:
                    pass
            g_done += 1

    df = pd.DataFrame(all_results)
    if verbose:
        print(f"\nTotal rows generated:   {len(df)}")
        feas = df[df['feasible'] == True]
        print(f"Feasible rows:          {len(feas)}")
    return df


# ── Save results ─────────────────────────────────────────────────────────────

def save_database(df: pd.DataFrame,
                  csv_path:  str = 'D_gate_stack_database.csv',
                  xlsx_path: str = 'D_gate_stack_database.xlsx'):
    df.to_csv(csv_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Full_Database', index=False)
            feas = df[df['feasible'] == True]
            feas.to_excel(writer, sheet_name='Feasible_Only', index=False)
    except Exception as e:
        print(f"Excel write warning: {e} — CSV still saved.")
    print(f"\nDatabase saved: {csv_path} | {xlsx_path}")


# ── Summary statistics ────────────────────────────────────────────────────────

def print_doe_summary(df: pd.DataFrame):
    feas = df[df['feasible'] == True]
    print("\n" + "="*60)
    print("MOSAIC DoE Summary")
    print("="*60)
    print(f"Total configurations simulated : {len(df)}")
    print(f"Feasible configurations        : {len(feas)}")
    print(f"Infeasible configurations      : {len(df) - len(feas)}")

    metrics = ['EOT_nm', 'VTH_V', 'ION_uAum', 'SS_mVdec', 'IG_Aum',
               'Eox_MVcm', 'VBD_V', 'dT_K', 'Rel_score']
    print(f"\nPerformance ranges (feasible only):")
    for m in metrics:
        if m in feas.columns:
            col = pd.to_numeric(feas[m], errors='coerce').dropna()
            if len(col):
                print(f"  {m:15s}: [{col.min():.3e}, {col.max():.3e}]  "
                      f"median={col.median():.3e}")

    def _safe(s): return str(s).encode('ascii', errors='replace').decode('ascii')
    print(f"\nTop gate materials (feasible):")
    print(_safe(feas['Gate'].value_counts().head(5).to_string()))
    print(f"\nTop dielectrics (feasible):")
    print(_safe(feas['Dielectric'].value_counts().head(5).to_string()))
    print(f"\nTop substrates (feasible):")
    print(_safe(feas['Substrate'].value_counts().head(5).to_string()))
    print("="*60)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("Loading materials database...")
    db = load_materials_db('MOSAIC_Materials_Database_1.xlsx')

    print(f"Running DoE over {len(GATE_MATERIALS)} gates x "
          f"{len(DIELECTRICS)} dielectrics x {len(SUBSTRATES)} substrates "
          f"x {N_STRUCT_SAMPLES} LHS samples...\n")

    df = run_doe(db, use_targeted_grid=False, n_lhs_per_combo=N_STRUCT_SAMPLES, verbose=True)

    # Save first — before any print that might fail
    save_database(df)
    print_doe_summary(df)

    print("\nSample of feasible configurations:")
    feas_sample = df[df['feasible'] == True].head(10)
    show_cols = ['Gate', 'Dielectric', 'Substrate', 't_HK_nm', 't_IL_nm',
                 'EOT_nm', 'VTH_V', 'ION_uAum', 'SS_mVdec', 'IG_Aum',
                 'VBD_V', 'dT_K']
    out = feas_sample[[c for c in show_cols if c in feas_sample.columns]].to_string(index=False)
    print(out.encode('ascii', errors='replace').decode('ascii'))
