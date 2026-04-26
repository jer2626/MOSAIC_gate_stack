"""
MOSAIC Gate Stack Recommendation Dashboard
Streamlit web interface — run with:  streamlit run F_recommendation_interface.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Pareto helper (inline to avoid sklearn import chain) ──────────────────────
def pareto_front_nd(values: np.ndarray) -> np.ndarray:
    n = len(values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = (
            np.all(values <= values[i], axis=1) &
            np.any(values <  values[i], axis=1)
        )
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False
    return is_pareto


# ── IL name lookup (inline to avoid importing C_device_modeling) ──────────────
_SI_COMPAT  = {'Si', 'Strained Si', 'SiGe'}
_GE_COMPAT  = {'Ge'}
_GAN_COMPAT = {'GaN'}

def _il_name(substrate: str) -> str:
    if substrate in _SI_COMPAT:  return 'SiOx'
    if substrate in _GE_COMPAT:  return 'Al2O3'
    if substrate in _GAN_COMPAT: return 'Ga2O3'
    return 'None'


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOSAIC Gate Stack Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(_DIR, 'E_surrogate_model.pkl')
DOE_CSV   = os.path.join(_DIR, 'D_gate_stack_database.csv')
MATS_XLSX = os.path.join(_DIR, 'MOSAIC_Materials_Database_1.xlsx')

N_ML_SAMPLES = 8   # LHS samples per material combo in ML mode


# ── Dielectric T_cryst lookup (from materials database Excel) ─────────────────
@st.cache_data(show_spinner=False)
def _load_t_cryst_map() -> dict:
    """Returns {dielectric_name: T_cryst_degC}. Falls back to empty dict if Excel missing."""
    if not os.path.exists(MATS_XLSX):
        return {}
    try:
        df = pd.read_excel(MATS_XLSX, sheet_name='2_Dielectrics', header=1)
        result = {}
        for _, row in df.iterrows():
            mat = str(row.iloc[0]).strip()
            if not mat or mat.lower() in ('nan', 'material', ''):
                continue
            try:
                raw = str(row.iloc[9]).replace('>', '').replace('~', '').strip()
                result[mat] = float(raw)
            except (ValueError, IndexError):
                pass
        return result
    except Exception:
        return {}


# ── Load database at startup (pandas only — safe) ─────────────────────────────
@st.cache_data(show_spinner="Loading gate stack database…")
def _load_database():
    if not os.path.exists(DOE_CSV):
        return None
    return pd.read_csv(DOE_CSV)


# ── Surrogate loader — lazy, only attempted when ML mode selected ──────────────
@st.cache_resource(show_spinner="Loading surrogate model…")
def _load_surrogate():
    """Returns (model, error_string). Model is None if loading failed."""
    if not os.path.exists(MODEL_PKL):
        return None, "E_surrogate_model.pkl not found — run E_surrogate_modeling.py first."
    try:
        import pickle
        import sys as _sys
        _sys.path.insert(0, _DIR)
        import E_surrogate_modeling as _esm  # noqa: F401  — registers module for pickle

        class _RemapUnpickler(pickle.Unpickler):
            """Redirect __main__.MOSAICSurrogate → E_surrogate_modeling.MOSAICSurrogate.
            Needed because the pkl was saved when E_surrogate_modeling ran as __main__."""
            def find_class(self, module, name):
                if module == '__main__':
                    module = 'E_surrogate_modeling'
                return super().find_class(module, name)

        with open(MODEL_PKL, 'rb') as f:
            return _RemapUnpickler(f).load(), None
    except Exception as exc:
        return None, str(exc)


df_raw = _load_database()

if df_raw is None:
    st.error("Gate stack database not found. Run `D_simulation_doe.py` first.")
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(v):
    _sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    return str(v).translate(_sub).encode('ascii', errors='replace').decode('ascii')


def _apply_filters(df: pd.DataFrame, req: dict) -> pd.DataFrame:
    df = df.copy()
    if 'min_ION' in req: df = df[df['ION_uAum']  >= req['min_ION']]
    if 'max_IG'  in req: df = df[df['IG_Aum']    <= req['max_IG']]
    if 'max_SS'  in req: df = df[df['SS_mVdec']  <= req['max_SS']]
    if 'max_EOT' in req: df = df[df['EOT_nm']    <= req['max_EOT']]
    if 'min_VTH' in req: df = df[df['VTH_V']     >= req['min_VTH']]
    if 'max_VTH' in req: df = df[df['VTH_V']     <= req['max_VTH']]
    if 'max_dT'  in req: df = df[df['dT_K']      <= req['max_dT']]
    if 'min_VBD' in req: df = df[df['VBD_V']     >= req['min_VBD']]
    if 'min_Rel' in req: df = df[df['Rel_score'] >= req['min_Rel']]
    if 'max_Eox_frac_EBD' in req and 'EBD_HK' in df.columns:
        frac = req['max_Eox_frac_EBD']
        df = df[df['Eox_MVcm'] <= frac * df['EBD_HK']]
    return df.dropna(subset=['ION_uAum', 'IG_Aum', 'SS_mVdec'])


def _score(df: pd.DataFrame) -> pd.Series:
    return (
        np.log10(df['ION_uAum'].clip(1, None))           * 0.35 +
        (-np.log10(df['IG_Aum'].clip(1e-30, None)))      * 0.25 +
        (-df['SS_mVdec'] / 60)                           * 0.20 +
        df['Rel_score']                                  * 0.20
    )


# ── Hard-coded tier requirements (mirror of E_surrogate_modeling.py) ─────────
TIER_REQUIREMENTS = {
    'Tier 1 — Foundational Design': {
        'min_ION': 100,   'max_IG': 1e-6, 'max_SS': 90,
        'max_EOT': 1.5,   'min_VTH': 0.3, 'max_VTH': 0.7,
    },
    'Tier 2 — Balanced Performance': {
        'min_ION': 300,   'max_IG': 1e-7, 'max_SS': 80,
        'max_EOT': 1.0,   'min_VTH': 0.3, 'max_VTH': 0.6,
    },
    'Tier 3 — Reliability + Thermal': {
        'min_ION': 250,   'max_IG': 1e-7,  'max_SS': 75,
        'max_EOT': 0.9,   'max_Eox_frac_EBD': 0.70,
        'min_VBD': 1.2,   'max_dT': 20,
    },
    'Tier 4 — Pareto + Electro-Thermal': {
        'min_ION': 300,   'max_IG': 1e-8,  'max_SS': 70,
        'max_EOT': 0.7,   'max_Eox_frac_EBD': 0.65,
        'max_dT': 15,     'min_VBD': 1.2,
    },
}

_REQ_LABELS = {
    'min_ION': ('ION ≥',          'µA/µm'),
    'max_IG':  ('IG ≤',           'A/µm'),
    'max_SS':  ('SS ≤',           'mV/dec'),
    'max_EOT': ('EOT ≤',          'nm'),
    'min_VTH': ('VTH ≥',          'V'),
    'max_VTH': ('VTH ≤',          'V'),
    'max_dT':  ('ΔT ≤',           'K'),
    'min_VBD': ('VBD ≥',          'V'),
    'min_Rel': ('Rel score ≥',    ''),
    'max_Eox_frac_EBD': ('Eox ≤ X% of EBD_HK', '%'),
}

def _render_tier_requirements(req: dict):
    """Display tier requirements as a compact table."""
    rows = []
    for key, val in req.items():
        label, unit = _REQ_LABELS.get(key, (key, ''))
        if key == 'max_Eox_frac_EBD':
            rows.append({'Constraint': label, 'Value': f'{val*100:.0f}', 'Unit': unit})
        elif key in ('max_IG',):
            rows.append({'Constraint': label, 'Value': f'{val:.0e}', 'Unit': unit})
        else:
            rows.append({'Constraint': label, 'Value': f'{val}', 'Unit': unit})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── ML candidate generation (with file cache) ────────────────────────────────
ML_CANDIDATES_CSV = os.path.join(_DIR, 'E_ml_candidates.csv')

@st.cache_data(show_spinner="Loading ML candidates…")
def _load_ml_candidates_cache():
    """Load pre-generated ML candidates from file if available."""
    if os.path.exists(ML_CANDIDATES_CSV):
        return pd.read_csv(ML_CANDIDATES_CSV)
    return None


def _generate_ml_candidates(surrogate_model, df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Load ML candidates from E_ml_candidates.csv if it exists (generated by
    E_surrogate_modeling.py). If not, generate fresh via surrogate.predict_batch().
    LHS seed is fixed (42) so candidates are identical every run — caching is safe.
    Delete E_ml_candidates.csv to force regeneration.
    """
    cached = _load_ml_candidates_cache()
    if cached is not None:
        return cached

    # No cache — generate fresh
    ebd_map   = (df_ref.groupby('Dielectric')['EBD_HK'].first().to_dict()
                 if 'EBD_HK' in df_ref.columns else {})
    kappa_map = (df_ref.groupby('Dielectric')['kappa'].first().to_dict()
                 if 'kappa'  in df_ref.columns else {})
    cbo_map   = (df_ref.groupby('Dielectric')['CBO_HK'].first().to_dict()
                 if 'CBO_HK' in df_ref.columns else {})

    combos = df_ref[['Gate', 'Dielectric', 'Substrate']].drop_duplicates().values
    rng    = np.random.default_rng(42)
    rows   = []

    for gate, diel, sub in combos:
        gate, diel, sub = str(gate), str(diel), str(sub)
        il = _il_name(sub)
        ebd   = float(ebd_map.get(diel,   8.0))
        kappa = float(kappa_map.get(diel, 10.0))
        cbo   = float(cbo_map.get(diel,   1.5))

        raw    = rng.uniform(size=(N_ML_SAMPLES, 6))
        lo = np.array([1.0,  0.0, 15.0,  20.0, 0.7, 300.0])
        hi = np.array([10.0, 1.0, 18.0, 100.0, 1.8, 400.0])
        scaled = lo + raw * (hi - lo)

        batch_inputs = []
        for s in scaled:
            batch_inputs.append(dict(
                Gate=gate, Dielectric=diel, Substrate=sub,
                t_HK_nm=float(s[0]),
                t_IL_nm=float(s[1]) if il != 'None' else 0.0,
                Nsub=float(10 ** s[2]),
                L_nm=float(s[3]), VDD=float(s[4]), T_K=float(s[5]),
                EBD_HK=ebd, CBO_HK=cbo, kappa=kappa,
            ))
        try:
            batch_df = pd.DataFrame(batch_inputs)
            preds_df = surrogate_model.predict_batch(batch_df)
            for idx in range(len(preds_df)):
                pred = preds_df.iloc[idx].to_dict()
                if 'error' in pred or pred.get('ION_uAum', 0) < 1.0:
                    continue
                pred.update(batch_inputs[idx])
                pred['feasible'] = True
                rows.append(pred)
        except Exception:
            pass

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df.to_csv(ML_CANDIDATES_CSV, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Page selector + Requirements + source selection
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("MOSAIC Gate Stack Recommender")
st.sidebar.markdown("### 🗂️ Choose a Page")
page_mode = st.sidebar.radio(
    "page_select",
    [
        "📊  Tier Analysis — Tiers 1–4 (pre-defined)",
        "🔍  Custom Search — set your own requirements",
    ],
    key="page_mode",
    label_visibility="collapsed",
)
page_mode = "Tier Analysis" if page_mode.startswith("📊") else "Custom Search"
st.sidebar.caption(
    "**Tier Analysis** evaluates all four pre-defined design tiers at once.  \n"
    "**Custom Search** lets you set your own performance constraints."
)
st.sidebar.divider()

requirements = {}

if page_mode == "Custom Search":
    st.sidebar.subheader("🎯 Design Requirements")
    st.sidebar.caption("Enable a constraint and tune its value.")
    st.sidebar.subheader("Performance")

    col_a, col_b = st.sidebar.columns([1, 2])
    use_ion = col_a.checkbox("ION ≥", value=True, key="use_ion")
    min_ion = col_b.number_input("µA/µm", min_value=0.0, max_value=50000.0,
                                  value=300.0, step=50.0, key="min_ion",
                                  label_visibility="collapsed")
    if use_ion: requirements['min_ION'] = min_ion

    col_a, col_b = st.sidebar.columns([1, 2])
    use_ss = col_a.checkbox("SS ≤", value=True, key="use_ss")
    max_ss = col_b.number_input("mV/dec", min_value=60.0, max_value=200.0,
                                 value=80.0, step=1.0, key="max_ss",
                                 label_visibility="collapsed")
    if use_ss: requirements['max_SS'] = max_ss

    col_a, col_b = st.sidebar.columns([1, 2])
    use_eot = col_a.checkbox("EOT ≤", value=True, key="use_eot")
    max_eot = col_b.number_input("nm", min_value=0.3, max_value=10.0,
                                  value=1.0, step=0.1, key="max_eot",
                                  label_visibility="collapsed")
    if use_eot: requirements['max_EOT'] = max_eot

    use_vth = st.sidebar.checkbox("VTH window", value=True, key="use_vth")
    if use_vth:
        vth_min, vth_max = st.sidebar.slider("VTH range (V)", min_value=0.0,
                                              max_value=1.5, value=(0.3, 0.6),
                                              step=0.05, key="vth_range")
        requirements['min_VTH'] = vth_min
        requirements['max_VTH'] = vth_max

    st.sidebar.subheader("Leakage")

    col_a, col_b = st.sidebar.columns([1, 2])
    use_ig = col_a.checkbox("IG ≤", value=True, key="use_ig")
    ig_map = {'1e-6': 1e-6, '1e-7': 1e-7, '1e-8': 1e-8, '1e-9': 1e-9}
    ig_sel = col_b.selectbox("A/µm", list(ig_map.keys()), index=1, key="max_ig",
                              label_visibility="collapsed")
    if use_ig: requirements['max_IG'] = ig_map[ig_sel]

    st.sidebar.subheader("Reliability & Thermal")

    col_a, col_b = st.sidebar.columns([1, 2])
    use_dt = col_a.checkbox("ΔT ≤", value=False, key="use_dt")
    max_dt = col_b.number_input("K", min_value=0.0, max_value=500.0,
                                 value=30.0, step=5.0, key="max_dt",
                                 label_visibility="collapsed")
    if use_dt: requirements['max_dT'] = max_dt

    col_a, col_b = st.sidebar.columns([1, 2])
    use_vbd = col_a.checkbox("VBD ≥", value=False, key="use_vbd")
    min_vbd = col_b.number_input("V", min_value=0.0, max_value=20.0,
                                  value=1.2, step=0.1, key="min_vbd",
                                  label_visibility="collapsed")
    if use_vbd: requirements['min_VBD'] = min_vbd

    use_eox = st.sidebar.checkbox("Eox ≤ X% of EBD_HK", value=False, key="use_eox")
    if use_eox:
        eox_pct = st.sidebar.slider("Eox limit (% of EBD_HK)", min_value=50,
                                     max_value=100, value=70, step=5, key="eox_pct")
        requirements['max_Eox_frac_EBD'] = eox_pct / 100.0

    col_a, col_b = st.sidebar.columns([1, 2])
    use_rel = col_a.checkbox("Rel ≥", value=False, key="use_rel")
    min_rel = col_b.number_input("(0–1)", min_value=0.0, max_value=1.0,
                                  value=0.5, step=0.05, key="min_rel",
                                  label_visibility="collapsed")
    if use_rel: requirements['min_Rel'] = min_rel

    st.sidebar.subheader("Results")
    top_n = st.sidebar.slider("Top N recommendations", min_value=1, max_value=20,
                               value=5, key="top_n")

# ── Data source selection (shared by both pages) ───────────────────────────────
st.sidebar.subheader("Data Source")
source_mode = st.sidebar.radio(
    "Recommendation source",
    ["Database only", "ML predictions", "Both (side by side)"],
    index=0,
    key="source_mode",
    help=(
        "**Database only** — fast, searches pre-simulated DoE CSV (~4 k rows).\n\n"
        "**ML predictions** — calls surrogate.predict() on fresh LHS samples "
        "across the full continuous parameter space (requires E_surrogate_model.pkl).\n\n"
        "**Both** — runs both and shows results side by side for comparison."
    ),
)

if page_mode == "Tier Analysis":
    top_n = st.sidebar.slider("Top N per tier", min_value=1, max_value=20,
                               value=5, key="top_n_tier")

st.sidebar.divider()
run_btn = st.sidebar.button(
    "▶  Run Tier Analysis" if page_mode == "Tier Analysis" else "▶  Find Gate Stacks",
    type="primary", use_container_width=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.title("⚡ MOSAIC Gate Stack Recommendation Dashboard")

if page_mode == "Tier Analysis":
    st.caption("Pre-defined Tier 1–4 requirements. Click **Run Tier Analysis** to evaluate all tiers.")
else:
    st.caption("Set requirements in the sidebar and click **Find Gate Stacks**.")

if not run_btn:
    if page_mode == "Tier Analysis":
        st.info("👈  Select a data source and click **Run Tier Analysis** to evaluate all four tiers.")
    else:
        st.info("👈  Set your requirements in the sidebar and click **Find Gate Stacks** to begin.")
    st.stop()


# ── Build candidate pools based on selected source ────────────────────────────
use_db = source_mode in ("Database only", "Both (side by side)")
use_ml = source_mode in ("ML predictions", "Both (side by side)")

df_db  = pd.DataFrame()
df_ml  = pd.DataFrame()
surrogate_err = None

if use_db:
    df_db = df_raw[df_raw['feasible'] == True].copy()

if use_ml:
    surrogate, surrogate_err = _load_surrogate()
    if surrogate is not None:
        n_combos = len(df_raw[['Gate', 'Dielectric', 'Substrate']].drop_duplicates())
        with st.spinner(f"Running ML predictions "
                        f"({N_ML_SAMPLES} samples × {n_combos} material combos)…"):
            df_ml = _generate_ml_candidates(surrogate, df_raw)
        if df_ml.empty:
            st.warning("ML prediction returned no results — falling back to database.")
            use_ml = False
            use_db = True
            df_db  = df_raw[df_raw['feasible'] == True].copy()
    else:
        st.warning(f"⚠️ Surrogate model unavailable: {surrogate_err}  "
                   "Showing database results instead.")
        use_ml = False
        use_db = True
        df_db  = df_raw[df_raw['feasible'] == True].copy()


# ── Filter + score + Pareto for each pool ─────────────────────────────────────
def _process_pool(df_cands: pd.DataFrame, req: dict, n: int):
    """Filter, score, compute Pareto. Returns (df_filtered, df_top, df_pareto)."""
    df_f = _apply_filters(df_cands, req)
    if df_f.empty:
        return df_f, pd.DataFrame(), pd.DataFrame()
    obj = np.column_stack([
        -np.log10(df_f['ION_uAum'].clip(1, None).values),
         np.log10(df_f['IG_Aum'].clip(1e-30, None).values),
         df_f['SS_mVdec'].values,
         df_f['dT_K'].values,
    ])
    mask   = pareto_front_nd(obj)
    df_par = df_f[mask].copy().sort_values('ION_uAum', ascending=False).reset_index(drop=True)
    df_f   = df_f.copy()
    df_f['_score'] = _score(df_f)
    df_top = df_f.sort_values('_score', ascending=False).head(n).reset_index(drop=True)
    return df_f, df_top, df_par


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _render_recommendation_cards(df_top: pd.DataFrame, label: str = ''):
    if df_top.empty:
        st.warning(f"No results from {label}.")
        return
    src_note = f"  ·  source: {label}" if label else ''
    st.markdown(
        f"**{len(df_top)} recommendations** ranked by composite score "
        f"(ION 35% · IG 25% · SS 20% · Reliability 20%){src_note}."
    )
    for i, row in df_top.iterrows():
        il_nm  = _il_name(str(row.get('Substrate', '')))
        t_il   = float(row.get('t_IL_nm', 0.0))
        il_str = f"{il_nm}, t_IL={t_il:.2f} nm" if (t_il > 0 and il_nm != 'None') else "None"

        gate = _safe(row.get('Gate', '?'))
        diel = _safe(row.get('Dielectric', '?'))
        sub  = str(row.get('Substrate', '?'))

        with st.expander(
            f"**Rank {i+1}** — {gate} / {diel} / {sub}  · score={row['_score']:.3f}",
            expanded=(i == 0),
        ):
            if 'La' in str(row.get('Dielectric', '')):
                st.warning("⚠️ La₂O₃ is highly hygroscopic — absorbs moisture from air, "
                           "degrading interface quality and dielectric stability. "
                           "Requires hermetic encapsulation or capping layer.")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Gate Stack**")
                st.write(f"Gate electrode : {gate}")
                st.write(f"Dielectric     : {diel}")
                st.write(f"Substrate      : {sub}")
                st.write(f"t_HK           : {row.get('t_HK_nm', float('nan')):.2f} nm")
                st.write(f"Interfacial L. : {il_str}")
                st.write(f"Nsub           : {row.get('Nsub', float('nan')):.2e} cm⁻³")
                st.write(f"L (gate length): {row.get('L_nm', float('nan')):.1f} nm")
                st.write(f"VDD            : {row.get('VDD', float('nan')):.3f} V")
            with c2:
                st.markdown("**Performance**")
                st.write(f"ION   : {row.get('ION_uAum', float('nan')):.1f} µA/µm")
                st.write(f"SS    : {row.get('SS_mVdec', float('nan')):.2f} mV/dec")
                st.write(f"VTH   : {row.get('VTH_V', float('nan')):.3f} V")
                st.write(f"EOT   : {row.get('EOT_nm', float('nan')):.3f} nm")
                if 'COX_Fm2'  in row: st.write(f"COX   : {row['COX_Fm2']:.4f} F/m²")
                st.write(f"IG    : {row.get('IG_Aum', float('nan')):.3e} A/µm")
                if 'Eox_MVcm' in row: st.write(f"Eox   : {row['Eox_MVcm']:.3f} MV/cm")
                if 'DIBL_mVV' in row: st.write(f"DIBL  : {row['DIBL_mVV']:.1f} mV/V")
            with c3:
                st.markdown("**Reliability**")
                st.write(f"VBD       : {row.get('VBD_V', float('nan')):.2f} V")
                st.write(f"ΔT        : {row.get('dT_K', float('nan')):.1f} K")
                st.write(f"Rel score : {row.get('Rel_score', float('nan')):.3f}")
                if 'ION_IOFF' in row: st.write(f"ION/IOFF  : {row['ION_IOFF']:.2e}")
                if 'Dit_flag' in row:
                    dit = str(row['Dit_flag'])
                    dit_icon = "✓" if dit == "Low" else ("⚠" if dit == "Med" else "✖")
                    st.write(f"Dit flag    : {dit_icon} {dit}")
                if 'cryst_risk' in row:
                    risk = "⚠ Yes" if row['cryst_risk'] else "✓ No"
                    t_cryst_map = _load_t_cryst_map()
                    t_c = t_cryst_map.get(str(row.get('Dielectric', '')))
                    t_c_str = f"  (T_cryst = {t_c:.0f}°C)" if t_c is not None else ""
                    st.write(f"Cryst. risk : {risk}{t_c_str}")

    dl_cols = ['Gate', 'Dielectric', 'Substrate', 't_HK_nm', 't_IL_nm',
               'Nsub', 'L_nm', 'VDD', 'EOT_nm', 'VTH_V', 'ION_uAum',
               'SS_mVdec', 'IG_Aum', 'Eox_MVcm', 'VBD_V', 'dT_K',
               'Rel_score', 'ION_IOFF', '_score']
    st.download_button(
        f"⬇  Download ({label}) as CSV",
        data=df_top[[c for c in dl_cols if c in df_top.columns]].to_csv(index=False),
        file_name=f"mosaic_{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


def _render_pareto_tab(df_filtered: pd.DataFrame, df_pareto: pd.DataFrame, label: str = ''):
    n_par = len(df_pareto)
    n_mat = len(df_filtered)
    if n_par == 0:
        st.warning(f"No Pareto-optimal designs found ({label}).")
        return
    st.markdown(
        f"**{n_par} Pareto-optimal designs** from {n_mat} matched ({label}). "
        "No other design beats each of these on all four objectives simultaneously "
        "(maximize ION · minimize IG · minimize SS · minimize ΔT)."
    )
    df_p = df_pareto.copy()
    df_p['Design#'] = [f"#{i+1}" for i in range(len(df_p))]
    df_p['Stack']   = (df_p['Gate'].apply(_safe) + ' / ' +
                       df_p['Dielectric'].astype(str) + ' / ' +
                       df_p['Substrate'].astype(str))
    df_p['log_IG']  = np.log10(df_p['IG_Aum'].clip(1e-30))
    df_filtered = df_filtered.copy()
    df_filtered['log_IG'] = np.log10(df_filtered['IG_Aum'].clip(1e-30))

    hover_template = (
        '<b>%{customdata[0]}</b><br>'
        'ION=%{customdata[1]:.1f} µA/µm<br>'
        'IG=%{customdata[2]:.2e} A/µm<br>'
        'SS=%{customdata[3]:.1f} mV/dec<br>'
        'ΔT=%{customdata[4]:.1f} K<br>'
        'EOT=%{customdata[5]:.3f} nm  VTH=%{customdata[6]:.3f} V<br>'
        'VBD=%{customdata[7]:.2f} V  Rel=%{customdata[8]:.3f}'
        '<extra></extra>'
    )
    hover_data = df_p[['Stack', 'ION_uAum', 'IG_Aum', 'SS_mVdec',
                        'dT_K', 'EOT_nm', 'VTH_V', 'VBD_V', 'Rel_score']].values

    subplot_cfg = [
        ('ION_uAum', 'log_IG',   1, 1, 'ION (µA/µm)', 'log₁₀(IG) [lower=better]'),
        ('ION_uAum', 'SS_mVdec', 1, 2, 'ION (µA/µm)', 'SS (mV/dec) [lower=better]'),
        ('ION_uAum', 'dT_K',     2, 1, 'ION (µA/µm)', 'ΔT (K) [lower=better]'),
        ('SS_mVdec', 'dT_K',     2, 2, 'SS (mV/dec)', 'ΔT (K)'),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['ION vs IG', 'ION vs SS', 'ION vs ΔT', 'SS vs ΔT'],
                        horizontal_spacing=0.12, vertical_spacing=0.15)

    for cx, cy, r, c, lx, ly in subplot_cfg:
        fig.add_trace(go.Scatter(
            x=df_filtered[cx], y=df_filtered[cy], mode='markers',
            marker=dict(size=4, color='lightsteelblue', opacity=0.35),
            name='All matched', showlegend=(r == 1 and c == 1), hoverinfo='skip',
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=df_p[cx], y=df_p[cy], mode='markers+text',
            marker=dict(size=13, color='crimson', line=dict(width=1.2, color='darkred')),
            text=df_p['Design#'], textposition='top center',
            textfont=dict(size=8, color='darkred'),
            name='Pareto-optimal', showlegend=(r == 1 and c == 1),
            customdata=hover_data, hovertemplate=hover_template,
        ), row=r, col=c)
        fig.update_xaxes(title_text=lx, row=r, col=c)
        fig.update_yaxes(title_text=ly, row=r, col=c)

    fig.update_layout(height=680, title_text=f'Pareto Front — {label}',
                      title_font_size=14,
                      legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
                      hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)

    show_cols = ['Design#', 'Gate', 'Dielectric', 'Substrate', 't_HK_nm', 't_IL_nm',
                 'L_nm', 'VDD', 'Nsub', 'ION_uAum', 'SS_mVdec', 'IG_Aum', 'dT_K',
                 'EOT_nm', 'VTH_V', 'VBD_V', 'Eox_MVcm', 'Rel_score']
    st.dataframe(df_p[[c for c in show_cols if c in df_p.columns]],
                 use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇  Download Pareto ({label}) as CSV",
        data=df_p[[c for c in show_cols if c in df_p.columns]].to_csv(index=False),
        file_name=f"mosaic_pareto_{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE RENDERING
# ══════════════════════════════════════════════════════════════════════════════

if page_mode == "Tier Analysis":
    # ── Run all 4 tiers ────────────────────────────────────────────────────────
    st.subheader("Tier Analysis — Tiers 1 to 4")
    st.caption(
        "Requirements are fixed per tier (matching E_surrogate_modeling.py). "
        "Tier 4 additionally shows the 4-objective Pareto-optimal designs."
    )

    tier_tabs = st.tabs([
        "Tier 1 — Foundational",
        "Tier 2 — Balanced",
        "Tier 3 — Reliability + Thermal",
        "Tier 4 — Pareto + Electro-Thermal",
    ])

    tier_names  = list(TIER_REQUIREMENTS.keys())
    tier_colors = ["🟢", "🔵", "🟠", "🔴"]

    for tab, tier_name, color in zip(tier_tabs, tier_names, tier_colors):
        req = TIER_REQUIREMENTS[tier_name]
        with tab:
            st.markdown(f"#### {color} {tier_name}")

            # Requirements table
            with st.expander("Target Requirements", expanded=True):
                _render_tier_requirements(req)

            # Process each source
            res_t_db = _process_pool(df_db, req, top_n) if use_db else (pd.DataFrame(),)*3
            res_t_ml = _process_pool(df_ml, req, top_n) if use_ml else (pd.DataFrame(),)*3
            filt_db, top_db, par_db = res_t_db
            filt_ml, top_ml, par_ml = res_t_ml

            # Summary metrics
            if source_mode == "Both (side by side)":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("DB feasible",   f"{len(df_db):,}")
                c2.metric("DB matched",    f"{len(filt_db):,}")
                c3.metric("ML candidates", f"{len(df_ml):,}")
                c4.metric("ML matched",    f"{len(filt_ml):,}")
            elif use_ml:
                c1, c2, c3 = st.columns(3)
                c1.metric("ML candidates",    f"{len(df_ml):,}")
                c2.metric("ML matched",       f"{len(filt_ml):,}")
                c3.metric("Pareto-optimal",   f"{len(par_ml):,}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("DB feasible",      f"{len(df_db):,}")
                c2.metric("DB matched",       f"{len(filt_db):,}")
                c3.metric("Pareto-optimal",   f"{len(par_db):,}")

            # Recommendations + Pareto tabs
            is_tier4 = (tier_name == tier_names[3])

            if source_mode == "Both (side by side)":
                sub_tabs = st.tabs(["🏆 Recommendations", "📊 Pareto-Optimal"])
                with sub_tabs[0]:
                    cl, cr = st.columns(2)
                    with cl:
                        st.markdown("**🗄️ Database**")
                        _render_recommendation_cards(top_db, f"DB-{tier_name[:6]}")
                    with cr:
                        st.markdown("**🤖 ML**")
                        _render_recommendation_cards(top_ml, f"ML-{tier_name[:6]}")
                with sub_tabs[1]:
                    cl, cr = st.columns(2)
                    with cl:
                        st.markdown("**🗄️ Database**")
                        _render_pareto_tab(filt_db, par_db, f"DB-{tier_name[:6]}")
                    with cr:
                        st.markdown("**🤖 ML**")
                        _render_pareto_tab(filt_ml, par_ml, f"ML-{tier_name[:6]}")

            elif use_ml:
                sub_tabs = st.tabs(["🏆 Recommendations (ML)", "📊 Pareto-Optimal (ML)"])
                with sub_tabs[0]:
                    _render_recommendation_cards(top_ml, f"ML-{tier_name[:6]}")
                with sub_tabs[1]:
                    _render_pareto_tab(filt_ml, par_ml, f"ML-{tier_name[:6]}")

            else:
                sub_tabs = st.tabs(["🏆 Recommendations", "📊 Pareto-Optimal"])
                with sub_tabs[0]:
                    _render_recommendation_cards(top_db, f"DB-{tier_name[:6]}")
                with sub_tabs[1]:
                    _render_pareto_tab(filt_db, par_db, f"DB-{tier_name[:6]}")


else:
    # ══════════════════════════════════════════════════════════════════════════
    #  CUSTOM SEARCH PAGE
    # ══════════════════════════════════════════════════════════════════════════
    res_db = _process_pool(df_db, requirements, top_n) if use_db else (pd.DataFrame(),)*3
    res_ml = _process_pool(df_ml, requirements, top_n) if use_ml else (pd.DataFrame(),)*3

    df_filt_db, df_top_db, df_par_db = res_db
    df_filt_ml, df_top_ml, df_par_ml = res_ml

    n_db     = len(df_filt_db)
    n_ml     = len(df_filt_ml)
    n_par_db = len(df_par_db)
    n_par_ml = len(df_par_ml)

    if source_mode == "Both (side by side)":
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Database feasible",    f"{len(df_db):,}")
        c2.metric("DB matched",           f"{n_db:,}")
        c3.metric("ML candidates",        f"{len(df_ml):,}")
        c4.metric("ML matched",           f"{n_ml:,}")
    elif use_ml:
        c1, c2, c3 = st.columns(3)
        c1.metric("ML candidates generated", f"{len(df_ml):,}")
        c2.metric("Matched requirements",    f"{n_ml:,}")
        c3.metric("Pareto-optimal",          f"{n_par_ml:,}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total feasible in database", f"{len(df_db):,}")
        c2.metric("Matched requirements",       f"{n_db:,}")
        c3.metric("Pareto-optimal",             f"{n_par_db:,}")

    if n_db == 0 and n_ml == 0:
        st.error("No gate stacks matched all requirements. Try relaxing one or more constraints.")
        st.stop()

    if source_mode == "Both (side by side)":
        tab1, tab2 = st.tabs(["🏆  Recommendations", "📊  Pareto-Optimal Designs"])
        with tab1:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🗄️ Database Search")
                _render_recommendation_cards(df_top_db, "database")
            with col_r:
                st.markdown("### 🤖 ML Predictions")
                _render_recommendation_cards(df_top_ml, "ML")
        with tab2:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🗄️ Database")
                _render_pareto_tab(df_filt_db, df_par_db, "database")
            with col_r:
                st.markdown("### 🤖 ML")
                _render_pareto_tab(df_filt_ml, df_par_ml, "ML")

    elif use_ml:
        tab1, tab2 = st.tabs([
            f"🏆  Top {top_n} Recommendations (ML)",
            f"📊  Pareto-Optimal Designs — ML  ({n_par_ml})",
        ])
        with tab1:
            st.info("🤖 Results from **surrogate.predict()** — exploring the full continuous "
                    "structural parameter space beyond the pre-sampled DoE points.")
            _render_recommendation_cards(df_top_ml, "ML")
        with tab2:
            _render_pareto_tab(df_filt_ml, df_par_ml, "ML predictions")

    else:
        tab1, tab2 = st.tabs([
            f"🏆  Top {top_n} Recommendations",
            f"📊  Pareto-Optimal Designs  ({n_par_db})",
        ])
        with tab1:
            _render_recommendation_cards(df_top_db, "database")
        with tab2:
            _render_pareto_tab(df_filt_db, df_par_db, "database")
