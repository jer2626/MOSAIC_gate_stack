"""
MOSAIC Task E: Compact / Surrogate Modeling Framework
Trains ML surrogate models on the DoE database (Task D) to map all inputs
to all outputs. Includes preprocessing, cross-validation, feature importance,
sensitivity analysis, and a prediction interface.
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
from C_device_modeling import get_il_props
import matplotlib
matplotlib.use('Agg')   # headless backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, KFold, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network     import MLPRegressor
from sklearn.pipeline           import Pipeline
from sklearn.multioutput        import MultiOutputRegressor
from sklearn.metrics            import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection         import permutation_importance

# ── Target outputs (log-transformed where appropriate) ─────────────────────
TARGETS_LINEAR = ['VTH_V', 'VTH_long_V', 'EOT_nm', 'SS_mVdec', 'Eox_MVcm', 'VBD_V',
                   'dT_K', 'T_junction_K', 'Rel_score', 'COX_Fm2', 'lambda_nm', 'DIBL_mVV']
TARGETS_LOG    = ['ION_uAum', 'IG_Aum', 'J_G_Acm2', 'ION_IOFF']
ALL_TARGETS    = TARGETS_LINEAR + TARGETS_LOG

# ── Categorical input columns ──────────────────────────────────────────────
CAT_COLS = ['Gate', 'Dielectric', 'Substrate']

# ── Numeric input columns ──────────────────────────────────────────────────
NUM_COLS = ['t_HK_nm', 't_IL_nm', 'Nsub', 'L_nm', 'VDD', 'T_K',
            'EBD_HK', 'CBO_HK', 'kappa']


# ── Data loading & cleaning ────────────────────────────────────────────────

def load_and_clean(csv_path: str = 'D_gate_stack_database.csv') -> pd.DataFrame:
    """Load DoE CSV, keep only feasible rows, drop columns not needed for ML."""
    df = pd.read_csv(csv_path)
    df = df[df['feasible'] == True].copy()
    df = df.dropna(subset=ALL_TARGETS + CAT_COLS + NUM_COLS, how='any')
    # Drop near-zero ION rows (depletion-mode: VTH > VDD) to avoid log-space bimodality
    if 'ION_uAum' in df.columns:
        df = df[df['ION_uAum'] >= 1.0]

    # Log-transform positive-definite targets for better regression
    for col in TARGETS_LOG:
        if col in df.columns:
            df[col + '_raw'] = df[col]
            df[col] = np.log10(np.clip(df[col].astype(float), 1e-30, None))

    # Log-transform Nsub (spans orders of magnitude)
    df['Nsub_log'] = np.log10(df['Nsub'].astype(float).clip(1e10, None))

    print(f"Loaded {len(df)} feasible samples.")
    return df


# ── Feature encoding pipeline ──────────────────────────────────────────────

class FeatureEncoder:
    """Encode categoricals with LabelEncoder, keep numerics."""

    def __init__(self):
        self.encoders = {col: LabelEncoder() for col in CAT_COLS}
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in CAT_COLS:
            enc = self.encoders[col]
            parts.append(enc.fit_transform(df[col].astype(str)).reshape(-1, 1))
        num_cols_use = ['t_HK_nm', 't_IL_nm', 'Nsub_log', 'L_nm', 'VDD', 'T_K',
                        'EBD_HK', 'CBO_HK', 'kappa']
        num_arr = df[num_cols_use].values.astype(float)
        self.fitted = True
        self.num_cols_use = num_cols_use
        X = np.hstack(parts + [num_arr])
        self.feature_names = CAT_COLS + num_cols_use
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in CAT_COLS:
            parts.append(self.encoders[col].transform(df[col].astype(str)).reshape(-1, 1))
        num_arr = df[self.num_cols_use].values.astype(float)
        return np.hstack(parts + [num_arr])

    def transform_dict(self, cfg: dict) -> np.ndarray:
        row_df = pd.DataFrame([cfg])
        if 'Nsub_log' not in row_df.columns:
            row_df['Nsub_log'] = np.log10(float(cfg.get('Nsub', 1e17)))
        return self.transform(row_df)


# ── Model definitions ──────────────────────────────────────────────────────

def build_models():
    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42),
        'GradientBoosting': MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42),
            n_jobs=-1),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(128, 128, 64), activation='relu',
            max_iter=500, learning_rate_init=1e-3,
            early_stopping=True, validation_fraction=0.1,
            random_state=42),
    }


# ── Training & evaluation ──────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       target_names, scaler_X, scaler_y):
    """Train all models, return results dict."""
    models_def = build_models()
    results    = {}

    Xs_train = scaler_X.transform(X_train)
    Xs_test  = scaler_X.transform(X_test)
    ys_train = scaler_y.transform(y_train)

    for name, model in models_def.items():
        print(f"  Training {name}...")
        model.fit(Xs_train, ys_train)
        ys_pred = model.predict(Xs_test)
        y_pred  = scaler_y.inverse_transform(ys_pred)

        r2s  = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
        maes = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]

        mean_r2 = np.mean(r2s)
        print(f"    Mean R²={mean_r2:.4f}  |  per-target R²:")
        for t, r in zip(target_names, r2s):
            print(f"      {t:20s}: R²={r:.4f}  MAE={maes[target_names.index(t)]:.4e}")

        results[name] = {
            'model': model,
            'r2_per_target': dict(zip(target_names, r2s)),
            'mae_per_target': dict(zip(target_names, maes)),
            'mean_r2': mean_r2,
        }

    return results


def cross_validate_best(X, y_scaled, model, scaler_X, k=5):
    """K-fold CV on the best model."""
    kf  = KFold(n_splits=k, shuffle=True, random_state=42)
    r2s = cross_val_score(model, scaler_X.transform(X), y_scaled,
                           cv=kf, scoring='r2', n_jobs=-1)
    print(f"\n  {k}-Fold CV R²: {r2s.mean():.4f} ± {r2s.std():.4f}")
    return r2s


# ── Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, target_names,
                             model_name='RandomForest', out_dir='.'):
    """Works for RandomForest and MultiOutputRegressor(GBR)."""
    imp = None
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # MultiOutputRegressor: average over per-target estimators
        imps = [est.feature_importances_ for est in model.estimators_
                if hasattr(est, 'feature_importances_')]
        if imps:
            imp = np.mean(imps, axis=0)
    if imp is None:
        print(f"  Feature importance not available for {model_name}")
        return None

    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': imp})
    df_imp = df_imp.sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_imp.head(12), x='Importance', y='Feature', ax=ax)
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_xlabel('Mean Decrease in Impurity')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'E_feature_importance_{model_name}.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")
    return df_imp


# ── Parity plots ──────────────────────────────────────────────────────────

def plot_parity(y_test, y_pred, target_names, model_name='RF', out_dir='.'):
    n = len(target_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, tname in enumerate(target_names):
        ax = axes[i]
        yt = y_test[:, i]
        yp = y_pred[:, i]
        ax.scatter(yt, yp, alpha=0.3, s=5)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        r2 = r2_score(yt, yp)
        ax.set_title(f'{tname}\nR²={r2:.3f}', fontsize=9)
        ax.set_xlabel('True', fontsize=8)
        ax.set_ylabel('Pred', fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Parity Plots — {model_name}', y=1.01, fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, f'E_parity_{model_name}.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Sensitivity analysis ──────────────────────────────────────────────────

def sensitivity_analysis(model, X_test_scaled, y_test, target_names,
                          feature_names, scaler_X, scaler_y, out_dir='.'):
    """
    Sensitivity via input perturbation (+-1 std in scaled space).
    Works with any sklearn model type, including MultiOutputRegressor.
    """
    print("\nSensitivity analysis (perturbation method)...")
    sensitivity = {t: {} for t in target_names}

    for fi, fname in enumerate(feature_names):
        X_pos = X_test_scaled.copy(); X_pos[:, fi] += 1.0
        X_neg = X_test_scaled.copy(); X_neg[:, fi] -= 1.0
        y_pos = scaler_y.inverse_transform(model.predict(X_pos))
        y_neg = scaler_y.inverse_transform(model.predict(X_neg))
        dy = np.abs(y_pos - y_neg) / 2.0
        for ti, tname in enumerate(target_names):
            sensitivity[tname][fname] = float(np.mean(dy[:, ti]))

    df_sens = pd.DataFrame(sensitivity, index=feature_names)
    df_norm = df_sens.div(df_sens.max(axis=0).replace(0, 1), axis=1)
    df_norm.to_csv(os.path.join(out_dir, 'E_sensitivity.csv'))

    fig, ax = plt.subplots(figsize=(max(12, len(target_names)), 5))
    sns.heatmap(df_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.3, annot_kws={'size': 7})
    ax.set_title('Input-Output Sensitivity (normalised)')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'E_sensitivity_heatmap.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return df_norm


# ── Surrogate model wrapper ────────────────────────────────────────────────

class MOSAICSurrogate:
    """
    Unified surrogate model: maps M-O-S inputs → all device outputs.
    Handles encoding, scaling, log-transforms automatically.
    """

    def __init__(self):
        self.encoder   = FeatureEncoder()
        self.scaler_X  = StandardScaler()
        self.scaler_y  = StandardScaler()
        self.model     = None
        self.model_name= None
        self.targets   = None
        self.log_targets = None

    def fit(self, df: pd.DataFrame, model_name: str = 'RandomForest'):
        targets_present = [t for t in ALL_TARGETS if t in df.columns]
        self.targets = targets_present
        self.log_targets = [t for t in TARGETS_LOG if t in targets_present]

        X = self.encoder.fit_transform(df)
        y = df[targets_present].values.astype(float)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.scaler_X.fit(X_tr)
        self.scaler_y.fit(y_tr)

        Xs_tr = self.scaler_X.transform(X_tr)
        ys_tr = self.scaler_y.transform(y_tr)

        models = build_models()
        self.model = models[model_name]
        self.model_name = model_name
        print(f"Fitting {model_name}...")
        self.model.fit(Xs_tr, ys_tr)

        ys_pred = self.model.predict(self.scaler_X.transform(X_te))
        y_pred  = self.scaler_y.inverse_transform(ys_pred)
        r2s = [r2_score(y_te[:, i], y_pred[:, i]) for i in range(len(targets_present))]
        print(f"Test-set R² (mean): {np.mean(r2s):.4f}")
        for t, r in zip(targets_present, r2s):
            print(f"  {t:20s}: R²={r:.4f}")

        return X_te, y_te, y_pred

    def predict_batch(self, cfgs: pd.DataFrame) -> pd.DataFrame:
        """
        Predict all device metrics for a batch of configurations at once.
        cfgs: DataFrame with columns Gate, Dielectric, Substrate, t_HK_nm,
              t_IL_nm, Nsub, L_nm, VDD, T_K, [EBD_HK, CBO_HK, kappa].
        Returns DataFrame with one row per input, columns = target names.
        Much faster than calling predict() in a loop — only one DataFrame
        creation and one model.predict() call for the whole batch.
        """
        df = cfgs.copy()
        df['Nsub_log'] = np.log10(df['Nsub'].astype(float).clip(1e10))
        for col, default in [('EBD_HK', 8.0), ('CBO_HK', 1.5),
                              ('kappa', 20.0), ('T_K', 300.0)]:
            if col not in df.columns:
                df[col] = default

        try:
            X = self.encoder.transform(df)
        except Exception as e:
            return pd.DataFrame([{'error': str(e)}] * len(df))

        Xs = self.scaler_X.transform(X)
        ys = self.model.predict(Xs)
        y  = self.scaler_y.inverse_transform(ys)

        BOUNDS = {
            'ION_uAum': (0, 1e5),    'IG_Aum':   (1e-20, 1e-2),
            'J_G_Acm2': (1e-20, 10), 'ION_IOFF': (1, 1e15),
            'SS_mVdec': (60, 300),   'EOT_nm':   (0.1, 15),
            'VBD_V':    (0, 30),     'dT_K':     (0, 1000),
        }
        rows = []
        for row_y in y:
            result = {}
            for i, t in enumerate(self.targets):
                val = float(row_y[i])
                if t in self.log_targets:
                    val = 10 ** val
                if t in BOUNDS:
                    lo, hi = BOUNDS[t]
                    val = float(np.clip(val, lo, hi))
                result[t] = val
            rows.append(result)
        return pd.DataFrame(rows)

    def predict(self, cfg: dict) -> dict:
        """
        Predict all device metrics for a single gate stack configuration dict.
        cfg keys: Gate, Dielectric, Substrate, t_HK_nm, t_IL_nm, Nsub,
                  L_nm, VDD, T_K, [EBD_HK, CBO_HK, kappa]
        """
        row = dict(cfg)
        row['Nsub_log'] = np.log10(float(row.get('Nsub', 1e17)))
        row.setdefault('EBD_HK', 8.0)
        row.setdefault('CBO_HK', 1.5)
        row.setdefault('kappa',  20.0)
        row.setdefault('T_K', 300.0)

        try:
            X = self.encoder.transform_dict(row)
        except Exception as e:
            return {'error': f'Encoding failed: {e}'}

        Xs = self.scaler_X.transform(X.reshape(1, -1))
        ys = self.model.predict(Xs)
        y  = self.scaler_y.inverse_transform(ys)[0]

        result = {}
        BOUNDS = {
            'ION_uAum': (0, 1e5), 'IG_Aum': (1e-20, 1e-2),
            'J_G_Acm2': (1e-20, 10), 'ION_IOFF': (1, 1e15),
            'SS_mVdec': (60, 300), 'EOT_nm': (0.1, 15),
            'VBD_V': (0, 30), 'dT_K': (0, 1000),
        }
        for i, t in enumerate(self.targets):
            val = float(y[i])
            if t in self.log_targets:
                val = 10 ** val
            if t in BOUNDS:
                lo, hi = BOUNDS[t]
                val = float(np.clip(val, lo, hi))
            result[t] = val

        return result

    def save(self, path: str = 'E_surrogate_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Surrogate model saved: {path}")

    @staticmethod
    def load(path: str = 'E_surrogate_model.pkl') -> 'MOSAICSurrogate':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Recommendation interface ──────────────────────────────────────────────

def recommend(surrogate: MOSAICSurrogate,
              requirements: dict,
              df_doe: pd.DataFrame,
              top_n: int = 10,
              rel_weight: float = 0.20,
              tier_label: str = '') -> pd.DataFrame:
    """
    Search the DoE database for gate stacks meeting user-specified requirements.
    Returns ranked gate stack material combinations with their performance metrics.

    requirements keys (all optional):
        min_ION          : minimum drive current (μA/μm)
        max_IG           : maximum gate leakage (A/μm)
        max_SS           : maximum subthreshold swing (mV/dec)
        max_EOT          : maximum equivalent oxide thickness (nm)
        min_VTH          : minimum threshold voltage (V)
        max_VTH          : maximum threshold voltage (V)
        max_dT           : maximum self-heating (K)
        min_VBD          : minimum breakdown voltage (V)
        min_Rel          : minimum reliability score (0-1)
        max_Eox          : maximum oxide electric field (MV/cm)  Eox = Vox/tox
        min_Eox          : minimum oxide electric field (MV/cm)  Eox = Vox/tox
        max_Eox_frac_EBD : Eox <= frac × EBD_HK per row  (e.g. 0.8 = max 80% of EBD)

    rel_weight : float (0–1)
        Weight given to reliability score in the composite ranking.
        Remaining weight (1 - rel_weight) is split across ION, IG, SS.
        Default 0.20; set higher (e.g. 0.40) to prioritise reliability.
    tier_label : str
        Optional label printed in the section header (e.g. 'Tier 3').
    """
    df = df_doe[df_doe['feasible'] == True].copy()

    filters_applied = []
    if 'min_ION' in requirements:
        df = df[df['ION_uAum'] >= requirements['min_ION']]
        filters_applied.append(f"ION >= {requirements['min_ION']} uA/um")
    if 'max_IG' in requirements:
        df = df[df['IG_Aum'] <= requirements['max_IG']]
        filters_applied.append(f"IG <= {requirements['max_IG']:.0e} A/um")
    if 'max_SS' in requirements:
        df = df[df['SS_mVdec'] <= requirements['max_SS']]
        filters_applied.append(f"SS <= {requirements['max_SS']} mV/dec")
    if 'max_EOT' in requirements:
        df = df[df['EOT_nm'] <= requirements['max_EOT']]
        filters_applied.append(f"EOT <= {requirements['max_EOT']} nm")
    if 'min_VTH' in requirements:
        df = df[df['VTH_V'] >= requirements['min_VTH']]
        filters_applied.append(f"VTH >= {requirements['min_VTH']} V")
    if 'max_VTH' in requirements:
        df = df[df['VTH_V'] <= requirements['max_VTH']]
        filters_applied.append(f"VTH <= {requirements['max_VTH']} V")
    if 'max_dT' in requirements:
        df = df[df['dT_K'] <= requirements['max_dT']]
        filters_applied.append(f"dT <= {requirements['max_dT']} K")
    if 'min_VBD' in requirements:
        df = df[df['VBD_V'] >= requirements['min_VBD']]
        filters_applied.append(f"VBD >= {requirements['min_VBD']} V")
    if 'min_Rel' in requirements:
        df = df[df['Rel_score'] >= requirements['min_Rel']]
        filters_applied.append(f"Rel >= {requirements['min_Rel']}")
    if 'max_Eox' in requirements:
        df = df[df['Eox_MVcm'] <= requirements['max_Eox']]
        filters_applied.append(f"Eox <= {requirements['max_Eox']} MV/cm")
    if 'min_Eox' in requirements:
        df = df[df['Eox_MVcm'] >= requirements['min_Eox']]
        filters_applied.append(f"Eox >= {requirements['min_Eox']} MV/cm")
    if 'max_Eox_frac_EBD' in requirements:
        frac = requirements['max_Eox_frac_EBD']
        df = df[df['Eox_MVcm'] <= frac * df['EBD_HK']]
        filters_applied.append(f"Eox <= {frac*100:.0f}% of EBD_HK")

    n_matched = len(df)
    if n_matched == 0:
        print("\nNo gate stacks matched all requirements.")
        print("Try relaxing one or more constraints.")
        return pd.DataFrame()

    # Rank by composite score: high ION, low IG, low SS, high Rel_score
    df['_score'] = (
        np.log10(df['ION_uAum'].clip(1, None)) * 0.35 +
        (-np.log10(df['IG_Aum'].clip(1e-30, None))) * 0.25 +
        (-df['SS_mVdec'] / 60) * 0.20 +
        df['Rel_score'] * 0.20
    )
    df = df.sort_values('_score', ascending=False).head(top_n)

    # ── Print results ──────────────────────────────────────────────────────
    sep = "=" * 62
    header = f"  GATE STACK RECOMMENDATIONS — {tier_label}" if tier_label else "  GATE STACK RECOMMENDATIONS"
    print(f"\n{sep}")
    print(header)
    print(sep)
    print(f"  Requirements: {' | '.join(filters_applied)}")
    print(f"  {n_matched} feasible designs found. Showing top {len(df)} ranked by score.")
    print(sep)

    def _s(v):
        _sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        return str(v).translate(_sub).encode('ascii', errors='replace').decode('ascii')

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        gate  = _s(row.get('Gate', '?'))
        diel  = _s(row.get('Dielectric', '?'))
        sub   = _s(row.get('Substrate', '?'))
        t_hk  = row.get('t_HK_nm', float('nan'))
        t_il  = row.get('t_IL_nm', 0.0)
        l_nm  = row.get('L_nm',    float('nan'))
        vdd   = row.get('VDD',     float('nan'))
        nsub  = row.get('Nsub',    float('nan'))
        t_k   = row.get('T_K',     float('nan'))
        ion   = row.get('ION_uAum', float('nan'))
        ig    = row.get('IG_Aum', float('nan'))
        ss    = row.get('SS_mVdec', float('nan'))
        vth   = row.get('VTH_V', float('nan'))
        eot   = row.get('EOT_nm', float('nan'))
        vbd   = row.get('VBD_V', float('nan'))
        dt    = row.get('dT_K', float('nan'))
        rel   = row.get('Rel_score', float('nan'))
        eox   = row.get('Eox_MVcm', float('nan'))
        score = row.get('_score', float('nan'))

        # IL material is fixed by substrate chemistry
        il_name, _ = get_il_props(sub)
        if t_il > 0 and il_name is not None:
            il_str = f", IL={il_name} t_IL={t_il:.2f}nm"
        else:
            il_str = ", no IL"

        print(f"\n  Rank {rank}  [score={score:.3f}]")
        print(f"    Gate Stack  : {gate}  /  {diel}  /  {sub}")
        if 'La' in diel:
            print(f"    *** WARNING : La2O3 is highly hygroscopic — absorbs moisture from air,")
            print(f"                  degrading interface quality and dielectric stability.")
            print(f"                  Requires hermetic encapsulation or capping layer.")
        print(f"    Oxide       : t_HK={t_hk:.2f}nm{il_str}")
        print(f"    Inputs      : L={l_nm:.1f}nm | VDD={vdd:.3f}V | Nsub={nsub:.2e} cm\u207b\u00b3 | T={t_k:.1f}K")
        dibl_mv  = row.get('DIBL_mVV',   float('nan'))
        lam_nm   = row.get('lambda_nm',  float('nan'))
        sce      = row.get('SCE_short',  None)
        vth_long = row.get('VTH_long_V', float('nan'))
        sce_str  = ('yes' if sce else 'no') if sce is not None else 'n/a'
        print(f"    Performance : ION={ion:.1f} uA/um | SS={ss:.1f} mV/dec | VTH={vth:.3f}V (long={vth_long:.3f}V) | EOT={eot:.3f}nm")
        print(f"    SCE         : lambda={lam_nm:.1f}nm | DIBL={dibl_mv:.1f}mV/V | short-channel={sce_str}")
        print(f"    Leakage     : IG={ig:.3e} A/um")
        print(f"    Oxide Field : Eox={eox:.3f} MV/cm")
        t_junc   = row.get('T_junction_K', float('nan'))
        dit_flag = row.get('Dit_flag', 'n/a')
        print(f"    Reliability : VBD={vbd:.2f}V | dT={dt:.1f}K | T_junction={t_junc:.1f}K | Rel={rel:.3f} | Dit={dit_flag}")

    print(f"\n{sep}")
    return df.drop(columns=['_score'])


# ── ML candidate generation (call once, reuse across all tiers) ───────────────

ML_CANDIDATES_CSV = 'E_ml_candidates.csv'


def build_ml_candidates(surrogate: 'MOSAICSurrogate',
                        df_doe: pd.DataFrame,
                        n_samples: int = 20,
                        cache_path: str = ML_CANDIDATES_CSV,
                        force_rebuild: bool = False) -> pd.DataFrame:
    """
    Generate ML-predicted candidates for every (Gate, Dielectric, Substrate)
    combo using LHS-sampled structural parameters.

    Results are cached to cache_path (default E_ml_candidates.csv).
    On subsequent runs the cached file is loaded instantly — no re-generation.
    Delete E_ml_candidates.csv (or set force_rebuild=True) to regenerate.

    Note: LHS uses a fixed seed (42), so candidates are identical every run
    unless n_samples changes or the surrogate model is retrained.
    """
    if not force_rebuild and os.path.exists(cache_path):
        print(f"\nLoading cached ML candidates from '{cache_path}'...")
        print("(Delete E_ml_candidates.csv to force regeneration)\n")
        return pd.read_csv(cache_path)

    ebd_map   = (df_doe.groupby('Dielectric')['EBD_HK'].first().to_dict()
                 if 'EBD_HK' in df_doe.columns else {})
    kappa_map = (df_doe.groupby('Dielectric')['kappa'].first().to_dict()
                 if 'kappa'  in df_doe.columns else {})
    cbo_map   = (df_doe.groupby('Dielectric')['CBO_HK'].first().to_dict()
                 if 'CBO_HK' in df_doe.columns else {})

    combos = df_doe[['Gate', 'Dielectric', 'Substrate']].drop_duplicates().values
    rng    = np.random.default_rng(42)
    rows   = []

    print(f"  Generating ML candidates: {n_samples} samples × {len(combos)} material combos "
          f"= {n_samples * len(combos)} predictions...")

    for gate, diel, sub in combos:
        gate, diel, sub = str(gate), str(diel), str(sub)
        il_name, _  = get_il_props(sub)
        ebd   = float(ebd_map.get(diel,   8.0))
        kappa = float(kappa_map.get(diel, 10.0))
        cbo   = float(cbo_map.get(diel,   1.5))

        raw    = rng.uniform(size=(n_samples, 6))
        lo = np.array([1.0,  0.0, 15.0,  20.0, 0.7, 300.0])
        hi = np.array([10.0, 1.0, 18.0, 100.0, 1.8, 400.0])
        scaled = lo + raw * (hi - lo)

        batch_inputs = []
        for s in scaled:
            batch_inputs.append(dict(
                Gate=gate, Dielectric=diel, Substrate=sub,
                t_HK_nm=float(s[0]),
                t_IL_nm=float(s[1]) if il_name is not None else 0.0,
                Nsub=float(10 ** s[2]),
                L_nm=float(s[3]), VDD=float(s[4]), T_K=float(s[5]),
                EBD_HK=ebd, CBO_HK=cbo, kappa=kappa,
            ))
        batch_df = pd.DataFrame(batch_inputs)
        preds_df = surrogate.predict_batch(batch_df)

        for idx in range(len(preds_df)):
            pred = preds_df.iloc[idx].to_dict()
            if 'error' in pred or pred.get('ION_uAum', 0) < 1.0:
                continue
            pred.update(batch_inputs[idx])
            rows.append(pred)

    if not rows:
        print("  No ML candidates generated.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"  Done — {len(df)} candidates saved to '{cache_path}'.")
    return df


# ── ML-powered recommendation (filters pre-built candidates) ─────────────────

def recommend_ml(surrogate: 'MOSAICSurrogate',
                 requirements: dict,
                 df_doe: pd.DataFrame,
                 top_n: int = 10,
                 n_samples: int = 20,
                 tier_label: str = '',
                 ml_candidates: pd.DataFrame = None) -> pd.DataFrame:
    """
    Filter and rank ML-predicted candidates by requirements.

    ml_candidates : pre-built DataFrame from build_ml_candidates().
        If None, candidates are generated here (convenient for single-tier use,
        but wasteful when called for multiple tiers — pass ml_candidates instead).
    """
    if ml_candidates is None:
        df = build_ml_candidates(surrogate, df_doe, n_samples)
    else:
        df = ml_candidates.copy()

    n_combos = len(df_doe[['Gate', 'Dielectric', 'Substrate']].drop_duplicates())

    if df.empty:
        print("\n  ML prediction returned no candidates.")
        return pd.DataFrame()

    n_total = len(df)

    filters_applied = []
    if 'min_ION' in requirements:
        df = df[df['ION_uAum'] >= requirements['min_ION']]
        filters_applied.append(f"ION >= {requirements['min_ION']} uA/um")
    if 'max_IG' in requirements:
        df = df[df['IG_Aum'] <= requirements['max_IG']]
        filters_applied.append(f"IG <= {requirements['max_IG']:.0e} A/um")
    if 'max_SS' in requirements:
        df = df[df['SS_mVdec'] <= requirements['max_SS']]
        filters_applied.append(f"SS <= {requirements['max_SS']} mV/dec")
    if 'max_EOT' in requirements:
        df = df[df['EOT_nm'] <= requirements['max_EOT']]
        filters_applied.append(f"EOT <= {requirements['max_EOT']} nm")
    if 'min_VTH' in requirements:
        df = df[df['VTH_V'] >= requirements['min_VTH']]
        filters_applied.append(f"VTH >= {requirements['min_VTH']} V")
    if 'max_VTH' in requirements:
        df = df[df['VTH_V'] <= requirements['max_VTH']]
        filters_applied.append(f"VTH <= {requirements['max_VTH']} V")
    if 'max_dT' in requirements:
        df = df[df['dT_K'] <= requirements['max_dT']]
        filters_applied.append(f"dT <= {requirements['max_dT']} K")
    if 'min_VBD' in requirements:
        df = df[df['VBD_V'] >= requirements['min_VBD']]
        filters_applied.append(f"VBD >= {requirements['min_VBD']} V")
    if 'min_Rel' in requirements:
        df = df[df['Rel_score'] >= requirements['min_Rel']]
        filters_applied.append(f"Rel >= {requirements['min_Rel']}")
    if 'max_Eox' in requirements:
        df = df[df['Eox_MVcm'] <= requirements['max_Eox']]
        filters_applied.append(f"Eox <= {requirements['max_Eox']} MV/cm")
    if 'min_Eox' in requirements:
        df = df[df['Eox_MVcm'] >= requirements['min_Eox']]
        filters_applied.append(f"Eox >= {requirements['min_Eox']} MV/cm")
    if 'max_Eox_frac_EBD' in requirements:
        frac = requirements['max_Eox_frac_EBD']
        df = df[df['Eox_MVcm'] <= frac * df['EBD_HK']]
        filters_applied.append(f"Eox <= {frac*100:.0f}% of EBD_HK (per dielectric)")

    n_matched = len(df)
    if n_matched == 0:
        print("\n  No ML-predicted gate stacks matched all requirements.")
        return pd.DataFrame()

    df['_score'] = (
        np.log10(df['ION_uAum'].clip(1, None)) * 0.35 +
        (-np.log10(df['IG_Aum'].clip(1e-30, None))) * 0.25 +
        (-df['SS_mVdec'] / 60) * 0.20 +
        df['Rel_score'] * 0.20
    )
    df = df.sort_values('_score', ascending=False).head(top_n)

    sep    = "=" * 62
    header = f"  ML RECOMMENDATIONS — {tier_label}" if tier_label else "  ML GATE STACK RECOMMENDATIONS"
    print(f"\n{sep}")
    print(header)
    print(f"  Source : surrogate.predict()  [{n_samples} LHS samples x {n_combos} material combos → {n_total} candidates]")
    print(sep)
    print(f"  Requirements : {' | '.join(filters_applied)}")
    print(f"  {n_matched} ML-predicted designs matched. Showing top {len(df)} ranked by score.")
    print(sep)

    def _s(v):
        _sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        return str(v).translate(_sub).encode('ascii', errors='replace').decode('ascii')

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        gate  = _s(row.get('Gate', '?'))
        diel  = _s(row.get('Dielectric', '?'))
        sub   = str(row.get('Substrate', '?'))
        t_hk  = float(row.get('t_HK_nm', float('nan')))
        t_il  = float(row.get('t_IL_nm', 0.0))
        l_nm  = float(row.get('L_nm', float('nan')))
        vdd   = float(row.get('VDD', float('nan')))
        nsub  = float(row.get('Nsub', float('nan')))
        t_k   = float(row.get('T_K', float('nan')))
        ion   = float(row.get('ION_uAum', float('nan')))
        ig    = float(row.get('IG_Aum', float('nan')))
        ss    = float(row.get('SS_mVdec', float('nan')))
        vth   = float(row.get('VTH_V', float('nan')))
        eot   = float(row.get('EOT_nm', float('nan')))
        vbd   = float(row.get('VBD_V', float('nan')))
        dt    = float(row.get('dT_K', float('nan')))
        rel   = float(row.get('Rel_score', float('nan')))
        eox   = float(row.get('Eox_MVcm', float('nan')))
        score = float(row.get('_score', float('nan')))
        ebd   = float(row.get('EBD_HK', float('nan')))

        il_label, _ = get_il_props(sub)
        il_str = f", IL={il_label} t_IL={t_il:.2f}nm" if (t_il > 0 and il_label is not None) else ", no IL"
        eox_str = (f"{eox:.3f} MV/cm ({eox/ebd*100:.1f}% of EBD={ebd:.1f} MV/cm)"
                   if ebd > 0 else f"{eox:.3f} MV/cm")
        dibl_mv  = row.get('DIBL_mVV',   float('nan'))
        lam_nm   = row.get('lambda_nm',  float('nan'))
        vth_long = row.get('VTH_long_V', float('nan'))
        t_junc   = row.get('T_junction_K', float('nan'))
        sce      = row.get('SCE_short', None)
        sce_str  = ('yes' if sce else 'no') if sce is not None else 'n/a'

        print(f"\n  Rank {rank}  [score={score:.3f}]  \u2190 ML prediction")
        print(f"    Gate Stack  : {gate}  /  {diel}  /  {sub}")
        if 'La' in diel:
            print(f"    *** WARNING : La2O3 is highly hygroscopic — absorbs moisture from air,")
            print(f"                  degrading interface quality and dielectric stability.")
            print(f"                  Requires hermetic encapsulation or capping layer.")
        print(f"    Oxide       : t_HK={t_hk:.2f}nm{il_str}")
        print(f"    Inputs      : L={l_nm:.1f}nm | VDD={vdd:.3f}V | Nsub={nsub:.2e} cm\u207b\u00b3 | T={t_k:.1f}K")
        print(f"    Performance : ION={ion:.1f} uA/um | SS={ss:.1f} mV/dec | VTH={vth:.3f}V (long={vth_long:.3f}V) | EOT={eot:.3f}nm")
        print(f"    SCE         : lambda={lam_nm:.1f}nm | DIBL={dibl_mv:.1f}mV/V | short-channel={sce_str}")
        print(f"    Leakage     : IG={ig:.3e} A/um")
        dit_flag = row.get('Dit_flag', 'n/a')
        print(f"    Oxide Field : Eox={eox_str}")
        print(f"    Reliability : VBD={vbd:.2f}V | dT={dt:.1f}K | T_junction={t_junc:.1f}K | Rel={rel:.3f} | Dit={dit_flag}")

    print(f"\n{sep}")
    return df.drop(columns=['_score'])


# ── Pareto front utility ──────────────────────────────────────────────────

def pareto_front_2d(df, obj1_col, obj2_col,
                    maximize_obj1=True, maximize_obj2=False):
    """Extract 2D Pareto front from DataFrame."""
    data = df[[obj1_col, obj2_col]].dropna().values
    idx  = df[[obj1_col, obj2_col]].dropna().index

    sign1 = 1 if maximize_obj1 else -1
    sign2 = 1 if maximize_obj2 else -1

    is_pareto = np.ones(len(data), dtype=bool)
    for i, p in enumerate(data):
        dominated = ((sign1 * data[:, 0] >= sign1 * p[0]) &
                     (sign2 * data[:, 1] >= sign2 * p[1]) &
                     ((sign1 * data[:, 0] > sign1 * p[0]) |
                      (sign2 * data[:, 1] > sign2 * p[1])))
        if dominated.any():
            is_pareto[i] = False

    return df.loc[idx[is_pareto]]


def plot_pareto_fronts(df_doe: pd.DataFrame, out_dir: str = '.'):
    """Plot Pareto fronts for the key trade-off pairs."""
    feas = df_doe[df_doe['feasible'] == True].copy()

    pairs = [
        ('ION_uAum', 'IG_Aum',    True,  False, 'ION vs IG'),
        ('ION_uAum', 'SS_mVdec',  True,  False, 'ION vs SS'),
        ('ION_uAum', 'dT_K',      True,  False, 'ION vs ΔT'),
        ('EOT_nm',   'ION_uAum',  False, True,  'EOT vs ION'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (c1, c2, m1, m2, title) in zip(axes, pairs):
        # For log-scale axes, set scale first and restrict to positive values
        plot_feas = feas.copy()
        if c2 == 'IG_Aum':
            ax.set_yscale('log')                      # must be set BEFORE scatter
            plot_feas = feas[feas[c2] > 0].copy()

        sub = plot_feas[[c1, c2]].dropna()
        if len(sub) < 5:
            continue
        pf = pareto_front_2d(plot_feas, c1, c2, m1, m2)

        ax.scatter(plot_feas[c1], plot_feas[c2], alpha=0.15, s=8,
                   color='steelblue', label='All feasible')
        ax.scatter(pf[c1], pf[c2], s=30, color='red', zorder=5,
                   edgecolors='darkred', linewidths=0.7, label='Pareto front')

        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle('Pareto Fronts — MOSAIC Gate Stack Design Space', y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, 'E_pareto_fronts.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Multi-objective Pareto (final design selection) ──────────────────────

def pareto_front_nd(values: np.ndarray) -> np.ndarray:
    """
    Find non-dominated (Pareto-optimal) rows.
    values : (n_points, n_objectives) — ALL objectives in MINIMISE convention.
    Returns boolean mask; True = non-dominated.
    """
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


def _plot_pareto_4obj(df_constrained: pd.DataFrame,
                      df_pareto: pd.DataFrame,
                      out_dir: str = '.',
                      fname: str = 'E_pareto_4obj.png',
                      title_suffix: str = ''):
    """2×2 grid of 2D projections of the 4-objective Pareto front."""
    pairs = [
        # (x_col,       y_col,       x_label,       y_label,      log_y)
        ('ION_uAum', 'IG_Aum',   'ION (\u03bcA/\u03bcm)', 'IG (A/\u03bcm)',   True),
        ('ION_uAum', 'SS_mVdec', 'ION (\u03bcA/\u03bcm)', 'SS (mV/dec)',       False),
        ('ION_uAum', 'dT_K',     'ION (\u03bcA/\u03bcm)', '\u0394T (K)',        False),
        ('SS_mVdec', 'dT_K',     'SS (mV/dec)',            '\u0394T (K)',        False),
    ]
    notes = [
        ('\u2192 max ION', '\u2193 min IG'),
        ('\u2192 max ION', '\u2193 min SS'),
        ('\u2192 max ION', '\u2193 min \u0394T'),
        ('\u2190 min SS',  '\u2193 min \u0394T'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (cx, cy, lx, ly, logy), (nx, ny) in zip(axes, pairs, notes):
        ax.scatter(df_constrained[cx], df_constrained[cy],
                   alpha=0.2, s=10, color='steelblue', label='Feasible (post-constraints)')
        ax.scatter(df_pareto[cx], df_pareto[cy],
                   s=70, color='red', zorder=5, edgecolors='darkred',
                   linewidths=0.7, label='Pareto-optimal')

        # Label each Pareto point with its design number
        for idx, row in df_pareto.iterrows():
            ax.annotate(
                f'#{idx + 1}',
                xy=(row[cx], row[cy]),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, color='darkred', fontweight='bold', zorder=6,
            )

        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(lx, fontsize=9)
        ax.set_ylabel(ly, fontsize=9)
        ax.set_title(f'{lx.split("(")[0].strip()} vs {ly.split("(")[0].strip()}',
                     fontsize=10)
        ax.text(0.97, 0.04, nx, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='darkgreen')
        ax.text(0.03, 0.96, ny, transform=ax.transAxes,
                ha='left',  va='top',    fontsize=8, color='darkgreen')
        ax.legend(fontsize=7, loc='upper left' if cx == 'SS_mVdec' else 'upper right')

    plt.suptitle(
        f'Constrained 4-Objective Pareto Front{title_suffix}\n'
        'Objectives: maximize ION  \u2502  minimize IG  '
        '\u2502  minimize SS  \u2502  minimize \u0394T',
        fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def run_pareto_analysis(requirements: dict,
                        df_doe: pd.DataFrame,
                        out_dir: str = '.',
                        top_n: int = None) -> pd.DataFrame:
    """
    Apply hard constraints from TARGET_REQUIREMENTS, then compute the
    4-objective Pareto front:
        maximize ION  |  minimize IG  |  minimize SS  |  minimize dT

    Supports all keys accepted by recommend(), plus:
        'max_Eox_frac_EBD' : float
            Eox <= frac x EBD_HK evaluated per row.
            e.g. 0.8 means the operating field must stay below 80% of
            that dielectric's breakdown field — automatically adapts
            to whichever dielectric is in each row.

    Returns DataFrame of Pareto-optimal final designs.
    """
    df = df_doe[df_doe['feasible'] == True].copy()

    # ── Apply hard constraints ──────────────────────────────────────
    filters = []
    if 'min_ION' in requirements:
        df = df[df['ION_uAum'] >= requirements['min_ION']]
        filters.append(f"ION >= {requirements['min_ION']} uA/um")
    if 'max_IG' in requirements:
        df = df[df['IG_Aum'] <= requirements['max_IG']]
        filters.append(f"IG <= {requirements['max_IG']:.0e} A/um")
    if 'max_SS' in requirements:
        df = df[df['SS_mVdec'] <= requirements['max_SS']]
        filters.append(f"SS <= {requirements['max_SS']} mV/dec")
    if 'max_EOT' in requirements:
        df = df[df['EOT_nm'] <= requirements['max_EOT']]
        filters.append(f"EOT <= {requirements['max_EOT']} nm")
    if 'min_VTH' in requirements:
        df = df[df['VTH_V'] >= requirements['min_VTH']]
        filters.append(f"VTH >= {requirements['min_VTH']} V")
    if 'max_VTH' in requirements:
        df = df[df['VTH_V'] <= requirements['max_VTH']]
        filters.append(f"VTH <= {requirements['max_VTH']} V")
    if 'max_dT' in requirements:
        df = df[df['dT_K'] <= requirements['max_dT']]
        filters.append(f"dT <= {requirements['max_dT']} K")
    if 'min_VBD' in requirements:
        df = df[df['VBD_V'] >= requirements['min_VBD']]
        filters.append(f"VBD >= {requirements['min_VBD']} V")
    if 'min_Rel' in requirements:
        df = df[df['Rel_score'] >= requirements['min_Rel']]
        filters.append(f"Rel >= {requirements['min_Rel']}")
    if 'max_Eox' in requirements:
        df = df[df['Eox_MVcm'] <= requirements['max_Eox']]
        filters.append(f"Eox <= {requirements['max_Eox']} MV/cm")
    if 'min_Eox' in requirements:
        df = df[df['Eox_MVcm'] >= requirements['min_Eox']]
        filters.append(f"Eox >= {requirements['min_Eox']} MV/cm")
    if 'max_Eox_frac_EBD' in requirements:
        frac = requirements['max_Eox_frac_EBD']
        df = df[df['Eox_MVcm'] <= frac * df['EBD_HK']]
        filters.append(f"Eox <= {frac*100:.0f}% of EBD_HK (per dielectric)")

    sep = '=' * 66
    print(f'\n{sep}')
    print('  MOSAIC FINAL DESIGN — CONSTRAINED 4-OBJECTIVE PARETO FRONT')
    print(sep)
    print('  Objectives : maximize ION  |  minimize IG  |  minimize SS  |  minimize dT')
    if filters:
        print('  Hard constraints:')
        for f in filters:
            print(f'    \u2022 {f}')
    print(f'  Feasible designs after constraints : {len(df)}')

    if len(df) == 0:
        print('  No designs satisfy all constraints — relax one or more requirements.')
        print(sep)
        return pd.DataFrame()

    # ── 4-objective matrix in MINIMISE convention ───────────────────
    # maximize ION  → minimise  -log10(ION)
    # minimize IG   → minimise   log10(IG)
    # minimize SS   → minimise   SS  (already in mV/dec)
    # minimize dT   → minimise   dT  (already in K)
    obj = np.column_stack([
        -np.log10(df['ION_uAum'].clip(1, None).values),
         np.log10(df['IG_Aum'].clip(1e-30, None).values),
        df['SS_mVdec'].values,
        df['dT_K'].values,
    ])

    mask      = pareto_front_nd(obj)
    df_pareto = (df[mask].copy()
                 .sort_values('ION_uAum', ascending=False)
                 .reset_index(drop=True))

    n_pareto  = len(df_pareto)
    if top_n is not None:
        df_pareto = df_pareto.head(top_n)

    print(f'  Pareto-optimal final designs       : {n_pareto}')
    if top_n is not None and top_n < n_pareto:
        print(f'  Showing top {top_n} (sorted by ION, highest first)')
    print(sep)

    def _s(v):
        _sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        return str(v).translate(_sub).encode('ascii', errors='replace').decode('ascii')

    for i, row in df_pareto.iterrows():
        il      = f"t_IL={row['t_IL_nm']:.2f} nm" if row['t_IL_nm'] > 0 else "no IL"
        ebd     = float(row.get('EBD_HK', float('nan')))
        eox     = float(row['Eox_MVcm'])
        eox_str = (f"{eox:.3f} MV/cm ({eox/ebd*100:.1f}% of EBD={ebd:.1f} MV/cm)"
                   if ebd > 0 else f"{eox:.3f} MV/cm")
        print(f"\n  Final design #{i+1}")
        print(f"    Stack      : {_s(row['Gate'])} / {_s(row['Dielectric'])} / {_s(row['Substrate'])}")
        print(f"    Oxide      : t_HK={row['t_HK_nm']:.2f} nm, {il}")
        print(f"    Inputs     : L={row['L_nm']:.1f}nm | VDD={row['VDD']:.3f}V | "
              f"Nsub={row['Nsub']:.2e} cm\u207b\u00b3 | T={row['T_K']:.1f}K")
        print(f"    ION        : {row['ION_uAum']:.1f} uA/um       \u2190 maximised")
        print(f"    IG         : {row['IG_Aum']:.3e} A/um    \u2190 minimised")
        print(f"    SS         : {row['SS_mVdec']:.2f} mV/dec        \u2190 minimised")
        t_junc   = float(row.get('T_junction_K', float('nan')))
        print(f"    dT         : {row['dT_K']:.2f} K (T_junction={t_junc:.1f} K)   \u2190 minimised")
        print(f"    Eox        : {eox_str}")
        dibl_mv  = float(row.get('DIBL_mVV',   float('nan')))
        lam_nm   = float(row.get('lambda_nm',  float('nan')))
        sce      = row.get('SCE_short', None)
        vth_long = float(row.get('VTH_long_V', float('nan')))
        sce_str  = ('yes' if sce else 'no') if sce is not None else 'n/a'
        dit_flag = row.get('Dit_flag', 'n/a')
        print(f"    EOT={row['EOT_nm']:.3f} nm | VTH={row['VTH_V']:.3f} V (long={vth_long:.3f} V) | "
              f"VBD={row['VBD_V']:.2f} V | Rel={row['Rel_score']:.3f} | Dit={dit_flag}")
        print(f"    SCE        : lambda={lam_nm:.1f}nm | DIBL={dibl_mv:.1f}mV/V | short-channel={sce_str}")

    print(f'\n{sep}')
    _plot_pareto_4obj(df, df_pareto, out_dir)
    return df_pareto


def run_pareto_analysis_ml(requirements: dict,
                           ml_candidates: pd.DataFrame,
                           out_dir: str = '.') -> pd.DataFrame:
    """
    Apply hard constraints from requirements to the ML candidate pool,
    compute the 4-objective Pareto front, print ALL Pareto-optimal designs,
    and save E_pareto_4obj_ml.png.
    """
    df = ml_candidates.copy()

    filters = []
    if 'min_ION' in requirements:
        df = df[df['ION_uAum'] >= requirements['min_ION']]
        filters.append(f"ION >= {requirements['min_ION']} uA/um")
    if 'max_IG' in requirements:
        df = df[df['IG_Aum'] <= requirements['max_IG']]
        filters.append(f"IG <= {requirements['max_IG']:.0e} A/um")
    if 'max_SS' in requirements:
        df = df[df['SS_mVdec'] <= requirements['max_SS']]
        filters.append(f"SS <= {requirements['max_SS']} mV/dec")
    if 'max_EOT' in requirements:
        df = df[df['EOT_nm'] <= requirements['max_EOT']]
        filters.append(f"EOT <= {requirements['max_EOT']} nm")
    if 'min_VTH' in requirements:
        df = df[df['VTH_V'] >= requirements['min_VTH']]
        filters.append(f"VTH >= {requirements['min_VTH']} V")
    if 'max_VTH' in requirements:
        df = df[df['VTH_V'] <= requirements['max_VTH']]
        filters.append(f"VTH <= {requirements['max_VTH']} V")
    if 'max_dT' in requirements:
        df = df[df['dT_K'] <= requirements['max_dT']]
        filters.append(f"dT <= {requirements['max_dT']} K")
    if 'min_VBD' in requirements:
        df = df[df['VBD_V'] >= requirements['min_VBD']]
        filters.append(f"VBD >= {requirements['min_VBD']} V")
    if 'min_Rel' in requirements:
        df = df[df['Rel_score'] >= requirements['min_Rel']]
        filters.append(f"Rel >= {requirements['min_Rel']}")
    if 'max_Eox_frac_EBD' in requirements:
        frac = requirements['max_Eox_frac_EBD']
        df = df[df['Eox_MVcm'] <= frac * df['EBD_HK']]
        filters.append(f"Eox <= {frac*100:.0f}% of EBD_HK (per dielectric)")

    sep = '=' * 66
    print(f'\n{sep}')
    print('  ML FINAL DESIGN — CONSTRAINED 4-OBJECTIVE PARETO FRONT')
    print(sep)
    print('  Objectives : maximize ION  |  minimize IG  |  minimize SS  |  minimize dT')
    print('  Source     : surrogate.predict() ML candidates')
    if filters:
        print('  Hard constraints:')
        for f in filters:
            print(f'    \u2022 {f}')
    print(f'  ML designs after constraints : {len(df)}')

    if len(df) == 0:
        print('  No ML designs satisfy all constraints.')
        print(sep)
        return pd.DataFrame()

    obj = np.column_stack([
        -np.log10(df['ION_uAum'].clip(1, None).values),
         np.log10(df['IG_Aum'].clip(1e-30, None).values),
        df['SS_mVdec'].values,
        df['dT_K'].values,
    ])
    mask      = pareto_front_nd(obj)
    df_pareto = (df[mask].copy()
                 .sort_values('ION_uAum', ascending=False)
                 .reset_index(drop=True))
    n_pareto  = len(df_pareto)

    print(f'  Pareto-optimal ML designs    : {n_pareto}')
    print(sep)

    def _s(v):
        _sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        return str(v).translate(_sub).encode('ascii', errors='replace').decode('ascii')

    for i, row in df_pareto.iterrows():
        il_label, _ = get_il_props(str(row.get('Substrate', '')))
        t_il   = float(row.get('t_IL_nm', 0.0))
        il_str = f"t_IL={t_il:.2f}nm" if t_il > 0 else "no IL"
        ebd    = float(row.get('EBD_HK', float('nan')))
        eox    = float(row.get('Eox_MVcm', float('nan')))
        eox_str = (f"{eox:.3f} MV/cm ({eox/ebd*100:.1f}% of EBD={ebd:.1f} MV/cm)"
                   if ebd > 0 else f"{eox:.3f} MV/cm")
        dibl_mv  = float(row.get('DIBL_mVV',   float('nan')))
        lam_nm   = float(row.get('lambda_nm',  float('nan')))
        vth_long = float(row.get('VTH_long_V', float('nan')))
        t_junc   = float(row.get('T_junction_K', float('nan')))

        print(f"\n  ML Final design #{i+1}")
        print(f"    Stack      : {_s(row['Gate'])} / {_s(row['Dielectric'])} / {_s(row['Substrate'])}")
        print(f"    Oxide      : t_HK={row['t_HK_nm']:.2f}nm, {il_str}")
        print(f"    Inputs     : L={row['L_nm']:.1f}nm | VDD={row['VDD']:.3f}V | "
              f"Nsub={row['Nsub']:.2e} cm\u207b\u00b3 | T={row['T_K']:.1f}K")
        print(f"    ION        : {row['ION_uAum']:.1f} uA/um       \u2190 maximised")
        print(f"    IG         : {row['IG_Aum']:.3e} A/um    \u2190 minimised")
        print(f"    SS         : {row['SS_mVdec']:.2f} mV/dec        \u2190 minimised")
        print(f"    dT         : {row['dT_K']:.2f} K (T_junction={t_junc:.1f} K)   \u2190 minimised")
        print(f"    Eox        : {eox_str}")
        dit_flag = row.get('Dit_flag', 'n/a')
        print(f"    EOT={row['EOT_nm']:.3f}nm | VTH={row['VTH_V']:.3f}V (long={vth_long:.3f}V) | "
              f"VBD={row['VBD_V']:.2f}V | Rel={row['Rel_score']:.3f} | Dit={dit_flag}")
        print(f"    SCE        : lambda={lam_nm:.1f}nm | DIBL={dibl_mv:.1f}mV/V")

    print(f'\n{sep}')
    _plot_pareto_4obj(df, df_pareto,
                      out_dir=out_dir,
                      fname='E_pareto_4obj_ml.png',
                      title_suffix=' (ML Predictions)')
    return df_pareto


# ── Main pipeline ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys as _sys
    import os as _os
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # ════════════════════════════════════════════════════════════════
    #  TIER TARGET REQUIREMENTS
    #  All four tiers are run in sequence.
    #  Edit any dict below to adjust the constraints for that tier.
    #  Remove a key entirely to leave that metric unconstrained.
    #
    #  Available keys:
    #    min_ION          : minimum drive current          (μA/μm)
    #    max_IG           : maximum gate leakage           (A/μm)
    #    max_SS           : maximum subthreshold swing     (mV/dec)
    #    max_EOT          : maximum equiv. oxide thickness (nm)
    #    min_VTH          : minimum threshold voltage      (V)
    #    max_VTH          : maximum threshold voltage      (V)
    #    max_dT           : maximum self-heating           (K)
    #    min_VBD          : minimum breakdown voltage      (V)
    #    min_Rel          : minimum reliability score      (0–1)
    #    max_Eox_frac_EBD : Eox <= frac × EBD_HK per row
    #                       e.g. 0.70 = must stay below 70% of that
    #                       dielectric's own breakdown field
    # ════════════════════════════════════════════════════════════════

    # ── Tier 1: Foundational Design ──────────────────────────────────
    TIER1_REQUIREMENTS = {
        'min_ION': 100,    # μA/μm
        'max_IG':  1e-6,   # A/μm
        'max_SS':  90,     # mV/dec
        'max_EOT': 1.5,    # nm
        'min_VTH': 0.3,    # V
        'max_VTH': 0.7,    # V
    }

    # ── Tier 2: Balanced Performance ─────────────────────────────────
    TIER2_REQUIREMENTS = {
        'min_ION': 300,    # μA/μm
        'max_IG':  1e-7,   # A/μm
        'max_SS':  80,     # mV/dec
        'max_EOT': 1.0,    # nm
        'min_VTH': 0.3,    # V
        'max_VTH': 0.6,    # V
    }

    # ── Tier 3: Reliability + Thermal Design ─────────────────────────
    TIER3_REQUIREMENTS = {
        'min_ION':          250,   # μA/μm
        'max_IG':           1e-7,  # A/μm
        'max_SS':           75,    # mV/dec
        'max_EOT':          0.9,   # nm
        'max_Eox_frac_EBD': 0.70,  # Eox <= 70% of EBD_HK per dielectric
        'min_VBD':          1.2,   # V
        'max_dT':           20,    # K
        # Reliability maximised via scoring (rel_weight=0.40 below)
    }

    # ── Tier 4: Pareto + Electro-Thermal Optimisation ────────────────
    TIER4_REQUIREMENTS = {
        'min_ION':          300,   # μA/μm
        'max_IG':           1e-8,  # A/μm
        'max_SS':           70,    # mV/dec
        'max_EOT':          0.7,   # nm
        'max_Eox_frac_EBD': 0.65,  # Eox <= 65% of EBD_HK per dielectric
        'max_dT':           15,    # K
        'min_VBD':          1.2,   # V
        # Pareto: maximise ION | minimise IG, SS, ΔT (run after recommend)
        # Reliability maximised via scoring (rel_weight=0.40 below)
    }

    TOP_N = 3   # how many recommendations to show per tier (user-editable)
    # ════════════════════════════════════════════════════════════════

    DOE_CSV   = 'D_gate_stack_database.csv'
    MODEL_PKL = 'E_surrogate_model.pkl'
    OUT_DIR   = '.'

    print("=" * 62)
    print("  MOSAIC Task E: Surrogate Modeling")
    print("=" * 62)

    # ── Load or train surrogate model ──────────────────────────────
    if _os.path.exists(MODEL_PKL):
        print(f"\nLoading saved surrogate model from '{MODEL_PKL}'...")
        print("(Delete E_surrogate_model.pkl to force retraining)\n")
        surrogate = MOSAICSurrogate.load(MODEL_PKL)
    else:
        print("\nNo saved model found — training surrogate model...")
        df = load_and_clean(DOE_CSV)

        targets_present = [t for t in ALL_TARGETS if t in df.columns]
        encoder   = FeatureEncoder()
        X = encoder.fit_transform(df)
        y = df[targets_present].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        scaler_X.fit(X_train)
        scaler_y.fit(y_train)

        print("\nTraining all surrogate models...")
        results = train_and_evaluate(X_train, X_test, y_train, y_test,
                                      targets_present, scaler_X, scaler_y)

        best_name  = max(results, key=lambda k: results[k]['mean_r2'])
        best_model = results[best_name]['model']
        print(f"\nBest model: {best_name}  (mean R²={results[best_name]['mean_r2']:.4f})")

        ys_train = scaler_y.transform(y_train)
        cross_validate_best(X_train, ys_train, best_model, scaler_X, k=5)

        print("\nComputing feature importances...")
        plot_feature_importance(best_model, encoder.feature_names, targets_present,
                                 best_name, OUT_DIR)

        ys_pred = best_model.predict(scaler_X.transform(X_test))
        y_pred  = scaler_y.inverse_transform(ys_pred)
        plot_parity(y_test, y_pred, targets_present, best_name, OUT_DIR)

        sensitivity_analysis(
            best_model, scaler_X.transform(X_test), y_test,
            targets_present, encoder.feature_names,
            scaler_X, scaler_y, OUT_DIR)

        print("\nGenerating Pareto front plots...")
        plot_pareto_fronts(pd.read_csv(DOE_CSV), OUT_DIR)  # raw CSV — IG_Aum in original units, not log-transformed

        print("\nSaving surrogate model...")
        surrogate = MOSAICSurrogate()
        surrogate.fit(df, model_name=best_name)
        surrogate.save(MODEL_PKL)
        print("Training complete. Model saved.")

    df_raw = pd.read_csv(DOE_CSV)

    # ── Part 1: CSV-based recommendations ─────────────────────────
    print("\n" + "=" * 62)
    print("  PART 1 — DATABASE SEARCH (pre-sampled DoE CSV)")
    print("=" * 62)

    recommend(surrogate, TIER1_REQUIREMENTS, df_raw,
              top_n=TOP_N, tier_label='Tier 1 — Foundational Design')
    recommend(surrogate, TIER2_REQUIREMENTS, df_raw,
              top_n=TOP_N, tier_label='Tier 2 — Balanced Performance')
    recommend(surrogate, TIER3_REQUIREMENTS, df_raw,
              top_n=TOP_N, tier_label='Tier 3 — Reliability + Thermal')
    recommend(surrogate, TIER4_REQUIREMENTS, df_raw,
              top_n=TOP_N, tier_label='Tier 4 — Pareto + Electro-Thermal')

    print("\nRunning Tier 4 constrained Pareto analysis (final design selection)...")
    run_pareto_analysis(TIER4_REQUIREMENTS, df_raw, OUT_DIR)

    # ── Part 2: ML-powered recommendations ────────────────────────
    print("\n" + "=" * 62)
    print("  PART 2 — ML PREDICTIONS (surrogate.predict())")
    print("  Explores full continuous structural parameter space")
    print("  — not limited to pre-sampled DoE points")
    print("=" * 62)

    # Generate candidates ONCE — all four tiers reuse the same pool
    df_ml = build_ml_candidates(surrogate, df_raw, n_samples=20)

    recommend_ml(surrogate, TIER1_REQUIREMENTS, df_raw,
                 top_n=TOP_N, n_samples=20,
                 tier_label='Tier 1 — Foundational Design',
                 ml_candidates=df_ml)
    recommend_ml(surrogate, TIER2_REQUIREMENTS, df_raw,
                 top_n=TOP_N, n_samples=20,
                 tier_label='Tier 2 — Balanced Performance',
                 ml_candidates=df_ml)
    recommend_ml(surrogate, TIER3_REQUIREMENTS, df_raw,
                 top_n=TOP_N, n_samples=20,
                 tier_label='Tier 3 — Reliability + Thermal',
                 ml_candidates=df_ml)
    print("\nRunning Tier 4 ML constrained Pareto analysis (all Pareto-optimal designs)...")
    run_pareto_analysis_ml(TIER4_REQUIREMENTS, df_ml, OUT_DIR)
