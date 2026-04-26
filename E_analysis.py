"""
MOSAIC Task E: Dataset Analysis
Generates correlation matrix and distribution plots per material category.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DOE_CSV = 'D_gate_stack_database.csv'
OUT_DIR = '.'

# ── Load feasible data ────────────────────────────────────────────────────────
df = pd.read_csv(DOE_CSV)
df = df[df['feasible'] == True].copy()
print(f"Loaded {len(df)} feasible rows.")

# Subscript translator for plot labels
_sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
def _s(v): return str(v).translate(_sub)

KEY_INPUTS  = ['t_HK_nm', 't_IL_nm', 'Nsub', 'L_nm', 'VDD', 'T_K',
               'kappa', 'EBD_HK', 'CBO_HK']
KEY_OUTPUTS = ['VTH_V', 'ION_uAum', 'SS_mVdec', 'IG_Aum',
               'EOT_nm', 'Eox_MVcm', 'VBD_V', 'dT_K', 'Rel_score', 'DIBL_mVV']

# ── 1. Correlation Matrix ─────────────────────────────────────────────────────
print("\nGenerating correlation matrix...")

cols = [c for c in KEY_INPUTS + KEY_OUTPUTS if c in df.columns]
df_corr = df[cols].copy()

# Log-transform skewed columns for cleaner correlation
for col in ['ION_uAum', 'IG_Aum', 'Nsub']:
    if col in df_corr.columns:
        df_corr[col] = np.log10(df_corr[col].clip(1e-30))
        df_corr.rename(columns={col: f'log10({col})'}, inplace=True)

corr = df_corr.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.zeros_like(corr, dtype=bool)   # show full matrix (inputs vs outputs)
sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
    linewidths=0.4, annot_kws={'size': 7},
    vmin=-1, vmax=1, ax=ax,
    cbar_kws={'label': 'Pearson r'}
)
ax.set_title('Correlation Matrix — Inputs vs Outputs (feasible designs)', fontsize=12, pad=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
out = f'{OUT_DIR}/E_correlation_matrix.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")


# ── 2. Distribution Plots per Material ───────────────────────────────────────
METRICS = ['ION_uAum', 'SS_mVdec', 'EOT_nm', 'Rel_score', 'dT_K', 'VBD_V']
LOG_METRICS = {'ION_uAum', 'IG_Aum'}

def plot_distributions(df, group_col, metrics, fname, figsize=(18, 14)):
    """Violin + box plots for each metric, grouped by material category."""
    df = df.copy()
    # Translate subscript characters in material names
    df[group_col] = df[group_col].apply(_s)

    order = (df.groupby(group_col)['ION_uAum'].median()
               .sort_values(ascending=False).index.tolist())

    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            axes[i].set_visible(False)
            continue
        ax = axes[i]
        plot_df = df[[group_col, metric]].dropna().copy()

        if metric in LOG_METRICS:
            plot_df[metric] = np.log10(plot_df[metric].clip(1e-30))
            ylabel = f'log10({metric})'
        else:
            ylabel = metric

        sns.violinplot(data=plot_df, x=group_col, y=metric, order=order,
                       ax=ax, inner='box', palette='muted', cut=0)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Performance Distribution by {group_col}  '
                 f'(n={len(df)} feasible designs, sorted by median ION)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


print("\nGenerating distribution plots per material...")
plot_distributions(df, 'Gate',       METRICS, f'{OUT_DIR}/E_dist_by_gate.png')
plot_distributions(df, 'Dielectric', METRICS, f'{OUT_DIR}/E_dist_by_dielectric.png')
plot_distributions(df, 'Substrate',  METRICS, f'{OUT_DIR}/E_dist_by_substrate.png')

print("\nAll analysis plots generated.")

# ── 4. Sensitivity heatmap — continuous inputs only ───────────────────────────
print("\nGenerating cleaned sensitivity heatmap (continuous inputs only)...")

df_sens = pd.read_csv(f'{OUT_DIR}/E_sensitivity.csv', index_col=0)

# Keep only continuous input parameters — exclude categorical labels
EXCLUDE = ['Gate', 'Dielectric', 'Substrate']
df_sens = df_sens.drop(index=[r for r in EXCLUDE if r in df_sens.index])

# Rename inputs for readability
rename_inputs = {
    't_HK_nm':  't_HK (nm)',
    't_IL_nm':  't_IL (nm)',
    'Nsub_log': 'Nsub (log)',
    'L_nm':     'L (nm)',
    'VDD':      'VDD (V)',
    'T_K':      'T (K)',
    'EBD_HK':   'EBD_HK',
    'CBO_HK':   'CBO_HK',
    'kappa':    'kappa',
}
df_sens.index = [rename_inputs.get(r, r) for r in df_sens.index]

# Keep key output columns only
KEY_OUT = ['VTH_V', 'EOT_nm', 'SS_mVdec', 'Eox_MVcm', 'VBD_V',
           'dT_K', 'Rel_score', 'ION_uAum', 'IG_Aum', 'ION_IOFF',
           'DIBL_mVV', 'lambda_nm']
df_sens = df_sens[[c for c in KEY_OUT if c in df_sens.columns]]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df_sens, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.4, annot_kws={'size': 8},
            vmin=0, vmax=1,
            cbar_kws={'label': 'Normalised sensitivity (0=none, 1=max)'})
ax.set_title('Input-Output Sensitivity — Continuous Parameters Only (normalised)',
             fontsize=11, pad=10)
ax.set_xlabel('Output metrics', fontsize=9)
ax.set_ylabel('Input parameters', fontsize=9)
plt.xticks(rotation=40, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
out = f'{OUT_DIR}/E_sensitivity_continuous.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")
