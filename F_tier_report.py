"""
MOSAIC Gate Stack Tier Recommendations — PDF Report
Generates a clean side-by-side summary table for all 4 tiers (top 2 each).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ── Load and filter ───────────────────────────────────────────────────────────
df = pd.read_csv('D_gate_stack_database.csv')
feas = df[df['feasible'] == True].copy()

def _s(v):
    sub = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    return str(v).translate(sub).encode('ascii', 'replace').decode('ascii')

def apply_filters(d, req):
    if 'min_ION' in req: d = d[d['ION_uAum'] >= req['min_ION']]
    if 'max_IG'  in req: d = d[d['IG_Aum']   <= req['max_IG']]
    if 'max_SS'  in req: d = d[d['SS_mVdec'] <= req['max_SS']]
    if 'max_EOT' in req: d = d[d['EOT_nm']   <= req['max_EOT']]
    if 'min_VTH' in req: d = d[d['VTH_V']    >= req['min_VTH']]
    if 'max_VTH' in req: d = d[d['VTH_V']    <= req['max_VTH']]
    if 'max_dT'  in req: d = d[d['dT_K']     <= req['max_dT']]
    if 'min_VBD' in req: d = d[d['VBD_V']    >= req['min_VBD']]
    if 'max_Eox_frac_EBD' in req:
        d = d[d['Eox_MVcm'] <= req['max_Eox_frac_EBD'] * d['EBD_HK']]
    return d.copy()

def composite_score(d):
    return (
        np.log10(d['ION_uAum'].clip(1))      * 0.35 +
        (-np.log10(d['IG_Aum'].clip(1e-30))) * 0.25 +
        (-d['SS_mVdec'] / 60)                * 0.20 +
        d['Rel_score']                        * 0.20
    )

TIERS = {
    'Tier 1\nFoundational Design':   {'min_ION':100,  'max_IG':1e-6, 'max_SS':90, 'max_EOT':1.5, 'min_VTH':0.3, 'max_VTH':0.7},
    'Tier 2\nBalanced Performance':  {'min_ION':300,  'max_IG':1e-7, 'max_SS':80, 'max_EOT':1.0, 'min_VTH':0.3, 'max_VTH':0.6},
    'Tier 3\nReliability + Thermal': {'min_ION':250,  'max_IG':1e-7, 'max_SS':75, 'max_EOT':0.9, 'max_Eox_frac_EBD':0.70, 'min_VBD':1.2, 'max_dT':20},
    'Tier 4\nPareto + Electro-Thermal': {'min_ION':300, 'max_IG':1e-8, 'max_SS':70, 'max_EOT':0.7, 'max_Eox_frac_EBD':0.65, 'max_dT':15, 'min_VBD':1.2},
}

TIER_COLORS = ['#1a6faf', '#2e8b57', '#c25f00', '#8b1a1a']

def get_top2(req):
    d = apply_filters(feas, req)
    d['_score'] = composite_score(d)
    return d.sort_values('_score', ascending=False).head(2)

def format_row(r):
    il = f"SiOx {r['t_IL_nm']:.2f}nm" if r['t_IL_nm'] > 0 else 'None'
    vth_note = ' (depletion)' if r['VTH_V'] < 0 else ''
    return [
        f"{_s(r['Gate'])}",
        f"{_s(r['Dielectric'])}",
        f"{r['Substrate']}",
        f"{r['t_HK_nm']:.2f} nm",
        f"{il}",
        f"{r['L_nm']:.0f} nm",
        f"{r['VDD']:.2f} V",
        f"{r['Nsub']:.1e} cm⁻³",
        f"{r['ION_uAum']:.0f} μA/μm",
        f"{r['SS_mVdec']:.1f} mV/dec",
        f"{r['VTH_V']:.3f} V{vth_note}",
        f"{r['EOT_nm']:.3f} nm",
        f"{r['IG_Aum']:.2e} A/μm",
        f"{r['Eox_MVcm']:.2f} MV/cm",
        f"{r['VBD_V']:.2f} V",
        f"{r['dT_K']:.1f} K",
        f"{r['Rel_score']:.3f}",
    ]

ROW_LABELS = [
    'Gate Electrode', 'Dielectric', 'Substrate',
    't_HK', 'Interfacial Layer', 'Gate Length (L)', 'VDD', 'Nsub',
    'ION', 'SS', 'VTH', 'EOT',
    'IG', 'Eox', 'VBD', 'ΔT', 'Rel Score',
]

SECTION_BREAKS = {0: 'Gate Stack', 4: 'Structural Parameters', 8: 'Performance', 12: 'Reliability'}

# ── Build PDF ─────────────────────────────────────────────────────────────────
out_pdf = 'MOSAIC_Tier_Recommendations.pdf'

with PdfPages(out_pdf) as pdf:
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('white')

    ax_title = fig.add_axes([0.01, 0.93, 0.98, 0.06])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5,
                  'MOSAIC Gate Stack Recommendations — Top 2 Designs per Tier (Database Search)',
                  ha='center', va='center', fontsize=16, fontweight='bold', color='#1a1a2e')

    n_tiers = 4
    col_w   = 0.235
    col_gap = 0.013
    left0   = 0.01
    top     = 0.91
    ax_h    = 0.87

    axes = []
    for ti in range(n_tiers):
        left = left0 + ti * (col_w + col_gap)
        ax = fig.add_axes([left, top - ax_h, col_w, ax_h])
        axes.append(ax)

    tier_items = list(TIERS.items())

    for ti, (tier_label, req) in enumerate(tier_items):
        ax    = axes[ti]
        color = TIER_COLORS[ti]
        top2  = get_top2(req)
        n_matched = len(apply_filters(feas, req))

        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        n_rows   = len(ROW_LABELS)
        row_h    = 0.040
        header_h = 0.065
        label_h  = 0.055
        top_y    = 0.995

        # Tier header
        fancy = mpatches.FancyBboxPatch((0, top_y - label_h), 1, label_h,
                                         boxstyle='round,pad=0.01',
                                         facecolor=color, edgecolor='none',
                                         transform=ax.transAxes, clip_on=False)
        ax.add_patch(fancy)
        lines = tier_label.split('\n')
        ax.text(0.5, top_y - label_h/2, lines[0],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', transform=ax.transAxes)
        if len(lines) > 1:
            ax.text(0.5, top_y - label_h/2 - 0.022, lines[1],
                    ha='center', va='center', fontsize=8,
                    color='#e0e0e0', transform=ax.transAxes)

        # Matched count
        y_count = top_y - label_h - 0.025
        ax.text(0.5, y_count, f'{n_matched} designs matched',
                ha='center', va='center', fontsize=7.5,
                color='#555555', transform=ax.transAxes,
                style='italic')

        # Column headers Rank 1 / Rank 2
        y_header = y_count - 0.030
        col1_x, col2_x = 0.42, 0.72
        for cx, lbl in [(col1_x, 'Rank 1'), (col2_x, 'Rank 2')]:
            ax.text(cx, y_header, lbl,
                    ha='center', va='center', fontsize=8.5, fontweight='bold',
                    color=color, transform=ax.transAxes)
        ax.plot([0.02, 0.98], [y_header - 0.018, y_header - 0.018],
                color=color, linewidth=0.8, transform=ax.transAxes)

        # Data rows
        y_cur = y_header - 0.028
        rows_data = [format_row(r) for _, r in top2.iterrows()]

        for ri, label in enumerate(ROW_LABELS):
            # Section header
            if ri in SECTION_BREAKS:
                sec_label = SECTION_BREAKS[ri]
                ax.text(0.01, y_cur, sec_label,
                        ha='left', va='center', fontsize=7, fontweight='bold',
                        color=color, transform=ax.transAxes)
                ax.plot([0.01, 0.99], [y_cur - 0.008, y_cur - 0.008],
                        linewidth=0.4, color=color, alpha=0.5,
                        transform=ax.transAxes)
                y_cur -= 0.022

            # Alternating row background
            if ri % 2 == 0:
                bg = mpatches.FancyBboxPatch((0.01, y_cur - row_h * 0.55), 0.98, row_h * 0.95,
                                              boxstyle='square,pad=0',
                                              facecolor='#f5f7fa', edgecolor='none',
                                              transform=ax.transAxes, clip_on=True)
                ax.add_patch(bg)

            # Row label
            ax.text(0.03, y_cur - row_h * 0.05, label,
                    ha='left', va='center', fontsize=7, color='#333333',
                    transform=ax.transAxes)

            # Rank 1 value
            if len(rows_data) > 0:
                val1 = rows_data[0][ri]
                is_depl = 'depletion' in val1
                ax.text(col1_x, y_cur - row_h * 0.05, val1.replace(' (depletion)', ''),
                        ha='center', va='center', fontsize=7,
                        color='#cc0000' if is_depl else '#1a1a1a',
                        fontweight='bold' if is_depl else 'normal',
                        transform=ax.transAxes)

            # Rank 2 value
            if len(rows_data) > 1:
                val2 = rows_data[1][ri]
                is_depl = 'depletion' in val2
                ax.text(col2_x, y_cur - row_h * 0.05, val2.replace(' (depletion)', ''),
                        ha='center', va='center', fontsize=7,
                        color='#cc0000' if is_depl else '#1a1a1a',
                        fontweight='bold' if is_depl else 'normal',
                        transform=ax.transAxes)

            y_cur -= row_h

        # Divider between rank cols
        ax.plot([0.565, 0.565], [0.02, y_header - 0.01],
                color='#cccccc', linewidth=0.5, transform=ax.transAxes)

        # Border
        for spine_pos in ['top','bottom','left','right']:
            ax.spines[spine_pos].set_visible(False)
        rect = mpatches.FancyBboxPatch((0, top_y - ax_h * 0.995), 1, ax_h * 0.995,
                                        boxstyle='round,pad=0.005',
                                        facecolor='none',
                                        edgecolor=color, linewidth=1.2,
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        # Depletion warning if needed
        has_depl = any(r['VTH_V'] < 0 for _, r in top2.iterrows())
        if has_depl:
            ax.text(0.5, 0.008,
                    '⚠ Red VTH = depletion-mode (VTH < 0)',
                    ha='center', va='bottom', fontsize=6.5, color='#cc0000',
                    style='italic', transform=ax.transAxes)

    # Footer
    ax_foot = fig.add_axes([0.01, 0.005, 0.98, 0.02])
    ax_foot.axis('off')
    ax_foot.text(0.5, 0.5,
                 'Scoring: ION 35% · IG 25% · SS 20% · Rel 20%  |  '
                 'Source: D_gate_stack_database.csv (physics simulation, NMOS, W=1μm)  |  '
                 'MOSAIC Gate Stack Co-Design Project',
                 ha='center', va='center', fontsize=7, color='#888888')

    pdf.savefig(fig, bbox_inches='tight', dpi=200)
    plt.close(fig)

print(f'Saved: {out_pdf}')
