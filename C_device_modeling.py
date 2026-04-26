"""
MOSAIC Task C: Device Modeling Framework
Physics-based MOSFET gate stack models (planar NMOS, with PMOS option).
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Physical constants ──────────────────────────────────────────────────────
q   = 1.602e-19   # C
eps0= 8.854e-12   # F/m
kB  = 1.381e-23   # J/K
h   = 6.626e-34   # J·s
m0  = 9.109e-31   # kg

# ── Inversion-layer effective mobilities [cm²/V·s] ─────────────────────────
# These reflect scattering in the inversion layer (lower than bulk values).
MU_INV = {
    # Effective inversion-layer mobilities [cm²/V·s].
    # III-V values are reduced vs bulk to reflect surface/remote-Coulomb scattering
    # at the high-k oxide interface in a nanoscale MOSFET.
    'Si':           {'n': 150,  'p': 70},
    'Strained Si':  {'n': 250,  'p': 120},
    'Ge':           {'n': 350,  'p': 220},
    'SiGe':         {'n': 200,  'p': 150},
    'GaAs':         {'n': 500,  'p': 150},
    'InGaAs':       {'n': 800,  'p': 100},   # limited by high-k interface scattering
    'InSb':         {'n': 1200, 'p': 200},
    'GaN':          {'n': 350,  'p': 15},
    'SiC (4H)':     {'n': 120,  'p': 50},
    'InP':          {'n': 600,  'p': 90},
    'Diamond':      {'n': 180,  'p': 250},
    'MoS\u2082':    {'n': 60,   'p': 30},
    'WS\u2082':     {'n': 90,   'p': 40},
    'Graphene':     {'n': 3000, 'p': 3000},
    'Black Phosphorus': {'n': 350, 'p': 250},
}

# ── IL pairing rules (from Sheet 7) ────────────────────────────────────────
SI_COMPATIBLE  = {'Si', 'Strained Si', 'SiGe'}
GE_COMPATIBLE  = {'Ge'}
GAN_COMPATIBLE = {'GaN'}
NO_IL_SUBSTRATES = {'GaAs', 'InGaAs', 'InP', 'InSb', 'SiC (4H)', 'Diamond',
                    'MoS\u2082', 'WS\u2082', 'Graphene', 'Black Phosphorus'}

IL_PROPS = {
    'SiOx': {
        'kappa': 3.9, 'Eg': 9.0, 'CBO': 3.5, 'VBO': 4.4,
        'EBD': 10.0, 'J_ref': 1e-11, 'Dit': 5e9
    },
    'Al2O3_Ge': {
        'kappa': 9.0, 'Eg': 8.8, 'CBO': 2.0, 'VBO': 3.8,
        'EBD': 9.0,  'J_ref': 1e-9,  'Dit': 5e10
    },
    'Ga2O3_GaN': {
        'kappa': 10.0, 'Eg': 4.9, 'CBO': 0.5, 'VBO': 2.1,
        'EBD': 6.0,  'J_ref': 1e-7,  'Dit': 2e10
    },
}

def get_il_props(substrate: str):
    """Return (il_name, il_dict) for the substrate, or (None, None)."""
    if substrate in SI_COMPATIBLE:
        return 'SiOx', IL_PROPS['SiOx']
    if substrate in GE_COMPATIBLE:
        return 'Al2O3_Ge', IL_PROPS['Al2O3_Ge']
    if substrate in GAN_COMPATIBLE:
        return 'Ga2O3_GaN', IL_PROPS['Ga2O3_GaN']
    return None, None   # No IL for III-V and wide-gap substrates


# ── Parse helper ────────────────────────────────────────────────────────────
def _parse_float(val, default=None):
    if val is None:
        return default
    s = str(val).replace('~', '').replace('≈', '').strip()
    if s.upper() in ('N/A', 'NONE', '', 'NAN'):
        return default
    try:
        return float(s)
    except Exception:
        return default


# ── Load materials database from Excel ─────────────────────────────────────
def load_materials_db(xlsx_path: str = 'MOSAIC_Materials_Database_1.xlsx'):
    wb_data = {}
    try:
        xl = pd.ExcelFile(xlsx_path)
        sheets = xl.sheet_names
    except Exception as e:
        raise FileNotFoundError(f"Cannot open {xlsx_path}: {e}")

    # Gate electrodes
    df_gate = xl.parse('1_Gate_Electrodes', header=1)
    df_gate.columns = [c.replace('\n', ' ') for c in df_gate.columns]
    gates = {}
    for _, row in df_gate.iterrows():
        mat = str(row.iloc[0]).strip()
        if not mat or mat.startswith('Gate') or mat.startswith('🟡'):
            continue
        try:
            gates[mat] = {
                'phi_M':     _parse_float(row.iloc[2], 4.5),
                'rho':       _parse_float(row.iloc[3], 50.0),
                'diffusivity': str(row.iloc[4]),
                'T_max':     _parse_float(row.iloc[5], 800),
                'CMOS_compat': str(row.iloc[7]),
                'type':      str(row.iloc[1]),
            }
        except Exception:
            continue
    wb_data['gates'] = gates

    # Dielectrics
    df_diel = xl.parse('2_Dielectrics', header=1)
    df_diel.columns = [c.replace('\n', ' ') for c in df_diel.columns]
    diels = {}
    for _, row in df_diel.iterrows():
        mat = str(row.iloc[0]).strip()
        if not mat or mat.startswith('Gate') or mat.startswith('CBO') or \
           mat.startswith('  SUBSTRATE') or mat.startswith('Al\u2082O\u2083 /') or \
           mat.startswith('Ga\u2082O\u2083 /') or mat.startswith('h-BN') or \
           mat.startswith('MoS\u2082 (diel)'):
            continue
        try:
            diels[mat] = {
                'kappa':  _parse_float(row.iloc[2], 10.0),
                'Eg':     _parse_float(row.iloc[3], 5.0),
                'CBO':    _parse_float(row.iloc[4], 1.5),
                'VBO':    _parse_float(row.iloc[5], 3.0),
                'EBD':    _parse_float(row.iloc[6], 8.0),
                'J_ref':  _parse_float(str(row.iloc[7]).replace('~', ''), 1e-7),
                'Dit':    _parse_float(str(row.iloc[8]).replace('~', ''), 1e11),
                'T_cryst':_parse_float(str(row.iloc[9]).replace('>', ''), 500),
                'category': str(row.iloc[1]),
            }
        except Exception:
            continue
    wb_data['diels'] = diels

    # Substrates
    df_sub = xl.parse('3_Substrates', header=1)
    df_sub.columns = [c.replace('\n', ' ') for c in df_sub.columns]
    subs = {}
    for _, row in df_sub.iterrows():
        mat = str(row.iloc[0]).strip()
        if not mat or mat.startswith('Substrate') or mat.startswith('Material'):
            continue
        ni_raw = str(row.iloc[6]).replace('~', '').strip()
        try:
            ni_val = _parse_float(ni_raw, 1.5e10)
            if ni_val is None or ni_val <= 0:
                ni_val = 1e-10
        except Exception:
            ni_val = 1e-10
        try:
            subs[mat] = {
                'Eg':    _parse_float(row.iloc[2], 1.12),
                'chi':   _parse_float(row.iloc[3], 4.05),
                'mu_n':  _parse_float(row.iloc[4], 1000),
                'mu_p':  _parse_float(row.iloc[5], 400),
                'ni':    ni_val,
                'kappa_th': _parse_float(row.iloc[7], 148),
                'Rth':   _parse_float(row.iloc[8], 7.5),
                'eps_r': _parse_float(row.iloc[9], 11.7),
                'vsat':  _parse_float(str(row.iloc[10]).replace('~', ''), 1e7),
                'type':  str(row.iloc[1]),
            }
        except Exception:
            continue
    wb_data['subs'] = subs

    return wb_data


# ── Core physics functions ──────────────────────────────────────────────────

def compute_COX(kappa_IL, t_IL_nm, kappa_HK, t_HK_nm):
    """Gate oxide capacitance [F/m²]. Series combination of IL + HK."""
    C_HK = eps0 * kappa_HK / (t_HK_nm * 1e-9)
    if t_IL_nm > 0:
        C_IL = eps0 * kappa_IL / (t_IL_nm * 1e-9)
        return C_IL * C_HK / (C_IL + C_HK)
    return C_HK


def compute_EOT(COX):
    """Equivalent oxide thickness [nm]."""
    return (3.9 * eps0 / COX) * 1e9


def compute_VTH(phi_M, chi_S, Eg_S, eps_r_S, Nsub, ni_S, COX, Dit_total, T=300):
    """
    Threshold voltage [V] for NMOS on p-type substrate.
    VTH = VFB + 2*phi_F + Qdep/COX
    """
    kT_q = kB * T / q
    # Guard against non-positive ni or Nsub/ni ratio
    ratio = max(Nsub / max(ni_S, 1.0), 1.0)
    phi_F = kT_q * np.log(ratio)       # Fermi potential [V] > 0 for p-type

    phi_S = chi_S + Eg_S / 2 + phi_F  # Semiconductor work function [eV→V]
    phi_MS = phi_M - phi_S             # Metal-semiconductor work function diff [V]

    # Fixed oxide charge from interface traps (simplified: Qox = q*Dit per cm²)
    Qox_Cm2 = q * Dit_total * 1e4     # C/m² (Dit in cm⁻²eV⁻¹, ΔE ~1 eV assumed)
    VFB = phi_MS - Qox_Cm2 / COX

    eps_s = eps_r_S * eps0
    Nsub_SI = Nsub * 1e6              # cm⁻³ → m⁻³
    Wdep = np.sqrt(2 * eps_s * 2 * phi_F / (q * Nsub_SI + 1e-30))
    Qdep = q * Nsub_SI * Wdep        # C/m²

    Cdep = eps_s / (Wdep + 1e-30)
    VTH = VFB + 2 * phi_F + Qdep / COX
    return VTH, phi_F, Cdep, Wdep


def compute_mu_eff(substrate, carrier='n', T=300, VOV=0.5, theta=0.15):
    """Effective inversion-layer mobility [cm²/V·s] with degradation and T scaling."""
    mu_inv = MU_INV.get(substrate, {}).get(carrier, 200)
    # Temperature scaling: μ ∝ T^(-1.5) for acoustic phonon (electrons in Si-like)
    alpha = 1.5 if carrier == 'n' else 1.0
    mu_T = mu_inv * (300 / max(T, 200)) ** alpha
    # Gate-field mobility degradation
    mu_eff = mu_T / (1 + theta * max(VOV, 0))
    return mu_eff


def compute_IDS_sat(mu_eff_cm2Vs, COX, L_nm, VGS, VTH, vsat_cms, W_um=1.0):
    """
    Saturation drive current per unit width [A/μm].
    Uses velocity-saturation limited model:
        Vdsat = VOV * Esat*L / (VOV + Esat*L)
        IDS = μ*COX/L * (VOV*Vdsat - Vdsat²/2)
    """
    VOV = VGS - VTH
    if VOV <= 0:
        return 0.0
    mu_SI = mu_eff_cm2Vs * 1e-4       # m²/Vs
    L_m  = L_nm * 1e-9
    L_cm = L_nm * 1e-7
    Esat_Vcm = 2 * vsat_cms / max(mu_eff_cm2Vs, 1)   # V/cm
    Esat_Vm  = Esat_Vcm * 1e2                          # V/m
    Vdsat = VOV * Esat_Vm * L_m / (VOV + Esat_Vm * L_m)
    IDS = mu_SI * COX / L_m * (VOV * Vdsat - 0.5 * Vdsat ** 2)  # A/m
    return IDS / W_um * W_um * 1e-6 / 1e-6  # already per m, convert to A/μm
    # Simpler: IDS [A/m] / 1 [m per μm] * 1e-6... let's be explicit:
    # IDS [A/m width] → A/μm = IDS * 1e-6
    # But W_um=1 → total A = IDS*W_m = IDS*1e-6; per μm = total/W_um = IDS*1e-6/1
    return IDS * 1e-6  # A/μm  (for W=1μm device, IDS [A/m] * 1e-6 m/μm)


def _compute_IDS_sat_clean(mu_eff_cm2Vs, COX, L_nm, VGS, VTH, vsat_cms):
    """Clean single-return saturation current [A/μm] (W=1μm normalised)."""
    VOV = VGS - VTH
    if VOV <= 0:
        return 0.0
    mu_SI   = mu_eff_cm2Vs * 1e-4
    L_m     = L_nm * 1e-9
    Esat_Vm = 2 * vsat_cms * 1e-2 / max(mu_eff_cm2Vs * 1e-4, 1e-10)  # V/m
    Vdsat   = VOV * Esat_Vm * L_m / (VOV + Esat_Vm * L_m)
    IDS_Am  = mu_SI * COX / L_m * (VOV * Vdsat - 0.5 * Vdsat ** 2)  # A/m (per m width)
    return IDS_Am * 1e-6   # A/μm (since 1 μm = 1e-6 m)


def compute_SS(COX, Cdep, T=300):
    """Subthreshold slope [mV/dec]."""
    SS_V = (kB * T / q) * np.log(10) * (1 + Cdep / max(COX, 1e-6))
    return SS_V * 1000   # mV/dec


def compute_gm(ION_Aum, VOV):
    """
    Saturation transconductance [μS/μm] = dIDS/dVGS ≈ IDS/VOV.
    Using IDS/VOV is consistent with the velocity-saturation IDS model.
    Returns value in μS/μm (ION in A/μm).
    """
    if VOV <= 0 or ION_Aum <= 0:
        return 0.0
    return ION_Aum / VOV * 1e6   # A/μm / V * 1e6 = μS/μm


def compute_leakage(J_ref_HK, EBD_HK, CBO_HK,
                    kappa_IL, t_IL_nm, J_ref_IL, EBD_IL,
                    kappa_HK, t_HK_nm,
                    VDD, L_nm):
    """
    Gate leakage current density J_G [A/cm²] and current IG [A/μm].
    Exponential field model calibrated to J_ref @ 1 MV/cm:
        J(E) = J_ref * exp(γ * (E - 1))
        γ = ln(J_BD / J_ref) / (EBD - 1),  J_BD = 1 A/cm²
    Voltage splits across IL and HK in series; leakage = min(J_IL, J_HK).
    """
    t_total_nm = t_HK_nm + t_IL_nm

    # Electric field through each layer (voltage divider by 1/C = t/ε)
    if t_IL_nm > 0 and kappa_IL > 0:
        C_IL = kappa_IL / t_IL_nm   # proportional (ε0 cancels)
        C_HK = kappa_HK / t_HK_nm
        C_tot = C_IL * C_HK / (C_IL + C_HK)
        V_HK = VDD * C_tot / C_HK
        V_IL = VDD * C_tot / C_IL
        E_HK = V_HK / (t_HK_nm * 0.1)  # MV/cm
        E_IL = V_IL / (t_IL_nm * 0.1)
    else:
        E_HK = VDD / (t_HK_nm * 0.1)
        E_IL = 0.0

    def _J(E_MV, J_r, E_BD):
        if E_MV <= 0:
            return 1e-30
        gamma = np.log(1.0 / max(J_r, 1e-30)) / max(E_BD - 1.0, 0.5)
        return J_r * np.exp(gamma * (E_MV - 1.0))

    J_HK = _J(E_HK, J_ref_HK, EBD_HK)
    if t_IL_nm > 0:
        J_IL = _J(E_IL, J_ref_IL, EBD_IL)
        J_G  = min(J_HK, J_IL)   # series: lower of the two limits total current
    else:
        J_G = J_HK

    J_G = np.clip(J_G, 1e-30, 1e6)
    IG  = J_G * L_nm * 1e-7   # A/μm  (J [A/cm²] × L [cm])
    return J_G, IG


def compute_natural_length(eps_r_S, W_dep_m, EOT_nm):
    """
    Electrostatic natural length λ [nm] for a single-gate planar bulk MOSFET.
        λ = √(εₛ × W_dep × EOT / ε_ox)
    Short-channel when L < 5λ.
    """
    eps_s  = eps_r_S * eps0
    eps_ox = 3.9 * eps0
    lam_m  = np.sqrt(eps_s * W_dep_m * (EOT_nm * 1e-9) / eps_ox)
    return lam_m * 1e9   # nm


def compute_DIBL_coeff(L_nm, lambda_nm):
    """
    DIBL coefficient η [V/V] from 2D Poisson exponential decay solution.
        η = 0.8 × exp(−π × L / (2λ))
    Acceptable limit: η ≤ 0.1 V/V (100 mV/V).
    V_TH_eff = V_TH_long − η × V_DS  (apply at V_DS = V_DD for saturation)
    """
    return 0.8 * np.exp(-np.pi * L_nm / (2.0 * max(lambda_nm, 1e-6)))


def compute_Eox(VDD, t_HK_nm, t_IL_nm):
    """Average oxide electric field [MV/cm]."""
    return VDD / ((t_HK_nm + t_IL_nm) * 0.1)


def compute_VBD(EBD_HK, t_HK_nm, t_IL_nm, EBD_IL=10.0):
    """Breakdown voltage [V]. Limited by the weaker dielectric layer."""
    t_total = t_HK_nm + t_IL_nm
    # For stacked dielectric, VBD is limited by the layer that breaks first
    VBD_HK = EBD_HK * t_HK_nm * 0.1
    if t_IL_nm > 0:
        VBD_IL = EBD_IL * t_IL_nm * 0.1
        return min(VBD_HK + VBD_IL, EBD_HK * t_total * 0.1)
    return VBD_HK


def compute_deltaT(ION_Aum, VDD, L_nm, kappa_th_WmK, W_um=1.0):
    """
    Self-heating temperature rise [K].
    Uses 3D spreading resistance model for a rectangular heat source on a half-space:
        Rth = 1 / (2 * kappa_th * sqrt(pi * L * W))
        DeltaT = P * Rth  where P = ION * VDD * W
    Gives physically grounded results across substrates and gate lengths.
    """
    L_m = L_nm * 1e-9
    W_m = W_um * 1e-6
    A_m2 = L_m * W_m
    Rth_KW = 1.0 / (2.0 * kappa_th_WmK * np.sqrt(np.pi * A_m2 + 1e-30))   # K/W
    P_W = ION_Aum * W_um * VDD    # total device power [W]
    return P_W * Rth_KW            # K


def compute_reliability_score(Eox, EBD, VBD, VBD_min, T_junction, Dit,
                               Dit_max=1e12, T_ref=300.0):
    """
    Composite reliability score in [0, 1]. Higher = more reliable.

    TDDB sub-score uses a coupled Eyring model that jointly captures oxide field
    and self-heated junction temperature acceleration:
        AF = exp(Ea/k * (1/T_ref - 1/T_junction)) * exp(beta * max(0, Eox/EBD - 0.3))
    where 0.3*EBD is the low-stress reference field baseline.
    score = 1 at reference conditions (300 K, Eox = 30% EBD), 0 at 100x acceleration.

    Weights: 0.40 * s_TDDB + 0.30 * s_VBD + 0.30 * s_Dit
    """
    k_eV  = 8.617e-5   # eV/K
    Ea    = 0.7        # eV  — activation energy for SiO2/high-k TDDB (JEDEC-typical)
    beta  = 3.0        # field acceleration coefficient, normalised to EBD (E-model)
    Eox_ref_frac = 0.3 # reference low-stress field = 30% of EBD

    T_j   = max(float(T_junction), 200.0)   # guard against unphysical values
    EBD_  = max(float(EBD), 1e-3)

    # Thermal acceleration vs 300 K (Arrhenius)
    AF_T  = np.exp((Ea / k_eV) * (1.0 / T_ref - 1.0 / T_j))
    # Field acceleration vs 30%-EBD baseline (E-model, normalised)
    AF_E  = np.exp(beta * max(0.0, Eox / EBD_ - Eox_ref_frac))
    # Combined TDDB acceleration factor; score = 0 at AF >= 100 (2 log-decades)
    AF    = AF_T * AF_E
    s_TDDB = max(0.0, 1.0 - np.log10(max(AF, 1.0)) / 2.0)

    s_VBD = min(1.0, VBD / max(VBD_min, 0.1))
    s_Dit = max(0.0, 1.0 - np.log10(max(Dit, 1.0)) / np.log10(max(Dit_max, 1.0)))

    return 0.4 * s_TDDB + 0.3 * s_VBD + 0.3 * s_Dit


def compute_CMOS_score(gate_props, diel_props, sub_props, t_HK_nm, t_IL_nm):
    """
    Qualitative CMOS process compatibility flags. Returns dict of pass/fail/score.
    """
    flags = {}
    T_max_gate = gate_props.get('T_max', 800)
    T_cryst    = diel_props.get('T_cryst', 500)
    compat     = gate_props.get('CMOS_compat', 'Good')

    flags['CMOS_process_pass']    = compat in ('Excellent', 'Good')
    flags['thermal_stability_pass'] = T_max_gate >= 400
    flags['crystallization_risk'] = T_cryst < 600  # true = at risk
    flags['manufacturability_pass'] = (0.5 <= t_HK_nm <= 10.0) and (t_IL_nm <= 1.5)
    flags['Dit_risk'] = diel_props.get('Dit', 1e11)
    flags['Dit_flag'] = 'Low' if flags['Dit_risk'] < 1e11 else \
                        'Med' if flags['Dit_risk'] < 1e12 else 'High'
    return flags


# ── Main evaluation function ────────────────────────────────────────────────

def evaluate_gate_stack(
    gate_name:  str,
    diel_name:  str,
    sub_name:   str,
    t_HK_nm:   float,
    t_IL_nm:   float,
    Nsub:      float,
    L_nm:      float,
    VDD:       float,
    T:         float = 300.0,
    carrier:   str   = 'n',
    db:        dict  = None,
):
    """
    Compute all device metrics for a given M-O-S gate stack configuration.

    Parameters
    ----------
    gate_name  : gate metal name (key in db['gates'])
    diel_name  : high-k dielectric name (key in db['diels'])
    sub_name   : substrate name (key in db['subs'])
    t_HK_nm   : high-k physical thickness [nm]
    t_IL_nm   : interfacial layer thickness [nm] (0 = no IL)
    Nsub      : channel doping [cm⁻³]
    L_nm      : gate length [nm]
    VDD       : supply voltage [V]
    T         : temperature [K]
    carrier   : 'n' (NMOS) or 'p' (PMOS)
    db        : materials database dict (from load_materials_db)

    Returns
    -------
    dict of computed metrics + feasibility flag.
    """
    if db is None:
        raise ValueError("Pass a materials database dict via the `db` argument.")

    gp = db['gates'].get(gate_name)
    dp = db['diels'].get(diel_name)
    sp = db['subs'].get(sub_name)
    if gp is None or dp is None or sp is None:
        return {'feasible': False, 'reason': 'Material not in database'}

    # ── IL selection based on substrate ──────────────────────────────────
    il_name, il_props = get_il_props(sub_name)
    if il_name is None:
        t_IL_nm = 0.0     # No IL for III-V / wide-gap substrates
        kappa_IL = dp['kappa']   # not used but needed as placeholder
        J_ref_IL = dp['J_ref']
        EBD_IL   = dp['EBD']
        Dit_IL   = dp['Dit']
    else:
        kappa_IL = il_props['kappa']
        J_ref_IL = il_props['J_ref']
        EBD_IL   = il_props['EBD']
        Dit_IL   = il_props['Dit']

    # Effective Dit = average of IL and HK contributions
    if t_IL_nm > 0 and il_name is not None:
        Dit_total = 0.5 * (il_props['Dit'] + dp['Dit'])
    else:
        Dit_total = dp['Dit']

    # ── Capacitance & EOT ────────────────────────────────────────────────
    COX = compute_COX(kappa_IL, t_IL_nm, dp['kappa'], t_HK_nm)
    EOT = compute_EOT(COX)

    # ── Threshold voltage ────────────────────────────────────────────────
    ni_raw = sp['ni']
    ni_S   = max(float(ni_raw) if ni_raw is not None else 1.5e10, 1.0)
    VTH, phi_F, Cdep, Wdep = compute_VTH(
        gp['phi_M'], sp['chi'], sp['Eg'],
        sp['eps_r'], Nsub, ni_S, COX, Dit_total, T
    )

    # ── Natural length & DIBL ────────────────────────────────────────────
    lambda_nm  = compute_natural_length(sp['eps_r'], Wdep, EOT)
    DIBL_coeff = compute_DIBL_coeff(L_nm, lambda_nm)
    DIBL_mVV   = DIBL_coeff * 1000.0          # mV/V for reporting
    SCE_short  = L_nm < 5.0 * lambda_nm       # short-channel flag

    # DIBL-corrected threshold (only shift when DIBL is within acceptable range)
    VTH_long = VTH
    VTH_eff  = VTH_long - DIBL_coeff * VDD    # V_TH_eff = V_TH_long − η·V_DS

    # ── Mobility ─────────────────────────────────────────────────────────
    # Use DIBL-corrected VTH for all downstream calculations
    VOV = VDD - VTH_eff
    mu_eff = compute_mu_eff(sub_name, carrier, T, max(VOV, 0))
    vsat   = sp.get('vsat', 1e7)
    if vsat is None:
        vsat = 1e7

    # ── Drive current ────────────────────────────────────────────────────
    ION = _compute_IDS_sat_clean(mu_eff, COX, L_nm, VDD, VTH_eff, float(vsat))

    # ── Subthreshold slope ───────────────────────────────────────────────
    SS = compute_SS(COX, Cdep, T)

    # ── Transconductance ─────────────────────────────────────────────────
    gm = compute_gm(ION, max(VOV, 0))

    # ── Gate leakage ─────────────────────────────────────────────────────
    J_G, IG = compute_leakage(
        dp['J_ref'], dp['EBD'], dp['CBO'],
        kappa_IL, t_IL_nm, J_ref_IL, EBD_IL,
        dp['kappa'], t_HK_nm,
        VDD, L_nm
    )

    # ── Reliability ──────────────────────────────────────────────────────
    Eox  = compute_Eox(VDD, t_HK_nm, t_IL_nm)
    VBD  = compute_VBD(dp['EBD'], t_HK_nm, t_IL_nm, EBD_IL if t_IL_nm > 0 else dp['EBD'])
    kappa_th = sp.get('kappa_th', 148) or 148
    dT          = compute_deltaT(ION, VDD, L_nm, float(kappa_th))
    T_junction  = T + dT   # self-heating fed back into reliability model
    rel         = compute_reliability_score(Eox, dp['EBD'], VBD, 1.2, T_junction, Dit_total)

    # ── CMOS flags ───────────────────────────────────────────────────────
    flags = compute_CMOS_score(gp, dp, sp, t_HK_nm, t_IL_nm)

    # ── ION/IOFF ratio (IOFF ≈ IG + subthreshold off-current) ───────────
    # Use DIBL-corrected VTH_eff for off-current (lower VTH → higher IOFF)
    I_sub_off = ION * np.exp(-max(VTH_eff, 0) / ((SS * 1e-3) / np.log(10))) if SS > 0 else 1e-15
    IOFF = max(IG, I_sub_off, 1e-20)
    ION_IOFF = ION / max(IOFF, 1e-30)

    # ── Intrinsic delay ──────────────────────────────────────────────────
    CGG = COX * (L_nm * 1e-9) * (1e-6)   # F/μm (W=1μm, L in m, COX in F/m²)
    tau = CGG * VDD / max(ION, 1e-30)     # s

    # ── Feasibility screening ────────────────────────────────────────────
    reasons = []
    if EOT > 10:
        reasons.append(f'EOT={EOT:.2f}nm > 10nm')
    if t_HK_nm > 10 or t_HK_nm < 0.5:
        reasons.append(f't_HK={t_HK_nm}nm out of range')
    if dp['CBO'] < 1.0:
        reasons.append(f'CBO={dp["CBO"]}eV < 1eV (excessive leakage risk)')
    if not flags['CMOS_process_pass']:
        reasons.append('CMOS process incompatible')
    if Eox > dp['EBD']:
        reasons.append(f'E_ox={Eox:.1f} > EBD={dp["EBD"]} MV/cm (breakdown)')
    if DIBL_coeff > 0.1:
        reasons.append(f'DIBL={DIBL_mVV:.0f}mV/V > 100mV/V (short-channel electrostatic failure)')

    feasible = len(reasons) == 0

    return {
        # Inputs echoed
        'Gate':     gate_name,
        'Dielectric': diel_name,
        'Substrate': sub_name,
        't_HK_nm':  t_HK_nm,
        't_IL_nm':  t_IL_nm,
        'Nsub':     Nsub,
        'L_nm':     L_nm,
        'VDD':      VDD,
        'T_K':      T,
        # Performance outputs
        'VTH_V':     round(VTH_eff, 4),        # DIBL-corrected threshold voltage
        'VTH_long_V':round(VTH_long, 4),       # long-channel (uncorrected) threshold
        'ION_uAum': round(ION * 1e6, 4),       # A/μm → μA/μm
        'SS_mVdec': round(SS, 2),
        'gm_uSum':  round(gm, 4),      # μS/μm
        'COX_Fm2':  round(COX, 4),
        'EOT_nm':   round(EOT, 4),
        'IG_Aum':   float(f'{IG:.4e}'),
        'J_G_Acm2': float(f'{J_G:.4e}'),
        # Reliability outputs
        'Eox_MVcm':    round(Eox, 3),
        'VBD_V':       round(VBD, 3),
        'dT_K':        round(dT, 3),
        'T_junction_K':round(T_junction, 2),
        'Rel_score':   round(rel, 4),
        'ION_IOFF': float(f'{ION_IOFF:.3e}'),
        'tau_ps':   round(tau * 1e12, 4),
        # Short-channel diagnostics
        'lambda_nm':  round(lambda_nm, 3),
        'SCE_short':  SCE_short,
        'DIBL_coeff': round(DIBL_coeff, 4),
        'DIBL_mVV':   round(DIBL_mVV, 2),
        # Flags
        'feasible': feasible,
        'infeasible_reasons': '; '.join(reasons),
        'CMOS_pass': flags['CMOS_process_pass'],
        'Dit_flag':  flags['Dit_flag'],
        'cryst_risk': flags['crystallization_risk'],
        'phi_F_V':   round(phi_F, 4),
        'Cdep_Fm2':  round(Cdep, 4),
    }


# ── evaluate_gate_stack_v2: thin alias kept for backward compatibility ───────
def evaluate_gate_stack_v2(gate_name, diel_name, sub_name,
                            t_HK_nm, t_IL_nm, Nsub, L_nm, VDD,
                            T=300.0, carrier='n', db=None):
    """Alias for evaluate_gate_stack (gm is now correctly computed inside)."""
    return evaluate_gate_stack(gate_name, diel_name, sub_name,
                               t_HK_nm, t_IL_nm, Nsub, L_nm, VDD,
                               T, carrier, db)


# ── Convenience: batch evaluate ─────────────────────────────────────────────
def batch_evaluate(configs: list, db: dict) -> pd.DataFrame:
    """
    Evaluate a list of gate stack config dicts. Returns a DataFrame.
    Each config dict: {gate, diel, sub, t_HK_nm, t_IL_nm, Nsub, L_nm, VDD, T, carrier}
    """
    results = []
    for cfg in configs:
        try:
            r = evaluate_gate_stack_v2(
                cfg['gate'], cfg['diel'], cfg['sub'],
                cfg.get('t_HK_nm', 3.0),
                cfg.get('t_IL_nm', 0.5),
                cfg.get('Nsub', 1e17),
                cfg.get('L_nm', 30.0),
                cfg.get('VDD', 1.0),
                cfg.get('T', 300.0),
                cfg.get('carrier', 'n'),
                db=db,
            )
        except Exception as e:
            r = {'feasible': False, 'reason': str(e)}
            r.update({k: cfg.get(k) for k in cfg})
        results.append(r)
    return pd.DataFrame(results)


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    xlsx = 'MOSAIC_Materials_Database_1.xlsx'
    db = load_materials_db(xlsx)
    print(f"Loaded: {len(db['gates'])} gates, {len(db['diels'])} dielectrics, "
          f"{len(db['subs'])} substrates\n")

    # Use ASCII-safe material names that match database keys
    test_configs = [
        dict(gate='TiN',            diel='HfO\u2082',       sub='Si',      t_HK_nm=3, t_IL_nm=0.5, Nsub=5e16, L_nm=30, VDD=1.0),
        dict(gate='TaN',            diel='HfSiON',          sub='Si',      t_HK_nm=2, t_IL_nm=0.5, Nsub=5e16, L_nm=20, VDD=0.8),
        dict(gate='TiN',            diel='Al\u2082O\u2083', sub='Ge',      t_HK_nm=4, t_IL_nm=0.5, Nsub=1e17, L_nm=30, VDD=1.0),
        dict(gate='W (Tungsten)',    diel='HfO\u2082',       sub='InGaAs',  t_HK_nm=3, t_IL_nm=0.0, Nsub=1e17, L_nm=20, VDD=0.7),
        dict(gate='Ru (Ruthenium)', diel='SiO\u2082',       sub='Si',      t_HK_nm=2, t_IL_nm=0,   Nsub=1e17, L_nm=50, VDD=1.0),
    ]

    df = batch_evaluate(test_configs, db)
    cols = ['Gate', 'Dielectric', 'Substrate', 't_HK_nm', 'EOT_nm', 'VTH_V',
            'ION_uAum', 'SS_mVdec', 'IG_Aum', 'Eox_MVcm', 'VBD_V', 'dT_K',
            'Rel_score', 'feasible']
    # Print with ASCII replacement for Windows compatibility
    out = df[cols].to_string(index=False)
    print(out.encode('ascii', errors='replace').decode('ascii'))
