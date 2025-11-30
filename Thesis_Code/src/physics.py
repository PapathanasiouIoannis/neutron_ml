# src/physics.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sympy import Symbol, exp, lambdify
from src.const import CONSTANTS

# --- 1. EoS Library ---
def get_eos_library():
    p = Symbol('p')
    # Analytic Core Models
    core_exprs = {
        "APR-1": 0.000719964 * p**1.85898 + 108.975 * p**0.340074,
        "SLy": 0.0018 * p**1.5 + 120.0 * p**0.35,
        "MDI-2": 5.97365 * p**0.77374 + 89.24 * p**0.30993,
        "NLD": 119.05 + 304.8 * (1 - exp(-p/48.61465)) + 33722.34448 * (1 - exp(-p/17499.47411)),
        "W": 0.2618 * p**1.1685 + 92.49 * p**0.3077,
        "MS1": 4.1844 * p**0.81449 + 95.00135 * p**0.31736,
        "H4": 119.05 + 304.8 * (1 - exp(-p/48.6)),
        "HHJ-1": 1.78429 * p**0.93761 + 106.93652 * p**0.31715,
        "HHJ-2": 1.18961 * p**0.96539 + 108.40302 * p**0.31264,
        "Ska": 0.53928 * p**1.01394 + 94.31452 * p**0.35135,
        "SkI4": 4.75668 * p**0.76537 + 105.722 * p**0.2745,
        "HLPS-2": 161.553 + 172.858 * (1 - exp(-p/22.8644)) + 2777.75 * (1 - exp(-p/1909.97)),
        "HLPS-3": 81.5682 + 131.811 * (1 - exp(-p/4.41577)) + 924.143 * (1 - exp(-p/523.736)),
        "SCVBB": 0.371414 * p**1.08004 + 109.258 * p**0.351019,
        "WFF-1": 0.00127717 * p**1.69617 + 135.233 * p**0.331471,
        "WFF-2": 0.00244523 * p**1.62692 + 122.076 * p**0.340401,
        "PS": 1.69483 + 9805.95 * (1 - exp(-p * 0.000193624)) + 212.072 * (1 - exp(-p * 0.401508)),
        "BGP": 0.0112475 * p**1.59689 + 102.302 * p**0.335526,
        "BL-1": 0.488686 * p**1.01457 + 102.26 * p**0.355095,
        "BL-2": 1.34241 * p**0.910079 + 100.756 * p**0.354129,
        "DH": 39.5021 * p**0.541485 + 96.0528 * p**0.00401285,
        "MDI-3": 15.55 * p**0.666 + 76.71 * p**0.247,
        "MDI-4": 25.99587 * p**0.61209 + 65.62193 * p**0.15512,
    }
    crust_expr = 0.00873 + 103.17 * (1 - exp(-p/0.385))
    core_funcs = {k: (lambdify(p, e, 'numpy'), lambdify(p, e.diff(p), 'numpy')) for k, e in core_exprs.items()}
    crust_func = (lambdify(p, crust_expr, 'numpy'), lambdify(p, crust_expr.diff(p), 'numpy'))
    return core_funcs, crust_func

# --- 2. TOV Solver ---
def tov_rhs(r, y_state, eos_data, is_quark):
    m, P, y_tidal = y_state
    P_safe = max(P, 1e-10)
    
    if is_quark:
        B, cs2 = eos_data
        epsilon = (P_safe / cs2) + (4.0 * B)
        dedp = 1.0 / cs2
    else:
        fA_e, fA_de, fB_e, fB_de, w, crust_e, crust_de, alpha = eos_data
        if P_safe > CONSTANTS['P_TRANSITION']:
            P_base = P_safe / alpha
            valA, valB = fA_e(P_base), fB_e(P_base)
            if valA <= 0 or valB <= 0: return [0, 0, 0]
            epsilon_base = (valA**w) * (valB**(1.0-w))
            epsilon = epsilon_base * alpha
            dedpA, dedpB = fA_de(P_base), fB_de(P_base)
            termA = (w * dedpA / valA)
            termB = ((1.0-w) * dedpB / valB)
            dedp = epsilon_base * (termA + termB)
        else:
            epsilon = crust_e(P_safe)
            dedp = crust_de(P_safe)

    if dedp <= 0: return [0, 0, 0]
    cs2_local = 1.0 / dedp
    if cs2_local > 1.0: cs2_local = 1.0
    if cs2_local < 1e-5: cs2_local = 1e-5
    
    if r < 1e-4 or epsilon <= 0: return [0, 0, 0]

    term_1 = (epsilon + P)
    term_2 = (m + (r**3 * P * CONSTANTS['G_CONV']))
    term_3 = r * (r - 2.0 * m * CONSTANTS['A_CONV'])
    
    if abs(term_3) < 1e-5: return [0, 0, 0]

    dP_dr = -CONSTANTS['A_CONV'] * (term_1 * term_2) / term_3
    dm_dr = (r**2) * epsilon * CONSTANTS['G_CONV']

    exp_lambda = 1.0 / (1.0 - 2.0 * CONSTANTS['A_CONV'] * m / r)
    Q = CONSTANTS['A_CONV'] * CONSTANTS['G_CONV'] * (5.0*epsilon + 9.0*P + (epsilon+P)/cs2_local) * (r**2)
    Q -= 6.0 * exp_lambda
    F = (1.0 - CONSTANTS['A_CONV'] * CONSTANTS['G_CONV'] * (r**2) * (epsilon - P)) * exp_lambda
    dy_dr = -(y_tidal**2 + y_tidal * F + Q) / r

    return [dm_dr, dP_dr, dy_dr]

def solve_sequence(eos_input, is_quark):
    r_min = 1e-4
    pressures = np.logspace(1.5, 4.0, 60) if is_quark else np.logspace(-3, 3.8, 80)
    curve_data = []
    max_m = 0.0

    for pc in pressures:
        if is_quark:
            B, cs2 = eos_input
            eps_init = (pc/cs2) + 4*B
        else:
            fA_e, _, fB_e, _, w, crust_e, _, alpha = eos_input
            if pc > CONSTANTS['P_TRANSITION']:
                vA = fA_e(pc/alpha); vB = fB_e(pc/alpha)
                if vA <= 0 or vB <= 0: continue
                eps_init = ((vA**w) * (vB**(1.0-w))) * alpha
            else:
                eps_init = crust_e(pc)

        m_init = (r_min**3) * eps_init * (CONSTANTS['G_CONV'] / 3.0)
        y0 = [m_init, pc, 2.0]

        def surface_event(t, y): return y[1]
        surface_event.terminal = True; surface_event.direction = -1

        try:
            sol = solve_ivp(fun=lambda r, y: tov_rhs(r, y, eos_input, is_quark), t_span=(r_min, 30.0), y0=y0, events=surface_event, method='RK45', rtol=1e-5, atol=1e-8)
            if sol.status == 1 and len(sol.t_events[0]) > 0:
                R = sol.t_events[0][0]; M = sol.y_events[0][0][0]; yR = sol.y_events[0][0][2]
                C = (M * CONSTANTS['A_CONV']) / R
                if R < 3.0 or M < 0.1 or C >= 0.5: continue
                
                num = (8/5)*(1-2*C)**2 * C**5 * (2*C*(yR-1) - yR + 2)
                den = 2*C*(6-3*yR+3*C*(5*yR-8)) + 4*C**3*(13-11*yR+C*(3*yR-2)+2*C**2*(1+yR)) + 3*(1-2*C)**2*(2-yR+2*C*(yR-1))*np.log(1-2*C)
                if abs(den) < 1e-10: continue
                Lam = (2/3)*(num/den)*(C**-5)

                if M < max_m: break
                if M > max_m: max_m = M
                curve_data.append([M, R, Lam, pc])
        except: continue
    return curve_data, max_m

# --- 3. Workers ---
def calculate_baselines():
    print("--- Phase 0: Calculating Baselines ---")
    core_lib, crust_funcs = get_eos_library()
    baselines = {}
    for name in core_lib.keys():
        f = core_lib[name]
        eos_input = (f[0], f[1], f[0], f[1], 1.0, crust_funcs[0], crust_funcs[1], 1.0)
        _, max_m = solve_sequence(eos_input, is_quark=False)
        if max_m > 1.0: baselines[name] = max_m
    return baselines

def worker_hadronic_gen(batch_size, baselines, seed_offset, batch_idx):
    np.random.seed(seed_offset)
    core_lib, crust_funcs = get_eos_library()
    model_names = list(baselines.keys())
    valid_data = []
    attempts = 0
    while len(valid_data) < batch_size and attempts < batch_size * 50:
        attempts += 1
        nA, nB = np.random.choice(model_names, 2, replace=False)
        w = np.random.uniform(0.2, 0.8)
        base_max_m = w * baselines[nA] + (1-w) * baselines[nB]
        target_m = base_max_m * (1.0 + np.random.uniform(-0.15, 0.05))
        alpha = (target_m / base_max_m)**2
        
        fA = core_lib[nA]; fB = core_lib[nB]
        eos_input = (fA[0], fA[1], fB[0], fB[1], w, crust_funcs[0], crust_funcs[1], alpha)
        curve, max_m = solve_sequence(eos_input, False)

        if max_m < CONSTANTS['H_M_MAX_LOWER'] or max_m > CONSTANTS['H_M_MAX_UPPER']: continue
        try:
            c_arr = np.array(curve); c_arr = c_arr[c_arr[:,0].argsort()]
            if c_arr[0,0] > 1.4 or c_arr[-1,0] < 2.0: continue
            f_R = interp1d(c_arr[:,0], c_arr[:,1]); f_L = interp1d(c_arr[:,0], c_arr[:,2])
            if f_R(1.4) < CONSTANTS['H_R14_MIN'] or f_L(1.4) > CONSTANTS['H_L14_MAX'] or f_L(2.0) < CONSTANTS['H_L20_MIN']: continue
        except: continue

        for pt in curve:
            if pt[0] > 1.1 and pt[0] < max_m:
                valid_data.append([pt[0], pt[1], pt[2], 0, f"H_{batch_idx}_{attempts}"])
    return valid_data

def worker_quark_gen(batch_size, seed_offset, batch_idx):
    np.random.seed(seed_offset)
    valid_data = []
    attempts = 0
    while len(valid_data) < batch_size and attempts < batch_size * 50:
        attempts += 1
        B = np.random.uniform(*CONSTANTS['Q_B_RANGE'])
        cs2 = np.random.uniform(*CONSTANTS['Q_CS2_RANGE'])
        curve, max_m = solve_sequence((B, cs2), True)
        if max_m < 1.4 or max_m > CONSTANTS['H_M_MAX_UPPER']: continue
        if np.max(np.array(curve)[:,1]) > CONSTANTS['Q_R_MAX']: continue
        for pt in curve:
            if pt[0] > CONSTANTS['Q_M_MIN'] and pt[0] < max_m:
                valid_data.append([pt[0], pt[1], pt[2], 1, f"Q_{batch_idx}_{attempts}"])
    return valid_data

def worker_get_plot_curve(mode, baselines, seed):
    np.random.seed(seed)
    if mode == 'hadronic':
        core_lib, crust_funcs = get_eos_library()
        model_names = list(baselines.keys())
        for _ in range(50): 
            nA, nB = np.random.choice(model_names, 2, replace=False)
            w = np.random.uniform(0.2, 0.8)
            alpha = ((w * baselines[nA] + (1-w) * baselines[nB]) * (1.0 + np.random.uniform(-0.15, 0.05)) / (w * baselines[nA] + (1-w) * baselines[nB]))**2
            fA = core_lib[nA]; fB = core_lib[nB]
            eos_input = (fA[0], fA[1], fB[0], fB[1], w, crust_funcs[0], crust_funcs[1], alpha)
            curve, max_m = solve_sequence(eos_input, False)
            if max_m > CONSTANTS['H_M_MAX_LOWER'] and max_m < CONSTANTS['H_M_MAX_UPPER']:
                c = np.array(curve)
                try:
                    c = c[c[:,0].argsort()]
                    f_R = interp1d(c[:,0], c[:,1]); f_L = interp1d(c[:,0], c[:,2])
                    if f_R(1.4) >= CONSTANTS['H_R14_MIN'] and f_L(1.4) <= CONSTANTS['H_L14_MAX']: return c
                except: continue
    else:
        for _ in range(50):
            B = np.random.uniform(*CONSTANTS['Q_B_RANGE'])
            cs2 = np.random.uniform(*CONSTANTS['Q_CS2_RANGE'])
            curve, max_m = solve_sequence((B, cs2), True)
            if max_m > 1.4 and max_m < CONSTANTS['H_M_MAX_UPPER']:
                c = np.array(curve)
                if np.max(c[:,1]) <= CONSTANTS['Q_R_MAX']: return c
    return None

def run_worker_wrapper(task_tuple, baselines):
    mode, batch_size, seed, batch_idx = task_tuple
    if mode == 'hadronic': return worker_hadronic_gen(batch_size, baselines, seed, batch_idx)
    return worker_quark_gen(batch_size, seed, batch_idx)