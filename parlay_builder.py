
from itertools import combinations
import numpy as np
import pandas as pd

def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return (1 + 100/abs(odds)) if odds < 0 else (1 + odds/100.0)

def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be > 1.")
    return int(round((decimal_odds - 1) * 100)) if decimal_odds >= 2 else int(round(-100 / (decimal_odds - 1)))

def single_leg_ev(p_true: float, american_odds: int) -> float:
    dec = american_to_decimal(american_odds)
    return p_true * (dec - 1.0) - (1.0 - p_true)

def norm_ppf(p: np.ndarray) -> np.ndarray:
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                  1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                  6.680131188771972e+01, -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                  -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00])
    plow, phigh = 0.02425, 1 - 0.02425
    p = np.clip(p, 1e-12, 1-1e-12)
    q = p - 0.5
    r = np.zeros_like(p)
    mask_c = (p >= plow) & (p <= phigh)
    if np.any(mask_c):
        qc = q[mask_c]
        rc = qc * qc
        r[mask_c] = (((((a[0]*rc + a[1])*rc + a[2])*rc + a[3])*rc + a[4])*rc + a[5]) * qc / \
                     (((((b[0]*rc + b[1])*rc + b[2])*rc + b[3])*rc + b[4])*rc + 1)
    mask_l = p < plow
    if np.any(mask_l):
        ql = np.sqrt(-2.0*np.log(p[mask_l]))
        r[mask_l] = (((((c[0]*ql + c[1])*ql + c[2])*ql + c[3])*ql + c[4])*ql + c[5]) / \
                     ((((d[0]*ql + d[1])*ql + d[2])*ql + d[3])*ql + 1)
    mask_u = p > phigh
    if np.any(mask_u):
        qu = np.sqrt(-2.0*np.log(1.0 - p[mask_u]))
        r[mask_u] = -(((((c[0]*qu + c[1])*qu + c[2])*qu + c[3])*qu + c[4])*qu + c[5]) / \
                       ((((d[0]*qu + d[1])*qu + d[2])*qu + d[3])*qu + 1)
    return r

def joint_prob_gaussian_copula(p_list, R=None, n_sims=25000, seed=42) -> float:
    k = len(p_list)
    if k == 0:
        return 0.0
    if R is None:
        jp = 1.0
        for p in p_list:
            jp *= p
        return float(jp)
    R = np.array(R, dtype=float)
    if R.shape != (k, k):
        raise ValueError("Correlation matrix shape must be (k, k).")
    R = (R + R.T)/2.0
    try:
        L = np.linalg.cholesky(R + 1e-10*np.eye(k))
    except np.linalg.LinAlgError:
        np.fill_diagonal(R, 1.0)
        R = np.clip(R, -0.99, 0.99)
        L = np.linalg.cholesky(R + 1e-6*np.eye(k))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(k, n_sims))
    X = L @ Z
    thr = norm_ppf(np.array(p_list))
    hits = (X <= thr[:, None]).all(axis=0)
    return float(hits.mean())

def build_parlays(
    legs,
    max_legs=5,
    min_leg_ev=0.02,
    max_bad_price=-300,
    min_joint_p_true=0.05,
    forbid_same_game_non_sgp=True,
    correlation_matrix=None,
    n_sims_joint=25000
):
    legs_df = pd.DataFrame(legs).copy()
    required = {"id","description","odds","p_true"}
    if not required.issubset(legs_df.columns):
        raise ValueError(f"Legs missing required fields: {required - set(legs_df.columns)}")
    legs_df["single_ev"] = legs_df.apply(lambda r: single_leg_ev(r["p_true"], int(r["odds"])), axis=1)
    legs_df = legs_df[legs_df["odds"].astype(int) >= max_bad_price]
    legs_df = legs_df[legs_df["single_ev"] >= min_leg_ev].reset_index(drop=True)
    if legs_df.empty:
        return pd.DataFrame(columns=["combo_ids","combo_desc","n_legs","joint_p_true","parlay_odds_decimal","parlay_odds_american","ev","notes"])

    def assemble_R(ids):
        k = len(ids)
        R = np.eye(k)
        if correlation_matrix is None:
            return R
        for i in range(k):
            for j in range(i+1,k):
                rho = 0.0
                key = (ids[i], ids[j])
                key2 = (ids[j], ids[i])
                if key in correlation_matrix: rho = correlation_matrix[key]
                elif key2 in correlation_matrix: rho = correlation_matrix[key2]
                R[i,j]=R[j,i]=float(np.clip(rho,-0.9,0.9))
        return R

    results = []
    records = legs_df.to_dict("records")
    for r in range(2, max_legs+1):
        for combo in combinations(records, r):
            ids = [c["id"] for c in combo]
            descs = [c["description"] for c in combo]
            odds_list = [int(c["odds"]) for c in combo]
            p_list = [float(c["p_true"]) for c in combo]
            game_ids = [c.get("game_id") for c in combo]

            if forbid_same_game_non_sgp:
                games = [g for g in game_ids if g is not None]
                if len(set(games)) < len(games) and correlation_matrix is None:
                    continue

            dec_odds = float(np.prod([american_to_decimal(o) for o in odds_list]))
            R = assemble_R(ids)
            joint_p = joint_prob_gaussian_copula(p_list, R, n_sims=n_sims_joint)
            if joint_p < min_joint_p_true:
                continue
            ev = joint_p*(dec_odds-1.0) - (1.0 - joint_p)
            notes = []
            games = [g for g in game_ids if g is not None]
            if len(set(games)) < len(games):
                notes.append("same-game (uses correlation model)")
            results.append({
                "combo_ids": ids,
                "combo_desc": "; ".join(descs),
                "n_legs": r,
                "joint_p_true": joint_p,
                "parlay_odds_decimal": dec_odds,
                "parlay_odds_american": decimal_to_american(dec_odds),
                "ev": ev,
                "notes": ", ".join(notes)
            })
    if not results:
        return pd.DataFrame(columns=["combo_ids","combo_desc","n_legs","joint_p_true","parlay_odds_decimal","parlay_odds_american","ev","notes"])
    return pd.DataFrame(results).sort_values(["ev","joint_p_true"], ascending=[False, False]).reset_index(drop=True)
