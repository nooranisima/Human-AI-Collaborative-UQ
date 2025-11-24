# src/llm/online_helpers.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import json
import math
import os
import re

import numpy as np
import pandas as pd

from . import offline_helpers as off


# ---------------------------------------------------------------------------
# Basic JSONL + age utilities
# ---------------------------------------------------------------------------

def _read_jsonl(path: str):
    """Simple JSONL reader, tolerant to garbage lines."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


# Very lightweight regexes for pulling ages from vignette text if needed
_AGE_NUM_KEYS = ("age", "Age", "age_years", "patient_age", "patient_age_years")
_AGE_RE       = re.compile(r"\bAge[:\s]*([0-9]{1,3})\b")
_YEARS_RE     = re.compile(r"([0-9]{1,3})\s*years?")


def _extract_age_years(rec: Dict[str, Any]) -> float:
    """Try numeric keys first, then vignette-style text."""
    # numeric keys
    for k in _AGE_NUM_KEYS:
        if k in rec:
            v = _to_float(rec[k])
            if not math.isnan(v):
                return v

    # text fields
    for k in ("vignette", "prompt", "prompt_text", "text"):
        s = rec.get(k)
        if isinstance(s, str):
            m = _AGE_RE.search(s) or _YEARS_RE.search(s)
            if m:
                v = _to_float(m.group(1))
                if not math.isnan(v):
                    return v
    return math.nan


def build_age_table(
    ai_jsonl: str,
    human_jsonl: str,
    ai_id_key: str = "subset_row_id",
) -> pd.DataFrame:
    """
    Build a table with columns:
        id, age_years, age_group, gt
    aligned on example ids present in BOTH AI and human files.
    """
    ai_rows = list(_read_jsonl(ai_jsonl))
    hu_rows = list(_read_jsonl(human_jsonl))

    # Guess id key for AI rows if needed
    if ai_rows and ai_id_key not in ai_rows[0]:
        for k in ["subset_row_id", "example_id", "id", "row_id"]:
            if k in ai_rows[0]:
                ai_id_key = k
                break

    ai_by_id = {r.get(ai_id_key): r for r in ai_rows if r.get(ai_id_key) is not None}

    # Try same key for human; else guess
    if hu_rows and ai_id_key in hu_rows[0]:
        hu_id_key = ai_id_key
    else:
        hu_id_key = ai_id_key
        if hu_rows and hu_id_key not in hu_rows[0]:
            for k in ["subset_row_id", "example_id", "id", "row_id"]:
                if k in hu_rows[0]:
                    hu_id_key = k
                    break

    hu_by_id = {r.get(hu_id_key): r for r in hu_rows if r.get(hu_id_key) is not None}

    ids = sorted(set(ai_by_id.keys()) & set(hu_by_id.keys()))
    rows = []
    for i in ids:
        ra = ai_by_id[i]
        rh = hu_by_id[i]

        age = _extract_age_years(ra)
        if math.isnan(age):
            age = _extract_age_years(rh)

        gt = ra.get("gt") or rh.get("gt")
        rows.append({"id": i, "age_years": _to_float(age), "gt": gt})

    df = pd.DataFrame(rows)

    # Age groups: 1–10, 10–20, …, 100+, and an "Unknown" bucket
    bins   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
    labels = [f"{i+1}–{i+10}" for i in range(0, 100, 10)] + ["100+"]

    df["age_group"] = pd.cut(
        df["age_years"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    df["age_group"] = df["age_group"].cat.add_categories(["Unknown"])
    df.loc[df["age_years"].isna(), "age_group"] = "Unknown"
    return df


def choose_calib_groups(
    df_age: pd.DataFrame,
    target_calib_n: int = 100,
    allowed_groups: Optional[List[str]] = None,
) -> List[str]:
    """
    Pick age groups for calibration greedily from youngest→oldest
    until reaching (approximately) target_calib_n examples.
    """
    order = [f"{i+1}–{i+10}" for i in range(0, 100, 10)] + ["100+"]

    if allowed_groups is not None:
        allowed = set(allowed_groups)
        order   = [g for g in order if g in allowed]

    counts = (
        df_age[df_age["age_group"] != "Unknown"]["age_group"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )

    picked: List[str] = []
    total = 0
    for g in order:
        c = int(counts.get(g, 0))
        if c == 0:
            continue
        picked.append(g)
        total += c
        if total >= target_calib_n:
            break
    return picked


# ---------------------------------------------------------------------------
# Conformal utilities (value thresholds)
# ---------------------------------------------------------------------------

def emp_quantile(scores: List[float], level: float) -> float:
    """Empirical quantile with 'higher' interpolation (CP-friendly)."""
    if not scores:
        return math.inf
    arr = np.asarray(scores, dtype=float)
    return float(np.quantile(arr, level, method="higher"))


@dataclass
class OnlineReplayResult:
    success: np.ndarray      # 1{Y in C_t}
    size:    np.ndarray      # |C_t|
    mask_inH: np.ndarray     # 1{Y in H_t}
    q_inH_trace: np.ndarray  # running q_inH values
    q_notH_trace: np.ndarray # running q_notH values


def replay_thresholds_values(
    ids: List[int],
    ai: off.AIModel,
    human: off.HumanExpert,
    label_space: List[str],
    strategy: str,
    eps: float,
    delta: float,
    q_inH0: float,
    q_notH0: float,
    adaptive: bool,
    eta: float = 0.05,
) -> OnlineReplayResult:
    """
    Replay CUP on a fixed test stream (ids) given initial VALUE thresholds q_inH0,q_notH0.
    If adaptive=True, update thresholds with a simple stochastic gradient step per round.
    """
    Lset = set(label_space)
    q_inH, q_notH = float(q_inH0), float(q_notH0)

    target_inH  = 1.0 - eps
    target_notH = float(delta)

    success: List[int] = []
    sizes:   List[int] = []
    mask_inH: List[int] = []
    q_inH_trace: List[float] = []
    q_notH_trace: List[float] = []

    for ex_id in ids:
        gt = ai.get_gt(ex_id) or human.get_gt(ex_id)
        if gt is None or gt not in Lset:
            success.append(0)
            sizes.append(0)
            mask_inH.append(0)
            q_inH_trace.append(q_inH)
            q_notH_trace.append(q_notH)
            continue

        H = human.predict_set(ex_id, strategy=strategy)
        p = ai.get_prob(ex_id)  # dict: label -> prob

        C: List[str] = []
        for lab in label_space:
            s = 1.0 - float(p.get(lab, 0.0))
            thr = q_inH if (lab in H) else q_notH
            if s <= thr:
                C.append(lab)

        y_in = int(gt in C)
        success.append(y_in)
        sizes.append(len(C))

        yinH = int(gt in H)
        mask_inH.append(yinH)

        s_true = 1.0 - float(p.get(gt, 0.0))

        if adaptive:
            if yinH:
                q_inH += eta * (target_inH - float(s_true <= q_inH))
            else:
                q_notH += eta * (target_notH - float(s_true <= q_notH))

        q_inH_trace.append(q_inH)
        q_notH_trace.append(q_notH)

    return OnlineReplayResult(
        success=np.asarray(success, dtype=int),
        size=np.asarray(sizes, dtype=int),
        mask_inH=np.asarray(mask_inH, dtype=int),
        q_inH_trace=np.asarray(q_inH_trace, dtype=float),
        q_notH_trace=np.asarray(q_notH_trace, dtype=float),
    )


# ---------------------------------------------------------------------------
# AI-only baseline (matched to HAI online coverage)
# ---------------------------------------------------------------------------

def _build_ai_alone_scores(ai: off.AIModel, calib_ids: List[int]) -> np.ndarray:
    scores: List[float] = []
    for ex_id in calib_ids:
        gt = ai.get_gt(ex_id)
        if gt is None:
            continue
        p = ai.get_prob(ex_id)
        scores.append(1.0 - float(p.get(gt, 0.0)))
    return np.asarray(scores, dtype=float)


def _ai_alone_replay_fixed(
    ids: List[int],
    ai: off.AIModel,
    q_value: float,
    label_space: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    succ: List[int] = []
    sizes: List[int] = []

    for ex_id in ids:
        gt = ai.get_gt(ex_id)
        if gt is None or gt not in label_space:
            succ.append(0)
            sizes.append(0)
            continue
        p = ai.get_prob(ex_id)
        C = [
            lab
            for lab in label_space
            if (1.0 - float(p.get(lab, 0.0))) <= q_value
        ]
        sizes.append(len(C))
        succ.append(int(gt in C))
    return np.asarray(succ, dtype=int), np.asarray(sizes, dtype=int)


def ai_alone_match_online_level(
    calib_ids: List[int],
    test_ids: List[int],
    ai: off.AIModel,
    label_space: List[str],
    target_cov: float,
    tol: float = 5e-4,
    maxit: int = 18,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Find an AI-only quantile level a* such that AI-alone coverage on test_ids
    roughly matches target_cov (the online CUP global coverage).
    Returns:
        a_star, success_stream, size_stream
    """
    S_cal = _build_ai_alone_scores(ai, calib_ids)
    if S_cal.size == 0:
        return 0.0, np.zeros(len(test_ids), dtype=int), np.zeros(len(test_ids), dtype=int)

    lo, hi = 0.0, 1.0
    best_a = 0.5
    best_succ = None

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        q = float(np.quantile(S_cal, mid, method="higher"))
        succ_mid, _ = _ai_alone_replay_fixed(test_ids, ai, q, label_space)
        cov = float(np.mean(succ_mid))
        best_a, best_succ = mid, succ_mid

        if abs(cov - target_cov) <= tol:
            break
        if cov < target_cov:
            lo = mid
        else:
            hi = mid

    q_star = float(np.quantile(S_cal, best_a, method="higher"))
    succ, sizes = _ai_alone_replay_fixed(test_ids, ai, q_star, label_space)
    return best_a, succ, sizes


# ---------------------------------------------------------------------------
# Main online run (age-based calibration + test stream)
# ---------------------------------------------------------------------------

def run_online_by_age(
    ai: off.AIModel,
    human: off.HumanExpert,
    label_space: List[str],
    df_age: pd.DataFrame,
    eps: float = 0.1,
    delta: float = 0.7,
    human_policy: str = "topk_2",
    target_calib_n: int = 1000,
    eta: float = 0.005,
) -> Dict[str, Any]:
    """
    1) Choose calibration age groups to total ~target_calib_n.
    2) Use those ids to estimate VALUE thresholds (q_inH0, q_notH0).
    3) Order remaining ids (test stream) by age group, then id.
    4) Replay fixed vs online CUP on this test stream.

    Returns a dict with:
        - 'config' (eps, delta, eta, human_policy, target_calib_n)
        - 'calib_groups', 'calib_ids'
        - 'test_ids', 'test_age_groups'
        - 'q_inH0', 'q_notH0'
        - 'fixed'  : OnlineReplayResult as dict
        - 'online' : OnlineReplayResult as dict
    """
    # Choose calibration age groups
    calib_groups = choose_calib_groups(df_age, target_calib_n=target_calib_n)

    is_calib = df_age["age_group"].isin(calib_groups)
    calib_ids = df_age[is_calib & df_age["gt"].notna()]["id"].tolist()
    test_df   = df_age[~is_calib & df_age["gt"].notna()].copy()

    # Order test stream by age_group (young→old → Unknown) and id for stability
    age_order = [f"{i+1}–{i+10}" for i in range(0, 100, 10)] + ["100+", "Unknown"]
    age_index = {g: i for i, g in enumerate(age_order)}

    test_df["age_order"] = test_df["age_group"].map(lambda g: age_index.get(str(g), len(age_order)))
    test_df = test_df.sort_values(["age_order", "id"])
    test_ids = test_df["id"].tolist()
    test_age_groups = test_df["age_group"].tolist()

    # Calibration conformal scores
    S_inH: List[float] = []
    S_notH: List[float] = []

    Lset = set(label_space)

    for ex_id in calib_ids:
        gt = ai.get_gt(ex_id) or human.get_gt(ex_id)
        if gt is None or gt not in Lset:
            continue
        H = human.predict_set(ex_id, strategy=human_policy)
        p = ai.get_prob(ex_id)
        s = 1.0 - float(p.get(gt, 0.0))
        if gt in H:
            S_inH.append(s)
        else:
            S_notH.append(s)

    if not S_inH or not S_notH:
        raise RuntimeError(
            f"Calibration empty: |S_inH|={len(S_inH)}, |S_notH|={len(S_notH)}. "
            "Try a different human_policy or increase target_calib_n."
        )

    # Initial VALUE thresholds
    q_notH0 = emp_quantile(S_notH, delta)
    q_inH0  = emp_quantile(S_inH, 1.0 - eps)

    # Replay fixed vs online on the ordered test stream
    fixed = replay_thresholds_values(
        test_ids, ai, human, label_space, human_policy,
        eps, delta, q_inH0, q_notH0,
        adaptive=False, eta=eta,
    )
    online = replay_thresholds_values(
        test_ids, ai, human, label_space, human_policy,
        eps, delta, q_inH0, q_notH0,
        adaptive=True, eta=eta,
    )

    return {
        "config": {
            "eps": eps,
            "delta": delta,
            "eta": eta,
            "human_policy": human_policy,
            "target_calib_n": target_calib_n,
        },
        "calib_groups": calib_groups,
        "calib_ids": calib_ids,
        "test_ids": test_ids,
        "test_age_groups": list(map(str, test_age_groups)),
        "q_inH0": q_inH0,
        "q_notH0": q_notH0,
        "fixed": {
            "success": fixed.success,
            "size": fixed.size,
            "mask_inH": fixed.mask_inH,
            "q_inH_trace": fixed.q_inH_trace,
            "q_notH_trace": fixed.q_notH_trace,
        },
        "online": {
            "success": online.success,
            "size": online.size,
            "mask_inH": online.mask_inH,
            "q_inH_trace": online.q_inH_trace,
            "q_notH_trace": online.q_notH_trace,
        },
    }


# ---------------------------------------------------------------------------
# Prepare data for plotting (no matplotlib here)
# ---------------------------------------------------------------------------

def _prefix_conditional_rate(mask: np.ndarray, success: np.ndarray) -> np.ndarray:
    """Prefix averages of 1{Y in C} restricted to where mask==1."""
    m = mask.astype(float)
    s = success.astype(float)
    num = np.cumsum(s * m)
    den = np.cumsum(m)
    out = np.full_like(num, np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _cum_mean(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    t = np.arange(1, len(x) + 1, dtype=float)
    return np.cumsum(x) / t


def _age_phase_bounds(test_age_groups: List[str]) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Given age_group sequence aligned with test stream, return:
        bounds: list of (start_t, end_t) 1-based segments of constant age_group
        labels: corresponding group labels as strings.
    """
    if not test_age_groups:
        return [], []

    bounds: List[Tuple[int, int]] = []
    labels: List[str] = []

    start = 1
    curr = test_age_groups[0]
    for i in range(1, len(test_age_groups)):
        if test_age_groups[i] != curr:
            bounds.append((start, i))  # inclusive indices in 1..T
            labels.append(str(curr))
            start, curr = i + 1, test_age_groups[i]
    bounds.append((start, len(test_age_groups)))
    labels.append(str(curr))
    return bounds, labels


def prepare_plot_data(
    res: Dict[str, Any],
    ai: off.AIModel,
    human: off.HumanExpert,
    label_space: List[str],
) -> Dict[str, Any]:
    """
    Prepare a single dict P with all traces needed for plotting:
      - prefix conditional coverage given Y∈H, Y∉H
      - cumulative marginal coverage for HAI/Human/AI
      - cumulative set size for HAI/Human/AI
      - age-phase boundaries
      - cumulative counts for masks (for burn-in decisions)
    """
    eps   = float(res["config"]["eps"])
    delta = float(res["config"]["delta"])

    test_ids  = res["test_ids"]
    age_groups = res["test_age_groups"]
    t = np.arange(1, len(test_ids) + 1)

    # Fixed & online streams
    suc_on = np.asarray(res["online"]["success"], dtype=int)
    suc_fx = np.asarray(res["fixed"]["success"], dtype=int)
    sz_on  = np.asarray(res["online"]["size"], dtype=float)
    sz_fx  = np.asarray(res["fixed"]["size"], dtype=float)
    mH_on  = np.asarray(res["online"]["mask_inH"], dtype=int)
    mH_fx  = np.asarray(res["fixed"]["mask_inH"], dtype=int)

    mNH_on = 1 - mH_on
    mNH_fx = 1 - mH_fx

    # Human-only baseline (on same test stream)
    succ_h: List[int] = []
    size_h: List[int] = []
    for ex_id in test_ids:
        gt = ai.get_gt(ex_id) or human.get_gt(ex_id)
        H  = human.predict_set(ex_id, strategy=res["config"]["human_policy"])
        succ_h.append(int(gt in H))
        size_h.append(len(H))
    succ_h = np.asarray(succ_h, dtype=int)
    size_h = np.asarray(size_h, dtype=float)

    # AI-only baseline matched to Online global coverage
    target_cov = float(np.mean(suc_on))
    a_match, succ_ai, size_ai = ai_alone_match_online_level(
        res["calib_ids"], test_ids, ai, label_space, target_cov
    )

    # Prefix conditional coverage (used for ε/δ panels)
    cov_on_inH  = _prefix_conditional_rate(mH_on,  suc_on)
    cov_fx_inH  = _prefix_conditional_rate(mH_fx,  suc_fx)
    cov_on_notH = _prefix_conditional_rate(mNH_on, suc_on)
    cov_fx_notH = _prefix_conditional_rate(mNH_fx, suc_fx)

    # Cumulative marginal coverage
    mc_on = _cum_mean(suc_on)
    mc_h  = _cum_mean(succ_h)
    mc_ai = _cum_mean(succ_ai)

    # Cumulative set size
    mz_on = _cum_mean(sz_on)
    mz_h  = _cum_mean(size_h)
    mz_ai = _cum_mean(size_ai)

    # Age-phase boundaries
    bounds, labels = _age_phase_bounds(age_groups)

    # Cumulative mask counts (for burn-in thresholds)
    mH_on_cum  = np.cumsum(mH_on)
    mH_fx_cum  = np.cumsum(mH_fx)
    mNH_on_cum = np.cumsum(mNH_on)
    mNH_fx_cum = np.cumsum(mNH_fx)

    return {
        "t": t,
        "eps": eps,
        "delta": delta,
        "test_ids": test_ids,
        "test_age_groups": age_groups,
        "a_match": a_match,
        # CUP streams
        "suc_on": suc_on,
        "suc_fx": suc_fx,
        "sz_on": sz_on,
        "sz_fx": sz_fx,
        "mH_on": mH_on,
        "mH_fx": mH_fx,
        "mNH_on": mNH_on,
        "mNH_fx": mNH_fx,
        # Human / AI baselines
        "succ_h": succ_h,
        "size_h": size_h,
        "succ_ai": succ_ai,
        "size_ai": size_ai,
        # prefix conditionals
        "cov_on_inH":  cov_on_inH,
        "cov_fx_inH":  cov_fx_inH,
        "cov_on_notH": cov_on_notH,
        "cov_fx_notH": cov_fx_notH,
        # cumulative means
        "mc_on": mc_on,
        "mc_h":  mc_h,
        "mc_ai": mc_ai,
        "mz_on": mz_on,
        "mz_h":  mz_h,
        "mz_ai": mz_ai,
        # age phases
        "bounds": bounds,
        "labels": labels,
        # supports
        "mH_on_cum":  mH_on_cum,
        "mH_fx_cum":  mH_fx_cum,
        "mNH_on_cum": mNH_on_cum,
        "mNH_fx_cum": mNH_fx_cum,
    }
