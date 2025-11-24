

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


# basic utilities


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate stray logging lines etc.
                continue
    return rows


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


# label space

def load_label_space(
    ai_jsonl_paths: Sequence[Path],
    human_jsonl_path: Path,
    allowed_labels_json: Optional[Path] = None,
) -> List[str]:
    """
    Build the canonical label space.
    """
    # 1) Explicit allowed labels file
    if allowed_labels_json is not None and allowed_labels_json.exists():
        with allowed_labels_json.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and "labels" in d:
            return list(d["labels"])
        if isinstance(d, list):
            return list(d)


    labs: set[str] = set()

    # AI files
    for p in ai_jsonl_paths:
        if not p.exists():
            continue
        for r in read_jsonl(p):
            # 'ranked': list of dicts with 'label'
            if isinstance(r.get("ranked"), list):
                for it in r["ranked"]:
                    lab = it.get("label")
                    if isinstance(lab, str):
                        labs.add(lab)
    
            if "gt" in r and r["gt"]:
                labs.add(str(r["gt"]))

    # Human file
    if human_jsonl_path.exists():
        for r in read_jsonl(human_jsonl_path):
            if "pred_top10" in r and isinstance(r["pred_top10"], list):
                labs.update(str(x) for x in r["pred_top10"] if x)
            if "gt" in r and r["gt"]:
                labs.add(str(r["gt"]))

    return sorted(labs)



class AIModel:
    """
    Wraps the AI JSONL file and exposes:
      - available_ids()
      - get_gt(example_id)
      - get_prob(example_id) -> {label: prob} over LABEL_SPACE.

    Expects each record to have a unique integer id (subset_row_id, example_id, etc.).
    """

    def __init__(self, ai_jsonl_path: Path, label_space: List[str]):
        self.path = ai_jsonl_path
        self.labels: List[str] = list(label_space)
        self.label_to_id: Dict[str, int] = {lab: i for i, lab in enumerate(self.labels)}
        self.id_to_label: Dict[int, str] = {i: lab for lab, i in self.label_to_id.items()}

        self._records: List[Dict[str, Any]] = read_jsonl(ai_jsonl_path)
        self._by_id: Dict[int, Dict[str, Any]] = {}
        self._gt: Dict[int, str] = {}
        self._id_key: str = self._guess_id_key(self._records)

        for r in self._records:
            ex_id = r.get(self._id_key)
            if ex_id is None:
                continue
            self._by_id[ex_id] = r
            if "gt" in r and r["gt"]:
                self._gt[ex_id] = str(r["gt"])

    # ---- internal helpers ----

    @staticmethod
    def _guess_id_key(rows: List[Dict[str, Any]]) -> str:
        for k in ("subset_row_id", "example_id", "id", "row_id"):
            if rows and k in rows[0]:
                return k
        if rows:
            for k, v in rows[0].items():
                if isinstance(v, int):
                    return k
        return "subset_row_id"

    def _prob_from_record(self, r: Dict[str, Any]) -> Dict[str, float]:
        L = self.labels
        nL = len(L)
        p = np.zeros(nL, dtype=float)

        # Case A: full probs as [[id, p], ...]
        if isinstance(r.get("probs"), list) and len(r["probs"]) > 0:
            for it in r["probs"]:
                try:
                    idx, pi = int(it[0]), safe_float(it[1])
                except Exception:
                    continue
                lab = self.id_to_label.get(idx)
                if lab is not None and lab in self.label_to_id:
                    p[self.label_to_id[lab]] += pi
            s = p.sum()
            if s > 0:
                p /= s
            return {lab: float(p[self.label_to_id[lab]]) for lab in L}

        # Case B: 'ranked' list of dicts with 'label'/'id'/'p'
        if isinstance(r.get("ranked"), list) and len(r["ranked"]) > 0:
            ranked = r["ranked"]
            remainder = safe_float(r.get("remainder_mass", 0.0))
            used_idx: set[int] = set()

            for it in ranked:
                lab = it.get("label")
                pi = safe_float(it.get("p", 0.0))

                if lab in self.label_to_id:
                    j = self.label_to_id[lab]
                    p[j] += pi
                    used_idx.add(j)
                elif "id" in it and isinstance(it["id"], int):
                    lab2 = self.id_to_label.get(it["id"])
                    if lab2 in self.label_to_id:
                        j = self.label_to_id[lab2]
                        p[j] += pi
                        used_idx.add(j)

            rest = [j for j in range(nL) if j not in used_idx]
            if remainder > 0.0 and rest:
                p[rest] += remainder / len(rest)

            s = p.sum()
            if s > 0:
                p /= s
            return {lab: float(p[self.label_to_id[lab]]) for lab in L}

        # Case C: fallback uniform over pred_top10
        if isinstance(r.get("pred_top10"), list) and len(r["pred_top10"]) > 0:
            top = [lab for lab in r["pred_top10"] if lab in self.label_to_id]
            if top:
                mass = 1.0 / len(top)
                for lab in top:
                    p[self.label_to_id[lab]] = mass
            return {lab: float(p[self.label_to_id[lab]]) for lab in L}

        # Otherwise, all zeros
        return {lab: 0.0 for lab in L}



    def available_ids(self) -> List[int]:
        return list(self._by_id.keys())

    def get_gt(self, example_id: int) -> Optional[str]:
        return self._gt.get(example_id)

    def get_prob(self, example_id: int) -> Dict[str, float]:
        r = self._by_id.get(example_id)
        if r is None:
            raise ValueError(f"AI record not found for id={example_id}")
        return self._prob_from_record(r)


# human expert


class HumanExpert:
    

    def __init__(
        self,
        human_jsonl_path: Path,
        label_space: List[str],
        id_to_label: Dict[int, str],
        rng: Optional[random.Random] = None,
    ):
        self.path = human_jsonl_path
        self._records: List[Dict[str, Any]] = read_jsonl(human_jsonl_path)
        self._by_id: Dict[int, Dict[str, Any]] = {}
        self._id_key: str = AIModel._guess_id_key(self._records)
        self._gt: Dict[int, str] = {}
        self._label_space: List[str] = list(label_space)
        self._id_to_label = dict(id_to_label)
        self._rng = rng if rng is not None else random

        for r in self._records:
            ex_id = r.get(self._id_key)
            if ex_id is None:
                continue
            self._by_id[ex_id] = r
            if "gt" in r and r["gt"]:
                self._gt[ex_id] = str(r["gt"])

    def available_ids(self) -> List[int]:
        return list(self._by_id.keys())

    def get_gt(self, example_id: int) -> Optional[str]:
        return self._gt.get(example_id)

    # ---- internal ----

    def _ranked_labels(self, r: Dict[str, Any]) -> List[str]:

        if isinstance(r.get("ranked"), list) and len(r["ranked"]) > 0:
            items = sorted(
                r["ranked"],
                key=lambda x: safe_float(x.get("p", 0.0)),
                reverse=True,
            )
            out: List[str] = []
            for it in items:
                lab = it.get("label")
                if lab:
                    out.append(lab)
                elif "id" in it and isinstance(it["id"], int):
                    lab2 = self._id_to_label.get(it["id"])
                    if lab2:
                        out.append(lab2)

            seen: set[str] = set()
            clean: List[str] = []
            for lab in out:
                if lab and lab in self._label_space and lab not in seen:
                    seen.add(lab)
                    clean.append(lab)
            return clean

        # Fallback: 'pred_top10'
        if isinstance(r.get("pred_top10"), list) and len(r["pred_top10"]) > 0:
            seen: set[str] = set()
            clean: List[str] = []
            for lab in r["pred_top10"]:
                if lab and lab in self._label_space and lab not in seen:
                    seen.add(lab)
                    clean.append(lab)
            return clean

        return []



    def predict_set(self, example_id: int, strategy: str = "topk_1") -> List[str]:
        r = self._by_id.get(example_id)
        if r is None:
            raise ValueError(f"Human record not found for id={example_id}")

        if strategy == "empty":
            return []

        if strategy == "all":
            return self._ranked_labels(r)

        if strategy.startswith("topk_"):
            try:
                k = int(strategy.split("_", 1)[1])
            except Exception:
                k = 1
            labs = self._ranked_labels(r)
            return labs[:k]

        # default: top-1
        return self._ranked_labels(r)[:1]


#  H/AI conformal prediction singlesplit

def calculate_thresholds(
    cal_ids: Iterable[int],
    ai_model: AIModel,
    human_expert: HumanExpert,
    epsilon: float,
    delta: float,
    human_strategy: str,
    jitter: float = 0.02,
) -> Tuple[float, float]:
    
    scores_in_human: List[float] = []
    scores_not_in_human: List[float] = []

    for ex_id in cal_ids:
        gt = ai_model.get_gt(ex_id) or human_expert.get_gt(ex_id)
        if not gt:
            continue

        try:
            probs = ai_model.get_prob(ex_id)
        except ValueError:
            continue

        p_true = float(probs.get(gt, 0.0))
        score = 1.0 - max(0.0, min(1.0, p_true))
        if jitter > 0.0:
            score += random.uniform(-jitter, jitter)

        Hx = set(human_expert.predict_set(ex_id, strategy=human_strategy))
        if gt in Hx:
            scores_in_human.append(score)
        else:
            scores_not_in_human.append(score)

   
    if scores_in_human:
        b = float(np.quantile(scores_in_human, 1.0 - epsilon))
    else:
        b = float("inf")

    if scores_not_in_human:
        a = float(np.quantile(scores_not_in_human, delta))
    else:
        a = float("inf")

    return a, b


def generate_prediction_sets(
    ids: Iterable[int],
    ai_model: AIModel,
    human_expert: HumanExpert,
    a_threshold: float,
    b_threshold: float,
    human_strategy: str,
    jitter: float = 0.02,
) -> Dict[int, Dict[str, Any]]:
    """
    Build H/AI prediction sets C(x) on the given ids.

    Scores: s(y|x) = 1 - p(y|x) (+ small jitter)

      - If y ∈ H(x): keep it if s <= b_threshold
      - If y ∉ H(x): add it if s <= a_threshold
    """
    out: Dict[int, Dict[str, Any]] = {}

    for ex_id in ids:
        probs = ai_model.get_prob(ex_id)  # {label: p}
        scores: Dict[str, float] = {}
        for lab, p in probs.items():
            s = 1.0 - max(0.0, min(1.0, float(p)))
            if jitter > 0.0:
                s += random.uniform(-jitter, jitter)
            scores[lab] = s

        H = set(human_expert.predict_set(ex_id, strategy=human_strategy))
        C: set[str] = set()

        # Keep human labels if score <= b
        for lab in H:
            if scores.get(lab, 1.0) <= b_threshold:
                C.add(lab)

        # Rescue AI labels outside H if score <= a
        for lab, s in scores.items():
            if lab not in H and s <= a_threshold:
                C.add(lab)

        out[ex_id] = {
            "prediction_set": sorted(C),
            "set_size": len(C),
        }

    return out


# metrics + AI-alone 


def human_alone_metrics(
    ids: Iterable[int],
    human: HumanExpert,
    ai: AIModel,
    strategy: str,
) -> Tuple[float, float, int]:
    """Coverage / avg size for the human-only set H(x)."""
    hits = 0
    sizes: List[int] = []
    n_with_gt = 0

    for ex_id in ids:
        gt = ai.get_gt(ex_id) or human.get_gt(ex_id)
        if not gt:
            continue
        Hx = human.predict_set(ex_id, strategy=strategy)
        sizes.append(len(Hx))
        hits += int(gt in Hx)
        n_with_gt += 1

    cov = hits / max(n_with_gt, 1)
    avg_sz = float(np.mean(sizes)) if sizes else 0.0
    return cov, avg_sz, n_with_gt


def eval_sets(
    preds: Dict[int, Dict[str, Any]],
    ai: AIModel,
    human: Optional[HumanExpert],
) -> Tuple[float, float, int]:
    """
    Evaluate coverage / avg size for a dict of prediction sets.
    """
    hits = 0
    sizes: List[int] = []
    n_with_gt = 0

    for ex_id, rec in preds.items():
        gt = ai.get_gt(ex_id)
        if not gt and human is not None:
            gt = human.get_gt(ex_id)
        if not gt:
            continue

        sizes.append(rec["set_size"])
        hits += int(gt in rec["prediction_set"])
        n_with_gt += 1

    cov = hits / max(n_with_gt, 1)
    avg_sz = float(np.mean(sizes)) if sizes else 0.0
    return cov, avg_sz, n_with_gt


def ai_alone_at_coverage(
    cal_ids: Iterable[int],
    test_ids: Iterable[int],
    ai: AIModel,
    target_cov: float,
    delta_grid: Iterable[float] = np.linspace(0.01, 0.99, 99),
) -> Tuple[float, float, Optional[float]]:
    """
    AI-only baseline tuned (on CAL) to match the target coverage (on TEST).

    We treat all points as Y ∉ H(x), so thresholds come from a single
    score distribution s = 1 - p_true over CAL.
    """
    cal_scores: List[float] = []
    for ex_id in cal_ids:
        gt = ai.get_gt(ex_id)
        if not gt:
            continue
        probs = ai.get_prob(ex_id)
        p_true = float(probs.get(gt, 0.0))
        cal_scores.append(1.0 - max(0.0, min(1.0, p_true)))

    if not cal_scores:
        return 0.0, 0.0, None

    cal_scores_arr = np.asarray(cal_scores, dtype=float)

    best_delta: Optional[float] = None
    best_gap = float("inf")
    best_cov = 0.0
    best_size = 0.0

    for delta in delta_grid:
        a = float(np.quantile(cal_scores_arr, delta))

        preds: Dict[int, Dict[str, Any]] = {}
        for ex_id in test_ids:
            probs = ai.get_prob(ex_id)
            C = [lab for lab, p in probs.items() if (1.0 - float(p)) <= a]
            preds[ex_id] = {"prediction_set": C, "set_size": len(C)}

        cov, avg_sz, _ = eval_sets(preds, ai, human=None)
        gap = abs(cov - target_cov)
        if gap < best_gap:
            best_gap = gap
            best_delta = float(delta)
            best_cov = float(cov)
            best_size = float(avg_sz)

    return best_cov, best_size, best_delta


# single offline split for a given (ε, δ, strategy)


@dataclass
class OfflineResult:
    human_cov: float
    human_sz: float
    hai_cov: float
    hai_sz: float
    ai_cov: float
    ai_sz: float
    a: float
    b: float
    tuned_delta: Optional[float]
    n_test_with_gt: int


def run_single_split(
    ai: AIModel,
    human: HumanExpert,
    ids: Sequence[int],
    human_strategy: str,
    epsilon: float,
    delta: float,
    test_size: float = 0.5,
    random_state: int = 1337,
    jitter: float = 0.02,
) -> OfflineResult:
    
    cal_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=random_state
    )

    # Human alone
    human_cov, human_sz, _ = human_alone_metrics(test_ids, human, ai, strategy=human_strategy)

    # H/AI
    a, b = calculate_thresholds(
        cal_ids,
        ai_model=ai,
        human_expert=human,
        epsilon=epsilon,
        delta=delta,
        human_strategy=human_strategy,
        jitter=jitter,
    )
    preds_hai = generate_prediction_sets(
        test_ids,
        ai_model=ai,
        human_expert=human,
        a_threshold=a,
        b_threshold=b,
        human_strategy=human_strategy,
        jitter=jitter,
    )
    hai_cov, hai_sz, n_test = eval_sets(preds_hai, ai, human)

    # AI-only tuned to CUP coverage
    ai_cov, ai_sz, tuned_delta = ai_alone_at_coverage(
        cal_ids, test_ids, ai, target_cov=hai_cov
    )

    return OfflineResult(
        human_cov=human_cov,
        human_sz=human_sz,
        hai_cov=hai_cov,
        hai_sz=hai_sz,
        ai_cov=ai_cov,
        ai_sz=ai_sz,
        a=a,
        b=b,
        tuned_delta=tuned_delta,
        n_test_with_gt=n_test,
    )
