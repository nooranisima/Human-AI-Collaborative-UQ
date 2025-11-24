
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, List, Tuple, Any

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



LABELS: List[str] = [
    "knife", "keyboard", "elephant", "bicycle", "airplane", "clock", "oven", "chair",
    "bear", "boat", "cat", "bottle", "truck", "car", "bird", "dog",
]

NOISE_LEVELS: List[int] = [80, 95, 110, 125]
MODEL_NAMES: List[str] = ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]


#path config
@dataclass
class Imagenet16HPaths:
    """
    Centralized paths for the ImageNet16H experiments.
    """
    data_root: Path
    raw_model_csv: Path
    human_alone_csv: Path
    models_dir: Path
    users_dir: Path
    meta_dir: Path
    true_labels_csv: Path
    all_expert_csv: Path

    @classmethod
    def from_data_root(cls, data_root: Path) -> "Imagenet16HPaths":
        base = data_root / "imagenet16h"
        models_dir = data_root / "models"
        users_dir = data_root / "users"
        meta_dir = base / "metadata"
        models_dir.mkdir(parents=True, exist_ok=True)
        users_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            data_root=data_root,
            raw_model_csv=base / "hai_epoch10_model_preds_max_normalized.csv",
            human_alone_csv=base / "human_only_classification_6per_img_export.csv",
            models_dir=models_dir,
            users_dir=users_dir,
            meta_dir=meta_dir,
            true_labels_csv=data_root / "imagenet16H.csv",
            all_expert_csv=base / "human_only_classification_6per_img_export.csv",
        )


#data prep utilities

def _ensure_exists(p: Path, hint: str = "") -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\n{hint}")


def build_labels_from_raw_csv(paths: Imagenet16HPaths) -> None:
    """
    Create imagenet16H.csv = mapping image_name -> category (one row per image),
    and classes.json describing the 16 classes.
    """
    _ensure_exists(paths.raw_model_csv, "Put the provided model-output CSV here.")

    raw = pd.read_csv(paths.raw_model_csv, dtype=str)
    base = raw[
        (raw["model_name"] == "vgg19") & (raw["noise_level"] == "80")
    ][["image_name", "category"]].drop_duplicates()
    base = base.set_index("image_name").sort_index()

    paths.true_labels_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(paths.true_labels_csv)

    classes_json = {i: {"name": l} for i, l in enumerate(LABELS)}
    (paths.meta_dir / "classes.json").write_text(
        json.dumps(classes_json, indent=2)
    )

    print(f"[labels] wrote {paths.true_labels_csv} ({len(base)} rows)")
    print(f"[labels] wrote {paths.meta_dir / 'classes.json'}")


def split_model_predictions(paths: Imagenet16HPaths, model_name: str) -> None:
    """
    Save per-noise CSV: models/noise{nl}/{model_name}.csv
    columns = probs for 16 labels + 'correct', index=image_name.
    """
    _ensure_exists(paths.raw_model_csv)

    dtypes = {
        "image_name": str,
        "correct": int,
        "noise_level": str,
        "model_name": str,
        "category": str,
    }
    for l in LABELS:
        dtypes[l] = float

    raw = pd.read_csv(paths.raw_model_csv, dtype=dtypes)
    raw = raw[raw["model_name"] == model_name]

    for nl in NOISE_LEVELS:
        sub = raw[raw["noise_level"] == str(nl)]
        cols = ["image_name"] + LABELS + ["correct"]
        df = sub[cols].drop_duplicates().set_index("image_name")
        out_dir = paths.models_dir / f"noise{nl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{model_name}.csv"
        df.to_csv(out)
        print(f"[model] {model_name} ω={nl}: {len(df)} rows -> {out}")


def sort_model_predictions(paths: Imagenet16HPaths, model_name: str) -> None:
    """
    Write models/noise{nl}/{model_name}_sorted.csv = rank-ordered labels per image
    (columns '1'..'16').
    """
    _ensure_exists(paths.raw_model_csv)

    dtypes = {"image_name": str, "model_name": str, "noise_level": str}
    for l in LABELS:
        dtypes[l] = float

    raw = pd.read_csv(paths.raw_model_csv, dtype=dtypes)
    raw = raw[raw["model_name"] == model_name]

    labels_arr = np.array(LABELS)
    out_cols = [str(i) for i in range(1, 17)]

    for nl in NOISE_LEVELS:
        sub = raw[raw["noise_level"] == str(nl)]
        probs = sub[["image_name"] + LABELS].drop_duplicates().set_index("image_name")

        row_sums = probs.sum(axis=1).replace(0, np.nan)
        probs = probs.div(row_sums, axis=0).fillna(1.0 / len(LABELS))

        order = np.argsort(probs.values, axis=1)[:, ::-1]
        sorted_labels = labels_arr[order]

        sorted_df = pd.DataFrame(sorted_labels, index=probs.index, columns=out_cols)
        out_dir = paths.models_dir / f"noise{nl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_sorted.csv"
        sorted_df.to_csv(out_path)
        print(f"[sorted] {model_name} ω={nl}: {len(sorted_df)} rows -> {out_path}")


def prep_human_tables(paths: Imagenet16HPaths) -> None:
    """
    From the human-alone CSV, write:
        users/noise{nl}/success.csv
        users/noise{nl}/predictions.csv
    """
    _ensure_exists(paths.human_alone_csv)

    raw = pd.read_csv(paths.human_alone_csv, dtype=str)
    for nl in NOISE_LEVELS:
        sub = raw[raw["noise_level"] == str(nl)]
        success = sub[["image_name", "participant_id", "correct"]].set_index("image_name")
        preds = sub[["image_name", "participant_id", "participant_classification"]].set_index("image_name")
        out_dir = paths.users_dir / f"noise{nl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        success.to_csv(out_dir / "success.csv")
        preds.to_csv(out_dir / "predictions.csv")
        print(f"[human] ω={nl}: success({len(success)}), predictions({len(preds)}) -> {out_dir}")


# expert prob

def _canon(s: str) -> str:
    return str(s).strip().lower()


def _label_mapper() -> Dict[str, str]:
    m = {l: l for l in LABELS}
    m.update({
        "aeroplane": "airplane", "plane": "airplane",
        "bike": "bicycle",
        "kitty": "cat", "cat.": "cat",
        "puppy": "dog",
        "sofa": "chair", "couch": "chair",
        "jug": "bottle", "bottle.": "bottle",
        "truck.": "truck", "car.": "car",
        "bird.": "bird", "dog.": "dog",
    })
    return m


def _map_to_vocab(raw: str, mapper: Dict[str, str]) -> Optional[str]:
    r = _canon(raw)
    if r in mapper:
        return mapper[r]
    if r.endswith("s") and r[:-1] in mapper:
        return mapper[r[:-1]]
    return None


def build_expert_freq_probs(
    paths: Imagenet16HPaths,
    noise_level: int,
    labels: Sequence[str] = LABELS,
) -> pd.DataFrame:
    """
    Build p(y|x) from expert classifications; write:
        users/noise{nl}/all_expert_pyx_strict.csv
    """
    _ensure_exists(paths.all_expert_csv)

    df = pd.read_csv(paths.all_expert_csv, dtype=str)
    req = {"image_name", "participant_classification", "noise_level"}
    if not req.issubset(df.columns):
        raise ValueError(f"Expert CSV must contain columns {req}, got {set(df.columns)}")

    df_noise = df[df["noise_level"] == str(noise_level)].copy()
    mapper = _label_mapper()
    df_noise["mapped"] = df_noise["participant_classification"].map(
        lambda s: _map_to_vocab(s, mapper)
    )
    df_noise = df_noise[df_noise["mapped"].isin(labels)].copy()

    counts = df_noise.groupby(["image_name", "mapped"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(labels), fill_value=0)

    row_sums = counts.sum(axis=1)
    images_with_votes = row_sums[row_sums > 0].index
    counts = counts.loc[images_with_votes]
    row_sums = row_sums.loc[images_with_votes]

    pyx = counts.div(row_sums, axis=0)
    pyx.index.name = "image_name"

    out_dir = paths.users_dir / f"noise{noise_level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_expert_pyx_strict.csv"
    pyx.to_csv(out_path)
    print(f"[all expert p(y|x)] strict ω={noise_level}: {len(pyx)} images -> {out_path}")
    return pyx


# human expert and ai model class

class HumanExpert:
    """
    Represents a human expert providing p(y|x) and prediction sets.
    Expects users/noise{nl}/all_expert_pyx_strict.csv 
    """
    def __init__(self, paths: Imagenet16HPaths, noise_level: int, labels: Sequence[str] = LABELS):
        self.paths = paths
        self.noise_level = noise_level
        self.labels = list(labels)

        path = paths.users_dir / f"noise{noise_level}" / "all_expert_pyx_strict.csv"
        if not path.exists():
            print(f"[HumanExpert] {path} not found. Building from ALL_EXPERT_CSV.")
            build_expert_freq_probs(paths, noise_level)
        if not path.exists():
            raise FileNotFoundError(f"Failed to build {path}")

        self.p = pd.read_csv(path, index_col="image_name").reindex(columns=self.labels, fill_value=0.0)
        row_sums = self.p.sum(axis=1).replace(0, np.nan)
        self.p = self.p.div(row_sums, axis=0).fillna(1.0 / len(self.labels))

    def available_images(self) -> List[str]:
        return self.p.index.tolist()

    def get_prob(self, image_name: str) -> pd.Series:
        if image_name not in self.p.index:
            raise ValueError(f"Image {image_name} not found in human expert data for noise level {self.noise_level}.")
        return self.p.loc[image_name].astype(float)

    def predict_set(
        self,
        image_name: str,
        strategy: str = "top1",
        k: Optional[int] = None,
        mass: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Build a prediction set from the human probability vector.

        Supported strategies:
          - "top1"
          - "empty"
          - "topk"      -> uses k argument or defaults to 2
          - "topk5"     -> parses k=5 from the strategy name (any k in [1, num_labels])
          - "mass"
          - "threshold"
        """
        strat = strategy.lower()
        p = self.get_prob(image_name).sort_values(ascending=False)

        # top-1
        if strat == "top1":
            return [p.index[0]]

        # empty set (AI-alone baseline)
        if strat == "empty":
            return []

        # top-k: accept "topk", "topk5", etc.
        if strat.startswith("topk"):
            parsed_k: Optional[int] = None
            suffix = strat[4:]  # part after "topk"
            if suffix:
                if suffix.isdigit():
                    parsed_k = int(suffix)
                else:
                    raise ValueError(f"Unrecognized topk strategy '{strategy}'")

        
            if k is not None:
                parsed_k = int(k)

         
            if parsed_k is None:
                parsed_k = 2


            parsed_k = max(1, min(parsed_k, len(p)))
            return list(p.index[:parsed_k])

        # cumulative mass strategy
        if strat == "mass":
            if not (isinstance(mass, (int, float)) and 0 < mass <= 1):
                raise ValueError("mass in (0,1] required for 'mass' strategy")
            cs = p.cumsum().values
            cut_idx = int(np.searchsorted(cs, float(mass), side="left"))
            cut_idx = min(cut_idx, len(p) - 1)
            return list(p.index[: cut_idx + 1])

        # probability threshold strategy
        if strat == "threshold":
            if not (isinstance(threshold, (int, float)) and 0 < threshold <= 1):
                raise ValueError("threshold in (0,1] required for 'threshold' strategy")
            chosen = [lbl for lbl, val in p.items() if val >= float(threshold)]
            return chosen or [p.index[0]]

        raise ValueError(f"Unknown strategy: {strategy}")



class AIModel:
    """
    Represents an AI model providing p(y|x).
    Expects models/noise{nl}/{model_name}.csv.
    """
    def __init__(self, paths: Imagenet16HPaths, noise_level: int, model_name: str, labels: Sequence[str] = LABELS):
        self.paths = paths
        self.noise_level = noise_level
        self.model_name = model_name
        self.labels = list(labels)

        path = paths.models_dir / f"noise{noise_level}" / f"{model_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model probabilities file: {path}. "
                f"Ensure split_model_predictions(paths, '{model_name}') was run for ω={noise_level}."
            )

        try:
            self.p = pd.read_csv(path, index_col="image_name")
        except ValueError:
            self.p = pd.read_csv(path, index_col=0)

        self.p = (
            self.p.reindex(columns=self.labels)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        row_sums = self.p.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        self.p = self.p.div(row_sums, axis=0)

    def get_prob(self, image_name: str) -> pd.Series:
        if image_name not in self.p.index:
            raise ValueError(
                f"Image {image_name} not found in {self.model_name} data for noise level {self.noise_level}."
            )
        return self.p.loc[image_name].astype(float)


# CP utilities


def get_common_image_names(
    ai_model: AIModel,
    human_expert: HumanExpert,
    true_labels_df: pd.DataFrame,
) -> List[str]:
    hs = set(human_expert.available_images())
    ms = set(ai_model.p.index)
    ts = set(true_labels_df.index)
    return sorted(hs & ms & ts)


def calculate_thresholds(
    cal_images: List[str],
    ai_model: AIModel,
    human_expert: HumanExpert,
    true_labels_df: pd.DataFrame,
    epsilon: float,
    delta: float,
    human_strategy: str,
) -> Tuple[float, float]:
    scores_in_h = []
    scores_not_in_h = []

    for image_name in cal_images:
        if image_name not in true_labels_df.index:
            continue
        try:
            ai_probs = ai_model.get_prob(image_name)
            human_set = human_expert.predict_set(image_name, strategy=human_strategy)
        except ValueError:
            continue

        true_label = true_labels_df.loc[image_name, "category"]
        score = 1.0 - ai_probs.get(true_label, 0.0)

        if true_label in human_set:
            scores_in_h.append(score)
        else:
            scores_not_in_h.append(score)

    if scores_in_h:
        b = np.quantile(scores_in_h, 1.0 - epsilon, method="higher")
    else:
        b = np.inf

    if scores_not_in_h:
        a = np.quantile(scores_not_in_h, delta, method="higher")
    else:
        a = np.inf

    return a, b


def generate_prediction_sets(
    test_images: List[str],
    ai_model: AIModel,
    human_expert: HumanExpert,
    a_threshold: float,
    b_threshold: float,
    human_strategy: str,
) -> Dict[str, Dict[str, Any]]:
    predictions: Dict[str, Dict[str, Any]] = {}

    for image_name in test_images:
        try:
            ai_probs = ai_model.get_prob(image_name)
            human_set = human_expert.predict_set(image_name, strategy=human_strategy)
        except ValueError:
            predictions[image_name] = {"prediction_set": [], "set_size": 0}
            continue

        pred_set: List[str] = []
        for label in ai_model.labels:
            score = 1.0 - ai_probs.get(label, 0.0)
            threshold = b_threshold if label in human_set else a_threshold
            if score <= threshold:
                pred_set.append(label)

        predictions[image_name] = {"prediction_set": pred_set, "set_size": len(pred_set)}

    return predictions


def calculate_coverage(
    predictions: Dict[str, Dict[str, Any]],
    true_labels_df: pd.DataFrame,
) -> float:
    if not predictions:
        return 0.0

    correct = 0
    total = 0

    for image_name, data in predictions.items():
        if image_name not in true_labels_df.index:
            continue
        total += 1
        true_label = true_labels_df.loc[image_name, "category"]
        if true_label in data.get("prediction_set", []):
            correct += 1

    if total == 0:
        return 0.0
    return correct / total


#sing run of experiment

def run_single_split(
    paths: Imagenet16HPaths,
    noise_level: int,
    model_name: str,
    human_strategy: str,
    epsilon: float,
    delta: float,
    test_size: float = 0.5,
    random_state: int = 123,
) -> Dict[str, float]:
    true_labels_df = pd.read_csv(paths.true_labels_csv, index_col="image_name")

    ai_model = AIModel(paths, noise_level=noise_level, model_name=model_name)
    human_expert = HumanExpert(paths, noise_level=noise_level)

    all_images = get_common_image_names(ai_model, human_expert, true_labels_df)
    if not all_images:
        raise RuntimeError(f"No common images for ω={noise_level}, model={model_name}.")

    cal_images, test_images = train_test_split(
        all_images,
        test_size=test_size,
        random_state=random_state,
    )

    # Human-alone on test
    human_preds_test: Dict[str, Dict[str, Any]] = {}
    for img in test_images:
        hset = human_expert.predict_set(img, strategy=human_strategy)
        human_preds_test[img] = {"prediction_set": hset, "set_size": len(hset)}

    human_cov = calculate_coverage(human_preds_test, true_labels_df)
    human_sizes = [d["set_size"] for d in human_preds_test.values()]
    human_size = float(np.mean(human_sizes)) if human_sizes else 0.0

    # Collaborative method
    a, b = calculate_thresholds(
        cal_images,
        ai_model,
        human_expert,
        true_labels_df,
        epsilon=epsilon,
        delta=delta,
        human_strategy=human_strategy,
    )

    preds_test = generate_prediction_sets(
        test_images,
        ai_model,
        human_expert,
        a_threshold=a,
        b_threshold=b,
        human_strategy=human_strategy,
    )

    method_cov = calculate_coverage(preds_test, true_labels_df)
    method_sizes = [d["set_size"] for d in preds_test.values()]
    method_size = float(np.mean(method_sizes)) if method_sizes else 0.0


    true_labels_dict = true_labels_df.loc[test_images, "category"].to_dict()

    count_Y_not_in_H = 0
    count_Y_in_C_and_Y_not_in_H = 0
    count_Y_in_H = 0
    count_Y_not_in_C_and_Y_in_H = 0

    for img in test_images:
        if img not in true_labels_dict:
            continue
        y = true_labels_dict[img]
        hset = human_preds_test.get(img, {}).get("prediction_set", [])
        cset = preds_test.get(img, {}).get("prediction_set", [])

        if y not in hset:
            count_Y_not_in_H += 1
            if y in cset:
                count_Y_in_C_and_Y_not_in_H += 1
        else:
            count_Y_in_H += 1
            if y not in cset:
                count_Y_not_in_C_and_Y_in_H += 1

    cond_cov_not_in_H = (
        count_Y_in_C_and_Y_not_in_H / count_Y_not_in_H
        if count_Y_not_in_H > 0 else np.nan
    )
    cond_err_in_H = (
        count_Y_not_in_C_and_Y_in_H / count_Y_in_H
        if count_Y_in_H > 0 else np.nan
    )

    return {
        "human_coverage": float(human_cov),
        "human_set_size": float(human_size),
        "method_coverage": float(method_cov),
        "method_set_size": float(method_size),
        "cond_coverage_not_in_H": float(cond_cov_not_in_H),
        "cond_error_in_H": float(cond_err_in_H),
    }



# Sweeps over strategies / (ε, δ)


def sweep_strategies_eps_delta(
    paths: Imagenet16HPaths,
    noise_level: int,
    model_name: str,
    strategies: Sequence[str],
    deltas: Sequence[float],
    epsilons: Sequence[float],
    num_splits: int = 10,
    test_size: float = 0.5,
    base_seed: int = 123,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for strategy in strategies:
        for delta in deltas:
            for eps in epsilons:
                metrics: Dict[str, List[float]] = {
                    "human_coverage": [],
                    "human_set_size": [],
                    "method_coverage": [],
                    "method_set_size": [],
                    "cond_coverage_not_in_H": [],
                    "cond_error_in_H": [],
                }

                for split_idx in range(num_splits):
                    seed = base_seed + split_idx
                    try:
                        out = run_single_split(
                            paths=paths,
                            noise_level=noise_level,
                            model_name=model_name,
                            human_strategy=strategy,
                            epsilon=eps,
                            delta=delta,
                            test_size=test_size,
                            random_state=seed,
                        )
                    except Exception as e:
                        print(f"[WARN] split {split_idx} failed for strategy={strategy}, δ={delta}, ε={eps}: {e}")
                        out = {k: np.nan for k in metrics.keys()}

                    for k in metrics:
                        metrics[k].append(out[k])

                row: Dict[str, Any] = {
                    "strategy": strategy,
                    "delta": float(delta),
                    "epsilon": float(eps),
                }
                for k, vals in metrics.items():
                    arr = np.array(vals, dtype=float)
                    valid = arr[~np.isnan(arr)]
                    if valid.size == 0:
                        row[f"{k}_mean"] = np.nan
                        row[f"{k}_std"] = np.nan
                    else:
                        row[f"{k}_mean"] = float(valid.mean())
                        row[f"{k}_std"] = float(valid.std())

                rows.append(row)

    return pd.DataFrame(rows)


def sweep_ai_alone(
    paths: Imagenet16HPaths,
    noise_level: int,
    model_name: str,
    deltas: Sequence[float],
    epsilon: float,
    num_splits: int = 10,
    test_size: float = 0.5,
    base_seed: int = 123,
) -> pd.DataFrame:
    """
    AI-alone baseline: H(x) = ∅, i.e. strategy="empty".
    """
    return sweep_strategies_eps_delta(
        paths=paths,
        noise_level=noise_level,
        model_name=model_name,
        strategies=["empty"],
        deltas=deltas,
        epsilons=[epsilon],
        num_splits=num_splits,
        test_size=test_size,
        base_seed=base_seed,
    )
