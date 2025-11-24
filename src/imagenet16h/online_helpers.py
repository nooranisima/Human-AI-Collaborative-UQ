# online_helper.py


from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# paths


DATA_PATH = Path("/content/drive/MyDrive/Colab Notebooks/HAI-UQ/data")

MODELS_DIR = DATA_PATH / "models"
USERS_DIR  = DATA_PATH / "users"
META_DIR   = DATA_PATH / "imagenet16h" / "metadata"
TRUE_LABELS_CSV = DATA_PATH / "imagenet16H.csv"
CLASSES_JSON    = META_DIR / "classes.json"  # optional

LABELS = [
    "knife","keyboard","elephant","bicycle","airplane","clock","oven","chair",
    "bear","boat","cat","bottle","truck","car","bird","dog"
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

NOISE_LEVELS = [80, 95, 110, 125]


DEFAULT_SEED = 1337



def load_true_labels() -> pd.Series:
    """Return Series: index=image_name, value=category ∈ LABELS."""
    df = pd.read_csv(TRUE_LABELS_CSV, dtype=str).set_index("image_name")
    df["category"] = df["category"].str.strip().str.lower()
    return df["category"]


def load_human_probs(noise_level: int) -> pd.DataFrame:
    """
    Load human p(y|x) at a given noise level.
    Returns DataFrame[image_name x LABELS] with row-normalized probabilities.
    """
    path = USERS_DIR / f"noise{noise_level}" / "all_expert_pyx_strict.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Build via offline prep.")
    df = pd.read_csv(path, index_col="image_name")
    df = df.reindex(columns=LABELS).fillna(0.0)
    rs = df.sum(axis=1).replace(0, np.nan)
    df = df.div(rs, axis=0).fillna(1.0 / len(LABELS))
    return df


def load_ai_probs(noise_level: int, model_name: str) -> pd.DataFrame:
    """
    Load AI p(y|x) at a given noise level for a given model.
    Returns DataFrame[image_name x LABELS] with row-normalized probabilities.
    """
    path = MODELS_DIR / f"noise{noise_level}" / f"{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run split_model_predictions first.")
    df = pd.read_csv(path, index_col="image_name")
    df = df.reindex(columns=LABELS).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rs = df.sum(axis=1)
    rs[rs == 0] = 1.0
    return df.div(rs, axis=0)


# huamn set strategies

def human_set_from_probs(pH: pd.Series, policy: Dict[str, Any]) -> List[str]:
    """
    Build the human set H(x) from human probabilities pH and a policy dict.

    policy:
      - {"mode": "topk",     "k": int}
      - {"mode": "mass",     "tau": float in (0,1]}
      - {"mode": "threshold","omega": float in (0,1]}
      - {"mode": "empty"}
    """
    mode = policy.get("mode", "topk")
    pH = pH.sort_values(ascending=False)

    if mode == "empty":
        return []

    if mode == "topk":
        k = int(policy.get("k", 1))
        k = max(1, min(k, len(pH)))
        return list(pH.index[:k])

    if mode == "mass":
        tau = float(policy.get("tau", 0.8))
        cs = pH.cumsum().values
        cut = int(np.searchsorted(cs, tau, side="left"))
        cut = min(cut, len(pH) - 1)
        return list(pH.index[:cut + 1])

    if mode == "threshold":
        omega = float(policy.get("omega", 0.2))
        chosen = [lbl for lbl, val in pH.items() if val >= omega]
        return chosen or [pH.index[0]]

    raise ValueError(f"Unknown human policy: {policy}")


# small utils

def emp_quantile(arr: List[float], level: float) -> float:
    """Conservative empirical quantile (VALUE cutoff, 'higher' method)."""
    if len(arr) == 0:
        return np.inf
    return float(np.quantile(np.asarray(arr, dtype=float), level, method="higher"))


def _stable_hash_int(*parts) -> int:
    s = "|".join(map(str, parts))
    return (abs(hash(s)) % (2**31 - 1)) or 1


def _phase_signature(mode: str,
                     noise: Optional[int],
                     human_policy: Optional[Dict[str, Any]],
                     class_subset: Optional[List[str]]) -> str:
    if mode == "noise":
        return f"noise:{noise}"
    if mode == "human":
        hp = tuple(sorted((human_policy or {}).items()))
        return f"human:{hp}|noise:{noise}"
    if mode == "class":
        subset = tuple(sorted(class_subset or []))
        return f"class:{subset}|noise:{noise}"
    return "unknown"


def _perm_split_indices(n: int, base_seed: int) -> Tuple[List[int], List[int]]:
    """Return (first_half_idx, second_half_idx) from a deterministic permutation."""
    rng = np.random.RandomState(base_seed)
    order = rng.permutation(n)
    mid = n // 2
    return order[:mid].tolist(), order[mid:].tolist()


@dataclass
class Phase:
    name: str
    steps: Optional[int] = None     # None => use all available examples in this pool
    noise_level: Optional[int] = None
    human_policy: Optional[Dict[str, Any]] = None
    class_subset: Optional[List[str]] = None
    # If provided, the stream will use exactly these indices into its pool.
    use_indices: Optional[List[int]] = None


@dataclass
class RunConfig:
    mode: str                     # "noise" | "human" | "class"
    model_name: str
    eps: float
    delta: float
    steps_report: int = 500

    # Modes:
    noise_phases: Optional[List[Phase]] = None
    human_phases: Optional[List[Phase]] = None
    class_phases: Optional[List[Phase]] = None

    # Shared knobs:
    fixed_policy: Optional[Dict[str, Any]] = None
    fixed_noise_level: Optional[int] = None
    seed: Optional[int] = DEFAULT_SEED


    eta: Optional[float] = None
    eta_a: Optional[float] = None
    eta_b: Optional[float] = None
    normalize_levels: bool = True
    min_pi_levels: float = 1e-3


class BaseStream:
    def __iter__(self):
        raise NotImplementedError


class NoiseShiftStream(BaseStream):
    """
    Stream where noise level changes across phases, human policy fixed.
    """
    def __init__(self, phases: List[Phase], model_name: str,
                 fixed_policy: Dict[str, Any], seed: Optional[int] = None):
        self.phases = phases
        self.model_name = model_name
        self.fixed_policy = fixed_policy
        self.seed = seed if seed is not None else DEFAULT_SEED

        self.y_true = load_true_labels()
        self.hp_by_noise: Dict[int, pd.DataFrame] = {}
        self.ai_by_noise: Dict[int, pd.DataFrame] = {}

        for p in phases:
            nl = p.noise_level
            assert nl in NOISE_LEVELS, f"Unsupported noise {nl}"
            if nl not in self.hp_by_noise:
                self.hp_by_noise[nl] = load_human_probs(nl)
            if nl not in self.ai_by_noise:
                self.ai_by_noise[nl] = load_ai_probs(nl, model_name)

        self.imgs_by_noise: Dict[int, List[str]] = {}
        for nl in self.hp_by_noise:
            hp, ai = self.hp_by_noise[nl], self.ai_by_noise[nl]
            imgs = sorted(set(hp.index).intersection(ai.index).intersection(self.y_true.index))
            if not imgs:
                raise RuntimeError(f"No overlapping images for noise {nl}")
            self.imgs_by_noise[nl] = imgs

    def __iter__(self):
        for p in self.phases:
            imgs = self.imgs_by_noise[p.noise_level]
            n = len(imgs)
            steps = p.steps if p.steps is not None else n

            if p.use_indices is not None:
                idxs = p.use_indices[:steps]
            else:
                sig = _phase_signature("noise", p.noise_level, None, None)
                seed = _stable_hash_int("noise", self.seed, sig)
                rng = np.random.RandomState(seed)
                perm = rng.permutation(n)
                idxs = perm[:steps]

            for j in idxs:
                image = imgs[int(j)]
                y = self.y_true.loc[image]
                yield {
                    "image": image,
                    "y": y,
                    "noise": p.noise_level,
                    "policy": self.fixed_policy,
                    "phase": p.name,
                }


class HumanShiftStream(BaseStream):
    """
    Stream where human policy changes across phases, noise fixed.
    """
    def __init__(self, phases: List[Phase], model_name: str,
                 noise_level: int, seed: Optional[int] = None):
        self.phases = phases
        self.model_name = model_name
        self.noise_level = noise_level
        self.seed = seed if seed is not None else DEFAULT_SEED

        self.y_true = load_true_labels()
        self.hp = load_human_probs(noise_level)
        self.ai = load_ai_probs(noise_level, model_name)

        self.imgs = sorted(set(self.hp.index).intersection(self.ai.index).intersection(self.y_true.index))
        if not self.imgs:
            raise RuntimeError("No overlapping images for selected noise level.")

    def __iter__(self):
        n = len(self.imgs)
        for p in self.phases:
            policy = p.human_policy or {"mode": "topk", "k": 1}
            steps = p.steps if p.steps is not None else n

            if p.use_indices is not None:
                idxs = p.use_indices[:steps]
            else:
                sig = _phase_signature("human", self.noise_level, policy, None)
                seed = _stable_hash_int("human", self.seed, sig)
                rng = np.random.RandomState(seed)
                perm = rng.permutation(n)
                idxs = perm[:steps]

            for j in idxs:
                image = self.imgs[int(j)]
                y = self.y_true.loc[image]
                yield {
                    "image": image,
                    "y": y,
                    "noise": self.noise_level,
                    "policy": policy,
                    "phase": p.name,
                }


class ClassShiftStream(BaseStream):
    """
    Stream where the available class subset shifts across phases, noise & human policy fixed.
    """
    def __init__(self, phases: List[Phase], model_name: str, noise_level: int,
                 base_policy: Dict[str, Any], seed: Optional[int] = None):
        self.phases = phases
        self.model_name = model_name
        self.noise_level = noise_level
        self.base_policy = base_policy
        self.seed = seed if seed is not None else DEFAULT_SEED

        self.y_true = load_true_labels()
        self.hp = load_human_probs(noise_level)
        self.ai = load_ai_probs(noise_level, model_name)

        self.imgs_all = sorted(set(self.hp.index).intersection(self.ai.index).intersection(self.y_true.index))
        if not self.imgs_all:
            raise RuntimeError("No overlapping images for selected noise level.")

        self.imgs_by_class: Dict[str, List[str]] = defaultdict(list)
        for img in self.imgs_all:
            y = self.y_true.loc[img]
            if y in LABELS:
                self.imgs_by_class[y].append(img)

    def __iter__(self):
        for p in self.phases:
            subset = p.class_subset or LABELS
            pool: List[str] = []
            for c in subset:
                pool.extend(self.imgs_by_class.get(c, []))
            if not pool:
                raise RuntimeError(f"No images for class subset: {subset}")

            n = len(pool)
            steps = p.steps if p.steps is not None else n

            if p.use_indices is not None:
                idxs = p.use_indices[:steps]
            else:
                sig = _phase_signature("class", self.noise_level, self.base_policy, subset)
                seed = _stable_hash_int("class", self.seed, sig)
                rng = np.random.RandomState(seed)
                perm = rng.permutation(n)
                idxs = perm[:steps]

            for j in idxs:
                image = pool[int(j)]
                y = self.y_true.loc[image]
                yield {
                    "image": image,
                    "y": y,
                    "noise": self.noise_level,
                    "policy": self.base_policy,
                    "phase": p.name,
                }


def collect_stream(cfg: RunConfig):
    """
    Materialize a finite stream for a given RunConfig.
    Returns:
      events: List[dict]
      phase_bounds: np.ndarray of starting indices per phase
      phase_names: List[str]
      title_prefix: str for plotting
    """
    if cfg.mode == "noise":
        stream = NoiseShiftStream(cfg.noise_phases, cfg.model_name, cfg.fixed_policy, seed=cfg.seed)
        title_prefix = "Noise shift · "
    elif cfg.mode == "human":
        stream = HumanShiftStream(cfg.human_phases, cfg.model_name, cfg.fixed_noise_level, seed=cfg.seed)
        title_prefix = "Human shift · "
    else:
        stream = ClassShiftStream(cfg.class_phases, cfg.model_name, cfg.fixed_noise_level,
                                  cfg.fixed_policy, seed=cfg.seed)
        title_prefix = "Class shift · "

    events, phase_bounds, phase_names = [], [], []
    prev = None
    for t, item in enumerate(stream, start=1):
        events.append(item)
        if item["phase"] != prev:
            phase_bounds.append(t)
            phase_names.append(item["phase"])
            prev = item["phase"]
    return events, np.array(phase_bounds), phase_names, title_prefix


# conformal sets / online alg

def conformal_set(ai_p: pd.Series, H: List[str], q_notH: float, q_inH: float) -> List[str]:
    """
    Build conformal set C(x) given AI probabilities, human set H,
    and VALUE thresholds q_notH (Y∉H branch) and q_inH (Y∈H branch).
    """
    out: List[str] = []
    for y in LABELS:
        s = 1.0 - float(ai_p.get(y, 0.0))
        thr = q_inH if (y in H) else q_notH
        if s <= thr:
            out.append(y)
    return out


def replay_thresholds(
    events: List[Dict],
    cfg: RunConfig,
    q_inH0: float,
    q_notH0: float,
    adaptive: bool,
    eta: float,
) -> Dict[str, Any]:
    """
    Replay online CUP on a given sequence of events, starting from VALUE thresholds
    q_inH0, q_notH0. If adaptive=True, thresholds are updated online via Robbins–Monro
    toward target levels 1−ε and δ.

    Returns dict with success/masks/q-traces and marginal coverage.
    """
    eta_a = float(cfg.eta_a if cfg.eta_a is not None else eta)  # Y∉H branch
    eta_b = float(cfg.eta_b if cfg.eta_b is not None else eta)  # Y∈H branch
    normalize = bool(cfg.normalize_levels)
    min_pi = float(cfg.min_pi_levels)

    target_notH = cfg.delta
    target_inH  = 1.0 - cfg.eps

    q_notH = float(q_notH0)
    q_inH  = float(q_inH0)

    hp_cache: Dict[int, pd.DataFrame] = {}
    ai_df_cache: Dict[int, pd.DataFrame] = {}
    ai_row_cache: Dict[Tuple[int, str], pd.Series] = {}

    success_any, mask_inH, mask_notH = [], [], []
    q_notH_trace, q_inH_trace = [], []

    n_inH = 0
    n_notH = 0
    tot = 0
    correct = 0

    for item in events:
        image = item["image"]
        y = item["y"]
        noise = item["noise"]
        policy = item["policy"]

        # human set
        if noise not in hp_cache:
            hp_cache[noise] = load_human_probs(noise)
        pH = hp_cache[noise].loc[image]
        H = human_set_from_probs(pH, policy)

        # AI probs
        key = (noise, image)
        if key in ai_row_cache:
            pA = ai_row_cache[key]
        else:
            if noise not in ai_df_cache:
                ai_df_cache[noise] = load_ai_probs(noise, cfg.model_name)
            pA = ai_df_cache[noise].loc[image]
            ai_row_cache[key] = pA

        Cx = conformal_set(pA, H, q_notH=q_notH, q_inH=q_inH)

        y_in_set = int(y in Cx)
        y_in_H   = int(y in H)

        success_any.append(y_in_set)
        mask_inH.append(y_in_H)
        mask_notH.append(1 - y_in_H)

        tot += 1
        correct += y_in_set

        # realized score
        s_true = 1.0 - float(pA.get(y, 0.0))

        if adaptive:
            if y_in_H:
                n_inH += 1
                eff_eta = eta_b
                if normalize:
                    pi = n_inH / max(1, (n_inH + n_notH))
                    eff_eta = eta_b / max(pi, min_pi)
                q_inH += eff_eta * (target_inH - float(s_true <= q_inH))
            else:
                n_notH += 1
                eff_eta = eta_a
                if normalize:
                    pi = n_notH / max(1, (n_inH + n_notH))
                    eff_eta = eta_a / max(pi, min_pi)
                q_notH += eff_eta * (target_notH - float(s_true <= q_notH))

        q_notH_trace.append(q_notH)
        q_inH_trace.append(q_inH)

    return dict(
        success=np.array(success_any, dtype=int),
        mask_inH=np.array(mask_inH, dtype=int),
        mask_notH=np.array(mask_notH, dtype=int),
        q_notH_trace=np.array(q_notH_trace, dtype=float),
        q_inH_trace=np.array(q_inH_trace, dtype=float),
        marginal_cov_final=correct / max(1, tot),
    )


def replay_thresholds_with_sizes(
    events: List[Dict],
    cfg: RunConfig,
    q_inH0: float,
    q_notH0: float,
    adaptive: bool,
    eta: float,
) -> Dict[str, Any]:
    """
    Same as replay_thresholds but also logs per-step |C_t(x)|.
    """
    eta_a = float(cfg.eta_a if cfg.eta_a is not None else eta)
    eta_b = float(cfg.eta_b if cfg.eta_b is not None else eta)
    normalize = bool(cfg.normalize_levels)
    min_pi = float(cfg.min_pi_levels)

    target_notH = cfg.delta
    target_inH  = 1.0 - cfg.eps

    q_notH = float(q_notH0)
    q_inH  = float(q_inH0)

    hp_cache: Dict[int, pd.DataFrame] = {}
    ai_df_cache: Dict[int, pd.DataFrame] = {}
    ai_row_cache: Dict[Tuple[int, str], pd.Series] = {}

    sizes, success, mask_inH = [], [], []
    q_notH_trace, q_inH_trace = [], []

    n_inH = 0
    n_notH = 0
    tot = 0
    correct = 0

    for it in events:
        img, y, nz, pol = it["image"], it["y"], it["noise"], it["policy"]

        # human set
        if nz not in hp_cache:
            hp_cache[nz] = load_human_probs(nz)
        pH = hp_cache[nz].loc[img]
        H = human_set_from_probs(pH, pol)

        # AI probs
        key = (nz, img)
        if key in ai_row_cache:
            pA = ai_row_cache[key]
        else:
            if nz not in ai_df_cache:
                ai_df_cache[nz] = load_ai_probs(nz, cfg.model_name)
            pA = ai_df_cache[nz].loc[img]
            ai_row_cache[key] = pA

        Cx = conformal_set(pA, H, q_notH=q_notH, q_inH=q_inH)
        sizes.append(len(Cx))

        y_in_set = int(y in Cx)
        success.append(y_in_set)
        yinH = int(y in H)
        mask_inH.append(yinH)

        tot += 1
        correct += y_in_set

        s_true = 1.0 - float(pA.get(y, 0.0))

        if adaptive:
            if yinH:
                n_inH += 1
                eff = eta_b
                if normalize:
                    pi = n_inH / max(1, (n_inH + n_notH))
                    eff = eta_b / max(pi, min_pi)
                q_inH += eff * (target_inH - float(s_true <= q_inH))
            else:
                n_notH += 1
                eff = eta_a
                if normalize:
                    pi = n_notH / max(1, (n_inH + n_notH))
                    eff = eta_a / max(pi, min_pi)
                q_notH += eff * (target_notH - float(s_true <= q_notH))

        q_notH_trace.append(q_notH)
        q_inH_trace.append(q_inH)

    return dict(
        success=np.asarray(success, dtype=int),
        size=np.asarray(sizes, dtype=int),
        mask_inH=np.asarray(mask_inH, dtype=int),
        q_inH_trace=np.asarray(q_inH_trace, dtype=float),
        q_notH_trace=np.asarray(q_notH_trace, dtype=float),
        marginal_cov_final=correct / max(1, tot),
    )

# calibration

def _calibrate_half_internal(
    cfg: RunConfig,
    pool_builder: callable,
    inject_indices: callable,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Internal half-split calibration:
      - pool_builder(y_true) -> (pool, human_prob_source, ai_prob_source, policy_for_image, phase_signature)
      - inject_indices(second_half_indices) -> bool (whether they were attached to some test phase)
    """
    y_true = load_true_labels()
    S_inH, S_notH = [], []

    pool, human_prob_source, ai_prob_source, policy_for_image, phase_sig = pool_builder(y_true)
    if not pool:
        raise RuntimeError("No images for calibration.")

    seed = _stable_hash_int("calib", cfg.seed, phase_sig)
    first, second = _perm_split_indices(len(pool), base_seed=seed)

    hp_cache = human_prob_source
    ai_df_cache = ai_prob_source["df"]
    ai_row_cache = ai_prob_source["row"]

    for j in first:
        rec = pool[j]
        image = rec["image"]
        noise = rec["noise"]
        policy = rec["policy"]

        if noise not in hp_cache:
            hp_cache[noise] = load_human_probs(noise)
        pH = hp_cache[noise].loc[image]
        H = human_set_from_probs(pH, policy)

        if noise not in ai_df_cache:
            ai_df_cache[noise] = load_ai_probs(noise, cfg.model_name)
        key = (noise, image)
        if key in ai_row_cache:
            pA = ai_row_cache[key]
        else:
            pA = ai_df_cache[noise].loc[image]
            ai_row_cache[key] = pA

        y = y_true.loc[image]
        s_true = 1.0 - float(pA.get(y, 0.0))
        (S_inH if y in H else S_notH).append(s_true)

    applied = inject_indices(second)
    if not applied:
        print("[calibration] Warning: no matching test phase found to receive use_indices.")

    stats = dict(
        n_pool=len(pool),
        n_calib=len(first),
        n_test=len(second),
        n_inH=len(S_inH),
        n_notH=len(S_notH),
    )
    return S_inH, S_notH, stats


def calibrate_half_noise(cfg: RunConfig, calib_noise_level: int):
    """
    Half-split calibration for NOISE shift at a given noise level.
    """
    def pool_builder(y_true):
        hp = load_human_probs(calib_noise_level)
        ai = load_ai_probs(calib_noise_level, cfg.model_name)
        imgs = sorted(set(hp.index).intersection(ai.index).intersection(y_true.index))

        human_prob_source = {calib_noise_level: hp}
        ai_prob_source = {"df": {calib_noise_level: ai}, "row": {}}
        pool = [{"image": im, "noise": calib_noise_level, "policy": cfg.fixed_policy} for im in imgs]
        sig = _phase_signature("noise", calib_noise_level, None, None)
        return pool, human_prob_source, ai_prob_source, (lambda im: cfg.fixed_policy), sig

    def inject_indices(second):
        applied = False
        if cfg.noise_phases:
            for p in cfg.noise_phases:
                if p.noise_level == calib_noise_level and p.use_indices is None:
                    p.use_indices = second
                    applied = True
                    break
        return applied

    return _calibrate_half_internal(cfg, pool_builder, inject_indices)


def build_calibration_histories_human(model_name: str,
                                      calib_policy: dict,
                                      fixed_noise_level: int,
                                      calib_steps: int):
    """
    Half-split calibration for HUMAN shift at a fixed noise level and policy.
    `calib_steps` is ignored; we use half of the available pool.
    """
    y_true = load_true_labels()
    hp = load_human_probs(fixed_noise_level)
    ai = load_ai_probs(fixed_noise_level, model_name)

    pool_imgs = sorted(set(hp.index).intersection(ai.index).intersection(y_true.index))
    if not pool_imgs:
        raise RuntimeError(f"No overlapping images for human calibration at ω={fixed_noise_level}.")

    sig = ("human", model_name, fixed_noise_level, tuple(sorted((calib_policy or {}).items())))
    seed = (abs(hash(sig)) % (2**31 - 1)) or 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pool_imgs))
    mid = len(perm) // 2
    calib_idx = perm[:mid].tolist()
    test_idx  = perm[mid:].tolist()

    S_inH, S_notH = [], []
    for j in calib_idx:
        img = pool_imgs[j]
        y = y_true.loc[img]
        pH = hp.loc[img]
        H = human_set_from_probs(pH, calib_policy)
        pA = ai.loc[img]
        s_true = 1.0 - float(pA.get(y, 0.0))
        (S_inH if y in H else S_notH).append(s_true)

    stats = dict(
        n_inH=len(S_inH),
        n_notH=len(S_notH),
        n_pool=len(pool_imgs),
        n_calib=len(calib_idx),
        n_test=len(test_idx),
        test_indices=test_idx,
        pool_imgs=pool_imgs,
        noise_level=fixed_noise_level,
        policy=calib_policy,
    )
    return S_inH, S_notH, stats


def build_calibration_histories_class(model_name: str,
                                      calib_class_subset: List[str],
                                      fixed_noise_level: int,
                                      fixed_policy: dict,
                                      calib_steps: int):
    """
    Half-split calibration for CLASS shift restricted to `calib_class_subset`
    at a fixed noise level and human policy.
    """
    y_true = load_true_labels()
    hp = load_human_probs(fixed_noise_level)
    ai = load_ai_probs(fixed_noise_level, model_name)

    subset_set = set(calib_class_subset)
    pool_imgs = [
        img for img in sorted(set(hp.index).intersection(ai.index).intersection(y_true.index))
        if y_true.loc[img] in subset_set
    ]
    if not pool_imgs:
        raise RuntimeError(
            f"No overlapping images for class calibration at ω={fixed_noise_level} "
            f"with subset={sorted(subset_set)}."
        )

    sig = ("class", model_name, fixed_noise_level,
           tuple(sorted(subset_set)), tuple(sorted((fixed_policy or {}).items())))
    seed = (abs(hash(sig)) % (2**31 - 1)) or 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pool_imgs))
    mid = len(perm) // 2
    calib_idx = perm[:mid].tolist()
    test_idx  = perm[mid:].tolist()

    S_inH, S_notH = [], []
    for j in calib_idx:
        img = pool_imgs[j]
        y = y_true.loc[img]
        pH = hp.loc[img]
        H = human_set_from_probs(pH, fixed_policy)
        pA = ai.loc[img]
        s_true = 1.0 - float(pA.get(y, 0.0))
        (S_inH if y in H else S_notH).append(s_true)

    stats = dict(
        n_inH=len(S_inH),
        n_notH=len(S_notH),
        n_pool=len(pool_imgs),
        n_calib=len(calib_idx),
        n_test=len(test_idx),
        test_indices=test_idx,
        pool_imgs=pool_imgs,
        subset=sorted(subset_set),
        noise_level=fixed_noise_level,
        policy=fixed_policy,
    )
    return S_inH, S_notH, stats


# top level func to use 
def compare_online_levels_with_calibration(cfg_test: "RunConfig",
                                           calib_noise_level: int = 80,
                                           calib_steps: int = 2000,  # kept for API parity; not used here
                                           W: int = 500):
    if getattr(cfg_test, "seed", None) is None:
        cfg_test.seed = 1337

    # histories from half-split calibration
    S_inH0, S_notH0, cstats = calibrate_half_noise(cfg_test, calib_noise_level=calib_noise_level)
    print(f"[calib] noise={calib_noise_level}  |pool|={cstats['n_pool']}  "
          f"|calib|={cstats['n_calib']}  |test|={cstats['n_test']}  "
          f"|inH|={cstats['n_inH']}  |notH|={cstats['n_notH']}")

    # test stream 
    events, phase_bounds, phase_names, title_prefix = collect_stream(cfg_test)

    # initial VALUE thresholds from the calibrated histories at the correct levels
    q_notH0 = emp_quantile(S_notH0, level=cfg_test.delta)
    q_inH0  = emp_quantile(S_inH0,  level=1.0 - cfg_test.eps)
    eta = float(getattr(cfg_test, "eta", 0.1))

    # fixed thresholds (no updates)
    fixed = replay_thresholds(events, cfg_test,
                              q_inH0=q_inH0, q_notH0=q_notH0,
                              adaptive=False, eta=eta)

    # adaptive thresholds (update values online)
    adaptive = replay_thresholds(events, cfg_test,
                                 q_inH0=q_inH0, q_notH0=q_notH0,
                                 adaptive=True, eta=eta)

    # plot
    plot_compare_local_coverage_levels(
        adaptive, fixed,
        eps=cfg_test.eps, delta=cfg_test.delta,
        phase_bounds=phase_bounds, phase_names=phase_names,
        W=W, title_prefix=title_prefix
    )

    return {
        "adaptive": adaptive,
        "fixed": fixed,
        "q_notH0": q_notH0,
        "q_inH0": q_inH0,
        "calib_stats": cstats,
        "events": events,
        "phase_bounds": phase_bounds,
        "phase_names": phase_names,
        "title_prefix": title_prefix,
    }


def compare_online_levels_with_calibration_human(cfg_human: RunConfig,
                                                 calib_policy: dict,
                                                 calib_steps: int,
                                                 W: int = 500) -> Dict[str, Any]:
    """
    HUMAN shift experiment using VALUE cutoffs and half-split calibration.
    Uses *all* data at cfg_human.fixed_noise_level; phases differ only in human policy.
    """
    S_inH0, S_notH0, stats = build_calibration_histories_human(
        model_name=cfg_human.model_name,
        calib_policy=calib_policy,
        fixed_noise_level=cfg_human.fixed_noise_level,
        calib_steps=calib_steps,
    )
    print(f"[calib/human] ω={cfg_human.fixed_noise_level} "
          f"|pool|={stats['n_pool']} |calib|={stats['n_calib']} |test|={stats['n_test']} "
          f"|inH|={stats['n_inH']} |notH|={stats['n_notH']}")

    applied = False
    if cfg_human.human_phases:
        for p in cfg_human.human_phases:
            pol = p.human_policy or {"mode": "topk", "k": 1}
            if pol == calib_policy and p.use_indices is None:
                p.use_indices = stats["test_indices"]
                applied = True
                break
    if not applied:
        print("[calib/human] Warning: no matching test phase found to receive use_indices.")

    events, phase_bounds, phase_names, title_prefix = collect_stream(cfg_human)

    q_notH0 = emp_quantile(S_notH0, level=cfg_human.delta)
    q_inH0  = emp_quantile(S_inH0,  level=1.0 - cfg_human.eps)
    eta = float(cfg_human.eta if cfg_human.eta is not None else 0.05)

    fixed = replay_thresholds(events, cfg_human, q_inH0=q_inH0, q_notH0=q_notH0,
                              adaptive=False, eta=eta)
    adaptive = replay_thresholds(events, cfg_human, q_inH0=q_inH0, q_notH0=q_notH0,
                                 adaptive=True, eta=eta)

    plot_compare_local_coverage_levels(
        adaptive, fixed,
        eps=cfg_human.eps, delta=cfg_human.delta,
        phase_bounds=phase_bounds, phase_names=phase_names,
        W=W, title_prefix="Human shift · "
    )

    return dict(
        adaptive=adaptive,
        fixed=fixed,
        calib_stats=stats,
        q_notH0=q_notH0,
        q_inH0=q_inH0,
        phase_bounds=phase_bounds,
        phase_names=phase_names,
        events=events,
    )


def compare_online_levels_with_calibration_class(cfg_class: RunConfig,
                                                 calib_class_subset: List[str],
                                                 calib_steps: int,
                                                 W: int = 500) -> Dict[str, Any]:
    """
    CLASS-LABEL shift experiment using VALUE cutoffs and half-split calibration.
    Calibration uses all data in the given class subset at cfg_class.fixed_noise_level.
    """
    S_inH0, S_notH0, stats = build_calibration_histories_class(
        model_name=cfg_class.model_name,
        calib_class_subset=calib_class_subset,
        fixed_noise_level=cfg_class.fixed_noise_level,
        fixed_policy=cfg_class.fixed_policy,
        calib_steps=calib_steps,
    )
    print(f"[calib/class] ω={cfg_class.fixed_noise_level} subset={stats['subset']} "
          f"|pool|={stats['n_pool']} |calib|={stats['n_calib']} |test|={stats['n_test']} "
          f"|inH|={stats['n_inH']} |notH|={stats['n_notH']}")

    applied = False
    if cfg_class.class_phases:
        want = set(stats["subset"])
        for p in cfg_class.class_phases:
            sub = set(p.class_subset or [])
            if sub == want and p.use_indices is None:
                p.use_indices = stats["test_indices"]
                applied = True
                break
    if not applied:
        print("[calib/class] Warning: no matching test phase found to receive use_indices.")

    events, phase_bounds, phase_names, title_prefix = collect_stream(cfg_class)

    q_notH0 = emp_quantile(S_notH0, level=cfg_class.delta)
    q_inH0  = emp_quantile(S_inH0,  level=1.0 - cfg_class.eps)
    eta = float(cfg_class.eta if cfg_class.eta is not None else 0.05)

    fixed = replay_thresholds(events, cfg_class, q_inH0=q_inH0, q_notH0=q_notH0,
                              adaptive=False, eta=eta)
    adaptive = replay_thresholds(events, cfg_class, q_inH0=q_inH0, q_notH0=q_notH0,
                                 adaptive=True, eta=eta)

    plot_compare_local_coverage_levels(
        adaptive, fixed,
        eps=cfg_class.eps, delta=cfg_class.delta,
        phase_bounds=phase_bounds, phase_names=phase_names,
        W=W, title_prefix="Class-label shift · "
    )

    return dict(
        adaptive=adaptive,
        fixed=fixed,
        calib_stats=stats,
        q_notH0=q_notH0,
        q_inH0=q_inH0,
        phase_bounds=phase_bounds,
        phase_names=phase_names,
        events=events,
    )

# bsaelines

COLORS = {
    "online": "#1f77b4",  # blue  (CUP / online)
    "human":  "#2ca02c",  # green (human-only)
    "ai":     "#9467bd",  # purple (AI-only)
}


def human_alone_metrics(events: List[Dict], cfg: RunConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Human-only baseline:
      success = 1{Y ∈ H_t}
      size    = |H_t|
    """
    hp_cache: Dict[int, pd.DataFrame] = {}
    y_true = load_true_labels()
    succ, sizes = [], []

    for it in events:
        nz, img, pol = it["noise"], it["image"], it["policy"]
        if nz not in hp_cache:
            hp_cache[nz] = load_human_probs(nz)
        H = human_set_from_probs(hp_cache[nz].loc[img], pol)
        sizes.append(len(H))
        succ.append(1 if y_true.loc[img] in H else 0)

    return np.asarray(succ, dtype=int), np.asarray(sizes, dtype=int)


def _ai_alone_build_S_notH(model_name: str, noise_level: int) -> List[float]:
    """Half-split calibration reservoir for AI-alone (H≡∅)."""
    y_true = load_true_labels()
    ai = load_ai_probs(noise_level, model_name)
    pool = sorted(set(ai.index).intersection(y_true.index))
    if not pool:
        raise RuntimeError(f"No images for AI-alone calib at ω={noise_level}.")

    sig = ("ai-alone", model_name, noise_level)
    seed = (abs(hash(sig)) % (2**31 - 1)) or 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pool))
    mid = len(perm) // 2
    calib_idx = perm[:mid]

    S_notH = []
    for j in calib_idx:
        img = pool[j]
        y = y_true.loc[img]
        pA = ai.loc[img]
        S_notH.append(1.0 - float(pA.get(y, 0.0)))
    return S_notH


def ai_alone_replay_online_matched(
    events: List[Dict],
    cfg: RunConfig,
    ai_calib_noise: int,
    target_cov: float,
    eta: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    AI-only baseline using the same online SA logic on the 'notH' branch only.
    H_t ≡ ∅. The coverage level is updated online toward target_cov.

    Returns:
      final_level, success, size, level_trace
    """
    S_notH = list(_ai_alone_build_S_notH(cfg.model_name, ai_calib_noise))
    eta_a = float(eta if eta is not None else (cfg.eta_a if cfg.eta_a is not None else (cfg.eta or 0.05)))
    a_level = float(target_cov)

    ai_df_cache: Dict[int, pd.DataFrame] = {}
    ai_row_cache: Dict[Tuple[int, str], pd.Series] = {}
    y_true = load_true_labels()

    succ, sizes, a_trace = [], [], []

    for it in events:
        img, nz = it["image"], it["noise"]

        key = (nz, img)
        if key in ai_row_cache:
            pA = ai_row_cache[key]
        else:
            if nz not in ai_df_cache:
                ai_df_cache[nz] = load_ai_probs(nz, cfg.model_name)
            pA = ai_df_cache[nz].loc[img]
            ai_row_cache[key] = pA

        q_notH = emp_quantile(S_notH, a_level)
        Cx = conformal_set(pA, H=[], q_notH=q_notH, q_inH=1.0)

        y = y_true.loc[img]
        hit = 1.0 if (y in Cx) else 0.0
        succ.append(int(hit))
        sizes.append(len(Cx))

        s_true = 1.0 - float(pA.get(y, 0.0))
        S_notH.append(s_true)

        a_level += eta_a * (target_cov - hit)
        a_level = float(np.clip(a_level, 0.0, 1.0))
        a_trace.append(a_level)

    return a_level, np.asarray(succ, int), np.asarray(sizes, int), np.asarray(a_trace, float)


# ---- smoothing utilities ----

def _centered_local_rate(success: np.ndarray, W: int) -> np.ndarray:
    """Centered moving average for a 0/1 sequence."""
    x = success.astype(float)
    k = max(1, int(W))
    ker = np.ones(k, dtype=float)
    num = np.convolve(x, ker, mode="same")
    den = np.convolve(np.ones_like(x), ker, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def _centered_running_mean(vals: np.ndarray, W: int) -> np.ndarray:
    """Centered running mean for arbitrary numeric sequence."""
    v = vals.astype(float)
    k = max(1, int(W))
    ker = np.ones(k, dtype=float)
    num = np.convolve(v, ker, mode="same")
    den = np.convolve(np.ones_like(v), ker, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def _cumulative_rate(x: np.ndarray) -> np.ndarray:
    """Prefix mean of a 0/1 array."""
    x = x.astype(float)
    t = np.arange(1, len(x) + 1, dtype=float)
    return np.cumsum(x) / t


def _cumulative_mean(x: np.ndarray) -> np.ndarray:
    """Prefix mean of a numeric array."""
    x = x.astype(float)
    t = np.arange(1, len(x) + 1, dtype=float)
    return np.cumsum(x) / t


def _auto_zoom_ylim(series_list, pad=0.01, hard=(0.0, 1.0)):
    vals = []
    for s in series_list:
        arr = np.asarray(s, float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.extend([float(arr.min()), float(arr.max())])
    if not vals:
        return hard
    lo, hi = min(vals), max(vals)
    if hi - lo < 0.02:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.02, mid + 0.02
    lo -= pad
    hi += pad
    lo = max(hard[0], lo)
    hi = min(hard[1], hi)
    return lo, hi


# running coverage plots 

def _centered_local_conditional_coverage(mask: np.ndarray, success: np.ndarray, W: int) -> np.ndarray:
    mask = mask.astype(float)
    success = success.astype(float)
    num = np.convolve(success * mask, np.ones(W), mode="same")
    den = np.convolve(mask, np.ones(W), mode="same")
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def plot_compare_local_coverage_levels(summaryA: Dict[str, Any],
                                       summaryF: Dict[str, Any],
                                       eps: float,
                                       delta: float,
                                       phase_bounds,
                                       phase_names,
                                       W: int = 500,
                                       title_prefix: str = "",
                                       target_window: Optional[float] = None,
                                       ylim_inH=None,
                                       ylim_notH=None,
                                       auto_pad: float = 0.05,
                                       show_means: bool = True,
                                       print_diffs: bool = True) -> Dict[str, Any]:
    """
    Centered local conditional coverage for Y∈H and Y∉H:
      - adaptive vs fixed thresholds
      - with target lines 1−ε and δ
    """
    colA = "#1f77b4"   # adaptive (blue)
    colF = "crimson"   # fixed (red)
    colT = "black"     # target

    def _vlines(ax):
        for bi, nm in zip(phase_bounds, phase_names):
            ax.axvline(bi, color="k", alpha=0.25, linestyle=":")
            ax.text(bi + 6, 0.95, nm, fontsize=9, alpha=0.7,
                    transform=ax.get_xaxis_transform())

    def _lim(ax, series, explicit, tgt=None):
        if explicit is not None:
            lo, hi = explicit
        elif target_window is not None and tgt is not None:
            lo, hi = tgt - target_window, tgt + target_window
        else:
            s = np.asarray(series, float)
            s = s[np.isfinite(s)]
            if s.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = float(s.min()), float(s.max())
                rng = max(1e-6, hi - lo)
                lo -= auto_pad * rng
                hi += auto_pad * rng
        ax.set_ylim(max(0.0, lo), min(1.0, hi))

    x = np.arange(1, len(summaryA["success"]) + 1)

    covA_inH  = _centered_local_conditional_coverage(summaryA["mask_inH"],  summaryA["success"], W)
    covA_notH = _centered_local_conditional_coverage(summaryA["mask_notH"], summaryA["success"], W)
    covF_inH  = _centered_local_conditional_coverage(summaryF["mask_inH"],  summaryF["success"], W)
    covF_notH = _centered_local_conditional_coverage(summaryF["mask_notH"], summaryF["success"], W)

    muA_inH,  muF_inH  = float(np.nanmean(covA_inH)),  float(np.nanmean(covF_inH))
    muA_notH, muF_notH = float(np.nanmean(covA_notH)), float(np.nanmean(covF_notH))

    tgt_inH  = 1.0 - eps
    tgt_notH = delta
    dA_inH   = muA_inH  - tgt_inH
    dF_inH   = muF_inH  - tgt_inH
    dA_notH  = muA_notH - tgt_notH
    dF_notH  = muF_notH - tgt_notH

    # Y ∈ H
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(x, covA_inH, color=colA, label=f"CUP (online, W={W})")
    ax.plot(x, covF_inH, color=colF, label="fixed")
    ax.axhline(tgt_inH, color=colT, linestyle="--", linewidth=1.5,
               label=f"target 1−ε={tgt_inH:.2f}")
    if show_means:
        ax.axhline(muA_inH, color=colA, linestyle=":", linewidth=1.5,
                   label=f"mean online={muA_inH:.3f} (Δ={dA_inH:+.3f})")
        ax.axhline(muF_inH, color=colF, linestyle=":", linewidth=1.5,
                   label=f"mean fixed={muF_inH:.3f} (Δ={dF_inH:+.3f})")
    _vlines(ax)
    ax.set_xlabel("t")
    ax.set_ylabel("coverage")
    ax.set_title(f"{title_prefix} Local coverage given Y∈H")
    _lim(ax, np.concatenate([covA_inH, covF_inH, [tgt_inH]]), ylim_inH, tgt=tgt_inH)
    ax.legend()
    plt.show()

    # Y ∉ H
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(x, covA_notH, color=colA, label=f"CUP (online, W={W})")
    ax.plot(x, covF_notH, color=colF, label="fixed")
    ax.axhline(tgt_notH, color=colT, linestyle="--", linewidth=1.5,
               label=f"target δ={tgt_notH:.2f}")
    if show_means:
        ax.axhline(muA_notH, color=colA, linestyle=":", linewidth=1.5,
                   label=f"mean online={muA_notH:.3f} (Δ={dA_notH:+.3f})")
        ax.axhline(muF_notH, color=colF, linestyle=":", linewidth=1.5,
                   label=f"mean fixed={muF_notH:.3f} (Δ={dF_notH:+.3f})")
    _vlines(ax)
    ax.set_xlabel("t")
    ax.set_ylabel("coverage")
    ax.set_title(f"{title_prefix} Local coverage given Y∉H")
    _lim(ax, np.concatenate([covA_notH, covF_notH, [tgt_notH]]), ylim_notH, tgt=tgt_notH)
    ax.legend()
    plt.show()

    if print_diffs:
        print(f"[Y∈H] mean online = {muA_inH:.4f} (Δ={dA_inH:+.4f}), "
              f"fixed = {muF_inH:.4f} (Δ={dF_inH:+.4f}), target = {tgt_inH:.4f}")
        print(f"[Y∉H] mean online = {muA_notH:.4f} (Δ={dA_notH:+.4f}), "
              f"fixed = {muF_notH:.4f} (Δ={dF_notH:+.4f}), target = {tgt_notH:.4f}")

    return {
        "inH":  {"mean_online": muA_inH,  "mean_fixed": muF_inH,  "target": tgt_inH,
                 "delta_online": dA_inH,  "delta_fixed": dF_inH},
        "notH": {"mean_online": muA_notH, "mean_fixed": muF_notH, "target": tgt_notH,
                 "delta_online": dA_notH, "delta_fixed": dF_notH},
    }





def plot_marginal_coverage_hai_vs_baselines(
    cfg: RunConfig,
    adaptive: Dict[str, Any],
    W: int = 500,
    ai_calib_noise: Optional[int] = None,
    title_suffix: str = "Running marginal coverage  P(Y∈C)",
):
    """
    Windowed (centered) marginal coverage for:
      - CUP (online)
      - human-only
      - AI-only (matched to CUP's global marginal coverage).
    """
    events, phase_bounds, phase_names, title_prefix = collect_stream(cfg)

    human_success, _ = human_alone_metrics(events, cfg)

    if ai_calib_noise is None:
        ai_calib_noise = cfg.noise_phases[0].noise_level if cfg.mode == "noise" else cfg.fixed_noise_level

    target_cov = float(np.mean(adaptive["success"]))
    a_T, ai_succ_match, ai_size_match, a_trace = ai_alone_replay_online_matched(
        events, cfg, ai_calib_noise=ai_calib_noise, target_cov=target_cov,
        eta=cfg.eta_a if cfg.eta_a is not None else (cfg.eta or 0.05),
    )

    lc_online = _centered_local_rate(adaptive["success"], W)
    lc_human  = _centered_local_rate(human_success, W)
    lc_ai     = _centered_local_rate(ai_succ_match, W)

    m_online = float(np.nanmean(lc_online))
    m_human  = float(np.nanmean(lc_human))
    m_ai     = float(np.nanmean(lc_ai))

    t = np.arange(1, len(lc_online) + 1)
    ylo, yhi = _auto_zoom_ylim([lc_online, lc_human, lc_ai], pad=0.01)

    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    ax.plot(t, lc_online, label="CUP (online)", color=COLORS["online"], linewidth=1.8)
    ax.plot(t, lc_human,  label="Human-only",  color=COLORS["human"],  linewidth=1.8)
    ax.plot(t, lc_ai,     label=f"AI-only (matched, a*={a_T:.3f})",
            color=COLORS["ai"], linewidth=1.8)

    for xb, name in zip(phase_bounds, phase_names):
        ax.axvline(x=xb, color="k", linestyle=":", alpha=0.25)
        ax.text(xb + 8, 0.02, name, fontsize=9, alpha=0.6,
                transform=ax.get_xaxis_transform())

    ax.hlines([m_online, m_human, m_ai], xmin=1, xmax=len(t),
              linestyles=":", colors=[COLORS["online"], COLORS["human"], COLORS["ai"]], alpha=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel("coverage")
    ax.set_ylim(ylo, yhi)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{title_prefix}{title_suffix}", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    return {
        "local_online": lc_online,
        "local_human":  lc_human,
        "local_ai":     lc_ai,
        "means": dict(online=m_online, human=m_human, ai=m_ai),
        "ai_match_level": a_T,
        "phase_bounds": phase_bounds,
        "phase_names": phase_names,
    }


def plot_set_sizes_hai_vs_baselines(
    cfg: RunConfig,
    q_inH0: float,
    q_notH0: float,
    W: int = 300,
    ai_calib_noise: Optional[int] = None,
    title_suffix: str = "Prediction set size (centered running mean)",
):
    """
    Windowed running mean of |C_t(x)| for CUP vs human vs AI.
    """
    events, phase_bounds, phase_names, title_prefix = collect_stream(cfg)

    eta = float(cfg.eta if cfg.eta is not None else 0.05)
    online = replay_thresholds_with_sizes(events, cfg, q_inH0, q_notH0, adaptive=True, eta=eta)

    _, human_size = human_alone_metrics(events, cfg)

    if ai_calib_noise is None:
        ai_calib_noise = cfg.noise_phases[0].noise_level if cfg.mode == "noise" else cfg.fixed_noise_level
    target_cov = float(np.mean(online["success"]))
    a_T, ai_succ_match, ai_size_match, a_trace = ai_alone_replay_online_matched(
        events, cfg, ai_calib_noise=ai_calib_noise, target_cov=target_cov,
        eta=cfg.eta_a if cfg.eta_a is not None else (cfg.eta or 0.05),
    )

    s_online = _centered_running_mean(online["size"], W)
    s_human  = _centered_running_mean(human_size, W)
    s_ai     = _centered_running_mean(ai_size_match, W)

    t = np.arange(1, len(s_online) + 1)

    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    ax.plot(t, s_online, label=f"CUP (online, W={W})", color=COLORS["online"], linewidth=1.8)
    ax.plot(t, s_human,  label="Human-only",         color=COLORS["human"],  linewidth=1.8)
    ax.plot(t, s_ai,     label=f"AI-only (matched, a*={a_T:.3f})",
            color=COLORS["ai"], linewidth=1.8)

    for xb, name in zip(phase_bounds, phase_names):
        ax.axvline(x=xb, color="k", linestyle=":", alpha=0.25)
        ax.text(xb + 8, 0.05, name, fontsize=9, alpha=0.6,
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("t")
    ax.set_ylabel("|C_t|")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{title_prefix}{title_suffix}", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    return {
        "size_online": s_online,
        "size_human":  s_human,
        "size_ai":     s_ai,
        "ai_match_level": a_T,
        "phase_bounds": phase_bounds,
        "phase_names": phase_names,
    }




def plot_cum_marginal_coverage_hai_vs_baselines(
    cfg: RunConfig,
    adaptive: Dict[str, Any],
    ai_calib_noise: Optional[int] = None,
    events: Optional[List[Dict]] = None,
    title_suffix: str = "",
):
    """
    Prefix (cumulative) marginal coverage for CUP vs human vs AI-matched.
    Uses the same `events` sequence as `adaptive` if provided.
    """
    if events is None:
        events, phase_bounds, phase_names, title_prefix = collect_stream(cfg)
    else:
        phase_bounds, phase_names, title_prefix = [], [], ""

    human_success, _ = human_alone_metrics(events, cfg)

    if ai_calib_noise is None:
        ai_calib_noise = cfg.noise_phases[0].noise_level if cfg.mode == "noise" else cfg.fixed_noise_level
    target_cov = float(np.mean(adaptive["success"]))
    a_T, ai_succ_match, ai_size_match, a_trace = ai_alone_replay_online_matched(
        events, cfg, ai_calib_noise=ai_calib_noise, target_cov=target_cov,
        eta=cfg.eta_a if cfg.eta_a is not None else (cfg.eta or 0.05),
    )

    cm_online = _cumulative_rate(adaptive["success"])
    cm_human  = _cumulative_rate(human_success)
    cm_ai     = _cumulative_rate(ai_succ_match)

    L = min(len(cm_online), len(cm_human), len(cm_ai))
    cm_online, cm_human, cm_ai = cm_online[:L], cm_human[:L], cm_ai[:L]

    m_online = float(cm_online[-1])
    m_human  = float(cm_human[-1])
    m_ai     = float(cm_ai[-1])

    t = np.arange(1, L + 1)
    fig, ax = plt.subplots(figsize=(10.2, 3.0))

    ax.plot(t, cm_online, label="CUP (online)", color=COLORS["online"], linewidth=1.8)
    ax.plot(t, cm_human,  label="Human-only",  color=COLORS["human"],  linewidth=1.8)
    ax.plot(t, cm_ai,     label=f"AI-only (matched, a*={a_T:.3f})",
            color=COLORS["ai"], linewidth=1.8)

    ax.hlines([m_online, m_human, m_ai], xmin=1, xmax=L,
              linestyles=":", colors=[COLORS["online"], COLORS["human"], COLORS["ai"]], alpha=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.65, 1.0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{title_prefix}{title_suffix}", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    return {
        "cum_online": cm_online,
        "cum_human":  cm_human,
        "cum_ai":     cm_ai,
        "final_means": dict(online=m_online, human=m_human, ai=m_ai),
        "ai_match_level": a_T,
        "phase_bounds": phase_bounds,
        "phase_names": phase_names,
    }


def plot_cum_set_sizes_hai_vs_baselines(
    cfg: RunConfig,
    q_inH0: float,
    q_notH0: float,
    ai_calib_noise: Optional[int] = None,
    events: Optional[List[Dict]] = None,
    title_suffix: str = "",
):
    """
    Prefix (cumulative) mean set size for CUP vs human vs AI-matched.
    """
    if events is None:
        events, phase_bounds, phase_names, title_prefix = collect_stream(cfg)
    else:
        phase_bounds, phase_names, title_prefix = [], [], ""

    eta = float(cfg.eta if cfg.eta is not None else 0.05)
    online = replay_thresholds_with_sizes(events, cfg, q_inH0, q_notH0, adaptive=True, eta=eta)

    _, human_size = human_alone_metrics(events, cfg)

    if ai_calib_noise is None:
        ai_calib_noise = cfg.noise_phases[0].noise_level if cfg.mode == "noise" else cfg.fixed_noise_level
    target_cov = float(np.mean(online["success"]))
    a_T, ai_succ_match, ai_size_match, a_trace = ai_alone_replay_online_matched(
        events, cfg, ai_calib_noise=ai_calib_noise, target_cov=target_cov,
        eta=cfg.eta_a if cfg.eta_a is not None else (cfg.eta or 0.05),
    )

    s_online = _cumulative_mean(online["size"])
    s_human  = _cumulative_mean(human_size)
    s_ai     = _cumulative_mean(ai_size_match)

    L = min(len(s_online), len(s_human), len(s_ai))
    s_online, s_human, s_ai = s_online[:L], s_human[:L], s_ai[:L]
    t = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    ax.plot(t, s_online, label="CUP (online)", color=COLORS["online"], linewidth=1.8)
    ax.plot(t, s_human,  label="Human-only",  color=COLORS["human"],  linewidth=1.8)
    ax.plot(t, s_ai,     label=f"AI-only (matched, a*={a_T:.3f})",
            color=COLORS["ai"], linewidth=1.8)

    m_online = float(s_online[-1])
    m_human  = float(s_human[-1])
    m_ai     = float(s_ai[-1])

    ax.hlines([m_online, m_human, m_ai], xmin=1, xmax=L,
              linestyles=":", colors=[COLORS["online"], COLORS["human"], COLORS["ai"]], alpha=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\overline{|C|}_{1:t}$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{title_prefix}{title_suffix}", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    return {
        "cumsize_online": s_online,
        "cumsize_human":  s_human,
        "cumsize_ai":     s_ai,
        "ai_match_level": a_T,
        "phase_bounds": phase_bounds,
        "phase_names": phase_names,
    }
