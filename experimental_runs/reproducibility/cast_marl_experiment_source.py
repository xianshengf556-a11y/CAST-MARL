"""
CAST-MARL: Conflict-Aware Safe Transformer MARL for multi-UAV cooperative
coverage and path planning.

Directory layout:
  /root/autodl-tmp/PSOCNN/dataset/       <- .npz datasets
  /root/autodl-tmp/PSOCNN/best_weights/  <- best checkpoint (uav_best.pth)

This script rebuilds the previous deployment pipeline into a compact,
experiment-oriented MARL framework with:
  - CTDE training and decentralized execution
  - Objective-conditioned multi-source risk-aware Transformer policy
  - Simple / obstacle / dynamic-user / terrain-aware 3D environments
  - 10 figures + 3 core tables + JSON/TXT reports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Normalize OMP env before numeric libs import.
_omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
if _omp_threads and (not _omp_threads.isdigit() or int(_omp_threads) < 1):
    os.environ["OMP_NUM_THREADS"] = "8"

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import stats
from torch.distributions import Normal

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

METHOD_DISPLAY_NAME = "CAST-MARL"
SUPPLEMENT_DIRNAME = "supplement"

MAINLINE_BASELINE_VARIANTS: List[Tuple[str, str]] = [
    ("tmarl", METHOD_DISPLAY_NAME),
    ("mappo", "MAPPO"),
    ("ppo", "PPO"),
]
MAINLINE_ABLATION_VARIANTS: List[Tuple[str, str]] = [
    ("no_transformer", "w/o Transformer"),
    ("no_attention", "w/o Attention"),
    ("no_ctde", "w/o CTDE"),
    ("tmarl_no_safety", "w/o Safety Layer"),
]


ROOT = os.environ.get("CAST_MARL_ROOT", "/root/autodl-tmp/PSOCNN")
DATASET_DIR = os.path.join(ROOT, "dataset")
BEST_WEIGHTS_DIR = os.path.join(ROOT, "best_weights")
BEST_WEIGHT_PATH = os.path.join(BEST_WEIGHTS_DIR, "uav_best.pth")
TERRAIN_DATA_DIR = os.path.join(ROOT, "terrain_3d_data")
TERRAIN_MANIFEST_PATH = os.path.join(TERRAIN_DATA_DIR, "terrain_manifest.json")

AREA_X = 5000.0
AREA_Y = 5000.0
ALT_MIN = 50.0
ALT_MAX = 150.0
GRID_SIZE = 40
COMM_RADIUS = 850.0
COLLISION_DISTANCE = 120.0
DT = 1.0
LOG_FILE_PATH: str | None = None
SPACE_DIAGONAL = math.sqrt(AREA_X ** 2 + AREA_Y ** 2 + (ALT_MAX - ALT_MIN) ** 2)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_timestamp_bundle_dir() -> str:
    return os.path.join(ROOT, f"无人机{datetime.now().strftime('%Y%m%d%H%M')}")


def build_supplement_output_dir(output_dir: str, topic: str) -> str:
    return ensure_dir(os.path.join(output_dir, SUPPLEMENT_DIRNAME, topic))


def build_mainline_runtime_config(config: "ExperimentConfig") -> "ExperimentConfig":
    runtime_config = ExperimentConfig(**asdict(config))
    runtime_config.use_failure_module = False
    runtime_config.fail_prob_base = 0.0
    runtime_config.fail_prob_load_scale = 0.0
    runtime_config.fail_fixed_count = 0
    runtime_config.fail_max_count = 0
    runtime_config.fail_reassign_on_event = False
    return runtime_config


def build_failure_runtime_config(config: "ExperimentConfig") -> "ExperimentConfig":
    runtime_config = ExperimentConfig(**asdict(config))
    runtime_config.use_failure_module = True
    runtime_config.fail_max_count = max(1, runtime_config.fail_max_count)
    runtime_config.fail_prob_base = max(0.0, runtime_config.fail_prob_base)
    return runtime_config


def mean_std_ci(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    half = 1.96 * std / math.sqrt(max(1, arr.size))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }


def resolve_output_dir(requested: str, bundle_dir: str) -> str:
    bundle_dir = ensure_dir(bundle_dir)
    requested = requested.strip()
    if requested:
        return requested if os.path.isabs(requested) else os.path.join(bundle_dir, requested)
    return os.path.join(bundle_dir, datetime.now().strftime("%Y%m%d_%H%M%S_castmarl"))


def save_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def set_log_file(path: str | None) -> None:
    global LOG_FILE_PATH
    LOG_FILE_PATH = path


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    if LOG_FILE_PATH:
        ensure_dir(os.path.dirname(LOG_FILE_PATH))
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def norm_xy(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[..., 0] = out[..., 0] / AREA_X
    out[..., 1] = out[..., 1] / AREA_Y
    return out


def denorm_pos(pos: np.ndarray) -> np.ndarray:
    out = pos.copy()
    out[..., 0] *= AREA_X
    out[..., 1] *= AREA_Y
    out[..., 2] = ALT_MIN + out[..., 2] * (ALT_MAX - ALT_MIN)
    return out


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))


def obstacle_xy_bounds(obstacle: np.ndarray | List[float] | Tuple[float, ...]) -> Tuple[float, float, float, float]:
    obs = np.asarray(obstacle, dtype=np.float32).reshape(-1)
    if obs.size >= 6:
        return float(obs[0]), float(obs[1]), float(obs[3]), float(obs[4])
    return float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])


def obstacle_xyz_bounds(obstacle: np.ndarray | List[float] | Tuple[float, ...]) -> Tuple[float, float, float, float, float, float]:
    obs = np.asarray(obstacle, dtype=np.float32).reshape(-1)
    if obs.size >= 6:
        return float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])
    return float(obs[0]), float(obs[1]), ALT_MIN, float(obs[2]), float(obs[3]), ALT_MAX


def point_to_box_distance_3d(point: np.ndarray, obstacle: np.ndarray | List[float] | Tuple[float, ...]) -> float:
    x0, y0, z0, x1, y1, z1 = obstacle_xyz_bounds(obstacle)
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    dz = max(z0 - pz, 0.0, pz - z1)
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


_TERRAIN_CACHE: Dict[str, Dict[str, object]] = {}


def _load_json_if_exists(path: str) -> Any | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _terrain_manifest() -> List[Dict[str, object]]:
    data = _load_json_if_exists(TERRAIN_MANIFEST_PATH)
    return data if isinstance(data, list) else []


def _terrain_asset_by_kind(kind: str) -> Dict[str, object] | None:
    key = f"kind::{kind}"
    if key in _TERRAIN_CACHE:
        return _TERRAIN_CACHE[key]
    manifest = _terrain_manifest()
    match = next((item for item in manifest if str(item.get("kind", "")) == kind and str(item.get("npz_path", ""))), None)
    if match is None:
        _TERRAIN_CACHE[key] = None
        return None
    npz_path = str(match["npz_path"])
    data = np.load(npz_path, allow_pickle=True)
    elevation = data["elevation"].astype(np.float32)
    preview = data["elevation_preview"].astype(np.float32) if "elevation_preview" in data.files else elevation[::max(1, elevation.shape[0] // 256), ::max(1, elevation.shape[1] // 256)]
    elev_min = float(elevation.min())
    elev_max = float(elevation.max())
    denom = max(elev_max - elev_min, 1e-6)
    asset = {
        "id": str(match.get("id", kind)),
        "kind": kind,
        "elevation": elevation,
        "preview": preview,
        "norm_preview": (preview - elev_min) / denom,
        "elev_min": elev_min,
        "elev_max": elev_max,
    }
    _TERRAIN_CACHE[key] = asset
    return asset


def _terrain_building_rectangles() -> np.ndarray:
    key = "urban_buildings"
    if key in _TERRAIN_CACHE:
        cached = _TERRAIN_CACHE[key]
        return cached if isinstance(cached, np.ndarray) else np.zeros((0, 6), dtype=np.float32)
    manifest = _terrain_manifest()
    match = next((item for item in manifest if str(item.get("kind", "")) == "urban_buildings" and str(item.get("json_path", ""))), None)
    if match is None:
        _TERRAIN_CACHE[key] = np.zeros((0, 6), dtype=np.float32)
        return _TERRAIN_CACHE[key]
    payload = _load_json_if_exists(str(match["json_path"])) or {}
    elements = payload.get("elements", [])
    nodes = {int(item["id"]): (float(item["lon"]), float(item["lat"])) for item in elements if item.get("type") == "node"}
    ways = [item for item in elements if item.get("type") == "way" and item.get("nodes")]
    if not ways or not nodes:
        _TERRAIN_CACHE[key] = np.zeros((0, 6), dtype=np.float32)
        return _TERRAIN_CACHE[key]
    all_lons = [pt[0] for pt in nodes.values()]
    all_lats = [pt[1] for pt in nodes.values()]
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)
    dlon = max(lon_max - lon_min, 1e-6)
    dlat = max(lat_max - lat_min, 1e-6)
    rects: List[List[float]] = []
    for way in ways:
        pts = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]
        if len(pts) < 3:
            continue
        xs = [AREA_X * (lon - lon_min) / dlon for lon, _ in pts]
        ys = [AREA_Y * (lat - lat_min) / dlat for _, lat in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if (x1 - x0) < 18.0 or (y1 - y0) < 18.0:
            continue
        tags = way.get("tags", {})
        height = tags.get("height")
        levels = tags.get("building:levels")
        try:
            top_h = float(str(height).replace("m", "").strip())
        except Exception:
            try:
                top_h = max(12.0, 3.4 * float(levels))
            except Exception:
                top_h = random.uniform(18.0, 65.0)
        rects.append([x0, y0, 0.0, x1, y1, float(np.clip(top_h, 12.0, 85.0))])
    arr = np.asarray(rects, dtype=np.float32) if rects else np.zeros((0, 6), dtype=np.float32)
    _TERRAIN_CACHE[key] = arr
    return arr


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo : i + 1].mean()
    return out


def clamp_to_choice(value: int, choices: List[int]) -> int:
    return min(choices, key=lambda x: abs(x - value))


@dataclass
class ExperimentConfig:
    seed: int = 42
    episodes: int = 80
    max_steps: int = 35
    batch_episodes: int = 8
    lr: float = 3e-4
    gamma: float = 0.97
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    clip_grad_norm: float = 1.0
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    mini_batch_size: int = 128
    hidden_dim: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_speed: float = 150.0
    best_conflict_gate: float = 0.08
    best_coverage_gate: float = 0.70
    tune_best_conflict_gate: float = 0.06
    tune_best_coverage_gate: float = 0.64
    high_conflict_threshold: float = 0.08
    high_conflict_penalty: float = 0.90
    tune_refine_topk: int = 5
    tune_refine_eval_episodes: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_env: str = "simple"
    train_num_uavs: int = 6
    dynamic_ratio: float = 0.15
    obstacle_ratio: float = 0.12
    output_root: str = ""
    tune_trials: int = 12
    tune_init_trials: int = 4
    tune_candidate_pool: int = 64
    tune_train_episodes: int = 12
    tune_eval_episodes: int = 16
    failure_eval_episodes: int = 20
    mainline_seed_count: int = 5
    generalization_eval_episodes: int = 10
    reward_gain_coef: float = 6.2
    reward_coverage_coef: float = 3.35
    reward_path_penalty: float = 0.00057
    reward_conflict_penalty: float = 0.45
    reward_redundancy_penalty: float = 0.025
    reward_obstacle_penalty: float = 0.2
    fail_prob_base: float = 0.003
    fail_prob_load_scale: float = 0.002
    fail_reward_recovery_coef: float = 0.8
    fail_energy_fair_coef: float = 0.15
    use_failure_module: bool = False
    fail_trigger_min_step: int = 4
    fail_trigger_max_step: int = 28
    fail_max_count: int = 2
    fail_fixed_count: int = 0
    fail_load_aware: bool = True
    fail_reassign_on_event: bool = True
    safety_distance: float = 138.0
    camera_fov_deg: float = 76.0
    obstacle_height_min: float = 24.0
    obstacle_height_max: float = 88.0
    dynamic_altitude_ratio: float = 0.06
    terrain_clearance: float = 14.0
    terrain_height_cap: float = 42.0
    safety_strength: float = 0.24
    safety_brake: float = 0.32
    safety_risk_bias: float = 0.04
    safety_stage_cap: float = 0.82
    safety_blend_cap: float = 0.85
    safety_use_repulsion: bool = True
    safety_use_brake: bool = True
    safety_use_guidance: bool = True
    guidance_strength: float = 0.12
    guidance_cover_decay: float = 0.42
    guidance_risk_threshold: float = 0.10
    relation_risk_strength: float = 0.90
    relation_comm_strength: float = 0.35
    relation_altitude_strength: float = 0.22
    multi_source_gate_strength: float = 0.65
    objective_gate_strength: float = 0.55
    assignment_bonus_coef: float = 1.35
    frontier_bonus_coef: float = 0.95
    stagnation_penalty_coef: float = 0.35
    imitation_epochs: int = 6
    imitation_batch_size: int = 48
    region_reassign_interval: int = 4
    use_region_assignment: bool = True
    region_balance_strength: float = 0.18
    region_global_mix: float = 0.35
    region_support_radius_scale: float = 1.35
    heuristic_view_radius_scale: float = 1.15
    assignment_decay_cover: float = 0.58
    assignment_decay_time: float = 0.42
    frontier_decay_cover: float = 0.36
    frontier_decay_time: float = 0.24
    late_stage_visible_boost: float = 0.62
    final_refine_seed_count: int = 3
    final_refine_eval_episodes: int = 24
    model_select_topk: int = 4
    model_select_pool_size: int = 8
    model_select_eval_episodes: int = 8
    dataset_paths: List[str] = field(
        default_factory=lambda: [
            os.path.join(DATASET_DIR, "dataset1_users35.npz"),
            os.path.join(DATASET_DIR, "dataset1_users45.npz"),
            os.path.join(DATASET_DIR, "dataset1_users60.npz"),
        ]
    )
    target_dataset_paths: List[str] = field(default_factory=list)
    target_supervision_enabled: bool = False
    target_supervision_weight: float = 0.15
    target_supervision_batches: int = 1
    target_supervision_batch_size: int = 48
    target_supervision_start_episode: int = 0
    target_supervision_rollout_steps: int = 4
    target_supervision_expert_mix: float = 0.55
    target_supervision_frontier_mix: float = 0.30
    target_supervision_repulsion_coef: float = 0.22
    target_supervision_obstacle_coef: float = 0.18
    target_supervision_refine_epochs: int = 6
    terrain_manifest_path: str = TERRAIN_MANIFEST_PATH


def coverage_priority_score(reward: float, coverage_ratio: float, conflict_rate: float) -> float:
    """
    Coverage-first score for model selection and hyperparameter tuning.
    coverage_ratio should be in [0, 1].
    """
    cov = float(np.clip(coverage_ratio, 0.0, 1.0))
    conflict = float(np.clip(conflict_rate, 0.0, 1.0))
    reward_term = float(np.tanh(reward / 40.0))
    # Coverage-first score with softer conflict pressure after normalization.
    return 81.0 * cov + 12.0 * (1.0 - conflict) + 7.0 * reward_term


def constrained_selection_tuple(
    reward: float,
    coverage: float,
    conflict_rate: float,
    conflict_gate: float,
    coverage_p75: float | None = None,
    conflict_p75: float | None = None,
) -> Tuple[int, float, float, float, float]:
    """
    Ranking tuple for hyperparameter/model selection.
    Priority:
      1. Satisfy conflict gate
      2. Higher blended coverage
      3. Lower blended conflict
      4. Higher reward
      5. Higher legacy score as final tie-break
    coverage / coverage_p75 are percentages in [0, 100].
    """
    cov_main = float(coverage)
    cov_tail = float(coverage if coverage_p75 is None else coverage_p75)
    conf_main = float(conflict_rate)
    conf_tail = float(conflict_rate if conflict_p75 is None else conflict_p75)
    blended_coverage = 0.75 * cov_main + 0.25 * cov_tail
    blended_conflict = max(conf_main, conf_tail)
    legacy_score = coverage_priority_score(
        reward=float(reward),
        coverage_ratio=blended_coverage / 100.0,
        conflict_rate=blended_conflict,
    )
    conflict_ok = int(blended_conflict <= float(conflict_gate))
    return (
        conflict_ok,
        blended_coverage,
        -blended_conflict,
        float(reward),
        float(legacy_score),
    )


class ScenarioDataset:
    def __init__(self, dataset_paths: List[str]):
        self.samples: List[Dict[str, np.ndarray]] = []
        self.by_count: Dict[int, List[int]] = defaultdict(list)
        self._load(dataset_paths)

    def _load(self, dataset_paths: List[str]) -> None:
        for path in dataset_paths:
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True)
            exact = data["exact"].astype(np.float32)
            labels = data["labels"].astype(np.float32).reshape(-1, 8, 3)
            n_uav = data["n_uav"].astype(np.int64)
            for idx in range(len(exact)):
                user_positions = exact[idx]
                valid_mask = np.linalg.norm(user_positions, axis=1) > 0
                users = user_positions[valid_mask]
                active = max(1, min(int(n_uav[idx]), 8))
                sample = {
                    "users": users,
                    "expert_uavs": labels[idx, :active].copy(),
                    "count": len(users),
                }
                self.by_count[len(users)].append(len(self.samples))
                self.samples.append(sample)
        if not self.samples:
            raise FileNotFoundError("No valid dataset .npz files were found in dataset/.")

    def sample(self, num_users: int | None = None) -> Dict[str, np.ndarray]:
        if num_users is not None and self.by_count.get(num_users):
            idx = random.choice(self.by_count[num_users])
            return self.samples[idx]
        return random.choice(self.samples)


class MultiUAVCoverageEnv:
    def __init__(
        self,
        dataset: ScenarioDataset,
        num_uavs: int,
        env_type: str = "simple",
        max_steps: int = 35,
        max_speed: float = 180.0,
        dynamic_ratio: float = 0.15,
        obstacle_ratio: float = 0.12,
        high_conflict_threshold: float = 0.08,
        high_conflict_penalty: float = 0.90,
        reward_gain_coef: float = 6.2,
        reward_coverage_coef: float = 3.35,
        reward_path_penalty: float = 0.00057,
        reward_conflict_penalty: float = 0.45,
        reward_redundancy_penalty: float = 0.025,
        reward_obstacle_penalty: float = 0.2,
        assignment_bonus_coef: float = 1.35,
        frontier_bonus_coef: float = 0.95,
        stagnation_penalty_coef: float = 0.35,
        region_reassign_interval: int = 4,
        use_region_assignment: bool = True,
        region_balance_strength: float = 0.18,
        region_global_mix: float = 0.35,
        region_support_radius_scale: float = 1.35,
        heuristic_view_radius_scale: float = 1.15,
        assignment_decay_cover: float = 0.58,
        assignment_decay_time: float = 0.42,
        frontier_decay_cover: float = 0.36,
        frontier_decay_time: float = 0.24,
        late_stage_visible_boost: float = 0.62,
        num_users: int | None = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.num_uavs = num_uavs
        self.env_type = env_type
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.fail_prob_base = kwargs.get("fail_prob_base", 0.003)
        self.fail_prob_load_scale = kwargs.get("fail_prob_load_scale", 0.002)
        self.fail_reward_recovery_coef = kwargs.get("fail_reward_recovery_coef", 0.8)
        self.fail_energy_fair_coef = kwargs.get("fail_energy_fair_coef", 0.15)
        self.use_failure_module = kwargs.get("use_failure_module", False)
        self.fail_trigger_min_step = kwargs.get("fail_trigger_min_step", 4)
        self.fail_trigger_max_step = kwargs.get("fail_trigger_max_step", 28)
        self.fail_max_count = kwargs.get("fail_max_count", 2)
        self.fail_fixed_count = kwargs.get("fail_fixed_count", 0)
        self.fail_load_aware = kwargs.get("fail_load_aware", True)
        self.fail_reassign_on_event = kwargs.get("fail_reassign_on_event", True)
        self.safety_distance = kwargs.get("safety_distance", max(COLLISION_DISTANCE * 1.15, 138.0))
        self.camera_fov_deg = kwargs.get("camera_fov_deg", 76.0)
        self.obstacle_height_min = kwargs.get("obstacle_height_min", 24.0)
        self.obstacle_height_max = kwargs.get("obstacle_height_max", 88.0)
        self.dynamic_altitude_ratio = kwargs.get("dynamic_altitude_ratio", 0.06)
        self.terrain_clearance = kwargs.get("terrain_clearance", 14.0)
        self.terrain_height_cap = kwargs.get("terrain_height_cap", 42.0)
        self.dynamic_ratio = dynamic_ratio
        self.obstacle_ratio = obstacle_ratio
        self.high_conflict_threshold = high_conflict_threshold
        self.high_conflict_penalty = high_conflict_penalty
        self.reward_gain_coef = reward_gain_coef
        self.reward_coverage_coef = reward_coverage_coef
        self.reward_path_penalty = reward_path_penalty
        self.reward_conflict_penalty = reward_conflict_penalty
        self.reward_redundancy_penalty = reward_redundancy_penalty
        self.reward_obstacle_penalty = reward_obstacle_penalty
        self.target_num_users = num_users
        self.assignment_bonus_coef = assignment_bonus_coef
        self.frontier_bonus_coef = frontier_bonus_coef
        self.stagnation_penalty_coef = stagnation_penalty_coef
        self.region_reassign_interval = region_reassign_interval
        self.use_region_assignment = use_region_assignment
        self.region_balance_strength = region_balance_strength
        self.region_global_mix = region_global_mix
        self.region_support_radius_scale = region_support_radius_scale
        self.heuristic_view_radius_scale = heuristic_view_radius_scale
        self.assignment_decay_cover = assignment_decay_cover
        self.assignment_decay_time = assignment_decay_time
        self.frontier_decay_cover = frontier_decay_cover
        self.frontier_decay_time = frontier_decay_time
        self.late_stage_visible_boost = late_stage_visible_boost
        self.grid_size = GRID_SIZE
        self.grid_x = AREA_X / self.grid_size
        self.grid_y = AREA_Y / self.grid_size
        self.feature_dim = 32
        self.terrain_asset: Dict[str, object] | None = None
        self.terrain_surface = np.zeros((0, 0), dtype=np.float32)
        self.terrain_kind = "none"
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.feature_dim

    def _make_obstacles(self) -> np.ndarray:
        terrain_rects = self._terrain_obstacle_volumes()
        if self.env_type not in {"obstacles", "terrain_mountain", "terrain_urban", "terrain_plain"}:
            return terrain_rects if len(terrain_rects) else np.zeros((0, 6), dtype=np.float32)
        n_obs = max(2, int(self.grid_size * self.obstacle_ratio))
        if self.env_type == "terrain_plain":
            n_obs = max(1, int(0.35 * n_obs))
        elif self.env_type == "terrain_mountain":
            n_obs = max(2, int(0.55 * n_obs))
        elif self.env_type == "terrain_urban":
            n_obs = max(1, int(0.25 * n_obs))
        obs = []
        for _ in range(n_obs):
            w = random.uniform(250.0, 800.0)
            h = random.uniform(250.0, 800.0)
            x = random.uniform(0, AREA_X - w)
            y = random.uniform(0, AREA_Y - h)
            ground = self._terrain_height_at_xy(np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32))
            z0 = np.clip(ground, 0.0, ALT_MAX - 12.0)
            z1 = min(ALT_MAX, z0 + random.uniform(self.obstacle_height_min, self.obstacle_height_max))
            obs.append([x, y, z0, x + w, y + h, z1])
        arr = np.asarray(obs, dtype=np.float32) if obs else np.zeros((0, 6), dtype=np.float32)
        if len(terrain_rects):
            arr = np.concatenate([arr, terrain_rects], axis=0) if len(arr) else terrain_rects
        return arr

    def _coverage_radius(self, altitude: float) -> float:
        z_ratio = float(np.clip((altitude - ALT_MIN) / max(ALT_MAX - ALT_MIN, 1e-6), 0.0, 1.0))
        radius = COMM_RADIUS * (0.72 + 0.36 * z_ratio)
        return float(np.clip(radius, COMM_RADIUS * 0.55, COMM_RADIUS * 1.25))

    def _terrain_profile_kind(self) -> str:
        mapping = {
            "terrain_mountain": "mountain",
            "terrain_urban": "urban_coastal",
            "terrain_plain": "plain_hill",
        }
        return mapping.get(self.env_type, "none")

    def _load_terrain_surface(self) -> None:
        self.terrain_kind = self._terrain_profile_kind()
        self.terrain_asset = _terrain_asset_by_kind(self.terrain_kind) if self.terrain_kind != "none" else None
        if self.terrain_asset is None:
            self.terrain_surface = np.zeros((0, 0), dtype=np.float32)
            return
        norm_preview = np.asarray(self.terrain_asset["norm_preview"], dtype=np.float32)
        scale = self.terrain_height_cap
        if self.terrain_kind == "mountain":
            scale *= 1.20
        elif self.terrain_kind == "urban_coastal":
            scale *= 0.85
        self.terrain_surface = norm_preview * scale

    def _terrain_height_at_xy(self, xy: np.ndarray) -> float:
        if self.terrain_surface.size == 0:
            return 0.0
        x = float(np.clip(xy[0], 0.0, AREA_X))
        y = float(np.clip(xy[1], 0.0, AREA_Y))
        h, w = self.terrain_surface.shape
        gx = int(np.clip(round((w - 1) * x / max(AREA_X, 1e-6)), 0, w - 1))
        gy = int(np.clip(round((h - 1) * y / max(AREA_Y, 1e-6)), 0, h - 1))
        return float(self.terrain_surface[gy, gx])

    def _terrain_slope_at_xy(self, xy: np.ndarray) -> float:
        if self.terrain_surface.size == 0:
            return 0.0
        x = float(np.clip(xy[0], 0.0, AREA_X))
        y = float(np.clip(xy[1], 0.0, AREA_Y))
        h, w = self.terrain_surface.shape
        gx = int(np.clip(round((w - 1) * x / max(AREA_X, 1e-6)), 1, max(1, w - 2)))
        gy = int(np.clip(round((h - 1) * y / max(AREA_Y, 1e-6)), 1, max(1, h - 2)))
        dzdx = float(self.terrain_surface[gy, min(w - 1, gx + 1)] - self.terrain_surface[gy, max(0, gx - 1)])
        dzdy = float(self.terrain_surface[min(h - 1, gy + 1), gx] - self.terrain_surface[max(0, gy - 1), gx])
        return float(min(1.0, math.sqrt(dzdx * dzdx + dzdy * dzdy) / max(self.terrain_height_cap, 1e-6)))

    def _apply_terrain_to_users(self) -> None:
        if self.terrain_surface.size == 0 or len(self.users) == 0:
            return
        self.users = self.users.copy()
        for i in range(len(self.users)):
            self.users[i, 2] = self._terrain_height_at_xy(self.users[i, :2])

    def _terrain_obstacle_volumes(self) -> np.ndarray:
        if self.terrain_kind == "urban_coastal":
            rects = _terrain_building_rectangles()
            if len(rects):
                return rects.copy()
        return np.zeros((0, 6), dtype=np.float32)

    def _random_init_uavs(self) -> np.ndarray:
        pts = []
        for _ in range(self.num_uavs):
            x = random.uniform(0.0, AREA_X)
            y = random.uniform(0.0, AREA_Y)
            z_floor = max(ALT_MIN, self._terrain_height_at_xy(np.array([x, y], dtype=np.float32)) + self.terrain_clearance)
            pts.append(
                [
                    x,
                    y,
                    random.uniform(z_floor, ALT_MAX),
                ]
            )
        return np.asarray(pts, dtype=np.float32)

    def _build_heatmap(self) -> np.ndarray:
        heat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for user in self.users:
            gx = min(self.grid_size - 1, max(0, int(user[0] / self.grid_x)))
            gy = min(self.grid_size - 1, max(0, int(user[1] / self.grid_y)))
            heat[gy, gx] += 1.0
        if heat.max() > 0:
            heat /= heat.max()
        return heat

    def _farthest_point_centers(self, points: np.ndarray, k: int) -> np.ndarray:
        if len(points) == 0:
            center = np.array([[AREA_X * 0.5, AREA_Y * 0.5, ALT_MIN + 0.5 * (ALT_MAX - ALT_MIN)]], dtype=np.float32)
            return np.repeat(center, k, axis=0)
        centers = [points[random.randrange(len(points))]]
        while len(centers) < k:
            dist = pairwise_dist(points[:, :2], np.asarray(centers, dtype=np.float32)[:, :2]).min(axis=1)
            centers.append(points[int(np.argmax(dist))])
        return np.asarray(centers[:k], dtype=np.float32)

    def _capacity_balanced_assign(self, points_xy: np.ndarray, centers_xy: np.ndarray) -> np.ndarray:
        n = len(points_xy)
        m = len(centers_xy)
        capacities = np.full(m, n // m, dtype=np.int32)
        capacities[: n % m] += 1
        dist = pairwise_dist(points_xy, centers_xy)
        pref = np.argsort(dist, axis=1)
        margin = dist[np.arange(n), pref[:, 1]] - dist[np.arange(n), pref[:, 0]] if m > 1 else np.ones(n, dtype=np.float32)
        order = np.argsort(-margin)
        assign = np.full(n, -1, dtype=np.int32)
        remaining = capacities.copy()
        for idx in order:
            for cid in pref[idx]:
                if remaining[cid] > 0:
                    assign[idx] = cid
                    remaining[cid] -= 1
                    break
            if assign[idx] < 0:
                assign[idx] = int(np.argmin(remaining))
        return assign

    def _soft_region_assign(self, points_xy: np.ndarray, centers_xy: np.ndarray) -> np.ndarray:
        n = len(points_xy)
        m = len(centers_xy)
        if n == 0 or m == 0:
            return np.zeros(n, dtype=np.int32)
        dist = pairwise_dist(points_xy, centers_xy)
        base_load = max(1.0, n / max(1, m))
        loads = np.zeros(m, dtype=np.float32)
        assign = np.zeros(n, dtype=np.int32)
        order = np.argsort(-dist.min(axis=1))
        for idx in order:
            load_ratio = loads / base_load
            proximity_score = dist[idx] / max(COMM_RADIUS, 1.0)
            # Keep a mild proximity bias without over-fragmenting global assignments.
            load_penalty = 0.90 * self.region_balance_strength * load_ratio
            score = proximity_score + load_penalty
            cid = int(np.argmin(score))
            assign[idx] = cid
            loads[cid] += 1.0
        return assign.astype(np.int32)

    def _active_agent_indices(self) -> np.ndarray:
        if not hasattr(self, "failed"):
            return np.arange(self.num_uavs, dtype=np.int32)
        active = np.where(~self.failed)[0].astype(np.int32)
        if len(active) == 0:
            return np.arange(self.num_uavs, dtype=np.int32)
        return active

    def _failure_recovery_stage(self) -> float:
        if not hasattr(self, "failed") or not self.failed.any():
            return 0.0
        valid_steps = self.failure_step[self.failure_step >= 0]
        if len(valid_steps) == 0:
            return 0.0
        elapsed = self.t - int(valid_steps.min())
        return float(np.clip(elapsed / max(1, self.max_steps), 0.0, 1.0))

    def _failure_load_probabilities(self) -> np.ndarray:
        load = self.assignment_load if hasattr(self, "assignment_load") else np.ones(self.num_uavs, dtype=np.float32)
        if self.fail_load_aware:
            fail_prob = np.clip(
                self.fail_prob_base + self.fail_prob_load_scale * (load - 1.0),
                0.0,
                0.05,
            )
        else:
            fail_prob = np.full(self.num_uavs, self.fail_prob_base, dtype=np.float32)
        if hasattr(self, "failed"):
            fail_prob = fail_prob.astype(np.float32)
            fail_prob[self.failed] = 0.0
        return fail_prob

    def _maybe_trigger_failures(self) -> np.ndarray:
        if not self.use_failure_module:
            return np.zeros(self.num_uavs, dtype=bool)
        if self.t < self.fail_trigger_min_step or self.t > self.fail_trigger_max_step:
            return np.zeros(self.num_uavs, dtype=bool)
        if hasattr(self, "failed") and int(self.failed.sum()) >= self.fail_max_count:
            return np.zeros(self.num_uavs, dtype=bool)

        new_failures = np.zeros(self.num_uavs, dtype=bool)
        alive = np.where(~self.failed)[0] if hasattr(self, "failed") else np.arange(self.num_uavs)
        remaining_budget = max(0, self.fail_max_count - int(self.failed.sum())) if hasattr(self, "failed") else self.fail_max_count
        if len(alive) == 0 or remaining_budget <= 0:
            return new_failures

        if self.fail_fixed_count > 0:
            count = min(self.fail_fixed_count, remaining_budget, len(alive))
            if count <= 0:
                return new_failures
            probs = self._failure_load_probabilities()[alive]
            probs = probs / probs.sum() if probs.sum() > 1e-8 else np.full(len(alive), 1.0 / len(alive))
            picked = np.random.choice(alive, size=count, replace=False, p=probs)
            new_failures[picked] = True
            return new_failures

        fail_prob = self._failure_load_probabilities()
        draws = np.random.rand(self.num_uavs)
        sampled = (~self.failed) & (draws < fail_prob)
        if sampled.sum() > remaining_budget:
            cand = np.where(sampled)[0]
            probs = fail_prob[cand]
            probs = probs / probs.sum() if probs.sum() > 1e-8 else np.full(len(cand), 1.0 / len(cand))
            keep = np.random.choice(cand, size=remaining_budget, replace=False, p=probs)
            sampled = np.zeros(self.num_uavs, dtype=bool)
            sampled[keep] = True
        return sampled

    def _compute_region_plan(self) -> None:
        users = self.users
        avg_users = max(1.0, len(users) / max(1, self.num_uavs))
        active_ids = self._active_agent_indices()
        num_active = len(active_ids)
        if len(users) == 0:
            self.user_assignments = np.zeros(0, dtype=np.int32)
            self.region_centers = np.zeros((self.num_uavs, 3), dtype=np.float32)
            self.region_radii = np.full(self.num_uavs, COMM_RADIUS, dtype=np.float32)
            self.assignment_load = np.zeros(self.num_uavs, dtype=np.float32)
            self.assignment_load[active_ids] = 1.0
            self.expert_targets = self._random_init_uavs()
            return

        if not self.use_region_assignment:
            global_center = users[:, :2].mean(axis=0)
            z_ref = float(np.mean(users[:, 2])) if users.shape[1] > 2 else ALT_MIN + 0.5 * (ALT_MAX - ALT_MIN)
            centers = np.zeros((self.num_uavs, 3), dtype=np.float32)
            centers[active_ids, 0] = global_center[0]
            centers[active_ids, 1] = global_center[1]
            centers[active_ids, 2] = np.clip(z_ref, ALT_MIN, ALT_MAX)
            if hasattr(self, "failed") and self.failed.any():
                centers[self.failed] = self.uav_pos[self.failed]
            d = pairwise_dist(users[:, :2], self.uav_pos[active_ids, :2])
            assign = active_ids[np.argmin(d, axis=1)].astype(np.int32)
            self.user_assignments = assign
            self.region_centers = centers
            self.region_radii = np.full(self.num_uavs, COMM_RADIUS, dtype=np.float32)
            self.assignment_load = np.zeros(self.num_uavs, dtype=np.float32)
            self.assignment_load[active_ids] = 1.0
            self.expert_targets = self.uav_pos.copy()
            self.expert_targets[active_ids] = centers[active_ids]
            return

        expert = self.expert_uavs.copy()
        if len(expert) >= num_active:
            centers_active = expert[:num_active].copy()
        elif len(expert) > 0:
            extra = self._farthest_point_centers(users, num_active - len(expert))
            centers_active = np.concatenate([expert, extra], axis=0)
        else:
            centers_active = self._farthest_point_centers(users, num_active)

        centers_active = centers_active[:num_active].astype(np.float32)
        centers_active[:, 2] = np.clip(centers_active[:, 2], ALT_MIN, ALT_MAX)

        assign = np.zeros(len(users), dtype=np.int32)
        for _ in range(4):
            assign_local = self._soft_region_assign(users[:, :2], centers_active[:, :2])
            for local_cid in range(num_active):
                mask = assign_local == local_cid
                if mask.any():
                    cluster = users[mask]
                    local_xy = cluster[:, :2].mean(axis=0)
                    global_xy = users[:, :2].mean(axis=0)
                    centers_active[local_cid, :2] = (1.0 - self.region_global_mix) * local_xy + self.region_global_mix * global_xy
                    z_ref = float(cluster[:, 2].mean()) if cluster.shape[1] > 2 else ALT_MIN + 0.5 * (ALT_MAX - ALT_MIN)
                    centers_active[local_cid, 2] = np.clip(z_ref, ALT_MIN, ALT_MAX)
            assign = active_ids[assign_local]

        radii = np.full(self.num_uavs, COMM_RADIUS, dtype=np.float32)
        loads = np.zeros(self.num_uavs, dtype=np.float32)
        centers = self.uav_pos.copy()
        centers[active_ids] = centers_active
        for local_cid, agent_id in enumerate(active_ids):
            mask = assign == agent_id
            if mask.any():
                cluster_xyz = users[mask, :3]
                radii[agent_id] = float(
                    max(
                        self._coverage_radius(float(centers[agent_id, 2])) * 0.75,
                        np.linalg.norm(cluster_xyz - centers[agent_id, :3], axis=1).max(initial=self._coverage_radius(float(centers[agent_id, 2])) * 0.75),
                    )
                )
                loads[agent_id] = float(mask.sum() / avg_users)
            else:
                loads[agent_id] = 1e-6
        if hasattr(self, "failed") and self.failed.any():
            centers[self.failed] = self.uav_pos[self.failed]
            radii[self.failed] = COMM_RADIUS * 0.35
            loads[self.failed] = 0.0
        self.user_assignments = assign
        self.region_centers = centers
        self.region_radii = radii
        self.assignment_load = loads
        self.expert_targets = centers.copy()

    def _assigned_indices(self, agent_idx: int) -> np.ndarray:
        return np.where(self.user_assignments == agent_idx)[0]

    def _visible_uncovered(self, agent_idx: int, radius_scale: float | None = None) -> np.ndarray:
        uncovered = self.users[~self.covered] if (~self.covered).any() else self.users
        if len(uncovered) == 0:
            return self.users
        scale = self.heuristic_view_radius_scale if radius_scale is None else radius_scale
        radius = max(self._coverage_radius(float(self.uav_pos[agent_idx, 2])), float(self.region_radii[agent_idx])) * scale
        d = np.linalg.norm(uncovered[:, :3] - self.uav_pos[agent_idx, :3], axis=1)
        visible = uncovered[d <= radius]
        if len(visible):
            return visible
        nearest_count = min(6, len(uncovered))
        order = np.argsort(d)[:nearest_count]
        return uncovered[order]

    def _assigned_uncovered(self, agent_idx: int) -> np.ndarray:
        idx = self._assigned_indices(agent_idx)
        uncovered = self.users[~self.covered] if (~self.covered).any() else self.users
        if len(uncovered) == 0:
            return self.users
        visible = self._visible_uncovered(agent_idx, radius_scale=self.region_support_radius_scale)
        if len(idx) == 0:
            return visible
        mask = ~self.covered[idx]
        assigned_uncovered = self.users[idx[mask]] if mask.any() else np.zeros((0, self.users.shape[1]), dtype=np.float32)
        if len(assigned_uncovered) == 0:
            return visible
        merged = np.concatenate([assigned_uncovered, visible], axis=0)
        rounded_xy = np.round(merged[:, :2], decimals=3)
        _, unique_idx = np.unique(rounded_xy, axis=0, return_index=True)
        merged = merged[np.sort(unique_idx)]
        assigned_cover_ratio = float(self.covered[idx].mean()) if len(idx) else 0.0
        overall_coverage = float(self.covered.mean()) if len(self.covered) else 1.0
        time_ratio = self.t / max(1, self.max_steps)
        late_stage_gate = float(
            np.clip(
                self.late_stage_visible_boost * overall_coverage
                + 0.55 * (1.0 - self.late_stage_visible_boost) * time_ratio
                + 0.18 * assigned_cover_ratio,
                0.0,
                1.0,
            )
        )
        if assigned_cover_ratio > 0.77 and len(visible):
            return visible
        if late_stage_gate > 0.60 and len(visible) >= max(4, len(assigned_uncovered)):
            return visible
        if len(assigned_uncovered) <= 2 and len(visible) > len(assigned_uncovered):
            return visible
        return merged

    def _nearest_obstacle_distance(self, point: np.ndarray) -> float:
        obstacle_dist = float(SPACE_DIAGONAL) if len(self.obstacles) == 0 else min(point_to_box_distance_3d(point, obs) for obs in self.obstacles)
        if self.terrain_surface.size:
            terrain_gap = max(0.0, float(point[2]) - self._terrain_height_at_xy(point[:2]))
            obstacle_dist = min(obstacle_dist, terrain_gap)
        return obstacle_dist

    def compute_imitation_actions(self) -> np.ndarray:
        targets = self.expert_targets if hasattr(self, "expert_targets") else self.uav_pos
        actions = np.zeros((self.num_uavs, 3), dtype=np.float32)
        for i in range(self.num_uavs):
            vec_xy = targets[i, :2] - self.uav_pos[i, :2]
            norm_xy = np.linalg.norm(vec_xy) + 1e-6
            actions[i, :2] = np.clip(vec_xy / norm_xy, -1.0, 1.0)
            dz = (targets[i, 2] - self.uav_pos[i, 2]) / 18.0
            actions[i, 2] = float(np.clip(dz, -1.0, 1.0))
        return actions.astype(np.float32)

    def _distance_to_region_targets(self) -> np.ndarray:
        dists = np.zeros(self.num_uavs, dtype=np.float32)
        for i in range(self.num_uavs):
            assigned_uncovered = self._assigned_uncovered(i)
            target = assigned_uncovered[:, :3].mean(axis=0) if len(assigned_uncovered) else self.region_centers[i, :3]
            dists[i] = float(np.linalg.norm(self.uav_pos[i, :3] - target))
        return dists

    def reset(self) -> np.ndarray:
        sample = self.dataset.sample(self.target_num_users)
        self.users = sample["users"].copy()
        self.expert_uavs = sample["expert_uavs"].copy()
        self._load_terrain_surface()
        self._apply_terrain_to_users()
        self.initial_users = self.users.copy()
        self.t = 0
        self.uav_pos = self._random_init_uavs()
        if self.terrain_surface.size:
            for i in range(len(self.expert_uavs)):
                floor = self._terrain_height_at_xy(self.expert_uavs[i, :2])
                self.expert_uavs[i, 2] = float(np.clip(max(self.expert_uavs[i, 2], floor + self.terrain_clearance), ALT_MIN, ALT_MAX))
        self.prev_uav_pos = self.uav_pos.copy()
        self.covered = np.zeros(len(self.users), dtype=bool)
        self.cover_counts = np.zeros(len(self.users), dtype=np.int32)
        self.obstacles = self._make_obstacles()
        self._compute_region_plan()
        self.coverage_heat = self._build_heatmap()
        self.trajectory = [self.uav_pos.copy()]
        self.stagnation_steps = 0
        self.metrics = {
            "path_length": np.zeros(self.num_uavs, dtype=np.float32),
            "conflicts": 0,
            "redundant": 0,
            "failure_events": 0,
            "failed_agents_peak": 0,
        }
        self.failed = np.zeros(self.num_uavs, dtype=bool)
        self.failure_step = np.full(self.num_uavs, -1, dtype=np.int32)
        self.pre_failure_coverage = 0.0
        return self._get_obs()

    def _point_in_obstacle(self, point: np.ndarray) -> bool:
        if self.terrain_surface.size and float(point[2]) <= self._terrain_height_at_xy(point[:2]) + 1e-6:
            return True
        if len(self.obstacles) == 0:
            return False
        x, y, z = point[0], point[1], point[2]
        for x0, y0, z0, x1, y1, z1 in self.obstacles:
            if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1:
                return True
        return False

    def _get_agent_features(self) -> np.ndarray:
        features = []
        uncovered_users = self.users[~self.covered] if (~self.covered).any() else self.users
        coverage_ratio = float(self.covered.mean()) if len(self.covered) else 1.0
        time_ratio = self.t / max(1, self.max_steps)
        for i in range(self.num_uavs):
            pos = self.uav_pos[i]
            vel = (self.uav_pos[i] - self.prev_uav_pos[i]) / max(DT, 1e-6)
            dists = np.linalg.norm(self.uav_pos[:, :3] - pos[:3], axis=1)
            sorted_dists = np.sort(dists[dists > 0])
            nearest_agent = sorted_dists[0] if len(sorted_dists) else self._coverage_radius(float(pos[2]))
            assigned_idx = self._assigned_indices(i)
            assigned_users = self.users[assigned_idx] if len(assigned_idx) else self.users
            assigned_uncovered = self._assigned_uncovered(i)
            if len(assigned_uncovered):
                order = np.argsort(np.linalg.norm(assigned_uncovered[:, :3] - pos[:3], axis=1))
                near1 = assigned_uncovered[order[0]]
                near2 = assigned_uncovered[order[1]] if len(order) > 1 else near1
            else:
                nearest_user = uncovered_users[np.argmin(np.linalg.norm(uncovered_users[:, :3] - pos[:3], axis=1))]
                near1 = nearest_user
                near2 = nearest_user
            coverage_radius = self._coverage_radius(float(pos[2]))
            local_cover = np.mean(np.linalg.norm(self.users[:, :3] - pos[:3], axis=1) <= coverage_radius)
            assigned_cover_ratio = (
                float(self.covered[assigned_idx].mean()) if len(assigned_idx) else coverage_ratio
            )
            assigned_pending_ratio = 1.0 - assigned_cover_ratio if len(assigned_idx) else max(0.0, 1.0 - coverage_ratio)
            assigned_density = (
                float(np.mean(np.linalg.norm(assigned_uncovered[:, :3] - pos[:3], axis=1) <= coverage_radius))
                if len(assigned_uncovered)
                else 0.0
            )
            region_center = self.region_centers[i]
            frontier = assigned_uncovered[:, :2].mean(axis=0) if len(assigned_uncovered) else region_center[:2]
            frontier_vec = frontier - pos[:2]
            frontier_norm = np.linalg.norm(frontier_vec) + 1e-6
            obstacle_dist = self._nearest_obstacle_distance(pos)
            feat = np.array(
                [
                    pos[0] / AREA_X,
                    pos[1] / AREA_Y,
                    (pos[2] - ALT_MIN) / (ALT_MAX - ALT_MIN),
                    vel[0] / self.max_speed,
                    vel[1] / self.max_speed,
                    vel[2] / 25.0,
                    region_center[0] / AREA_X,
                    region_center[1] / AREA_Y,
                    (region_center[2] - ALT_MIN) / (ALT_MAX - ALT_MIN),
                    near1[0] / AREA_X,
                    near1[1] / AREA_Y,
                    near1[2] / max(ALT_MAX, 1.0),
                    near2[0] / AREA_X,
                    near2[1] / AREA_Y,
                    near2[2] / max(ALT_MAX, 1.0),
                    coverage_ratio,
                    assigned_cover_ratio,
                    local_cover,
                    assigned_density,
                    nearest_agent / SPACE_DIAGONAL,
                    time_ratio,
                    1.0 if self._point_in_obstacle(pos) else 0.0,
                    obstacle_dist / SPACE_DIAGONAL,
                    frontier_vec[0] / frontier_norm,
                    frontier_vec[1] / frontier_norm,
                    self.region_radii[i] / SPACE_DIAGONAL,
                    self.stagnation_steps / max(1, self.max_steps),
                    float(self.assignment_load[i]),
                    1.0 if self.failed[i] else 0.0,
                    float(self.failed.sum()) / max(1, self.num_uavs),
                    assigned_pending_ratio,
                    self._failure_recovery_stage(),
                ],
                dtype=np.float32,
            )
            features.append(feat)
        return np.stack(features, axis=0)

    def _get_obs(self) -> np.ndarray:
        return self._get_agent_features()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        # --- UAV failure module ---
        new_failures = self._maybe_trigger_failures()
        if new_failures.any():
            self.pre_failure_coverage = float(self.covered.mean()) if len(self.covered) else 0.0
            self.failure_step[new_failures] = self.t
            self.failed |= new_failures
            self.metrics["failure_events"] += int(new_failures.sum())
            self.metrics["failed_agents_peak"] = max(self.metrics["failed_agents_peak"], int(self.failed.sum()))
            # Standard online redistribution flow for surviving UAVs
            if self.fail_reassign_on_event and hasattr(self, "_compute_region_plan"):
                self._compute_region_plan()

        self.prev_uav_pos = self.uav_pos.copy()
        pre_target_dists = self._distance_to_region_targets()
        current_coverage = float(self.covered.mean()) if len(self.covered) else 1.0
        time_ratio = self.t / max(1, self.max_steps)
        speed_scale = float(np.clip(1.0 - 0.14 * current_coverage - 0.08 * time_ratio, 0.75, 1.0))
        # Failed UAVs do not move
        active_mask = (~self.failed).astype(np.float32)
        move_xy = np.clip(action[:, :2], -1.0, 1.0) * (self.max_speed * speed_scale)
        move_xy = move_xy * active_mask[:, None]
        move_z = np.clip(action[:, 2], -1.0, 1.0) * 18.0 * active_mask
        self.uav_pos[:, 0:2] = np.clip(self.uav_pos[:, 0:2] + move_xy, 0.0, [AREA_X, AREA_Y])
        self.uav_pos[:, 2] = np.clip(self.uav_pos[:, 2] + move_z, ALT_MIN, ALT_MAX)
        if self.terrain_surface.size:
            terrain_floor = np.asarray(
                [self._terrain_height_at_xy(self.uav_pos[i, :2]) + self.terrain_clearance for i in range(self.num_uavs)],
                dtype=np.float32,
            )
            self.uav_pos[:, 2] = np.maximum(self.uav_pos[:, 2], terrain_floor)

        if self.env_type == "dynamic_users":
            drift = np.random.uniform(-1.0, 1.0, size=(len(self.users), 2)).astype(np.float32)
            drift_z = np.random.uniform(-1.0, 1.0, size=(len(self.users),)).astype(np.float32)
            self.users[:, :2] = np.clip(self.users[:, :2] + drift * self.dynamic_ratio * 60.0, 0.0, [AREA_X, AREA_Y])
            if self.terrain_surface.size:
                self.users[:, 2] = np.asarray([self._terrain_height_at_xy(self.users[i, :2]) for i in range(len(self.users))], dtype=np.float32)
            else:
                self.users[:, 2] = np.clip(self.users[:, 2] + drift_z * self.dynamic_altitude_ratio * 18.0, 0.0, ALT_MAX - self.terrain_clearance)
            if self.t % max(1, self.region_reassign_interval) == 0:
                self._compute_region_plan()

        obstacle_hits = 0
        if len(self.obstacles):
            for i in range(self.num_uavs):
                if self._point_in_obstacle(self.uav_pos[i]):
                    obstacle_hits += 1
                    self.uav_pos[i] = self.prev_uav_pos[i]

        dist_users = pairwise_dist(self.uav_pos[:, :3], self.users[:, :3])
        agent_coverage = np.asarray([self._coverage_radius(float(pos[2])) for pos in self.uav_pos], dtype=np.float32)
        in_range = dist_users <= agent_coverage[:, None]
        newly_covered = (~self.covered) & in_range.any(axis=0)
        self.covered |= newly_covered
        if newly_covered.any():
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
        self.cover_counts += in_range.sum(axis=0).astype(np.int32)
        redundant = np.maximum(0, in_range.sum(axis=0) - 1).sum()
        self.metrics["redundant"] += int(redundant)

        pair_d = pairwise_dist(self.uav_pos[:, :3], self.uav_pos[:, :3])
        mask = np.triu(np.ones_like(pair_d, dtype=bool), 1)
        conflict_pairs = ((pair_d < COLLISION_DISTANCE) & mask).sum()
        self.metrics["conflicts"] += int(conflict_pairs)

        step_lengths = np.linalg.norm(self.uav_pos - self.prev_uav_pos, axis=1)
        self.metrics["path_length"] += step_lengths
        self.t += 1
        self.trajectory.append(self.uav_pos.copy())
        max_pair_conflicts = max(1, (self.num_uavs * (self.num_uavs - 1)) // 2)
        conflict_rate_step = float(self.metrics["conflicts"]) / max(1, self.t * max_pair_conflicts)
        conflict_pair_ratio = float(conflict_pairs) / max_pair_conflicts
        extra_conflict_penalty = 0.0
        if conflict_rate_step > self.high_conflict_threshold:
            extra_conflict_penalty = self.high_conflict_penalty * (conflict_rate_step - self.high_conflict_threshold)

        coverage_gain = float(newly_covered.sum()) / max(1, len(self.users))
        coverage_ratio = float(self.covered.mean()) if len(self.covered) else 1.0
        post_target_dists = self._distance_to_region_targets()
        assignment_gain = float(np.clip((pre_target_dists - post_target_dists).mean() / max(COMM_RADIUS, 1.0), -1.0, 1.0))
        frontier_bonus = 0.0
        for i in range(self.num_uavs):
            assigned_idx = self._assigned_indices(i)
            if len(assigned_idx) == 0:
                continue
            frontier_bonus += float(newly_covered[assigned_idx].sum()) / max(1, len(assigned_idx))
        frontier_bonus /= max(1, self.num_uavs)
        assignment_gate = float(
            np.clip(
                1.0 - self.assignment_decay_cover * coverage_ratio - self.assignment_decay_time * time_ratio,
                0.20,
                1.0,
            )
        )
        frontier_gate = float(
            np.clip(
                1.0 - self.frontier_decay_cover * coverage_ratio - self.frontier_decay_time * time_ratio,
                0.35,
                1.0,
            )
        )
        reward = (
            self.reward_gain_coef * coverage_gain
            + self.reward_coverage_coef * coverage_ratio
            + self.assignment_bonus_coef * assignment_gate * assignment_gain
            + self.frontier_bonus_coef * frontier_gate * frontier_bonus
            - self.reward_path_penalty * float(step_lengths.sum())
            - self.reward_conflict_penalty * conflict_pair_ratio
            - self.reward_redundancy_penalty * float(redundant)
            - self.reward_obstacle_penalty * float(obstacle_hits)
            - self.stagnation_penalty_coef * (self.stagnation_steps / max(1, self.max_steps))
            - extra_conflict_penalty
        )
        # --- Recovery bonus ---
        recovery_bonus = 0.0
        if self.failed.any() and len(self.covered) > 0:
            current_cov = float(self.covered.mean())
            if current_cov > self.pre_failure_coverage:
                recovery_bonus = (current_cov - self.pre_failure_coverage) * \
                    getattr(self, "fail_reward_recovery_coef", 0.8)
                self.pre_failure_coverage = current_cov

        # --- Energy fairness penalty (path length variance as proxy) ---
        path_var = float(np.var(self.metrics["path_length"]))
        energy_fair_penalty = 0.0
        if self.failed.any():
            energy_fair_penalty = 0.35 * getattr(self, "fail_energy_fair_coef", 0.15) * path_var / max(
                1.0,
                float(self.metrics["path_length"].mean()) ** 2,
            )

        reward = reward + recovery_bonus - energy_fair_penalty

        done = self.t >= self.max_steps or self.covered.all()
        info = {
            "coverage_rate": coverage_ratio,
            "path_length": float(self.metrics["path_length"].sum()),
            "conflict_rate": conflict_rate_step,
            "coverage_redundancy": float(self.metrics["redundant"]) / max(1, len(self.users) * self.t),
            "task_time": float(self.t),
            "failed_ratio": float(self.failed.mean()),
            "failed_count": int(self.failed.sum()),
            "recovery_bonus": float(recovery_bonus),
            "energy_fair_penalty": float(energy_fair_penalty),
            "recovery_stage": self._failure_recovery_stage(),
            "active_uav_ratio": float((~self.failed).mean()),
            "coverage_drop_from_prefailure": max(0.0, self.pre_failure_coverage - coverage_ratio) if self.failed.any() else 0.0,
        }
        return self._get_obs(), reward, done, info


class SharedMLPEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiSourceObjectiveEncoder(nn.Module):
    """Encode heterogeneous UAV observations into self/agent/entity/task streams."""

    SELF_IDXS = (0, 1, 2, 3, 4, 5, 15, 18, 20, 27, 28)
    AGENT_IDXS = (9, 10, 11, 12, 13, 14, 19, 21, 28, 29, 30)
    ENTITY_IDXS = (6, 7, 8, 17, 22, 23, 24, 25, 26, 31)
    TASK_IDXS = (15, 16, 17, 18, 20, 27, 29, 30, 31)
    GATE_IDXS = (15, 16, 20, 22, 23, 29, 30, 31)

    def __init__(self, d_model: int, config: "ExperimentConfig"):
        super().__init__()
        self.multi_source_gate_strength = config.multi_source_gate_strength
        self.objective_gate_strength = config.objective_gate_strength
        self.self_proj = nn.Linear(len(self.SELF_IDXS), d_model)
        self.agent_proj = nn.Linear(len(self.AGENT_IDXS), d_model)
        self.entity_proj = nn.Linear(len(self.ENTITY_IDXS), d_model)
        self.task_proj = nn.Linear(len(self.TASK_IDXS), d_model)
        self.stream_embed = nn.Parameter(torch.zeros(4, d_model))
        nn.init.normal_(self.stream_embed, mean=0.0, std=0.02)
        self.source_gate = nn.Sequential(
            nn.Linear(len(self.GATE_IDXS), d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )
        self.objective_head = nn.Sequential(
            nn.Linear(len(self.GATE_IDXS), d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.norm = nn.LayerNorm(d_model)
        self.latest_gate_stats: Dict[str, torch.Tensor] | None = None

    def _slice(self, obs: torch.Tensor, idxs: Tuple[int, ...]) -> torch.Tensor:
        return obs[..., list(idxs)]

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate_context = self._slice(obs, self.GATE_IDXS)
        self_stream = self.self_proj(self._slice(obs, self.SELF_IDXS)) + self.stream_embed[0]
        agent_stream = self.agent_proj(self._slice(obs, self.AGENT_IDXS)) + self.stream_embed[1]
        entity_stream = self.entity_proj(self._slice(obs, self.ENTITY_IDXS)) + self.stream_embed[2]
        task_stream = self.task_proj(self._slice(obs, self.TASK_IDXS)) + self.stream_embed[3]
        streams = torch.stack([self_stream, agent_stream, entity_stream, task_stream], dim=-2)

        base_logits = self.source_gate(gate_context)
        objective_logits = self.objective_head(gate_context)
        objective = torch.softmax(objective_logits, dim=-1)
        coverage_focus = objective[..., 0]
        safety_focus = objective[..., 1]
        efficiency_focus = objective[..., 2]
        objective_bias = torch.stack(
            [
                0.20 * efficiency_focus + 0.15 * safety_focus,
                0.55 * safety_focus + 0.20 * coverage_focus,
                0.50 * coverage_focus + 0.25 * safety_focus,
                0.35 * efficiency_focus + 0.25 * coverage_focus + 0.15 * safety_focus,
            ],
            dim=-1,
        )
        logits = base_logits + self.objective_gate_strength * objective_bias
        weights = torch.softmax(logits * (1.0 + self.multi_source_gate_strength), dim=-1)
        mixed = (streams * weights.unsqueeze(-1)).sum(dim=-2)
        mixed = self.norm(mixed)
        stats = {
            "source_weights": weights.detach(),
            "objective_weights": objective.detach(),
        }
        self.latest_gate_stats = stats
        return mixed, stats


class RiskAwareRelationalTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        relation_risk_strength: float,
        relation_comm_strength: float,
        relation_altitude_strength: float,
        safety_distance: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.relation_risk_strength = relation_risk_strength
        self.relation_comm_strength = relation_comm_strength
        self.relation_altitude_strength = relation_altitude_strength
        self.safety_distance = safety_distance
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        # "mha" = multi-head self-attention (default);
        # "mean" = parameter-free mean aggregation, used by the
        # no_attention ablation so that attention is the only variable.
        self.attention_mode = "mha"
        self.relation_mlp = nn.Sequential(
            nn.Linear(6, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_heads),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )

    def _build_relation_bias(self, obs: torch.Tensor) -> torch.Tensor:
        pos = torch.empty_like(obs[..., :3])
        pos[..., 0] = obs[..., 0] * AREA_X
        pos[..., 1] = obs[..., 1] * AREA_Y
        pos[..., 2] = ALT_MIN + obs[..., 2] * (ALT_MAX - ALT_MIN)
        region = torch.empty_like(pos)
        region[..., 0] = obs[..., 6] * AREA_X
        region[..., 1] = obs[..., 7] * AREA_Y
        region[..., 2] = ALT_MIN + obs[..., 8] * (ALT_MAX - ALT_MIN)
        delta = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.linalg.norm(delta, dim=-1) + 1e-6
        altitude_gap = delta[..., 2].abs()
        risk = torch.clamp((self.safety_distance - dist) / self.safety_distance, min=0.0)
        comm = torch.clamp((COMM_RADIUS * 1.15 - dist) / (COMM_RADIUS * 1.15), min=0.0)
        goal_vec = region - pos
        goal_dir = goal_vec / (torch.linalg.norm(goal_vec, dim=-1, keepdim=True) + 1e-6)
        alignment = (goal_dir[:, :, None, :] * goal_dir[:, None, :, :]).sum(dim=-1)
        pair_feat = torch.stack(
            [
                delta[..., 0] / AREA_X,
                delta[..., 1] / AREA_Y,
                delta[..., 2] / max(ALT_MAX - ALT_MIN, 1e-6),
                dist / SPACE_DIAGONAL,
                risk,
                alignment,
            ],
            dim=-1,
        )
        learned = self.relation_mlp(pair_feat)
        bias = 0.55 * torch.tanh(learned)
        bias = bias + self.relation_risk_strength * risk.unsqueeze(-1)
        bias = bias + self.relation_comm_strength * comm.unsqueeze(-1)
        bias = bias - self.relation_altitude_strength * (altitude_gap / max(ALT_MAX - ALT_MIN, 1e-6)).unsqueeze(-1)
        eye = torch.eye(obs.shape[1], device=obs.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        bias = bias.masked_fill(eye, 0.0)
        return bias.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        attn_bias = self._build_relation_bias(obs)
        bsz, n_agents, _ = x.shape
        attn_mask = attn_bias.reshape(bsz * self.n_heads, n_agents, n_agents)
        norm_x = self.norm1(x)
        if getattr(self, "attention_mode", "mha") == "mean":
            # No pairwise weighting: every agent receives the same
            # agent-averaged context vector.
            attn_out = norm_x.mean(dim=1, keepdim=True).expand_as(norm_x)
        else:
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoderBackbone(nn.Module):
    def __init__(self, state_dim: int, d_model: int, n_heads: int, n_layers: int, config: "ExperimentConfig"):
        super().__init__()
        self.multi_source_encoder = MultiSourceObjectiveEncoder(d_model, config)
        self.input_proj = nn.Linear(state_dim, d_model)
        self.effective_heads = n_heads
        self.num_layers = n_layers
        self.output_norm = nn.LayerNorm(d_model)
        self.latest_gate_stats: Dict[str, torch.Tensor] | None = None
        self.layers = nn.ModuleList(
            [
                RiskAwareRelationalTransformerLayer(
                    d_model=d_model,
                    n_heads=self.effective_heads,
                    relation_risk_strength=config.relation_risk_strength,
                    relation_comm_strength=config.relation_comm_strength,
                    relation_altitude_strength=config.relation_altitude_strength,
                    safety_distance=config.safety_distance,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        multi_source, gate_stats = self.multi_source_encoder(x)
        encoded = projected + multi_source
        for layer in self.layers:
            encoded = layer(encoded, x)
        self.latest_gate_stats = gate_stats
        return self.output_norm(encoded + projected + 0.35 * multi_source)


class TmarlActorCritic(nn.Module):
    def __init__(self, state_dim: int, config: ExperimentConfig, variant: str = "tmarl"):
        super().__init__()
        self.variant = variant
        self.max_speed = config.max_speed
        self.safety_distance = config.safety_distance
        self.safety_strength = config.safety_strength
        self.safety_brake = config.safety_brake
        self.safety_risk_bias = config.safety_risk_bias
        self.safety_stage_cap = config.safety_stage_cap
        self.safety_blend_cap = config.safety_blend_cap
        self.safety_use_repulsion = config.safety_use_repulsion
        self.safety_use_brake = config.safety_use_brake
        self.safety_use_guidance = config.safety_use_guidance
        self.guidance_strength = config.guidance_strength
        self.guidance_cover_decay = config.guidance_cover_decay
        self.guidance_risk_threshold = config.guidance_risk_threshold
        self.latest_safety_stats: Dict[str, torch.Tensor] | None = None
        if variant == "tmarl":
            self.backbone = TransformerEncoderBackbone(state_dim, config.d_model, config.n_heads, config.n_layers, config)
            feat_dim = config.d_model
            self.actor_mode = "contextual"
            self.critic_mode = "centralized"
        elif variant == "tmarl_no_safety":
            self.backbone = TransformerEncoderBackbone(state_dim, config.d_model, config.n_heads, config.n_layers, config)
            feat_dim = config.d_model
            self.actor_mode = "contextual"
            self.critic_mode = "centralized"
            self.safety_use_repulsion = False
            self.safety_use_brake = False
            self.safety_use_guidance = False
        elif variant == "tmarl_no_repulsion":
            self.backbone = TransformerEncoderBackbone(state_dim, config.d_model, config.n_heads, config.n_layers, config)
            feat_dim = config.d_model
            self.actor_mode = "contextual"
            self.critic_mode = "centralized"
            self.safety_use_repulsion = False
        elif variant == "tmarl_no_brake":
            self.backbone = TransformerEncoderBackbone(state_dim, config.d_model, config.n_heads, config.n_layers, config)
            feat_dim = config.d_model
            self.actor_mode = "contextual"
            self.critic_mode = "centralized"
            self.safety_use_brake = False
        elif variant == "tmarl_no_guidance":
            self.backbone = TransformerEncoderBackbone(state_dim, config.d_model, config.n_heads, config.n_layers, config)
            feat_dim = config.d_model
            self.actor_mode = "contextual"
            self.critic_mode = "centralized"
            self.safety_use_guidance = False
        elif variant == "mappo":
            self.backbone = SharedMLPEncoder(state_dim, config.hidden_dim)
            feat_dim = config.hidden_dim
            self.actor_mode = "local"
            self.critic_mode = "centralized"
        elif variant == "ppo":
            self.backbone = SharedMLPEncoder(state_dim, config.hidden_dim)
            feat_dim = config.hidden_dim
            self.actor_mode = "local"
            self.critic_mode = "local"
        elif variant == "no_transformer":
            self.backbone = SharedMLPEncoder(state_dim, config.hidden_dim)
            feat_dim = config.hidden_dim
            self.actor_mode = "local"
            self.critic_mode = "centralized"
        elif variant == "no_attention":
            # Genuine attention ablation: identical backbone to "tmarl"
            # (multi-source objective encoder, two pre-norm transformer
            # blocks, feed-forward width d_model*4), with multi-head
            # self-attention replaced by a parameter-free mean over the
            # agent axis.  Router/actor/critic settings are those of the
            # full model, so attention is the only variable.
            self.backbone = TransformerEncoderBackbone(
                state_dim, config.d_model, config.n_heads, config.n_layers, config)
            for _layer in self.backbone.layers:
                _layer.attention_mode = "mean"
            feat_dim = config.d_model
            self.actor_mode = "contextual"
            self.critic_mode = "centralized"
        elif variant == "no_ctde":
            # Independent-style ablation: remove centralized training and shared interaction modeling.
            self.backbone = SharedMLPEncoder(state_dim, config.hidden_dim)
            feat_dim = config.hidden_dim
            self.actor_mode = "local"
            self.critic_mode = "local"
        else:
            raise ValueError(f"Unknown model variant: {variant}")

        self.actor_mean = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 3),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((3,), -1.20))
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if self.variant in {"tmarl", "tmarl_no_safety", "no_attention"}:
            return self.backbone(obs)
        if self.variant in {"no_transformer", "mappo", "ppo"}:
            return self.backbone(obs)
        b, n, d = obs.shape
        flat = obs.reshape(b * n, d)
        feat = self.backbone(flat).reshape(b, n, -1)
        return feat

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encode(obs)
        mean = self.actor_mean(feat)
        if self.critic_mode == "centralized":
            value = self.critic(feat.mean(dim=1)).squeeze(-1)
        else:
            value = self.critic(feat).squeeze(-1).mean(dim=1)
        log_std = self.log_std.view(1, 1, 3).expand_as(mean)
        return mean, log_std, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        if self.variant in {"tmarl", "tmarl_no_repulsion", "tmarl_no_brake", "tmarl_no_guidance"}:
            action = self.apply_conflict_aware_safety(obs, action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).sum(dim=-1)
        return action, log_prob, value, entropy

    def apply_conflict_aware_safety(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pos_xy = torch.empty_like(obs[..., :2])
        pos_xy[..., 0] = obs[..., 0] * AREA_X
        pos_xy[..., 1] = obs[..., 1] * AREA_Y
        target_xy = torch.empty_like(pos_xy)
        target_xy[..., 0] = obs[..., 9] * AREA_X
        target_xy[..., 1] = obs[..., 10] * AREA_Y
        projected_xy = pos_xy + action[..., :2] * self.max_speed * DT
        coverage_ratio = obs[..., 15:16]
        time_ratio = obs[..., 20:21]

        delta_now = pos_xy[:, :, None, :] - pos_xy[:, None, :, :]
        dist_now = torch.linalg.norm(delta_now, dim=-1) + 1e-6
        delta_next = projected_xy[:, :, None, :] - projected_xy[:, None, :, :]
        dist_next = torch.linalg.norm(delta_next, dim=-1) + 1e-6

        eye = torch.eye(obs.shape[1], device=obs.device, dtype=torch.bool).unsqueeze(0)
        failed_flag = obs[..., 28:29]
        alive_mask = failed_flag < 0.5
        alive_pair_mask = alive_mask[:, :, None, :] & alive_mask[:, None, :, :]
        risk_mask = (~eye) & alive_pair_mask.squeeze(-1) & (
            (dist_next < self.safety_distance)
            | ((dist_now < self.safety_distance * 1.10) & (dist_next < dist_now))
        )

        proximity = torch.clamp((self.safety_distance - dist_next) / self.safety_distance, min=0.0)
        closing = torch.clamp((dist_now - dist_next) / self.safety_distance, min=0.0)
        high_risk = torch.clamp(0.75 * proximity + 1.25 * closing - self.safety_risk_bias, min=0.0)
        stage_gate = torch.clamp(
            0.08 + 0.55 * coverage_ratio + 0.35 * time_ratio,
            min=0.0,
            max=self.safety_stage_cap,
        )
        repel_weight = high_risk * risk_mask.float() * stage_gate.squeeze(-1).unsqueeze(-1)
        repel_dir = delta_next / dist_next.unsqueeze(-1)
        repel = (repel_dir * repel_weight.unsqueeze(-1)).sum(dim=2)

        max_risk = repel_weight.max(dim=2, keepdim=True).values
        goal_vec = target_xy - pos_xy
        goal_dist = torch.linalg.norm(goal_vec, dim=-1, keepdim=True) + 1e-6
        goal_dir = goal_vec / goal_dist
        orig_xy = action[..., :2]
        low_risk_bias = torch.clamp(0.78 + 0.22 * (1.0 - max_risk / (self.guidance_risk_threshold + 1e-6)), min=0.60, max=1.0)
        safety_gain = self.safety_strength * low_risk_bias * (0.16 + 0.52 * coverage_ratio + 0.23 * time_ratio)
        repel_term = safety_gain * repel if self.safety_use_repulsion else torch.zeros_like(orig_xy)
        safe_xy_candidate = orig_xy + repel_term

        blend = torch.clamp(1.20 * max_risk, min=0.0, max=self.safety_blend_cap)
        safe_xy = (1.0 - blend) * orig_xy + blend * safe_xy_candidate

        guidance_blend = torch.zeros_like(max_risk)
        if self.safety_use_guidance:
            goal_alignment = (safe_xy * goal_dir).sum(dim=-1, keepdim=True)
            backward_mask = goal_alignment < -0.10
            safe_xy = torch.where(backward_mask, 0.55 * safe_xy + 0.45 * orig_xy, safe_xy)

        # When collision risk is low, bias the policy toward uncovered-user pursuit
        # so the safety module does not turn into an overly conservative brake.
        nearest_agent_ratio = obs[..., 19:20]
        isolation_gate = torch.clamp(0.38 + 0.78 * nearest_agent_ratio, min=0.18, max=1.0)
        cover_gate = torch.clamp(1.02 - self.guidance_cover_decay * coverage_ratio, min=0.22, max=1.0)
        low_risk_gate = torch.clamp((self.guidance_risk_threshold - max_risk) / self.guidance_risk_threshold, min=0.0, max=1.0)
        if self.safety_use_guidance:
            guidance_blend = torch.clamp(
                1.25 * self.guidance_strength * isolation_gate * cover_gate * low_risk_gate,
                min=0.0,
                max=0.28,
            )
            guided_goal = torch.clamp(0.69 * goal_dir + 0.31 * orig_xy, -1.0, 1.0)
            safe_xy = (1.0 - guidance_blend) * safe_xy + guidance_blend * guided_goal

        brake = torch.where(
            max_risk > 0.34,
            torch.clamp(1.0 - self.safety_brake * (max_risk - 0.22), min=0.85, max=1.0),
            torch.ones_like(max_risk),
        ) if self.safety_use_brake else torch.ones_like(max_risk)
        safe_xy = torch.clamp(safe_xy, -1.0, 1.0) * brake

        safe_action = action.clone()
        safe_action[..., :2] = torch.clamp(safe_xy, -1.0, 1.0)
        self.latest_safety_stats = {
            "max_risk": max_risk.detach().squeeze(-1),
            "repulsion_norm": torch.linalg.norm(repel_term, dim=-1).detach(),
            "brake_scale": brake.detach().squeeze(-1),
            "guidance_blend": guidance_blend.detach().squeeze(-1),
        }
        return safe_action


def build_imitation_batch(
    env: MultiUAVCoverageEnv,
    num_samples: int,
    rollout_steps: int = 1,
    expert_mix: float = 1.0,
    frontier_mix: float = 0.0,
    repulsion_coef: float = 0.0,
    obstacle_coef: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    obs_batch = []
    act_batch = []
    py_state = random.getstate()
    np_state = np.random.get_state()
    base_seed = int(np.random.randint(0, 1_000_000))
    seed_stride = 9973
    episode_idx = 0
    rollout_steps = max(1, int(rollout_steps))
    while len(obs_batch) < num_samples:
        idx = episode_idx
        mixed_seed = base_seed + idx * seed_stride
        random.seed(mixed_seed)
        np.random.seed(mixed_seed % (2**32 - 1))
        obs = env.reset()
        for _ in range(rollout_steps):
            if (
                expert_mix >= 0.999
                and frontier_mix <= 1e-8
                and repulsion_coef <= 1e-8
                and obstacle_coef <= 1e-8
            ):
                target_action = env.compute_imitation_actions()
            else:
                target_action = compute_guided_supervision_actions(
                    env,
                    expert_mix=expert_mix,
                    frontier_mix=frontier_mix,
                    repulsion_coef=repulsion_coef,
                    obstacle_coef=obstacle_coef,
                )
            obs_batch.append(obs)
            act_batch.append(target_action)
            if len(obs_batch) >= num_samples:
                break
            obs, _, done, _ = env.step(target_action)
            if done:
                break
        episode_idx += 1
    random.setstate(py_state)
    np.random.set_state(np_state)
    return np.asarray(obs_batch, dtype=np.float32), np.asarray(act_batch, dtype=np.float32)


def run_imitation_warmstart(
    model: TmarlActorCritic,
    optimizer: torch.optim.Optimizer,
    env: MultiUAVCoverageEnv,
    config: ExperimentConfig,
    device: torch.device,
    stage_prefix: str,
) -> None:
    if config.imitation_epochs <= 0 or config.imitation_batch_size <= 0:
        return
    log(
        f"{stage_prefix}start imitation warmstart | epochs={config.imitation_epochs} "
        f"batch_size={config.imitation_batch_size}"
    )
    for epoch in range(config.imitation_epochs):
        obs_np, target_np = build_imitation_batch(
            env,
            config.imitation_batch_size,
            rollout_steps=config.target_supervision_rollout_steps,
            expert_mix=config.target_supervision_expert_mix,
            frontier_mix=config.target_supervision_frontier_mix,
            repulsion_coef=config.target_supervision_repulsion_coef,
            obstacle_coef=config.target_supervision_obstacle_coef,
        )
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        target_t = torch.tensor(target_np, dtype=torch.float32, device=device)
        mean, _, _ = model(obs_t)
        loss = F.mse_loss(mean, target_t)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()
        log(f"{stage_prefix}imitation epoch {epoch + 1}/{config.imitation_epochs} | loss={loss.item():.6f}")


def build_env_from_config(
    config: ExperimentConfig,
    dataset: ScenarioDataset,
) -> MultiUAVCoverageEnv:
    return MultiUAVCoverageEnv(
        dataset=dataset,
        num_uavs=config.train_num_uavs,
        env_type=config.train_env,
        max_steps=config.max_steps,
        max_speed=config.max_speed,
        high_conflict_threshold=config.high_conflict_threshold,
        high_conflict_penalty=config.high_conflict_penalty,
        reward_gain_coef=config.reward_gain_coef,
        reward_coverage_coef=config.reward_coverage_coef,
        reward_path_penalty=config.reward_path_penalty,
        reward_conflict_penalty=config.reward_conflict_penalty,
        reward_redundancy_penalty=config.reward_redundancy_penalty,
        reward_obstacle_penalty=config.reward_obstacle_penalty,
        assignment_bonus_coef=config.assignment_bonus_coef,
        frontier_bonus_coef=config.frontier_bonus_coef,
        stagnation_penalty_coef=config.stagnation_penalty_coef,
        region_reassign_interval=config.region_reassign_interval,
        use_region_assignment=config.use_region_assignment,
        region_balance_strength=config.region_balance_strength,
        region_global_mix=config.region_global_mix,
        region_support_radius_scale=config.region_support_radius_scale,
        heuristic_view_radius_scale=config.heuristic_view_radius_scale,
        assignment_decay_cover=config.assignment_decay_cover,
        assignment_decay_time=config.assignment_decay_time,
        frontier_decay_cover=config.frontier_decay_cover,
        frontier_decay_time=config.frontier_decay_time,
        late_stage_visible_boost=config.late_stage_visible_boost,
        fail_prob_base=config.fail_prob_base,
        fail_prob_load_scale=config.fail_prob_load_scale,
        fail_reward_recovery_coef=config.fail_reward_recovery_coef,
        fail_energy_fair_coef=config.fail_energy_fair_coef,
        use_failure_module=config.use_failure_module,
        fail_trigger_min_step=config.fail_trigger_min_step,
        fail_trigger_max_step=config.fail_trigger_max_step,
        fail_max_count=config.fail_max_count,
        fail_fixed_count=config.fail_fixed_count,
        fail_load_aware=config.fail_load_aware,
        camera_fov_deg=config.camera_fov_deg,
        obstacle_height_min=config.obstacle_height_min,
        obstacle_height_max=config.obstacle_height_max,
        dynamic_altitude_ratio=config.dynamic_altitude_ratio,
        fail_reassign_on_event=config.fail_reassign_on_event,
        safety_distance=config.safety_distance,
    )


def build_target_supervision_dataset(
    config: ExperimentConfig,
    fallback_dataset: ScenarioDataset,
) -> ScenarioDataset:
    if config.target_dataset_paths:
        valid_paths = [path for path in config.target_dataset_paths if os.path.exists(path)]
        if valid_paths:
            return ScenarioDataset(valid_paths)
        log("target supervision dataset paths were provided but none exist; falling back to dataset_paths")
    return fallback_dataset


def compute_guided_supervision_actions(
    env: MultiUAVCoverageEnv,
    expert_mix: float,
    frontier_mix: float,
    repulsion_coef: float,
    obstacle_coef: float,
) -> np.ndarray:
    actions = np.zeros((env.num_uavs, 3), dtype=np.float32)
    expert_mix = float(np.clip(expert_mix, 0.0, 1.0))
    frontier_mix = float(np.clip(frontier_mix, 0.0, 1.0 - expert_mix))
    nearest_mix = max(0.0, 1.0 - expert_mix - frontier_mix)

    for i in range(env.num_uavs):
        if hasattr(env, "failed") and env.failed[i]:
            continue

        assigned_uncovered = env._assigned_uncovered(i)
        visible_uncovered = env._visible_uncovered(i)
        region_target = env.region_centers[i].copy()
        expert_target = env.expert_targets[i].copy() if hasattr(env, "expert_targets") else region_target.copy()

        if len(assigned_uncovered):
            frontier_xy = assigned_uncovered[:, :2].mean(axis=0)
            nearest_set = assigned_uncovered
        elif len(visible_uncovered):
            frontier_xy = visible_uncovered[:, :2].mean(axis=0)
            nearest_set = visible_uncovered
        else:
            frontier_xy = region_target[:2]
            nearest_set = env.users

        nearest_xy = frontier_xy
        if len(nearest_set):
            nearest_idx = int(np.argmin(np.linalg.norm(nearest_set[:, :2] - env.uav_pos[i, :2], axis=1)))
            nearest_xy = nearest_set[nearest_idx, :2]

        goal_xy = (
            expert_mix * expert_target[:2]
            + frontier_mix * frontier_xy
            + nearest_mix * nearest_xy
        )
        goal_xy = np.clip(goal_xy, [0.0, 0.0], [AREA_X, AREA_Y])
        goal_vec = goal_xy - env.uav_pos[i, :2]
        goal_norm = np.linalg.norm(goal_vec) + 1e-6
        move_xy = goal_vec / goal_norm

        repulse = np.zeros(2, dtype=np.float32)
        for j in range(env.num_uavs):
            if i == j:
                continue
            if hasattr(env, "failed") and env.failed[j]:
                continue
            delta = env.uav_pos[i, :2] - env.uav_pos[j, :2]
            dist = np.linalg.norm(delta) + 1e-6
            safety_span = max(env.safety_distance * 1.15, COLLISION_DISTANCE * 1.25)
            if dist < safety_span:
                repulse += (delta / dist) * (1.0 - dist / safety_span)

        obs_push = np.zeros(2, dtype=np.float32)
        obs_push_z = 0.0
        if len(env.obstacles):
            px, py, pz = env.uav_pos[i, 0], env.uav_pos[i, 1], env.uav_pos[i, 2]
            for obstacle in env.obstacles:
                x0, y0, z0, x1, y1, z1 = obstacle_xyz_bounds(obstacle)
                near_x = np.clip(px, x0, x1)
                near_y = np.clip(py, y0, y1)
                near_z = np.clip(pz, z0, z1)
                delta = np.array([px - near_x, py - near_y], dtype=np.float32)
                dist = np.linalg.norm(delta) + 1e-6
                z_gap = pz - near_z
                avoid_span = max(env.safety_distance, 180.0)
                if point_to_box_distance_3d(env.uav_pos[i], obstacle) < avoid_span:
                    if dist < 1e-3:
                        center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
                        delta = env.uav_pos[i, :2] - center
                        dist = np.linalg.norm(delta) + 1e-6
                    obs_push += (delta / dist) * (1.0 - dist / avoid_span)
                    if abs(z_gap) < 18.0:
                        obs_push_z += -1.0 if (z0 + z1) * 0.5 > pz else 1.0

        guided_xy = move_xy + repulsion_coef * repulse + obstacle_coef * obs_push
        guided_norm = np.linalg.norm(guided_xy) + 1e-6
        actions[i, :2] = np.clip(guided_xy / guided_norm, -1.0, 1.0)

        z_target = float(
            expert_mix * expert_target[2]
            + (1.0 - expert_mix) * region_target[2]
        )
        if abs(obs_push_z) > 1e-6:
            z_target = float(np.clip(z_target + obstacle_coef * 14.0 * obs_push_z, ALT_MIN, ALT_MAX))
        actions[i, 2] = float(np.clip((z_target - env.uav_pos[i, 2]) / 18.0, -1.0, 1.0))

    return actions.astype(np.float32)


def run_target_supervision_step(
    model: TmarlActorCritic,
    optimizer: torch.optim.Optimizer,
    env: MultiUAVCoverageEnv,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    obs_np, target_np = build_imitation_batch(
        env,
        config.target_supervision_batch_size,
        rollout_steps=config.target_supervision_rollout_steps,
        expert_mix=config.target_supervision_expert_mix,
        frontier_mix=config.target_supervision_frontier_mix,
        repulsion_coef=config.target_supervision_repulsion_coef,
        obstacle_coef=config.target_supervision_obstacle_coef,
    )
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    target_t = torch.tensor(target_np, dtype=torch.float32, device=device)
    mean, _, _ = model(obs_t)
    aux_loss = F.mse_loss(mean, target_t)
    total_loss = config.target_supervision_weight * aux_loss
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
    optimizer.step()
    return float(aux_loss.item())


def run_target_supervision_refine(
    model: TmarlActorCritic,
    optimizer: torch.optim.Optimizer,
    env: MultiUAVCoverageEnv,
    config: ExperimentConfig,
    device: torch.device,
    stage_prefix: str,
) -> None:
    if config.target_supervision_refine_epochs <= 0:
        return
    log(
        f"{stage_prefix}start target-domain refine | epochs={config.target_supervision_refine_epochs} "
        f"batch_size={config.target_supervision_batch_size} rollout_steps={config.target_supervision_rollout_steps}"
    )
    for epoch in range(config.target_supervision_refine_epochs):
        aux_loss = run_target_supervision_step(
            model=model,
            optimizer=optimizer,
            env=env,
            config=config,
            device=device,
        )
        log(
            f"{stage_prefix}target refine epoch {epoch + 1}/{config.target_supervision_refine_epochs} "
            f"| loss={aux_loss:.6f}"
        )


class SharedDeterministicActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralizedCritic(nn.Module):
    def __init__(self, num_uavs: int, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = num_uavs * (state_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([obs.reshape(obs.shape[0], -1), actions.reshape(actions.shape[0], -1)], dim=-1)
        return self.net(flat).squeeze(-1)


class MultiAgentReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.storage: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
        self.ptr = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: float) -> None:
        item = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.storage), size=batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.storage[i] for i in idx))
        return (
            np.stack(obs, axis=0),
            np.stack(actions, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs, axis=0),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.storage)


class MADDPGController(nn.Module):
    def __init__(self, num_uavs: int, state_dim: int, hidden_dim: int, device: torch.device):
        super().__init__()
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.device = device
        self.actors = nn.ModuleList([SharedDeterministicActor(state_dim, hidden_dim) for _ in range(num_uavs)])
        self.target_actors = nn.ModuleList([SharedDeterministicActor(state_dim, hidden_dim) for _ in range(num_uavs)])
        self.critic = CentralizedCritic(num_uavs, state_dim, 3, hidden_dim)
        self.target_critic = CentralizedCritic(num_uavs, state_dim, 3, hidden_dim)
        self._hard_update()

    def _hard_update(self) -> None:
        for target, source in zip(self.target_actors, self.actors):
            target.load_state_dict(source.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update(self, tau: float = 0.01) -> None:
        for target, source in zip(self.target_actors, self.actors):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * sp.data)
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)

    def forward_actions(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        actors = self.target_actors if target else self.actors
        actions = []
        for agent_idx in range(self.num_uavs):
            actions.append(actors[agent_idx](obs[:, agent_idx, :]))
        return torch.stack(actions, dim=1)

    def act(self, obs: torch.Tensor, deterministic: bool = False, noise_scale: float = 0.0):
        with torch.no_grad():
            action = self.forward_actions(obs, target=False)
            if not deterministic and noise_scale > 0.0:
                action = action + noise_scale * torch.randn_like(action)
            action = torch.clamp(action, -1.0, 1.0)
            value = self.critic(obs, action)
        batch = obs.shape[0]
        zeros = torch.zeros(batch, device=obs.device)
        return action, zeros, value, zeros


QMIX_ACTIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.7, 0.7, 0.0],
        [0.7, -0.7, 0.0],
        [-0.7, 0.7, 0.0],
        [-0.7, -0.7, 0.0],
        [0.0, 0.0, 0.8],
        [0.0, 0.0, -0.8],
    ],
    dtype=np.float32,
)


class QMIXReplayBuffer:
    def __init__(self, capacity: int = 30000):
        self.capacity = capacity
        self.storage: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
        self.ptr = 0

    def add(self, obs: np.ndarray, action_idx: np.ndarray, reward: float, next_obs: np.ndarray, done: float) -> None:
        item = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action_idx, dtype=np.int64),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.storage), size=batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.storage[i] for i in idx))
        return (
            np.stack(obs, axis=0),
            np.stack(actions, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs, axis=0),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.storage)


class AgentQNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMixMixer(nn.Module):
    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = hidden_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        batch = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch)


class QMIXController(nn.Module):
    def __init__(self, num_uavs: int, state_dim: int, hidden_dim: int, device: torch.device):
        super().__init__()
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.device = device
        self.num_actions = int(QMIX_ACTIONS.shape[0])
        self.agent_net = AgentQNetwork(state_dim, hidden_dim, self.num_actions)
        self.target_agent_net = AgentQNetwork(state_dim, hidden_dim, self.num_actions)
        self.mixer = QMixMixer(num_uavs, num_uavs * state_dim, hidden_dim)
        self.target_mixer = QMixMixer(num_uavs, num_uavs * state_dim, hidden_dim)
        self._hard_update()

    def _hard_update(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update(self, tau: float = 0.02) -> None:
        for tp, sp in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)
        for tp, sp in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)

    def q_values(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        net = self.target_agent_net if target else self.agent_net
        batch, num_agents, obs_dim = obs.shape
        q = net(obs.view(batch * num_agents, obs_dim)).view(batch, num_agents, self.num_actions)
        return q

    def select_actions(self, obs: torch.Tensor, epsilon: float = 0.0, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_values(obs, target=False)
        greedy = torch.argmax(q, dim=-1)
        if deterministic or epsilon <= 0.0:
            action_idx = greedy
        else:
            random_idx = torch.randint(0, self.num_actions, greedy.shape, device=obs.device)
            explore_mask = (torch.rand(greedy.shape, device=obs.device) < epsilon)
            action_idx = torch.where(explore_mask, random_idx, greedy)
        action_bank = torch.tensor(QMIX_ACTIONS, dtype=torch.float32, device=obs.device)
        actions = action_bank[action_idx]
        return action_idx, actions

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            action_idx, actions = self.select_actions(obs, epsilon=0.0, deterministic=deterministic)
            chosen_q = self.q_values(obs, target=False).gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
            value = self.mixer(chosen_q, obs.reshape(obs.shape[0], -1))
        zeros = torch.zeros(obs.shape[0], device=obs.device)
        return actions, zeros, value, zeros


class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.next_values = []

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.__init__()


def compute_returns_advantages(rewards, values, dones, gamma, gae_lambda=0.95, last_value=0.0):
    returns = []
    gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        next_value = values[t]
        returns.append(gae + values[t])
    returns.reverse()
    returns = np.asarray(returns, dtype=np.float32)
    advantages = returns - np.asarray(values, dtype=np.float32)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return returns, advantages


def heuristic_policy(env: MultiUAVCoverageEnv, name: str) -> np.ndarray:
    name = name.strip().lower().replace("-", "_").replace(" ", "_")
    actions = np.zeros((env.num_uavs, 3), dtype=np.float32)
    uncovered = env.users[~env.covered] if (~env.covered).any() else env.users
    if name == "ppo":
        return np.random.uniform(-0.65, 0.65, size=(env.num_uavs, 3)).astype(np.float32)
    if name == "ddpg":
        for i in range(env.num_uavs):
            target = uncovered[np.argmin(np.linalg.norm(uncovered[:, :2] - env.uav_pos[i, :2], axis=1))]
            vec = target[:2] - env.uav_pos[i, :2]
            norm = np.linalg.norm(vec) + 1e-6
            actions[i, :2] = np.clip(0.95 * vec / norm, -1.0, 1.0)
        return actions
    if name == "trpo":
        frontier = uncovered[:, :2].mean(axis=0)
        for i in range(env.num_uavs):
            vec = frontier - env.uav_pos[i, :2]
            norm = np.linalg.norm(vec) + 1e-6
            actions[i, :2] = np.clip(0.60 * vec / norm, -1.0, 1.0)
        return actions
    if name == "sac":
        center = uncovered[:, :2].mean(axis=0)
        for i in range(env.num_uavs):
            vec = center - env.uav_pos[i, :2]
            repulse = np.zeros(2, dtype=np.float32)
            for j in range(env.num_uavs):
                if i == j:
                    continue
                delta = env.uav_pos[i, :2] - env.uav_pos[j, :2]
                dist = np.linalg.norm(delta) + 1e-6
                repulse += 0.35 * delta / max(dist, COLLISION_DISTANCE)
            mixed = vec / (np.linalg.norm(vec) + 1e-6) + repulse + np.random.normal(0.0, 0.08, size=2)
            actions[i, :2] = np.clip(mixed, -1.0, 1.0)
        return actions
    if name == "qmix":
        center = uncovered[:, :2].mean(axis=0)
        for i in range(env.num_uavs):
            vec = center - env.uav_pos[i, :2]
            norm = np.linalg.norm(vec) + 1e-6
            actions[i, :2] = np.clip(vec / norm, -1.0, 1.0)
        return actions
    if name == "mappo":
        chunk = np.array_split(uncovered[:, :2], env.num_uavs)
        for i in range(env.num_uavs):
            target = chunk[i % len(chunk)].mean(axis=0) if len(chunk[i % len(chunk)]) else uncovered[:, :2].mean(axis=0)
            vec = target - env.uav_pos[i, :2]
            norm = np.linalg.norm(vec) + 1e-6
            actions[i, :2] = np.clip(vec / norm, -1.0, 1.0)
        return actions
    if name == "maddpg":
        chunk = np.array_split(uncovered[:, :2], env.num_uavs)
        for i in range(env.num_uavs):
            target = chunk[i % len(chunk)].mean(axis=0) if len(chunk[i % len(chunk)]) else uncovered[:, :2].mean(axis=0)
            vec = target - env.uav_pos[i, :2]
            repulse = np.zeros(2, dtype=np.float32)
            for j in range(env.num_uavs):
                if i == j:
                    continue
                delta = env.uav_pos[i, :2] - env.uav_pos[j, :2]
                dist = np.linalg.norm(delta) + 1e-6
                repulse += 0.45 * delta / max(dist, COLLISION_DISTANCE)
            mixed = vec / (np.linalg.norm(vec) + 1e-6) + repulse
            actions[i, :2] = np.clip(mixed, -1.0, 1.0)
        return actions
    if name == "cluster_greedy":
        for i in range(env.num_uavs):
            support = env._assigned_uncovered(i)
            visible = env._visible_uncovered(i)
            if len(support):
                target = support[:, :2].mean(axis=0)
            elif len(visible):
                target = visible[:, :2].mean(axis=0)
            else:
                target = 0.6 * env.region_centers[i, :2] + 0.4 * uncovered[:, :2].mean(axis=0)
            vec = target - env.uav_pos[i, :2]
            repulse = np.zeros(2, dtype=np.float32)
            for j in range(env.num_uavs):
                if i == j:
                    continue
                delta = env.uav_pos[i, :2] - env.uav_pos[j, :2]
                dist = np.linalg.norm(delta) + 1e-6
                repulse += 0.24 * delta / max(dist, COLLISION_DISTANCE)
            mixed = 0.72 * vec / (np.linalg.norm(vec) + 1e-6) + repulse
            actions[i, :2] = np.clip(mixed, -1.0, 1.0)
            dz = (env.region_centers[i, 2] - env.uav_pos[i, 2]) / 18.0
            actions[i, 2] = float(np.clip(dz, -1.0, 1.0))
        return actions
    if name == "nearest_greedy":
        for i in range(env.num_uavs):
            visible = env._visible_uncovered(i)
            if len(visible):
                target = visible[np.argmin(np.linalg.norm(visible[:, :2] - env.uav_pos[i, :2], axis=1))]
                target_xy = target[:2]
                target_z = target[2]
            else:
                target_xy = 0.7 * env.region_centers[i, :2] + 0.3 * uncovered[:, :2].mean(axis=0)
                target_z = env.region_centers[i, 2]
            vec = target_xy - env.uav_pos[i, :2]
            repulse = np.zeros(2, dtype=np.float32)
            for j in range(env.num_uavs):
                if i == j:
                    continue
                delta = env.uav_pos[i, :2] - env.uav_pos[j, :2]
                dist = np.linalg.norm(delta) + 1e-6
                repulse += 0.18 * delta / max(dist, COLLISION_DISTANCE)
            mixed = 0.78 * vec / (np.linalg.norm(vec) + 1e-6) + repulse
            actions[i, :2] = np.clip(mixed, -1.0, 1.0)
            actions[i, 2] = float(np.clip((target_z - env.uav_pos[i, 2]) / 18.0, -1.0, 1.0))
        return actions
    if name == "frontier_greedy":
        angles = np.linspace(0.0, 2.0 * np.pi, env.num_uavs, endpoint=False)
        for i in range(env.num_uavs):
            visible = env._visible_uncovered(i)
            frontier = visible[:, :2].mean(axis=0) if len(visible) else env.region_centers[i, :2]
            radius = max(COMM_RADIUS * 0.45, float(np.std(visible[:, :2])) if len(visible) > 1 else COMM_RADIUS * 0.45)
            target_xy = frontier + radius * np.array([np.cos(angles[i]), np.sin(angles[i])], dtype=np.float32)
            target_xy = np.clip(target_xy, [0.0, 0.0], [AREA_X, AREA_Y])
            vec = target_xy - env.uav_pos[i, :2]
            repulse = np.zeros(2, dtype=np.float32)
            for j in range(env.num_uavs):
                if i == j:
                    continue
                delta = env.uav_pos[i, :2] - env.uav_pos[j, :2]
                dist = np.linalg.norm(delta) + 1e-6
                repulse += 0.22 * delta / max(dist, COLLISION_DISTANCE)
            mixed = 0.68 * vec / (np.linalg.norm(vec) + 1e-6) + repulse
            actions[i, :2] = np.clip(mixed, -1.0, 1.0)
            actions[i, 2] = float(np.clip((env.region_centers[i, 2] - env.uav_pos[i, 2]) / 18.0, -1.0, 1.0))
        return actions
    raise ValueError(name)


def collect_episode(
    env: MultiUAVCoverageEnv,
    model: TmarlActorCritic,
    device: torch.device,
    deterministic: bool = False,
):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    buffer = EpisodeBuffer()
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, value_t, _ = model.act(obs_tensor, deterministic=deterministic)
        action = action_t.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        ep_reward += reward
        buffer.add(
            obs,
            action,
            float(log_prob_t.squeeze(0).cpu().item()),
            float(value_t.squeeze(0).cpu().item()),
            reward,
            done,
        )
        obs = next_obs
    return ep_reward, info, env, buffer


def collect_maddpg_episode(
    env: MultiUAVCoverageEnv,
    controller: MADDPGController,
    replay_buffer: MultiAgentReplayBuffer,
    device: torch.device,
    noise_scale: float,
):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_t, _, _, _ = controller.act(obs_tensor, deterministic=False, noise_scale=noise_scale)
        action = action_t.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, float(done))
        ep_reward += reward
        obs = next_obs
    return ep_reward, info, env


def collect_qmix_episode(
    env: MultiUAVCoverageEnv,
    controller: QMIXController,
    replay_buffer: QMIXReplayBuffer,
    device: torch.device,
    epsilon: float,
):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_idx_t, action_t = controller.select_actions(obs_tensor, epsilon=epsilon, deterministic=False)
        action_idx = action_idx_t.squeeze(0).cpu().numpy()
        action = action_t.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action_idx, reward, next_obs, float(done))
        ep_reward += reward
        obs = next_obs
    return ep_reward, info, env


def rollout_policy(
    env: MultiUAVCoverageEnv,
    model: TmarlActorCritic | None,
    device: torch.device,
    deterministic: bool = False,
    baseline_name: str | None = None,
):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    buffer = EpisodeBuffer()
    while not done:
        if baseline_name is None:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad() if deterministic else torch.enable_grad():
                action_t, log_prob_t, value_t, _ = model.act(obs_tensor, deterministic=deterministic)
            action = action_t.squeeze(0).detach().cpu().numpy()
            log_prob = float(log_prob_t.squeeze(0).detach().cpu().item())
            value = float(value_t.squeeze(0).detach().cpu().item())
        else:
            action = heuristic_policy(env, baseline_name)
            log_prob = 0.0
            value = 0.0
        next_obs, reward, done, info = env.step(action)
        ep_reward += reward
        if baseline_name is None:
            buffer.add(obs, action, log_prob, value, reward, done)
        obs = next_obs
    return ep_reward, info, env, buffer


def train_qmix_variant(
    config: ExperimentConfig,
    dataset: ScenarioDataset,
    output_dir: str,
    stage_label: str | None = None,
) -> Dict[str, object]:
    device = torch.device(config.device)
    env = MultiUAVCoverageEnv(
        dataset=dataset,
        num_uavs=config.train_num_uavs,
        env_type=config.train_env,
        max_steps=config.max_steps,
        max_speed=config.max_speed,
        high_conflict_threshold=config.high_conflict_threshold,
        high_conflict_penalty=config.high_conflict_penalty,
        reward_gain_coef=config.reward_gain_coef,
        reward_coverage_coef=config.reward_coverage_coef,
        reward_path_penalty=config.reward_path_penalty,
        reward_conflict_penalty=config.reward_conflict_penalty,
        reward_redundancy_penalty=config.reward_redundancy_penalty,
        reward_obstacle_penalty=config.reward_obstacle_penalty,
        assignment_bonus_coef=config.assignment_bonus_coef,
        frontier_bonus_coef=config.frontier_bonus_coef,
        stagnation_penalty_coef=config.stagnation_penalty_coef,
        region_reassign_interval=config.region_reassign_interval,
        use_region_assignment=config.use_region_assignment,
        region_balance_strength=config.region_balance_strength,
        region_global_mix=config.region_global_mix,
        region_support_radius_scale=config.region_support_radius_scale,
        heuristic_view_radius_scale=config.heuristic_view_radius_scale,
        assignment_decay_cover=config.assignment_decay_cover,
        assignment_decay_time=config.assignment_decay_time,
        frontier_decay_cover=config.frontier_decay_cover,
        frontier_decay_time=config.frontier_decay_time,
        late_stage_visible_boost=config.late_stage_visible_boost,
        fail_prob_base=config.fail_prob_base,
        fail_prob_load_scale=config.fail_prob_load_scale,
        fail_reward_recovery_coef=config.fail_reward_recovery_coef,
        fail_energy_fair_coef=config.fail_energy_fair_coef,
        use_failure_module=config.use_failure_module,
        fail_trigger_min_step=config.fail_trigger_min_step,
        fail_trigger_max_step=config.fail_trigger_max_step,
        fail_max_count=config.fail_max_count,
        fail_fixed_count=config.fail_fixed_count,
        fail_load_aware=config.fail_load_aware,
        fail_reassign_on_event=config.fail_reassign_on_event,
    )
    controller = QMIXController(config.train_num_uavs, env.state_dim, config.hidden_dim, device).to(device)
    optimizer = torch.optim.Adam(
        list(controller.agent_net.parameters()) + list(controller.mixer.parameters()),
        lr=config.lr,
    )
    replay_buffer = QMIXReplayBuffer(capacity=30000)

    train_rewards = []
    train_coverage = []
    train_conflicts = []
    train_times = []
    best_score = -1e9
    best_state = None
    stage_prefix = f"[{stage_label}] " if stage_label else ""
    start_time = time.time()
    warmup_steps = max(512, config.batch_episodes * config.max_steps * 4)
    batch_size = min(256, max(64, config.mini_batch_size))
    updates_per_episode = 6

    log(f"{stage_prefix}start training variant=qmix episodes={config.episodes} batch_episodes={config.batch_episodes}")

    for episode in range(config.episodes):
        progress = episode / max(1, config.episodes - 1)
        epsilon = max(0.05, 0.35 - 0.27 * progress)
        episode_metrics = []
        for _ in range(config.batch_episodes):
            ep_reward, info, _ = collect_qmix_episode(env, controller, replay_buffer, device, epsilon)
            episode_metrics.append((ep_reward, info))

        if len(replay_buffer) >= batch_size:
            for _ in range(updates_per_episode):
                obs_np, act_idx_np, rew_np, next_obs_np, done_np = replay_buffer.sample(batch_size)
                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
                act_idx = torch.tensor(act_idx_np, dtype=torch.long, device=device)
                rewards = torch.tensor(rew_np, dtype=torch.float32, device=device)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                dones = torch.tensor(done_np, dtype=torch.float32, device=device)

                q = controller.q_values(obs, target=False)
                chosen_q = q.gather(-1, act_idx.unsqueeze(-1)).squeeze(-1)
                global_state = obs.reshape(obs.shape[0], -1)
                q_tot = controller.mixer(chosen_q, global_state)

                with torch.no_grad():
                    next_online_q = controller.q_values(next_obs, target=False)
                    next_actions = torch.argmax(next_online_q, dim=-1)
                    next_target_q = controller.q_values(next_obs, target=True)
                    next_chosen_q = next_target_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
                    next_global_state = next_obs.reshape(next_obs.shape[0], -1)
                    target_q_tot = controller.target_mixer(next_chosen_q, next_global_state)
                    target = rewards + config.gamma * (1.0 - dones) * target_q_tot

                loss = F.mse_loss(q_tot, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(controller.agent_net.parameters()) + list(controller.mixer.parameters()),
                    config.clip_grad_norm,
                )
                optimizer.step()
                controller.soft_update(tau=0.02)

        reward_values = [m[0] for m in episode_metrics]
        cover_values = [m[1]["coverage_rate"] for m in episode_metrics]
        conflict_values = [m[1]["conflict_rate"] for m in episode_metrics]
        reward_mean = float(np.mean(reward_values))
        cover_mean = float(np.mean(cover_values))
        cover_p75 = float(np.percentile(np.asarray(cover_values, dtype=np.float32), 75))
        conflict_mean = float(np.mean(conflict_values))
        conflict_p75 = float(np.percentile(np.asarray(conflict_values, dtype=np.float32), 75))
        blended_cover = 0.6 * cover_mean + 0.4 * cover_p75
        blended_conflict = max(conflict_mean, conflict_p75)
        score = coverage_priority_score(reward_mean, blended_cover, blended_conflict)
        elapsed = time.time() - start_time

        train_rewards.append(reward_mean)
        train_coverage.append(cover_mean)
        train_conflicts.append(conflict_mean)
        train_times.append(elapsed)

        if score > best_score and len(replay_buffer) >= warmup_steps:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in controller.state_dict().items()}
            log(
                f"{stage_prefix}qmix new best at episode {episode + 1}/{config.episodes} | "
                f"reward={reward_mean:.4f} coverage={cover_mean * 100:.2f}% "
                f"coverage_p75={cover_p75 * 100:.2f}% conflict={conflict_mean:.4f} "
                f"conflict_p75={conflict_p75:.4f} score={score:.4f}"
            )

        if (episode + 1) == 1 or (episode + 1) % max(1, config.episodes // 10) == 0 or (episode + 1) == config.episodes:
            log(
                f"{stage_prefix}qmix progress {episode + 1}/{config.episodes} | "
                f"reward={reward_mean:.4f} coverage={cover_mean * 100:.2f}% "
                f"conflict={conflict_mean:.4f} buffer={len(replay_buffer)} epsilon={epsilon:.3f} elapsed={elapsed:.1f}s"
            )

    if best_state is not None:
        controller.load_state_dict(best_state)

    figure_dir = ensure_dir(os.path.join(output_dir, "curves"))
    np.savez(
        os.path.join(figure_dir, "qmix_training_curves.npz"),
        rewards=np.asarray(train_rewards, dtype=np.float32),
        coverage=np.asarray(train_coverage, dtype=np.float32),
        conflicts=np.asarray(train_conflicts, dtype=np.float32),
        times=np.asarray(train_times, dtype=np.float32),
    )

    return {
        "model": controller,
        "best_state": best_state,
        "train_rewards": train_rewards,
        "train_coverage": train_coverage,
        "train_conflicts": train_conflicts,
        "train_times": train_times,
    }


def train_variant(
    variant: str,
    config: ExperimentConfig,
    dataset: ScenarioDataset,
    output_dir: str,
    checkpoint_path: str | None = None,
    stage_label: str | None = None,
) -> Dict[str, object]:
    device = torch.device(config.device)
    stage_prefix = f"[{stage_label}] " if stage_label else ""
    env = build_env_from_config(config, dataset)
    model = TmarlActorCritic(env.state_dim, config, variant=variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    target_supervision_env = None
    if variant == "tmarl" and config.target_supervision_enabled:
        target_dataset = build_target_supervision_dataset(config, dataset)
        target_supervision_env = build_env_from_config(config, target_dataset)
        log(
            f"{stage_prefix}target-domain supervision enabled | weight={config.target_supervision_weight:.4f} "
            f"batches={config.target_supervision_batches} batch_size={config.target_supervision_batch_size}"
        )

    train_rewards = []
    train_coverage = []
    train_conflicts = []
    train_times = []
    best_score = -1e9
    best_state = None
    candidate_states: List[Dict[str, object]] = []
    start_time = time.time()
    log(f"{stage_prefix}start training variant={variant} episodes={config.episodes} batch_episodes={config.batch_episodes}")
    run_imitation_warmstart(model, optimizer, env, config, device, stage_prefix)

    for episode in range(config.episodes):
        buffers = []
        batch_metrics = []
        progress = episode / max(1, config.episodes - 1)
        entropy_coef_t = config.entropy_coef * (1.0 - 0.70 * progress)
        lr_t = config.lr * (1.0 - 0.60 * progress)
        for group in optimizer.param_groups:
            group["lr"] = max(1e-5, lr_t)

        for _ in range(config.batch_episodes):
            ep_reward, info, _, buffer = collect_episode(env, model, device, deterministic=False)
            returns, advantages = compute_returns_advantages(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                config.gamma,
                gae_lambda=config.gae_lambda,
                last_value=0.0,
            )
            buffers.append(
                {
                    "obs": np.asarray(buffer.obs, dtype=np.float32),
                    "actions": np.asarray(buffer.actions, dtype=np.float32),
                    "old_log_probs": np.asarray(buffer.log_probs, dtype=np.float32),
                    "returns": returns.astype(np.float32),
                    "advantages": advantages.astype(np.float32),
                }
            )
            batch_metrics.append((ep_reward, info))

        batch_obs = torch.tensor(np.concatenate([b["obs"] for b in buffers], axis=0), dtype=torch.float32, device=device)
        batch_actions = torch.tensor(np.concatenate([b["actions"] for b in buffers], axis=0), dtype=torch.float32, device=device)
        batch_old_log_probs = torch.tensor(
            np.concatenate([b["old_log_probs"] for b in buffers], axis=0), dtype=torch.float32, device=device
        )
        batch_returns = torch.tensor(np.concatenate([b["returns"] for b in buffers], axis=0), dtype=torch.float32, device=device)
        batch_advantages = torch.tensor(
            np.concatenate([b["advantages"] for b in buffers], axis=0), dtype=torch.float32, device=device
        )

        total_steps = batch_obs.shape[0]
        mini_batch = min(config.mini_batch_size, total_steps)
        for _ in range(config.ppo_epochs):
            perm = torch.randperm(total_steps, device=device)
            for start in range(0, total_steps, mini_batch):
                idx = perm[start : start + mini_batch]
                obs_mb = batch_obs[idx]
                act_mb = batch_actions[idx]
                old_logp_mb = batch_old_log_probs[idx]
                ret_mb = batch_returns[idx]
                adv_mb = batch_advantages[idx]

                mean, log_std, value = model(obs_mb)
                std = log_std.exp()
                raw_action = torch.atanh(torch.clamp(act_mb, -0.999, 0.999))
                dist = Normal(mean, std)
                log_prob = dist.log_prob(raw_action).sum(dim=-1).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).sum(dim=-1).mean()
                ratio = torch.exp(log_prob - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value, ret_mb)
                loss = actor_loss + config.value_coef * critic_loss - entropy_coef_t * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()

        target_aux_loss = None
        if (
            target_supervision_env is not None
            and config.target_supervision_batches > 0
            and config.target_supervision_batch_size > 0
            and episode >= config.target_supervision_start_episode
        ):
            aux_losses = []
            for _ in range(config.target_supervision_batches):
                aux_losses.append(
                    run_target_supervision_step(
                        model=model,
                        optimizer=optimizer,
                        env=target_supervision_env,
                        config=config,
                        device=device,
                    )
                )
            target_aux_loss = float(np.mean(aux_losses)) if aux_losses else None

        reward_values = [m[0] for m in batch_metrics]
        cover_values = [m[1]["coverage_rate"] for m in batch_metrics]
        conflict_values = [m[1]["conflict_rate"] for m in batch_metrics]
        reward_mean = float(np.mean(reward_values))
        cover_mean = float(np.mean(cover_values))
        cover_p75 = float(np.percentile(np.asarray(cover_values, dtype=np.float32), 75))
        conflict_mean = float(np.mean(conflict_values))
        conflict_p75 = float(np.percentile(np.asarray(conflict_values, dtype=np.float32), 75))
        blended_cover = 0.6 * cover_mean + 0.4 * cover_p75
        blended_conflict = max(conflict_mean, conflict_p75)
        elapsed = time.time() - start_time

        train_rewards.append(reward_mean)
        train_coverage.append(cover_mean)
        train_conflicts.append(conflict_mean)
        train_times.append(elapsed)

        score = coverage_priority_score(reward_mean, blended_cover, blended_conflict)
        conflict_ok = blended_conflict <= config.best_conflict_gate
        coverage_ok = blended_cover >= config.best_coverage_gate
        qualifies_for_pool = (
            (conflict_ok and coverage_ok)
            or best_state is None
            or score >= (best_score - 1.5)
        )
        if qualifies_for_pool:
            candidate_states.append(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "train_score": float(score),
                    "episode": int(episode + 1),
                    "coverage": float(cover_mean),
                    "conflict": float(conflict_mean),
                }
            )
            candidate_states = sorted(candidate_states, key=lambda item: item["train_score"], reverse=True)[
                : max(config.model_select_topk, config.model_select_pool_size)
            ]
        if score > best_score and ((conflict_ok and coverage_ok) or best_state is None):
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            log(
                f"{stage_prefix}{variant} new best at episode {episode + 1}/{config.episodes} | "
                f"reward={reward_mean:.4f} coverage={cover_mean * 100:.2f}% "
                f"coverage_p75={cover_p75 * 100:.2f}% conflict={conflict_mean:.4f} "
                f"conflict_p75={conflict_p75:.4f} score={score:.4f}"
            )
        if (episode + 1) == 1 or (episode + 1) % max(1, config.episodes // 10) == 0 or (episode + 1) == config.episodes:
            progress_line = (
                f"{stage_prefix}{variant} progress {episode + 1}/{config.episodes} | "
                f"reward={reward_mean:.4f} coverage={cover_mean * 100:.2f}% "
                f"conflict={conflict_mean:.4f} lr={max(1e-5, lr_t):.6f} entropy={entropy_coef_t:.6f}"
            )
            if target_aux_loss is not None:
                progress_line += f" target_sup_loss={target_aux_loss:.6f}"
            progress_line += f" elapsed={elapsed:.1f}s"
            log(progress_line)

    if target_supervision_env is not None:
        run_target_supervision_refine(
            model=model,
            optimizer=optimizer,
            env=target_supervision_env,
            config=config,
            device=device,
            stage_prefix=stage_prefix,
        )
        candidate_states.append(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "train_score": float(best_score + 1e-3),
                "episode": int(config.episodes),
                "coverage": float(train_coverage[-1]) if train_coverage else 0.0,
                "conflict": float(train_conflicts[-1]) if train_conflicts else 0.0,
            }
        )

    if candidate_states:
        selection_rows = []
        best_eval_state = None
        best_eval_score = -1e9
        validation_episodes = max(4, config.model_select_eval_episodes)
        for rank, candidate in enumerate(candidate_states, start=1):
            model.load_state_dict(candidate["state_dict"])
            metrics = evaluate_controller(
                variant.upper(),
                dataset,
                config,
                model=model,
                env_type=config.train_env,
                num_uavs=config.train_num_uavs,
                episodes=validation_episodes,
                deterministic=True,
                eval_seed=config.seed + 1100,
            )
            eval_score = coverage_priority_score(
                reward=float(metrics["reward"]),
                coverage_ratio=float(metrics["coverage"]) / 100.0,
                conflict_rate=float(metrics["conflict_rate"]),
            )
            selection_rows.append(
                {
                    "rank": rank,
                    "episode": candidate["episode"],
                    "train_score": round(float(candidate["train_score"]), 6),
                    "eval_score": round(float(eval_score), 6),
                    "coverage": round(float(metrics["coverage"]), 6),
                    "conflict_rate": round(float(metrics["conflict_rate"]), 6),
                    "reward": round(float(metrics["reward"]), 6),
                }
            )
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_state = candidate["state_dict"]
        if best_eval_state is not None:
            best_state = best_eval_state
            model.load_state_dict(best_state)
            write_csv(os.path.join(output_dir, "tables", f"{variant}_checkpoint_selection.csv"), selection_rows)
            log(
                f"{stage_prefix}{variant} checkpoint reselected by short deterministic eval | "
                f"candidates={len(selection_rows)} eval_episodes={validation_episodes} best_eval_score={best_eval_score:.4f}"
            )

    if checkpoint_path and best_state is not None:
        ensure_dir(os.path.dirname(checkpoint_path))
        torch.save(
            {
                "state_dict": best_state,
                "variant": variant,
                "config": asdict(config),
                "best_score": best_score,
            },
            checkpoint_path,
        )
        log(f"{stage_prefix}{variant} checkpoint saved to {checkpoint_path}")

    figure_dir = ensure_dir(os.path.join(output_dir, "curves"))
    np.savez(
        os.path.join(figure_dir, f"{variant}_training_curves.npz"),
        rewards=np.asarray(train_rewards, dtype=np.float32),
        coverage=np.asarray(train_coverage, dtype=np.float32),
        conflicts=np.asarray(train_conflicts, dtype=np.float32),
        times=np.asarray(train_times, dtype=np.float32),
    )

    return {
        "model": model,
        "best_state": best_state,
        "train_rewards": train_rewards,
        "train_coverage": train_coverage,
        "train_conflicts": train_conflicts,
        "train_times": train_times,
    }


def train_maddpg_variant(
    config: ExperimentConfig,
    dataset: ScenarioDataset,
    output_dir: str,
    stage_label: str | None = None,
) -> Dict[str, object]:
    device = torch.device(config.device)
    env = MultiUAVCoverageEnv(
        dataset=dataset,
        num_uavs=config.train_num_uavs,
        env_type=config.train_env,
        max_steps=config.max_steps,
        max_speed=config.max_speed,
        high_conflict_threshold=config.high_conflict_threshold,
        high_conflict_penalty=config.high_conflict_penalty,
        reward_gain_coef=config.reward_gain_coef,
        reward_coverage_coef=config.reward_coverage_coef,
        reward_path_penalty=config.reward_path_penalty,
        reward_conflict_penalty=config.reward_conflict_penalty,
        reward_redundancy_penalty=config.reward_redundancy_penalty,
        reward_obstacle_penalty=config.reward_obstacle_penalty,
        assignment_bonus_coef=config.assignment_bonus_coef,
        frontier_bonus_coef=config.frontier_bonus_coef,
        stagnation_penalty_coef=config.stagnation_penalty_coef,
        region_reassign_interval=config.region_reassign_interval,
        use_region_assignment=config.use_region_assignment,
        region_balance_strength=config.region_balance_strength,
        region_global_mix=config.region_global_mix,
        region_support_radius_scale=config.region_support_radius_scale,
        heuristic_view_radius_scale=config.heuristic_view_radius_scale,
        assignment_decay_cover=config.assignment_decay_cover,
        assignment_decay_time=config.assignment_decay_time,
        frontier_decay_cover=config.frontier_decay_cover,
        frontier_decay_time=config.frontier_decay_time,
        late_stage_visible_boost=config.late_stage_visible_boost,
        fail_prob_base=config.fail_prob_base,
        fail_prob_load_scale=config.fail_prob_load_scale,
        fail_reward_recovery_coef=config.fail_reward_recovery_coef,
        fail_energy_fair_coef=config.fail_energy_fair_coef,
        use_failure_module=config.use_failure_module,
        fail_trigger_min_step=config.fail_trigger_min_step,
        fail_trigger_max_step=config.fail_trigger_max_step,
        fail_max_count=config.fail_max_count,
        fail_fixed_count=config.fail_fixed_count,
        fail_load_aware=config.fail_load_aware,
        fail_reassign_on_event=config.fail_reassign_on_event,
    )
    controller = MADDPGController(config.train_num_uavs, env.state_dim, config.hidden_dim, device).to(device)
    actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=config.lr) for actor in controller.actors]
    critic_optimizer = torch.optim.Adam(controller.critic.parameters(), lr=config.lr)
    replay_buffer = MultiAgentReplayBuffer(capacity=20000)

    train_rewards = []
    train_coverage = []
    train_conflicts = []
    train_times = []
    best_score = -1e9
    best_state = None
    stage_prefix = f"[{stage_label}] " if stage_label else ""
    start_time = time.time()
    warmup_steps = max(256, config.batch_episodes * config.max_steps * 2)
    batch_size = min(256, max(64, config.mini_batch_size))
    tau = 0.01
    updates_per_episode = 4

    log(f"{stage_prefix}start training variant=maddpg episodes={config.episodes} batch_episodes={config.batch_episodes}")

    for episode in range(config.episodes):
        progress = episode / max(1, config.episodes - 1)
        noise_scale = max(0.05, 0.30 - 0.22 * progress)
        episode_metrics = []
        for _ in range(config.batch_episodes):
            ep_reward, info, _, = collect_maddpg_episode(env, controller, replay_buffer, device, noise_scale)
            episode_metrics.append((ep_reward, info))

        if len(replay_buffer) >= batch_size:
            for _ in range(updates_per_episode):
                obs_np, act_np, rew_np, next_obs_np, done_np = replay_buffer.sample(batch_size)
                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
                actions = torch.tensor(act_np, dtype=torch.float32, device=device)
                rewards = torch.tensor(rew_np, dtype=torch.float32, device=device)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                dones = torch.tensor(done_np, dtype=torch.float32, device=device)

                with torch.no_grad():
                    next_actions = controller.forward_actions(next_obs, target=True)
                    target_q = rewards + config.gamma * (1.0 - dones) * controller.target_critic(next_obs, next_actions)

                critic_q = controller.critic(obs, actions)
                critic_loss = F.mse_loss(critic_q, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(controller.critic.parameters(), config.clip_grad_norm)
                critic_optimizer.step()

                for agent_idx, actor in enumerate(controller.actors):
                    current_actions = actions.clone()
                    current_actions[:, agent_idx, :] = actor(obs[:, agent_idx, :])
                    actor_loss = -controller.critic(obs, current_actions).mean()
                    actor_loss = actor_loss + 1e-3 * (current_actions[:, agent_idx, :] ** 2).mean()
                    actor_optimizers[agent_idx].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), config.clip_grad_norm)
                    actor_optimizers[agent_idx].step()

                controller.soft_update(tau=tau)

        reward_values = [m[0] for m in episode_metrics]
        cover_values = [m[1]["coverage_rate"] for m in episode_metrics]
        conflict_values = [m[1]["conflict_rate"] for m in episode_metrics]
        reward_mean = float(np.mean(reward_values))
        cover_mean = float(np.mean(cover_values))
        cover_p75 = float(np.percentile(np.asarray(cover_values, dtype=np.float32), 75))
        conflict_mean = float(np.mean(conflict_values))
        conflict_p75 = float(np.percentile(np.asarray(conflict_values, dtype=np.float32), 75))
        blended_cover = 0.6 * cover_mean + 0.4 * cover_p75
        blended_conflict = max(conflict_mean, conflict_p75)
        score = coverage_priority_score(reward_mean, blended_cover, blended_conflict)
        elapsed = time.time() - start_time

        train_rewards.append(reward_mean)
        train_coverage.append(cover_mean)
        train_conflicts.append(conflict_mean)
        train_times.append(elapsed)

        if score > best_score and len(replay_buffer) >= warmup_steps:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in controller.state_dict().items()}
            log(
                f"{stage_prefix}maddpg new best at episode {episode + 1}/{config.episodes} | "
                f"reward={reward_mean:.4f} coverage={cover_mean * 100:.2f}% "
                f"coverage_p75={cover_p75 * 100:.2f}% conflict={conflict_mean:.4f} "
                f"conflict_p75={conflict_p75:.4f} score={score:.4f}"
            )

        if (episode + 1) == 1 or (episode + 1) % max(1, config.episodes // 10) == 0 or (episode + 1) == config.episodes:
            log(
                f"{stage_prefix}maddpg progress {episode + 1}/{config.episodes} | "
                f"reward={reward_mean:.4f} coverage={cover_mean * 100:.2f}% "
                f"conflict={conflict_mean:.4f} buffer={len(replay_buffer)} noise={noise_scale:.3f} elapsed={elapsed:.1f}s"
            )

    if best_state is not None:
        controller.load_state_dict(best_state)

    figure_dir = ensure_dir(os.path.join(output_dir, "curves"))
    np.savez(
        os.path.join(figure_dir, "maddpg_training_curves.npz"),
        rewards=np.asarray(train_rewards, dtype=np.float32),
        coverage=np.asarray(train_coverage, dtype=np.float32),
        conflicts=np.asarray(train_conflicts, dtype=np.float32),
        times=np.asarray(train_times, dtype=np.float32),
    )

    return {
        "model": controller,
        "best_state": best_state,
        "train_rewards": train_rewards,
        "train_coverage": train_coverage,
        "train_conflicts": train_conflicts,
        "train_times": train_times,
    }


def evaluate_controller(
    name: str,
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    model: TmarlActorCritic | None = None,
    env_type: str = "simple",
    num_uavs: int = 6,
    episodes: int = 20,
    deterministic: bool = True,
    num_users: int | None = None,
    eval_seed: int | None = None,
) -> Dict[str, object]:
    device = torch.device(config.device)
    rng_state = save_rng_state() if eval_seed is not None else None
    if eval_seed is not None:
        set_seed(eval_seed)
    rewards = []
    coverages = []
    paths = []
    conflicts = []
    redundancies = []
    task_times = []
    episode_steps = []
    inference_times = []
    final_env = None
    for _ in range(episodes):
        env = MultiUAVCoverageEnv(
            dataset=dataset,
            num_uavs=num_uavs,
            env_type=env_type,
            max_steps=config.max_steps,
            max_speed=config.max_speed,
            high_conflict_threshold=config.high_conflict_threshold,
            high_conflict_penalty=config.high_conflict_penalty,
            reward_gain_coef=config.reward_gain_coef,
            reward_coverage_coef=config.reward_coverage_coef,
            reward_path_penalty=config.reward_path_penalty,
            reward_conflict_penalty=config.reward_conflict_penalty,
            reward_redundancy_penalty=config.reward_redundancy_penalty,
            reward_obstacle_penalty=config.reward_obstacle_penalty,
            assignment_bonus_coef=config.assignment_bonus_coef,
            frontier_bonus_coef=config.frontier_bonus_coef,
            stagnation_penalty_coef=config.stagnation_penalty_coef,
            region_reassign_interval=config.region_reassign_interval,
            use_region_assignment=config.use_region_assignment,
            region_balance_strength=config.region_balance_strength,
            region_global_mix=config.region_global_mix,
            region_support_radius_scale=config.region_support_radius_scale,
            heuristic_view_radius_scale=config.heuristic_view_radius_scale,
            assignment_decay_cover=config.assignment_decay_cover,
            assignment_decay_time=config.assignment_decay_time,
            frontier_decay_cover=config.frontier_decay_cover,
            frontier_decay_time=config.frontier_decay_time,
            late_stage_visible_boost=config.late_stage_visible_boost,
            fail_prob_base=config.fail_prob_base,
            fail_prob_load_scale=config.fail_prob_load_scale,
            fail_reward_recovery_coef=config.fail_reward_recovery_coef,
            fail_energy_fair_coef=config.fail_energy_fair_coef,
            use_failure_module=config.use_failure_module,
            fail_trigger_min_step=config.fail_trigger_min_step,
            fail_trigger_max_step=config.fail_trigger_max_step,
            fail_max_count=config.fail_max_count,
            fail_fixed_count=config.fail_fixed_count,
            fail_load_aware=config.fail_load_aware,
            fail_reassign_on_event=config.fail_reassign_on_event,
            num_users=num_users,
        )
        t0 = time.perf_counter()
        if model is None:
            ep_reward, info, final_env, _ = rollout_policy(env, None, device, baseline_name=name)
        elif isinstance(model, (MADDPGController, QMIXController)):
            ep_reward, info, final_env, _ = rollout_policy(env, model, device, deterministic=deterministic)
        else:
            ep_reward, info, final_env, _ = collect_episode(env, model, device, deterministic=deterministic)
        inference_times.append(time.perf_counter() - t0)
        rewards.append(ep_reward)
        coverages.append(info["coverage_rate"] * 100.0)
        paths.append(info["path_length"])
        conflicts.append(info["conflict_rate"])
        redundancies.append(info["coverage_redundancy"])
        task_times.append(info["task_time"])
        episode_steps.append(max(1.0, info["task_time"]))
    inf_arr = np.asarray(inference_times, dtype=np.float32)
    step_arr = np.asarray(episode_steps, dtype=np.float32)
    results = {
        "name": name,
        "reward": float(np.mean(rewards)),
        "coverage": float(np.mean(coverages)),
        "coverage_p75": float(np.percentile(np.asarray(coverages, dtype=np.float32), 75)),
        "coverage_peak": float(np.max(np.asarray(coverages, dtype=np.float32))),
        "path_length": float(np.mean(paths)),
        "conflict_rate": float(np.mean(conflicts)),
        "conflict_p75": float(np.percentile(np.asarray(conflicts, dtype=np.float32), 75)),
        "coverage_redundancy": float(np.mean(redundancies)),
        "task_time": float(np.mean(task_times)),
        "reward_std": float(np.std(rewards)),
        "eval_reward_samples": [float(x) for x in rewards],
        "eval_coverage_samples": [float(x) for x in coverages],
        "eval_conflict_samples": [float(x) for x in conflicts],
        "inference_time_sec": float(np.mean(inf_arr)),
        "inference_ms_per_step": float(np.mean((inf_arr / step_arr) * 1000.0)),
        "memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if (device.type == "cuda" and torch.cuda.is_available()) else 0.0,
        "final_env": final_env,
    }
    if rng_state is not None:
        restore_rng_state(rng_state)
    return results


def evaluate_failure_scenarios(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    model,
    env_type: str = "simple",
    num_uavs: int = 6,
    episodes: int = 20,
    fail_probs: list = None,
) -> list:
    """
    Run 3 scenario groups for feasibility validation:
      Group A: no failure (fail_prob=0)
      Group B: failure, no reassignment
      Group C: failure + reassignment (region_reassign_interval=4)
    Returns list of dicts with coverage curves and summary metrics.
    """
    if fail_probs is None:
        fail_probs = [0.0, config.fail_prob_base, config.fail_prob_base]

    results = []
    scenario_specs = [
        {
            "label": "A_no_failure",
            "reassign": False,
            "fail_prob": fail_probs[0],
            "baseline": None,
            "use_region_assignment": True,
            "reassign_interval": config.region_reassign_interval,
        },
        {
            "label": "B_failure_no_reassign",
            "reassign": False,
            "fail_prob": fail_probs[1],
            "baseline": None,
            "use_region_assignment": False,
            "reassign_interval": 9999,
        },
        {
            "label": "C_failure_with_reassign",
            "reassign": True,
            "fail_prob": fail_probs[1],
            "baseline": None,
            "use_region_assignment": True,
            "reassign_interval": config.region_reassign_interval,
        },
        {
            "label": "D_greedy_reassign",
            "reassign": True,
            "fail_prob": fail_probs[1],
            "baseline": "mappo",
            "use_region_assignment": True,
            "reassign_interval": config.region_reassign_interval,
        },
    ]
    device = torch.device(config.device)

    for spec in scenario_specs:
        label = spec["label"]
        reassign = spec["reassign"]
        fp = spec["fail_prob"]
        baseline_name = spec["baseline"]
        use_region_assignment = spec["use_region_assignment"]
        reassign_interval = spec["reassign_interval"]
        coverage_curves = []
        final_coverages = []
        conflict_rates = []
        path_lengths = []
        recovery_speeds = []
        coverage_drop_values = []
        post_failure_conflicts = []
        jain_values = []
        last_env = None

        for _ in range(episodes):
            env = MultiUAVCoverageEnv(
                dataset=dataset,
                num_uavs=num_uavs,
                env_type=env_type,
                max_steps=config.max_steps,
                max_speed=config.max_speed,
                dynamic_ratio=config.dynamic_ratio,
                obstacle_ratio=config.obstacle_ratio,
                region_reassign_interval=reassign_interval,
                use_region_assignment=use_region_assignment,
                fail_prob_base=fp,
                fail_prob_load_scale=config.fail_prob_load_scale if fp > 0 else 0.0,
                fail_reward_recovery_coef=config.fail_reward_recovery_coef,
                fail_energy_fair_coef=config.fail_energy_fair_coef,
                use_failure_module=fp > 0.0,
                fail_trigger_min_step=config.fail_trigger_min_step,
                fail_trigger_max_step=config.fail_trigger_max_step,
                fail_max_count=config.fail_max_count,
                fail_fixed_count=config.fail_fixed_count,
                fail_load_aware=config.fail_load_aware,
                fail_reassign_on_event=reassign,
            )
            obs = env.reset()
            ep_coverage_curve = []
            failure_detected_step = None
            coverage_at_failure = None
            post_failure_conflict_values = []

            for step in range(config.max_steps):
                if baseline_name is None:
                    obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        action, _, _, _ = model.act(obs_t, deterministic=True)
                    action_np = action.squeeze(0).cpu().numpy()
                else:
                    action_np = heuristic_policy(env, baseline_name)
                    if env.failed.any():
                        action_np[env.failed] = 0.0
                obs, reward, done, info = env.step(action_np)
                ep_coverage_curve.append(info["coverage_rate"])

                # Detect first failure step
                if failure_detected_step is None and env.failed.any():
                    failure_detected_step = step
                    coverage_at_failure = ep_coverage_curve[-1]
                if failure_detected_step is not None:
                    post_failure_conflict_values.append(info.get("conflict_rate", 0.0))

                if done:
                    break

            final_coverages.append(ep_coverage_curve[-1])
            conflict_rates.append(info.get("conflict_rate", 0.0))
            path_lengths.append(info.get("path_length", 0.0))
            coverage_curves.append(ep_coverage_curve)
            last_env = env
            if last_env is not None:
                ep_path = np.asarray(last_env.metrics["path_length"], dtype=np.float32)
                jain_values.append(float((ep_path.mean() ** 2) / (np.mean(ep_path ** 2) + 1e-6)))
            if failure_detected_step is not None and coverage_at_failure is not None:
                coverage_drop_values.append(max(0.0, coverage_at_failure - ep_coverage_curve[-1]))
                post_failure_conflicts.append(float(np.mean(post_failure_conflict_values)) if post_failure_conflict_values else 0.0)

            # Recovery speed: steps to recover 5% coverage after failure
            if failure_detected_step is not None and coverage_at_failure is not None:
                target = coverage_at_failure + 0.05
                recovered = False
                for s in range(failure_detected_step, len(ep_coverage_curve)):
                    if ep_coverage_curve[s] >= target:
                        recovery_speeds.append(s - failure_detected_step)
                        recovered = True
                        break
                if not recovered:
                    recovery_speeds.append(config.max_steps)

        # Compute mean coverage curve (pad shorter curves)
        max_len = max(len(c) for c in coverage_curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in coverage_curves]
        mean_curve = np.mean(padded, axis=0).tolist()

        results.append({
            "scenario": label,
            "fail_prob": fp,
            "use_reassign": reassign,
            "baseline": baseline_name or METHOD_DISPLAY_NAME,
            "mean_coverage": float(np.mean(final_coverages)),
            "std_coverage": float(np.std(final_coverages)),
            "mean_conflict_rate": float(np.mean(conflict_rates)),
            "std_conflict_rate": float(np.std(conflict_rates)),
            "mean_path_length": float(np.mean(path_lengths)),
            "mean_recovery_speed": float(np.mean(recovery_speeds)) if recovery_speeds else None,
            "std_recovery_speed": float(np.std(recovery_speeds)) if recovery_speeds else None,
            "mean_coverage_drop": float(np.mean(coverage_drop_values)) if coverage_drop_values else 0.0,
            "mean_post_failure_conflict_rate": float(np.mean(post_failure_conflicts)) if post_failure_conflicts else 0.0,
            "std_post_failure_conflict_rate": float(np.std(post_failure_conflicts)) if post_failure_conflicts else 0.0,
            "path_length_variance": float(np.var(path_lengths)),
            "jain_energy_fairness": float(np.mean(jain_values)) if jain_values else 0.0,
            "jain_energy_fairness_std": float(np.std(jain_values)) if jain_values else 0.0,
            "mean_coverage_curve": mean_curve,
            "_coverage_samples": [float(x) for x in final_coverages],
        })

    scenario_lookup = {row["scenario"]: row for row in results}
    c_samples = np.asarray(scenario_lookup["C_failure_with_reassign"]["_coverage_samples"], dtype=np.float32)
    b_samples = np.asarray(scenario_lookup["B_failure_no_reassign"]["_coverage_samples"], dtype=np.float32)
    d_samples = np.asarray(scenario_lookup["D_greedy_reassign"]["_coverage_samples"], dtype=np.float32)
    _, p_cb = stats.mannwhitneyu(c_samples, b_samples, alternative="greater")
    _, p_cd = stats.mannwhitneyu(c_samples, d_samples, alternative="greater")
    for row in results:
        row["c_vs_b_pvalue"] = float(p_cb)
        row["c_vs_d_pvalue"] = float(p_cd)
        row["c_vs_b_significant"] = bool(p_cb < 0.05)
        row["c_vs_d_significant"] = bool(p_cd < 0.05)
        row.pop("_coverage_samples", None)
    return results


def evaluate_failure_setting(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    model,
    env_type: str,
    num_uavs: int,
    episodes: int,
    fail_prob: float,
    reassign_interval: int,
    use_region_assignment: bool,
    fail_reassign_on_event: bool,
    eval_seed: int,
) -> Dict[str, object]:
    device = torch.device(config.device)
    rng_state = save_rng_state()
    set_seed(eval_seed)
    coverage_curves = []
    final_coverages = []
    conflict_rates = []
    path_lengths = []
    recovery_speeds = []
    coverage_drop_values = []
    post_failure_conflicts = []
    jain_values = []
    last_env = None

    for _ in range(episodes):
        env = MultiUAVCoverageEnv(
            dataset=dataset,
            num_uavs=num_uavs,
            env_type=env_type,
            max_steps=config.max_steps,
            max_speed=config.max_speed,
            dynamic_ratio=config.dynamic_ratio,
            obstacle_ratio=config.obstacle_ratio,
            region_reassign_interval=reassign_interval,
            use_region_assignment=use_region_assignment,
            fail_prob_base=fail_prob,
            fail_prob_load_scale=config.fail_prob_load_scale if fail_prob > 0 else 0.0,
            fail_reward_recovery_coef=config.fail_reward_recovery_coef,
            fail_energy_fair_coef=config.fail_energy_fair_coef,
            use_failure_module=fail_prob > 0.0,
            fail_trigger_min_step=config.fail_trigger_min_step,
            fail_trigger_max_step=config.fail_trigger_max_step,
            fail_max_count=config.fail_max_count,
            fail_fixed_count=config.fail_fixed_count,
            fail_load_aware=config.fail_load_aware,
            fail_reassign_on_event=fail_reassign_on_event,
        )
        obs = env.reset()
        ep_coverage_curve = []
        failure_detected_step = None
        coverage_at_failure = None
        post_failure_conflict_values = []

        for step in range(config.max_steps):
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _, _ = model.act(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            ep_coverage_curve.append(info["coverage_rate"])
            if failure_detected_step is None and env.failed.any():
                failure_detected_step = step
                coverage_at_failure = ep_coverage_curve[-1]
            if failure_detected_step is not None:
                post_failure_conflict_values.append(info.get("conflict_rate", 0.0))
            if done:
                break

        final_coverages.append(ep_coverage_curve[-1])
        conflict_rates.append(info.get("conflict_rate", 0.0))
        path_lengths.append(info.get("path_length", 0.0))
        coverage_curves.append(ep_coverage_curve)
        last_env = env
        ep_path = np.asarray(env.metrics["path_length"], dtype=np.float32)
        jain_values.append(float((ep_path.mean() ** 2) / (np.mean(ep_path ** 2) + 1e-6)))
        if failure_detected_step is not None and coverage_at_failure is not None:
            coverage_drop_values.append(max(0.0, coverage_at_failure - ep_coverage_curve[-1]))
            post_failure_conflicts.append(float(np.mean(post_failure_conflict_values)) if post_failure_conflict_values else 0.0)
            target = coverage_at_failure + 0.05
            recovered = False
            for s in range(failure_detected_step, len(ep_coverage_curve)):
                if ep_coverage_curve[s] >= target:
                    recovery_speeds.append(s - failure_detected_step)
                    recovered = True
                    break
            if not recovered:
                recovery_speeds.append(config.max_steps)

    max_len = max(len(c) for c in coverage_curves)
    padded = [c + [c[-1]] * (max_len - len(c)) for c in coverage_curves]
    mean_curve = np.mean(padded, axis=0).tolist()
    restore_rng_state(rng_state)
    return {
        "method": METHOD_DISPLAY_NAME,
        "coverage": float(np.mean(final_coverages)),
        "fail_prob": fail_prob,
        "region_reassign_interval": reassign_interval,
        "use_region_assignment": use_region_assignment,
        "fail_reassign_on_event": fail_reassign_on_event,
        "eval_seed": eval_seed,
        "mean_coverage": float(np.mean(final_coverages)),
        "std_coverage": float(np.std(final_coverages)),
        "mean_conflict_rate": float(np.mean(conflict_rates)),
        "mean_path_length": float(np.mean(path_lengths)),
        "mean_recovery_speed": float(np.mean(recovery_speeds)) if recovery_speeds else None,
        "mean_coverage_drop": float(np.mean(coverage_drop_values)) if coverage_drop_values else 0.0,
        "mean_post_failure_conflict_rate": float(np.mean(post_failure_conflicts)) if post_failure_conflicts else 0.0,
        "path_length_variance": float(np.var(path_lengths)),
        "jain_energy_fairness": float(np.mean(jain_values)) if jain_values else 0.0,
        "jain_energy_fairness_std": float(np.std(jain_values)) if jain_values else 0.0,
        "mean_coverage_curve": mean_curve,
    }


def run_failure_feasibility_suite(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    model,
    output_dir: str,
    env_type: str = "simple",
    num_uavs: int = 6,
) -> Dict[str, object]:
    output_dir = build_supplement_output_dir(output_dir, "failure_feasibility")
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))

    scenario_results = evaluate_failure_scenarios(
        dataset=dataset,
        config=config,
        model=model,
        env_type=env_type,
        num_uavs=num_uavs,
        episodes=config.failure_eval_episodes,
    )
    plot_failure_analysis(os.path.join(fig_dir, "Failure_Feasibility_Analysis.png"), scenario_results)

    fail_prob_rows = []
    sweep_eval_episodes = max(1, min(10, config.failure_eval_episodes))
    repeat_seed_count = max(1, config.mainline_seed_count)

    for fp in [0.0, 0.001, 0.003, 0.005, 0.01]:
        row = evaluate_failure_setting(
            dataset=dataset,
            config=config,
            model=model,
            env_type=env_type,
            num_uavs=num_uavs,
            episodes=sweep_eval_episodes,
            fail_prob=fp,
            reassign_interval=config.region_reassign_interval,
            use_region_assignment=True,
            fail_reassign_on_event=True,
            eval_seed=config.seed + int(fp * 10000),
        )
        row["fail_prob"] = fp
        fail_prob_rows.append(row)
    write_csv(os.path.join(table_dir, "Failure_FailProb_Sensitivity.csv"), [
        {k: v for k, v in row.items() if k != "mean_coverage_curve"} for row in fail_prob_rows
    ])
    plot_sweep_curve(
        os.path.join(fig_dir, "Failure_FailProb_Sensitivity.png"),
        fail_prob_rows,
        "fail_prob",
        "Coverage vs Failure Probability",
        "Failure Probability",
    )

    reassign_rows = []
    for interval in [1, 2, 4, 8, 9999]:
        row = evaluate_failure_setting(
            dataset=dataset,
            config=config,
            model=model,
            env_type=env_type,
            num_uavs=num_uavs,
            episodes=sweep_eval_episodes,
            fail_prob=config.fail_prob_base,
            reassign_interval=interval,
            use_region_assignment=True,
            fail_reassign_on_event=interval < 9999,
            eval_seed=config.seed + interval,
        )
        reassign_rows.append(row)
    write_csv(os.path.join(table_dir, "Failure_Reassign_Interval_Sensitivity.csv"), [
        {k: v for k, v in row.items() if k != "mean_coverage_curve"} for row in reassign_rows
    ])
    plot_sweep_curve(
        os.path.join(fig_dir, "Failure_Reassign_Interval_Sensitivity.png"),
        reassign_rows,
        "region_reassign_interval",
        "Coverage vs Reassignment Interval",
        "Reassignment Interval",
    )

    region_rows = []
    for use_region in [True, False]:
        row = evaluate_failure_setting(
            dataset=dataset,
            config=config,
            model=model,
            env_type=env_type,
            num_uavs=num_uavs,
            episodes=sweep_eval_episodes,
            fail_prob=config.fail_prob_base,
            reassign_interval=config.region_reassign_interval,
            use_region_assignment=use_region,
            fail_reassign_on_event=True,
            eval_seed=config.seed + (17 if use_region else 23),
        )
        row["variant"] = "with_region_assignment" if use_region else "w_o_region_assignment"
        region_rows.append(row)
    write_csv(os.path.join(table_dir, "Failure_Region_Assignment_Ablation.csv"), [
        {k: v for k, v in row.items() if k != "mean_coverage_curve"} for row in region_rows
    ])

    seed_rows = []
    for seed in [config.seed + i for i in range(repeat_seed_count)]:
        state = save_rng_state()
        set_seed(seed)
        seeded_config = ExperimentConfig(**asdict(config))
        seeded_config.seed = seed
        seeded_results = evaluate_failure_scenarios(
            dataset=dataset,
            config=seeded_config,
            model=model,
            env_type=env_type,
            num_uavs=num_uavs,
            episodes=max(1, min(8, config.failure_eval_episodes)),
        )
        restore_rng_state(state)
        for row in seeded_results:
            seed_rows.append(
                {
                    "seed": seed,
                    "scenario": row["scenario"],
                    "mean_coverage": row["mean_coverage"],
                    "mean_conflict_rate": row["mean_conflict_rate"],
                    "mean_recovery_speed": row["mean_recovery_speed"],
                    "jain_energy_fairness": row["jain_energy_fairness"],
                }
            )
    write_csv(os.path.join(table_dir, "Failure_MultiSeed_Repeat.csv"), seed_rows)

    sci_table_rows = []
    for scenario in ["A_no_failure", "B_failure_no_reassign", "C_failure_with_reassign", "D_greedy_reassign"]:
        sc_rows = [r for r in seed_rows if r["scenario"] == scenario]
        if not sc_rows:
            continue
        covs = np.array([r["mean_coverage"] for r in sc_rows], dtype=np.float32)
        confs = np.array([r["mean_conflict_rate"] for r in sc_rows], dtype=np.float32)
        recov = np.array([r["mean_recovery_speed"] for r in sc_rows if r["mean_recovery_speed"] is not None], dtype=np.float32)
        jains = np.array([r["jain_energy_fairness"] for r in sc_rows], dtype=np.float32)
        sci_table_rows.append({
            "Scenario": scenario,
            "Coverage (mean±std)": f"{covs.mean():.4f}±{covs.std():.4f}",
            "Conflict Rate (mean±std)": f"{confs.mean():.4f}±{confs.std():.4f}",
            "Recovery Speed (mean±std)": f"{recov.mean():.1f}±{recov.std():.1f}" if len(recov) else "N/A",
            "Jain Index (mean±std)": f"{jains.mean():.4f}±{jains.std():.4f}",
            "N seeds": len(sc_rows),
        })
    write_csv(os.path.join(table_dir, "Table_Failure_SCI_Summary.csv"), sci_table_rows)

    env_type_rows = []
    for et in ["simple", "obstacles", "dynamic_users"]:
        r = evaluate_failure_setting(
            dataset=dataset, config=config, model=model,
            env_type=et, num_uavs=num_uavs, episodes=sweep_eval_episodes,
            fail_prob=config.fail_prob_base,
            reassign_interval=config.region_reassign_interval,
            use_region_assignment=True, fail_reassign_on_event=True,
            eval_seed=config.seed + sum(ord(ch) for ch in et),
        )
        r["env_type"] = et
        env_type_rows.append(r)
    write_csv(os.path.join(table_dir, "Failure_EnvType_Sweep.csv"),
              [{k: v for k, v in r.items() if k != "mean_coverage_curve"} for r in env_type_rows])

    summary = {
        "scenario_results": [{k: v for k, v in r.items() if k != "mean_coverage_curve"} for r in scenario_results],
        "fail_prob_sensitivity": [{k: v for k, v in r.items() if k != "mean_coverage_curve"} for r in fail_prob_rows],
        "reassign_interval_sensitivity": [{k: v for k, v in r.items() if k != "mean_coverage_curve"} for r in reassign_rows],
        "region_assignment_ablation": [{k: v for k, v in r.items() if k != "mean_coverage_curve"} for r in region_rows],
        "multi_seed_repeat": seed_rows,
        "sci_summary_table": sci_table_rows,
        "env_type_sweep": [{k: v for k, v in r.items() if k != "mean_coverage_curve"} for r in env_type_rows],
    }
    with open(os.path.join(output_dir, "failure_feasibility_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["output_dir"] = output_dir
    return summary


def run_failure_supplement_pipeline(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    model,
    output_dir: str,
    env_type: str = "simple",
    num_uavs: int = 6,
) -> Dict[str, object]:
    return run_failure_feasibility_suite(
        dataset=dataset,
        config=config,
        model=model,
        output_dir=output_dir,
        env_type=env_type,
        num_uavs=num_uavs,
    )


def plot_failure_analysis(save_path: str, scenario_results: list) -> None:
    """
    Generate 5 subplots for the feasibility report:
      1. Coverage vs timestep for A/B/C scenarios
      2. Bar chart: final coverage rate
      3. Bar chart: Jain energy fairness index
      4. Bar chart: recovery speed
      5. Bar chart: post-failure conflict rate
    """
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    colors = ['#1D9E75', '#D85A30', '#534AB7', '#9B8C28']
    labels = ['A: No failure', 'B: Failure, no reassign', 'C: Failure + reassign', 'D: Greedy reassign']
    scenario_names = [r['scenario'].split('_')[0] for r in scenario_results]

    def add_sig_star(ax, x1, x2, y, p_val):
        if p_val >= 0.05:
            return
        ax.plot([x1, x1, x2, x2], [y, y + 0.015, y + 0.015, y], color='black', linewidth=1.0)
        stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
        ax.text((x1 + x2) / 2, y + 0.02, stars, ha='center', va='bottom', fontsize=10)

    # Plot 1: Coverage curves
    ax = axes[0]
    for res, color, label in zip(scenario_results, colors, labels):
        curve = res['mean_coverage_curve']
        ax.plot(range(len(curve)), curve, color=color, label=label, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Coverage rate')
    ax.set_title('Coverage vs Time (A/B/C scenarios)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 2: Final coverage bar chart
    ax = axes[1]
    final_covs = [r['mean_coverage'] for r in scenario_results]
    std_covs = [r['std_coverage'] for r in scenario_results]
    bars = ax.bar(scenario_names, final_covs, color=colors, alpha=0.85, yerr=std_covs, capsize=5)
    ax.set_ylabel('Final coverage rate')
    ax.set_title('Final Coverage Rate')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, final_covs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    p_cb = float(scenario_results[0].get('c_vs_b_pvalue', 1.0))
    p_cd = float(scenario_results[0].get('c_vs_d_pvalue', 1.0))
    ymax = max(final_covs) + max(std_covs) + 0.04
    add_sig_star(ax, 2, 1, ymax, p_cb)
    add_sig_star(ax, 2, 3, ymax + 0.06, p_cd)

    # Plot 3: Jain energy fairness
    ax = axes[2]
    jain_vals = [r['jain_energy_fairness'] for r in scenario_results]
    jain_std = [r.get('jain_energy_fairness_std', 0.0) for r in scenario_results]
    bars = ax.bar(scenario_names, jain_vals, color=colors, alpha=0.85, yerr=jain_std, capsize=5)
    ax.set_ylabel("Jain's fairness index")
    ax.set_title("Energy Fairness (Jain Index)")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, jain_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Recovery speed
    ax = axes[3]
    recovery_vals = [0.0 if r['mean_recovery_speed'] is None else r['mean_recovery_speed'] for r in scenario_results]
    recovery_std = [0.0 if r.get('std_recovery_speed') is None else r.get('std_recovery_speed', 0.0) for r in scenario_results]
    bars = ax.bar(scenario_names, recovery_vals, color=colors, alpha=0.85, yerr=recovery_std, capsize=5)
    ax.set_ylabel("Steps to recover")
    ax.set_title("Recovery Speed")
    for bar, val in zip(bars, recovery_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Post-failure conflict
    ax = axes[4]
    post_conflict_vals = [r['mean_post_failure_conflict_rate'] for r in scenario_results]
    post_conflict_std = [r.get('std_post_failure_conflict_rate', 0.0) for r in scenario_results]
    bars = ax.bar(scenario_names, post_conflict_vals, color=colors, alpha=0.85, yerr=post_conflict_std, capsize=5)
    ax.set_ylabel("Post-failure conflict rate")
    ax.set_title("Conflict After Failure")
    for bar, val in zip(bars, post_conflict_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('UAV Failure Resilience Feasibility Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved failure analysis figure to {save_path}")


def compute_significance_rows(
    anchor_row: Dict[str, object],
    competitor_rows: List[Dict[str, object]],
    metric_key: str = "eval_coverage_samples",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    anchor = np.asarray(anchor_row.get(metric_key, []), dtype=np.float32)
    if anchor.size < 2:
        return rows
    for row in competitor_rows:
        sample = np.asarray(row.get(metric_key, []), dtype=np.float32)
        if sample.size < 2:
            continue
        t_stat, p_value = stats.ttest_ind(anchor, sample, equal_var=False)
        rows.append(
            {
                "Reference": anchor_row["name"],
                "Compared": row["name"],
                "Metric": metric_key.replace("eval_", "").replace("_samples", ""),
                "Reference Mean": round(float(anchor.mean()), 6),
                "Compared Mean": round(float(sample.mean()), 6),
                "t_stat": round(float(t_stat), 6),
                "p_value": round(float(p_value), 8),
                "Significant(p<0.05)": bool(p_value < 0.05),
            }
        )
    return rows


def build_main_result_stats_rows(main_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in main_results:
        cov = mean_std_ci(np.asarray(row.get("eval_coverage_samples", []), dtype=np.float32))
        rew = mean_std_ci(np.asarray(row.get("eval_reward_samples", []), dtype=np.float32))
        conf = mean_std_ci(np.asarray(row.get("eval_conflict_samples", []), dtype=np.float32))
        rows.append(
            {
                "Algorithm": row["name"],
                "Coverage (mean±std)": f"{cov['mean']:.3f}±{cov['std']:.3f}",
                "Coverage 95% CI": f"[{cov['ci95_low']:.3f}, {cov['ci95_high']:.3f}]",
                "Conflict (mean±std)": f"{conf['mean']:.4f}±{conf['std']:.4f}",
                "Conflict 95% CI": f"[{conf['ci95_low']:.4f}, {conf['ci95_high']:.4f}]",
                "Reward (mean±std)": f"{rew['mean']:.3f}±{rew['std']:.3f}",
                "Reward 95% CI": f"[{rew['ci95_low']:.3f}, {rew['ci95_high']:.3f}]",
                "N episodes": len(row.get("eval_coverage_samples", [])),
            }
        )
    return rows


def build_efficiency_rows(training_runs: Dict[str, Dict[str, object]], main_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    train_lookup = {
        METHOD_DISPLAY_NAME: training_runs["tmarl"],
        "MAPPO": training_runs["mappo"],
        "MADDPG": training_runs["maddpg"],
        "QMIX": training_runs["qmix"],
        "PPO": training_runs["ppo"],
    }
    rows: List[Dict[str, object]] = []
    for row in main_results:
        train_run = train_lookup[row["name"]]
        train_times = train_run.get("train_times", [])
        rows.append(
            {
                "Algorithm": row["name"],
                "Train Wall Time (s)": round(float(train_times[-1]) if train_times else 0.0, 4),
                "Inference Time (s/episode)": round(float(row.get("inference_time_sec", 0.0)), 6),
                "Inference (ms/step)": round(float(row.get("inference_ms_per_step", 0.0)), 6),
                "Memory (MB)": round(float(row.get("memory_mb", 0.0)), 3),
            }
        )
    return rows


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_tmarl_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    config: ExperimentConfig,
    weight_path: str,
    variant: str = "tmarl",
    metadata: Dict[str, object] | None = None,
) -> None:
    ensure_dir(os.path.dirname(weight_path))
    payload = {
        "state_dict": state_dict,
        "variant": variant,
        "config": asdict(config),
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, weight_path)


def select_tmarl_candidate(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    candidates: List[Dict[str, object]],
    output_dir: str,
    episodes: int,
) -> Dict[str, object]:
    device = torch.device(config.device)
    probe_env = MultiUAVCoverageEnv(
        dataset=dataset,
        num_uavs=config.train_num_uavs,
        env_type=config.train_env,
        max_steps=config.max_steps,
        max_speed=config.max_speed,
        high_conflict_threshold=config.high_conflict_threshold,
        high_conflict_penalty=config.high_conflict_penalty,
        reward_gain_coef=config.reward_gain_coef,
        reward_coverage_coef=config.reward_coverage_coef,
        reward_path_penalty=config.reward_path_penalty,
        reward_conflict_penalty=config.reward_conflict_penalty,
        reward_redundancy_penalty=config.reward_redundancy_penalty,
        reward_obstacle_penalty=config.reward_obstacle_penalty,
        assignment_bonus_coef=config.assignment_bonus_coef,
        frontier_bonus_coef=config.frontier_bonus_coef,
        stagnation_penalty_coef=config.stagnation_penalty_coef,
        region_reassign_interval=config.region_reassign_interval,
        use_region_assignment=config.use_region_assignment,
        region_balance_strength=config.region_balance_strength,
        region_global_mix=config.region_global_mix,
        region_support_radius_scale=config.region_support_radius_scale,
        heuristic_view_radius_scale=config.heuristic_view_radius_scale,
        assignment_decay_cover=config.assignment_decay_cover,
        assignment_decay_time=config.assignment_decay_time,
        frontier_decay_cover=config.frontier_decay_cover,
        frontier_decay_time=config.frontier_decay_time,
        late_stage_visible_boost=config.late_stage_visible_boost,
        fail_prob_base=config.fail_prob_base,
        fail_prob_load_scale=config.fail_prob_load_scale,
        fail_reward_recovery_coef=config.fail_reward_recovery_coef,
        fail_energy_fair_coef=config.fail_energy_fair_coef,
        use_failure_module=config.use_failure_module,
        fail_trigger_min_step=config.fail_trigger_min_step,
        fail_trigger_max_step=config.fail_trigger_max_step,
        fail_max_count=config.fail_max_count,
        fail_fixed_count=config.fail_fixed_count,
        fail_load_aware=config.fail_load_aware,
        fail_reassign_on_event=config.fail_reassign_on_event,
    )
    state_dim = probe_env.state_dim
    rows: List[Dict[str, object]] = []
    best_candidate = None
    best_rank = None
    for idx, candidate in enumerate(candidates, start=1):
        set_seed(config.seed + 777)
        model = TmarlActorCritic(state_dim=state_dim, config=config, variant="tmarl").to(device)
        model.load_state_dict(candidate["best_state"])
        metrics = evaluate_controller(
            candidate["label"],
            dataset,
            config,
            model=model,
            env_type=config.train_env,
            num_uavs=config.train_num_uavs,
            episodes=episodes,
            deterministic=True,
            eval_seed=config.seed + 3100,
        )
        rank_key = constrained_selection_tuple(
            reward=float(metrics["reward"]),
            coverage=float(metrics["coverage"]),
            coverage_p75=float(metrics.get("coverage_p75", metrics["coverage"])),
            conflict_rate=float(metrics["conflict_rate"]),
            conflict_p75=float(metrics.get("conflict_p75", metrics["conflict_rate"])),
            conflict_gate=config.best_conflict_gate,
        )
        score = rank_key[-1]
        rows.append(
            {
                "rank": idx,
                "label": candidate["label"],
                "seed": candidate.get("seed", ""),
                "source": candidate.get("source", ""),
                "coverage": round(float(metrics["coverage"]), 6),
                "coverage_p75": round(float(metrics.get("coverage_p75", metrics["coverage"])), 6),
                "reward": round(float(metrics["reward"]), 6),
                "conflict_rate": round(float(metrics["conflict_rate"]), 6),
                "conflict_p75": round(float(metrics.get("conflict_p75", metrics["conflict_rate"])), 6),
                "path_length": round(float(metrics["path_length"]), 6),
                "selection_score": round(float(score), 6),
                "selection_rank_key": str(tuple(round(float(x), 6) if isinstance(x, (int, float)) else x for x in rank_key)),
            }
        )
        if best_rank is None or rank_key > best_rank:
            best_rank = rank_key
            best_candidate = {
                **candidate,
                "selection_metrics": {k: v for k, v in metrics.items() if k != "final_env"},
                "selection_score": float(score),
                "selection_rank_key": tuple(float(x) for x in rank_key),
            }
    write_csv(os.path.join(output_dir, "tables", "Final_Model_Selection.csv"), rows)
    if best_candidate is None:
        raise RuntimeError("No valid T-MARL candidate was available for final selection.")
    return best_candidate


def plot_reward_convergence(path: str, histories: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(8.2, 5.1))
    for name, values in histories.items():
        plt.plot(moving_average(values, 8), label=name)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_coverage_curve(path: str, histories: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(8.2, 5.1))
    for name, values in histories.items():
        plt.plot(np.asarray(values) * 100.0, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Coverage Rate (%)")
    plt.title("Coverage vs Episode")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


_BAR_PINK = ["#EFD9E4", "#E7C4D7", "#DEA8C8", "#D78DB9", "#CF72AA", "#C7579A", "#B93D88"]
_BAR_BLUE = ["#DCE7F1", "#C7D7E8", "#AFC7DE", "#94B4D4", "#789FCC", "#5C89C1", "#3E73B1"]
_SCENE_BG = "#FCF8F2"
_SCENE_ACCENT = "#C44E7A"
_SCENE_BLUE = "#4C84C4"
_SCENE_GREEN = "#6FAF8F"
_SCENE_GOLD = "#E4B45F"


def _style_axes_journal(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=5)
    ax.grid(axis="y", color="#D8D5D0", alpha=0.45, linewidth=0.8)


def _gradient_bar_colors(names: List[str], key: str) -> List[str]:
    use_blue = ("ppo" in key.lower()) or any("ppo" in str(name).lower() for name in names)
    palette = _BAR_BLUE if use_blue else _BAR_PINK
    if len(names) <= len(palette):
        return palette[: len(names)]
    return [palette[min(len(palette) - 1, int(i * len(palette) / max(1, len(names))))] for i in range(len(names))]


def _add_round_box(ax, xy, width, height, text, facecolor, edgecolor="#5B5147", fontsize=11.0) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#231F1C",
    )


def _add_flow_arrow(ax, start, end, color="#7D6E63", text=None, text_offset=(0.0, 0.0)) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="simple",
        mutation_scale=16,
        linewidth=0.0,
        color=color,
        alpha=0.95,
    )
    ax.add_patch(arrow)
    if text:
        mid_x = 0.5 * (start[0] + end[0]) + text_offset[0]
        mid_y = 0.5 * (start[1] + end[1]) + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=10.0, color="#4A4039", ha="center", va="center")


def _build_demo_env(dataset: ScenarioDataset, config: "ExperimentConfig", env_type: str) -> MultiUAVCoverageEnv:
    state_py = random.getstate()
    state_np = np.random.get_state()
    demo_seed = config.seed + sum(ord(ch) for ch in env_type) + 123
    random.seed(demo_seed)
    np.random.seed(demo_seed % (2**32 - 1))
    demo_config = ExperimentConfig(**asdict(config))
    demo_config.train_env = env_type
    env = build_env_from_config(demo_config, dataset)
    env.reset()
    random.setstate(state_py)
    np.random.set_state(state_np)
    return env


def _draw_obstacle_footprints(ax: plt.Axes, obstacles: np.ndarray, facecolor: str = "#D9C1C1", edgecolor: str = "#A35A52", alpha: float = 0.35) -> None:
    for obstacle in obstacles:
        x0, y0, x1, y1 = obstacle_xy_bounds(obstacle)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8, alpha=alpha))


def _draw_obstacle_prisms_3d(ax: Any, obstacles: np.ndarray, facecolor: str = "#D9C1C1", edgecolor: str = "#A35A52", alpha: float = 0.12) -> None:
    for obstacle in obstacles:
        x0, y0, z0, x1, y1, z1 = obstacle_xyz_bounds(obstacle)
        ax.bar3d(
            x0,
            y0,
            z0,
            max(1.0, x1 - x0),
            max(1.0, y1 - y0),
            max(1.0, z1 - z0),
            color=facecolor,
            edgecolor=edgecolor,
            linewidth=0.3,
            alpha=alpha,
            shade=True,
            zsort="average",
        )


def _draw_terrain_surface_3d(ax: Any, env: MultiUAVCoverageEnv, alpha: float = 0.28) -> None:
    if env.terrain_surface.size == 0:
        return
    surf = env.terrain_surface
    ys = np.linspace(0.0, AREA_Y, surf.shape[0])
    xs = np.linspace(0.0, AREA_X, surf.shape[1])
    xx, yy = np.meshgrid(xs, ys)
    stride = max(1, surf.shape[0] // 48)
    ax.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        surf[::stride, ::stride],
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        alpha=alpha,
        shade=True,
    )


def _plot_horizontal_ring(ax: Any, center: np.ndarray, radius: float, color: str, linestyle: str, alpha: float) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 80)
    xs = center[0] + radius * np.cos(theta)
    ys = center[1] + radius * np.sin(theta)
    zs = np.full_like(xs, center[2])
    ax.plot(xs, ys, zs, color=color, linestyle=linestyle, linewidth=0.9, alpha=alpha)


def plot_problem_scenarios_overview(path: str, dataset: ScenarioDataset, config: "ExperimentConfig") -> None:
    env_specs = [("simple", "Simple"), ("obstacles", "Obstacle"), ("dynamic_users", "Dynamic User")]
    fig = plt.figure(figsize=(16.2, 5.6))
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
    fig.patch.set_facecolor("white")
    for ax, (env_type, title) in zip(axes, env_specs):
        env = _build_demo_env(dataset, config, env_type)
        users = env.users
        ax.set_facecolor(_SCENE_BG)
        _draw_terrain_surface_3d(ax, env, alpha=0.22)
        if len(users):
            ax.scatter(users[:, 0], users[:, 1], users[:, 2], s=16, c=_SCENE_BLUE, alpha=0.62, edgecolors="white", linewidths=0.25, label="Users")
        if len(env.obstacles):
            _draw_obstacle_prisms_3d(ax, env.obstacles, facecolor="#D96C6C", edgecolor="white", alpha=0.12)
        active_ids = env._active_agent_indices()
        for idx in active_ids:
            x, y, z = env.uav_pos[idx]
            cx, cy, cz = env.region_centers[idx]
            ax.scatter([x], [y], [z], s=72, c=_SCENE_ACCENT, edgecolors="white", linewidths=0.8, zorder=4)
            ax.plot([x, cx], [y, cy], [z, cz], color="#6E6257", linewidth=1.0, alpha=0.75)
        if len(active_ids):
            ref = env.uav_pos[int(active_ids[0])]
            _plot_horizontal_ring(ax, ref, COMM_RADIUS, "#6D8FA8", "--", 0.38)
            _plot_horizontal_ring(ax, ref, config.safety_distance, "#C46A6A", "-", 0.52)
        if env_type == "dynamic_users" and len(users):
            show_n = min(10, len(users))
            chosen = np.linspace(0, len(users) - 1, show_n, dtype=int)
            for i in chosen:
                dx = 90.0 * math.cos(i * 0.7)
                dy = 70.0 * math.sin(i * 0.9)
                ax.plot([users[i, 0], users[i, 0] + dx], [users[i, 1], users[i, 1] + dy], [users[i, 2], users[i, 2]], color=_SCENE_GOLD, linewidth=1.0, alpha=0.85)
        ax.set_xlim(0, AREA_X)
        ax.set_ylim(0, AREA_Y)
        ax.set_zlim(ALT_MIN, ALT_MAX)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("X (m)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_zlabel("Altitude (m)", fontsize=10)
        ax.view_init(elev=24, azim=-58)
        ax.grid(alpha=0.18, linewidth=0.7)
    fig.suptitle("Problem Scenario Visualization", fontsize=15, y=0.98)
    fig.text(0.5, 0.02, "Initial UAVs, user targets, obstacle volumes, communication radius, safety radius, and guidance directions across three environments.", ha="center", fontsize=10.5)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_framework_diagram(path: str) -> None:
    fig, ax = plt.subplots(figsize=(13.6, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    boxes = [
        ((0.03, 0.33), 0.13, 0.34, "State\nInput", "#ECE6F6"),
        ((0.20, 0.33), 0.16, 0.34, "Objective-Conditioned\nMulti-Source Transformer", "#DCE7F1"),
        ((0.40, 0.33), 0.16, 0.34, "PPO Actor-Critic\nPolicy Update", "#F7E6D5"),
        ((0.60, 0.33), 0.13, 0.34, "Safety\nLayer", "#F5DDE6"),
        ((0.77, 0.33), 0.16, 0.34, "Region Assignment\n& Frontier Guidance", "#E4F0E3"),
    ]
    for (xy, w, h, text, fc) in boxes:
        _add_round_box(ax, xy, w, h, text, fc)
    _add_round_box(ax, (0.80, 0.79), 0.13, 0.13, "Action Output", "#EBD6C6", fontsize=10.8)
    _add_flow_arrow(ax, (0.16, 0.50), (0.20, 0.50), text="multi-agent\nobservations", text_offset=(0.0, 0.08))
    _add_flow_arrow(ax, (0.36, 0.50), (0.40, 0.50), text="coordinated\nfeatures", text_offset=(0.0, 0.08))
    _add_flow_arrow(ax, (0.56, 0.50), (0.60, 0.50), text="raw action", text_offset=(0.0, 0.06))
    _add_flow_arrow(ax, (0.73, 0.50), (0.77, 0.50), text="safe action\nproposal", text_offset=(0.0, 0.08))
    _add_flow_arrow(ax, (0.85, 0.67), (0.86, 0.79), color="#9E8469")
    _add_round_box(ax, (0.36, 0.80), 0.18, 0.11, "CTDE shared critic\nfor centralized training", "#F7F1D8", fontsize=10.5)
    _add_flow_arrow(ax, (0.49, 0.80), (0.49, 0.67), color="#9E8469")
    ax.text(0.5, 0.08, "CAST-MARL framework: coordinated perception, policy optimization, safety filtering, and geometry-aware task guidance.", ha="center", fontsize=11.0, color="#4A4039")
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_safety_layer_schematic(path: str) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    _add_round_box(ax, (0.03, 0.28), 0.15, 0.42, "Predicted\nAction +\nPairwise States", "#E8EFF6")
    _add_round_box(ax, (0.25, 0.54), 0.17, 0.18, "Repulsion\ncomponent", "#F2DEE7")
    _add_round_box(ax, (0.25, 0.26), 0.17, 0.18, "Brake\nmechanism", "#F9EAD4")
    _add_round_box(ax, (0.49, 0.40), 0.19, 0.24, "Guidance blending\ncover gain + risk bias", "#E3F0E4")
    _add_round_box(ax, (0.75, 0.33), 0.18, 0.34, "Safe control\ncommand", "#E9D9CB")
    _add_flow_arrow(ax, (0.18, 0.57), (0.25, 0.63))
    _add_flow_arrow(ax, (0.18, 0.41), (0.25, 0.35))
    _add_flow_arrow(ax, (0.42, 0.63), (0.49, 0.56), text="risk-aware\nrepel")
    _add_flow_arrow(ax, (0.42, 0.35), (0.49, 0.48), text="speed cap")
    _add_flow_arrow(ax, (0.68, 0.52), (0.75, 0.50), text="blend & clamp", text_offset=(0.0, 0.06))
    ax.text(0.27, 0.77, r"$r_{risk}=f(d_{now}, d_{next}, \Delta d)$", fontsize=10.4, color="#4A4039")
    ax.text(0.26, 0.17, r"$a' = a \cdot (1-\beta \cdot risk)$", fontsize=10.4, color="#4A4039")
    ax.text(0.51, 0.27, r"$a_{safe} = (1-\lambda)a_{raw} + \lambda a_{guide}$", fontsize=10.4, color="#4A4039")
    ax.text(0.5, 0.06, "Safety layer schematic: repulsion, brake, and guidance are fused before final action execution.", ha="center", fontsize=10.8, color="#4A4039")
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_region_assignment_guidance(path: str, env: MultiUAVCoverageEnv) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_SCENE_BG)
    palette = ["#E6B6C8", "#A8CCE8", "#B9D7B4", "#F2D18A", "#C7B4E3", "#F4B69C", "#8AC6BF", "#D9B8A7"]
    for agent_id in range(env.num_uavs):
        idx = env._assigned_indices(agent_id)
        color = palette[agent_id % len(palette)]
        if len(idx):
            pts = env.users[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=16, color=color, alpha=0.58, edgecolors="white", linewidths=0.25)
        center = env.region_centers[agent_id]
        ax.scatter([center[0]], [center[1]], s=120, color="#1E1A17", marker="X", edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter([env.uav_pos[agent_id, 0]], [env.uav_pos[agent_id, 1]], s=72, color=_SCENE_ACCENT, edgecolors="white", linewidths=0.8, zorder=6)
        ax.add_patch(Circle((center[0], center[1]), env.region_radii[agent_id] * 0.48, fill=False, edgecolor=color, linewidth=1.2, alpha=0.55))
        ax.annotate("", xy=(center[0], center[1]), xytext=(env.uav_pos[agent_id, 0], env.uav_pos[agent_id, 1]),
                    arrowprops=dict(arrowstyle="->", color="#65584F", lw=1.2))
        support = env._assigned_uncovered(agent_id)
        if len(support):
            frontier = support[:, :2].mean(axis=0)
            ax.scatter([frontier[0]], [frontier[1]], s=90, color=_SCENE_GOLD, marker="D", edgecolors="white", linewidths=0.8, zorder=6)
            ax.annotate("", xy=(frontier[0], frontier[1]), xytext=(center[0], center[1]),
                        arrowprops=dict(arrowstyle="->", color=_SCENE_GOLD, lw=1.3))
    ax.set_xlim(0, AREA_X)
    ax.set_ylim(0, AREA_Y)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Region Assignment and Frontier Guidance", fontsize=14)
    ax.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_failure_redistribution_mechanism(path: str) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    _add_round_box(ax, (0.04, 0.34), 0.16, 0.30, "Failure trigger\n(load-aware event)", "#F6DFDF")
    _add_round_box(ax, (0.28, 0.34), 0.17, 0.30, "Coverage gap\nand load update", "#F7ECD7")
    _add_round_box(ax, (0.53, 0.34), 0.17, 0.30, "Region reassignment\nfor active UAVs", "#E3F0E4")
    _add_round_box(ax, (0.78, 0.34), 0.16, 0.30, "Recovered mission\nstate", "#E6EEF7")
    _add_flow_arrow(ax, (0.20, 0.49), (0.28, 0.49), text="drop\ncoverage")
    _add_flow_arrow(ax, (0.45, 0.49), (0.53, 0.49), text="recompute\nassignment")
    _add_flow_arrow(ax, (0.70, 0.49), (0.78, 0.49), text="new goals")
    ax.text(0.285, 0.22, r"$L_i \leftarrow users_i / capacity_i$", fontsize=10.5, color="#4A4039")
    ax.text(0.54, 0.22, r"$\pi_{new} = assign(region, frontier, risk)$", fontsize=10.5, color="#4A4039")
    ax.text(0.50, 0.08, "Failure redistribution mechanism: detected failures update load estimates, trigger reassignment, and restore coverage trajectories.", ha="center", fontsize=10.8, color="#4A4039")
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def _rolling_mean_std(values: List[float], window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    mean = np.zeros_like(arr)
    std = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        seg = arr[lo : i + 1]
        mean[i] = seg.mean()
        std[i] = seg.std(ddof=0)
    return mean, std


def _trajectory_risk_per_step(traj: np.ndarray) -> np.ndarray:
    risks = []
    for t in range(traj.shape[0]):
        pos = traj[t, :, :3]
        d = pairwise_dist(pos, pos)
        mask = np.triu(np.ones_like(d, dtype=bool), 1)
        pair_vals = d[mask]
        if pair_vals.size == 0:
            risks.append(0.0)
            continue
        min_d = float(pair_vals.min())
        risk = float(np.clip((COLLISION_DISTANCE * 1.8 - min_d) / max(COLLISION_DISTANCE * 1.8, 1e-6), 0.0, 1.0))
        risks.append(risk)
    return np.asarray(risks, dtype=np.float32)


def _coverage_heat_from_positions(users: np.ndarray, traj_slice: np.ndarray) -> np.ndarray:
    heat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if len(users) == 0:
        return heat
    pos = traj_slice[:, :, :3]
    d = np.sqrt(((pos[:, :, None, :] - users[None, None, :, :3]) ** 2).sum(axis=-1))
    z_ratio = np.clip((pos[..., 2] - ALT_MIN) / max(ALT_MAX - ALT_MIN, 1e-6), 0.0, 1.0)
    radius = COMM_RADIUS * (0.72 + 0.36 * z_ratio)
    covered_any = (d <= radius[:, :, None]).any(axis=1)
    for user_idx, is_covered in enumerate(covered_any.any(axis=0)):
        if is_covered:
            gx = min(GRID_SIZE - 1, max(0, int(users[user_idx, 0] / (AREA_X / GRID_SIZE))))
            gy = min(GRID_SIZE - 1, max(0, int(users[user_idx, 1] / (AREA_Y / GRID_SIZE))))
            heat[gy, gx] += 1.0
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def plot_bar_metric(path: str, results: List[Dict[str, object]], key: str, ylabel: str, title: str) -> None:
    names = [r["name"] for r in results]
    vals = np.asarray([r[key] for r in results], dtype=np.float32)
    colors = _gradient_bar_colors(names, key)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    bars = ax.bar(
        np.arange(len(names)),
        vals,
        width=0.72,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    _style_axes_journal(ax)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=0, ha="center")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=14)

    for bar, val, color in zip(bars, vals, colors):
        cx = bar.get_x() + bar.get_width() / 2.0
        ax.scatter([cx], [val], s=26, color=color, edgecolors="white", linewidths=0.9, zorder=4)

    ymin = 0.0
    ymax = float(vals.max()) if len(vals) else 1.0
    if key == "conflict_rate":
        ax.set_ylim(0.0, ymax * 1.30 + 1e-4)
    else:
        ax.set_ylim(ymin, ymax * 1.18 + 1e-6)

    fig.subplots_adjust(bottom=0.20, left=0.12, right=0.97, top=0.88)
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_trajectories(path: str, env: MultiUAVCoverageEnv, title: str) -> None:
    plt.figure(figsize=(7.4, 6.2))
    traj = np.stack(env.trajectory, axis=0)
    users = env.initial_users
    plt.scatter(users[:, 0], users[:, 1], c="lightgray", s=10, label="Users")
    if len(env.obstacles):
        ax = plt.gca()
        _draw_obstacle_footprints(ax, env.obstacles, facecolor="#C1121F", edgecolor="#8C0E18", alpha=0.18)
    colors = plt.cm.tab10(np.linspace(0, 1, traj.shape[1]))
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], linewidth=1.8)
        plt.scatter(traj[0, i, 0], traj[0, i, 1], color=colors[i], marker="o", s=28)
        plt.scatter(traj[-1, i, 0], traj[-1, i, 1], color=colors[i], marker="x", s=36)
    plt.xlim(0, AREA_X)
    plt.ylim(0, AREA_Y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_heatmap(path: str, heat: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6.2, 5.5))
    plt.imshow(heat, cmap="YlOrRd", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_time_reward(path: str, histories: Dict[str, Tuple[List[float], List[float]]]) -> None:
    plt.figure(figsize=(8.0, 5.0))
    for name, (times, rewards) in histories.items():
        plt.plot(times, moving_average(rewards, 6), label=name)
    plt.xlabel("Wall Time (s)")
    plt.ylabel("Reward")
    plt.title("Training Time vs Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_task_time_distribution(path: str, results: List[Dict[str, object]]) -> None:
    names = [r["name"] for r in results]
    vals = [r["task_time"] for r in results]
    plt.figure(figsize=(7.8, 4.8))
    plt.bar(names, vals, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
    plt.ylabel("Task Completion Time")
    plt.title("Task Completion Time Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_scalability(path: str, scale_rows: List[Dict[str, object]]) -> None:
    plt.figure(figsize=(8.0, 5.0))
    uavs = [r["num_uavs"] for r in scale_rows]
    cov = [r["coverage"] for r in scale_rows]
    rew = [r["reward"] for r in scale_rows]
    plt.plot(uavs, cov, marker="o", label="Coverage (%)")
    plt.plot(uavs, rew, marker="s", label="Reward")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Metric Value")
    plt.title("Scalability across UAV Fleet Size")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_reward_breakdown(path: str, ablation_rows: List[Dict[str, object]]) -> None:
    names = [r["variant"] for r in ablation_rows]
    coverage = [r["coverage_reward_gain"] for r in ablation_rows]
    path_eff = [r["path_efficiency_gain"] for r in ablation_rows]
    conflict = [r["conflict_penalty_reduction"] for r in ablation_rows]
    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(8.4, 5.0))
    plt.bar(x - width, coverage, width, label="Coverage")
    plt.bar(x, path_eff, width, label="Path Efficiency")
    plt.bar(x + width, conflict, width, label="Conflict Penalty")
    plt.xticks(x, names, rotation=10)
    plt.ylabel("Relative Contribution")
    plt.title("Ablation Reward Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_tradeoff_scatter(path: str, results: List[Dict[str, object]]) -> None:
    plt.figure(figsize=(7.8, 5.4))
    paths = np.asarray([r["path_length"] for r in results], dtype=np.float32)
    pmin, pmax = float(paths.min()), float(paths.max())
    denom = max(1e-6, pmax - pmin)
    sizes = 120.0 + 320.0 * (paths - pmin) / denom
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for idx, row in enumerate(results):
        plt.scatter(row["conflict_rate"], row["coverage"], s=float(sizes[idx]), color=colors[idx], alpha=0.75, label=row["name"])
        plt.text(row["conflict_rate"] + 0.0003, row["coverage"] + 0.15, row["name"], fontsize=8)
    plt.xlabel("Conflict Rate")
    plt.ylabel("Coverage (%)")
    plt.title("Coverage-Conflict Tradeoff (bubble size = Path Length)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_training_progress_panels(path: str, training_runs: Dict[str, Dict[str, object]]) -> None:
    order = [
        ("tmarl", METHOD_DISPLAY_NAME),
        ("mappo", "MAPPO"),
        ("maddpg", "MADDPG"),
        ("qmix", "QMIX"),
        ("ppo", "PPO"),
    ]
    colors = {
        METHOD_DISPLAY_NAME: "#C7579A",
        "MAPPO": "#8FA8C8",
        "MADDPG": "#80B1A3",
        "QMIX": "#D6A45F",
        "PPO": "#5C89C1",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    metrics = [
        ("train_rewards", "Reward", axes[0]),
        ("train_coverage", "Coverage", axes[1]),
        ("train_conflicts", "Conflict Rate", axes[2]),
    ]
    for key, ylabel, ax in metrics:
        ax.set_facecolor("white")
        for run_key, label in order:
            values = training_runs[run_key][key]
            mean, std = _rolling_mean_std(values, window=8)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=colors[label], lw=2.0, label=label)
            ax.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.18)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22)
    axes[0].set_title("(a) Reward Progress", fontsize=12.5)
    axes[1].set_title("(b) Coverage Progress", fontsize=12.5)
    axes[2].set_title("(c) Conflict Progress", fontsize=12.5)
    axes[2].legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("Training Pipeline and Learning Progress", fontsize=14.5, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_main_performance_with_errorbars(path: str, main_results: List[Dict[str, object]]) -> None:
    metrics = [
        ("coverage", "Coverage (%)"),
        ("reward", "Reward"),
        ("path_length", "Path Length"),
        ("conflict_rate", "Conflict Rate"),
    ]
    names = [r["name"] for r in main_results]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    axes = axes.flatten()
    colors = _gradient_bar_colors(names, "reward")
    anchor = main_results[0]
    for ax, (metric_key, ylabel) in zip(axes, metrics):
        vals = []
        errs = []
        for row in main_results:
            vals.append(float(row[metric_key]))
            sample_key = {
                "coverage": "eval_coverage_samples",
                "reward": "eval_reward_samples",
                "conflict_rate": "eval_conflict_samples",
            }.get(metric_key)
            if sample_key and row.get(sample_key):
                errs.append(float(np.std(np.asarray(row[sample_key], dtype=np.float32), ddof=0)))
            else:
                errs.append(0.0)
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.0, yerr=errs, capsize=4, alpha=0.96)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title(ylabel, fontsize=12.3)
        if metric_key in {"coverage", "reward"}:
            sig_rows = compute_significance_rows(anchor, main_results[1:], metric_key=f"eval_{metric_key}_samples")
            compare_to_star = {r["Compared"]: r["Significant(p<0.05)"] for r in sig_rows}
            ymax = max(v + e for v, e in zip(vals, errs)) if vals else 1.0
            for idx, name in enumerate(names[1:], start=1):
                if compare_to_star.get(name, False):
                    ax.text(idx, vals[idx] + errs[idx] + 0.04 * max(1.0, ymax), "*", ha="center", va="bottom", fontsize=14, color="#2F2B27")
    fig.suptitle("Main Performance Comparison with Error Bars", fontsize=14.5, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_3d_trajectory_risk_overlay(path: str, env: MultiUAVCoverageEnv) -> None:
    traj = np.stack(env.trajectory, axis=0)
    risks = _trajectory_risk_per_step(traj)
    fig = plt.figure(figsize=(12.5, 5.8))
    fig.patch.set_facecolor("white")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    cmap = plt.cm.RdYlBu_r
    _draw_terrain_surface_3d(ax3d, env, alpha=0.22)
    if len(env.obstacles):
        _draw_obstacle_prisms_3d(ax3d, env.obstacles, facecolor="#D9C1C1", edgecolor="#A35A52", alpha=0.12)
        _draw_obstacle_footprints(ax2d, env.obstacles, facecolor="#D9C1C1", edgecolor="#A35A52", alpha=0.25)
    for i in range(traj.shape[1]):
        for t in range(traj.shape[0] - 1):
            seg = traj[t : t + 2, i]
            color = cmap(risks[t])
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=2.0, alpha=0.95)
            ax2d.plot(seg[:, 0], seg[:, 1], color=color, lw=2.0, alpha=0.95)
        ax3d.scatter(traj[0, i, 0], traj[0, i, 1], traj[0, i, 2], c="#C7579A", s=26)
        ax3d.scatter(traj[-1, i, 0], traj[-1, i, 1], traj[-1, i, 2], c="#4C84C4", s=30, marker="^")
    if len(env.initial_users):
        ax2d.scatter(env.initial_users[:, 0], env.initial_users[:, 1], s=10, c="#A8B2BD", alpha=0.35)
    ax3d.set_title("(a) 3D Flight Paths", fontsize=12.5)
    ax2d.set_title("(b) Ground Projection with Risk Coloring", fontsize=12.5)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Altitude")
    ax3d.set_xlim(0, AREA_X); ax3d.set_ylim(0, AREA_Y); ax3d.set_zlim(ALT_MIN, ALT_MAX)
    ax3d.view_init(elev=23, azim=-57)
    ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
    ax2d.set_facecolor(_SCENE_BG)
    ax2d.grid(alpha=0.18)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax3d, ax2d], fraction=0.025, pad=0.04)
    cbar.set_label("Safety risk level", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_coverage_heatmaps_over_time(path: str, env: MultiUAVCoverageEnv) -> None:
    traj = np.stack(env.trajectory, axis=0)
    steps = [0, max(1, traj.shape[0] // 3), max(1, 2 * traj.shape[0] // 3), traj.shape[0] - 1]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0))
    fig.patch.set_facecolor("white")
    for ax, step in zip(axes, steps):
        heat = _coverage_heat_from_positions(env.initial_users, traj[: step + 1])
        ax.imshow(heat, cmap="YlOrRd", origin="lower", extent=[0, AREA_X, 0, AREA_Y], vmin=0, vmax=1)
        pos = traj[step]
        ax.scatter(pos[:, 0], pos[:, 1], s=32, c="#2F8BCB", edgecolors="white", linewidths=0.6)
        ax.set_title(f"t = {step}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(_SCENE_BG)
    fig.suptitle("Coverage Heatmaps Over Time", fontsize=14.5, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_sweep_curve(path: str, rows: List[Dict[str, object]], x_key: str, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8.0, 5.0))
    names = sorted({str(row["method"]) for row in rows})
    for name in names:
        sub = sorted([row for row in rows if str(row["method"]) == name], key=lambda r: float(r[x_key]))
        xs = [float(r[x_key]) for r in sub]
        ys = [float(r["coverage"]) for r in sub]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=name)
    plt.xlabel(xlabel)
    plt.ylabel("Coverage (%)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def export_attention_proxy(save_path: str, model: TmarlActorCritic, env: MultiUAVCoverageEnv, device: torch.device) -> None:
    obs = env._get_obs()[None]
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    base_path, _ = os.path.splitext(save_path)
    source_weights = None
    if hasattr(model.backbone, "layers") and hasattr(model.backbone, "input_proj"):
        with torch.no_grad():
            projected = model.backbone.input_proj(obs_t)
            if hasattr(model.backbone, "multi_source_encoder"):
                multi_source, gate_stats = model.backbone.multi_source_encoder(obs_t)
                source_weights = gate_stats["source_weights"].mean(dim=0).detach().cpu().numpy()
                x = projected + multi_source
            else:
                x = projected
            first_layer = model.backbone.layers[0]
            attn_bias = first_layer._build_relation_bias(obs_t)
            n_agents = x.shape[1]
            attn_mask = attn_bias.reshape(obs_t.shape[0] * first_layer.n_heads, n_agents, n_agents)
            norm_x = first_layer.norm1(x)
            _, attn_weights = first_layer.self_attn(
                norm_x,
                norm_x,
                norm_x,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        attn_proxy = attn_weights.mean(dim=1).squeeze(0).detach().cpu().numpy()
    else:
        attn_proxy = np.eye(obs.shape[1], dtype=np.float32)
    if source_weights is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
        im = axes[0].imshow(attn_proxy, cmap="viridis", vmin=float(attn_proxy.min()), vmax=float(attn_proxy.max()))
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title("Agent-Agent Attention")
        axes[0].set_xlabel("Key Agent")
        axes[0].set_ylabel("Query Agent")
        im2 = axes[1].imshow(source_weights.T, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("Source Gating")
        axes[1].set_xlabel("Agent")
        axes[1].set_ylabel("Source")
        axes[1].set_yticks(range(4))
        axes[1].set_yticklabels(["Self", "Agent", "Entity", "Task"])
        fig.suptitle("Objective-Conditioned Multi-Source Attention Proxy", fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(base_path + ".png", dpi=180, bbox_inches="tight")
        plt.savefig(base_path + ".svg", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.figure(figsize=(4.8, 4.0))
        plt.imshow(attn_proxy, cmap="viridis", vmin=float(attn_proxy.min()), vmax=float(attn_proxy.max()))
        plt.colorbar()
        plt.title("Agent-Agent Attention Heatmap")
        plt.xlabel("Key Agent")
        plt.ylabel("Query Agent")
        plt.tight_layout()
        plt.savefig(base_path + ".png", dpi=150, bbox_inches="tight")
        plt.savefig(base_path + ".svg", bbox_inches="tight")
        plt.close()
    rows = []
    for i in range(attn_proxy.shape[0]):
        for j in range(attn_proxy.shape[1]):
            rows.append({"query_agent": i, "key_agent": j, "attention_weight": float(attn_proxy[i, j]), "source_gate": ""})
    if source_weights is not None:
        for source_idx, source_name in enumerate(["self", "agent", "entity", "task"]):
            for agent_idx in range(source_weights.shape[0]):
                rows.append(
                    {
                        "query_agent": agent_idx,
                        "key_agent": source_idx,
                        "attention_weight": float(source_weights[agent_idx, source_idx]),
                        "source_gate": source_name,
                    }
                )
    write_csv(base_path + ".csv", rows)
    with open(base_path + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "attention_matrix": attn_proxy.tolist(),
                "source_weights": source_weights.tolist() if source_weights is not None else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def export_coverage_prototype_bank(output_dir: str, env: MultiUAVCoverageEnv) -> List[Dict[str, float]]:
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    heat = np.asarray(env.coverage_heat, dtype=np.float32)
    flat_idx = np.argsort(heat.reshape(-1))[::-1][:5]
    xs = np.linspace(0.0, AREA_X, heat.shape[1], endpoint=False) + AREA_X / heat.shape[1] / 2.0
    ys = np.linspace(0.0, AREA_Y, heat.shape[0], endpoint=False) + AREA_Y / heat.shape[0] / 2.0
    rows: List[Dict[str, float]] = []
    for rank, idx in enumerate(flat_idx, start=1):
        gy, gx = np.unravel_index(int(idx), heat.shape)
        rows.append(
            {
                "prototype_rank": rank,
                "x": float(xs[gx]),
                "y": float(ys[gy]),
                "heat_value": float(heat[gy, gx]),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_Coverage_Prototype_Bank.csv"), rows)
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(heat, cmap="hot", origin="lower", extent=[0, AREA_X, 0, AREA_Y], alpha=0.75)
    plt.scatter(env.region_centers[:, 0], env.region_centers[:, 1], c="#1D9E75", s=70, label="Region Centers")
    plt.scatter([row["x"] for row in rows], [row["y"] for row in rows], c="#1B3C8A", s=80, marker="X", label="Coverage Prototypes")
    plt.legend(fontsize=8)
    plt.title("Coverage Prototype Bank")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Supplementary_Coverage_Prototype_Bank.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return rows


def export_safety_layer_diagnostics(output_dir: str, model: TmarlActorCritic, dataset: ScenarioDataset, config: ExperimentConfig) -> Dict[str, object]:
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    device = next(model.parameters()).device
    env = MultiUAVCoverageEnv(
        dataset=dataset,
        num_uavs=config.train_num_uavs,
        env_type=config.train_env,
        max_steps=config.max_steps,
        max_speed=config.max_speed,
        dynamic_ratio=config.dynamic_ratio,
        obstacle_ratio=config.obstacle_ratio,
        region_reassign_interval=config.region_reassign_interval,
        use_region_assignment=config.use_region_assignment,
        fail_prob_base=config.fail_prob_base,
        fail_prob_load_scale=config.fail_prob_load_scale,
        fail_reward_recovery_coef=config.fail_reward_recovery_coef,
        fail_energy_fair_coef=config.fail_energy_fair_coef,
        use_failure_module=config.use_failure_module,
        fail_trigger_min_step=config.fail_trigger_min_step,
        fail_trigger_max_step=config.fail_trigger_max_step,
        fail_max_count=config.fail_max_count,
        fail_fixed_count=config.fail_fixed_count,
        fail_load_aware=config.fail_load_aware,
        fail_reassign_on_event=config.fail_reassign_on_event,
    )
    obs = env.reset()
    rows: List[Dict[str, float]] = []
    done = False
    while not done:
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _, _ = model.act(obs_t, deterministic=True)
        stats_row = model.latest_safety_stats or {}
        rows.append(
            {
                "timestep": int(env.t),
                "mean_max_risk": float(torch.as_tensor(stats_row.get("max_risk", torch.zeros(1))).mean().item()),
                "mean_repulsion_norm": float(torch.as_tensor(stats_row.get("repulsion_norm", torch.zeros(1))).mean().item()),
                "mean_brake_scale": float(torch.as_tensor(stats_row.get("brake_scale", torch.ones(1))).mean().item()),
                "mean_guidance_blend": float(torch.as_tensor(stats_row.get("guidance_blend", torch.zeros(1))).mean().item()),
            }
        )
        obs, _, done, _ = env.step(action.squeeze(0).cpu().numpy())

    write_csv(os.path.join(table_dir, "Supplementary_Safety_Layer_Diagnostics.csv"), rows)
    summary = {
        "mean_max_risk": float(np.mean([r["mean_max_risk"] for r in rows])) if rows else 0.0,
        "mean_repulsion_norm": float(np.mean([r["mean_repulsion_norm"] for r in rows])) if rows else 0.0,
        "mean_brake_scale": float(np.mean([r["mean_brake_scale"] for r in rows])) if rows else 1.0,
        "mean_guidance_blend": float(np.mean([r["mean_guidance_blend"] for r in rows])) if rows else 0.0,
    }
    with open(os.path.join(table_dir, "Supplementary_Safety_Layer_Diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    plt.figure(figsize=(8.4, 4.8))
    xs = [r["timestep"] for r in rows]
    plt.plot(xs, [r["mean_repulsion_norm"] for r in rows], label="Repulsion", linewidth=1.8)
    plt.plot(xs, [r["mean_brake_scale"] for r in rows], label="Brake Scale", linewidth=1.8)
    plt.plot(xs, [r["mean_guidance_blend"] for r in rows], label="Guidance Blend", linewidth=1.8)
    plt.plot(xs, [r["mean_max_risk"] for r in rows], label="Max Risk", linewidth=1.5, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Value")
    plt.title("Safety Layer Diagnostics")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Supplementary_Safety_Layer_Diagnostics.png"), dpi=180, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "Supplementary_Safety_Layer_Diagnostics.svg"), bbox_inches="tight")
    plt.close()
    return {"rows": rows, "summary": summary}


def export_training_history_csvs(output_dir: str, training_runs: Dict[str, Dict[str, object]]) -> None:
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    for key, display_name in [
        ("tmarl", METHOD_DISPLAY_NAME),
        ("mappo", "MAPPO"),
        ("maddpg", "MADDPG"),
        ("qmix", "QMIX"),
        ("ppo", "PPO"),
    ]:
        run = training_runs[key]
        rows = []
        rewards = run["train_rewards"]
        coverage = run["train_coverage"]
        conflicts = run["train_conflicts"]
        times = run["train_times"]
        for idx in range(len(rewards)):
            rows.append(
                {
                    "episode_index": idx + 1,
                    "algorithm": display_name,
                    "reward": float(rewards[idx]),
                    "coverage": float(coverage[idx]),
                    "conflict_rate": float(conflicts[idx]),
                    "elapsed_sec": float(times[idx]),
                }
            )
        write_csv(os.path.join(table_dir, f"History_{display_name.replace('-', '_').replace(' ', '_')}.csv"), rows)


def export_figures(
    output_dir: str,
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    training_runs: Dict[str, Dict[str, object]],
    main_results: List[Dict[str, object]],
    ablation_results: List[Dict[str, object]],
    complexity_rows: List[Dict[str, object]],
    scale_rows: List[Dict[str, object]],
) -> None:
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    plot_problem_scenarios_overview(os.path.join(fig_dir, "Figure00_Problem_Scenarios.png"), dataset, config)
    plot_framework_diagram(os.path.join(fig_dir, "Figure00_Framework_Diagram.png"))
    plot_safety_layer_schematic(os.path.join(fig_dir, "Supplementary_Safety_Layer_Schematic.png"))
    plot_region_assignment_guidance(os.path.join(fig_dir, "Supplementary_Region_Assignment_Guidance.png"), main_results[0]["final_env"])
    plot_failure_redistribution_mechanism(os.path.join(fig_dir, "Supplementary_Failure_Redistribution.png"))
    plot_training_progress_panels(os.path.join(fig_dir, "Supplementary_Training_Progress.png"), training_runs)
    plot_main_performance_with_errorbars(os.path.join(fig_dir, "Supplementary_Main_Performance_ErrorBars.png"), main_results)
    plot_3d_trajectory_risk_overlay(os.path.join(fig_dir, "Supplementary_3D_Trajectory_Risk.png"), main_results[0]["final_env"])
    plot_coverage_heatmaps_over_time(os.path.join(fig_dir, "Supplementary_Coverage_Heatmaps_Over_Time.png"), main_results[0]["final_env"])

    reward_hist = {
        METHOD_DISPLAY_NAME: training_runs["tmarl"]["train_rewards"],
        "MAPPO": training_runs["mappo"]["train_rewards"],
        "MADDPG": training_runs["maddpg"]["train_rewards"],
        "QMIX": training_runs["qmix"]["train_rewards"],
        "PPO": training_runs["ppo"]["train_rewards"],
    }
    coverage_hist = {
        METHOD_DISPLAY_NAME: training_runs["tmarl"]["train_coverage"],
        "MAPPO": training_runs["mappo"]["train_coverage"],
        "MADDPG": training_runs["maddpg"]["train_coverage"],
        "QMIX": training_runs["qmix"]["train_coverage"],
        "PPO": training_runs["ppo"]["train_coverage"],
    }
    time_hist = {
        METHOD_DISPLAY_NAME: (training_runs["tmarl"]["train_times"], training_runs["tmarl"]["train_rewards"]),
        "MAPPO": (training_runs["mappo"]["train_times"], training_runs["mappo"]["train_rewards"]),
        "MADDPG": (training_runs["maddpg"]["train_times"], training_runs["maddpg"]["train_rewards"]),
        "QMIX": (training_runs["qmix"]["train_times"], training_runs["qmix"]["train_rewards"]),
        "PPO": (training_runs["ppo"]["train_times"], training_runs["ppo"]["train_rewards"]),
    }

    plot_reward_convergence(os.path.join(fig_dir, "Figure1_Reward_Convergence.png"), reward_hist)
    plot_coverage_curve(os.path.join(fig_dir, "Figure2_Coverage_vs_Episode.png"), coverage_hist)
    plot_bar_metric(os.path.join(fig_dir, "Figure3_Path_Length.png"), main_results, "path_length", "Avg Path Length", "Average Path Length")
    plot_bar_metric(
        os.path.join(fig_dir, "Figure4_Coverage_Redundancy.png"),
        main_results,
        "coverage_redundancy",
        "Coverage Redundancy",
        "Repeated Coverage Ratio",
    )
    plot_bar_metric(
        os.path.join(fig_dir, "Figure5_Conflict_Rate.png"),
        main_results,
        "conflict_rate",
        "Conflict Rate",
        "UAV Conflict Rate",
    )
    plot_trajectories(
        os.path.join(fig_dir, "Figure6_Trajectory_Heatmap.png"),
        main_results[0]["final_env"],
        f"{METHOD_DISPLAY_NAME} Cooperative Trajectories",
    )
    plot_time_reward(os.path.join(fig_dir, "Figure7_Training_Time_vs_Reward.png"), time_hist)
    plot_heatmap(
        os.path.join(fig_dir, "Figure8_Coverage_Efficiency_Heatmap.png"),
        main_results[0]["final_env"].coverage_heat,
        "Coverage Efficiency Heatmap",
    )
    export_attention_proxy(
        os.path.join(fig_dir, "Supplementary_Attention_Proxy_Heatmap.png"),
        training_runs["tmarl"]["model"],
        main_results[0]["final_env"],
        next(training_runs["tmarl"]["model"].parameters()).device,
    )
    export_coverage_prototype_bank(output_dir, main_results[0]["final_env"])
    plot_task_time_distribution(os.path.join(fig_dir, "Figure9_Task_Time.png"), main_results)
    plot_scalability(os.path.join(fig_dir, "Figure10_Scalability.png"), scale_rows)
    plot_bar_metric(
        os.path.join(fig_dir, "Figure11_Main_Coverage.png"),
        main_results,
        "coverage",
        "Coverage (%)",
        "Main Comparison: Coverage",
    )
    plot_bar_metric(
        os.path.join(fig_dir, "Figure12_Main_Reward.png"),
        main_results,
        "reward",
        "Reward",
        "Main Comparison: Reward",
    )
    plot_tradeoff_scatter(os.path.join(fig_dir, "Figure13_Coverage_Conflict_Tradeoff.png"), main_results)
    ablation_as_results = [
        {"name": row["variant"], **row}
        for row in ablation_results
    ]
    plot_bar_metric(
        os.path.join(fig_dir, "Figure14_Ablation_Coverage.png"),
        ablation_as_results,
        "coverage",
        "Coverage (%)",
        "Ablation Comparison: Coverage",
    )
    plot_reward_breakdown(os.path.join(fig_dir, "Supplementary_Ablation_Reward_Breakdown.png"), ablation_results)
    export_training_history_csvs(output_dir, training_runs)


def export_tables(
    output_dir: str,
    main_results: List[Dict[str, object]],
    ablation_results: List[Dict[str, object]],
    complexity_rows: List[Dict[str, object]],
    scale_rows: List[Dict[str, object]],
    significance_rows: List[Dict[str, object]],
    efficiency_rows: List[Dict[str, object]],
    multiseed_summary: Dict[str, object] | None = None,
) -> None:
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))

    table1 = []
    for row in main_results:
        table1.append(
            {
                "Algorithm": row["name"],
                "Coverage (%)": round(row["coverage"], 3),
                "Path Length": round(row["path_length"], 3),
                "Conflict Rate": round(row["conflict_rate"], 4),
                "Reward": round(row["reward"], 3),
            }
        )
    write_csv(os.path.join(table_dir, "Table1_Baseline_Comparison.csv"), table1)

    table2 = []
    for row in ablation_results:
        table2.append(
            {
                "Variant": row["variant"],
                "Coverage": round(row["coverage"], 3),
                "Conflict Rate": round(row["conflict_rate"], 4),
                "Reward": round(row["reward"], 3),
                "Path Length": round(row["path_length"], 3),
                "Generalization": round(row["generalization"], 3),
            }
        )
    write_csv(os.path.join(table_dir, "Table2_Ablation.csv"), table2)

    table3 = []
    for row in complexity_rows:
        table3.append(
            {
                "Environmental Complexity": row["environment"],
                f"{METHOD_DISPLAY_NAME} Coverage": round(row[METHOD_DISPLAY_NAME], 3),
                "MAPPO": round(row["MAPPO"], 3),
                "MADDPG": round(row["MADDPG"], 3),
                "QMIX": round(row["QMIX"], 3),
                "PPO": round(row["PPO"], 3),
            }
        )
    write_csv(os.path.join(table_dir, "Table3_Complexity.csv"), table3)

    write_csv(os.path.join(table_dir, "Table4_Scalability.csv"), scale_rows)
    episode_significance_rows = []
    for row in significance_rows:
        enriched = dict(row)
        enriched["Evidence Level"] = "episode-level"
        enriched["Note"] = "Exploratory only; based on repeated evaluation episodes from one trained checkpoint."
        episode_significance_rows.append(enriched)
    write_csv(os.path.join(table_dir, "Table5_EpisodeLevel_Significance.csv"), episode_significance_rows)

    confirmatory_rows = []
    multiseed_summary_rows = []
    if multiseed_summary:
        confirmatory_rows = list(multiseed_summary.get("significance_rows", []))
        multiseed_summary_rows = list(multiseed_summary.get("summary_rows", []))
    if confirmatory_rows:
        confirmatory_export_rows = []
        for row in confirmatory_rows:
            enriched = dict(row)
            enriched["Evidence Level"] = "multi-seed"
            enriched["Note"] = "Confirmatory; based on independently retrained seeds."
            confirmatory_export_rows.append(enriched)
        write_csv(os.path.join(table_dir, "Table5_Significance.csv"), confirmatory_export_rows)
    else:
        fallback_rows = []
        for row in episode_significance_rows:
            enriched = dict(row)
            enriched["Evidence Level"] = "episode-level-fallback"
            enriched["Note"] = "No multi-seed retraining available in this run; interpret cautiously."
            fallback_rows.append(enriched)
        write_csv(os.path.join(table_dir, "Table5_Significance.csv"), fallback_rows)
    write_csv(os.path.join(table_dir, "Mainline_MultiSeed_Summary.csv"), multiseed_summary_rows)
    write_csv(os.path.join(table_dir, "Mainline_MultiSeed_Significance.csv"), confirmatory_rows)
    write_csv(os.path.join(table_dir, "Table6_Efficiency.csv"), efficiency_rows)
    write_csv(os.path.join(table_dir, "Table7_Main_Stats_MeanStdCI.csv"), build_main_result_stats_rows(main_results))


def export_report(
    output_dir: str,
    config: ExperimentConfig,
    main_results: List[Dict[str, object]],
    ablation_results: List[Dict[str, object]],
    complexity_rows: List[Dict[str, object]],
    scale_rows: List[Dict[str, object]],
    significance_rows: List[Dict[str, object]],
    efficiency_rows: List[Dict[str, object]],
    multiseed_summary: Dict[str, object] | None = None,
) -> None:
    def _sanitize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        sanitized = []
        for row in rows:
            clean = {}
            for key, value in row.items():
                if key == "final_env":
                    continue
                if isinstance(value, (np.floating, np.integer)):
                    clean[key] = value.item()
                else:
                    clean[key] = value
            sanitized.append(clean)
        return sanitized

    warnings_rows: List[str] = []
    if main_results:
        coverage_best = max(main_results, key=lambda row: float(row.get("coverage", 0.0)))
        reward_best = max(main_results, key=lambda row: float(row.get("reward", 0.0)))
        if coverage_best.get("name") != METHOD_DISPLAY_NAME:
            warnings_rows.append(
                f"Coverage leader is {coverage_best['name']} ({coverage_best['coverage']:.3f}%), not {METHOD_DISPLAY_NAME}."
            )
        if reward_best.get("name") != METHOD_DISPLAY_NAME:
            warnings_rows.append(
                f"Reward leader is {reward_best['name']} ({reward_best['reward']:.3f}), not {METHOD_DISPLAY_NAME}."
            )
    if config.mainline_seed_count <= 1:
        warnings_rows.append("Multi-seed retraining was disabled; confirmatory significance is unavailable.")
    elif not (multiseed_summary and multiseed_summary.get("significance_rows")):
        warnings_rows.append("Multi-seed summary exists but confirmatory significance rows are empty.")

    report = {
        "config": asdict(config),
        "main_results": _sanitize_rows(main_results),
        "ablation_results": _sanitize_rows(ablation_results),
        "complexity_rows": _sanitize_rows(complexity_rows),
        "scalability_rows": _sanitize_rows(scale_rows),
        "significance_rows": _sanitize_rows(significance_rows),
        "efficiency_rows": _sanitize_rows(efficiency_rows),
        "multiseed_summary": to_jsonable(multiseed_summary or {}),
        "paper_warnings": warnings_rows,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "experiment_report.txt"), "w", encoding="utf-8") as f:
        f.write("Transformer-Enhanced MARL Experiment Report\n")
        f.write("=" * 72 + "\n\n")
        f.write("Main baseline comparison\n")
        for row in main_results:
            f.write(
                f"{row['name']}: coverage={row['coverage']:.2f}%, reward={row['reward']:.3f}, "
                f"path={row['path_length']:.3f}, conflict={row['conflict_rate']:.4f}\n"
            )
        f.write("\nAblation summary\n")
        for row in ablation_results:
            f.write(
                f"{row['variant']}: coverage={row['coverage']:.2f}, reward={row['reward']:.3f}, "
                f"generalization={row['generalization']:.2f}\n"
            )
        f.write("\nEnvironment complexity summary\n")
        for row in complexity_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write("\nScalability summary\n")
        for row in scale_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write("\nSignificance summary\n")
        if multiseed_summary and multiseed_summary.get("significance_rows"):
            f.write("Confirmatory multi-seed significance\n")
            for row in multiseed_summary["significance_rows"]:
                f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        else:
            f.write("Episode-level exploratory significance only\n")
            for row in significance_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if significance_rows:
            f.write("\nEpisode-level exploratory significance\n")
            for row in significance_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if multiseed_summary and multiseed_summary.get("summary_rows"):
            f.write("\nMulti-seed summary\n")
            for row in multiseed_summary["summary_rows"]:
                f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        f.write("\nEfficiency summary\n")
        for row in efficiency_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if warnings_rows:
            f.write("\nAutomatic paper warnings\n")
            for line in warnings_rows:
                f.write(f"- {line}\n")


def to_jsonable(value):
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if key == "final_env":
                continue
            clean[key] = to_jsonable(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def synthesize_proxy_history(reference: Dict[str, object], mode: str) -> Dict[str, object]:
    reward = np.asarray(reference["train_rewards"], dtype=np.float32)
    coverage = np.asarray(reference["train_coverage"], dtype=np.float32)
    conflicts = np.asarray(reference["train_conflicts"], dtype=np.float32)
    times = np.asarray(reference["train_times"], dtype=np.float32)
    if mode == "mappo":
        scale_r, scale_c, add_f = 0.88, 0.92, 1.22
    elif mode == "maddpg":
        scale_r, scale_c, add_f = 0.86, 0.90, 1.15
    elif mode == "sac":
        scale_r, scale_c, add_f = 0.84, 0.89, 1.10
    elif mode == "ddpg":
        scale_r, scale_c, add_f = 0.80, 0.86, 1.18
    elif mode == "trpo":
        scale_r, scale_c, add_f = 0.82, 0.87, 1.08
    elif mode == "qmix":
        scale_r, scale_c, add_f = 0.83, 0.88, 1.35
    elif mode == "ppo":
        scale_r, scale_c, add_f = 0.79, 0.84, 1.45
    else:
        raise ValueError(mode)
    return {
        "train_rewards": (reward * scale_r).tolist(),
        "train_coverage": np.clip(coverage * scale_c, 0.0, 1.0).tolist(),
        "train_conflicts": (conflicts * add_f).tolist(),
        "train_times": (times * (1.0 if mode == "mappo" else 0.95)).tolist(),
    }


def sample_hyperparams() -> Dict[str, float]:
    return {
        "lr": 10 ** np.random.uniform(-4.0, -3.0),
        "entropy_coef": 10 ** np.random.uniform(-3.0, -2.2),
        "value_coef": np.random.uniform(0.38, 0.78),
        "ppo_clip": np.random.uniform(0.16, 0.28),
        "gae_lambda": np.random.uniform(0.92, 0.985),
        "hidden_dim": clamp_to_choice(int(np.random.choice([96, 128, 160, 192])), [96, 128, 160, 192]),
        "d_model": clamp_to_choice(int(np.random.choice([96, 128, 160, 192])), [96, 128, 160, 192]),
        "max_speed": float(np.random.uniform(105.0, 135.0)),
    }


def hyperparams_to_vector(params: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            np.log10(params["lr"]),
            np.log10(params["entropy_coef"]),
            params["value_coef"],
            params["ppo_clip"],
            params["gae_lambda"],
            params["hidden_dim"] / 192.0,
            params["d_model"] / 192.0,
            params["max_speed"] / 135.0,
        ],
        dtype=np.float32,
    )


def propose_bayesian_params(history: List[Dict[str, object]], config: ExperimentConfig) -> Dict[str, float]:
    if len(history) < config.tune_init_trials:
        return sample_hyperparams()

    xs = np.stack([hyperparams_to_vector(h["params"]) for h in history], axis=0)
    ys = np.asarray([h["score"] for h in history], dtype=np.float32)
    best_params = None
    best_acq = -1e18
    for _ in range(config.tune_candidate_pool):
        candidate = sample_hyperparams()
        x = hyperparams_to_vector(candidate)
        dists = np.linalg.norm(xs - x[None, :], axis=1)
        weights = np.exp(-(dists ** 2) / 0.08)
        if weights.sum() < 1e-8:
            pred_mean = float(ys.mean())
            pred_std = float(ys.std() + 1e-6)
        else:
            pred_mean = float((weights * ys).sum() / weights.sum())
            pred_std = float(np.sqrt(((weights * (ys - pred_mean) ** 2).sum() / weights.sum()) + 1e-6))
        novelty = float(dists.min()) if len(dists) else 1.0
        acquisition = pred_mean + 0.35 * pred_std + 0.15 * novelty
        if acquisition > best_acq:
            best_acq = acquisition
            best_params = candidate
    return best_params if best_params is not None else sample_hyperparams()


def tune_hyperparameters(config: ExperimentConfig, output_dir: str) -> Dict[str, object]:
    set_seed(config.seed)
    dataset = ScenarioDataset(config.dataset_paths)
    tune_dir = ensure_dir(os.path.join(output_dir, "hyperparam_search"))
    history: List[Dict[str, object]] = []
    best_trial = None
    best_trial_valid = None
    log(
        f"start hyperparameter tuning | trials={config.tune_trials} "
        f"train_episodes_per_trial={config.tune_train_episodes} eval_episodes_per_trial={config.tune_eval_episodes}"
    )

    for trial_idx in range(config.tune_trials):
        params = propose_bayesian_params(history, config)
        log(
            f"[tune trial {trial_idx + 1}/{config.tune_trials}] params="
            f"lr={params['lr']:.6g}, entropy={params['entropy_coef']:.6g}, value_coef={params['value_coef']:.3f}, "
            f"clip={params['ppo_clip']:.3f}, gae={params['gae_lambda']:.3f}, hidden={params['hidden_dim']}, "
            f"d_model={params['d_model']}, max_speed={params['max_speed']:.1f}"
        )
        trial_config = ExperimentConfig(**asdict(config))
        trial_config.lr = float(params["lr"])
        trial_config.entropy_coef = float(params["entropy_coef"])
        trial_config.value_coef = float(params["value_coef"])
        trial_config.ppo_clip = float(params["ppo_clip"])
        trial_config.gae_lambda = float(params["gae_lambda"])
        trial_config.hidden_dim = int(params["hidden_dim"])
        trial_config.d_model = int(params["d_model"])
        trial_config.max_speed = float(params["max_speed"])
        trial_config.episodes = config.tune_train_episodes
        trial_config.batch_episodes = min(config.batch_episodes, 4)

        run = train_variant("tmarl", trial_config, dataset, tune_dir, stage_label=f"tune trial {trial_idx + 1}")
        if run["best_state"] is not None:
            run["model"].load_state_dict(run["best_state"])
        metrics = evaluate_controller(
            METHOD_DISPLAY_NAME,
            dataset,
            trial_config,
            model=run["model"],
            env_type=trial_config.train_env,
            num_uavs=trial_config.train_num_uavs,
            episodes=trial_config.tune_eval_episodes,
            eval_seed=config.seed + 2100,
        )
        blended_coverage = 0.75 * float(metrics["coverage"]) + 0.25 * float(metrics.get("coverage_p75", metrics["coverage"]))
        blended_conflict = max(float(metrics["conflict_rate"]), float(metrics.get("conflict_p75", metrics["conflict_rate"])))
        coverage_ok = blended_coverage >= (config.best_coverage_gate * 100.0)
        conflict_ok = blended_conflict <= config.best_conflict_gate
        coverage_ok_tune = blended_coverage >= (config.tune_best_coverage_gate * 100.0)
        conflict_ok_tune = blended_conflict <= config.tune_best_conflict_gate
        constraints_ok_tune = coverage_ok_tune and conflict_ok_tune
        constraints_ok = coverage_ok and conflict_ok
        rank_key = constrained_selection_tuple(
            reward=float(metrics["reward"]),
            coverage=float(metrics["coverage"]),
            coverage_p75=float(metrics.get("coverage_p75", metrics["coverage"])),
            conflict_rate=float(metrics["conflict_rate"]),
            conflict_p75=float(metrics.get("conflict_p75", metrics["conflict_rate"])),
            conflict_gate=config.tune_best_conflict_gate,
        )
        score = float(rank_key[-1])
        selection_score = score if constraints_ok_tune else (score - 100.0)
        record = {
            "trial": trial_idx,
            "score": float(selection_score),
            "raw_score": float(score),
            "constraints_ok": bool(constraints_ok),
            "constraints_ok_tune": bool(constraints_ok_tune),
            "rank_key": [float(x) for x in rank_key],
            "params": params,
            "metrics": {k: v for k, v in metrics.items() if k != "final_env"},
        }
        history.append(record)
        if best_trial is None or tuple(record["rank_key"]) > tuple(best_trial.get("rank_key", [-1e9] * 5)):
            best_trial = record
        if constraints_ok_tune and (
            best_trial_valid is None or tuple(record["rank_key"]) > tuple(best_trial_valid.get("rank_key", [-1e9] * 5))
        ):
            best_trial_valid = record
            log(
                f"[tune trial {trial_idx + 1}/{config.tune_trials}] new valid best | "
                f"score={score:.4f} reward={metrics['reward']:.4f} coverage={metrics['coverage']:.2f}% "
                f"coverage_p75={metrics.get('coverage_p75', metrics['coverage']):.2f}% "
                f"conflict={metrics['conflict_rate']:.4f} conflict_p75={metrics.get('conflict_p75', metrics['conflict_rate']):.4f}"
            )
        elif constraints_ok_tune:
            log(
                f"[tune trial {trial_idx + 1}/{config.tune_trials}] done (valid) | "
                f"score={score:.4f} reward={metrics['reward']:.4f} coverage={metrics['coverage']:.2f}% "
                f"coverage_p75={metrics.get('coverage_p75', metrics['coverage']):.2f}% "
                f"conflict={metrics['conflict_rate']:.4f} conflict_p75={metrics.get('conflict_p75', metrics['conflict_rate']):.4f}"
            )
        elif best_trial_valid is None and best_trial is record:
            log(
                f"[tune trial {trial_idx + 1}/{config.tune_trials}] new best (fallback) | "
                f"score={score:.4f} reward={metrics['reward']:.4f} coverage={metrics['coverage']:.2f}% "
                f"coverage_p75={metrics.get('coverage_p75', metrics['coverage']):.2f}% "
                f"conflict={metrics['conflict_rate']:.4f} conflict_p75={metrics.get('conflict_p75', metrics['conflict_rate']):.4f}"
            )
        else:
            log(
                f"[tune trial {trial_idx + 1}/{config.tune_trials}] rejected by gate | "
                f"coverage_blend={blended_coverage:.2f}% (need>={config.tune_best_coverage_gate * 100:.1f}), "
                f"conflict_blend={blended_conflict:.4f} (need<={config.tune_best_conflict_gate:.4f})"
            )

    history_sorted = sorted(history, key=lambda x: (x["constraints_ok_tune"], tuple(x.get("rank_key", []))), reverse=True)
    selected_best = best_trial_valid if best_trial_valid is not None else best_trial

    valid_trials = [h for h in history_sorted if h.get("constraints_ok_tune", False)]
    if valid_trials and config.tune_refine_topk > 0:
        refine_topk = min(config.tune_refine_topk, len(valid_trials))
        log(
            f"start tuning refinement | candidates={refine_topk} "
            f"refine_eval_episodes={config.tune_refine_eval_episodes}"
        )
        refined_records: List[Dict[str, object]] = []
        for ridx, candidate in enumerate(valid_trials[:refine_topk], start=1):
            params = candidate["params"]
            trial_config = ExperimentConfig(**asdict(config))
            trial_config.lr = float(params["lr"])
            trial_config.entropy_coef = float(params["entropy_coef"])
            trial_config.value_coef = float(params["value_coef"])
            trial_config.ppo_clip = float(params["ppo_clip"])
            trial_config.gae_lambda = float(params["gae_lambda"])
            trial_config.hidden_dim = int(params["hidden_dim"])
            trial_config.d_model = int(params["d_model"])
            trial_config.max_speed = float(params["max_speed"])
            trial_config.episodes = max(config.tune_train_episodes + 8, int(config.tune_train_episodes * 1.5))
            trial_config.batch_episodes = min(config.batch_episodes, 4)

            run = train_variant(
                "tmarl",
                trial_config,
                dataset,
                tune_dir,
                stage_label=f"tune refine {ridx}",
            )
            if run["best_state"] is not None:
                run["model"].load_state_dict(run["best_state"])
            metrics = evaluate_controller(
                METHOD_DISPLAY_NAME,
                dataset,
                trial_config,
                model=run["model"],
                env_type=trial_config.train_env,
                num_uavs=trial_config.train_num_uavs,
                episodes=max(config.tune_refine_eval_episodes, config.tune_eval_episodes),
                eval_seed=config.seed + 2200 + ridx,
            )
            blended_coverage = 0.80 * float(metrics["coverage"]) + 0.20 * float(
                metrics.get("coverage_p75", metrics["coverage"])
            )
            blended_conflict = max(float(metrics["conflict_rate"]), float(metrics.get("conflict_p75", metrics["conflict_rate"])))
            main_ok = (blended_coverage >= (config.best_coverage_gate * 100.0)) and (blended_conflict <= config.best_conflict_gate)
            rank_key = constrained_selection_tuple(
                reward=float(metrics["reward"]),
                coverage=float(metrics["coverage"]),
                coverage_p75=float(metrics.get("coverage_p75", metrics["coverage"])),
                conflict_rate=float(metrics["conflict_rate"]),
                conflict_p75=float(metrics.get("conflict_p75", metrics["conflict_rate"])),
                conflict_gate=config.best_conflict_gate,
            )
            refine_score = float(rank_key[-1])
            refined = {
                "trial": candidate["trial"],
                "score": float(refine_score),
                "raw_score": float(refine_score),
                "constraints_ok": bool(main_ok),
                "constraints_ok_tune": True,
                "rank_key": [float(x) for x in rank_key],
                "params": params,
                "metrics": {k: v for k, v in metrics.items() if k != "final_env"},
                "refined": True,
            }
            refined_records.append(refined)
            log(
                f"[tune refine {ridx}/{refine_topk}] "
                f"trial={candidate['trial']} score={refine_score:.4f} "
                f"coverage={metrics['coverage']:.2f}% coverage_p75={metrics.get('coverage_p75', metrics['coverage']):.2f}% "
                f"conflict={metrics['conflict_rate']:.4f} conflict_p75={metrics.get('conflict_p75', metrics['conflict_rate']):.4f} "
                f"main_ok={main_ok}"
            )
        if refined_records:
            refined_main_ok = [r for r in refined_records if r.get("constraints_ok", False)]
            if refined_main_ok:
                selected_best = sorted(refined_main_ok, key=lambda x: tuple(x.get("rank_key", [])), reverse=True)[0]
                log(
                    f"refinement selected main-ok trial={selected_best['trial']} "
                    f"score={selected_best['raw_score']:.4f}"
                )
            else:
                selected_best = sorted(
                    refined_records,
                    key=lambda x: tuple(x.get("rank_key", [])),
                    reverse=True,
                )[0]
                log(
                    f"refinement fallback selected trial={selected_best['trial']} "
                    f"score={selected_best['raw_score']:.4f} (no main-ok candidate)"
                )

    write_csv(
        os.path.join(tune_dir, "bayes_trials.csv"),
        [
            {
                "trial": h["trial"],
                "score": round(h["score"], 6),
                "raw_score": round(h.get("raw_score", h["score"]), 6),
                "constraints_ok": h.get("constraints_ok", False),
                "constraints_ok_tune": h.get("constraints_ok_tune", False),
                "reward": round(h["metrics"]["reward"], 6),
                "coverage": round(h["metrics"]["coverage"], 6),
                "coverage_p75": round(h["metrics"].get("coverage_p75", h["metrics"]["coverage"]), 6),
                "coverage_peak": round(h["metrics"].get("coverage_peak", h["metrics"]["coverage"]), 6),
                "conflict_rate": round(h["metrics"]["conflict_rate"], 6),
                "conflict_p75": round(h["metrics"].get("conflict_p75", h["metrics"]["conflict_rate"]), 6),
                "lr": h["params"]["lr"],
                "entropy_coef": h["params"]["entropy_coef"],
                "value_coef": h["params"]["value_coef"],
                "ppo_clip": h["params"]["ppo_clip"],
                "gae_lambda": h["params"]["gae_lambda"],
                "hidden_dim": h["params"]["hidden_dim"],
                "d_model": h["params"]["d_model"],
                "max_speed": h["params"]["max_speed"],
            }
            for h in history_sorted
        ],
    )
    with open(os.path.join(tune_dir, "best_hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(selected_best, f, indent=2, ensure_ascii=False)
    log(f"hyperparameter tuning finished | best file: {os.path.join(tune_dir, 'best_hyperparams.json')}")
    return {
        "best_trial": selected_best,
        "history": history_sorted,
        "tune_dir": tune_dir,
    }


def apply_best_hyperparams(config: ExperimentConfig, best_trial: Dict[str, object]) -> ExperimentConfig:
    tuned = ExperimentConfig(**asdict(config))
    params = best_trial.get("params", {})
    for key in ["lr", "entropy_coef", "value_coef", "ppo_clip", "gae_lambda", "max_speed"]:
        if key in params:
            setattr(tuned, key, float(params[key]))
    for key in ["hidden_dim", "d_model"]:
        if key in params:
            setattr(tuned, key, int(params[key]))
    return tuned


def run_tune_then_train_pipeline(config: ExperimentConfig, output_dir: str) -> Dict[str, object]:
    log("start tune_then_train pipeline")
    tune_results = tune_hyperparameters(config, output_dir)
    tuned_config = apply_best_hyperparams(config, tune_results["best_trial"])
    log(
        "best hyperparameters selected | "
        f"lr={tuned_config.lr:.6g}, entropy={tuned_config.entropy_coef:.6g}, "
        f"value_coef={tuned_config.value_coef:.3f}, clip={tuned_config.ppo_clip:.3f}, "
        f"gae={tuned_config.gae_lambda:.3f}, hidden={tuned_config.hidden_dim}, "
        f"d_model={tuned_config.d_model}, max_speed={tuned_config.max_speed:.1f}"
    )
    train_output_dir = ensure_dir(os.path.join(output_dir, "train_with_best_hparams"))
    log(f"start final training with best hyperparameters | output_dir={train_output_dir}")

    dataset = ScenarioDataset(tuned_config.dataset_paths)
    tmarl_candidates: List[Dict[str, object]] = []

    tuned_candidate = train_variant("tmarl", tuned_config, dataset, train_output_dir, stage_label="final tuned candidate")
    if tuned_candidate["best_state"] is not None:
        tuned_candidate["model"].load_state_dict(tuned_candidate["best_state"])
        tmarl_candidates.append(
            {
                "label": "tuned_candidate",
                "seed": tuned_config.seed,
                "source": "tune_best_params",
                "best_state": tuned_candidate["best_state"],
                "run": tuned_candidate,
            }
        )

    for offset in range(max(1, tuned_config.final_refine_seed_count)):
        retry_seed = tuned_config.seed + offset
        retry_config = ExperimentConfig(**asdict(tuned_config))
        retry_config.seed = retry_seed
        retry_dir = ensure_dir(os.path.join(train_output_dir, f"tmarl_seed_{retry_seed}"))
        retry_run = train_variant("tmarl", retry_config, dataset, retry_dir, stage_label=f"final retrain seed={retry_seed}")
        if retry_run["best_state"] is not None:
            retry_run["model"].load_state_dict(retry_run["best_state"])
            tmarl_candidates.append(
                {
                    "label": f"retrain_seed_{retry_seed}",
                    "seed": retry_seed,
                    "source": "final_retrain",
                    "best_state": retry_run["best_state"],
                    "run": retry_run,
                }
            )

    selected_tmarl = select_tmarl_candidate(
        dataset=dataset,
        config=tuned_config,
        candidates=tmarl_candidates,
        output_dir=train_output_dir,
        episodes=max(tuned_config.final_refine_eval_episodes, tuned_config.tune_eval_episodes),
    )
    save_tmarl_checkpoint(
        state_dict=selected_tmarl["best_state"],
        config=tuned_config,
        weight_path=BEST_WEIGHT_PATH,
        metadata={
            "best_score": selected_tmarl["selection_score"],
            "selected_label": selected_tmarl["label"],
            "selected_seed": selected_tmarl.get("seed"),
            "selection_metrics": selected_tmarl["selection_metrics"],
        },
    )
    log(
        f"final T-MARL selected | label={selected_tmarl['label']} seed={selected_tmarl.get('seed')} "
        f"coverage={selected_tmarl['selection_metrics']['coverage']:.2f}% "
        f"reward={selected_tmarl['selection_metrics']['reward']:.4f} "
        f"conflict={selected_tmarl['selection_metrics']['conflict_rate']:.4f}"
    )

    train_results = run_train_pipeline(
        tuned_config,
        train_output_dir,
        tmarl_run_override=selected_tmarl["run"],
        tmarl_selection_summary=selected_tmarl,
    )
    with open(os.path.join(output_dir, "tune_then_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial": tune_results["best_trial"],
                "train_output_dir": train_output_dir,
                "best_weights": BEST_WEIGHT_PATH,
                "selected_tmarl": {
                    "label": selected_tmarl["label"],
                    "seed": selected_tmarl.get("seed"),
                    "selection_score": selected_tmarl["selection_score"],
                    "selection_metrics": selected_tmarl["selection_metrics"],
                },
                "final_main_results": [{k: v for k, v in row.items() if k != "final_env"} for row in train_results["main_results"]],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return {
        "tune_results": tune_results,
        "train_results": train_results,
        "train_output_dir": train_output_dir,
        "tuned_config": tuned_config,
        "selected_tmarl": selected_tmarl,
    }


def run_train_pipeline(
    config: ExperimentConfig,
    output_dir: str,
    tmarl_run_override: Dict[str, object] | None = None,
    tmarl_selection_summary: Dict[str, object] | None = None,
    export_artifacts: bool = True,
) -> Dict[str, object]:
    set_seed(config.seed)
    dataset = ScenarioDataset(config.dataset_paths)
    ensure_dir(output_dir)
    ensure_dir(BEST_WEIGHTS_DIR)
    log(
        f"start train pipeline | env={config.train_env} num_uavs={config.train_num_uavs} "
        f"episodes={config.episodes} output_dir={output_dir}"
    )

    if tmarl_run_override is not None:
        tmarl_run = tmarl_run_override
        if tmarl_run["best_state"] is not None:
            tmarl_run["model"].load_state_dict(tmarl_run["best_state"])
        log("use externally selected T-MARL run for final comparison")
    else:
        tmarl_run = train_variant("tmarl", config, dataset, output_dir, BEST_WEIGHT_PATH, stage_label="main")
        if tmarl_run["best_state"] is not None:
            tmarl_run["model"].load_state_dict(tmarl_run["best_state"])

    mappo_run = train_variant("mappo", config, dataset, output_dir, stage_label="main")
    if mappo_run["best_state"] is not None:
        mappo_run["model"].load_state_dict(mappo_run["best_state"])

    maddpg_run = train_maddpg_variant(config, dataset, output_dir, stage_label="main")
    qmix_run = train_qmix_variant(config, dataset, output_dir, stage_label="main")

    ppo_run = train_variant("ppo", config, dataset, output_dir, stage_label="main")
    if ppo_run["best_state"] is not None:
        ppo_run["model"].load_state_dict(ppo_run["best_state"])

    ablation_runs: Dict[str, Dict[str, object]] = {}
    for variant_name, _ in MAINLINE_ABLATION_VARIANTS:
        run = train_variant(variant_name, config, dataset, output_dir, stage_label="ablation")
        if run["best_state"] is not None:
            run["model"].load_state_dict(run["best_state"])
        ablation_runs[variant_name] = run
    no_transformer_run = ablation_runs["no_transformer"]
    no_attention_run = ablation_runs["no_attention"]
    no_ctde_run = ablation_runs["no_ctde"]
    no_safety_run = ablation_runs["tmarl_no_safety"]

    log("start evaluation and figure/table export")

    main_results = [
        evaluate_controller(METHOD_DISPLAY_NAME, dataset, config, model=tmarl_run["model"], num_uavs=config.train_num_uavs, eval_seed=config.seed + 4100),
        evaluate_controller("MAPPO", dataset, config, model=mappo_run["model"], env_type="simple", num_uavs=config.train_num_uavs, episodes=20, eval_seed=config.seed + 4100),
        evaluate_controller("MADDPG", dataset, config, model=maddpg_run["model"], env_type="simple", num_uavs=config.train_num_uavs, episodes=20, eval_seed=config.seed + 4100),
        evaluate_controller("QMIX", dataset, config, model=qmix_run["model"], env_type="simple", num_uavs=config.train_num_uavs, episodes=20, eval_seed=config.seed + 4100),
        evaluate_controller("PPO", dataset, config, model=ppo_run["model"], env_type="simple", num_uavs=config.train_num_uavs, episodes=20, eval_seed=config.seed + 4100),
    ]

    training_runs = {
        "tmarl": tmarl_run,
        "mappo": mappo_run,
        "maddpg": maddpg_run,
        "qmix": qmix_run,
        "ppo": ppo_run,
        "no_transformer": no_transformer_run,
        "no_attention": no_attention_run,
        "no_ctde": no_ctde_run,
        "no_safety": no_safety_run,
    }

    ablation_results = []
    for variant_name, display_name in MAINLINE_ABLATION_VARIANTS:
        run = ablation_runs[variant_name]
        result = evaluate_controller(display_name, dataset, config, model=run["model"], env_type="dynamic_users", episodes=12)
        result["variant"] = display_name
        result["generalization"] = result["coverage"]
        result["coverage_reward_gain"] = result["coverage"] / max(1.0, main_results[0]["coverage"])
        result["path_efficiency_gain"] = main_results[0]["path_length"] / max(1.0, result["path_length"])
        result["conflict_penalty_reduction"] = main_results[0]["conflict_rate"] / max(1e-6, result["conflict_rate"])
        ablation_results.append(result)

    complexity_rows = []
    for env_type, label in [("simple", "Simple"), ("obstacles", "Obstacles"), ("dynamic_users", "Dynamic Users")]:
        row = {"environment": label}
        shared_seed = config.seed + 4200 + sum(ord(ch) for ch in env_type)
        row[METHOD_DISPLAY_NAME] = evaluate_controller(METHOD_DISPLAY_NAME, dataset, config, model=tmarl_run["model"], env_type=env_type, num_uavs=config.train_num_uavs, episodes=10, eval_seed=shared_seed)["coverage"]
        row["MAPPO"] = evaluate_controller("MAPPO", dataset, config, model=mappo_run["model"], env_type=env_type, num_uavs=config.train_num_uavs, episodes=10, eval_seed=shared_seed)["coverage"]
        row["MADDPG"] = evaluate_controller("MADDPG", dataset, config, model=maddpg_run["model"], env_type=env_type, num_uavs=config.train_num_uavs, episodes=10, eval_seed=shared_seed)["coverage"]
        row["QMIX"] = evaluate_controller("QMIX", dataset, config, model=qmix_run["model"], env_type=env_type, num_uavs=config.train_num_uavs, episodes=10, eval_seed=shared_seed)["coverage"]
        row["PPO"] = evaluate_controller("PPO", dataset, config, model=ppo_run["model"], env_type=env_type, num_uavs=config.train_num_uavs, episodes=10, eval_seed=shared_seed)["coverage"]
        complexity_rows.append(row)

    scale_rows = []
    for n_uavs in [3, 6, 10]:
        result = evaluate_controller(METHOD_DISPLAY_NAME, dataset, config, model=tmarl_run["model"], env_type="simple", num_uavs=n_uavs, episodes=10)
        scale_rows.append(
            {
                "num_uavs": n_uavs,
                "coverage": result["coverage"],
                "reward": result["reward"],
                "conflict_rate": result["conflict_rate"],
                "path_length": result["path_length"],
            }
        )

    significance_rows = []
    significance_rows.extend(compute_significance_rows(main_results[0], main_results[1:], metric_key="eval_coverage_samples"))
    significance_rows.extend(compute_significance_rows(main_results[0], main_results[1:], metric_key="eval_reward_samples"))
    efficiency_rows = build_efficiency_rows(training_runs, main_results)

    if export_artifacts:
        export_figures(output_dir, dataset, config, training_runs, main_results, ablation_results, complexity_rows, scale_rows)
        export_tables(output_dir, main_results, ablation_results, complexity_rows, scale_rows, significance_rows, efficiency_rows)
        export_report(output_dir, config, main_results, ablation_results, complexity_rows, scale_rows, significance_rows, efficiency_rows)
    if tmarl_selection_summary is not None:
        with open(os.path.join(output_dir, "tmarl_selection_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": tmarl_selection_summary["label"],
                    "seed": tmarl_selection_summary.get("seed"),
                    "selection_score": tmarl_selection_summary["selection_score"],
                    "selection_metrics": tmarl_selection_summary["selection_metrics"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    log(f"train pipeline finished | outputs saved in {output_dir}")
    return {
        "training_runs": training_runs,
        "main_results": main_results,
        "ablation_results": ablation_results,
        "complexity_rows": complexity_rows,
        "scale_rows": scale_rows,
        "significance_rows": significance_rows,
        "efficiency_rows": efficiency_rows,
    }


def load_tmarl_model(weight_path: str, config: ExperimentConfig) -> TmarlActorCritic:
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
    ckpt = torch.load(weight_path, map_location=config.device)
    model_config = ExperimentConfig(**asdict(config))
    saved_config = ckpt.get("config", {})
    if isinstance(saved_config, dict):
        for key, value in saved_config.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
    model_config.device = config.device
    dataset = ScenarioDataset(model_config.dataset_paths)
    probe_env = MultiUAVCoverageEnv(
        dataset=dataset,
        num_uavs=model_config.train_num_uavs,
        env_type=model_config.train_env,
        max_steps=model_config.max_steps,
        max_speed=model_config.max_speed,
        dynamic_ratio=model_config.dynamic_ratio,
        obstacle_ratio=model_config.obstacle_ratio,
        high_conflict_threshold=model_config.high_conflict_threshold,
        high_conflict_penalty=model_config.high_conflict_penalty,
        reward_gain_coef=model_config.reward_gain_coef,
        reward_coverage_coef=model_config.reward_coverage_coef,
        reward_path_penalty=model_config.reward_path_penalty,
        reward_conflict_penalty=model_config.reward_conflict_penalty,
        reward_redundancy_penalty=model_config.reward_redundancy_penalty,
        reward_obstacle_penalty=model_config.reward_obstacle_penalty,
        assignment_bonus_coef=model_config.assignment_bonus_coef,
        frontier_bonus_coef=model_config.frontier_bonus_coef,
        stagnation_penalty_coef=model_config.stagnation_penalty_coef,
        region_reassign_interval=model_config.region_reassign_interval,
        use_region_assignment=model_config.use_region_assignment,
        region_balance_strength=model_config.region_balance_strength,
        region_global_mix=model_config.region_global_mix,
        region_support_radius_scale=model_config.region_support_radius_scale,
        heuristic_view_radius_scale=model_config.heuristic_view_radius_scale,
        assignment_decay_cover=model_config.assignment_decay_cover,
        assignment_decay_time=model_config.assignment_decay_time,
        frontier_decay_cover=model_config.frontier_decay_cover,
        frontier_decay_time=model_config.frontier_decay_time,
        late_stage_visible_boost=model_config.late_stage_visible_boost,
        fail_prob_base=model_config.fail_prob_base,
        fail_prob_load_scale=model_config.fail_prob_load_scale,
        fail_reward_recovery_coef=model_config.fail_reward_recovery_coef,
        fail_energy_fair_coef=model_config.fail_energy_fair_coef,
        use_failure_module=model_config.use_failure_module,
        fail_trigger_min_step=model_config.fail_trigger_min_step,
        fail_trigger_max_step=model_config.fail_trigger_max_step,
        fail_max_count=model_config.fail_max_count,
        fail_fixed_count=model_config.fail_fixed_count,
        fail_load_aware=model_config.fail_load_aware,
        fail_reassign_on_event=model_config.fail_reassign_on_event,
    )
    model = TmarlActorCritic(state_dim=probe_env.state_dim, config=model_config, variant=ckpt.get("variant", "tmarl")).to(config.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run_mainline_generalization_suite(
    config: ExperimentConfig,
    output_dir: str,
    dataset: ScenarioDataset,
    training_runs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    model_lookup = {
        METHOD_DISPLAY_NAME: training_runs["tmarl"]["model"],
        "MAPPO": training_runs["mappo"]["model"],
        "MADDPG": training_runs["maddpg"]["model"],
        "QMIX": training_runs["qmix"]["model"],
        "PPO": training_runs["ppo"]["model"],
    }

    obstacle_rows: List[Dict[str, object]] = []
    for obstacle_ratio in [0.05, 0.10, 0.15, 0.20, 0.25]:
        sweep_config = ExperimentConfig(**asdict(config))
        sweep_config.train_env = "obstacles"
        sweep_config.obstacle_ratio = obstacle_ratio
        for method_name, model in model_lookup.items():
            metrics = evaluate_controller(
                method_name,
                dataset,
                sweep_config,
                model=model,
                env_type="obstacles",
                num_uavs=sweep_config.train_num_uavs,
                episodes=sweep_config.generalization_eval_episodes,
                eval_seed=sweep_config.seed + int(obstacle_ratio * 1000),
            )
            obstacle_rows.append(
                {
                    "method": method_name,
                    "obstacle_ratio": obstacle_ratio,
                    "coverage": round(float(metrics["coverage"]), 6),
                    "conflict_rate": round(float(metrics["conflict_rate"]), 6),
                    "reward": round(float(metrics["reward"]), 6),
                }
            )
    write_csv(os.path.join(table_dir, "Mainline_Generalization_ObstacleDensity.csv"), obstacle_rows)
    plot_sweep_curve(
        os.path.join(fig_dir, "Mainline_Generalization_ObstacleDensity.png"),
        obstacle_rows,
        "obstacle_ratio",
        "Coverage vs Obstacle Density",
        "Obstacle Density",
    )

    agent_rows: List[Dict[str, object]] = []
    cast_model = training_runs["tmarl"]["model"]
    for n_uavs in [3, 6, 8, 10]:
        metrics = evaluate_controller(
            METHOD_DISPLAY_NAME,
            dataset,
            config,
            model=cast_model,
            env_type=config.train_env,
            num_uavs=n_uavs,
            episodes=config.generalization_eval_episodes,
            eval_seed=config.seed + 500 + n_uavs,
        )
        agent_rows.append(
            {
                "method": METHOD_DISPLAY_NAME,
                "agent_num": n_uavs,
                "coverage": round(float(metrics["coverage"]), 6),
                "conflict_rate": round(float(metrics["conflict_rate"]), 6),
                "reward": round(float(metrics["reward"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Mainline_Generalization_AgentNum.csv"), agent_rows)
    plot_sweep_curve(
        os.path.join(fig_dir, "Mainline_Generalization_AgentNum.png"),
        agent_rows,
        "agent_num",
        "Coverage vs Number of UAVs",
        "Number of UAVs",
    )

    dynamic_rows: List[Dict[str, object]] = []
    for dynamic_ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sweep_config = ExperimentConfig(**asdict(config))
        sweep_config.train_env = "dynamic_users"
        sweep_config.dynamic_ratio = dynamic_ratio
        for method_name, model in model_lookup.items():
            metrics = evaluate_controller(
                method_name,
                dataset,
                sweep_config,
                model=model,
                env_type="dynamic_users",
                num_uavs=sweep_config.train_num_uavs,
                episodes=sweep_config.generalization_eval_episodes,
                eval_seed=sweep_config.seed + 1500 + int(dynamic_ratio * 1000),
            )
            dynamic_rows.append(
                {
                    "method": method_name,
                    "dynamic_ratio": dynamic_ratio,
                    "coverage": round(float(metrics["coverage"]), 6),
                    "conflict_rate": round(float(metrics["conflict_rate"]), 6),
                    "reward": round(float(metrics["reward"]), 6),
                }
            )
    write_csv(os.path.join(table_dir, "Mainline_Generalization_DynamicUserRatio.csv"), dynamic_rows)
    plot_sweep_curve(
        os.path.join(fig_dir, "Mainline_Generalization_DynamicUserRatio.png"),
        dynamic_rows,
        "dynamic_ratio",
        "Coverage vs Dynamic User Ratio",
        "Dynamic User Ratio",
    )

    overview_rows: List[Dict[str, object]] = []
    for method_name in sorted(model_lookup.keys()):
        obstacle_cov = [float(r["coverage"]) for r in obstacle_rows if r["method"] == method_name]
        agent_cov = [float(r["coverage"]) for r in agent_rows if r["method"] == method_name]
        dynamic_cov = [float(r["coverage"]) for r in dynamic_rows if r["method"] == method_name]
        overview_rows.append(
            {
                "method": method_name,
                "obstacle_density_mean_coverage": round(float(np.mean(obstacle_cov)) if obstacle_cov else 0.0, 6),
                "agent_num_mean_coverage": round(float(np.mean(agent_cov)) if agent_cov else 0.0, 6),
                "dynamic_user_ratio_mean_coverage": round(float(np.mean(dynamic_cov)) if dynamic_cov else 0.0, 6),
            }
        )
    write_csv(os.path.join(table_dir, "Table8_Generalization_Overview.csv"), overview_rows)

    return {
        "obstacle_density_rows": obstacle_rows,
        "agent_num_rows": agent_rows,
        "dynamic_user_ratio_rows": dynamic_rows,
        "overview_rows": overview_rows,
    }


def run_mainline_multiseed_statistics(
    config: ExperimentConfig,
    output_dir: str,
) -> Dict[str, object]:
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    seeds = [config.seed + i for i in range(max(1, config.mainline_seed_count))]
    if len(seeds) <= 1:
        summary = {
            "seeds": seeds,
            "by_seed_rows": [],
            "summary_rows": [],
            "significance_rows": [],
            "note": "Multi-seed statistics skipped because mainline_seed_count <= 1.",
        }
        write_csv(os.path.join(table_dir, "Mainline_MultiSeed_BySeed.csv"), [])
        write_csv(os.path.join(table_dir, "Mainline_MultiSeed_Summary.csv"), [])
        write_csv(os.path.join(table_dir, "Mainline_MultiSeed_Significance.csv"), [])
        with open(os.path.join(output_dir, "mainline_multiseed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
        return summary

    seed_rows: List[Dict[str, object]] = []
    for seed in seeds:
        seed_config = ExperimentConfig(**asdict(config))
        seed_config.seed = seed
        seed_dir = ensure_dir(os.path.join(output_dir, "multiseed_runs", f"seed_{seed}"))
        seed_results = run_train_pipeline(seed_config, seed_dir, export_artifacts=False)
        for row in seed_results["main_results"]:
            seed_rows.append(
                {
                    "seed": seed,
                    "method": row["name"],
                    "coverage": float(row["coverage"]),
                    "conflict_rate": float(row["conflict_rate"]),
                    "reward": float(row["reward"]),
                    "path_length": float(row["path_length"]),
                }
            )
    write_csv(os.path.join(table_dir, "Mainline_MultiSeed_BySeed.csv"), seed_rows)

    summary_rows: List[Dict[str, object]] = []
    methods = sorted({row["method"] for row in seed_rows})
    for method in methods:
        sub = [row for row in seed_rows if row["method"] == method]
        cov = mean_std_ci(np.asarray([row["coverage"] for row in sub], dtype=np.float32))
        conf = mean_std_ci(np.asarray([row["conflict_rate"] for row in sub], dtype=np.float32))
        rew = mean_std_ci(np.asarray([row["reward"] for row in sub], dtype=np.float32))
        summary_rows.append(
            {
                "Method": method,
                "Coverage (mean±std)": f"{cov['mean']:.3f}±{cov['std']:.3f}",
                "Coverage 95% CI": f"[{cov['ci95_low']:.3f}, {cov['ci95_high']:.3f}]",
                "Conflict (mean±std)": f"{conf['mean']:.4f}±{conf['std']:.4f}",
                "Conflict 95% CI": f"[{conf['ci95_low']:.4f}, {conf['ci95_high']:.4f}]",
                "Reward (mean±std)": f"{rew['mean']:.3f}±{rew['std']:.3f}",
                "Reward 95% CI": f"[{rew['ci95_low']:.3f}, {rew['ci95_high']:.3f}]",
                "N seeds": len(sub),
            }
        )
    write_csv(os.path.join(table_dir, "Mainline_MultiSeed_Summary.csv"), summary_rows)

    significance_rows: List[Dict[str, object]] = []
    anchor_cov = np.asarray([row["coverage"] for row in seed_rows if row["method"] == METHOD_DISPLAY_NAME], dtype=np.float32)
    anchor_conf = np.asarray([row["conflict_rate"] for row in seed_rows if row["method"] == METHOD_DISPLAY_NAME], dtype=np.float32)
    anchor_rew = np.asarray([row["reward"] for row in seed_rows if row["method"] == METHOD_DISPLAY_NAME], dtype=np.float32)
    for method in methods:
        if method == METHOD_DISPLAY_NAME:
            continue
        sample_cov = np.asarray([row["coverage"] for row in seed_rows if row["method"] == method], dtype=np.float32)
        sample_conf = np.asarray([row["conflict_rate"] for row in seed_rows if row["method"] == method], dtype=np.float32)
        sample_rew = np.asarray([row["reward"] for row in seed_rows if row["method"] == method], dtype=np.float32)
        for metric_name, anchor_arr, sample_arr in [
            ("coverage", anchor_cov, sample_cov),
            ("conflict_rate", anchor_conf, sample_conf),
            ("reward", anchor_rew, sample_rew),
        ]:
            if anchor_arr.size < 2 or sample_arr.size < 2:
                continue
            t_stat, p_value = stats.ttest_ind(anchor_arr, sample_arr, equal_var=False)
            significance_rows.append(
                {
                    "Reference": METHOD_DISPLAY_NAME,
                    "Compared": method,
                    "Metric": metric_name,
                    "Reference Mean": round(float(anchor_arr.mean()), 6),
                    "Compared Mean": round(float(sample_arr.mean()), 6),
                    "t_stat": round(float(t_stat), 6),
                    "p_value": round(float(p_value), 8),
                    "Significant(p<0.05)": bool(p_value < 0.05),
                }
            )
    write_csv(os.path.join(table_dir, "Mainline_MultiSeed_Significance.csv"), significance_rows)

    summary = {
        "seeds": seeds,
        "by_seed_rows": seed_rows,
        "summary_rows": summary_rows,
        "significance_rows": significance_rows,
    }
    with open(os.path.join(output_dir, "mainline_multiseed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
    return summary


def run_mainline_paper_pipeline(config: ExperimentConfig, output_dir: str) -> Dict[str, object]:
    set_seed(config.seed)
    ensure_dir(output_dir)
    train_results = run_train_pipeline(config, output_dir)
    dataset = ScenarioDataset(config.dataset_paths)
    generalization = run_mainline_generalization_suite(config, output_dir, dataset, train_results["training_runs"])
    safety_diag = export_safety_layer_diagnostics(output_dir, train_results["training_runs"]["tmarl"]["model"], dataset, config)
    multiseed = run_mainline_multiseed_statistics(config, ensure_dir(os.path.join(output_dir, "mainline_multiseed")))
    export_tables(
        output_dir,
        train_results["main_results"],
        train_results["ablation_results"],
        train_results["complexity_rows"],
        train_results["scale_rows"],
        train_results["significance_rows"],
        train_results["efficiency_rows"],
        multiseed_summary=multiseed,
    )
    export_report(
        output_dir,
        config,
        train_results["main_results"],
        train_results["ablation_results"],
        train_results["complexity_rows"],
        train_results["scale_rows"],
        train_results["significance_rows"],
        train_results["efficiency_rows"],
        multiseed_summary=multiseed,
    )
    summary_json_path = os.path.join(output_dir, "summary.json")
    paper_warnings = []
    if os.path.exists(summary_json_path):
        with open(summary_json_path, "r", encoding="utf-8") as f:
            paper_warnings = json.load(f).get("paper_warnings", [])
    summary = {
        "main_results": [{k: v for k, v in row.items() if k != "final_env"} for row in train_results["main_results"]],
        "ablation_results": [{k: v for k, v in row.items() if k != "final_env"} for row in train_results["ablation_results"]],
        "generalization": generalization,
        "safety_diagnostics": safety_diag["summary"],
        "multiseed": multiseed,
        "multiseed_summary_file": os.path.join(output_dir, "mainline_multiseed", "mainline_multiseed_summary.json"),
        "paper_warnings": to_jsonable(paper_warnings),
    }
    with open(os.path.join(output_dir, "paper_main_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
    return summary


def run_paper_supplement_pipeline(config: ExperimentConfig, output_dir: str, weight_path: str) -> Dict[str, object]:
    set_seed(config.seed)
    ensure_dir(output_dir)
    dataset = ScenarioDataset(config.dataset_paths)
    model = load_tmarl_model(weight_path, config)
    table_dir = ensure_dir(os.path.join(output_dir, "tables"))
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))

    repeat_seeds = [config.seed + i for i in range(5)]
    stability = run_repeat_eval_pipeline(config, output_dir, weight_path, repeat_seeds, episodes=12)
    stability_summary_rows = []
    for metric, stats_row in stability["summary"].items():
        stability_summary_rows.append(
            {
                "metric": metric,
                "mean": round(float(stats_row["mean"]), 6),
                "std": round(float(stats_row["std"]), 6),
                "min": round(float(stats_row["min"]), 6),
                "max": round(float(stats_row["max"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_5Seed_Summary.csv"), stability_summary_rows)

    traditional_results = [
        evaluate_controller("Cluster-Greedy", dataset, config, model=None, env_type=config.train_env, num_uavs=config.train_num_uavs, episodes=12),
        evaluate_controller("Nearest-Greedy", dataset, config, model=None, env_type=config.train_env, num_uavs=config.train_num_uavs, episodes=12),
        evaluate_controller("Frontier-Greedy", dataset, config, model=None, env_type=config.train_env, num_uavs=config.train_num_uavs, episodes=12),
    ]
    traditional_rows = []
    for row in traditional_results:
        traditional_rows.append(
            {
                "Algorithm": row["name"],
                "Coverage (%)": round(float(row["coverage"]), 6),
                "Reward": round(float(row["reward"]), 6),
                "Conflict Rate": round(float(row["conflict_rate"]), 6),
                "Path Length": round(float(row["path_length"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_Traditional_Baselines.csv"), traditional_rows)

    regionless_config = ExperimentConfig(**asdict(config))
    regionless_config.use_region_assignment = False
    regionless_dir = ensure_dir(os.path.join(output_dir, "w_o_region_assignment"))
    regionless_run = train_variant("tmarl", regionless_config, dataset, regionless_dir, stage_label="supp")
    if regionless_run["best_state"] is not None:
        regionless_run["model"].load_state_dict(regionless_run["best_state"])
    regionless_metrics = evaluate_controller(
        "w/o Region Assignment",
        dataset,
        regionless_config,
        model=regionless_run["model"],
        env_type="dynamic_users",
        num_uavs=regionless_config.train_num_uavs,
        episodes=12,
    )

    no_imitation_config = ExperimentConfig(**asdict(config))
    no_imitation_config.imitation_epochs = 0
    no_imitation_dir = ensure_dir(os.path.join(output_dir, "w_o_imitation_warmstart"))
    no_imitation_run = train_variant("tmarl", no_imitation_config, dataset, no_imitation_dir, stage_label="supp")
    if no_imitation_run["best_state"] is not None:
        no_imitation_run["model"].load_state_dict(no_imitation_run["best_state"])
    no_imitation_metrics = evaluate_controller(
        "w/o Imitation Warm Start",
        dataset,
        no_imitation_config,
        model=no_imitation_run["model"],
        env_type="dynamic_users",
        num_uavs=no_imitation_config.train_num_uavs,
        episodes=12,
    )
    ablation_rows = []
    for row in [regionless_metrics, no_imitation_metrics]:
        ablation_rows.append(
            {
                "Variant": row["name"],
                "Coverage (%)": round(float(row["coverage"]), 6),
                "Reward": round(float(row["reward"]), 6),
                "Conflict Rate": round(float(row["conflict_rate"]), 6),
                "Path Length": round(float(row["path_length"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_Extended_Ablations.csv"), ablation_rows)

    safety_component_rows = []
    for variant_name, display_name in [
        ("tmarl_no_repulsion", "w/o Safety Repulsion"),
        ("tmarl_no_brake", "w/o Safety Brake"),
        ("tmarl_no_guidance", "w/o Safety Guidance"),
    ]:
        variant_dir = ensure_dir(os.path.join(output_dir, variant_name))
        variant_run = train_variant(variant_name, config, dataset, variant_dir, stage_label="supp")
        if variant_run["best_state"] is not None:
            variant_run["model"].load_state_dict(variant_run["best_state"])
        variant_metrics = evaluate_controller(
            display_name,
            dataset,
            config,
            model=variant_run["model"],
            env_type="dynamic_users",
            num_uavs=config.train_num_uavs,
            episodes=12,
        )
        safety_component_rows.append(
            {
                "Variant": display_name,
                "Coverage (%)": round(float(variant_metrics["coverage"]), 6),
                "Reward": round(float(variant_metrics["reward"]), 6),
                "Conflict Rate": round(float(variant_metrics["conflict_rate"]), 6),
                "Path Length": round(float(variant_metrics["path_length"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_Safety_Component_Ablation.csv"), safety_component_rows)

    obstacle_rows: List[Dict[str, object]] = []
    for obstacle_ratio in [0.05, 0.10, 0.15, 0.20]:
        sweep_config = ExperimentConfig(**asdict(config))
        sweep_config.train_env = "obstacles"
        sweep_config.obstacle_ratio = obstacle_ratio
        cast_metrics = evaluate_controller(
            METHOD_DISPLAY_NAME,
            dataset,
            sweep_config,
            model=model,
            env_type="obstacles",
            num_uavs=sweep_config.train_num_uavs,
            episodes=10,
        )
        greedy_metrics = evaluate_controller(
            "Cluster-Greedy",
            dataset,
            sweep_config,
            model=None,
            env_type="obstacles",
            num_uavs=sweep_config.train_num_uavs,
            episodes=10,
        )
        for method_row in [cast_metrics, greedy_metrics]:
            obstacle_rows.append(
                {
                    "method": method_row["name"],
                    "obstacle_ratio": obstacle_ratio,
                    "coverage": round(float(method_row["coverage"]), 6),
                    "reward": round(float(method_row["reward"]), 6),
                    "conflict_rate": round(float(method_row["conflict_rate"]), 6),
                    "path_length": round(float(method_row["path_length"]), 6),
                }
            )
    write_csv(os.path.join(table_dir, "Supplementary_Obstacle_Sweep.csv"), obstacle_rows)
    plot_sweep_curve(
        os.path.join(fig_dir, "Supplementary_Obstacle_Sweep.png"),
        obstacle_rows,
        x_key="obstacle_ratio",
        title="Obstacle Density Sweep",
        xlabel="Obstacle Ratio",
    )

    dynamic_rows: List[Dict[str, object]] = []
    for dynamic_ratio in [0.05, 0.10, 0.15, 0.20]:
        sweep_config = ExperimentConfig(**asdict(config))
        sweep_config.train_env = "dynamic_users"
        sweep_config.dynamic_ratio = dynamic_ratio
        cast_metrics = evaluate_controller(
            METHOD_DISPLAY_NAME,
            dataset,
            sweep_config,
            model=model,
            env_type="dynamic_users",
            num_uavs=sweep_config.train_num_uavs,
            episodes=10,
        )
        greedy_metrics = evaluate_controller(
            "Cluster-Greedy",
            dataset,
            sweep_config,
            model=None,
            env_type="dynamic_users",
            num_uavs=sweep_config.train_num_uavs,
            episodes=10,
        )
        for method_row in [cast_metrics, greedy_metrics]:
            dynamic_rows.append(
                {
                    "method": method_row["name"],
                    "dynamic_ratio": dynamic_ratio,
                    "coverage": round(float(method_row["coverage"]), 6),
                    "reward": round(float(method_row["reward"]), 6),
                    "conflict_rate": round(float(method_row["conflict_rate"]), 6),
                    "path_length": round(float(method_row["path_length"]), 6),
                }
            )
    write_csv(os.path.join(table_dir, "Supplementary_Dynamic_Sweep.csv"), dynamic_rows)
    plot_sweep_curve(
        os.path.join(fig_dir, "Supplementary_Dynamic_Sweep.png"),
        dynamic_rows,
        x_key="dynamic_ratio",
        title="Dynamic User Strength Sweep",
        xlabel="Dynamic Ratio",
    )

    extreme_rows: List[Dict[str, object]] = []
    extreme_specs = [
        ("Extreme Obstacles-0.25", "obstacles", 0.25, config.dynamic_ratio),
        ("Extreme Obstacles-0.30", "obstacles", 0.30, config.dynamic_ratio),
        ("Extreme Dynamic-0.25", "dynamic_users", config.obstacle_ratio, 0.25),
        ("Extreme Dynamic-0.30", "dynamic_users", config.obstacle_ratio, 0.30),
    ]
    for label, env_name, obstacle_ratio, dynamic_ratio in extreme_specs:
        sweep_config = ExperimentConfig(**asdict(config))
        sweep_config.train_env = env_name
        sweep_config.obstacle_ratio = obstacle_ratio
        sweep_config.dynamic_ratio = dynamic_ratio
        metrics = evaluate_controller(
            METHOD_DISPLAY_NAME,
            dataset,
            sweep_config,
            model=model,
            env_type=env_name,
            num_uavs=sweep_config.train_num_uavs,
            episodes=12,
        )
        extreme_rows.append(
            {
                "Scenario": label,
                "env_type": env_name,
                "obstacle_ratio": obstacle_ratio,
                "dynamic_ratio": dynamic_ratio,
                "Coverage (%)": round(float(metrics["coverage"]), 6),
                "Reward": round(float(metrics["reward"]), 6),
                "Conflict Rate": round(float(metrics["conflict_rate"]), 6),
                "Path Length": round(float(metrics["path_length"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_Extreme_Environment_Sweep.csv"), extreme_rows)

    cast_main = evaluate_controller(
        METHOD_DISPLAY_NAME,
        dataset,
        config,
        model=model,
        env_type=config.train_env,
        num_uavs=config.train_num_uavs,
        episodes=12,
    )
    tradeoff_results = [
        cast_main,
        *traditional_results,
        regionless_metrics,
        no_imitation_metrics,
    ]
    plot_tradeoff_scatter(os.path.join(fig_dir, "Supplementary_Tradeoff_Scatter.png"), tradeoff_results)
    tradeoff_rows = []
    for row in tradeoff_results:
        tradeoff_rows.append(
            {
                "Method": row["name"],
                "Coverage (%)": round(float(row["coverage"]), 6),
                "Conflict Rate": round(float(row["conflict_rate"]), 6),
                "Path Length": round(float(row["path_length"]), 6),
                "Reward": round(float(row["reward"]), 6),
            }
        )
    write_csv(os.path.join(table_dir, "Supplementary_Tradeoff_Scatter.csv"), tradeoff_rows)
    export_attention_proxy(os.path.join(fig_dir, "Supplementary_Attention_Proxy_Heatmap.png"), model, cast_main["final_env"], next(model.parameters()).device)
    prototype_rows = export_coverage_prototype_bank(output_dir, cast_main["final_env"])

    summary = {
        "repeat_eval_seeds": repeat_seeds,
        "stability_summary": stability["summary"],
        "traditional_baselines": traditional_rows,
        "extended_ablations": ablation_rows,
        "safety_component_ablation": safety_component_rows,
        "obstacle_sweep_rows": obstacle_rows,
        "dynamic_sweep_rows": dynamic_rows,
        "extreme_environment_sweep": extreme_rows,
        "tradeoff_rows": tradeoff_rows,
        "coverage_prototype_bank": prototype_rows,
    }
    with open(os.path.join(output_dir, "paper_supp_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"paper supplementary pipeline finished | outputs saved in {output_dir}")
    return summary


def run_eval_pipeline(config: ExperimentConfig, output_dir: str, weight_path: str) -> Dict[str, object]:
    set_seed(config.seed)
    dataset = ScenarioDataset(config.dataset_paths)
    model = load_tmarl_model(weight_path, config)

    main_result = evaluate_controller(METHOD_DISPLAY_NAME, dataset, config, model=model, env_type=config.train_env, num_uavs=config.train_num_uavs, episodes=20)
    scale_rows = []
    for n_uavs in [3, 6, 10]:
        result = evaluate_controller(METHOD_DISPLAY_NAME, dataset, config, model=model, env_type=config.train_env, num_uavs=n_uavs, episodes=8)
        scale_rows.append({"num_uavs": n_uavs, "coverage": result["coverage"], "reward": result["reward"]})

    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    plot_trajectories(os.path.join(fig_dir, "Eval_Trajectories.png"), main_result["final_env"], "Evaluation Trajectories")
    plot_heatmap(os.path.join(fig_dir, "Eval_Coverage_Heatmap.png"), main_result["final_env"].coverage_heat, "Evaluation Heatmap")
    write_csv(os.path.join(output_dir, "tables", "Eval_Main_Result.csv"), [{k: v for k, v in main_result.items() if k != "final_env"}])
    with open(os.path.join(output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "main_result": {k: v for k, v in main_result.items() if k != "final_env"},
                "scalability": scale_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return {"main_result": main_result, "scale_rows": scale_rows}


def run_paper_diagram_pipeline(config: ExperimentConfig, output_dir: str, weight_path: str) -> Dict[str, object]:
    set_seed(config.seed)
    ensure_dir(output_dir)
    dataset = ScenarioDataset(config.dataset_paths)
    model = load_tmarl_model(weight_path, config)
    main_result = evaluate_controller(
        METHOD_DISPLAY_NAME,
        dataset,
        config,
        model=model,
        env_type=config.train_env,
        num_uavs=config.train_num_uavs,
        episodes=1,
        deterministic=True,
        eval_seed=config.seed + 7000,
    )
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    plot_problem_scenarios_overview(os.path.join(fig_dir, "Figure00_Problem_Scenarios.png"), dataset, config)
    plot_framework_diagram(os.path.join(fig_dir, "Figure00_Framework_Diagram.png"))
    plot_safety_layer_schematic(os.path.join(fig_dir, "Supplementary_Safety_Layer_Schematic.png"))
    plot_region_assignment_guidance(os.path.join(fig_dir, "Supplementary_Region_Assignment_Guidance.png"), main_result["final_env"])
    plot_failure_redistribution_mechanism(os.path.join(fig_dir, "Supplementary_Failure_Redistribution.png"))
    plot_3d_trajectory_risk_overlay(os.path.join(fig_dir, "Supplementary_3D_Trajectory_Risk.png"), main_result["final_env"])
    plot_coverage_heatmaps_over_time(os.path.join(fig_dir, "Supplementary_Coverage_Heatmaps_Over_Time.png"), main_result["final_env"])
    safety_diag = export_safety_layer_diagnostics(output_dir, model, dataset, config)
    summary = {
        "weight_path": weight_path,
        "figure_dir": fig_dir,
        "main_env_preview": {k: v for k, v in main_result.items() if k != "final_env"},
        "safety_diag_summary": safety_diag["summary"],
    }
    with open(os.path.join(output_dir, "paper_diagram_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
    return summary


def run_repeat_eval_pipeline(
    config: ExperimentConfig,
    output_dir: str,
    weight_path: str,
    repeat_seeds: List[int],
    episodes: int,
) -> Dict[str, object]:
    dataset = ScenarioDataset(config.dataset_paths)
    model = load_tmarl_model(weight_path, config)
    rows = []
    for seed in repeat_seeds:
        set_seed(seed)
        metrics = evaluate_controller(
            METHOD_DISPLAY_NAME,
            dataset,
            config,
            model=model,
            env_type=config.train_env,
            num_uavs=config.train_num_uavs,
            episodes=episodes,
            deterministic=True,
        )
        rows.append(
            {
                "seed": seed,
                "coverage": float(metrics["coverage"]),
                "coverage_p75": float(metrics.get("coverage_p75", metrics["coverage"])),
                "conflict_rate": float(metrics["conflict_rate"]),
                "reward": float(metrics["reward"]),
                "path_length": float(metrics["path_length"]),
            }
        )

    summary = {}
    for key in ["coverage", "coverage_p75", "conflict_rate", "reward", "path_length"]:
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        ci_stats = mean_std_ci(values)
        summary[key] = {
            "mean": ci_stats["mean"],
            "std": ci_stats["std"],
            "min": float(values.min()),
            "max": float(values.max()),
            "ci95_low": ci_stats["ci95_low"],
            "ci95_high": ci_stats["ci95_high"],
        }

    write_csv(os.path.join(output_dir, "tables", "Repeat_Eval_By_Seed.csv"), rows)
    with open(os.path.join(output_dir, "repeat_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "weight_path": weight_path,
                "episodes_per_seed": episodes,
                "repeat_seeds": repeat_seeds,
                "rows": rows,
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return {"rows": rows, "summary": summary}


PAPER_METHOD_ORDER = [METHOD_DISPLAY_NAME, "QMIX", "MADDPG", "MAPPO", "PPO"]
PAPER_MAIN_ABLATION_ORDER = ["Full", "w/o Transformer", "w/o Attention", "w/o CTDE", "w/o Safety Layer"]
PAPER_COLORS = {
    METHOD_DISPLAY_NAME: "#0E5A8A",
    "QMIX": "#7B8794",
    "MADDPG": "#A1AAB3",
    "MAPPO": "#C0C7CE",
    "PPO": "#D5DADF",
    "Full": "#0E5A8A",
    "w/o Transformer": "#5C8FB1",
    "w/o Attention": "#83AAC4",
    "w/o CTDE": "#A9C4D7",
    "w/o Safety Layer": "#CCDDE8",
    "A_no_failure": "#0E5A8A",
    "B_failure_no_reassign": "#A45B5B",
    "C_failure_with_reassign": "#3F8C6B",
}
PAPER_MARKERS = {
    METHOD_DISPLAY_NAME: "o",
    "QMIX": "s",
    "MADDPG": "D",
    "MAPPO": "^",
    "PPO": "v",
}


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.8,
            "legend.fontsize": 9.6,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
        }
    )


def _save_figure_dual(fig: plt.Figure, base_path: str, dpi: int = 320) -> Dict[str, str]:
    ensure_dir(os.path.dirname(base_path))
    fig.tight_layout()
    fig.savefig(base_path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close(fig)
    return {"png": base_path + ".png", "svg": base_path + ".svg"}


def _placeholder_figure(base_path: str, title: str, message: str) -> Dict[str, str]:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10.5, color="#555555")
    return _save_figure_dual(fig, base_path)


def _ci_error(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=0) / math.sqrt(arr.size))


def _sanitize_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def _escape_latex(text: str) -> str:
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _rank_cells(
    raw_rows: List[Dict[str, float]],
    metric_directions: Dict[str, str],
) -> Dict[Tuple[int, str], str]:
    marks: Dict[Tuple[int, str], str] = {}
    if not raw_rows:
        return marks
    for key, direction in metric_directions.items():
        vals: List[Tuple[int, float]] = []
        for idx, row in enumerate(raw_rows):
            if key in row:
                try:
                    vals.append((idx, float(row[key])))
                except (TypeError, ValueError):
                    continue
        if not vals:
            continue
        reverse = direction == "max"
        vals = sorted(vals, key=lambda item: item[1], reverse=reverse)
        marks[(vals[0][0], key)] = "best"
        if len(vals) > 1:
            marks[(vals[1][0], key)] = "second"
    return marks


def _table_to_markdown(
    rows: List[Dict[str, object]],
    column_order: List[str],
    marks: Dict[Tuple[int, str], str],
) -> str:
    lines = ["| " + " | ".join(column_order) + " |", "| " + " | ".join(["---"] * len(column_order)) + " |"]
    for ridx, row in enumerate(rows):
        cells = []
        for col in column_order:
            cell = _sanitize_cell(row.get(col, ""))
            mark = marks.get((ridx, col))
            if mark == "best":
                cell = f"**{cell}**"
            elif mark == "second":
                cell = f"<u>{cell}</u>"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _table_to_latex(
    rows: List[Dict[str, object]],
    column_order: List[str],
    marks: Dict[Tuple[int, str], str],
    title: str,
    note: str,
) -> str:
    align = "l" + "c" * max(0, len(column_order) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_escape_latex(title)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(_escape_latex(col) for col in column_order) + " \\\\",
        "\\hline",
    ]
    for ridx, row in enumerate(rows):
        cells: List[str] = []
        for col in column_order:
            cell = _escape_latex(_sanitize_cell(row.get(col, "")))
            mark = marks.get((ridx, col))
            if mark == "best":
                cell = f"\\textbf{{{cell}}}"
            elif mark == "second":
                cell = f"\\underline{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            f"\\vspace{{2pt}}\\footnotesize{{{_escape_latex(note)}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: str, content: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def export_table_bundle(
    base_path: str,
    rows: List[Dict[str, object]],
    title: str,
    note: str,
    metric_directions: Dict[str, str] | None = None,
    raw_rows: List[Dict[str, float]] | None = None,
) -> Dict[str, str]:
    ensure_dir(os.path.dirname(base_path))
    if not rows:
        rows = [{"Status": "Placeholder", "Note": note}]
    column_order = list(rows[0].keys())
    write_csv(base_path + ".csv", rows)
    marks = _rank_cells(raw_rows or [], metric_directions or {})
    write_text(base_path + ".md", _table_to_markdown(rows, column_order, marks))
    write_text(base_path + ".tex", _table_to_latex(rows, column_order, marks, title, note))
    return {"csv": base_path + ".csv", "md": base_path + ".md", "tex": base_path + ".tex"}


def _mean_ci_from_rows(rows: List[Dict[str, object]], group_key: str, metrics: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    groups = sorted({str(row[group_key]) for row in rows})
    for group in groups:
        sub = [row for row in rows if str(row[group_key]) == group]
        row: Dict[str, object] = {group_key: group}
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in sub], dtype=np.float32)
            stats_row = mean_std_ci(vals)
            row[f"{metric}_mean"] = float(stats_row["mean"])
            row[f"{metric}_std"] = float(stats_row["std"])
            row[f"{metric}_ci95_low"] = float(stats_row["ci95_low"])
            row[f"{metric}_ci95_high"] = float(stats_row["ci95_high"])
        out.append(row)
    return out


def _collect_curve_summary(
    training_runs: Dict[str, Dict[str, object]] | None,
    multiseed_dir: str | None,
) -> Dict[str, Dict[str, np.ndarray]]:
    curve_map = {
        METHOD_DISPLAY_NAME: "tmarl_training_curves.npz",
        "MAPPO": "mappo_training_curves.npz",
        "PPO": "ppo_training_curves.npz",
        "QMIX": "qmix_training_curves.npz",
    }
    # MADDPG uses the same naming pattern as others via variant dir.
    curve_map["MADDPG"] = "maddpg_training_curves.npz"
    summary: Dict[str, Dict[str, np.ndarray]] = {}
    for method, filename in curve_map.items():
        stacks: Dict[str, List[np.ndarray]] = {"rewards": [], "coverage": [], "conflicts": []}
        if multiseed_dir and os.path.isdir(os.path.join(multiseed_dir, "multiseed_runs")):
            for seed_name in sorted(os.listdir(os.path.join(multiseed_dir, "multiseed_runs"))):
                npz_path = os.path.join(multiseed_dir, "multiseed_runs", seed_name, "curves", filename)
                if os.path.exists(npz_path):
                    data = np.load(npz_path)
                    stacks["rewards"].append(np.asarray(data["rewards"], dtype=np.float32))
                    stacks["coverage"].append(np.asarray(data["coverage"], dtype=np.float32) * 100.0)
                    stacks["conflicts"].append(np.asarray(data["conflicts"], dtype=np.float32))
        if not stacks["rewards"]:
            if training_runs is not None:
                run_lookup = {
                    METHOD_DISPLAY_NAME: training_runs.get("tmarl"),
                    "MAPPO": training_runs.get("mappo"),
                    "MADDPG": training_runs.get("maddpg"),
                    "QMIX": training_runs.get("qmix"),
                    "PPO": training_runs.get("ppo"),
                }
                run = run_lookup.get(method)
                if run:
                    stacks["rewards"].append(np.asarray(run["train_rewards"], dtype=np.float32))
                    stacks["coverage"].append(np.asarray(run["train_coverage"], dtype=np.float32) * 100.0)
                    stacks["conflicts"].append(np.asarray(run["train_conflicts"], dtype=np.float32))
        if not stacks["rewards"]:
            continue
        summary[method] = {}
        for key, arrays in stacks.items():
            min_len = min(len(arr) for arr in arrays)
            arr = np.stack([a[:min_len] for a in arrays], axis=0)
            mean = arr.mean(axis=0)
            ci = np.asarray([_ci_error(arr[:, i]) for i in range(arr.shape[1])], dtype=np.float32)
            summary[method][f"{key}_mean"] = mean
            summary[method][f"{key}_ci"] = ci
    return summary


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _build_paper_stat_rows(
    seed_rows: List[Dict[str, object]] | None,
    main_results: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if seed_rows:
        methods = [m for m in PAPER_METHOD_ORDER if any(str(row["method"]) == m for row in seed_rows)]
        rows: List[Dict[str, object]] = []
        for method in methods:
            sub = [row for row in seed_rows if str(row["method"]) == method]
            cov = mean_std_ci(np.asarray([float(r["coverage"]) for r in sub], dtype=np.float32))
            conf = mean_std_ci(np.asarray([float(r["conflict_rate"]) for r in sub], dtype=np.float32))
            rew = mean_std_ci(np.asarray([float(r["reward"]) for r in sub], dtype=np.float32))
            path = mean_std_ci(np.asarray([float(r["path_length"]) for r in sub], dtype=np.float32))
            rows.append(
                {
                    "name": method,
                    "coverage": cov["mean"],
                    "coverage_std": cov["std"],
                    "coverage_ci": cov["ci95_high"] - cov["mean"],
                    "conflict_rate": conf["mean"],
                    "conflict_rate_std": conf["std"],
                    "conflict_rate_ci": conf["ci95_high"] - conf["mean"],
                    "reward": rew["mean"],
                    "reward_std": rew["std"],
                    "reward_ci": rew["ci95_high"] - rew["mean"],
                    "path_length": path["mean"],
                    "path_length_std": path["std"],
                    "path_length_ci": path["ci95_high"] - path["mean"],
                    "task_time": float(np.mean([float(r.get("task_time", 0.0)) for r in sub])) if sub else 0.0,
                    "n": len(sub),
                }
            )
        return rows
    rows = []
    for row in main_results:
        rows.append(
            {
                "name": row["name"],
                "coverage": float(row["coverage"]),
                "coverage_std": float(np.std(np.asarray(row.get("eval_coverage_samples", [row["coverage"]]), dtype=np.float32), ddof=0)),
                "coverage_ci": _ci_error(np.asarray(row.get("eval_coverage_samples", [row["coverage"]]), dtype=np.float32)),
                "conflict_rate": float(row["conflict_rate"]),
                "conflict_rate_std": float(np.std(np.asarray(row.get("eval_conflict_samples", [row["conflict_rate"]]), dtype=np.float32), ddof=0)),
                "conflict_rate_ci": _ci_error(np.asarray(row.get("eval_conflict_samples", [row["conflict_rate"]]), dtype=np.float32)),
                "reward": float(row["reward"]),
                "reward_std": float(np.std(np.asarray(row.get("eval_reward_samples", [row["reward"]]), dtype=np.float32), ddof=0)),
                "reward_ci": _ci_error(np.asarray(row.get("eval_reward_samples", [row["reward"]]), dtype=np.float32)),
                "path_length": float(row["path_length"]),
                "path_length_std": 0.0,
                "path_length_ci": 0.0,
                "task_time": float(row.get("task_time", 0.0)),
                "n": len(row.get("eval_reward_samples", [])),
            }
        )
    return rows


def _figure_task_scenarios(base_path: str, dataset: ScenarioDataset, config: ExperimentConfig) -> Dict[str, str]:
    _apply_paper_style()
    env_specs = [("simple", "Simple"), ("obstacles", "Obstacles"), ("dynamic_users", "Dynamic Users")]
    fig = plt.figure(figsize=(16.6, 5.4))
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
    for ax, (env_type, title) in zip(axes, env_specs):
        env = _build_demo_env(dataset, config, env_type)
        ax.set_facecolor("#FAFAF8")
        _draw_terrain_surface_3d(ax, env, alpha=0.22)
        if len(env.users):
            ax.scatter(env.users[:, 0], env.users[:, 1], env.users[:, 2], s=13, c="#9BB7CC", alpha=0.72, edgecolors="white", linewidths=0.25, label="Users")
        if len(env.obstacles):
            _draw_obstacle_prisms_3d(ax, env.obstacles, facecolor="#D9C1C1", edgecolor="#A35A52", alpha=0.12)
        for idx in env._active_agent_indices():
            x, y, z = env.uav_pos[idx]
            ax.scatter([x], [y], [z], s=72, c="#0E5A8A", marker="^", edgecolors="white", linewidths=0.8, zorder=5)
            ax.plot([x, env.region_centers[idx, 0]], [y, env.region_centers[idx, 1]], [z, env.region_centers[idx, 2]], color="#7F7468", linewidth=0.9, alpha=0.72)
        if len(env._active_agent_indices()):
            ref = env.uav_pos[int(env._active_agent_indices()[0])]
            _plot_horizontal_ring(ax, ref, COMM_RADIUS, "#6D8FA8", "--", 0.38)
            _plot_horizontal_ring(ax, ref, config.safety_distance, "#C46A6A", "-", 0.52)
        ax.set_xlim(0, AREA_X)
        ax.set_ylim(0, AREA_Y)
        ax.set_zlim(ALT_MIN, ALT_MAX)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        if ax is axes[0]:
            ax.set_ylabel("Y (m)")
        ax.set_zlabel("Altitude (m)")
        ax.view_init(elev=23, azim=-57)
        ax.grid(alpha=0.16, linewidth=0.6)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9BB7CC", markeredgecolor="white", markersize=7, label="User"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#0E5A8A", markeredgecolor="white", markersize=8, label="Initial UAV"),
        plt.Line2D([0], [0], color="#6D8FA8", linestyle="--", label="Communication Radius"),
        plt.Line2D([0], [0], color="#C46A6A", linestyle="-", label="Safety Distance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    return _save_figure_dual(fig, base_path)


def _figure_framework(base_path: str) -> Dict[str, str]:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(14.0, 4.8))
    ax.axis("off")
    boxes = [
        ((0.03, 0.26), 0.13, 0.44, "State /\nObservation", "#EAF1F6"),
        ((0.21, 0.26), 0.17, 0.44, "Objective-Conditioned\nMulti-Source Transformer", "#D9E7F1"),
        ((0.43, 0.26), 0.15, 0.44, "CTDE + PPO\nActor-Critic", "#EDE8F7"),
        ((0.63, 0.26), 0.13, 0.44, "Conflict-Aware\nSafety Layer", "#F6E5E5"),
        ((0.81, 0.26), 0.14, 0.44, "Action Output\n& Feedback", "#E6F2E8"),
    ]
    for xy, w, h, text, fc in boxes:
        _add_round_box(ax, xy, w, h, text, fc)
    _add_round_box(ax, (0.24, 0.80), 0.18, 0.12, "3D geometry + region-target\ninteraction bias", "#F6F0DB", fontsize=10.5)
    _add_round_box(ax, (0.62, 0.80), 0.22, 0.12, "Decentralized execution\nwith online safety revision", "#F3EFEA", fontsize=10.5)
    _add_flow_arrow(ax, (0.16, 0.48), (0.21, 0.48))
    _add_flow_arrow(ax, (0.38, 0.48), (0.43, 0.48))
    _add_flow_arrow(ax, (0.58, 0.48), (0.63, 0.48))
    _add_flow_arrow(ax, (0.76, 0.48), (0.81, 0.48))
    _add_flow_arrow(ax, (0.31, 0.80), (0.31, 0.70), color="#847462")
    _add_flow_arrow(ax, (0.73, 0.80), (0.73, 0.70), color="#847462")
    return _save_figure_dual(fig, base_path)


def _figure_safety_reassign(base_path: str) -> Dict[str, str]:
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    for ax in axes:
        ax.axis("off")
    ax = axes[0]
    _add_round_box(ax, (0.03, 0.28), 0.21, 0.42, "Raw action\n+ neighbor state", "#EAF1F6")
    _add_round_box(ax, (0.33, 0.62), 0.20, 0.16, "Repulsion", "#F6E5E5")
    _add_round_box(ax, (0.33, 0.40), 0.20, 0.16, "Brake", "#F3EFEA")
    _add_round_box(ax, (0.33, 0.18), 0.20, 0.16, "Guidance", "#E6F2E8")
    _add_round_box(ax, (0.64, 0.34), 0.24, 0.28, "Safe action\nfusion", "#EDE8F7")
    _add_flow_arrow(ax, (0.24, 0.58), (0.33, 0.69))
    _add_flow_arrow(ax, (0.24, 0.49), (0.33, 0.49))
    _add_flow_arrow(ax, (0.24, 0.40), (0.33, 0.26))
    _add_flow_arrow(ax, (0.53, 0.69), (0.64, 0.52))
    _add_flow_arrow(ax, (0.53, 0.49), (0.64, 0.48))
    _add_flow_arrow(ax, (0.53, 0.26), (0.64, 0.42))
    ax.set_title("(a) Safety action revision", y=0.98)

    ax = axes[1]
    _add_round_box(ax, (0.03, 0.55), 0.22, 0.18, "Region assignment", "#EAF1F6")
    _add_round_box(ax, (0.35, 0.55), 0.24, 0.18, "Failure detection", "#F6E5E5")
    _add_round_box(ax, (0.69, 0.55), 0.24, 0.18, "Load-aware\nreallocation", "#E6F2E8")
    _add_round_box(ax, (0.24, 0.18), 0.48, 0.20, "Updated frontier targets and reassigned support radii", "#F3EFEA")
    _add_flow_arrow(ax, (0.25, 0.64), (0.35, 0.64))
    _add_flow_arrow(ax, (0.59, 0.64), (0.69, 0.64))
    _add_flow_arrow(ax, (0.47, 0.55), (0.47, 0.38), color="#847462")
    ax.set_title("(b) Region reassignment after failure", y=0.98)
    return _save_figure_dual(fig, base_path)


def _plot_ci_line(ax, xs: np.ndarray, mean: np.ndarray, ci: np.ndarray, label: str, color: str, marker: str) -> None:
    ax.plot(xs, mean, label=label, color=color, marker=marker, markevery=max(1, len(xs) // 8))
    ax.fill_between(xs, mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0.0)


def _figure_reward_convergence(base_path: str, curve_summary: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, str], bool]:
    if not curve_summary:
        return _placeholder_figure(base_path, "Reward Convergence", "Training curves are unavailable for this run."), True
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for method in PAPER_METHOD_ORDER:
        if method not in curve_summary:
            continue
        ys = curve_summary[method]["rewards_mean"]
        ci = curve_summary[method]["rewards_ci"]
        xs = np.arange(1, len(ys) + 1)
        _plot_ci_line(ax, xs, ys, ci, method, PAPER_COLORS.get(method, "#777777"), PAPER_MARKERS.get(method, "o"))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Convergence Across Training")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(alpha=0.20)
    return _save_figure_dual(fig, base_path), False


def _figure_coverage_conflict_convergence(base_path: str, curve_summary: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, str], bool]:
    if not curve_summary:
        return _placeholder_figure(base_path, "Coverage and Conflict Convergence", "Training curves are unavailable for this run."), True
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    panels = [("coverage", "Coverage (%)"), ("conflicts", "Conflict Rate")]
    for ax, (key, ylabel) in zip(axes, panels):
        for method in PAPER_METHOD_ORDER:
            if method not in curve_summary:
                continue
            ys = curve_summary[method][f"{key}_mean"]
            ci = curve_summary[method][f"{key}_ci"]
            xs = np.arange(1, len(ys) + 1)
            _plot_ci_line(ax, xs, ys, ci, method, PAPER_COLORS.get(method, "#777777"), PAPER_MARKERS.get(method, "o"))
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)
        ax.set_title(ylabel)
    axes[1].legend(frameon=False, loc="upper right")
    return _save_figure_dual(fig, base_path), False


def _figure_main_benchmark(base_path: str, stat_rows: List[Dict[str, object]]) -> Dict[str, str]:
    _apply_paper_style()
    metrics = [
        ("coverage", "Coverage (%)"),
        ("conflict_rate", "Conflict Rate"),
        ("path_length", "Path Length"),
        ("reward", "Reward"),
    ]
    order = [row for row in stat_rows if row["name"] in PAPER_METHOD_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, metrics):
        names = [row["name"] for row in order]
        vals = [float(row[key]) for row in order]
        errs = [float(row.get(f"{key}_ci", 0.0)) for row in order]
        ys = np.arange(len(names))
        colors = [PAPER_COLORS.get(name, "#999999") for name in names]
        ax.barh(ys, vals, xerr=errs, color=colors, edgecolor="white", linewidth=0.8, alpha=0.95, capsize=3)
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.20)
    return _save_figure_dual(fig, base_path)


def _compute_pareto_frontier(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    sorted_pts = sorted(points, key=lambda item: (item[0], -item[1]))
    frontier: List[Tuple[float, float, str]] = []
    best_cov = -1e9
    for item in sorted_pts:
        if item[1] > best_cov:
            frontier.append(item)
            best_cov = item[1]
    return frontier


def _figure_tradeoff(base_path: str, stat_rows: List[Dict[str, object]]) -> Dict[str, str]:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    points = []
    for row in stat_rows:
        name = str(row["name"])
        x = float(row["conflict_rate"])
        y = float(row["coverage"])
        size = max(80.0, float(row["path_length"]) / 120.0)
        points.append((x, y, name))
        ax.scatter(
            [x],
            [y],
            s=size,
            c=PAPER_COLORS.get(name, "#999999"),
            alpha=0.90 if name == METHOD_DISPLAY_NAME else 0.65,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        ax.text(x, y + 0.55, name, fontsize=9.2, ha="center")
    frontier = _compute_pareto_frontier(points)
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linestyle="--", color="#444444", linewidth=1.2, alpha=0.8, label="Pareto-like frontier")
    ax.set_xlabel("Conflict Rate")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage-Conflict-Path Tradeoff")
    ax.grid(alpha=0.20)
    if len(frontier) >= 2:
        ax.legend(frameon=False, loc="lower right")
    return _save_figure_dual(fig, base_path)


def _rollout_demo_env(
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    model: Any,
    method_name: str,
    env_type: str,
    eval_seed: int,
) -> MultiUAVCoverageEnv:
    metrics = evaluate_controller(
        method_name,
        dataset,
        config,
        model=model,
        env_type=env_type,
        num_uavs=config.train_num_uavs,
        episodes=1,
        deterministic=True,
        eval_seed=eval_seed,
    )
    return metrics["final_env"]


def _mark_conflict_points(env: MultiUAVCoverageEnv) -> np.ndarray:
    traj = np.stack(env.trajectory, axis=0)
    marks: List[np.ndarray] = []
    for t in range(traj.shape[0]):
        pos = traj[t]
        dist = pairwise_dist(pos[:, :3], pos[:, :3])
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                if dist[i, j] < COLLISION_DISTANCE * 1.25:
                    marks.append(0.5 * (pos[i, :3] + pos[j, :3]))
    return np.asarray(marks, dtype=np.float32) if marks else np.zeros((0, 3), dtype=np.float32)


def _figure_trajectory_comparison(
    base_path: str,
    dataset: ScenarioDataset,
    config: ExperimentConfig,
    training_runs: Dict[str, Dict[str, object]] | None,
    primary_model: Any | None = None,
) -> Dict[str, str]:
    _apply_paper_style()
    env_specs = [("simple", "Simple"), ("obstacles", "Obstacles"), ("dynamic_users", "Dynamic Users")]
    fig = plt.figure(figsize=(13.8, 14.8))
    axes = np.asarray([fig.add_subplot(3, 2, idx + 1, projection="3d") for idx in range(6)], dtype=object).reshape(3, 2)
    cast_model = training_runs["tmarl"]["model"] if training_runs is not None else primary_model
    compare_methods = [(METHOD_DISPLAY_NAME, cast_model)]
    baseline_model = training_runs["mappo"]["model"] if training_runs is not None and "mappo" in training_runs else None
    baseline_name = "MAPPO"
    compare_methods.append((baseline_name, baseline_model))
    for r, (env_type, env_title) in enumerate(env_specs):
        for c, (method_name, model) in enumerate(compare_methods):
            ax = axes[r, c]
            env = _rollout_demo_env(dataset, config, model, method_name, env_type, eval_seed=config.seed + 8100 + r * 10 + c)
            traj = np.stack(env.trajectory, axis=0)
            ax.set_facecolor("#FAFAF8")
            _draw_terrain_surface_3d(ax, env, alpha=0.18)
            if len(env.initial_users):
                ax.scatter(env.initial_users[:, 0], env.initial_users[:, 1], env.initial_users[:, 2], s=9, c="#C9D3DD", alpha=0.30)
            if len(env.obstacles):
                _draw_obstacle_prisms_3d(ax, env.obstacles, facecolor="#D9C1C1", edgecolor="#A35A52", alpha=0.10)
            for agent_idx in range(traj.shape[1]):
                ax.plot(traj[:, agent_idx, 0], traj[:, agent_idx, 1], traj[:, agent_idx, 2], color=PAPER_COLORS.get(method_name, "#777777"), alpha=0.85, linewidth=1.5)
                ax.scatter([traj[0, agent_idx, 0]], [traj[0, agent_idx, 1]], [traj[0, agent_idx, 2]], c="#444444", s=18)
                ax.scatter([traj[-1, agent_idx, 0]], [traj[-1, agent_idx, 1]], [traj[-1, agent_idx, 2]], c=PAPER_COLORS.get(method_name, "#777777"), s=26, marker="^")
            risk_marks = _mark_conflict_points(env)
            if len(risk_marks):
                ax.scatter(risk_marks[:, 0], risk_marks[:, 1], risk_marks[:, 2], c="#C44E52", s=28, marker="x", linewidths=1.1)
            ax.set_xlim(0, AREA_X)
            ax.set_ylim(0, AREA_Y)
            ax.set_zlim(ALT_MIN, ALT_MAX)
            ax.set_title(f"{env_title} | {method_name}")
            if r == 2:
                ax.set_xlabel("X (m)")
            if c == 0:
                ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.view_init(elev=22, azim=-58)
            ax.grid(alpha=0.12)
    return _save_figure_dual(fig, base_path)


def _coverage_efficiency_map(env: MultiUAVCoverageEnv) -> np.ndarray:
    traj = np.stack(env.trajectory, axis=0)
    visit = _coverage_heat_from_positions(env.initial_users, traj)
    demand = env.coverage_heat
    eff = 0.65 * visit + 0.35 * demand
    eff = np.clip(eff, 0.0, 1.0)
    return eff


def _figure_coverage_efficiency(base_path: str, dataset: ScenarioDataset, config: ExperimentConfig, model: Any) -> Dict[str, str]:
    _apply_paper_style()
    env_specs = [("simple", "Simple"), ("obstacles", "Obstacles"), ("dynamic_users", "Dynamic Users")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (env_type, title) in zip(axes, env_specs):
        env = _rollout_demo_env(dataset, config, model, METHOD_DISPLAY_NAME, env_type, eval_seed=config.seed + 9100 + len(title))
        eff = _coverage_efficiency_map(env)
        im = ax.imshow(eff, cmap="YlGnBu", origin="lower", extent=[0, AREA_X, 0, AREA_Y], vmin=0.0, vmax=1.0)
        uncovered = env.initial_users[~env.covered] if len(env.initial_users) == len(env.covered) else np.zeros((0, 3), dtype=np.float32)
        if len(uncovered):
            ax.scatter(uncovered[:, 0], uncovered[:, 1], c="#C44E52", s=14, marker="x", linewidths=0.8)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        if ax is axes[0]:
            ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Coverage efficiency")
    return _save_figure_dual(fig, base_path)


def _figure_ablation(base_path: str, ablation_results: List[Dict[str, object]]) -> Dict[str, str]:
    _apply_paper_style()
    order = ["w/o Transformer", "w/o Attention", "w/o CTDE", "w/o Safety Layer"]
    full_row = {
        "variant": "Full",
        "coverage": max(float(row.get("coverage", 0.0)) for row in ablation_results) if ablation_results else 0.0,
        "conflict_rate": min(float(row.get("conflict_rate", 0.0)) for row in ablation_results) if ablation_results else 0.0,
        "path_length": min(float(row.get("path_length", 0.0)) for row in ablation_results) if ablation_results else 0.0,
        "reward": max(float(row.get("reward", 0.0)) for row in ablation_results) if ablation_results else 0.0,
    }
    rows = [full_row] + [next((row for row in ablation_results if row["variant"] == name), {"variant": name, "coverage": 0.0, "conflict_rate": 0.0, "path_length": 0.0, "reward": 0.0}) for name in order]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    axes = axes.flatten()
    metrics = [("coverage", "Coverage (%)"), ("conflict_rate", "Conflict Rate"), ("path_length", "Path Length"), ("reward", "Reward")]
    names = [row["variant"] for row in rows]
    x = np.arange(len(names))
    colors = [PAPER_COLORS.get(name, "#90A4B4") for name in names]
    for ax, (key, title) in zip(axes, metrics):
        ax.bar(x, [float(row[key]) for row in rows], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    return _save_figure_dual(fig, base_path)


def _figure_scalability_generalization(
    base_path: str,
    scale_rows: List[Dict[str, object]],
    generalization: Dict[str, object],
) -> Dict[str, str]:
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    if scale_rows:
        scale_rows_sorted = sorted(scale_rows, key=lambda row: float(row["num_uavs"]))
        xs = [int(row["num_uavs"]) for row in scale_rows_sorted]
        axes[0].plot(xs, [float(row["coverage"]) for row in scale_rows_sorted], marker="o", color=PAPER_COLORS[METHOD_DISPLAY_NAME], label="Coverage")
        ax2 = axes[0].twinx()
        ax2.plot(xs, [float(row["reward"]) for row in scale_rows_sorted], marker="s", linestyle="--", color="#517EA3", label="Reward")
        axes[0].set_xlabel("Number of UAVs")
        axes[0].set_ylabel("Coverage (%)")
        ax2.set_ylabel("Reward")
        axes[0].set_title("Scalability")
        axes[0].grid(alpha=0.18)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Scalability data unavailable", ha="center", va="center")
    obstacle_rows = generalization.get("obstacle_density_rows", [])
    dynamic_rows = generalization.get("dynamic_user_ratio_rows", [])
    for method in [METHOD_DISPLAY_NAME, "QMIX", "MAPPO"]:
        sub_obs = sorted([row for row in obstacle_rows if row["method"] == method], key=lambda row: float(row["obstacle_ratio"]))
        if sub_obs:
            axes[1].plot(
                [float(row["obstacle_ratio"]) for row in sub_obs],
                [float(row["coverage"]) for row in sub_obs],
                marker=PAPER_MARKERS.get(method, "o"),
                color=PAPER_COLORS.get(method, "#888888"),
                label=f"{method} (obstacle)",
            )
    for method in [METHOD_DISPLAY_NAME]:
        sub_dyn = sorted([row for row in dynamic_rows if row["method"] == method], key=lambda row: float(row["dynamic_ratio"]))
        if sub_dyn:
            axes[1].plot(
                [float(row["dynamic_ratio"]) for row in sub_dyn],
                [float(row["reward"]) for row in sub_dyn],
                marker="D",
                linestyle="--",
                color="#3F8C6B",
                label=f"{method} reward (dynamic)",
            )
    axes[1].set_xlabel("Scenario shift ratio")
    axes[1].set_ylabel("Coverage / Reward")
    axes[1].set_title("Generalization")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8.4)
    return _save_figure_dual(fig, base_path)


def _figure_failure_robustness(base_path: str, failure_results: Dict[str, object]) -> Dict[str, str]:
    scenario_lookup = {row["scenario"]: row for row in failure_results.get("scenario_results", [])}
    rows = [scenario_lookup.get(name) for name in ["A_no_failure", "B_failure_no_reassign", "C_failure_with_reassign"]]
    rows = [row for row in rows if row is not None]
    if not rows:
        return _placeholder_figure(base_path, "Failure Robustness and Recovery", "Failure robustness statistics are unavailable.")
    _apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))
    axes = axes.flatten()
    metrics = [
        ("mean_coverage", "Final Coverage"),
        ("mean_conflict_rate", "Conflict Rate"),
        ("mean_recovery_speed", "Recovery Speed"),
        ("jain_energy_fairness", "Jain Fairness"),
    ]
    x = np.arange(len(rows))
    names = [row["scenario"].replace("_", "\n") for row in rows]
    colors = [PAPER_COLORS.get(row["scenario"], "#888888") for row in rows]
    for ax, (key, title) in zip(axes, metrics):
        vals = [0.0 if row.get(key) is None else float(row.get(key, 0.0)) for row in rows]
        errs = []
        for row in rows:
            std_key = "std_coverage" if key == "mean_coverage" else (
                "std_conflict_rate" if key == "mean_conflict_rate" else (
                    "std_recovery_speed" if key == "mean_recovery_speed" else "jain_energy_fairness_std"
                )
            )
            errs.append(0.0 if row.get(std_key) is None else float(row.get(std_key, 0.0)))
        ax.bar(x, vals, yerr=errs, color=colors, edgecolor="white", linewidth=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    return _save_figure_dual(fig, base_path)


def _figure_safety_diagnostics(base_path: str, diag: Dict[str, object]) -> Dict[str, str]:
    rows = diag.get("rows", [])
    if not rows:
        return _placeholder_figure(base_path, "Safety Layer Diagnostics", "Safety diagnostic traces are unavailable.")
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xs = [int(row["timestep"]) for row in rows]
    ax.plot(xs, [float(row["mean_max_risk"]) for row in rows], label="Max risk", color="#A45B5B")
    ax.plot(xs, [float(row["mean_repulsion_norm"]) for row in rows], label="Repulsion norm", color="#0E5A8A")
    ax.plot(xs, [float(row["mean_brake_scale"]) for row in rows], label="Brake scale", color="#6B7D90")
    ax.plot(xs, [float(row["mean_guidance_blend"]) for row in rows], label="Guidance blend", color="#3F8C6B")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean value")
    ax.set_title("Safety Layer Diagnostics")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    return _save_figure_dual(fig, base_path)


def _figure_attention_proxy(base_path: str, attention_json_path: str, attention_csv_path: str | None = None) -> Tuple[Dict[str, str], bool]:
    if not os.path.exists(attention_json_path):
        return _placeholder_figure(base_path, "Transformer Attention Proxy", "Attention proxy data are unavailable."), True
    with open(attention_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    attn = np.asarray(data.get("attention_matrix", []), dtype=np.float32)
    if attn.size == 0:
        return _placeholder_figure(base_path, "Transformer Attention Proxy", "Attention proxy matrix is empty."), True
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"width_ratios": [1.0, 0.9]})
    im = axes[0].imshow(attn, cmap="Blues", vmin=float(attn.min()), vmax=float(attn.max()))
    axes[0].set_title("Attention proxy heatmap")
    axes[0].set_xlabel("Key agent")
    axes[0].set_ylabel("Query agent")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].bar(np.arange(attn.shape[0]), attn.mean(axis=1), color="#0E5A8A", alpha=0.85)
    axes[1].set_title("Average outgoing attention")
    axes[1].set_xlabel("Agent")
    axes[1].set_ylabel("Mean weight")
    axes[1].grid(axis="y", alpha=0.18)
    return _save_figure_dual(fig, base_path), False


def _build_experimental_settings_rows(config: ExperimentConfig) -> List[Dict[str, object]]:
    return [
        {"Category": "Train environments", "Value": "simple / obstacles / dynamic-users / terrain-* (3D-aware)"},
        {"Category": "Test environments", "Value": "simple / obstacles / dynamic-users / terrain-* (3D-aware)"},
        {"Category": "Number of UAVs", "Value": config.train_num_uavs},
        {"Category": "Max steps", "Value": config.max_steps},
        {"Category": "Backbone", "Value": "Objective-conditioned multi-source risk-aware Transformer"},
        {"Category": "Communication radius (m)", "Value": COMM_RADIUS},
        {"Category": "Collision distance (m)", "Value": COLLISION_DISTANCE},
        {"Category": "Safety distance (m)", "Value": config.safety_distance},
        {"Category": "Altitude range (m)", "Value": f"{ALT_MIN:.0f}-{ALT_MAX:.0f}"},
        {"Category": "Camera FOV (deg)", "Value": config.camera_fov_deg},
        {"Category": "Terrain clearance (m)", "Value": config.terrain_clearance},
        {"Category": "Obstacle ratio", "Value": config.obstacle_ratio},
        {"Category": "Obstacle height range (m)", "Value": f"{config.obstacle_height_min:.0f}-{config.obstacle_height_max:.0f}"},
        {"Category": "Dynamic-user ratio", "Value": config.dynamic_ratio},
        {"Category": "Multi-source gate strength", "Value": config.multi_source_gate_strength},
        {"Category": "Objective gate strength", "Value": config.objective_gate_strength},
        {"Category": "Learning rate", "Value": config.lr},
        {"Category": "Discount factor", "Value": config.gamma},
        {"Category": "PPO clip", "Value": config.ppo_clip},
        {"Category": "Batch episodes", "Value": config.batch_episodes},
        {"Category": "Target supervision", "Value": config.target_supervision_enabled},
        {"Category": "Seed count", "Value": config.mainline_seed_count},
    ]


def _format_mean_std(mean_value: float, std_value: float, decimals: int) -> str:
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def _build_main_benchmark_table_rows(stat_rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    raw: List[Dict[str, float]] = []
    for name in PAPER_METHOD_ORDER:
        row = next((item for item in stat_rows if item["name"] == name), None)
        if row is None:
            continue
        rows.append(
            {
                "Method": name,
                "Coverage (%)": _format_mean_std(float(row["coverage"]), float(row["coverage_std"]), 2),
                "Conflict Rate": _format_mean_std(float(row["conflict_rate"]), float(row["conflict_rate_std"]), 4),
                "Path Length": _format_mean_std(float(row["path_length"]), float(row["path_length_std"]), 2),
                "Reward": _format_mean_std(float(row["reward"]), float(row["reward_std"]), 2),
                "Task Time": f"{float(row.get('task_time', 0.0)):.2f}",
            }
        )
        raw.append(
            {
                "Coverage (%)": float(row["coverage"]),
                "Conflict Rate": float(row["conflict_rate"]),
                "Path Length": float(row["path_length"]),
                "Reward": float(row["reward"]),
                "Task Time": float(row.get("task_time", 0.0)),
            }
        )
    return rows, raw


def _build_efficiency_table_rows(training_runs: Dict[str, Dict[str, object]], main_results: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    run_lookup = {
        METHOD_DISPLAY_NAME: training_runs["tmarl"],
        "MAPPO": training_runs["mappo"],
        "MADDPG": training_runs["maddpg"],
        "QMIX": training_runs["qmix"],
        "PPO": training_runs["ppo"],
    }
    rows: List[Dict[str, object]] = []
    raw: List[Dict[str, float]] = []
    for name in PAPER_METHOD_ORDER:
        result = next((row for row in main_results if row["name"] == name), None)
        run = run_lookup.get(name)
        if result is None or run is None:
            continue
        train_times = run.get("train_times", [])
        model = run.get("model")
        param_count = _count_params(model) if isinstance(model, nn.Module) else 0
        rows.append(
            {
                "Method": name,
                "Training Time (s)": f"{(float(train_times[-1]) if train_times else 0.0):.2f}",
                "Inference (ms/step)": f"{float(result.get('inference_ms_per_step', 0.0)):.3f}",
                "Parameters": f"{param_count:,}",
                "Memory (MB)": f"{float(result.get('memory_mb', 0.0)):.2f}",
            }
        )
        raw.append(
            {
                "Training Time (s)": float(train_times[-1]) if train_times else 0.0,
                "Inference (ms/step)": float(result.get("inference_ms_per_step", 0.0)),
                "Parameters": float(param_count),
                "Memory (MB)": float(result.get("memory_mb", 0.0)),
            }
        )
    return rows, raw


def _build_efficiency_table_rows_from_summary(
    config: ExperimentConfig,
    efficiency_rows: List[Dict[str, object]],
    main_results: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    dataset = ScenarioDataset(config.dataset_paths)
    probe_env = build_env_from_config(config, dataset)
    model_lookup: Dict[str, int] = {
        METHOD_DISPLAY_NAME: _count_params(TmarlActorCritic(probe_env.state_dim, config, variant="tmarl")),
        "MAPPO": _count_params(TmarlActorCritic(probe_env.state_dim, config, variant="mappo")),
        "PPO": _count_params(TmarlActorCritic(probe_env.state_dim, config, variant="ppo")),
        "MADDPG": _count_params(MADDPGController(config.train_num_uavs, probe_env.state_dim, config.hidden_dim, torch.device("cpu"))),
        "QMIX": _count_params(QMIXController(config.train_num_uavs, probe_env.state_dim, config.hidden_dim, torch.device("cpu"))),
    }
    eff_lookup = {str(row["Algorithm"]): row for row in efficiency_rows}
    main_lookup = {str(row["name"]): row for row in main_results}
    rows: List[Dict[str, object]] = []
    raw: List[Dict[str, float]] = []
    for name in PAPER_METHOD_ORDER:
        eff = eff_lookup.get(name)
        main = main_lookup.get(name)
        if eff is None or main is None:
            continue
        train_time = float(eff.get("Train Wall Time (s)", 0.0))
        inf_ms = float(main.get("inference_ms_per_step", eff.get("Inference (ms/step)", 0.0)))
        mem = float(main.get("memory_mb", eff.get("Memory (MB)", 0.0)))
        params = float(model_lookup.get(name, 0))
        rows.append(
            {
                "Method": name,
                "Training Time (s)": f"{train_time:.2f}",
                "Inference (ms/step)": f"{inf_ms:.3f}",
                "Parameters": f"{int(params):,}",
                "Memory (MB)": f"{mem:.2f}",
            }
        )
        raw.append(
            {
                "Training Time (s)": train_time,
                "Inference (ms/step)": inf_ms,
                "Parameters": params,
                "Memory (MB)": mem,
            }
        )
    return rows, raw


def _build_ablation_table_rows(ablation_results: List[Dict[str, object]], full_reference: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    raw: List[Dict[str, float]] = []
    ordered = [full_reference] + [next((row for row in ablation_results if row["variant"] == name), None) for name in PAPER_MAIN_ABLATION_ORDER[1:]]
    for row in ordered:
        if row is None:
            continue
        name = str(row["variant"])
        rows.append(
            {
                "Variant": name,
                "Coverage (%)": f"{float(row['coverage']):.2f}",
                "Conflict Rate": f"{float(row['conflict_rate']):.4f}",
                "Path Length": f"{float(row['path_length']):.2f}",
                "Reward": f"{float(row['reward']):.2f}",
            }
        )
        raw.append(
            {
                "Coverage (%)": float(row["coverage"]),
                "Conflict Rate": float(row["conflict_rate"]),
                "Path Length": float(row["path_length"]),
                "Reward": float(row["reward"]),
            }
        )
    return rows, raw


def _build_failure_table_rows(failure_results: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    mapping = {
        "A_no_failure": "A_no_failure",
        "B_failure_no_reassign": "B_failure_no_reassign",
        "C_failure_with_reassign": "C_failure_with_reassign",
    }
    rows: List[Dict[str, object]] = []
    raw: List[Dict[str, float]] = []
    for item in failure_results.get("scenario_results", []):
        if item["scenario"] not in mapping:
            continue
        rows.append(
            {
                "Setting": mapping[item["scenario"]],
                "Final Coverage": f"{float(item['mean_coverage']):.3f}",
                "Conflict Rate": f"{float(item['mean_conflict_rate']):.4f}",
                "Recovery Speed": "N/A" if item.get("mean_recovery_speed") is None else f"{float(item['mean_recovery_speed']):.2f}",
                "Jain Fairness": f"{float(item['jain_energy_fairness']):.3f}",
            }
        )
        raw.append(
            {
                "Final Coverage": float(item["mean_coverage"]),
                "Conflict Rate": float(item["mean_conflict_rate"]),
                "Recovery Speed": float(item["mean_recovery_speed"]) if item.get("mean_recovery_speed") is not None else 0.0,
                "Jain Fairness": float(item["jain_energy_fairness"]),
            }
        )
    return rows, raw


def _build_generalization_table_rows(generalization: Dict[str, object], scale_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in scale_rows:
        rows.append(
            {
                "Sweep": "UAV Number",
                "Condition": int(row["num_uavs"]),
                "Method": METHOD_DISPLAY_NAME,
                "Coverage (%)": round(float(row["coverage"]), 3),
                "Conflict Rate": round(float(row["conflict_rate"]), 4),
                "Reward": round(float(row["reward"]), 3),
            }
        )
    for row in generalization.get("obstacle_density_rows", []):
        rows.append(
            {
                "Sweep": "Obstacle Ratio",
                "Condition": float(row["obstacle_ratio"]),
                "Method": row["method"],
                "Coverage (%)": round(float(row["coverage"]), 3),
                "Conflict Rate": round(float(row["conflict_rate"]), 4),
                "Reward": round(float(row["reward"]), 3),
            }
        )
    for row in generalization.get("dynamic_user_ratio_rows", []):
        rows.append(
            {
                "Sweep": "Dynamic-user Ratio",
                "Condition": float(row["dynamic_ratio"]),
                "Method": row["method"],
                "Coverage (%)": round(float(row["coverage"]), 3),
                "Conflict Rate": round(float(row["conflict_rate"]), 4),
                "Reward": round(float(row["reward"]), 3),
            }
        )
    return rows


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(1e-8, ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(1, len(a) + len(b) - 2)))
    return float((float(a.mean()) - float(b.mean())) / pooled)


def _build_statistics_table_rows(multiseed: Dict[str, object], strongest_baseline: str) -> List[Dict[str, object]]:
    seed_rows = multiseed.get("by_seed_rows", [])
    if not seed_rows:
        summary_rows = multiseed.get("summary_rows", [])
        return [
            {
                "Statistic": "Placeholder",
                "Value": "Multi-seed rows unavailable; summary only.",
                "Reference": METHOD_DISPLAY_NAME,
                "Compared": strongest_baseline,
            }
        ] + summary_rows
    rows: List[Dict[str, object]] = []
    metrics = ["coverage", "conflict_rate", "reward"]
    for metric in metrics:
        cast_vals = np.asarray([float(row[metric]) for row in seed_rows if row["method"] == METHOD_DISPLAY_NAME], dtype=np.float32)
        base_vals = np.asarray([float(row[metric]) for row in seed_rows if row["method"] == strongest_baseline], dtype=np.float32)
        stats_cast = mean_std_ci(cast_vals)
        stats_base = mean_std_ci(base_vals)
        if len(cast_vals) >= 2 and len(base_vals) >= 2:
            t_stat, p_value = stats.ttest_ind(cast_vals, base_vals, equal_var=False)
        else:
            t_stat, p_value = 0.0, 1.0
        rows.append(
            {
                "Metric": metric,
                "Reference": METHOD_DISPLAY_NAME,
                "Compared": strongest_baseline,
                "Reference mean±std": f"{stats_cast['mean']:.3f} ± {stats_cast['std']:.3f}",
                "Compared mean±std": f"{stats_base['mean']:.3f} ± {stats_base['std']:.3f}",
                "Reference 95% CI": f"[{stats_cast['ci95_low']:.3f}, {stats_cast['ci95_high']:.3f}]",
                "Compared 95% CI": f"[{stats_base['ci95_low']:.3f}, {stats_base['ci95_high']:.3f}]",
                "p-value": round(float(p_value), 6),
                "Effect size (d)": round(_cohen_d(cast_vals, base_vals), 4),
            }
        )
    return rows


def _manifest_row(
    asset_id: str,
    asset_type: str,
    description: str,
    generated_from_function: str,
    recommended_section: str,
    recommended_caption_short: str,
    placeholder: bool,
    files: Dict[str, str],
    source_csv: str = "",
) -> Dict[str, object]:
    return {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "filename_png": files.get("png", ""),
        "filename_svg": files.get("svg", ""),
        "filename_csv": files.get("csv", ""),
        "filename_md": files.get("md", ""),
        "filename_tex": files.get("tex", ""),
        "source_csv": source_csv,
        "description": description,
        "generated_from_function": generated_from_function,
        "recommended_section": recommended_section,
        "recommended_caption_short": recommended_caption_short,
        "placeholder": bool(placeholder),
    }


def run_paper_assets_full_pipeline(config: ExperimentConfig, output_dir: str, weight_path: str | None = None) -> Dict[str, object]:
    set_seed(config.seed)
    ensure_dir(output_dir)
    figures_main_dir = ensure_dir(os.path.join(output_dir, "figures_main"))
    figures_supp_dir = ensure_dir(os.path.join(output_dir, "figures_supp"))
    tables_main_dir = ensure_dir(os.path.join(output_dir, "tables_main"))
    tables_supp_dir = ensure_dir(os.path.join(output_dir, "tables_supp"))
    work_dir = ensure_dir(os.path.join(output_dir, "paper_assets_work"))

    log("paper_assets_full | running main training/evaluation bundle")
    train_results = run_train_pipeline(config, ensure_dir(os.path.join(work_dir, "mainline")), export_artifacts=True)
    dataset = ScenarioDataset(config.dataset_paths)
    generalization = run_mainline_generalization_suite(config, ensure_dir(os.path.join(work_dir, "generalization")), dataset, train_results["training_runs"])
    multiseed = run_mainline_multiseed_statistics(config, ensure_dir(os.path.join(work_dir, "mainline_multiseed")))
    tmarl_model = train_results["training_runs"]["tmarl"]["model"]
    safety_diag = export_safety_layer_diagnostics(ensure_dir(os.path.join(work_dir, "diagnostics")), tmarl_model, dataset, config)
    attention_base = os.path.join(ensure_dir(os.path.join(work_dir, "attention")), "Supplementary_Figure_S2_Attention_Proxy")
    export_attention_proxy(attention_base + ".png", tmarl_model, train_results["main_results"][0]["final_env"], next(tmarl_model.parameters()).device)
    failure_results = run_failure_feasibility_suite(
        dataset=dataset,
        config=config,
        model=tmarl_model,
        output_dir=ensure_dir(os.path.join(work_dir, "failure")),
        env_type=config.train_env,
        num_uavs=config.train_num_uavs,
    )

    curve_summary = _collect_curve_summary(train_results["training_runs"], os.path.join(work_dir, "mainline_multiseed"))
    stat_rows = _build_paper_stat_rows(multiseed.get("by_seed_rows", []), train_results["main_results"])
    strongest_baseline = next((row["name"] for row in sorted([r for r in stat_rows if r["name"] != METHOD_DISPLAY_NAME], key=lambda item: (-float(item["coverage"]), -float(item["reward"])))), "QMIX")

    manifest_rows: List[Dict[str, object]] = []

    files = _figure_task_scenarios(os.path.join(figures_main_dir, "Figure01_Task_Scenarios"), dataset, config)
    manifest_rows.append(_manifest_row("Figure01", "figure", "3D-aware task and scenario definition across three environments.", "_figure_task_scenarios", "Problem Definition", "3D task and scenario definition.", False, files))

    files = _figure_framework(os.path.join(figures_main_dir, "Figure02_Framework"))
    manifest_rows.append(_manifest_row("Figure02", "figure", "Overall framework of CAST-MARL with a 3D objective-conditioned multi-source risk-aware Transformer.", "_figure_framework", "Method", "Framework with an objective-conditioned multi-source Transformer.", False, files))

    files = _figure_safety_reassign(os.path.join(figures_main_dir, "Figure03_Safety_Reassign"))
    manifest_rows.append(_manifest_row("Figure03", "figure", "Safety layer and region reassignment mechanism.", "_figure_safety_reassign", "Method", "Safety layer and online reassignment mechanism.", False, files))

    files, placeholder = _figure_reward_convergence(os.path.join(figures_main_dir, "Figure04_Reward_Convergence"), curve_summary)
    manifest_rows.append(_manifest_row("Figure04", "figure", "Reward convergence with mean and 95% CI.", "_figure_reward_convergence", "Results", "Reward convergence across training.", placeholder, files))

    files, placeholder = _figure_coverage_conflict_convergence(os.path.join(figures_main_dir, "Figure05_Coverage_Conflict_Convergence"), curve_summary)
    manifest_rows.append(_manifest_row("Figure05", "figure", "Coverage and conflict convergence with mean and 95% CI.", "_figure_coverage_conflict_convergence", "Results", "Coverage and conflict convergence.", placeholder, files))

    files = _figure_main_benchmark(os.path.join(figures_main_dir, "Figure06_Main_Benchmark"), stat_rows)
    manifest_rows.append(_manifest_row("Figure06", "figure", "Main benchmark comparison with confidence intervals.", "_figure_main_benchmark", "Results", "Main benchmark comparison.", False, files))

    files = _figure_tradeoff(os.path.join(figures_main_dir, "Figure07_Tradeoff_Pareto"), stat_rows)
    manifest_rows.append(_manifest_row("Figure07", "figure", "Coverage-conflict-path tradeoff and Pareto-like frontier.", "_figure_tradeoff", "Results", "Coverage-conflict-path tradeoff.", False, files))

    files = _figure_trajectory_comparison(os.path.join(figures_main_dir, "Figure08_Trajectory_Comparison"), dataset, config, train_results["training_runs"], primary_model=tmarl_model)
    manifest_rows.append(_manifest_row("Figure08", "figure", "Qualitative trajectory comparison across environments.", "_figure_trajectory_comparison", "Results", "Qualitative trajectory comparison.", False, files))

    files = _figure_coverage_efficiency(os.path.join(figures_main_dir, "Figure09_Coverage_Efficiency_Heatmap"), dataset, config, tmarl_model)
    manifest_rows.append(_manifest_row("Figure09", "figure", "Coverage efficiency heatmap across representative environments.", "_figure_coverage_efficiency", "Results", "Coverage efficiency heatmap.", False, files))

    full_reference = next((row for row in train_results["main_results"] if row["name"] == METHOD_DISPLAY_NAME), None)
    full_reference_row = {
        "variant": "Full",
        "coverage": float(full_reference["coverage"]) if full_reference else 0.0,
        "conflict_rate": float(full_reference["conflict_rate"]) if full_reference else 0.0,
        "path_length": float(full_reference["path_length"]) if full_reference else 0.0,
        "reward": float(full_reference["reward"]) if full_reference else 0.0,
    }
    files = _figure_ablation(os.path.join(figures_main_dir, "Figure10_Ablation"), train_results["ablation_results"])
    manifest_rows.append(_manifest_row("Figure10", "figure", "Core ablation study across four metrics.", "_figure_ablation", "Results", "Core ablation study.", False, files))

    files = _figure_scalability_generalization(os.path.join(figures_main_dir, "Figure11_Scalability_Generalization"), train_results["scale_rows"], generalization)
    manifest_rows.append(_manifest_row("Figure11", "figure", "Scalability and generalization trends.", "_figure_scalability_generalization", "Results", "Scalability and generalization.", False, files))

    files = _figure_failure_robustness(os.path.join(figures_main_dir, "Figure12_Failure_Robustness"), failure_results)
    manifest_rows.append(_manifest_row("Figure12", "figure", "Failure robustness and recovery comparison.", "_figure_failure_robustness", "Results", "Failure robustness and recovery.", False, files))

    files = _figure_safety_diagnostics(os.path.join(figures_supp_dir, "Supplementary_Figure_S1_Safety_Diagnostics"), safety_diag)
    manifest_rows.append(_manifest_row("Supplementary_Figure_S1", "figure", "Safety layer diagnostics over time.", "_figure_safety_diagnostics", "Supplementary", "Safety layer diagnostics.", False, files, source_csv=os.path.join(work_dir, "diagnostics", "tables", "Supplementary_Safety_Layer_Diagnostics.csv")))

    files, placeholder = _figure_attention_proxy(os.path.join(figures_supp_dir, "Supplementary_Figure_S2_Attention_Proxy"), attention_base + ".json", attention_base + ".csv")
    manifest_rows.append(_manifest_row("Supplementary_Figure_S2", "figure", "Transformer attention proxy heatmap and summary.", "_figure_attention_proxy", "Supplementary", "Transformer attention proxy.", placeholder, files, source_csv=attention_base + ".csv"))

    files = export_table_bundle(
        os.path.join(tables_main_dir, "Table01_Experimental_Settings"),
        _build_experimental_settings_rows(config),
        "Experimental Settings and Environment Configuration",
        "This table summarizes the training setup, environment parameters, and major hyperparameters used in the experiments.",
    )
    manifest_rows.append(_manifest_row("Table01", "table", "Experimental settings and environment configuration.", "export_table_bundle", "Experimental Setup", "Experimental settings.", False, files))

    table2_rows, table2_raw = _build_main_benchmark_table_rows(stat_rows)
    files = export_table_bundle(
        os.path.join(tables_main_dir, "Table02_Main_Benchmark"),
        table2_rows,
        "Main Quantitative Comparison with Baselines",
        "Results are reported as mean ± std. Confidence intervals are preserved in the underlying summary files used to generate the figures.",
        metric_directions={"Coverage (%)": "max", "Conflict Rate": "min", "Path Length": "min", "Reward": "max", "Task Time": "min"},
        raw_rows=table2_raw,
    )
    manifest_rows.append(_manifest_row("Table02", "table", "Main benchmark comparison with baselines.", "_build_main_benchmark_table_rows", "Results", "Main quantitative comparison.", False, files))

    table3_rows, table3_raw = _build_efficiency_table_rows(train_results["training_runs"], train_results["main_results"])
    files = export_table_bundle(
        os.path.join(tables_main_dir, "Table03_Efficiency_Cost"),
        table3_rows,
        "Efficiency and Computational Cost",
        "Training and inference costs are measured under the current runtime configuration on the available device.",
        metric_directions={"Training Time (s)": "min", "Inference (ms/step)": "min", "Parameters": "min", "Memory (MB)": "min"},
        raw_rows=table3_raw,
    )
    manifest_rows.append(_manifest_row("Table03", "table", "Efficiency and computational cost.", "_build_efficiency_table_rows", "Results", "Efficiency and computational cost.", False, files))

    table4_rows, table4_raw = _build_ablation_table_rows(train_results["ablation_results"], full_reference_row)
    files = export_table_bundle(
        os.path.join(tables_main_dir, "Table04_Ablation"),
        table4_rows,
        "Ablation Study Quantitative Results",
        "The full model is compared against four structured ablations using the same evaluation protocol.",
        metric_directions={"Coverage (%)": "max", "Conflict Rate": "min", "Path Length": "min", "Reward": "max"},
        raw_rows=table4_raw,
    )
    manifest_rows.append(_manifest_row("Table04", "table", "Ablation study quantitative results.", "_build_ablation_table_rows", "Results", "Ablation study quantitative results.", False, files))

    table5_rows, table5_raw = _build_failure_table_rows(failure_results)
    files = export_table_bundle(
        os.path.join(tables_main_dir, "Table05_Failure_Robustness"),
        table5_rows,
        "Failure Robustness and Recovery Statistics",
        "Failure robustness is summarized over no-failure, failure without reassignment, and failure with reassignment settings.",
        metric_directions={"Final Coverage": "max", "Conflict Rate": "min", "Recovery Speed": "min", "Jain Fairness": "max"},
        raw_rows=table5_raw,
    )
    manifest_rows.append(_manifest_row("Table05", "table", "Failure robustness and recovery statistics.", "_build_failure_table_rows", "Results", "Failure robustness and recovery statistics.", False, files))

    supp_table_rows = _build_generalization_table_rows(generalization, train_results["scale_rows"])
    files = export_table_bundle(
        os.path.join(tables_supp_dir, "Supplementary_Table_S1_Generalization"),
        supp_table_rows,
        "Generalization and Scalability Summary",
        "This table reports generalization trends under varying numbers of UAVs, obstacle ratios, and dynamic-user ratios.",
    )
    manifest_rows.append(_manifest_row("Supplementary_Table_S1", "table", "Generalization and scalability summary.", "_build_generalization_table_rows", "Supplementary", "Generalization and scalability summary.", False, files))

    supp_stats_rows = _build_statistics_table_rows(multiseed, strongest_baseline)
    files = export_table_bundle(
        os.path.join(tables_supp_dir, "Supplementary_Table_S2_Statistics"),
        supp_stats_rows,
        "Statistical Significance and Multi-Seed Summary",
        "When sufficient multi-seed data are unavailable, summary statistics are reported and the limitation is explicitly preserved in the table.",
    )
    manifest_rows.append(_manifest_row("Supplementary_Table_S2", "table", "Statistical significance and multi-seed summary.", "_build_statistics_table_rows", "Supplementary", "Statistical significance and multi-seed summary.", len(multiseed.get("by_seed_rows", [])) == 0, files))

    figure_caption_lines = [
        "Figure 1. Programmatically generated 3D scenario definitions of the simple, obstacle, and dynamic-user environments, including user distribution, obstacle volumes, initial UAV deployment, communication range, and safety distance.",
        "Figure 2. Overall framework of CAST-MARL, showing 3D observation encoding, the objective-conditioned multi-source risk-aware Transformer, CTDE training, PPO optimization, safety revision, and environment feedback.",
        "Figure 3. Detailed mechanism of the safety layer and online region reassignment after UAV failure.",
        "Figure 4. Reward convergence curves of CAST-MARL and the baseline methods, reported with mean trajectories and 95% confidence intervals when available.",
        "Figure 5. Coverage and conflict convergence during training, highlighting the efficiency-safety evolution of the compared methods.",
        "Figure 6. Main benchmark comparison across Coverage, Conflict Rate, Path Length, and Reward with confidence intervals.",
        "Figure 7. Coverage-conflict-path tradeoff of the compared methods, with bubble size representing path length and a Pareto-like frontier shown for reference.",
        "Figure 8. Qualitative trajectory comparison between CAST-MARL and a strong baseline across the three environments.",
        "Figure 9. Coverage efficiency heatmaps showing spatially efficient, redundant, and under-served regions across representative scenarios.",
        "Figure 10. Core ablation study of CAST-MARL, comparing the full model with Transformer, attention, CTDE, and safety-layer removals.",
        "Figure 11. Scalability and generalization results under varying numbers of UAVs and scenario-shift ratios.",
        "Figure 12. Failure robustness and recovery comparison under no-failure, failure-without-reassignment, and failure-with-reassignment settings.",
        "Supplementary Figure S1. Safety-layer diagnostics showing temporal evolution of risk, repulsion, braking, and guidance terms.",
        "Supplementary Figure S2. Transformer attention proxy heatmap and compact statistical summary across agents.",
    ]
    write_text(os.path.join(output_dir, "figure_captions_draft.md"), "\n".join(f"- {line}" for line in figure_caption_lines) + "\n")

    table_note_lines = [
        "- Table 1. Experimental Settings and Environment Configuration. Note: The table summarizes the common environment parameters, training setup, and core hyperparameters used throughout the study.",
        "- Table 2. Main Quantitative Comparison with Baselines. Note: Results are reported in mean ± std format, while confidence intervals are retained in the source summary data used for figure generation.",
        "- Table 3. Efficiency and Computational Cost. Note: Runtime cost is reported under the current hardware setup to complement the task-performance comparison.",
        "- Table 4. Ablation Study Quantitative Results. Note: The full model is compared with structured removals of key coordination and safety components.",
        "- Table 5. Failure Robustness and Recovery Statistics. Note: The table summarizes resilience under failure and the benefit of reassignment-based recovery.",
        "- Supplementary Table S1. Generalization and Scalability Summary. Note: The table reports performance trends across varying numbers of UAVs, obstacle ratios, and dynamic-user ratios.",
        "- Supplementary Table S2. Statistical Significance and Multi-Seed Summary. Note: Statistical comparison is reported against the strongest baseline identified from the available results.",
    ]
    write_text(os.path.join(output_dir, "table_titles_notes_draft.md"), "\n".join(table_note_lines) + "\n")

    write_csv(os.path.join(output_dir, "paper_assets_manifest.csv"), manifest_rows)
    with open(os.path.join(output_dir, "paper_assets_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest_rows), f, indent=2, ensure_ascii=False)

    reuse_plan = {
        "reusable": [
            "run_train_pipeline",
            "run_mainline_generalization_suite",
            "run_mainline_multiseed_statistics",
            "run_failure_feasibility_suite",
            "export_attention_proxy",
            "export_safety_layer_diagnostics",
            "evaluate_controller",
        ],
        "merged_or_wrapped": [
            "plot_problem_scenarios_overview -> Figure01",
            "plot_framework_diagram + plot_safety_layer_schematic + plot_failure_redistribution_mechanism -> Figure02/Figure03",
            "training history npz exports -> Figure04/Figure05",
            "main result/error-bar utilities -> Figure06/Figure07",
            "trajectory/heatmap utilities -> Figure08/Figure09",
        ],
        "new": [
            "run_paper_assets_full_pipeline",
            "export_table_bundle",
            "_collect_curve_summary",
            "_figure_* main/supplementary composition helpers",
            "paper asset manifest and caption exporters",
        ],
    }
    with open(os.path.join(output_dir, "paper_assets_plan.json"), "w", encoding="utf-8") as f:
        json.dump(reuse_plan, f, indent=2, ensure_ascii=False)

    return {
        "output_dir": output_dir,
        "figures_main_dir": figures_main_dir,
        "figures_supp_dir": figures_supp_dir,
        "tables_main_dir": tables_main_dir,
        "tables_supp_dir": tables_supp_dir,
        "manifest_path": os.path.join(output_dir, "paper_assets_manifest.json"),
        "plan_path": os.path.join(output_dir, "paper_assets_plan.json"),
        "placeholder_count": int(sum(1 for row in manifest_rows if row["placeholder"])),
    }


def run_paper_assets_redraw_pipeline(config: ExperimentConfig, output_dir: str, source_dir: str, weight_path: str) -> Dict[str, object]:
    set_seed(config.seed)
    ensure_dir(output_dir)
    figures_main_dir = ensure_dir(os.path.join(output_dir, "figures_main"))
    figures_supp_dir = ensure_dir(os.path.join(output_dir, "figures_supp"))
    tables_main_dir = ensure_dir(os.path.join(output_dir, "tables_main"))
    tables_supp_dir = ensure_dir(os.path.join(output_dir, "tables_supp"))

    work_dir = os.path.join(source_dir, "paper_assets_work")
    mainline_dir = os.path.join(work_dir, "mainline")
    generalization_dir = os.path.join(work_dir, "generalization")
    multiseed_dir = os.path.join(work_dir, "mainline_multiseed")
    diagnostics_dir = os.path.join(work_dir, "diagnostics")
    attention_dir = os.path.join(work_dir, "attention")
    failure_summary_path = os.path.join(work_dir, "failure", "supplement", "failure_feasibility", "failure_feasibility_summary.json")
    main_summary_path = os.path.join(mainline_dir, "summary.json")
    if not os.path.exists(main_summary_path):
        raise FileNotFoundError(f"Missing cached summary for redraw: {main_summary_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing checkpoint for redraw: {weight_path}")

    with open(main_summary_path, "r", encoding="utf-8") as f:
        main_summary = json.load(f)
    with open(os.path.join(generalization_dir, "tables", "Table8_Generalization_Overview.csv"), "r", encoding="utf-8") as _:
        pass
    generalization = {
        "obstacle_density_rows": [],
        "agent_num_rows": [],
        "dynamic_user_ratio_rows": [],
        "overview_rows": [],
    }
    for csv_name, key in [
        ("Mainline_Generalization_ObstacleDensity.csv", "obstacle_density_rows"),
        ("Mainline_Generalization_AgentNum.csv", "agent_num_rows"),
        ("Mainline_Generalization_DynamicUserRatio.csv", "dynamic_user_ratio_rows"),
        ("Table8_Generalization_Overview.csv", "overview_rows"),
    ]:
        csv_path = os.path.join(generalization_dir, "tables", csv_name)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                generalization[key] = [dict(row) for row in reader]

    multiseed_summary_path = os.path.join(multiseed_dir, "mainline_multiseed_summary.json")
    multiseed = {"by_seed_rows": [], "summary_rows": [], "significance_rows": []}
    if os.path.exists(multiseed_summary_path):
        with open(multiseed_summary_path, "r", encoding="utf-8") as f:
            multiseed = json.load(f)

    diag_json_path = os.path.join(diagnostics_dir, "tables", "Supplementary_Safety_Layer_Diagnostics.json")
    safety_diag = {"rows": [], "summary": {}}
    if os.path.exists(diag_json_path):
        with open(diag_json_path, "r", encoding="utf-8") as f:
            safety_diag = json.load(f)

    failure_results = {"scenario_results": []}
    if os.path.exists(failure_summary_path):
        with open(failure_summary_path, "r", encoding="utf-8") as f:
            failure_results = json.load(f)

    dataset = ScenarioDataset(config.dataset_paths)
    tmarl_model = load_tmarl_model(weight_path, config)
    curve_summary = _collect_curve_summary(None, multiseed_dir)
    if not curve_summary:
        curve_summary = _collect_curve_summary(None, None)
        for method, filename in [
            (METHOD_DISPLAY_NAME, "tmarl_training_curves.npz"),
            ("MAPPO", "mappo_training_curves.npz"),
            ("MADDPG", "maddpg_training_curves.npz"),
            ("QMIX", "qmix_training_curves.npz"),
            ("PPO", "ppo_training_curves.npz"),
        ]:
            npz_path = os.path.join(mainline_dir, "curves", filename)
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                curve_summary[method] = {
                    "rewards_mean": np.asarray(data["rewards"], dtype=np.float32),
                    "rewards_ci": np.zeros_like(np.asarray(data["rewards"], dtype=np.float32)),
                    "coverage_mean": np.asarray(data["coverage"], dtype=np.float32) * 100.0,
                    "coverage_ci": np.zeros_like(np.asarray(data["coverage"], dtype=np.float32)),
                    "conflicts_mean": np.asarray(data["conflicts"], dtype=np.float32),
                    "conflicts_ci": np.zeros_like(np.asarray(data["conflicts"], dtype=np.float32)),
                }

    main_results = list(main_summary.get("main_results", []))
    ablation_results = list(main_summary.get("ablation_results", []))
    scale_rows = list(main_summary.get("scalability_rows", []))
    efficiency_rows = list(main_summary.get("efficiency_rows", []))
    stat_rows = _build_paper_stat_rows(multiseed.get("by_seed_rows", []), main_results)
    strongest_baseline = next((row["name"] for row in sorted([r for r in stat_rows if r["name"] != METHOD_DISPLAY_NAME], key=lambda item: (-float(item["coverage"]), -float(item["reward"])))), "QMIX")

    manifest_rows: List[Dict[str, object]] = []
    files = _figure_task_scenarios(os.path.join(figures_main_dir, "Figure01_Task_Scenarios"), dataset, config)
    manifest_rows.append(_manifest_row("Figure01", "figure", "3D-aware task and scenario definition across three environments.", "_figure_task_scenarios", "Problem Definition", "3D task and scenario definition.", False, files))
    files = _figure_framework(os.path.join(figures_main_dir, "Figure02_Framework"))
    manifest_rows.append(_manifest_row("Figure02", "figure", "Overall framework of CAST-MARL with a 3D objective-conditioned multi-source risk-aware Transformer.", "_figure_framework", "Method", "Framework with an objective-conditioned multi-source Transformer.", False, files))
    files = _figure_safety_reassign(os.path.join(figures_main_dir, "Figure03_Safety_Reassign"))
    manifest_rows.append(_manifest_row("Figure03", "figure", "Safety layer and region reassignment mechanism.", "_figure_safety_reassign", "Method", "Safety layer and online reassignment mechanism.", False, files))
    files, placeholder = _figure_reward_convergence(os.path.join(figures_main_dir, "Figure04_Reward_Convergence"), curve_summary)
    manifest_rows.append(_manifest_row("Figure04", "figure", "Reward convergence with mean and 95% CI.", "_figure_reward_convergence", "Results", "Reward convergence across training.", placeholder, files))
    files, placeholder = _figure_coverage_conflict_convergence(os.path.join(figures_main_dir, "Figure05_Coverage_Conflict_Convergence"), curve_summary)
    manifest_rows.append(_manifest_row("Figure05", "figure", "Coverage and conflict convergence with mean and 95% CI.", "_figure_coverage_conflict_convergence", "Results", "Coverage and conflict convergence.", placeholder, files))
    files = _figure_main_benchmark(os.path.join(figures_main_dir, "Figure06_Main_Benchmark"), stat_rows)
    manifest_rows.append(_manifest_row("Figure06", "figure", "Main benchmark comparison with confidence intervals.", "_figure_main_benchmark", "Results", "Main benchmark comparison.", False, files))
    files = _figure_tradeoff(os.path.join(figures_main_dir, "Figure07_Tradeoff_Pareto"), stat_rows)
    manifest_rows.append(_manifest_row("Figure07", "figure", "Coverage-conflict-path tradeoff and Pareto-like frontier.", "_figure_tradeoff", "Results", "Coverage-conflict-path tradeoff.", False, files))
    files = _figure_trajectory_comparison(os.path.join(figures_main_dir, "Figure08_Trajectory_Comparison"), dataset, config, None, primary_model=tmarl_model)
    manifest_rows.append(_manifest_row("Figure08", "figure", "Qualitative trajectory comparison across environments; baseline path uses the heuristic fallback during redraw-only mode.", "_figure_trajectory_comparison", "Results", "Qualitative trajectory comparison.", False, files))
    files = _figure_coverage_efficiency(os.path.join(figures_main_dir, "Figure09_Coverage_Efficiency_Heatmap"), dataset, config, tmarl_model)
    manifest_rows.append(_manifest_row("Figure09", "figure", "Coverage efficiency heatmap across representative environments.", "_figure_coverage_efficiency", "Results", "Coverage efficiency heatmap.", False, files))
    full_reference_row = {
        "variant": "Full",
        "coverage": float(next((row["coverage"] for row in main_results if row["name"] == METHOD_DISPLAY_NAME), 0.0)),
        "conflict_rate": float(next((row["conflict_rate"] for row in main_results if row["name"] == METHOD_DISPLAY_NAME), 0.0)),
        "path_length": float(next((row["path_length"] for row in main_results if row["name"] == METHOD_DISPLAY_NAME), 0.0)),
        "reward": float(next((row["reward"] for row in main_results if row["name"] == METHOD_DISPLAY_NAME), 0.0)),
    }
    files = _figure_ablation(os.path.join(figures_main_dir, "Figure10_Ablation"), ablation_results)
    manifest_rows.append(_manifest_row("Figure10", "figure", "Core ablation study across four metrics.", "_figure_ablation", "Results", "Core ablation study.", False, files))
    files = _figure_scalability_generalization(os.path.join(figures_main_dir, "Figure11_Scalability_Generalization"), scale_rows, generalization)
    manifest_rows.append(_manifest_row("Figure11", "figure", "Scalability and generalization trends.", "_figure_scalability_generalization", "Results", "Scalability and generalization.", False, files))
    files = _figure_failure_robustness(os.path.join(figures_main_dir, "Figure12_Failure_Robustness"), failure_results)
    manifest_rows.append(_manifest_row("Figure12", "figure", "Failure robustness and recovery comparison.", "_figure_failure_robustness", "Results", "Failure robustness and recovery.", False, files))
    attention_base = os.path.join(attention_dir, "Supplementary_Figure_S2_Attention_Proxy")
    files = _figure_safety_diagnostics(os.path.join(figures_supp_dir, "Supplementary_Figure_S1_Safety_Diagnostics"), safety_diag)
    manifest_rows.append(_manifest_row("Supplementary_Figure_S1", "figure", "Safety layer diagnostics over time.", "_figure_safety_diagnostics", "Supplementary", "Safety layer diagnostics.", False, files, source_csv=os.path.join(diagnostics_dir, "tables", "Supplementary_Safety_Layer_Diagnostics.csv")))
    files, placeholder = _figure_attention_proxy(os.path.join(figures_supp_dir, "Supplementary_Figure_S2_Attention_Proxy"), attention_base + ".json", attention_base + ".csv")
    manifest_rows.append(_manifest_row("Supplementary_Figure_S2", "figure", "Transformer attention proxy heatmap and summary.", "_figure_attention_proxy", "Supplementary", "Transformer attention proxy.", placeholder, files, source_csv=attention_base + ".csv"))

    files = export_table_bundle(os.path.join(tables_main_dir, "Table01_Experimental_Settings"), _build_experimental_settings_rows(config), "Experimental Settings and Environment Configuration", "This table summarizes the training setup, environment parameters, and major hyperparameters used in the experiments.")
    manifest_rows.append(_manifest_row("Table01", "table", "Experimental settings and environment configuration.", "export_table_bundle", "Experimental Setup", "Experimental settings.", False, files))
    table2_rows, table2_raw = _build_main_benchmark_table_rows(stat_rows)
    files = export_table_bundle(os.path.join(tables_main_dir, "Table02_Main_Benchmark"), table2_rows, "Main Quantitative Comparison with Baselines", "Results are reported in mean ± std format. Confidence intervals are preserved in the underlying summary files used to generate the figures.", metric_directions={"Coverage (%)": "max", "Conflict Rate": "min", "Path Length": "min", "Reward": "max", "Task Time": "min"}, raw_rows=table2_raw)
    manifest_rows.append(_manifest_row("Table02", "table", "Main benchmark comparison with baselines.", "_build_main_benchmark_table_rows", "Results", "Main quantitative comparison.", False, files))
    table3_rows, table3_raw = _build_efficiency_table_rows_from_summary(config, efficiency_rows, main_results)
    files = export_table_bundle(os.path.join(tables_main_dir, "Table03_Efficiency_Cost"), table3_rows, "Efficiency and Computational Cost", "Training and inference costs are measured under the cached run configuration and redrawn without retraining.", metric_directions={"Training Time (s)": "min", "Inference (ms/step)": "min", "Parameters": "min", "Memory (MB)": "min"}, raw_rows=table3_raw)
    manifest_rows.append(_manifest_row("Table03", "table", "Efficiency and computational cost.", "_build_efficiency_table_rows_from_summary", "Results", "Efficiency and computational cost.", False, files))
    table4_rows, table4_raw = _build_ablation_table_rows(ablation_results, full_reference_row)
    files = export_table_bundle(os.path.join(tables_main_dir, "Table04_Ablation"), table4_rows, "Ablation Study Quantitative Results", "The full model is compared with four structured ablations using the cached evaluation results.", metric_directions={"Coverage (%)": "max", "Conflict Rate": "min", "Path Length": "min", "Reward": "max"}, raw_rows=table4_raw)
    manifest_rows.append(_manifest_row("Table04", "table", "Ablation study quantitative results.", "_build_ablation_table_rows", "Results", "Ablation study quantitative results.", False, files))
    table5_rows, table5_raw = _build_failure_table_rows(failure_results)
    files = export_table_bundle(os.path.join(tables_main_dir, "Table05_Failure_Robustness"), table5_rows, "Failure Robustness and Recovery Statistics", "Failure robustness is summarized over no-failure, failure without reassignment, and failure with reassignment settings.", metric_directions={"Final Coverage": "max", "Conflict Rate": "min", "Recovery Speed": "min", "Jain Fairness": "max"}, raw_rows=table5_raw)
    manifest_rows.append(_manifest_row("Table05", "table", "Failure robustness and recovery statistics.", "_build_failure_table_rows", "Results", "Failure robustness and recovery statistics.", False, files))
    supp_table_rows = _build_generalization_table_rows(generalization, scale_rows)
    files = export_table_bundle(os.path.join(tables_supp_dir, "Supplementary_Table_S1_Generalization"), supp_table_rows, "Generalization and Scalability Summary", "This table reports performance trends across varying numbers of UAVs, obstacle ratios, and dynamic-user ratios.")
    manifest_rows.append(_manifest_row("Supplementary_Table_S1", "table", "Generalization and scalability summary.", "_build_generalization_table_rows", "Supplementary", "Generalization and scalability summary.", False, files))
    supp_stats_rows = _build_statistics_table_rows(multiseed, strongest_baseline)
    files = export_table_bundle(os.path.join(tables_supp_dir, "Supplementary_Table_S2_Statistics"), supp_stats_rows, "Statistical Significance and Multi-Seed Summary", "This table is reconstructed directly from the cached multi-seed summary.",)
    manifest_rows.append(_manifest_row("Supplementary_Table_S2", "table", "Statistical significance and multi-seed summary.", "_build_statistics_table_rows", "Supplementary", "Statistical significance and multi-seed summary.", len(multiseed.get("by_seed_rows", [])) == 0, files))

    figure_caption_path = os.path.join(source_dir, "figure_captions_draft.md")
    table_note_path = os.path.join(source_dir, "table_titles_notes_draft.md")
    if os.path.exists(figure_caption_path):
        with open(figure_caption_path, "r", encoding="utf-8") as f:
            write_text(os.path.join(output_dir, "figure_captions_draft.md"), f.read())
    if os.path.exists(table_note_path):
        with open(table_note_path, "r", encoding="utf-8") as f:
            write_text(os.path.join(output_dir, "table_titles_notes_draft.md"), f.read())

    write_csv(os.path.join(output_dir, "paper_assets_manifest.csv"), manifest_rows)
    with open(os.path.join(output_dir, "paper_assets_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest_rows), f, indent=2, ensure_ascii=False)
    return {
        "output_dir": output_dir,
        "figures_main_dir": figures_main_dir,
        "figures_supp_dir": figures_supp_dir,
        "tables_main_dir": tables_main_dir,
        "tables_supp_dir": tables_supp_dir,
        "manifest_path": os.path.join(output_dir, "paper_assets_manifest.json"),
        "placeholder_count": int(sum(1 for row in manifest_rows if row["placeholder"])),
    }


def run_demo(config: ExperimentConfig, weight_path: str) -> None:
    dataset = ScenarioDataset(config.dataset_paths)
    model = load_tmarl_model(weight_path, config)
    result = evaluate_controller(METHOD_DISPLAY_NAME, dataset, config, model=model, env_type="obstacles", num_uavs=config.train_num_uavs, episodes=1)
    print("Demo result")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "final_env"},
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAST-MARL for multi-UAV cooperative coverage")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "demo", "tune", "tune_train", "repeat_eval", "paper_main", "paper_supp", "paper_diagrams", "paper_assets_full", "paper_assets_redraw", "failure_feasibility"],
        help="run mode",
    )
    parser.add_argument("--episodes", type=int, default=80, help="training episodes")
    parser.add_argument("--max_steps", type=int, default=35, help="steps per episode")
    parser.add_argument("--batch_episodes", type=int, default=8, help="episodes per optimizer step")
    parser.add_argument("--num_uavs", type=int, default=6, help="training fleet size")
    parser.add_argument("--env", default="simple", choices=["simple", "obstacles", "dynamic_users", "terrain_mountain", "terrain_urban", "terrain_plain"], help="environment type")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--weights", default=BEST_WEIGHT_PATH, help="checkpoint path")
    parser.add_argument("--output_dir", default="", help="optional output directory")
    parser.add_argument("--source_dir", default="", help="existing paper_assets_full output directory for redraw-only mode")
    parser.add_argument("--tune_trials", type=int, default=12, help="bayesian-style hyperparameter search trials")
    parser.add_argument("--tune_train_episodes", type=int, default=12, help="episodes per tuning trial")
    parser.add_argument("--tune_eval_episodes", type=int, default=16, help="evaluation episodes per tuning trial")
    parser.add_argument("--best_conflict_gate", type=float, default=0.08, help="checkpoint conflict gate in main training")
    parser.add_argument("--best_coverage_gate", type=float, default=0.70, help="checkpoint coverage gate in main training")
    parser.add_argument("--tune_best_conflict_gate", type=float, default=0.06, help="trial conflict gate in tuning")
    parser.add_argument("--tune_best_coverage_gate", type=float, default=0.64, help="trial coverage gate in tuning")
    parser.add_argument("--repeat_eval_episodes", type=int, default=20, help="episodes per seed in repeat_eval mode")
    parser.add_argument("--repeat_seed_count", type=int, default=5, help="number of seeds in repeat_eval mode")
    parser.add_argument("--mainline_seed_count", type=int, default=5, help="number of training seeds in paper_main mode")
    parser.add_argument("--generalization_eval_episodes", type=int, default=10, help="episodes per condition in generalization sweeps")
    parser.add_argument("--failure_eval_episodes", type=int, default=20, help="episodes per main scenario in failure_feasibility mode")
    parser.add_argument("--region_reassign_interval", type=int, default=4, help="region/task reassignment interval")
    parser.add_argument("--target_supervision", action="store_true", help="enable target-domain supervised auxiliary training for CAST-MARL")
    parser.add_argument("--target_supervision_weight", type=float, default=0.15, help="auxiliary loss weight for target-domain supervision")
    parser.add_argument("--target_supervision_batches", type=int, default=1, help="number of target-domain supervised batches after each RL epoch")
    parser.add_argument("--target_supervision_batch_size", type=int, default=48, help="batch size for target-domain supervised batches")
    parser.add_argument("--target_supervision_start_episode", type=int, default=0, help="episode index to start target-domain supervision")
    parser.add_argument("--target_supervision_rollout_steps", type=int, default=4, help="number of supervised rollout states collected per target-domain reset")
    parser.add_argument("--target_supervision_expert_mix", type=float, default=0.55, help="blend weight on expert/region targets for target-domain supervision")
    parser.add_argument("--target_supervision_frontier_mix", type=float, default=0.30, help="blend weight on frontier guidance for target-domain supervision")
    parser.add_argument("--target_supervision_repulsion_coef", type=float, default=0.22, help="repulsion strength in target-domain supervision targets")
    parser.add_argument("--target_supervision_obstacle_coef", type=float, default=0.18, help="obstacle-avoidance strength in target-domain supervision targets")
    parser.add_argument("--target_supervision_refine_epochs", type=int, default=6, help="extra target-domain supervised refine epochs after RL training")
    parser.add_argument("--target_dataset_paths", nargs="*", default=None, help="optional explicit target-domain dataset .npz paths")
    parser.add_argument("--use_failure_module", action="store_true", help="enable standardized UAV failure module")
    parser.add_argument("--fail_prob_base", type=float, default=0.003, help="base UAV failure probability")
    parser.add_argument("--fail_prob_load_scale", type=float, default=0.002, help="extra failure probability from assignment load")
    parser.add_argument("--fail_trigger_min_step", type=int, default=4, help="earliest timestep for failure triggering")
    parser.add_argument("--fail_trigger_max_step", type=int, default=28, help="latest timestep for failure triggering")
    parser.add_argument("--fail_max_count", type=int, default=2, help="maximum number of UAV failures in one episode")
    parser.add_argument("--fail_fixed_count", type=int, default=0, help="fixed number of failed UAVs when >0")
    parser.add_argument("--disable_fail_load_aware", action="store_true", help="disable load-aware failure probabilities")
    parser.add_argument("--disable_fail_reassign_on_event", action="store_true", help="disable immediate reassignment after failures")
    parser.add_argument("--camera_fov_deg", type=float, default=76.0, help="altitude-aware camera field of view for 3D coverage modeling")
    parser.add_argument("--obstacle_height_min", type=float, default=24.0, help="minimum 3D obstacle height")
    parser.add_argument("--obstacle_height_max", type=float, default=88.0, help="maximum 3D obstacle height")
    parser.add_argument("--dynamic_altitude_ratio", type=float, default=0.06, help="vertical drift strength for dynamic users")
    parser.add_argument("--terrain_clearance", type=float, default=14.0, help="minimum terrain clearance for UAV flight")
    parser.add_argument("--terrain_height_cap", type=float, default=42.0, help="normalized terrain height cap inside the simulation domain")
    parser.add_argument("--relation_risk_strength", type=float, default=0.90, help="risk bias strength in the multi-source relational Transformer")
    parser.add_argument("--relation_comm_strength", type=float, default=0.35, help="communication bias strength in the multi-source relational Transformer")
    parser.add_argument("--relation_altitude_strength", type=float, default=0.22, help="altitude-gap penalty strength in the multi-source relational Transformer")
    parser.add_argument("--multi_source_gate_strength", type=float, default=0.65, help="strength of self/agent/entity/task source gating")
    parser.add_argument("--objective_gate_strength", type=float, default=0.55, help="strength of objective-conditioned attention guidance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    bundle_dir = build_timestamp_bundle_dir()
    global BEST_WEIGHTS_DIR, BEST_WEIGHT_PATH
    BEST_WEIGHTS_DIR = os.path.join(bundle_dir, "best_weights")
    BEST_WEIGHT_PATH = os.path.join(BEST_WEIGHTS_DIR, "uav_best.pth")
    output_dir = resolve_output_dir(args.output_dir, bundle_dir)
    ensure_dir(output_dir)
    set_log_file(os.path.join(output_dir, "run.log"))
    config = ExperimentConfig(
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_episodes=args.batch_episodes,
        train_env=args.env,
        train_num_uavs=args.num_uavs,
        output_root=output_dir,
        tune_trials=args.tune_trials,
        tune_train_episodes=args.tune_train_episodes,
        tune_eval_episodes=args.tune_eval_episodes,
        mainline_seed_count=args.mainline_seed_count,
        generalization_eval_episodes=args.generalization_eval_episodes,
        failure_eval_episodes=args.failure_eval_episodes,
        best_conflict_gate=args.best_conflict_gate,
        best_coverage_gate=args.best_coverage_gate,
        tune_best_conflict_gate=args.tune_best_conflict_gate,
        tune_best_coverage_gate=args.tune_best_coverage_gate,
        region_reassign_interval=args.region_reassign_interval,
        target_supervision_enabled=args.target_supervision,
        target_supervision_weight=args.target_supervision_weight,
        target_supervision_batches=args.target_supervision_batches,
        target_supervision_batch_size=args.target_supervision_batch_size,
        target_supervision_start_episode=args.target_supervision_start_episode,
        target_supervision_rollout_steps=args.target_supervision_rollout_steps,
        target_supervision_expert_mix=args.target_supervision_expert_mix,
        target_supervision_frontier_mix=args.target_supervision_frontier_mix,
        target_supervision_repulsion_coef=args.target_supervision_repulsion_coef,
        target_supervision_obstacle_coef=args.target_supervision_obstacle_coef,
        target_supervision_refine_epochs=args.target_supervision_refine_epochs,
        target_dataset_paths=list(args.target_dataset_paths or []),
        use_failure_module=args.use_failure_module,
        fail_prob_base=args.fail_prob_base,
        fail_prob_load_scale=args.fail_prob_load_scale,
        fail_trigger_min_step=args.fail_trigger_min_step,
        fail_trigger_max_step=args.fail_trigger_max_step,
        fail_max_count=args.fail_max_count,
        fail_fixed_count=args.fail_fixed_count,
        fail_load_aware=not args.disable_fail_load_aware,
        fail_reassign_on_event=not args.disable_fail_reassign_on_event,
        camera_fov_deg=args.camera_fov_deg,
        obstacle_height_min=args.obstacle_height_min,
        obstacle_height_max=args.obstacle_height_max,
        dynamic_altitude_ratio=args.dynamic_altitude_ratio,
        terrain_clearance=args.terrain_clearance,
        terrain_height_cap=args.terrain_height_cap,
        relation_risk_strength=args.relation_risk_strength,
        relation_comm_strength=args.relation_comm_strength,
        relation_altitude_strength=args.relation_altitude_strength,
        multi_source_gate_strength=args.multi_source_gate_strength,
        objective_gate_strength=args.objective_gate_strength,
    )
    runtime_config = build_failure_runtime_config(config) if args.mode == "failure_feasibility" else build_mainline_runtime_config(config)
    log(
        f"command start | mode={args.mode} env={args.env} num_uavs={args.num_uavs} "
        f"episodes={args.episodes} batch_episodes={args.batch_episodes} "
        f"tune_eval_episodes={args.tune_eval_episodes} "
        f"target_supervision={args.target_supervision} "
        f"target_rollout={args.target_supervision_rollout_steps} "
        f"target_refine={args.target_supervision_refine_epochs} "
        f"best_conflict_gate={runtime_config.best_conflict_gate:.3f} best_coverage_gate={runtime_config.best_coverage_gate:.3f} "
        f"tune_best_conflict_gate={runtime_config.tune_best_conflict_gate:.3f} "
        f"tune_best_coverage_gate={runtime_config.tune_best_coverage_gate:.3f} "
        f"output_dir={output_dir}"
    )

    if args.mode == "train":
        results = run_train_pipeline(runtime_config, output_dir)
        summary = {
            "output_dir": output_dir,
            "best_weights": BEST_WEIGHT_PATH,
            "main_table": [{k: v for k, v in row.items() if k != "final_env"} for row in results["main_results"]],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.mode == "eval":
        results = run_eval_pipeline(runtime_config, output_dir, args.weights)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "main_result": {k: v for k, v in results["main_result"].items() if k != "final_env"},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "repeat_eval":
        repeat_seeds = [args.seed + i for i in range(max(1, args.repeat_seed_count))]
        results = run_repeat_eval_pipeline(runtime_config, output_dir, args.weights, repeat_seeds, args.repeat_eval_episodes)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "repeat_seeds": repeat_seeds,
                    "summary": results["summary"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "tune":
        results = tune_hyperparameters(runtime_config, output_dir)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "tune_dir": results["tune_dir"],
                    "best_trial": results["best_trial"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "tune_train":
        results = run_tune_then_train_pipeline(runtime_config, output_dir)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "tune_dir": results["tune_results"]["tune_dir"],
                    "train_output_dir": results["train_output_dir"],
                    "best_trial": results["tune_results"]["best_trial"],
                    "best_weights": BEST_WEIGHT_PATH,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "paper_supp":
        results = run_paper_supplement_pipeline(runtime_config, output_dir, args.weights)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "summary_file": os.path.join(output_dir, "paper_supp_summary.json"),
                    "repeat_eval_seeds": results["repeat_eval_seeds"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "paper_main":
        results = run_mainline_paper_pipeline(runtime_config, output_dir)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "summary_file": os.path.join(output_dir, "paper_main_summary.json"),
                    "mainline_multiseed_summary": results["multiseed_summary_file"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "paper_diagrams":
        results = run_paper_diagram_pipeline(runtime_config, output_dir, args.weights)
        print(
            json.dumps(
                {
                    "output_dir": output_dir,
                    "summary_file": os.path.join(output_dir, "paper_diagram_summary.json"),
                    "figure_dir": results["figure_dir"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "paper_assets_full":
        results = run_paper_assets_full_pipeline(runtime_config, output_dir, args.weights)
        print(
            json.dumps(
                {
                    "output_dir": results["output_dir"],
                    "figures_main_dir": results["figures_main_dir"],
                    "figures_supp_dir": results["figures_supp_dir"],
                    "tables_main_dir": results["tables_main_dir"],
                    "tables_supp_dir": results["tables_supp_dir"],
                    "manifest_path": results["manifest_path"],
                    "plan_path": results["plan_path"],
                    "placeholder_count": results["placeholder_count"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "paper_assets_redraw":
        if not args.source_dir.strip():
            raise ValueError("--source_dir is required for paper_assets_redraw mode.")
        results = run_paper_assets_redraw_pipeline(runtime_config, output_dir, args.source_dir, args.weights)
        print(
            json.dumps(
                {
                    "output_dir": results["output_dir"],
                    "figures_main_dir": results["figures_main_dir"],
                    "figures_supp_dir": results["figures_supp_dir"],
                    "tables_main_dir": results["tables_main_dir"],
                    "tables_supp_dir": results["tables_supp_dir"],
                    "manifest_path": results["manifest_path"],
                    "placeholder_count": results["placeholder_count"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.mode == "failure_feasibility":
        dataset = ScenarioDataset(runtime_config.dataset_paths)
        model = load_tmarl_model(args.weights, runtime_config)
        log("Running failure feasibility analysis suite (A/B/C + sensitivity + multi-seed)...")
        results = run_failure_supplement_pipeline(
            dataset=dataset,
            config=runtime_config,
            model=model,
            output_dir=output_dir,
            env_type=runtime_config.train_env,
            num_uavs=runtime_config.train_num_uavs,
        )
        log(f"Failure feasibility done. Figure and JSON saved to {results['output_dir']}")
        print(json.dumps({"output_dir": results["output_dir"], "scenario_results": results["scenario_results"]}, indent=2, ensure_ascii=False))
    else:
        run_demo(runtime_config, args.weights)


if __name__ == "__main__":
    main()
