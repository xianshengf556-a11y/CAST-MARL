# -*- coding: utf-8 -*-
"""Multi-domain mixed training for CAST-MARL.

Data-side change only: during training, each episode samples one of the four
terrain types (plain / urban / mountain / mountain-dense) and builds the
corresponding environment. The CAST-MARL network, reward function, and the
CVA-SP projection logic are unchanged. After training, the checkpoint is
evaluated separately on each of the four terrains under the cross-map
protocol, and the results are compared with the single-terrain (plain)
pretrained checkpoint.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal


PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / "experimental_runs" / "reproducibility" / "cast_marl_experiment_source.py"
spec = importlib.util.spec_from_file_location("cast_marl_source", SOURCE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.TERRAIN_MANIFEST_PATH = str(PACKAGE / "experimental_runs" / "reproducibility" / "missing_terrain_manifest.json")
mod._TERRAIN_CACHE.clear()

DATASET = [
    str(PACKAGE / "dataset" / "dataset1_users35.npz"),
    str(PACKAGE / "dataset" / "dataset1_users45.npz"),
    str(PACKAGE / "dataset" / "dataset1_users60.npz"),
]
CHECKPOINT = PACKAGE / "experimental_runs" / "localized_uav_best.pth"
OUT = PACKAGE / "experimental_runs" / "reproducibility" / "cross_map"
OUT.mkdir(parents=True, exist_ok=True)

# Terrain mix used for multi-domain training (same obstacle ratios as the
# cross-map protocol).
TERRAIN_MIX = [
    ("terrain_plain", "plain", 0.08),
    ("terrain_urban", "urban", 0.20),
    ("terrain_mountain", "mountain", 0.16),
    ("terrain_mountain", "mountain-dense", 0.30),
]

NUM_UAVS = 6
MAX_STEPS = 35
MAX_SPEED = 150.0

# Training schedule (adaptable): episodes per terrain type, total episodes,
# batch size per optimizer update.
EPISODES_PER_TERRAIN = 40        # 4 terrains x 40 = 160 training episodes
TOTAL_EPISODES = EPISODES_PER_TERRAIN * len(TERRAIN_MIX)
BATCH_EPISODES = 8               # episodes per PPO update
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 128
CLIP_EPS = 0.2
OVERRIDE_LR = 1.5e-4             # lower than the single-domain 3e-4 for
                                 # stability under mixed terrain sampling
TRAIN_SEED = 2026
EVAL_SEEDS = [42, 43, 44, 45, 46]
EVAL_EPISODES_PER_SEED = 12


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    if not isinstance(cfg_dict, dict):
        cfg_dict = {k: getattr(cfg_dict, k) for k in dir(cfg_dict)
                    if not k.startswith("_")}
    config = mod.ExperimentConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in mod.ExperimentConfig.__dataclass_fields__})
    dataset = mod.ScenarioDataset(DATASET)
    state_dim = mod.MultiUAVCoverageEnv(
        dataset, NUM_UAVS, env_type="terrain_plain", max_steps=MAX_STEPS,
        max_speed=MAX_SPEED, obstacle_ratio=0.08, use_failure_module=False,
        region_reassign_interval=4, num_users=None).state_dim
    model = mod.TmarlActorCritic(state_dim, config, variant="tmarl")
    model.load_state_dict(ckpt["state_dict"])
    return model, config, dataset


def fresh_model(config, dataset):
    """Randomly initialized CAST-MARL model with the same architecture."""
    state_dim = mod.MultiUAVCoverageEnv(
        dataset, NUM_UAVS, env_type="terrain_plain", max_steps=MAX_STEPS,
        max_speed=MAX_SPEED, obstacle_ratio=0.08, use_failure_module=False,
        region_reassign_interval=4, num_users=None).state_dim
    return mod.TmarlActorCritic(state_dim, config, variant="tmarl")


def make_env(config, dataset, env_type, obstacle_ratio):
    return mod.MultiUAVCoverageEnv(
        dataset=dataset, num_uavs=NUM_UAVS, env_type=env_type,
        max_steps=MAX_STEPS, max_speed=MAX_SPEED,
        obstacle_ratio=obstacle_ratio,
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


def train_multidomain(model, config, dataset, device):
    """PPO training with per-episode terrain sampling. Network and reward
    logic are identical to the single-terrain training; only the environment
    type changes between episodes."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=OVERRIDE_LR)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    terrain_cycle = TERRAIN_MIX * EPISODES_PER_TERRAIN
    random.shuffle(terrain_cycle)
    train_log = []
    start = time.time()
    for ep, (env_type, label, ratio) in enumerate(terrain_cycle):
        env = make_env(config, dataset, env_type, ratio)
        mod.set_seed(TRAIN_SEED + ep * 1000)
        buffers = []
        batch_metrics = []
        for _ in range(BATCH_EPISODES):
            ep_reward, info, _, buffer = mod.collect_episode(
                env, model, device, deterministic=False)
            returns, advantages = mod.compute_returns_advantages(
                buffer.rewards, buffer.values, buffer.dones,
                config.gamma, gae_lambda=config.gae_lambda, last_value=0.0)
            buffers.append({
                "obs": np.asarray(buffer.obs, dtype=np.float32),
                "actions": np.asarray(buffer.actions, dtype=np.float32),
                "old_log_probs": np.asarray(buffer.log_probs, dtype=np.float32),
                "returns": returns.astype(np.float32),
                "advantages": advantages.astype(np.float32),
            })
            batch_metrics.append((ep_reward, info))
        batch_obs = torch.tensor(
            np.concatenate([b["obs"] for b in buffers], axis=0),
            dtype=torch.float32, device=device)
        batch_actions = torch.tensor(
            np.concatenate([b["actions"] for b in buffers], axis=0),
            dtype=torch.float32, device=device)
        batch_old_log_probs = torch.tensor(
            np.concatenate([b["old_log_probs"] for b in buffers], axis=0),
            dtype=torch.float32, device=device)
        batch_returns = torch.tensor(
            np.concatenate([b["returns"] for b in buffers], axis=0),
            dtype=torch.float32, device=device)
        batch_advantages = torch.tensor(
            np.concatenate([b["advantages"] for b in buffers], axis=0),
            dtype=torch.float32, device=device)
        total_steps = batch_obs.shape[0]
        mini_batch = min(MINI_BATCH_SIZE, total_steps)
        progress = ep / max(1, len(terrain_cycle) - 1)
        entropy_coef_t = config.entropy_coef * (1.0 - 0.70 * progress)
        lr_t = OVERRIDE_LR * (1.0 - 0.60 * progress)
        for group in optimizer.param_groups:
            group["lr"] = max(1e-5, lr_t)
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(total_steps, device=device)
            for s in range(0, total_steps, mini_batch):
                idx = perm[s:s + mini_batch]
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
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS,
                                    1.0 + CLIP_EPS) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.nn.functional.mse_loss(value, ret_mb)
                loss = (actor_loss + config.value_coef * critic_loss
                        - entropy_coef_t * entropy)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm)
                optimizer.step()
        avg_rew = float(np.mean([m[0] for m in batch_metrics]))
        avg_cov = float(np.mean([m[1]["coverage_rate"] for m in batch_metrics])) * 100
        train_log.append({
            "episode": ep, "terrain": label, "reward": avg_rew,
            "coverage": avg_cov, "loss": float(loss.item()),
        })
        print(f"ep {ep + 1}/{len(terrain_cycle)} [{label}] "
              f"rew={avg_rew:.2f} cov={avg_cov:.2f} loss={loss.item():.4f}",
              flush=True)
    model.eval()
    return model, train_log, time.time() - start


def evaluate(model, config, dataset, device):
    """Evaluate on each of the four terrains under the cross-map protocol."""
    rows = []
    for env_type, label, ratio in TERRAIN_MIX:
        for seed in EVAL_SEEDS:
            for ep in range(EVAL_EPISODES_PER_SEED):
                mod.set_seed(seed * 1000 + ep)
                env = make_env(config, dataset, env_type, ratio)
                ep_reward, info, _, _ = mod.collect_episode(
                    env, model, device, deterministic=True)
                rows.append({
                    "target_map": label, "seed": seed, "episode": ep,
                    "coverage": info["coverage_rate"] * 100.0,
                    "conflict_rate": info["conflict_rate"],
                    "reward": ep_reward,
                    "path_length": info["path_length"],
                })
    return rows


def summarize(rows, tag):
    out = []
    for _, label, _ in TERRAIN_MIX:
        g = [r for r in rows if r["target_map"] == label]
        out.append({
            "model": tag, "target_map": label,
            **{f"{k}_mean": round(float(np.mean([x[k] for x in g])), 4)
               for k in ("coverage", "conflict_rate", "reward", "path_length")},
            **{f"{k}_std": round(float(np.std([x[k] for x in g], ddof=1)), 4)
               for k in ("coverage", "conflict_rate", "reward", "path_length")},
            "n": len(g),
        })
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    from_scratch = "--from_scratch" in sys.argv
    if from_scratch:
        _, config, dataset = load_checkpoint()
        model = fresh_model(config, dataset)
        print("training from scratch (random init)")
    else:
        model, config, dataset = load_checkpoint()
        print("continuing from single-domain checkpoint")
    model.to(device)
    model, train_log, train_time = train_multidomain(model, config, dataset, device)

    # Save multi-domain checkpoint
    ckpt_path = PACKAGE / "experimental_runs" / "multidomain_uav_best.pth"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {k: getattr(config, k) for k in
                   mod.ExperimentConfig.__dataclass_fields__},
        "variant": "tmarl",
        "train_tag": "multi-domain 4-terrain mix",
    }, ckpt_path)
    print("saved", ckpt_path)

    rows = evaluate(model, config, dataset, device)
    summary = summarize(rows, "multi-domain")
    with (OUT / "CrossMap_MultiDomain_ByEpisode.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "CrossMap_MultiDomain_Summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    max_mem = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else 0.0)
    (OUT / "CrossMap_MultiDomain_Protocol.json").write_text(json.dumps({
        "training": {
            "episodes_per_terrain": EPISODES_PER_TERRAIN,
            "total_episodes": TOTAL_EPISODES,
            "batch_episodes": BATCH_EPISODES,
            "ppo_epochs": PPO_EPOCHS,
            "mini_batch_size": MINI_BATCH_SIZE,
            "clip_eps": CLIP_EPS,
            "lr": OVERRIDE_LR,
            "terrain_mix": [l for _, l, _ in TERRAIN_MIX],
            "seed": TRAIN_SEED,
            "train_seconds": round(train_time, 1),
            "max_gpu_memory_mb": round(max_mem, 1),
        },
        "evaluation": {
            "seeds": EVAL_SEEDS,
            "episodes_per_seed": EVAL_EPISODES_PER_SEED,
            "protocol": "cross-map, deterministic rollout",
            "metrics": "unchanged simulator step metrics",
        },
        "note": "only the data supply (per-episode terrain sampling) changed; "
                "network, reward, and CVA-SP logic unchanged.",
    }, indent=2), encoding="utf-8")
    print("TRAIN_LOG:")
    for r in train_log:
        print(r)
    print("SUMMARY:")
    for s in summary:
        print(s)
    print("MULTI-DOMAIN TRAINING DONE")


if __name__ == "__main__":
    main()
