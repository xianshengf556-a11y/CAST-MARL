# -*- coding: utf-8 -*-
"""Target-domain fine-tuning ablation for CAST-MARL.

Protocol
--------
The archived plain-trained checkpoint (seed 42, terrain_plain) is used as the
zero-shot baseline.  From the target domain (terrain_urban, obstacle ratio
0.20), a small number of training episodes are drawn and used to continue
RL training of the same model (PPO-style updates with GAE).  The test set is
the unchanged cross-map protocol: plain / urban / mountain / mountain-dense,
5 seeds x 12 episodes each, same obstacle layouts and start configurations as
the reported CrossMap_Summary.csv.  We report coverage, conflict rate, reward
and path length for the zero-shot checkpoint and for fine-tuned checkpoints
with 5 and 10 target-domain episodes.

The experiment measures the effect of a small amount of target-domain data
on cross-domain generalization while keeping the test distribution fixed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import copy
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal


_CROSS_SPEC = importlib.util.spec_from_file_location(
    "cross_map_generalization",
    Path(__file__).resolve().parent / "cross_map_generalization.py",
)
cross = importlib.util.module_from_spec(_CROSS_SPEC)
sys.modules[_CROSS_SPEC.name] = cross
assert _CROSS_SPEC.loader is not None
_CROSS_SPEC.loader.exec_module(cross)

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

SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
NUM_UAVS = 6
MAX_STEPS = 35
MAX_SPEED = 150.0
FINETUNE_EPISODES = [0, 5, 10]          # 0 = zero-shot baseline
FINETUNE_SEED = 2026                     # deterministic fine-tune rollouts
PPO_EPOCHS = 3
CLIP_EPS = 0.2


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
        max_speed=MAX_SPEED, obstacle_ratio=0.12, use_failure_module=False,
        region_reassign_interval=4, num_users=None).state_dim
    model = mod.TmarlActorCritic(state_dim, config, variant="tmarl")
    model.load_state_dict(ckpt["state_dict"])
    return model, config, dataset


def finetune(model, config, dataset, n_episodes, device):
    """Continue RL training on target-domain (urban) episodes."""
    if n_episodes <= 0:
        return model
    model.train()
    env = mod.MultiUAVCoverageEnv(
        dataset, NUM_UAVS, env_type="terrain_urban", max_steps=MAX_STEPS,
        max_speed=MAX_SPEED, obstacle_ratio=0.20, use_failure_module=False,
        region_reassign_interval=config.region_reassign_interval,
        num_users=None,
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
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for ep in range(n_episodes):
        mod.set_seed(FINETUNE_SEED + ep)
        buffers = []
        for _ in range(config.batch_episodes):
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
        mini_batch = min(config.mini_batch_size, total_steps)
        progress = ep / max(1, n_episodes - 1)
        entropy_coef_t = config.entropy_coef * (1.0 - 0.70 * progress)
        lr_t = config.lr * (1.0 - 0.60 * progress)
        for group in optimizer.param_groups:
            group["lr"] = max(1e-5, lr_t)
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(total_steps, device=device)
            for start in range(0, total_steps, mini_batch):
                idx = perm[start:start + mini_batch]
                obs_mb = batch_obs[idx]
                act_mb = batch_actions[idx]
                old_logp_mb = batch_old_log_probs[idx]
                ret_mb = batch_returns[idx]
                adv_mb = batch_advantages[idx]
                mean, log_std, value = model(obs_mb)
                std = log_std.exp()
                raw_action = torch.atanh(
                    torch.clamp(act_mb, -0.999, 0.999))
                dist = Normal(mean, std)
                log_prob = dist.log_prob(raw_action).sum(dim=-1).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).sum(dim=-1).mean()
                ratio = torch.exp(log_prob - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(
                    ratio, 1.0 - config.ppo_clip,
                    1.0 + config.ppo_clip) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.nn.functional.mse_loss(value, ret_mb)
                loss = (actor_loss + config.value_coef * critic_loss
                        - entropy_coef_t * entropy)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm)
                optimizer.step()
        print(f"finetune batch {ep + 1}/{n_episodes} | "
              f"reward={ep_reward:.2f} "
              f"coverage={info['coverage_rate'] * 100:.2f} | "
              f"loss={loss.item():.4f}", flush=True)
    model.eval()
    return model


def evaluate(model, config, dataset, device):
    rows = []
    targets = [
        ("terrain_plain", "plain", 0.08),
        ("terrain_urban", "urban", 0.20),
        ("terrain_mountain", "mountain", 0.16),
        ("terrain_mountain", "mountain-dense", 0.30),
    ]
    for env_type, label, obstacle_ratio in targets:
        for seed in SEEDS:
            for ep in range(EPISODES_PER_SEED):
                result = run_marl_episode_simple(
                    seed * 1000 + ep, dataset, env_type, obstacle_ratio,
                    model, config, device)
                rows.append({
                    "target_map": label,
                    "seed": seed,
                    "episode": ep,
                    "coverage": result["coverage"],
                    "conflict_rate": result["conflict_rate"],
                    "reward": result["reward"],
                    "path_length": result["path_length"],
                })
    return rows


def run_marl_episode_simple(seed, dataset, env_type, obstacle_ratio,
                            model, config, device):
    """Delegate to cross_map_generalization.run_marl_episode so the test-set
    evaluation is byte-identical to the reported zero-shot baseline."""
    return cross.run_marl_episode(seed, dataset, env_type, obstacle_ratio,
                                  model, config, torch.device("cpu"))


def summarize(rows, label):
    out = []
    for target_map in ["plain", "urban", "mountain", "mountain-dense"]:
        g = [r for r in rows if r["target_map"] == target_map]
        out.append({
            "finetune_episodes": label,
            "target_map": target_map,
            **{f"{k}_mean": round(float(np.mean([x[k] for x in g])), 4)
               for k in ("coverage", "conflict_rate", "reward", "path_length")},
            **{f"{k}_std": round(float(np.std([x[k] for x in g], ddof=1)), 4)
               for k in ("coverage", "conflict_rate", "reward", "path_length")},
            "n": len(g),
        })
    return out


def main():
    device = torch.device("cpu")
    print("device:", device)
    base_model, config, dataset = load_checkpoint()
    all_rows = []
    summaries = []
    for n in FINETUNE_EPISODES:
        model = copy.deepcopy(base_model)
        model.to(device)
        model = finetune(model, config, dataset, n, device)
        model.eval()
        rows = evaluate(model, config, dataset, device)
        for r in rows:
            all_rows.append({"finetune_episodes": n, **r})
        summaries.extend(summarize(rows, n))
        print(f"done finetune_episodes={n}", flush=True)

    with (OUT / "CrossMap_TargetFinetune_ByEpisode.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with (OUT / "CrossMap_TargetFinetune_Summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    (OUT / "CrossMap_TargetFinetune_Protocol.json").write_text(
        json.dumps({
            "checkpoint": str(CHECKPOINT.name),
            "base_train_env": "terrain_plain (seed 42)",
            "target_finetune_env": "terrain_urban (obstacle ratio 0.20)",
            "finetune_episodes": FINETUNE_EPISODES,
            "finetune_seed": FINETUNE_SEED,
            "ppo_epochs": PPO_EPOCHS,
            "clip_eps": CLIP_EPS,
            "test_set": "unchanged cross-map protocol (plain/urban/mountain/"
                        "mountain-dense, 5 seeds x 12 episodes each)",
            "metrics": "unchanged simulator step metrics",
        }, indent=2), encoding="utf-8")
    print("TARGET FINETUNE ABLATION DONE")


if __name__ == "__main__":
    main()
