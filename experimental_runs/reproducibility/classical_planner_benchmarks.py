"""Executable A*, PRM*, RRT* and CBS baselines for the CAST-MARL simulator.

The planners operate on a 2-D occupancy grid extracted from each simulator
episode. The resulting actions are passed through the unchanged simulator
step function, so coverage, path length, conflict rate and reward use the same
metric implementation as the learning methods.
"""

from __future__ import annotations

import csv
import heapq
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / "experimental_runs" / "reproducibility" / "cast_marl_experiment_source.py"
spec = importlib.util.spec_from_file_location("cast_marl_source", SOURCE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

# The local reproducibility bundle does not include the remote terrain manifest.
# The protocol therefore uses the simulator's terrain_urban label with its
# episode-generated urban obstacle set, keeping the obstacle generator active.
mod.TERRAIN_MANIFEST_PATH = str(PACKAGE / "experimental_runs" / "reproducibility" / "missing_terrain_manifest.json")
mod._TERRAIN_CACHE.clear()

DATASET = [
    str(PACKAGE / "dataset" / "dataset1_users35.npz"),
    str(PACKAGE / "dataset" / "dataset1_users45.npz"),
    str(PACKAGE / "dataset" / "dataset1_users60.npz"),
]
OUT = PACKAGE / "experimental_runs" / "reproducibility"
SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
NUM_UAVS = 6
MAX_STEPS = 35
GRID_N = 20
INFLATION = 135.0
MAX_SPEED = 150.0
ALT_TARGET = 120.0


def cell_xy(cell):
    return np.array([(cell[0] + 0.5) * mod.AREA_X / GRID_N, (cell[1] + 0.5) * mod.AREA_Y / GRID_N], dtype=np.float32)


def xy_cell(xy):
    return (int(np.clip(xy[0] / mod.AREA_X * GRID_N, 0, GRID_N - 1)), int(np.clip(xy[1] / mod.AREA_Y * GRID_N, 0, GRID_N - 1)))


def neighbours(cell):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        q = (x + dx, y + dy)
        if 0 <= q[0] < GRID_N and 0 <= q[1] < GRID_N:
            yield q


def segment_cells(a, b):
    n = max(abs(a[0] - b[0]), abs(a[1] - b[1]), 1)
    return [(round(a[0] + (b[0] - a[0]) * i / n), round(a[1] + (b[1] - a[1]) * i / n)) for i in range(n + 1)]


def occupancy(env):
    occ = np.zeros((GRID_N, GRID_N), dtype=bool)
    for gx in range(GRID_N):
        for gy in range(GRID_N):
            p = cell_xy((gx, gy))
            for obs in env.obstacles:
                x0, y0, _, x1, y1, _ = obs
                if (x0 - INFLATION <= p[0] <= x1 + INFLATION and y0 - INFLATION <= p[1] <= y1 + INFLATION):
                    occ[gx, gy] = True
                    break
    return occ


def nearest_free(cell, occ):
    if not occ[cell]:
        return cell
    for radius in range(1, GRID_N):
        for x in range(max(0, cell[0] - radius), min(GRID_N, cell[0] + radius + 1)):
            for y in range(max(0, cell[1] - radius), min(GRID_N, cell[1] + radius + 1)):
                if not occ[x, y]:
                    return (x, y)
    return cell


def valid_edge(a, b, occ, reservations=None, t0=0):
    if occ[b] or any(occ[c] for c in segment_cells(a, b)):
        return False
    if reservations is None:
        return True
    for dt, c in enumerate(segment_cells(a, b)):
        t = t0 + dt
        if (t, c) in reservations or (t, b, c) in reservations:
            return False
    return True


def astar(start, goal, occ, reservations=None, max_time=None):
    start, goal = nearest_free(start, occ), nearest_free(goal, occ)
    if start == goal:
        return [start]
    def h(a):
        return math.hypot(a[0] - goal[0], a[1] - goal[1])
    open_set = [(h(start), 0.0, 0, start)]
    parent, gscore = {}, {start: 0.0}
    serial = 0
    while open_set:
        _, g, t, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return list(reversed(path))
        if max_time is not None and t >= max_time:
            continue
        for nxt in neighbours(cur):
            if reservations is not None and (t + 1, nxt) in reservations:
                continue
            if not valid_edge(cur, nxt, occ):
                continue
            ng = g + math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if ng < gscore.get(nxt, float("inf")):
                gscore[nxt] = ng
                parent[nxt] = cur
                serial += 1
                heapq.heappush(open_set, (ng + h(nxt), ng, t + 1, nxt))
    return [start]


def prm_star(start, goal, occ):
    start, goal = nearest_free(start, occ), nearest_free(goal, occ)
    free = [(x, y) for x in range(GRID_N) for y in range(GRID_N) if not occ[x, y]]
    samples = [start, goal] + random.sample(free, min(180, len(free)))
    graph = {p: [] for p in samples}
    for i, p in enumerate(samples):
        distances = sorted((math.hypot(p[0] - q[0], p[1] - q[1]), q) for q in samples if q != p)
        for _, q in distances[: min(12, len(distances))]:
            if valid_edge(p, q, occ):
                graph[p].append(q); graph[q].append(p)
    frontier = [(0.0, start)]; dist = {start: 0.0}; parent = {}
    while frontier:
        d, cur = heapq.heappop(frontier)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]; path.append(cur)
            return list(reversed(path))
        for nxt in graph[cur]:
            nd = d + math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd; parent[nxt] = cur; heapq.heappush(frontier, (nd, nxt))
    return astar(start, goal, occ)


def rrt_star(start, goal, occ):
    start, goal = nearest_free(start, occ), nearest_free(goal, occ)
    if start == goal:
        return [start]
    nodes = [start]; parent = {start: None}; cost = {start: 0.0}
    for _ in range(450):
        qrand = goal if random.random() < 0.12 else random.choice([(x, y) for x in range(GRID_N) for y in range(GRID_N) if not occ[x, y]])
        near = min(nodes, key=lambda q: (q[0] - qrand[0]) ** 2 + (q[1] - qrand[1]) ** 2)
        dx, dy = qrand[0] - near[0], qrand[1] - near[1]
        scale = max(abs(dx), abs(dy), 1)
        new = (int(round(near[0] + dx / scale)), int(round(near[1] + dy / scale)))
        if new == near or occ[new] or not valid_edge(near, new, occ) or new in parent:
            continue
        near_set = sorted(nodes, key=lambda q: (q[0] - new[0]) ** 2 + (q[1] - new[1]) ** 2)[:12]
        best = near
        best_cost = cost[near] + math.hypot(new[0] - near[0], new[1] - near[1])
        for q in near_set:
            c = cost[q] + math.hypot(new[0] - q[0], new[1] - q[1])
            if c < best_cost and valid_edge(q, new, occ):
                best, best_cost = q, c
        nodes.append(new); parent[new] = best; cost[new] = best_cost
        if math.hypot(new[0] - goal[0], new[1] - goal[1]) <= 1.5 and valid_edge(new, goal, occ):
            parent[goal] = new; nodes.append(goal); break
    if goal not in parent:
        return astar(start, goal, occ)
    path = [goal]
    visited = set()
    while path[-1] is not None and path[-1] in parent and path[-1] not in visited:
        visited.add(path[-1])
        path.append(parent[path[-1]])
    return list(reversed(path[:-1])) if len(path) > 1 else [start]


def timed_astar(start, goal, occ, vertex_constraints, edge_constraints, max_time):
    start, goal = nearest_free(start, occ), nearest_free(goal, occ)
    open_set = [(math.hypot(start[0] - goal[0], start[1] - goal[1]), 0.0, 0, start)]
    parent, gscore = {}, {(0, start): 0.0}
    moves = list(neighbours(start))
    while open_set:
        _, g, t, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]; state = (t, cur)
            while state in parent:
                state = parent[state]; path.append(state[1])
            return list(reversed(path))
        if t >= max_time:
            continue
        candidates = list(neighbours(cur)) + [cur]
        for nxt in candidates:
            if occ[nxt] or (t + 1, nxt) in vertex_constraints or (t, cur, nxt) in edge_constraints:
                continue
            ng = g + (0.05 if nxt == cur else math.hypot(nxt[0] - cur[0], nxt[1] - cur[1]))
            state = (t + 1, nxt)
            if ng < gscore.get(state, float("inf")):
                gscore[state] = ng; parent[state] = (t, cur)
                h = math.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
                heapq.heappush(open_set, (ng + h, ng, t + 1, nxt))
    return None


def first_cbs_conflict(paths):
    horizon = max(len(p) for p in paths)
    for t in range(horizon):
        positions = [p[min(t, len(p) - 1)] for p in paths]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if positions[i] == positions[j]:
                    return {"type": "vertex", "time": t, "a": i, "b": j, "cell": positions[i]}
                if t > 0:
                    prev = [p[min(t - 1, len(p) - 1)] for p in paths]
                    if prev[i] == positions[j] and prev[j] == positions[i]:
                        return {"type": "edge", "time": t - 1, "a": i, "b": j, "from": prev[i], "to": positions[i]}
    return None


def cbs_paths(starts, goals, occ):
    # Standard CBS: a high-level constraint tree branches on the first
    # conflict, while each child replans one agent with time-indexed A*.
    n = len(starts)
    root_v = [set() for _ in range(n)]
    root_e = [set() for _ in range(n)]
    root_paths = [timed_astar(s, g, occ, root_v[i], root_e[i], MAX_STEPS + GRID_N) for i, (s, g) in enumerate(zip(starts, goals))]
    if any(p is None for p in root_paths):
        return [astar(s, g, occ) for s, g in zip(starts, goals)]
    root_cost = sum(len(p) for p in root_paths)
    queue = [(root_cost, 0, root_paths, root_v, root_e)]
    serial = 0
    expansions = 0
    while queue and expansions < 80:
        _, _, paths, vcons, econs = heapq.heappop(queue)
        conflict = first_cbs_conflict(paths)
        if conflict is None:
            return paths
        expansions += 1
        for agent in (conflict["a"], conflict["b"]):
            child_v = [set(x) for x in vcons]
            child_e = [set(x) for x in econs]
            if conflict["type"] == "vertex":
                child_v[agent].add((conflict["time"], conflict["cell"]))
            else:
                if agent == conflict["a"]:
                    child_e[agent].add((conflict["time"], conflict["from"], conflict["to"]))
                else:
                    child_e[agent].add((conflict["time"], conflict["to"], conflict["from"]))
            new_path = timed_astar(starts[agent], goals[agent], occ, child_v[agent], child_e[agent], MAX_STEPS + GRID_N)
            if new_path is None:
                continue
            child_paths = list(paths); child_paths[agent] = new_path
            serial += 1
            heapq.heappush(queue, (sum(len(p) for p in child_paths), serial, child_paths, child_v, child_e))
    return [astar(s, g, occ) for s, g in zip(starts, goals)]


def target_for(env, i):
    support = env._assigned_uncovered(i)
    if len(support):
        return support[int(np.argmin(np.linalg.norm(support[:, :2] - env.uav_pos[i, :2], axis=1)))][:2]
    return env.region_centers[i, :2]


def action_from_paths(env, paths, planner):
    actions = np.zeros((env.num_uavs, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        if not path:
            continue
        cur = xy_cell(env.uav_pos[i, :2])
        idx = min(1, len(path) - 1)
        if path[0] != cur:
            idx = 0
        goal_xy = cell_xy(path[idx])
        vec = goal_xy - env.uav_pos[i, :2]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            vec = np.asarray(target_for(env, i), dtype=np.float32) - env.uav_pos[i, :2]
            norm = np.linalg.norm(vec)
        actions[i, :2] = vec / max(norm, 1e-6)
        actions[i, 2] = np.clip((ALT_TARGET - env.uav_pos[i, 2]) / 18.0, -1.0, 1.0)
    return actions


def run_episode(seed, planner, dataset):
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(dataset, NUM_UAVS, env_type="terrain_urban", max_steps=MAX_STEPS, max_speed=MAX_SPEED, obstacle_ratio=0.12, use_failure_module=False, region_reassign_interval=4, num_users=None)
    env.reset()
    occ = occupancy(env)
    paths = [[] for _ in range(NUM_UAVS)]
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        if planner == "CBS" or planner == "A*":
            starts = [xy_cell(p[:2]) for p in env.uav_pos]
            goals = [xy_cell(target_for(env, i)) for i in range(NUM_UAVS)]
            paths = cbs_paths(starts, goals, occ) if planner == "CBS" else [astar(s, g, occ) for s, g in zip(starts, goals)]
        elif planner == "PRM*":
            paths = [prm_star(xy_cell(env.uav_pos[i, :2]), xy_cell(target_for(env, i)), occ) for i in range(NUM_UAVS)]
        else:
            paths = [rrt_star(xy_cell(env.uav_pos[i, :2]), xy_cell(target_for(env, i)), occ) for i in range(NUM_UAVS)]
        _, reward, done, info = env.step(action_from_paths(env, paths, planner))
        total_reward += float(reward)
        if done:
            break
    return {"coverage": info["coverage_rate"] * 100.0, "conflict_rate": info["conflict_rate"], "reward": total_reward, "path_length": info["path_length"]}


def main():
    dataset = mod.ScenarioDataset(DATASET)
    rows = []
    for planner in ("A*", "PRM*", "RRT*", "CBS"):
        for seed in SEEDS:
            values = []
            for ep in range(EPISODES_PER_SEED):
                values.append(run_episode(seed * 1000 + ep, planner, dataset))
            for ep, val in enumerate(values):
                rows.append({"planner": planner, "seed": seed, "episode": ep, **val})
            print(planner, seed, {k: round(float(np.mean([v[k] for v in values])), 5) for k in ("coverage", "conflict_rate", "path_length")}, flush=True)
    with (OUT / "Classical_Planner_BySeed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = []
    for planner in sorted({r["planner"] for r in rows}):
        for seed in SEEDS:
            group = [r for r in rows if r["planner"] == planner and r["seed"] == seed]
            summary.append({"planner": planner, "seed": seed, **{f"{k}_mean": float(np.mean([x[k] for x in group])) for k in ("coverage", "conflict_rate", "reward", "path_length")}})
    with (OUT / "Classical_Planner_Seed_Summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    pooled = []
    for planner in sorted({r["planner"] for r in rows}):
        group = [r for r in rows if r["planner"] == planner]
        pooled.append({"planner": planner, **{f"{k}_mean": float(np.mean([x[k] for x in group])) for k in ("coverage", "conflict_rate", "reward", "path_length")}, **{f"{k}_std": float(np.std([x[k] for x in group], ddof=1)) for k in ("coverage", "conflict_rate", "reward", "path_length")}, "n": len(group)})
    with (OUT / "Classical_Planner_Summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pooled[0].keys()); writer.writeheader(); writer.writerows(pooled)
    (OUT / "Classical_Planner_Protocol.json").write_text(json.dumps({"planners": ["A*", "PRM*", "RRT*", "CBS"], "seeds": SEEDS, "episodes_per_seed": EPISODES_PER_SEED, "num_uavs": NUM_UAVS, "max_steps": MAX_STEPS, "grid_n": GRID_N, "obstacle_inflation": INFLATION, "environment": "terrain_urban label with episode-generated urban obstacle set", "metrics": "unchanged simulator step metrics"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
