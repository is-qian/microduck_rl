"""Evaluate a headstand-exit checkpoint against the deployment spec:

headstand spawn (inverted, head-top on floor) → smooth exit → stand and hold.

Per env:
  reached    — stand gate (upright within 20 deg, feet on floor, head off
               floor, root z at standing height) sustained >= 0.5 s within
               the first 5 s
  stable     — after the first sustained stand, stand gate active >= 95% of
               the remaining steps until episode end
  flight-free — no sustained run without ANY robot-body ground contact
               (a body-supported roll keeps contact; only a real hop reads
               as fully unsupported), after the 0.3 s spawn settle window

Usage:
  uv run python scripts/eval_headstand_exit.py --checkpoint <model.pt> [--num-envs 256] [--video]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

TASK_ID = "Mjlab-HeadstandExit-Flat-MicroDuck"
STAND_COS_MIN = math.cos(math.radians(20.0))
STAND_Z = 0.115
STAND_Z_TOL = 0.02
ENTRY_WINDOW_S = 5.0
EPISODE_S = 8.0
SUSTAIN_S = 0.5
STABLE_FRAC = 0.95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--video", action="store_true", help="record env 0 (exit + first stand seconds)")
    ap.add_argument("--video-dir", type=Path, default=Path("logs/eval_videos"))
    ap.add_argument("--video-length", type=int, default=400)
    args = ap.parse_args()

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    # Whole-robot ground contact: the true support signal (feet/head sensors
    # alone read a body-supported fold phase as unsupported).
    existing = {s.name for s in env_cfg.scene.sensors}
    extra = () if "body_ground_contact" in existing else (
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + extra

    device = "cuda:0"
    if args.video:
        env_cfg.viewer.width = 1280
        env_cfg.viewer.height = 720
    env = ManagerBasedRlEnv(
        cfg=env_cfg, device=device, render_mode="rgb_array" if args.video else None
    )
    if args.video:
        env = VideoRecorder(
            env,
            video_folder=args.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID)
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    raw = env.unwrapped
    hz = 1.0 / raw.step_dt
    total_steps = int(EPISODE_S * hz)
    entry_steps = int(ENTRY_WINDOW_S * hz)
    sustain_steps = int(SUSTAIN_S * hz)

    n = args.num_envs
    stand_hist = torch.zeros(n, total_steps, dtype=torch.bool, device=device)
    settle_steps = int(0.3 * hz)
    flight_runs = torch.zeros(n, dtype=torch.long, device=device)
    cur_run = torch.zeros(n, dtype=torch.long, device=device)

    obs, _ = env.reset()
    with torch.inference_mode():
        for step in range(total_steps):
            obs = policy(obs)
            obs, _, _, _ = env.step(obs)

            asset = raw.scene["robot"]
            quat = asset.data.root_link_quat_w
            cos_up = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
            root_z = asset.data.root_link_pos_w[:, 2] - raw.scene.terrain.env_origins[:, 2]
            head = raw.scene.sensors["head_ground_contact"].data.force
            head_force = torch.nan_to_num(torch.linalg.norm(head, dim=-1), nan=0.0)
            if head_force.dim() > 1:
                head_force = head_force.sum(dim=-1)
            feet = raw.scene.sensors["feet_ground_contact"].data.found
            feet_found = torch.nan_to_num(feet, nan=0.0)
            if feet_found.dim() > 1:
                feet_found = feet_found.sum(dim=-1)
            feet_down = feet_found > 0.0
            head_down = head_force > 1.0
            stand = (
                (cos_up > STAND_COS_MIN)
                & feet_down
                & ~head_down
                & (root_z > STAND_Z - STAND_Z_TOL)
                & (root_z < STAND_Z + STAND_Z_TOL)
            )
            stand_hist[:, step] = stand
            if step >= settle_steps:
                body_found = raw.scene.sensors["body_ground_contact"].data.found
                body_found = torch.nan_to_num(body_found, nan=0.0)
                if body_found.dim() > 1:
                    body_found = body_found.sum(dim=-1)
                unsupported = body_found <= 0.0
                cur_run = torch.where(unsupported, cur_run + 1, torch.zeros_like(cur_run))
                flight_runs = torch.maximum(flight_runs, cur_run)

            if step % int(2 * hz) == 0:
                print(f"t={step / hz:5.1f}s  standing: {stand.sum().item():4d}/{n}")

    # reached: >= sustain_steps of stand within the entry window
    cum_entry = stand_hist[:, :entry_steps].cumsum(dim=1)
    reached = (cum_entry >= sustain_steps).any(dim=1)

    # stable: from the first step where the trailing window was all-stand,
    # >=95% stand over the rest of the episode
    sustained_start = torch.full((n,), -1, dtype=torch.long, device=device)
    rolling = torch.nn.functional.conv1d(
        stand_hist.long().float().unsqueeze(1),
        torch.ones(1, 1, sustain_steps, device=device),
    ).squeeze(1)
    sustained = rolling >= sustain_steps
    for i in range(n):
        idx = sustained[i].nonzero()
        if len(idx) > 0:
            sustained_start[i] = idx[0]
    has_stand = sustained_start >= 0
    exit_times = sustained_start[has_stand].float() / hz
    stable = torch.zeros(n, dtype=torch.bool, device=device)
    for i in range(n):
        if not has_stand[i]:
            continue
        start = int(sustained_start[i]) + sustain_steps
        rest = stand_hist[i, start:]
        if len(rest) > 0 and rest.float().mean() >= STABLE_FRAC:
            stable[i] = True

    print(f"\n=== {args.checkpoint.name} | {n} envs | headstand start, {EPISODE_S:.0f} s ===")
    print(f"reached stand within {ENTRY_WINDOW_S:.0f}s  : {reached.sum().item():4d}/{n} ({reached.float().mean():.1%})")
    print(f"stable stand to end         : {stable.sum().item():4d}/{n} ({stable.float().mean():.1%})")
    flight_free = flight_runs <= 2  # >2 unsupported steps (~40 ms) = real hop
    print(f"flight-free exit (<=2 steps): {flight_free.sum().item():4d}/{n} ({flight_free.float().mean():.1%})")
    print(f"max unsupported run (steps) : median {flight_runs.float().median():.0f}  p90 {flight_runs.float().quantile(0.9):.0f}  max {flight_runs.max().item()}")
    if has_stand.any():
        print(f"time to stand               : median {exit_times.median():.2f}s  p90 {exit_times.quantile(0.9):.2f}s")
    both = reached & stable & flight_free
    print(f"SUCCESS (all three)         : {both.sum().item():4d}/{n} ({both.float().mean():.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
