#!/usr/bin/env python3
"""
Convert RoboMimic HDF5 dataset to LeRobot v2 format for NVIDIA GR00T-N1.6 finetuning.

FIX: GR00T globs for parquet files as "data/*/*.parquet" — files must be inside
     a subdirectory, e.g. data/chunk-000/episode_000000.parquet

Usage:
    /isaac-sim/kit/python/bin/python3 convert_robomimic_to_lerobot.py \
        --input /workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5 \
        --output /workspace/isaaclab/ImitationLearning/demonstrations/lerobot_dataset \
        --task-description "robot manipulation task" \
        --fps 20
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


INFO_FILE     = "meta/info.json"
STATS_FILE    = "meta/stats.json"
EPISODES_FILE = "meta/episodes.json"
TASKS_FILE    = "meta/tasks.json"
CHUNK_DIR     = "data/chunk-000"   # GR00T glob: data/*/*.parquet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-description", default="robot manipulation task")
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def compute_stats(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
        "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
        "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
    }


def main():
    args = parse_args()
    src  = Path(args.input)
    dst  = Path(args.output)

    assert src.exists(), f"Input not found: {src}"

    if dst.exists():
        print(f"[WARN] Removing existing output: {dst}")
        shutil.rmtree(dst)

    (dst / "meta").mkdir(parents=True)
    (dst / CHUNK_DIR).mkdir(parents=True)

    print(f"[INFO] Reading {src}")

    episodes_meta = []
    all_actions, all_ee, all_joints, all_states = [], [], [], []
    total_frames = 0

    with h5py.File(src, "r") as f:
        demo_keys  = sorted(f["data"].keys())
        n_episodes = len(demo_keys)
        print(f"[INFO] {n_episodes} episodes: {demo_keys}\n")

        for ep_idx, key in enumerate(demo_keys):
            g       = f["data"][key]
            actions = g["actions"][:]              # (T, 8)
            ee      = g["obs/ee_poses"][:]          # (T, 7)
            joints  = g["obs/joint_positions"][:]   # (T, 7)
            state   = g["obs/state"][:]             # (T, 22)
            dones   = g["dones"][:]
            rewards = g["rewards"][:]
            T       = actions.shape[0]

            print(f"  ep {ep_idx:02d} ({key}): {T} frames")

            rows = []
            for t in range(T):
                rows.append({
                    "episode_index":               ep_idx,
                    "frame_index":                 t,
                    "timestamp":                   round(t / args.fps, 6),
                    "task_index":                  0,
                    "action":                      actions[t].tolist(),
                    "observation.state":           state[t].tolist(),
                    "observation.ee_pose":         ee[t].tolist(),
                    "observation.joint_positions": joints[t].tolist(),
                    "next.done":                   bool(dones[t]),
                    "next.reward":                 float(rewards[t]),
                })

            df = pd.DataFrame(rows)
            out_path = dst / CHUNK_DIR / f"episode_{ep_idx:06d}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"    → {out_path}")

            episodes_meta.append({
                "episode_index": ep_idx,
                "tasks":         [args.task_description],
                "length":        T,
            })

            all_actions.append(actions)
            all_ee.append(ee)
            all_joints.append(joints)
            all_states.append(state)
            total_frames += T

    # ── Stats ────────────────────────────────────────────────────────────────
    print("\n[INFO] Computing statistics...")
    stats = {
        "action":                      compute_stats(np.concatenate(all_actions)),
        "observation.state":           compute_stats(np.concatenate(all_states)),
        "observation.ee_pose":         compute_stats(np.concatenate(all_ee)),
        "observation.joint_positions": compute_stats(np.concatenate(all_joints)),
    }

    with open(dst / STATS_FILE,    "w") as f: json.dump(stats,         f, indent=2)
    with open(dst / EPISODES_FILE, "w") as f: json.dump(episodes_meta, f, indent=2)
    with open(dst / TASKS_FILE,    "w") as f:
        json.dump([{"task_index": 0, "task": args.task_description}], f, indent=2)

    info = {
        "codebase_version": "v2.0",
        "robot_type":       "custom",
        "total_episodes":   n_episodes,
        "total_frames":     total_frames,
        "fps":              args.fps,
        "splits":           {"train": f"0:{n_episodes}"},
        "data_path":        f"{CHUNK_DIR}/episode_{{episode_index:06d}}.parquet",
        "features": {
            "action": {
                "dtype": "float32", "shape": [8],
                "names": ["joint_0","joint_1","joint_2","joint_3",
                          "joint_4","joint_5","joint_6","gripper"],
            },
            "observation.state": {
                "dtype": "float32", "shape": [22],
                "names": [f"state_{i}" for i in range(22)],
            },
            "observation.ee_pose": {
                "dtype": "float32", "shape": [7],
                "names": ["x","y","z","qx","qy","qz","qw"],
            },
            "observation.joint_positions": {
                "dtype": "float32", "shape": [7],
                "names": [f"joint_{i}" for i in range(7)],
            },
        },
    }
    with open(dst / INFO_FILE, "w") as f: json.dump(info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅  Conversion complete!")
    print(f"   Output   : {dst}")
    print(f"   Episodes : {n_episodes}  |  Frames: {total_frames}")
    print(f"   Parquet  : {dst / CHUNK_DIR}/*.parquet")
    print(f"{'='*60}")
    print(f"""
Run training from /workspace/Isaac-GR00T:

  cd /workspace/Isaac-GR00T
  TRANSFORMERS_ATTN_IMPLEMENTATION=eager \\
  /isaac-sim/kit/python/bin/python3 gr00t/experiment/launch_finetune.py \\
      --base-model-path "nvidia/GR00T-N1.6-3B" \\
      --dataset-path "{dst}" \\
      --embodiment-tag "NEW_EMBODIMENT" \\
      --modality-config-path "/workspace/Isaac-GR00T/gr00t/configs/data/custom_embodiment.py" \\
      --output-dir "/workspace/isaaclab/ImitationLearning/checkpoints" \\
      --global-batch-size 64 \\
      --learning-rate 1e-4 \\
      --max-steps 10000
""")


if __name__ == "__main__":
    main()