#!/usr/bin/env python3
"""
Convert RoboMimic HDF5 dataset to LeRobot format for NVIDIA GR00T-N1.6 finetuning.

Usage:
    /isaac-sim/kit/python/bin/python3 convert_robomimic_to_lerobot.py \
        --input /workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5 \
        --output /workspace/isaaclab/ImitationLearning/demonstrations/lerobot_dataset

Dataset structure detected:
    - actions:          (N, 8)  float32  — 7 joint targets + 1 gripper
    - obs/ee_poses:     (N, 7)  float32  — end-effector pos (3) + quat (4)
    - obs/joint_positions: (N, 7) float32 — joint angles
    - obs/state:        (N, 22) float32  — full state vector
    - dones:            (N,)    float64
    - rewards:          (N,)    float64
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


# ── LeRobot v2 filenames ────────────────────────────────────────────────────
LEROBOT_INFO_FILENAME   = "meta/info.json"
LEROBOT_STATS_FILENAME  = "meta/stats.json"
LEROBOT_EPISODES_FILENAME = "meta/episodes.json"
LEROBOT_TASKS_FILENAME  = "meta/tasks.json"
DATA_DIR                = "data"
CHUNK_SIZE              = 1000   # episodes per parquet chunk


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RoboMimic HDF5 → LeRobot v2")
    parser.add_argument("--input",  required=True, help="Path to robomimic_dataset.hdf5")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--task-description", default="robot manipulation task",
                        help="Natural language task description (used as language annotation)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Dataset collection framerate (default: 20 Hz)")
    return parser.parse_args()


def compute_stats(all_values: np.ndarray) -> dict:
    """Compute mean/std/min/max statistics for a modality array."""
    return {
        "mean": all_values.mean(axis=0).tolist(),
        "std":  all_values.std(axis=0).tolist(),
        "min":  all_values.min(axis=0).tolist(),
        "max":  all_values.max(axis=0).tolist(),
    }


def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    assert input_path.exists(), f"Input file not found: {input_path}"

    # ── Clean and create output directory ──────────────────────────────────
    if output_path.exists():
        print(f"[WARN] Output directory exists, removing: {output_path}")
        shutil.rmtree(output_path)

    (output_path / "meta").mkdir(parents=True)
    (output_path / DATA_DIR).mkdir(parents=True)

    print(f"[INFO] Reading: {input_path}")

    with h5py.File(input_path, "r") as f:
        demo_keys = sorted(f["data"].keys())  # demo_0, demo_1, ...
        n_episodes = len(demo_keys)
        print(f"[INFO] Found {n_episodes} episodes: {demo_keys}")

        # ── Collect all data ────────────────────────────────────────────────
        episodes_data   = []   # list of dicts per episode
        all_actions     = []
        all_ee_poses    = []
        all_joint_pos   = []
        all_states      = []

        total_frames = 0

        for ep_idx, key in enumerate(demo_keys):
            grp     = f["data"][key]
            actions = grp["actions"][:]              # (T, 8)
            ee      = grp["obs/ee_poses"][:]         # (T, 7)
            joints  = grp["obs/joint_positions"][:]  # (T, 7)
            state   = grp["obs/state"][:]            # (T, 22)
            dones   = grp["dones"][:]
            rewards = grp["rewards"][:]

            T = actions.shape[0]
            print(f"  Episode {ep_idx:3d} ({key}): {T} frames")

            episodes_data.append({
                "episode_index": ep_idx,
                "tasks":  [args.task_description],
                "length": T,
                "actions":     actions,
                "ee_poses":    ee,
                "joint_positions": joints,
                "state":       state,
                "dones":       dones,
                "rewards":     rewards,
            })

            all_actions.append(actions)
            all_ee_poses.append(ee)
            all_joint_pos.append(joints)
            all_states.append(state)
            total_frames += T

    print(f"\n[INFO] Total frames across all episodes: {total_frames}")

    # ── Write data/episode_XXXXXX.jsonl files ───────────────────────────────
    # LeRobot stores each episode as a JSONL file (one JSON object per frame)
    print("[INFO] Writing episode data files...")

    for ep in episodes_data:
        ep_idx = ep["episode_index"]
        ep_dir = output_path / DATA_DIR
        ep_file = ep_dir / f"episode_{ep_idx:06d}.jsonl"

        T = ep["length"]
        with open(ep_file, "w") as fout:
            for t in range(T):
                frame = {
                    "episode_index":    ep_idx,
                    "frame_index":      t,
                    "timestamp":        round(t / args.fps, 6),
                    "task":             ep["tasks"][0],

                    # Actions
                    "action":           ep["actions"][t].tolist(),

                    # Observations
                    "observation.state": ep["state"][t].tolist(),
                    "observation.ee_pose": ep["ee_poses"][t].tolist(),
                    "observation.joint_positions": ep["joint_positions"][t].tolist(),

                    # Episode metadata
                    "next.done":        bool(ep["dones"][t]),
                    "next.reward":      float(ep["rewards"][t]),
                }
                fout.write(json.dumps(frame) + "\n")

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(episodes_data) - 1:
            print(f"  Written episode {ep_idx + 1}/{len(episodes_data)}")

    # ── Compute statistics ──────────────────────────────────────────────────
    print("[INFO] Computing dataset statistics...")

    all_actions_np  = np.concatenate(all_actions,  axis=0)
    all_ee_np       = np.concatenate(all_ee_poses, axis=0)
    all_joints_np   = np.concatenate(all_joint_pos, axis=0)
    all_states_np   = np.concatenate(all_states,   axis=0)

    stats = {
        "action":                        compute_stats(all_actions_np),
        "observation.state":             compute_stats(all_states_np),
        "observation.ee_pose":           compute_stats(all_ee_np),
        "observation.joint_positions":   compute_stats(all_joints_np),
    }

    # ── Write meta/stats.json ───────────────────────────────────────────────
    stats_path = output_path / LEROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Wrote {stats_path}")

    # ── Write meta/episodes.json ────────────────────────────────────────────
    episodes_meta = [
        {
            "episode_index": ep["episode_index"],
            "tasks":         ep["tasks"],
            "length":        ep["length"],
        }
        for ep in episodes_data
    ]
    ep_path = output_path / LEROBOT_EPISODES_FILENAME
    with open(ep_path, "w") as f:
        json.dump(episodes_meta, f, indent=2)
    print(f"[INFO] Wrote {ep_path}")

    # ── Write meta/tasks.json ───────────────────────────────────────────────
    tasks_meta = [{"task_index": 0, "task": args.task_description}]
    tasks_path = output_path / LEROBOT_TASKS_FILENAME
    with open(tasks_path, "w") as f:
        json.dump(tasks_meta, f, indent=2)
    print(f"[INFO] Wrote {tasks_path}")

    # ── Write meta/info.json ────────────────────────────────────────────────
    info = {
        "codebase_version": "v2.0",
        "robot_type":       "custom",
        "total_episodes":   n_episodes,
        "total_frames":     total_frames,
        "fps":              args.fps,
        "splits":           {"train": f"0:{n_episodes}"},
        "data_path":        f"{DATA_DIR}/episode_{{episode_index:06d}}.jsonl",
        "features": {
            "action": {
                "dtype":  "float32",
                "shape":  [8],
                "names":  [
                    "joint_0", "joint_1", "joint_2", "joint_3",
                    "joint_4", "joint_5", "joint_6", "gripper"
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [22],
                "names": [f"state_{i}" for i in range(22)],
            },
            "observation.ee_pose": {
                "dtype": "float32",
                "shape": [7],
                "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
            },
            "observation.joint_positions": {
                "dtype": "float32",
                "shape": [7],
                "names": [f"joint_{i}" for i in range(7)],
            },
        },
    }

    info_path = output_path / LEROBOT_INFO_FILENAME
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[INFO] Wrote {info_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅  Conversion complete!")
    print(f"   Output directory : {output_path}")
    print(f"   Episodes         : {n_episodes}")
    print(f"   Total frames     : {total_frames}")
    print(f"   FPS              : {args.fps}")
    print(f"   Task             : {args.task_description}")
    print("=" * 60)
    print("\nNext step — run training:")
    print(f"""
TRANSFORMERS_ATTN_IMPLEMENTATION=eager \\
/isaac-sim/kit/python/bin/python3 gr00t/experiment/launch_finetune.py \\
    --base-model-path "nvidia/GR00T-N1.6-3B" \\
    --dataset-path "{output_path}" \\
    --embodiment-tag "NEW_EMBODIMENT" \\
    --modality-config-path "/workspace/Isaac-GR00T/gr00t/configs/data/custom_embodiment.py" \\
    --output-dir "/workspace/isaaclab/ImitationLearning/checkpoints" \\
    --global-batch-size 64 \\
    --learning-rate 1e-4 \\
    --max-steps 10000
""")


if __name__ == "__main__":
    main()