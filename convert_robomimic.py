#!/usr/bin/env python3
import argparse, json, shutil, subprocess
from pathlib import Path
import h5py, numpy as np, pandas as pd

CHUNKS_SIZE = 1000
CAMERA_KEYS = {
    "camera":   "observation.images.camera",
    "camera_3": "observation.images.camera_3",
    "camera_9": "observation.images.camera_9",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-description", default="robot manipulation task")
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()

def compute_stats(arr):
    return {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(), "q99": np.quantile(arr, 0.99, axis=0).tolist()}

def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def get_video_res(p):
    try:
        r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_streams",str(p)], capture_output=True, text=True)
        for s in json.loads(r.stdout)["streams"]:
            if s.get("codec_type")=="video": return s["width"], s["height"]
    except: pass
    return 640, 480

def main():
    args = parse_args()
    src, vid_src, dst = Path(args.input), Path(args.videos), Path(args.output)
    assert src.exists(), f"Not found: {src}"
    if dst.exists(): shutil.rmtree(dst)
    (dst/"meta").mkdir(parents=True)
    episodes_meta, all_actions, all_ee, all_joints, all_states, total_frames = [], [], [], [], [], 0

    with h5py.File(src, "r") as f:
        demo_keys = sorted(f["data"].keys())
        n_episodes = len(demo_keys)
        print(f"[INFO] {n_episodes} episodes")
        for ep_idx, key in enumerate(demo_keys):
            g = f["data"][key]
            actions = g["actions"][:]
            ee = g["obs/ee_poses"][:]
            joints = g["obs/joint_positions"][:]
            state = g["obs/state"][:]
            dones = g["dones"][:]
            rewards = g["rewards"][:]
            T = actions.shape[0]
            print(f"  ep {ep_idx:02d} ({key}): {T} frames")
            chunk_idx = ep_idx // CHUNKS_SIZE
            chunk_dir = dst / f"data/chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for t in range(T):
                rows.append({
                    "episode_index": ep_idx, "frame_index": t,
                    "timestamp": round(t/args.fps, 6), "task_index": 0,
                    "action": actions[t].tolist(),
                    "observation.state": state[t].tolist(),
                    "observation.ee_pose": ee[t].tolist(),
                    "observation.joint_positions": joints[t].tolist(),
                    "annotation.human.task_description": 0,
                    "next.done": bool(dones[t]), "next.reward": float(rewards[t]),
                })
            pd.DataFrame(rows).to_parquet(chunk_dir / f"episode_{ep_idx:06d}.parquet", index=False)
            for cam_suffix, obs_key in CAMERA_KEYS.items():
                sv = vid_src / f"episode_{ep_idx}_{cam_suffix}.mp4"
                if not sv.exists(): print(f"  [WARN] missing {sv.name}"); continue
                vd = dst / f"videos/chunk-{chunk_idx:03d}/{obs_key}"
                vd.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sv, vd / f"episode_{ep_idx:06d}.mp4")
                print(f"    -> {obs_key}/episode_{ep_idx:06d}.mp4")
            episodes_meta.append({"episode_index": ep_idx, "tasks": [args.task_description], "length": T})
            all_actions.append(actions); all_ee.append(ee)
            all_joints.append(joints); all_states.append(state)
            total_frames += T

    stats = {
        "action": compute_stats(np.concatenate(all_actions)),
        "observation.state": compute_stats(np.concatenate(all_states)),
        "observation.ee_pose": compute_stats(np.concatenate(all_ee)),
        "observation.joint_positions": compute_stats(np.concatenate(all_joints)),
    }
    with open(dst/"meta/stats.json","w") as f: json.dump(stats, f, indent=2)
    write_jsonl(dst/"meta/episodes.jsonl", episodes_meta)
    write_jsonl(dst/"meta/tasks.jsonl", [{"task_index": 0, "task": args.task_description}])

    modality = {
        "video": {
            "camera":   {"original_key": "observation.images.camera"},
            "camera_3": {"original_key": "observation.images.camera_3"},
            "camera_9": {"original_key": "observation.images.camera_9"},
        },
        "state": {
            "joint_positions": {"start": 0, "end": 7, "original_key": "observation.joint_positions"},
            "ee_poses":        {"start": 0, "end": 7, "original_key": "observation.ee_pose"},
        },
        "action": {
            "joint_positions": {"start": 0, "end": 7, "original_key": "action"},
            "gripper":         {"start": 7, "end": 8, "original_key": "action"},
        },
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"},
        },
    }
    with open(dst/"meta/modality.json","w") as f: json.dump(modality, f, indent=2)

    w, h = get_video_res(vid_src/"episode_0_camera.mp4")
    vf = {"dtype":"video","shape":[h,w,3],"names":["height","width","channel"],
          "video_info":{"video.fps":float(args.fps),"video.codec":"h264",
                        "video.pix_fmt":"yuv420p","video.is_depth_map":False,"has_audio":False}}
    info = {
        "codebase_version": "v2.0", "robot_type": "custom",
        "total_episodes": n_episodes, "total_frames": total_frames,
        "total_videos": 3, "total_chunks": 0, "chunks_size": CHUNKS_SIZE,
        "fps": float(args.fps), "splits": {"train": f"0:{n_episodes}"},
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.camera":   vf,
            "observation.images.camera_3": vf,
            "observation.images.camera_9": vf,
            "action":                      {"dtype":"float32","shape":[8]},
            "observation.state":           {"dtype":"float32","shape":[22]},
            "observation.ee_pose":         {"dtype":"float32","shape":[7]},
            "observation.joint_positions": {"dtype":"float32","shape":[7]},
            "annotation.human.task_description": {"dtype":"int64","shape":[1]},
            "task_index": {"dtype":"int64","shape":[1]},
            "next.done":  {"dtype":"bool","shape":[1]},
            "next.reward":{"dtype":"float64","shape":[1]},
        },
    }
    with open(dst/"meta/info.json","w") as f: json.dump(info, f, indent=2)
    print(f"Done! {n_episodes} episodes, {total_frames} frames -> {dst}")

if __name__ == "__main__":
    main()
