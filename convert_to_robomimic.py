import os
import h5py
import numpy as np
import json
from pathlib import Path

def convert_to_robomimic(input_dir, output_file):
    input_path = Path(input_dir)
    hdf5_files = sorted(list(input_path.glob("robot_demos_*.hdf5")))
    
    if not hdf5_files:
        print("No HDF5 files found in", input_dir)
        return

    print(f"Found {len(hdf5_files)} demonstration files. Converting...")
    
    with h5py.File(output_file, "w") as out_f:
        data_grp = out_f.create_group("data")
        
        # Store env info
        env_args = {
            "env_name": "IsaacLab-TaskSpace",
            "type": 1,
            "env_kwargs": {}
        }
        data_grp.attrs["env_args"] = json.dumps(env_args)
        
        total_samples = 0
        
        for i, h5_file in enumerate(hdf5_files):
            demo_name = f"demo_{i}"
            ep_grp = data_grp.create_group(demo_name)
            
            with h5py.File(h5_file, "r") as in_f:
                # The input files have 'episode_X' at root
                root_keys = list(in_f.keys())
                if not root_keys:
                    continue
                in_ep = in_f[root_keys[0]]
                
                # Read and squeeze the arrays from (N, 1, D) to (N, D)
                actions = np.array(in_ep["actions"]).squeeze(axis=1) if "actions" in in_ep else None
                obs_state = np.array(in_ep["observations"]).squeeze(axis=1) if "observations" in in_ep else None
                joint_pos = np.array(in_ep["joint_positions"]).squeeze(axis=1) if "joint_positions" in in_ep else None
                ee_poses = np.array(in_ep["ee_poses"]).squeeze(axis=1) if "ee_poses" in in_ep else None
                
                num_samples = actions.shape[0] if actions is not None else 0
                total_samples += num_samples
                
                ep_grp.attrs["num_samples"] = num_samples
                ep_grp.create_dataset("actions", data=actions)
                
                # Rewards and Dones (basic zeros for imitation learning if absent)
                ep_grp.create_dataset("rewards", data=np.zeros(num_samples))
                dones = np.zeros(num_samples)
                dones[-1] = 1.0  # Mark last step as done
                ep_grp.create_dataset("dones", data=dones)
                
                obs_grp = ep_grp.create_group("obs")
                if obs_state is not None:
                    obs_grp.create_dataset("state", data=obs_state)
                if joint_pos is not None:
                    obs_grp.create_dataset("joint_positions", data=joint_pos)
                if ee_poses is not None:
                    obs_grp.create_dataset("ee_poses", data=ee_poses)
                    
            print(f"Converted {h5_file.name} -> {demo_name} ({num_samples} samples)")
            
        data_grp.attrs["total"] = total_samples
        print(f"Conversion complete! Saved to {output_file} with {total_samples} total samples.")

if __name__ == "__main__":
    convert_to_robomimic("demonstrations", "demonstrations/robomimic_dataset.hdf5")
