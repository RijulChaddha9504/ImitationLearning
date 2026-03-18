import argparse
import sys
import numpy as np
from pathlib import Path

"""Robot Arm GR00T Inference - runs fine-tuned model to control robot in IsaacLab"""

parser = argparse.ArgumentParser(description="GR00T Inference for Robot Arm Control")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--checkpoint", type=str,
                    default="/workspace/isaaclab/ImitationLearning/checkpoints/checkpoint-5000",
                    help="Path to GR00T checkpoint.")
parser.add_argument("--task-description", type=str,
                    default="robot manipulation task",
                    help="Task description for the model.")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.enable_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_from_euler_xyz
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Monkey-patch warp
try:
    import warp as wp
    if hasattr(wp.types, 'array'):
        _original_warp_array = wp.types.array
        def _patched_warp_array(*args, **kwargs):
            if 'owner' in kwargs:
                del kwargs['owner']
            return _original_warp_array(*args, **kwargs)
        wp.types.array = _patched_warp_array
        print("[INFO] Applied Warp array monkey-patch")
except ImportError:
    pass


# ============================================================
# GR00T Model Wrapper
# ============================================================
class GR00TInference:
    """Wraps the fine-tuned GR00T model for inference."""

    def __init__(self, checkpoint_path: str, task_description: str, device: str = "cuda"):
        self.device = device
        self.task_description = task_description

        print(f"[GR00T] Loading model from {checkpoint_path}...")
        sys.path.insert(0, '/workspace/Isaac-GR00T')

        # Load modality config
        import gr00t.configs.data.custom_embodiment  # registers NEW_EMBODIMENT

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(device)
        print("[GR00T] Model loaded successfully!")

        # Load processor
        processor_path = str(Path(checkpoint_path).parent / "processor")
        from gr00t.data.dataset.gr00t_dataset import GR00tDatasetConfig
        print(f"[GR00T] Processor ready.")

    @torch.no_grad()
    def predict(self, joint_positions: np.ndarray, ee_pose: np.ndarray) -> np.ndarray:
        """
        Given current joint positions and ee pose, predict next action.

        Args:
            joint_positions: shape (7,) - current joint positions
            ee_pose: shape (7,) - current end-effector pose [x, y, z, qw, qx, qy, qz]

        Returns:
            action: shape (8,) - [7 joint positions + 1 gripper]
        """
        from gr00t.data.embodiment_tags import EmbodimentTag

        # Build input batch
        batch = {
            "joint_positions": torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            "ee_poses": torch.tensor(ee_pose, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            "annotation.human.task_description": [self.task_description],
            "embodiment_id": torch.tensor([EmbodimentTag.NEW_EMBODIMENT.value], device=self.device),
        }

        output = self.model(batch)
        action_pred = output["action_pred"][0, 0].cpu().numpy()  # (8,)
        return action_pred


# ============================================================
# Scene Configuration (same as task_space_test.py)
# ============================================================
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ============================================================
# Helper: gripper norm to joint positions
# ============================================================
def gripper_norm_to_joint_positions(norm_val: float, device: str = "cuda"):
    """Convert normalized gripper value [0=open, 1=closed] to joint positions."""
    open_val = 0.04
    closed_val = 0.0
    pos = open_val + (closed_val - open_val) * norm_val
    return torch.tensor([[pos, pos]], device=device)


# ============================================================
# Main Simulation Loop
# ============================================================
def run_simulator(sim, scene, groot: GR00TInference):
    sim_dt = sim.get_physics_dt()

    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    # IK controller
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim.device)

    # Get joint IDs
    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    arm_joint_ids = []
    gripper_joint_ids = []
    for name in arm_joint_names:
        idx = robot.find_joints(name)[0]
        if idx is not None:
            arm_joint_ids.extend(idx)
    for name in gripper_joint_names:
        idx = robot.find_joints(name)[0]
        if idx is not None:
            gripper_joint_ids.extend(idx)

    # Initialize goal pose from current EE pose
    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose = ee_pose_w.clone()
    smooth_target_pose = goal_pose.clone()

    # Inference state
    gripper_target_norm = 0.0  # Start open
    step_count = 0
    inference_interval = 5  # Run GR00T every N sim steps

    print("[INFO] Starting GR00T inference loop...")
    print(f"[INFO] Running inference every {inference_interval} steps")

    while simulation_app.is_running():
        # Get current robot state
        joint_pos_full = robot.data.joint_pos[:, :7]   # (1, 7) arm joints
        joint_vel_full = robot.data.joint_vel[:, :7]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # Run GR00T inference every N steps
        if step_count % inference_interval == 0:
            joint_pos_np = joint_pos_full[0].cpu().numpy()   # (7,)
            ee_pose_np = ee_pose_w[0].cpu().numpy()           # (7,)

            try:
                action = groot.predict(joint_pos_np, ee_pose_np)  # (8,)
                # action[:7] = joint position targets
                # action[7]  = gripper command
                predicted_joints = action[:7]
                gripper_target_norm = float(np.clip(action[7], 0.0, 1.0))

                # Update goal pose using predicted joint positions via FK approximation
                # Convert joint targets to ee pose delta
                joint_targets = torch.tensor(predicted_joints, dtype=torch.float32, device=sim.device).unsqueeze(0)

                print(f"[Step {step_count}] Action: joints={predicted_joints.round(3)}, gripper={gripper_target_norm:.2f}")

            except Exception as e:
                print(f"[WARN] Inference failed at step {step_count}: {e}")

        # Smooth approach to goal
        position_smoothing = 0.1
        rotation_smoothing = 0.1
        max_linear_velocity = 0.5
        max_angular_velocity = 1.0

        position_error = goal_pose[:, 0:3] - smooth_target_pose[:, 0:3]
        position_delta = position_smoothing * position_error
        position_delta_norm = torch.norm(position_delta)
        max_pos_delta = max_linear_velocity * sim_dt
        if position_delta_norm > max_pos_delta:
            position_delta = position_delta / position_delta_norm * max_pos_delta
        smooth_target_pose[:, 0:3] += position_delta

        # IK solve
        ik_controller.set_command(smooth_target_pose)
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        jacobian = robot.root_physx_view.get_jacobians()[:, robot_entity_cfg.body_ids[0] - 1, :, :7]
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_full)

        # Apply arm joint targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=list(range(7)))

        # Apply gripper
        gripper_positions = gripper_norm_to_joint_positions(gripper_target_norm, sim.device)
        if len(gripper_joint_ids) > 0:
            robot.set_joint_position_target(gripper_positions, joint_ids=gripper_joint_ids)

        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1


def main():
    # Load GR00T model
    groot = GR00TInference(
        checkpoint_path=args_cli.checkpoint,
        task_description=args_cli.task_description,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, groot)


if __name__ == "__main__":
    main()
    simulation_app.close()