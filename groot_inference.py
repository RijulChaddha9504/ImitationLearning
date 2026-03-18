import argparse
import sys
import numpy as np
from PIL import Image as PILImage
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
    """Wraps the fine-tuned GR00T model for inference using official Gr00tPolicy."""

    def __init__(self, checkpoint_path: str, task_description: str, device: str = "cuda"):
        self.task_description = task_description
        self.device = device

        print(f"[GR00T] Loading model from {checkpoint_path}...")
        sys.path.insert(0, '/workspace/Isaac-GR00T')

        import gr00t.configs.data.custom_embodiment
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
        from transformers import AutoProcessor

        self.model = Gr00tN1d6.from_pretrained(
            str(Path(checkpoint_path).resolve()),
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        processor_path = str(Path(checkpoint_path).parent / "processor")
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor.eval()
        print("[GR00T] Model loaded successfully!")

    @torch.no_grad()
    def predict(self, joint_positions: np.ndarray, ee_pose: np.ndarray) -> np.ndarray:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData

        # Create dummy black images (H, W, C) uint8 - required by processor
        dummy_pil = PILImage.new('RGB', (256, 256), color=(128, 128, 128))
        dummy_image = np.array(dummy_pil, dtype=np.uint8)

        vla_step = VLAStepData(
            images={
                "camera":   dummy_image,
                "camera_3": dummy_image,
                "camera_9": dummy_image,
            },
            states={
                "joint_positions": joint_positions.reshape(1, -1).astype(np.float32),
                "ee_poses": ee_pose.reshape(1, -1).astype(np.float32),
            },
            actions={},
            text=self.task_description,
            embodiment=EmbodimentTag.NEW_EMBODIMENT,
        )

        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
        processed = self.processor(messages)
        collated = self.processor.collator([processed])

        # Move to device
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                collated[k] = v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)

        output = self.model.get_action(**collated)
        action_pred = output["action_pred"].float().cpu().numpy()

        states = {"joint_positions": joint_positions.reshape(1, 1, -1).astype(np.float32),
                "ee_poses": ee_pose.reshape(1, 1, -1).astype(np.float32)}
        decoded = self.processor.decode_action(action_pred, EmbodimentTag.NEW_EMBODIMENT, states)

        joints = decoded['joint_positions'][0, 0]  # (7,)
        gripper = decoded['gripper'][0, 0, 0]       # scalar
        return np.append(joints, gripper)


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
                import traceback
                traceback.print_exc()
                print(f"[WARN] Inference failed at step {step_count}: {e}")
                break  # stop after first error to see full traceback

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