import argparse
import sys
import numpy as np
from PIL import Image as PILImage
from pathlib import Path
from datetime import datetime

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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_from_euler_xyz
from isaaclab_assets import UR10_CFG, FRANKA_PANDA_HIGH_PD_CFG

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
        print("[INFO] Applied Warp array monkey-patch for compatibility")
except ImportError:
    pass


# ============================================================
# GR00T Model Wrapper
# ============================================================
class GR00TInference:
    """Wraps the fine-tuned GR00T model for inference."""

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
    def predict(self, joint_positions: np.ndarray, ee_pose: np.ndarray,
                camera_frame: np.ndarray = None,
                camera_3_frame: np.ndarray = None,
                camera_9_frame: np.ndarray = None) -> np.ndarray:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData

        dummy_pil = PILImage.new('RGB', (64, 64), color=(128, 128, 128))
        dummy_image = np.array(dummy_pil, dtype=np.uint8)

        cam  = camera_frame   if camera_frame   is not None else dummy_image
        cam3 = camera_3_frame if camera_3_frame is not None else dummy_image
        cam9 = camera_9_frame if camera_9_frame is not None else dummy_image

        vla_step = VLAStepData(
            images={
                "camera":   cam,
                "camera_3": cam3,
                "camera_9": cam9,
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

        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                collated[k] = v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)

        output = self.model.get_action(**collated)
        action_pred = output["action_pred"].float().cpu().numpy()

        states = {
            "joint_positions": joint_positions.reshape(1, 1, -1).astype(np.float32),
            "ee_poses": ee_pose.reshape(1, 1, -1).astype(np.float32),
        }
        decoded = self.processor.decode_action(action_pred, EmbodimentTag.NEW_EMBODIMENT, states)

        joints  = decoded['joint_positions'][0, 0]
        gripper = decoded['gripper'][0, 0, 0]
        return np.append(joints, gripper)


# ============================================================
# Scene Configuration (full scene from task_space_test.py)
# ============================================================
def _get_cam_quat(angle_deg):
    try:
        angle_rad = np.deg2rad(angle_deg)
        q = quat_from_euler_xyz(
            torch.tensor([0.0]),
            torch.tensor([np.deg2rad(30.0)]),
            torch.tensor([angle_rad + np.pi])
        )
        return tuple(q[0].tolist())
    except:
        return (1.0, 0.0, 0.0, 0.0)


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0)
        ),
    )
    cube_table = AssetBaseCfg(
        prim_path="/World/CubeTable",
        spawn=sim_utils.CuboidCfg(
            size=[0.4, 0.4, 0.6],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.4)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0, -0.1)),
    )
    cube = AssetBaseCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(size=[0.1, 0.1, 0.1]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )
    cube2 = AssetBaseCfg(
        prim_path="/World/cube2",
        spawn=sim_utils.CuboidCfg(
            size=[0.03, 0.03, 0.03],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0, 0.23)),
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.71, 0.88, 1.0), rot=_get_cam_quat(36.0), convention="world"),
    )
    camera_3 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_3",
        update_period=0.1, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.04, 1.43, 1.0), rot=_get_cam_quat(108.0), convention="world"),
    )
    camera_9 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_9",
        update_period=0.1, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.71, -0.88, 1.0), rot=_get_cam_quat(324.0), convention="world"),
    )
    camera_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CameraMarker",
        spawn=sim_utils.CuboidCfg(
            size=[0.1, 0.1, 0.2],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.71, 0.88, 1.0), rot=_get_cam_quat(36.0)),
    )


# ============================================================
# Helper: get camera frame as numpy array (256x256)
# ============================================================
def _get_camera_frame(sensor):
    if sensor is None:
        return None
    try:
        rgb_data = sensor.data.output["rgb"]
        if rgb_data is None:
            return None
        frame_data = rgb_data[0] if len(rgb_data.shape) == 4 else rgb_data
        if hasattr(frame_data, 'cpu'):
            v_frame = frame_data.clone().cpu().numpy()
        else:
            v_frame = np.asarray(frame_data).copy()
        if v_frame.dtype != np.uint8:
            v_frame = (v_frame * 255).astype(np.uint8) if v_frame.max() <= 1.0 else v_frame.astype(np.uint8)
        pil = PILImage.fromarray(v_frame).resize((64, 64))
        return np.array(pil, dtype=np.uint8)
    except Exception:
        return None


# ============================================================
# Helper: gripper norm to joint positions
# ============================================================
def gripper_norm_to_joint_positions(norm_val: float, device: str = "cuda"):
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

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker   = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    # Initialize joint states
    joint_position = robot.data.default_joint_pos.clone()
    joint_vel      = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_position, joint_vel)

    # Gripper setup
    joint_names = [n.lower() for n in robot.data.joint_names]
    gripper_joint_ids = [i for i, n in enumerate(joint_names)
                         if any(k in n for k in ["finger", "gripper", "panda_finger"])]
    gripper_target_norm = 0.0

    # Cameras
    camera_sensor   = scene["camera"]   if "camera"   in scene.keys() else None
    camera_3_sensor = scene["camera_3"] if "camera_3" in scene.keys() else None
    camera_9_sensor = scene["camera_9"] if "camera_9" in scene.keys() else None

    ee_pose_w  = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose  = ee_pose_w.clone()

    step_count = 0
    inference_interval = 10

    print("[INFO] Starting GR00T inference loop...")
    print(f"[INFO] Running inference every {inference_interval} steps")

    while simulation_app.is_running():
        ee_pose_w     = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        joint_pos     = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_pos_full = robot.data.joint_pos[:, :7]

        if step_count % inference_interval == 0:
            joint_pos_np = joint_pos_full[0].cpu().numpy()
            ee_pose_np   = ee_pose_w[0].cpu().numpy()

            cam_frame  = _get_camera_frame(camera_sensor)
            cam3_frame = _get_camera_frame(camera_3_sensor)
            cam9_frame = _get_camera_frame(camera_9_sensor)

            try:
                action = groot.predict(
                    joint_pos_np, ee_pose_np,
                    camera_frame=cam_frame,
                    camera_3_frame=cam3_frame,
                    camera_9_frame=cam9_frame,
                )
                predicted_joints    = action[:7]
                gripper_target_norm = float(np.clip(action[7], 0.0, 1.0))

                joint_targets = torch.tensor(predicted_joints, dtype=torch.float32, device=sim.device).unsqueeze(0)
                robot.set_joint_position_target(joint_targets, joint_ids=robot_entity_cfg.joint_ids)

                print(f"[Step {step_count}] joints={predicted_joints.round(3)}, gripper={gripper_target_norm:.2f}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[WARN] Inference failed at step {step_count}: {e}")
                break

        # Apply gripper
        if len(gripper_joint_ids) > 0:
            gripper_positions = gripper_norm_to_joint_positions(gripper_target_norm, sim.device)
            robot.set_joint_position_target(gripper_positions, joint_ids=gripper_joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(goal_pose[:, 0:3] + scene.env_origins, goal_pose[:, 3:7])

        step_count += 1


def main():
    groot = GR00TInference(
        checkpoint_path=args_cli.checkpoint,
        task_description=args_cli.task_description,
        device="cuda:1",
    )

    sim_cfg  = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim      = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene     = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, groot)


if __name__ == "__main__":
    main()
    simulation_app.close()