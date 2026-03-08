import argparse
from isaaclab.app import AppLauncher
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_from_euler_xyz

"""Robot Arm Teleoperation (headless-compatible) with Task Space IK Control"""

parser = argparse.ArgumentParser(description="Robot Arm Teleoperation with Task Space IK Control")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
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
from isaaclab.utils.math import subtract_frame_transforms

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

from isaaclab_assets import UR10_CFG, FRANKA_PANDA_HIGH_PD_CFG

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] OpenCV not available, video recording will use imageio instead")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

if not args_cli.headless:
    from isaaclab.devices import Se3Keyboard
    import omni.ui as ui


class DemonstrationRecorder:
    def __init__(self, save_dir="demonstrations", recordings_dir="recordings"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(exist_ok=True, parents=True)
        self.episodes = []
        self.episode_counter = 0
        if self.recordings_dir.exists():
            import re
            pattern = re.compile(r"episode_(\d+)_")
            max_ep = -1
            for file_path in self.recordings_dir.glob("*.mp4"):
                match = pattern.search(file_path.name)
                if match:
                    try:
                        ep_num = int(match.group(1))
                        if ep_num > max_ep:
                            max_ep = ep_num
                    except ValueError:
                        pass
            if max_ep >= 0:
                self.episode_counter = max_ep + 1
                print(f"[INFO] Found existing recordings up to episode_{max_ep}. Resuming from episode_{self.episode_counter}")
        self.current_episode = {
            'episode_num': 0, 'observations': [], 'actions': [],
            'ee_poses': [], 'joint_positions': [], 'video_frames': {}
        }
        self.recording = False
        self.video_fps = 10
        print(f"[INFO] HDF5 and MP4 files will save per-episode to: {self.save_dir} and {self.recordings_dir}")

    def start_episode(self):
        self.recording = True
        self.current_episode = {
            'episode_num': self.episode_counter, 'observations': [], 'actions': [],
            'ee_poses': [], 'joint_positions': [], 'video_frames': {}
        }
        print(f"[RECORDING] Started episode (will be episode_{self.episode_counter})")

    def add_transition(self, obs, action, ee_pose, joint_pos, video_frames_dict=None):
        if self.recording:
            self.current_episode['observations'].append(obs.cpu().numpy())
            self.current_episode['actions'].append(action.cpu().numpy())
            self.current_episode['ee_poses'].append(ee_pose.cpu().numpy())
            self.current_episode['joint_positions'].append(joint_pos.cpu().numpy())
            if video_frames_dict is not None and isinstance(video_frames_dict, dict):
                for cam_name, frame_data in video_frames_dict.items():
                    if cam_name not in self.current_episode['video_frames']:
                        self.current_episode['video_frames'][cam_name] = []
                    if hasattr(frame_data, 'cpu'):
                        frame = frame_data.cpu().numpy()
                    elif hasattr(frame_data, 'numpy'):
                        frame = frame_data.numpy()
                    elif isinstance(frame_data, list):
                        frame = np.array(frame_data)
                    else:
                        frame = frame_data
                    self.current_episode['video_frames'][cam_name].append(frame)

    def _save_video(self, frames, episode_num, cam_suffix=""):
        if len(frames) == 0:
            return None
        name_part = f"episode_{episode_num}"
        if cam_suffix:
            name_part += f"_{cam_suffix}"
        video_path = self.recordings_dir / f"{name_part}.mp4"
        try:
            if CV2_AVAILABLE:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, self.video_fps, (width, height))
                for frame in frames:
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame.astype(np.uint8)
                    out.write(frame_bgr)
                out.release()
                print(f"[VIDEO] Saved {len(frames)} frames to {video_path}")
            elif IMAGEIO_AVAILABLE:
                imageio.mimwrite(str(video_path), [f.astype(np.uint8) for f in frames], fps=self.video_fps)
                print(f"[VIDEO] Saved {len(frames)} frames to {video_path}")
            else:
                print("[WARN] No video library available. Video not saved.")
                return None
            return str(video_path)
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
            return None

    def end_episode(self):
        if self.recording and len(self.current_episode['observations']) > 0:
            episode_num = self.current_episode['episode_num']
            num_steps = len(self.current_episode['observations'])
            video_paths = {}
            for cam_name, frames in self.current_episode['video_frames'].items():
                if len(frames) > 0:
                    path = self._save_video(frames, episode_num, cam_suffix=cam_name)
                    if path:
                        video_paths[cam_name] = path
            self.current_episode['video_paths'] = video_paths
            self.current_episode['video_frames'] = {}
            self.episodes.append(self.current_episode)
            self.episode_counter += 1
            print(f"[RECORDING] Episode {episode_num} completed with {num_steps} steps.")
            self.save()
            self.get_quick_summary()
        elif self.recording:
            print("[WARN] Episode ended but no data was recorded")
        self.recording = False

    def save(self):
        if len(self.episodes) == 0:
            print("[WARN] No episodes to save")
            return
        saved_count = 0
        try:
            for episode in self.episodes:
                episode_num = episode.get('episode_num', 0)
                file_path = self.save_dir / f"robot_demos_{episode_num}.hdf5"
                with h5py.File(file_path, 'w') as f:
                    grp = f.create_group(f'episode_{episode_num}')
                    for key, value in episode.items():
                        if key in ('video_frames', 'episode_num'):
                            continue
                        elif key == 'video_paths':
                            for cam_name, path in value.items():
                                grp.attrs[f'video_path_{cam_name}'] = str(path)
                        else:
                            grp.create_dataset(key, data=np.array(value))
                saved_count += 1
            print(f"[SUCCESS] Saved {saved_count} episodes to {self.save_dir.absolute()}")
            self.episodes = []
        except Exception as e:
            print(f"[ERROR] Failed to save demonstrations: {e}")

    def get_quick_summary(self):
        unsaved = len(self.episodes)
        status = "✓" if unsaved == 0 else f"⚠ {unsaved} unsaved"
        print(f"[STATS] This session: {len(self.episodes)} episodes {status}")


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
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
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
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.71, 0.88, 1.0), rot=_get_cam_quat(36.0), convention="world"),
    )
    camera_3 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_3",
        update_period=0.1, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.04, 1.43, 1.0), rot=_get_cam_quat(108.0), convention="world"),
    )
    camera_9 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_9",
        update_period=0.1, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # -----------------------
    # SMOOTH APPROACH PARAMETERS
    # -----------------------
    position_smoothing = 0.15
    rotation_smoothing = 0.12
    max_linear_velocity = 0.6
    max_angular_velocity = 1.0
    slow_zone_threshold = 0.15
    min_speed_ratio = 0.1
    position_deadband = 0.002
    rotation_deadband = 0.02

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    robot_entity_cfg.resolve(scene)

    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    sim_dt = sim.get_physics_dt()

    if args_cli.robot == "franka_panda":
        joint_position = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)
    else:
        joint_position = torch.zeros((1, 6), device=sim.device)
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)

    joint_names = [n.lower() for n in robot.data.joint_names]
    gripper_candidates = [i for i, n in enumerate(joint_names) if any(k in n for k in ["finger", "gripper", "hand", "claw", "panda_finger"])]
    gripper_joint_ids = gripper_candidates

    if len(gripper_joint_ids) > 0:
        print("[INFO] Detected gripper joints:", [robot.data.joint_names[i] for i in gripper_joint_ids])
    else:
        print("[WARN] No gripper joints detected.")

    gripper_open_pos = 0.04
    gripper_closed_pos = 0.0
    gripper_target_norm = 0.0
    gripper_open_bool = True

    def gripper_norm_to_joint_positions(norm):
        pos = gripper_open_pos + (gripper_closed_pos - gripper_open_pos) * norm
        if len(gripper_joint_ids) == 0:
            return None
        return torch.tensor([[pos] * len(gripper_joint_ids)], device=sim.device)

    recorder = DemonstrationRecorder("demonstrations", "recordings")
    print("[INFO] Demonstration recorder initialized")

    camera = scene["camera"] if "camera" in scene.keys() else None

    if not args_cli.headless:
        try:
            teleop = Se3Keyboard(pos_sensitivity=0.05, rot_sensitivity=0.05)
        except TypeError:
            try:
                teleop = Se3Keyboard(0.05, 0.05)
            except TypeError:
                teleop = Se3Keyboard()
                teleop.pos_sensitivity = 0.05
                teleop.rot_sensitivity = 0.05
        teleop.reset()
        print("[INFO] Teleoperation active — smooth approach ENABLED")
        teleop_has_extra_keys = False
        gripper_state = {"open": True}

        def _toggle_gripper_cb():
            gripper_state["open"] = not gripper_state["open"]
            print(f"[UI] Gripper toggled -> {'OPEN' if gripper_state['open'] else 'CLOSED'}")

        def _start_recording():
            recorder.start_episode()

        def _stop_recording():
            recorder.end_episode()

        gripper_window = ui.Window("Gripper", width=180, height=80)
        with gripper_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Gripper Control")
                ui.Button("Toggle Gripper", clicked_fn=_toggle_gripper_cb, height=40)

        recording_window = ui.Window("Recording", width=180, height=110, position_x=200)
        with recording_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Demonstration Recording")
                ui.Button("Start Recording", clicked_fn=_start_recording, height=40)
                ui.Button("Stop Recording / Save", clicked_fn=_stop_recording, height=40)
    else:
        step = 0
        headless_cycle_t = 0
        print("[INFO] Running headless simulation with scripted motion...")

    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose = ee_pose_w.clone()

    # -----------------------------------------------------------------------
    # RE-ENABLED: Smooth approach state variables
    # -----------------------------------------------------------------------
    smooth_target_pose = goal_pose.clone()
    previous_smooth_pose = smooth_target_pose.clone()

    while simulation_app.is_running():
        if not args_cli.headless:
            try:
                ret = teleop.advance()
                if isinstance(ret, tuple) and len(ret) == 3:
                    pos_delta, rot_delta, extra_keys = ret
                    teleop_has_extra_keys = True
                else:
                    pos_delta, rot_delta = ret
            except TypeError:
                pos_delta, rot_delta = teleop.advance()

            # Update GOAL pose with teleop input (goal is where you want to go)
            goal_pose[:, 0:3] += torch.tensor(pos_delta[:3], device=goal_pose.device).unsqueeze(0)

            if isinstance(rot_delta, (list, tuple, np.ndarray)) and len(rot_delta) == 3:
                current_quat = goal_pose[:, 3:7]
                delta_quat = quat_from_euler_xyz(
                    torch.tensor([rot_delta[0]], device=sim.device),
                    torch.tensor([rot_delta[1]], device=sim.device),
                    torch.tensor([rot_delta[2]], device=sim.device),
                )
                goal_pose[:, 3:7] = quat_mul(delta_quat, current_quat)
                goal_pose[:, 3:7] = goal_pose[:, 3:7] / torch.norm(goal_pose[:, 3:7], dim=1, keepdim=True)

            gripper_open_bool = gripper_state["open"]
            gripper_target_norm = 0.0 if gripper_open_bool else 1.0

        else:
            delta_pos = 0.01 * torch.sin(torch.tensor(step * 0.1))
            goal_pose[:, 0] += delta_pos
            step += 1
            headless_cycle_t += 1
            cycle_len = 200
            tmod = headless_cycle_t % cycle_len
            if tmod < (cycle_len // 2):
                gripper_target_norm = 0.0
                gripper_open_bool = True
            else:
                gripper_target_norm = 1.0
                gripper_open_bool = False

        # -----------------------------------------------------------------------
        # RE-ENABLED: Smooth approach logic — robot moves smoothly toward goal
        # -----------------------------------------------------------------------
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # Position smoothing
        position_error = goal_pose[:, 0:3] - smooth_target_pose[:, 0:3]
        distance_to_goal = torch.norm(position_error, dim=1, keepdim=True)

        speed_scale = torch.clamp(
            distance_to_goal / slow_zone_threshold,
            min=min_speed_ratio,
            max=1.0
        )

        if distance_to_goal.item() > position_deadband:
            position_delta = position_smoothing * speed_scale * position_error
            max_position_delta = max_linear_velocity * sim_dt
            position_delta_norm = torch.norm(position_delta)
            if position_delta_norm > max_position_delta:
                position_delta = position_delta / position_delta_norm * max_position_delta
            smooth_target_pose[:, 0:3] += position_delta

        # Rotation smoothing
        goal_quat = goal_pose[:, 3:7]
        current_quat = smooth_target_pose[:, 3:7]
        dot_product = torch.sum(goal_quat * current_quat, dim=1, keepdim=True)
        goal_quat_corrected = torch.where(dot_product < 0, -goal_quat, goal_quat)
        rotation_error = goal_quat_corrected - current_quat
        rotation_error_magnitude = torch.norm(rotation_error, dim=1, keepdim=True)

        if rotation_error_magnitude.item() > rotation_deadband:
            rotation_delta = rotation_smoothing * rotation_error
            max_rotation_delta = max_angular_velocity * sim_dt
            rotation_delta_norm = torch.norm(rotation_delta)
            if rotation_delta_norm > max_rotation_delta:
                rotation_delta = rotation_delta / rotation_delta_norm * max_rotation_delta
            smooth_target_pose[:, 3:7] += rotation_delta
            smooth_target_pose[:, 3:7] = smooth_target_pose[:, 3:7] / torch.norm(smooth_target_pose[:, 3:7], dim=1, keepdim=True)

        # Temporal smoothing blend
        alpha = 0.7
        smooth_target_pose = alpha * smooth_target_pose + (1 - alpha) * previous_smooth_pose
        smooth_target_pose[:, 3:7] = smooth_target_pose[:, 3:7] / torch.norm(smooth_target_pose[:, 3:7], dim=1, keepdim=True)
        previous_smooth_pose = smooth_target_pose.clone()

        # -----------------------------------------------------------------------
        # IK now uses smooth_target_pose instead of goal_pose directly
        # -----------------------------------------------------------------------
        diff_ik_controller.set_command(smooth_target_pose)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        if len(gripper_joint_ids) > 0:
            gripper_joint_positions = gripper_norm_to_joint_positions(gripper_target_norm)
            if gripper_joint_positions is not None:
                robot.set_joint_position_target(gripper_joint_positions, joint_ids=gripper_joint_ids)

        # Record transition
        joint_pos_full = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel_full = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]

        obs = torch.cat([
            joint_pos_full, joint_vel_full, ee_pose_w,
            torch.tensor([[gripper_target_norm]], device=sim.device)
        ], dim=-1)

        action = torch.cat([
            joint_pos_des,
            torch.tensor([[gripper_target_norm]], device=sim.device)
        ], dim=-1)

        video_frames_dict = {}

        def _get_camera_frame(sensor):
            if sensor is None:
                return None
            try:
                rgb_data = sensor.data.output["rgb"]
                if rgb_data is None:
                    return None
                if hasattr(rgb_data, 'shape') and len(rgb_data.shape) >= 3:
                    frame_data = rgb_data[0] if len(rgb_data.shape) == 4 else rgb_data
                    if hasattr(frame_data, 'cpu'):
                        v_frame = frame_data.clone().cpu().numpy()
                    elif hasattr(frame_data, 'numpy'):
                        v_frame = frame_data.numpy().copy()
                    else:
                        v_frame = np.asarray(frame_data).copy()
                    if v_frame.dtype != np.uint8:
                        v_frame = (v_frame * 255).astype(np.uint8) if v_frame.max() <= 1.0 else v_frame.astype(np.uint8)
                    return v_frame
            except Exception as e:
                if not hasattr(sensor, '_error_printed'):
                    print(f"[WARN] Frame capture error for {sensor}: {e}")
                    sensor._error_printed = True
            return None

        for key in scene.keys():
            if "camera" in key.lower() and "marker" not in key.lower():
                try:
                    sensor = scene[key]
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                        frame = _get_camera_frame(sensor)
                        if frame is not None:
                            video_frames_dict[key] = frame
                except:
                    pass

        recorder.add_transition(
            obs=obs, action=action, ee_pose=ee_pose_w,
            joint_pos=joint_pos_full, video_frames_dict=video_frames_dict
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(goal_pose[:, 0:3] + scene.env_origins, goal_pose[:, 3:7])


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()