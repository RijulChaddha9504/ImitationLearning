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
# AppLauncher adds --enable_cameras, so we don't need to add it manually here
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force enable cameras if not set, as they are required for sensors
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

from isaaclab_assets import UR10_CFG, FRANKA_PANDA_HIGH_PD_CFG

# Video recording imports
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

# Import Se3Keyboard and Omni UI only if GUI mode
if not args_cli.headless:
    from isaaclab.devices import Se3Keyboard
    import omni.ui as ui


# Add recording functionality
import h5py
import numpy as np
from pathlib import Path

class DemonstrationRecorder:
    """
    Records robot demonstrations with timestamped filenames.
    Each session creates a new file: robot_demos_YYYYMMDD_HHMMSS.hdf5
    Also saves video recordings to the 'recordings' folder.
    """
    
    def __init__(self, save_dir="demonstrations", recordings_dir="recordings"):
        # Create save directories if they don't exist
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = self.save_dir / f"robot_demos_{timestamp}.hdf5"
        self.session_timestamp = timestamp
        
        self.episodes = []
        self.current_episode = {
            'observations': [],
            'actions': [],
            'ee_poses': [],
            'joint_positions': [],
            'video_frames': []  # NEW: Store video frames
        }
        self.recording = False
        self.video_fps = 10  # Frames per second for saved videos
        
        print(f"[INFO] This session will save to: {self.save_path}")
        print(f"[INFO] Video recordings will save to: {self.recordings_dir}")
    
    def start_episode(self):
        """Start recording a new episode"""
        self.recording = True
        self.current_episode = {
            'observations': [],
            'actions': [],
            'ee_poses': [],
            'joint_positions': [],
            'video_frames': []  # NEW: Store video frames
        }
        next_ep_num = len(self.episodes)
        print(f"[RECORDING] Started episode (will be episode_{next_ep_num})")
    
    def add_transition(self, obs, action, ee_pose, joint_pos, video_frame=None):
        """Add a single transition to the current episode"""
        if self.recording:
            self.current_episode['observations'].append(obs.cpu().numpy())
            self.current_episode['actions'].append(action.cpu().numpy())
            self.current_episode['ee_poses'].append(ee_pose.cpu().numpy())
            self.current_episode['joint_positions'].append(joint_pos.cpu().numpy())
            # Store video frame if provided
            if video_frame is not None:
                if hasattr(video_frame, 'cpu'):
                    frame = video_frame.cpu().numpy()
                else:
                    frame = np.array(video_frame)
                self.current_episode['video_frames'].append(frame)
    
    def _save_video(self, frames, episode_num):
        """Save video frames to MP4 file"""
        if len(frames) == 0:
            return None
        
        # Create unique timestamp for this recording
        recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"episode_{episode_num}_{recording_timestamp}.mp4"
        video_path = self.recordings_dir / video_filename
        
        try:
            if CV2_AVAILABLE:
                # Use OpenCV for video writing
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, self.video_fps, (width, height))
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame.astype(np.uint8)
                    out.write(frame_bgr)
                out.release()
                print(f"[VIDEO] Saved {len(frames)} frames to {video_path}")
            elif IMAGEIO_AVAILABLE:
                # Use imageio for video writing
                imageio.mimwrite(str(video_path), [f.astype(np.uint8) for f in frames], fps=self.video_fps)
                print(f"[VIDEO] Saved {len(frames)} frames to {video_path}")
            else:
                print("[WARN] No video library available (cv2 or imageio). Video not saved.")
                return None
            return str(video_path)
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
            return None
    
    def end_episode(self):
        """End the current episode, save video, and store episode data"""
        if self.recording and len(self.current_episode['observations']) > 0:
            episode_num = len(self.episodes)
            num_steps = len(self.current_episode['observations'])
            num_frames = len(self.current_episode['video_frames'])
            
            # Save video if frames were recorded
            video_path = None
            if num_frames > 0:
                video_path = self._save_video(self.current_episode['video_frames'], episode_num)
            
            # Store video path reference and clear frames from memory
            self.current_episode['video_path'] = video_path if video_path else ""
            self.current_episode['video_frames'] = []  # Clear frames to save memory
            
            self.episodes.append(self.current_episode)
            print(f"[RECORDING] Episode {episode_num} completed with {num_steps} steps, {num_frames} frames")
        elif self.recording:
            print("[WARN] Episode ended but no data was recorded")
        self.recording = False
    
    def save(self):
        """Save all episodes from this session to timestamped file"""
        if len(self.episodes) == 0:
            print("[WARN] No episodes to save")
            return
        
        try:
            with h5py.File(self.save_path, 'w') as f:
                for i, episode in enumerate(self.episodes):
                    grp = f.create_group(f'episode_{i}')
                    for key, value in episode.items():
                        # Skip video_frames (already saved as MP4) but keep video_path
                        if key == 'video_frames':
                            continue
                        if key == 'video_path':
                            # Store video path as string
                            grp.attrs['video_path'] = value if value else ""
                        else:
                            grp.create_dataset(key, data=np.array(value))
            
            print(f"[SUCCESS] Saved {len(self.episodes)} episodes to:")
            print(f"  {self.save_path.absolute()}")
            
            # Clear episodes from memory after successful save
            self.episodes = []
            
        except Exception as e:
            print(f"[ERROR] Failed to save demonstrations: {e}")
    
    def get_stats(self, detailed=False):
        """Print comprehensive statistics about recorded demonstrations."""
        print("\n" + "="*60)
        print("DEMONSTRATION STATISTICS")
        print("="*60)
        
        print(f"\n📁 Current Session File:")
        print(f"   {self.save_path}")
        
        if self.save_path.exists():
            file_size_kb = self.save_path.stat().st_size / 1024
            print(f"   Size: {file_size_kb:.2f} KB")
        else:
            print(f"   Status: Not saved yet")
        
        # Episodes in memory
        episodes_in_memory = [len(ep['observations']) for ep in self.episodes]
        total_in_memory = len(self.episodes)
        
        print(f"\n📊 This Session:")
        print(f"   Episodes recorded: {total_in_memory}")
        
        if episodes_in_memory:
            total_timesteps = sum(episodes_in_memory)
            avg_length = np.mean(episodes_in_memory)
            min_length = min(episodes_in_memory)
            max_length = max(episodes_in_memory)
            
            print(f"\n📈 Episode Statistics:")
            print(f"   Total timesteps: {total_timesteps:,}")
            print(f"   Average: {avg_length:.1f} steps")
            print(f"   Min: {min_length} steps")
            print(f"   Max: {max_length} steps")
            
            if detailed:
                print(f"\n📋 Per-Episode Breakdown:")
                for i, steps in enumerate(episodes_in_memory):
                    status = "✓ Saved" if self.save_path.exists() else "⚠ Unsaved"
                    print(f"   episode_{i}: {steps} steps - {status}")
        
        # Show all files in directory
        if self.save_dir.exists():
            all_files = sorted(self.save_dir.glob("robot_demos_*.hdf5"))
            if all_files:
                print(f"\n📂 All Sessions in {self.save_dir}:")
                total_all_episodes = 0
                for file_path in all_files:
                    try:
                        with h5py.File(file_path, 'r') as f:
                            num_eps = len(f.keys())
                            total_all_episodes += num_eps
                            marker = "← Current" if file_path == self.save_path else ""
                            print(f"   {file_path.name}: {num_eps} episodes {marker}")
                    except:
                        print(f"   {file_path.name}: Error reading")
                print(f"\n   Total episodes across all sessions: {total_all_episodes}")
        
        print(f"\n🎯 Training Readiness:")
        if total_in_memory > 0:
            print(f"   ⚠ WARNING: {total_in_memory} unsaved episodes!")
            print(f"   → Click 'Save All Demos' to save them")
        else:
            print(f"   ✓ All episodes saved")
        
        print("="*60 + "\n")
    
    def get_quick_summary(self):
        """Print a quick one-line summary"""
        unsaved = len(self.episodes)
        status = "✓" if unsaved == 0 else f"⚠ {unsaved} unsaved"
        print(f"[STATS] This session: {len(self.episodes)} episodes {status}")

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple tabletop scene with a robot."""

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

    # Elevated platform/table for the pickable cube
    cube_table = AssetBaseCfg(
        prim_path="/World/CubeTable",
        spawn=sim_utils.CuboidCfg(
            size=[0.4, 0.4, 0.6],  # 40cm x 40cm surface, 60cm tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.6, 0.4),  # Wood-like color
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0, -0.1)),  # Moved away from origin, base at ground
    )

    # Original cube
    cube = AssetBaseCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(size=[0.1, 0.1, 0.1]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )

    # Second pickable cube - now on elevated table
    cube2 = AssetBaseCfg(
        prim_path="/World/cube2",
        spawn=sim_utils.CuboidCfg(
            size=[0.03, 0.03, 0.03],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.9),  # Blue color for visibility
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0, 0.23)),  # On top of table, matching table position
    )

    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    
    # Camera sensor for recording demonstrations
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,  # 10 Hz
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.5),  # Position: behind and above the workspace
            rot=(0.9239, 0.0, 0.3827, 0.0),  # Looking down at 45 degrees
            convention="world",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    
    # Enable collisions for gripper to allow grasping
    print("[INFO] Enabling gripper collisions for object interaction...")

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

    # Initialize joint states
    if args_cli.robot == "franka_panda":
        joint_position = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)
    else:
        joint_position = torch.zeros((1, 6), device=sim.device)
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)

    # -----------------------
    # GRIPPER SETUP
    # -----------------------
    joint_names = [n.lower() for n in robot.data.joint_names]
    gripper_candidates = [i for i, n in enumerate(joint_names) if any(k in n for k in ["finger", "gripper", "hand", "claw", "panda_finger"])]
    gripper_joint_ids = gripper_candidates

    if len(gripper_joint_ids) > 0:
        print("[INFO] Detected gripper joints:", [robot.data.joint_names[i] for i in gripper_joint_ids])
    else:
        print("[WARN] No gripper joints detected. Gripper control will be disabled.")

    gripper_open_pos = 0.04
    gripper_closed_pos = 0.0
    gripper_target_norm = 0.0
    gripper_open_bool = True

    def gripper_norm_to_joint_positions(norm):
        pos = gripper_open_pos + (gripper_closed_pos - gripper_open_pos) * norm
        if len(gripper_joint_ids) == 0:
            return None
        return torch.tensor([[pos] * len(gripper_joint_ids)], device=sim.device)

    # ========================================================================
    # ADD DEMONSTRATION RECORDER HERE - STEP 1: Initialize recorder
    # ========================================================================
    recorder = DemonstrationRecorder("demonstrations", "recordings")
    print("[INFO] Demonstration recorder initialized")
    
    # Get camera from scene for video recording
    print(f"[DEBUG] Scene keys: {list(scene.keys())}")
    camera = scene["camera"] if "camera" in scene.keys() else None
    if camera is not None:
        print(f"[INFO] Camera sensor found in scene")
        # Don't access camera.data here as it might not be initialized yet
    else:
        print("[WARN] No camera sensor found in scene. Video recording disabled.")

    # -----------------------
    # TELEOP / HEADLESS SETUP
    # -----------------------
    if not args_cli.headless:
        teleop = Se3Keyboard(pos_sensitivity=0.05, rot_sensitivity=0.05)
        teleop.reset()
        print("[INFO] Teleoperation active — use WASDQE to move and arrow keys to rotate.")
        print("[INFO] Smooth approach enabled: arm will slow down near goal")
        teleop_has_extra_keys = False

        gripper_state = {"open": True}

        def _toggle_gripper_cb():
            gripper_state["open"] = not gripper_state["open"]
            print(f"[UI] Gripper toggled -> {'OPEN' if gripper_state['open'] else 'CLOSED'}")

        # ========================================================================
        # ADD RECORDING CONTROLS HERE - STEP 2: Add UI buttons for recording
        # ========================================================================
        def _start_recording():
            recorder.start_episode()
            print("[RECORDING] Started new episode - perform your demonstration")

        def _stop_recording():
            recorder.end_episode()
            recorder.get_quick_summary()
            print("[RECORDING] Stopped episode")

        def _save_demos():
            recorder.save()
            recorder.get_stats(detailed=True) 
            print("[RECORDING] All demonstrations saved to file")

        # Create UI windows
        gripper_window = ui.Window("Gripper", width=180, height=80)
        with gripper_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Gripper Control")
                ui.Button("Toggle Gripper", clicked_fn=_toggle_gripper_cb, height=40)

        # Recording control window
        recording_window = ui.Window("Recording", width=180, height=150, position_x=200)
        with recording_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Demonstration Recording")
                ui.Button("Start Recording", clicked_fn=_start_recording, height=40)
                ui.Button("Stop Recording", clicked_fn=_stop_recording, height=40)
                ui.Button("Save All Demos", clicked_fn=_save_demos, height=40)

    else:
        step = 0
        headless_cycle_t = 0
        print("[INFO] Running headless simulation with scripted motion + gripper sequence...")

    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose = ee_pose_w.clone()
    #smooth_target_pose = goal_pose.clone()
    #previous_smooth_pose = smooth_target_pose.clone()

    # Main simulation loop
    while simulation_app.is_running():
        if not args_cli.headless:
            # GUI teleop - update goal based on user input
            '''try:
                ret = teleop.advance()
                if isinstance(ret, tuple) and len(ret) == 3:
                    pos_delta, rot_delta, extra_keys = ret
                    teleop_has_extra_keys = True
                else:
                    pos_delta, rot_delta = ret
            except TypeError:
                pos_delta, rot_delta = teleop.advance()

            # Update goal pose with teleop input
            goal_pose[:, 0:3] += torch.tensor(pos_delta[:3], device=goal_pose.device).unsqueeze(0)

            gripper_open_bool = gripper_state["open"]
            gripper_target_norm = 0.0 if gripper_open_bool else 1.0'''
            try:
                ret = teleop.advance()
                if isinstance(ret, tuple) and len(ret) == 3:
                    pos_delta, rot_delta, extra_keys = ret
                    teleop_has_extra_keys = True
                else:
                    pos_delta, rot_delta = ret
            except TypeError:
                pos_delta, rot_delta = teleop.advance()

            # Update goal pose DIRECTLY with teleop input (no smoothing)
            goal_pose[:, 0:3] += torch.tensor(pos_delta[:3], device=goal_pose.device).unsqueeze(0)
            
            # Also update rotation if available
            # Apply rotation only if rot_delta is a valid vector
            if isinstance(rot_delta, (list, tuple, np.ndarray)) and len(rot_delta) == 3:
                current_quat = goal_pose[:, 3:7]

                delta_quat = quat_from_euler_xyz(
                    torch.tensor([rot_delta[0]], device=sim.device),
                    torch.tensor([rot_delta[1]], device=sim.device),
                    torch.tensor([rot_delta[2]], device=sim.device),
                )

                goal_pose[:, 3:7] = quat_mul(delta_quat, current_quat)
                goal_pose[:, 3:7] = goal_pose[:, 3:7] / torch.norm(
                    goal_pose[:, 3:7], dim=1, keepdim=True
                )

            gripper_open_bool = gripper_state["open"]
            gripper_target_norm = 0.0 if gripper_open_bool else 1.0

        else:
            # Headless scripted motion
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

        # -----------------------
        # SMOOTH APPROACH LOGIC
        # -----------------------
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        
        '''position_error = goal_pose[:, 0:3] - smooth_target_pose[:, 0:3]
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
        
        alpha = 0.7
        smooth_target_pose = alpha * smooth_target_pose + (1 - alpha) * previous_smooth_pose
        smooth_target_pose[:, 3:7] = smooth_target_pose[:, 3:7] / torch.norm(smooth_target_pose[:, 3:7], dim=1, keepdim=True)
        previous_smooth_pose = smooth_target_pose.clone()'''

        # IK computation
        # diff_ik_controller.set_command(smooth_target_pose)
        diff_ik_controller.set_command(goal_pose)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Set arm joint targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # Apply gripper joint targets
        if len(gripper_joint_ids) > 0:
            gripper_joint_positions = gripper_norm_to_joint_positions(gripper_target_norm)
            if gripper_joint_positions is not None:
                robot.set_joint_position_target(
                    gripper_joint_positions,
                    joint_ids=gripper_joint_ids
                )

        # ========================================================================
        # RECORD DEMONSTRATION DATA HERE - STEP 3: Record transitions
        # ========================================================================
        # Create observation (matching what you'll use for training)
        joint_pos_full = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel_full = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]
        
        # Observation: [joint_pos(7), joint_vel(7), ee_pose(7), gripper_state(1)]
        obs = torch.cat([
            joint_pos_full,
            joint_vel_full,
            ee_pose_w,
            torch.tensor([[gripper_target_norm]], device=sim.device)
        ], dim=-1)
        
        # Action: joint position targets + gripper target
        action = torch.cat([
            joint_pos_des,
            torch.tensor([[gripper_target_norm]], device=sim.device)
        ], dim=-1)
        
        # Record the transition
        # Capture camera frame if camera is available
        video_frame = None
        if camera is not None:
            try:
                # Get RGB image from camera - IsaacLab returns warp arrays
                rgb_data = camera.data.output["rgb"]
                if rgb_data is not None:
                    # Handle different tensor formats
                    if hasattr(rgb_data, 'shape') and len(rgb_data.shape) >= 3:
                        # Get first environment's data if batched
                        frame_data = rgb_data[0] if len(rgb_data.shape) == 4 else rgb_data
                        # Convert to numpy - handle torch tensors and warp arrays
                        if hasattr(frame_data, 'cpu'):
                            # PyTorch tensor
                            video_frame = frame_data.clone().cpu().numpy()
                        elif hasattr(frame_data, 'numpy'):
                            # Has numpy method
                            video_frame = frame_data.numpy().copy()
                        else:
                            # Try direct numpy array conversion (assuming np is imported)
                            video_frame = np.asarray(frame_data).copy()
                        # Ensure uint8 format for video
                        if video_frame.dtype != np.uint8:
                            video_frame = (video_frame * 255).astype(np.uint8) if video_frame.max() <= 1.0 else video_frame.astype(np.uint8)
            except Exception as e:
                # Print error once to help debug
                if not hasattr(camera, '_error_printed'):
                    print(f"[WARN] Camera frame capture error: {e}")
                    import traceback
                    traceback.print_exc()
                    camera._error_printed = True
        
        recorder.add_transition(
            obs=obs,
            action=action,
            ee_pose=ee_pose_w,
            joint_pos=joint_pos_full,
            video_frame=video_frame
        )

        # Write/update simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Visualize EE & goal
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