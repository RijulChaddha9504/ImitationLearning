import argparse
from isaaclab.app import AppLauncher

"""Robot Arm Teleoperation (headless-compatible) with Task Space IK Control"""

parser = argparse.ArgumentParser(description="Robot Arm Teleoperation with Task Space IK Control")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import UR10_CFG, FRANKA_PANDA_HIGH_PD_CFG

# Import Se3Keyboard and Omni UI only if GUI mode
if not args_cli.headless:
    from isaaclab.devices import Se3Keyboard
    import omni.ui as ui


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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0, -0.1)),  # Moved away from origin, base at ground
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0, 0.23)),  # On top of table, matching table position
    )

    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    
    # Enable collisions for gripper to allow grasping
    # This ensures the gripper fingers can interact with objects
    print("[INFO] Enabling gripper collisions for object interaction...")

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # -----------------------
    # SMOOTH APPROACH PARAMETERS
    # -----------------------
    # Proportional gain for smooth convergence (lower = smoother/slower)
    position_p_gain = 0.15  # Reduced for stability (tune between 0.1-0.3)
    rotation_p_gain = 0.1   # Lower for rotation stability
    
    # Distance-based scaling parameters
    slow_zone_threshold = 0.15  # Start slowing down within 15cm
    min_gain = 0.05  # Minimum speed when very close (5% of max speed)
    
    # Deadband threshold to stop oscillations when very close
    position_deadband = 0.001  # Stop moving if error < 1mm
    rotation_deadband = 0.01   # Stop rotating if quaternion difference < 0.01

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

        gripper_window = ui.Window("Gripper", width=180, height=80)
        with gripper_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Gripper Control")
                ui.Button("Toggle Gripper", clicked_fn=_toggle_gripper_cb, height=40)

    else:
        step = 0
        headless_cycle_t = 0
        print("[INFO] Running headless simulation with scripted motion + gripper sequence...")

    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose = ee_pose_w.clone()
    
    # Smooth target pose that converges to goal_pose
    smooth_target_pose = goal_pose.clone()

    while simulation_app.is_running():
        if not args_cli.headless:
            # GUI teleop - update goal based on user input
            try:
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
        # Get current end-effector position
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        
        # Calculate position error
        position_error = goal_pose[:, 0:3] - smooth_target_pose[:, 0:3]
        distance_to_goal = torch.norm(position_error, dim=1, keepdim=True)
        
        # Apply deadband to prevent tiny oscillations
        if distance_to_goal.item() > position_deadband:
            # Distance-based adaptive gain (slows down when close)
            adaptive_gain = torch.clamp(
                distance_to_goal / slow_zone_threshold, 
                min=min_gain, 
                max=1.0
            )
            
            # Apply proportional control with adaptive gain
            smooth_target_pose[:, 0:3] += position_p_gain * adaptive_gain * position_error
        
        # Smooth rotation with quaternion sign handling
        goal_quat = goal_pose[:, 3:7]
        current_quat = smooth_target_pose[:, 3:7]
        
        # Handle quaternion sign ambiguity (q and -q represent same rotation)
        # Choose the shortest path by ensuring dot product is positive
        dot_product = torch.sum(goal_quat * current_quat, dim=1, keepdim=True)
        goal_quat_corrected = torch.where(dot_product < 0, -goal_quat, goal_quat)
        
        # Calculate rotation error with corrected quaternion
        rotation_error = goal_quat_corrected - current_quat
        rotation_error_magnitude = torch.norm(rotation_error, dim=1, keepdim=True)
        
        # Apply deadband for rotation
        if rotation_error_magnitude.item() > rotation_deadband:
            smooth_target_pose[:, 3:7] += rotation_p_gain * rotation_error
            
            # Normalize quaternion to maintain valid rotation
            smooth_target_pose[:, 3:7] = smooth_target_pose[:, 3:7] / torch.norm(smooth_target_pose[:, 3:7], dim=1, keepdim=True)

        # Feed smoothed target to IK controller
        diff_ik_controller.set_command(smooth_target_pose)

        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # set arm joint targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # Apply gripper joint targets
        if len(gripper_joint_ids) > 0:
            gripper_joint_positions = gripper_norm_to_joint_positions(gripper_target_norm)
            if gripper_joint_positions is not None:
                robot.set_joint_position_target(
                    gripper_joint_positions,
                    joint_ids=gripper_joint_ids
                )

        # write/update simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # visualize EE & goal
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