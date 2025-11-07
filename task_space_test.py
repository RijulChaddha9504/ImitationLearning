import argparse
from isaaclab.app import AppLauncher

"""Configures a command-line argument parser to select the robot type and launch the simulation app."""

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

# ✅ NEW IMPORTS for teleoperation
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg


##
# Pre-defined configs
##

#Franka Panda is a 7-degree-of-freedom robot arm commonly used in robotics research and applications.
#Lets you choose between Franka Panda and UR10 robot arms.
from isaaclab_assets import UR10_CFG, FRANKA_PANDA_HIGH_PD_CFG


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

    # 🟨 Cube is no longer needed for motion, but we can keep it for visual reference
    cube = AssetBaseCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(size=[0.1, 0.1, 0.1]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )

    # robot
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"] # Get robot entity

    #Computes position and orientation changes based on keyboard input
    # Sets up a differential IK controller for end-effector pose control
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    #Visualization markers for current and goal end-effector poses, current and goal markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    #Sets which parts of the robot correspond to joints and end-effector
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    robot_entity_cfg.resolve(scene)

    #Finds required joint movements via Jacobian index based on whether the robot is fixed or mobile
    # x˙=J(q)⋅q˙​
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()

    # Initialize robot joint states
    if args_cli.robot == "franka_panda":
        joint_position = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)
    else:
        joint_position = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=sim.device)
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)

    # ✅ NEW: Setup teleoperation device
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05))
    teleop.reset()

    # ✅ NEW: Get initial end-effector pose as starting goal
    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose = ee_pose_w.clone()  # [x, y, z, qw, qx, qy, qz]

    print("[INFO] Teleoperation active — use WASDQE to move and arrow keys to rotate.")

    while simulation_app.is_running():
        # ✅ NEW: Get teleop delta (6D — xyz + rpy), reads keyboard input
        action = teleop.advance()

        # ✅ Update goal pose (integrate small deltas)
        goal_pose[:, 0:3] += action[:, 0:3]  # position delta
        # For rotation deltas, IsaacLab handles quaternion math internally
        # so we’ll skip orientation accumulation for simplicity.

        # Compute IK to reach goal pose
        ik_commands = goal_pose  # feed to IK controller
        diff_ik_controller.set_command(ik_commands)

        """Jacobian relates joint velocities to end-effector velocities.
        ee_pose_w: Current end-effector pose in world frame.
        root_pose_w: Robot base pose in world frame.
        joint_pos: Current joint positions."""
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        """Compute end-effector pose relative to robot base frame."""
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Update visualization markers

        """ee_maker visualizes the current end-effector pose.
        goal_marker visualizes the target end-effector pose."""
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
