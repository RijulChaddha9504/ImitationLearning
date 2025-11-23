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

# ✅ Import keyboard teleoperation only if GUI mode
if not args_cli.headless:
    from isaaclab.devices import Se3Keyboard

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

    cube = AssetBaseCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(size=[0.1, 0.1, 0.1]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )

        # Second pickable cube
    cube2 = AssetBaseCfg(
        prim_path="/World/cube2",
        spawn=sim_utils.CuboidCfg(
            size=[0.05, 0.05, 0.05],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.45, 0.15, 0.02),
        ),
    )


    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    robot_entity_cfg.resolve(scene)

    # find ee jacobi index (existing)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()

    # Initialize joint states (existing)
    if args_cli.robot == "franka_panda":
        joint_position = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)
    else:
        joint_position = torch.zeros((1, 6), device=sim.device)
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_position, joint_vel)

    # -----------------------
    # GRIPPER SETUP (new)
    # -----------------------
    # find gripper joint indices by name heuristics
    joint_names = [n.lower() for n in robot.data.joint_names]  # list of joint name strings
    gripper_candidates = []
    for i, n in enumerate(joint_names):
        if ("finger" in n) or ("gripper" in n) or ("hand" in n) or ("claw" in n) or ("panda_finger" in n):
            gripper_candidates.append(i)

    # If none found, leave empty and we simply skip gripper actions
    gripper_joint_ids = gripper_candidates  # list of integers
    if len(gripper_joint_ids) > 0:
        print("[INFO] Detected gripper joints:", [robot.data.joint_names[i] for i in gripper_joint_ids])
    else:
        print("[WARN] No gripper joints detected. Gripper control will be disabled.")

    # mapping from normalized target [0..1] to actual joint positions (tune as needed)
    # for Franka Panda finger joints open ~0.04, closed ~0.0 (tune per robot)
    # we pick defaults that are reasonable; you may adjust per your URDF.
    gripper_open_pos = 0.04
    gripper_closed_pos = 0.0

    # normalized in [0..1], 0=open, 1=closed
    gripper_target_norm = 0.0

    # helper to convert normalized target to per-joint tensor
    def gripper_norm_to_joint_positions(norm):
        pos = gripper_open_pos + (gripper_closed_pos - gripper_open_pos) * norm
        # return shape (1, num_gripper_joints)
        if len(gripper_joint_ids) == 0:
            return None
        return torch.tensor([[pos] * len(gripper_joint_ids)], device=sim.device)

    # -----------------------
    # TELEOP / HEADLESS SETUP (existing + small edits)
    # -----------------------
    if not args_cli.headless:
        # GUI mode: use keyboard teleop
        teleop = Se3Keyboard(pos_sensitivity=0.05, rot_sensitivity=0.05)
        teleop.reset()
        print("[INFO] Teleoperation active — use WASDQE to move and arrow keys to rotate.")
        print("[INFO] Press 'q' to toggle gripper open/close (keyboard package) or use teleop extra-key API if available.")
        # Option A (recommended): use `keyboard` python package to detect presses
        try:
            import keyboard as _kb  # pip install keyboard
            kb_available = True
            print("[INFO] 'keyboard' package found — using it for gripper toggle.")
        except Exception:
            _kb = None
            kb_available = False
            print("[WARN] 'keyboard' package not available. If Se3Keyboard exposes a key API you can adapt the code to use it.")
    else:
        # Headless: use scripted motion and scripted gripper sequence
        step = 0
        headless_cycle_t = 0
        print("[INFO] Running headless simulation with scripted motion + gripper sequence...")

    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    goal_pose = ee_pose_w.clone()

    # helper state for gripper toggle (GUI)
    last_q_state = False

    while simulation_app.is_running():
        if not args_cli.headless:
            # GUI teleoperation
            pos_delta, rot_delta = teleop.advance()
            trans_delta = pos_delta[:3]

            goal_pose[:, 0:3] += torch.tensor(trans_delta, device=goal_pose.device).unsqueeze(0)

            # --- gripper toggle via keyboard package (press 'q' to toggle)
            if kb_available:
                # non-blocking; returns True only while key is pressed; we want toggle on key-down edge
                q_pressed = _kb.is_pressed("q")
                if q_pressed and not last_q_state:
                    # toggle between open (0.0) and closed (1.0)
                    gripper_target_norm = 1.0 if gripper_target_norm < 0.5 else 0.0
                    print("[INFO] Gripper toggled ->", "CLOSED" if gripper_target_norm > 0.5 else "OPEN")
                last_q_state = q_pressed
            else:
                # If keyboard package not available, you can extend Se3Keyboard (if it exposes keys)
                # Example stub (uncomment & adapt if Se3Keyboard has a method like get_button):
                # if hasattr(teleop, "get_button") and teleop.get_button("q"):
                #     gripper_target_norm = 1.0 if gripper_target_norm < 0.5 else 0.0
                pass

        else:
            # Headless scripted motion (existing)
            delta_pos = 0.01 * torch.sin(torch.tensor(step * 0.1))
            goal_pose[:, 0] += delta_pos
            step += 1

            # simple gripper sequence: open for 0..100 steps, close for 101..200, repeat
            headless_cycle_t += 1
            cycle_len = 200
            tmod = headless_cycle_t % cycle_len
            if tmod < (cycle_len // 2):
                gripper_target_norm = 0.0  # open
            else:
                gripper_target_norm = 1.0  # closed

        # Feed IK
        diff_ik_controller.set_command(goal_pose)

        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # set arm joint targets (existing)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # -----------------------
        # Apply gripper joint targets (new)
        # -----------------------
        if len(gripper_joint_ids) > 0:
            gripper_joint_positions = gripper_norm_to_joint_positions(gripper_target_norm)
            if gripper_joint_positions is not None:
                # robot.set_joint_position_target expects shape matching robot.data.joint_pos
                # Provide per-joint targets for each gripper joint id
                # gripper_joint_positions shape: (1, num_gripper_joints)
                robot.set_joint_position_target(
                    gripper_joint_positions,
                    joint_ids=gripper_joint_ids
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
