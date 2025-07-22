# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

import copy
import dataclasses
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .so_100_robot_cfg import SO100_CFG

CAMERA_HEIGHT = 144
CAMERA_WIDTH = 256

VIEWER_EYE = (0.15, 0.05, 2)
VIEWER_LOOKAT = (-0.15, -0.05, 1.9)

TABLE_POS = (0.5, 0.0, 1.05)
TABLE_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

ROBOT_POS = (0.0, 0.0, 1.05)
ROBOT_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

INITIAL_CUBE_POS = (0.2, 0.0, 1.065)
INITIAL_CUBE_ROT = (1.0, 0.0, 0.0, 0.0)
RANDOMIZATION_RANGE_CUBE_X = 0.05
RANDOMIZATION_RANGE_CUBE_Y = 0.2
RANDOMIZATION_RANGE_CUBE_Z = 0.0


@configclass
class So100GraspEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 6
    action_scale_robot = 0.0
    state_space = 0
    observation_space = spaces.Dict({
        "camera": spaces.Box(low=0.0, high=1.0, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.float32),
        "proprioceptive": spaces.Box(low=float("-inf"), high=float("inf"), shape=(18,), dtype=np.float32),
    })

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,  # 100Hz
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
    )

    viewer = ViewerCfg(eye=VIEWER_EYE, lookat=VIEWER_LOOKAT)

    # robot(s)
    robot_cfg: ArticulationCfg = dataclasses.replace(SO100_CFG, prim_path="/World/envs/env_.*/Robot")
    if robot_cfg.init_state is None:
        robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
    robot_cfg.init_state = dataclasses.replace(robot_cfg.init_state, pos=ROBOT_POS,rot=ROBOT_ROT)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.5, replicate_physics=True)

    # Joint names for action mapping
    # JOINT NAMES:  ['Shoulder_Rotation', 'Shoulder_Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Gripper']
    # BODY NAMES:  ['Base', 'Shoulder_Rotation_Pitch', 'Upper_Arm', 'Lower_Arm', 'Wrist_Pitch_Roll', 'Fixed_Gripper', 'Moving_Jaw']
    dof_names = ["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Gripper"]
    
    # Object configuration
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=INITIAL_CUBE_POS, 
            rot=INITIAL_CUBE_ROT
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Camera configuration
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/Wrist_Pitch_Roll/Gripper_Camera/Camera_SG2_OX03CC_5200_GMSL2_H60YA",
        update_period=0.04,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(180.0, 0.0, 0.0, 0.0), convention="ros"),
        spawn=None
    )

    table_cfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS, rot=TABLE_ROT),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    # Configure end-effector marker
    goal_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
    # Properly replace the frame marker configuration
    goal_marker_cfg.markers = {
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.03, 0.03, 0.03),
        )
    }
    goal_marker_cfg.prim_path = "/Visuals/ee_goal"


    # Configure end-effector marker
    marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
    # Properly replace the frame marker configuration
    marker_cfg.markers = {
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        )
    }
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Base",
        visualizer_cfg=marker_cfg,
        debug_vis=False,  # disable visualization
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # Original path in comments for reference
                # prim_path="{ENV_REGEX_NS}/Robot/SO_100/SO_5DOF_ARM100_05d_SLDASM/Fixed_Gripper",
                # Updated path for the new USD structure
                prim_path="/World/envs/env_.*/Robot/Fixed_Gripper",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.01, -0.0, 0.1),
                ),
            ),
        ],
    )

    # Configure cube marker with different color and path
    cube_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
    cube_marker_cfg.markers = {
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
    cube_marker_cfg.prim_path = "/Visuals/CubeFrameMarker"
    
    cube_marker_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Object",
        visualizer_cfg=cube_marker_cfg,
        debug_vis=False,  # disable visualization
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object",
                name="cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    initial_cube_pos = INITIAL_CUBE_POS
    initial_cube_rot = INITIAL_CUBE_ROT
    randomization_range_cube_x = RANDOMIZATION_RANGE_CUBE_X
    randomization_range_cube_y = RANDOMIZATION_RANGE_CUBE_Y
    randomization_range_cube_z = RANDOMIZATION_RANGE_CUBE_Z
    
    # Reward parameters
    grasping_reward_weight = 2.0
    object_ee_distance_std = 0.05
    action_penalty_weight = -1e-4
    joint_vel_penalty_weight = -1e-4