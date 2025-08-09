# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import numpy as np
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation as LabArticulation
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, combine_frame_transforms, subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot_motion.motion_generation")

from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver, interface_config_loader
from isaacsim.core.prims import SingleArticulation as SimSingleArticulation


from .so_100_grasp_env_cfg import So100GraspEnvCfg

_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DESCRIPTOR_PATH = os.path.join(_THIS_SCRIPT_DIR, "asset", "SO_5DOF_ARM100_WITH_CAMERA_Descriptor.yaml")
ROBOT_URDF_PATH = os.path.join(_THIS_SCRIPT_DIR, "asset", "urdf", "SO_5DOF_ARM100_8j_URDF.SLDASM", "urdf", "SO_5DOF_ARM100_8j_URDF.SLDASM.urdf")


class So100GraspEnv(DirectRLEnv):
    cfg: So100GraspEnvCfg

    def __init__(self, cfg: So100GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=ROBOT_DESCRIPTOR_PATH,
            urdf_path=ROBOT_URDF_PATH,
        )
        print(f"kinematics solver all frame names: {self.kinematics_solver.get_all_frame_names()}")


        # Get joint indices for action mapping
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.action_scale_robot = self.cfg.action_scale_robot

        # # Create controller
        # diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        # self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.device)

        # indices, body_names = self.robot.find_bodies(self.robot.body_names, preserve_order=True)     
        # self.body_ids = indices[body_names.index("Moving_Jaw")]
        # print("body_ids: ", self.body_ids)
        # # Obtain the frame index of the end-effector
        # # For a fixed base robot, the frame index is one less than the body index. This is because
        # # the root body is not included in the returned Jacobians.
        # self.ee_jacobi_idx = indices[self.body_ids] - 1
        # print("ee_jacobi_idx: ", self.ee_jacobi_idx)
        
        # self.ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)
        # self.ik_commands[:, :3] = torch.tensor(self.cfg.initial_cube_pos, device=self.device)
        # self.ik_commands[:, 3:7] = torch.tensor(self.cfg.initial_cube_rot, device=self.device)

    def _setup_scene(self):
        """Set up the simulation scene."""
        self.robot = LabArticulation(self.cfg.robot_cfg)
        self.ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.cube_marker = FrameTransformer(self.cfg.cube_marker_cfg)
        self.camera = Camera(self.cfg.camera_cfg)
        self.goal_marker = VisualizationMarkers(self.cfg.goal_marker_cfg)
        table_cfg = self.cfg.table_cfg
        table_cfg.spawn.func(
            table_cfg.prim_path, table_cfg.spawn,
            translation=table_cfg.init_state.pos,
            orientation=table_cfg.init_state.rot
        )
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["cube_marker"] = self.cube_marker
        self.scene.sensors["camera"] = self.camera
        self.scene.extras["goal_marker"] = self.goal_marker
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.actions = actions.clone()
        self.actions[:, :6] = self.action_scale_robot * self.actions[:, :6]
        self.actions[:, 5:6] = torch.clamp(self.actions[:, 5:6], min=0.0, max=0.5)

    def _apply_action(self) -> None:
        # apply arm actions
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.robot.set_joint_position_target(self.actions[:, :5], joint_ids=self.dof_idx[:5])
        # apply gripper actions
        self.robot.set_joint_position_target(self.actions[:, 5:6], joint_ids=self.dof_idx[5:6])
        self.last_actions = self.actions.clone()
        # if self.common_step_counter % 1000==0:
        #     print(f"last actions: {self.last_actions}")

    def _get_observations(self) -> dict:
        """Get observations for the policy."""
        # Joint positions (relative to initial positions)
        joint_pos_rel = self._joint_pos_rel()
        # Joint velocities (relative to initial velocities)
        joint_vel_rel = self._joint_vel_rel()
        # get camera rgb
        camera_rgb = self._get_camera_rgb()
        # Concatenate all observations
        states = torch.cat([
            joint_pos_rel,      # 6 dims
            joint_vel_rel,      # 6 dims
            self.last_actions,  # 6 dims
        ], dim=-1)
        obs = {
            "proprioceptive": states,
            "camera": camera_rgb,
        }
        observations = {
            "policy": obs
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on the manager-based environment reward structure."""
        grasping_object = self._object_grasped()
        
        action_rate_penalty = self._action_rate_penalty(self.actions, self.last_actions)

        joint_vel_penalty = self._joint_vel_penalty()
        
        if self.common_step_counter > 12000:
            action_rate_penalty_weight = -5e-4
            joint_vel_penalty_weight = -5e-4
        else:
            action_rate_penalty_weight = self.cfg.action_penalty_weight
            joint_vel_penalty_weight = self.cfg.joint_vel_penalty_weight
        # Combine all rewards with weights
        total_reward = (
            self.cfg.grasping_reward_weight * grasping_object +
            action_rate_penalty_weight * action_rate_penalty +
            joint_vel_penalty_weight * joint_vel_penalty
        )
        # if self.common_step_counter % 1000 == 0:
        #     print(f"reward at step {self.common_step_counter} is {total_reward.unsqueeze(-1)}")
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags based on manager-based environment termination conditions."""
        # 1. Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 2. Object dropping (root height below minimum)
        object_height = self.object.data.root_pos_w[:, 2]
        object_dropping = object_height < 1.0
        return object_dropping, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specific environments based on manager-based environment reset logic."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Call super to manage internal buffers (episode length, etc.)
        super()._reset_idx(env_ids)
        # Get the origins for the environments being reset
        env_origins = self.scene.env_origins[env_ids]  # shape: (num_envs, 3)
        default_root_state = self.object.data.default_root_state[env_ids]
        default_object_pos = default_root_state[:, :3] + env_origins
        # Randomize the object position
        object_pos_x = sample_uniform(lower=-self.cfg.randomization_range_cube_x, upper=self.cfg.randomization_range_cube_x, size=(self.num_envs,), device=self.device)
        object_pos_y = sample_uniform(lower=-self.cfg.randomization_range_cube_y, upper=self.cfg.randomization_range_cube_y, size=(self.num_envs,), device=self.device)
        object_pos_z = sample_uniform(lower=-self.cfg.randomization_range_cube_z, upper=self.cfg.randomization_range_cube_z, size=(self.num_envs,), device=self.device)
        object_pos = torch.stack([object_pos_x, object_pos_y, object_pos_z], dim=1)
        object_pos = default_object_pos + object_pos
        object_quat = default_root_state[:, 3:7]
        object_vel = default_root_state[:, 7:13]
        self.object.data.root_pos_w[env_ids] = object_pos
        self.object.data.root_quat_w[env_ids] = object_quat
        self.object.data.root_vel_w[env_ids] = object_vel
        default_root_state[:, :3] = object_pos
        self.object.write_root_state_to_sim(default_root_state, env_ids)

        self.goal_marker.visualize(object_pos)

        target_pos = object_pos.clone().detach().cpu().numpy()
        target_quat = object_quat.clone().detach().cpu().numpy()

        for env_id in env_ids:
            prim_path = f"/World/envs/env_{env_id}/Robot"

            # 1) Sim articulation per env
            sim_art = SimSingleArticulation(prim_path=prim_path)
            sim_art.initialize()

            # 2) Build per-env ArticulationKinematicsSolver AFTER init
            end_effector_frame = "Fixed_Gripper"  # <- ensure this exists in solver.get_all_frame_names()
            #art_ik = ArticulationKinematicsSolver(sim_art, self.kinematics_solver, end_effector_frame)

            # 3) Update base pose for this env
            base_t, base_q = sim_art.get_world_pose()
            self.kinematics_solver.set_robot_base_pose(
                base_t.clone().detach().cpu().numpy(),
                base_q.clone().detach().cpu().numpy()
            )

            # 4) Use current arm joints as a single-row seed (5 DoF arm)
            # current 5-DoF arm joints as seed (no gripper)
            # q_seed = self.robot.data.joint_pos[env_id, self.dof_idx[:5]].detach().cpu().numpy()
            # q_seed = q_seed.astype(np.float64).reshape(-1, 1)     # (5, 1) float64 column
            # self.kinematics_solver.set_default_cspace_seeds(q_seed)  # shape (1,5)
            q_seed = (
                self.robot.data.joint_pos[env_id, self.dof_idx[:5]]  # tensor shape (5,)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1, 1)  # shape (5, 1) as required
            )
            print(f"q_seed shape: {q_seed.shape}, dtype: {q_seed.dtype}")

            # Register this seed with the solver; must be a list of np.ndarray
            self.kinematics_solver.set_default_cspace_seeds([q_seed])

            # robot_base_translation, robot_base_orientation = sim_art.get_world_pose()
            # robot_base_translation = robot_base_translation.clone().detach().cpu().numpy()
            # robot_base_orientation = robot_base_orientation.clone().detach().cpu().numpy()

            #self.kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)

            # 5) Target pose for THIS env (world frame)
            pos_w  = np.asarray(target_pos[env_id],  dtype=np.float64).reshape(3)
            quat_w = np.asarray(target_quat[env_id], dtype=np.float64).reshape(4)

            # 6) Solve IK and apply
            q_sol, success = self.kinematics_solver.compute_inverse_kinematics(
                frame_name=end_effector_frame, 
                target_position=pos_w, 
                target_orientation=quat_w,
                warm_start=q_seed,
                position_tolerance=10,      # relax tolerances if needed
                orientation_tolerance=0.9
            )
            if not success:
                print(f"[WARNING] IK failed for env {env_id}")
                continue

            # q_sol is full arm vector; send first 5 joints to Lab (gripper handled separately)
            self.robot.set_joint_position_target(
                torch.tensor(q_sol[:5], device=self.device),
                joint_ids=self.dof_idx[:5],
                env_ids=[env_id],
            )
        self.robot.write_data_to_sim()

        self.last_actions[env_ids] = 0
        print("now the robot is reset")

    def _joint_pos_rel(self) -> torch.Tensor:
        """Get joint positions relative to initial positions."""
        return self.robot.data.joint_pos[:, self.dof_idx] - self.robot.data.default_joint_pos[:, self.dof_idx]

    def _joint_vel_rel(self) -> torch.Tensor:
        """Get joint velocities relative to initial velocities."""
        return self.robot.data.joint_vel[:, self.dof_idx] - self.robot.data.default_joint_vel[:, self.dof_idx]

    def _get_camera_rgb(self) -> torch.Tensor:
        """Get camera RGB images."""
        camera_data = self.camera.data.output
        if camera_data is None or "rgb" not in camera_data:
            print("[WARNING] Camera data is not available. Returning zero tensor.")
            return torch.zeros((self.num_envs, self.cfg.CAMERA_HEIGHT, self.cfg.CAMERA_WIDTH, 3), device=self.device)
        return camera_data["rgb"] / 255.0

    def _object_grasped(self, gripper_threshold: float = 0.3) -> torch.Tensor:
        z = self.object.data.root_pos_w[:, 2]
        gripper_index = self.robot.data.joint_names.index("Gripper")
        gripper_state = self.robot.data.joint_pos[:, gripper_index]
        #grasped = (z < 1.08) & (gripper_state < 0.3)
        return (gripper_state < gripper_threshold).float()
        
    def _action_rate_penalty(self, actions, prev_actions) -> torch.Tensor:
        """Penalize the rate of change of the actions using L2 squared kernel."""
        return torch.sum(torch.square(actions - prev_actions), dim=1)

    def _joint_vel_penalty(self) -> torch.Tensor:
        """Penalize the joint velocities using L2 norm."""
        return torch.sum(torch.square(self.robot.data.joint_vel[:, self.dof_idx]), dim=1) 