
import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import  get_args, task_registry
import torch
import numpy as np
import time

# --- Monkey Patch Start ---
# Fix missing attributes for "plane" terrain to prevent initialization crashes caused by Parkour logic
_original_get_env_origins = LeggedRobot._get_env_origins

def _patched_get_env_origins(self):
    _original_get_env_origins(self)
    if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
        device = self.device
        num_envs = self.num_envs
        # Goals for Parkour Logic
        if not hasattr(self, 'cur_goal_idx') or self.cur_goal_idx is None:
             self.cur_goal_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        if not hasattr(self, 'reach_goal_timer') or self.reach_goal_timer is None:
             self.reach_goal_timer = torch.zeros(num_envs, dtype=torch.float, device=device)
        if not hasattr(self, 'cur_goals') or self.cur_goals is None:
            self.cur_goals = torch.zeros(num_envs, 3, device=device, requires_grad=False)
        if not hasattr(self, 'next_goals') or self.next_goals is None:
            self.next_goals = torch.zeros(num_envs, 3, device=device, requires_grad=False)
        if not hasattr(self, 'env_goals') or self.env_goals is None:
             self.env_goals = torch.zeros(num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=device, requires_grad=False)
        
        # Terrain info
        if not hasattr(self, 'env_class') or self.env_class is None:
            self.env_class = torch.zeros(num_envs, device=device, requires_grad=False)
        if not hasattr(self, 'terrain_levels') or self.terrain_levels is None:
            self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
        if not hasattr(self, 'terrain_types') or self.terrain_types is None:
            self.terrain_types = torch.zeros(num_envs, dtype=torch.long, device=device)
        if not hasattr(self, 'terrain_goals') or self.terrain_goals is None:
            self.terrain_goals = torch.zeros(num_envs, self.cfg.terrain.num_goals, 3, device=device, requires_grad=False)

    # Values missing for observations (because reset_idx didn't run fully or compute_observations crashed)
    if not hasattr(self, 'target_yaw') or self.target_yaw is None:
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
    if not hasattr(self, 'next_target_yaw') or self.next_target_yaw is None:
        self.next_target_yaw = torch.zeros(self.num_envs, device=self.device)
    if not hasattr(self, 'yaw') or self.yaw is None:
        self.yaw = torch.zeros(self.num_envs, device=self.device)

LeggedRobot._get_env_origins = _patched_get_env_origins
# Also patch global methods to prevent other crashes
LeggedRobot._draw_goals = lambda self: None
LeggedRobot._update_terrain_curriculum = lambda self, env_ids: None 
LeggedRobot._update_goals = lambda self: None
# --- Monkey Patch End ---

def tune_pd(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 1. Setup Environment for Standing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane' # Flat ground
    
    # IMPORTANT: Enable gravity and physics execution
    env_cfg.asset.fix_base_link = False # Allow robot to fall/stand
    env_cfg.asset.disable_gravity = False
    
    # Initial Position (dropped slightly above ground)
    env_cfg.init_state.pos = [0.0, 0.0, 0.4] # Adjust height slightly above standing height
    env_cfg.init_state.rot = [0.0, 0.0, 0.0, 1.0]
    
    # Ensure correct URDF settings
    env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hustdog2/urdf/hustdog2.urdf'
    env_cfg.asset.collapse_fixed_joints = False 
    env_cfg.asset.flip_visual_attachments = False
    
    # Prepare PD Config to be tunable (initially from config)
    tune_stiffness = env_cfg.control.stiffness['joint']
    tune_damping = env_cfg.control.damping['joint']
    
    # Disable rewards to avoid crashes with terrain-based rewards
    class EmptyScales: pass
    env_cfg.rewards.scales = EmptyScales()

    # Create Env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # --- Disable Rewards Logic in Runtime too ---
    env.reward_scales = {}
    env.reward_functions = [] 
    
    if env.viewer is None:
        print("No viewer found. Use GUI mode.")
        return

    gym = env.gym
    viewer = env.viewer

    # Setup Controls
    print(f"\n{'='*50}")
    print("PD Tuning Mode (Standing)")
    print("Robot should be standing on the ground under gravity.")
    print("-" * 50)
    print("Controls:")
    print("  UP/DOWN    : Increase/Decrease Stiffness (P) by 5.0")
    print("  LEFT/RIGHT : Increase/Decrease Damping (D) by 0.5")
    print("  R          : Reset Robot State")
    print("  Q          : Quit")
    print("-" * 50)
    print(f"Initial P: {tune_stiffness} | D: {tune_damping}")
    print(f"{'='*50}\n")
    
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "P_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "P_DOWN")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "D_DOWN")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "D_UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "RESET")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "QUIT")

    action_scale = env.cfg.control.action_scale
    
    # Main Loop
    last_print = 0
    while not gym.query_viewer_has_closed(viewer):
        # Handle Input
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0:
                cmd = evt.action
                if cmd == "QUIT": return
                elif cmd == "RESET": 
                    env.reset_idx(torch.arange(env.num_envs, device=env.device))
                    print("\nRobot Reset!")
                elif cmd == "P_UP": tune_stiffness += 5.0
                elif cmd == "P_DOWN": tune_stiffness = max(0.0, tune_stiffness - 5.0)
                elif cmd == "D_UP": tune_damping += 0.5
                elif cmd == "D_DOWN": tune_damping = max(0.0, tune_damping - 0.5)

        # Apply new PD gains
        # Note: LeggedRobot usually reads p_gains from a tensor, we need to update it
        # The logic in _compute_torques uses self.p_gains
        env.p_gains[:] = tune_stiffness
        env.d_gains[:] = tune_damping

        # Set Actions: Target is strictly the default standing pose (action = 0)
        # We want to see if it holds the stand pose under gravity
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Step Physics
        env.step(actions)
        
        # Logging
        if time.time() - last_print > 0.5:
            # Calculate average hip height or just current params
            base_height = env.root_states[0, 2].item()
            print(f"\rStiffness (P): {tune_stiffness:5.1f} | Damping (D): {tune_damping:4.1f} | Height: {base_height:.3f}m", end="")
            last_print = time.time()

    print("\nExiting...")

if __name__ == '__main__':
    args = get_args()
    tune_pd(args)
