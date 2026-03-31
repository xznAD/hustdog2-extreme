import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import  get_args, task_registry
import torch
import numpy as np
import time

# --- Monkey Patch Start ---
# Fix missing attributes for "plane" terrain to prevent initialization crashes
_original_get_env_origins = LeggedRobot._get_env_origins

def _patched_get_env_origins(self):
    _original_get_env_origins(self)
    if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
        device = self.device
        num_envs = self.num_envs
        # Goals
        if not hasattr(self, 'cur_goal_idx') or self.cur_goal_idx is None:
             self.cur_goal_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        if not hasattr(self, 'reach_goal_timer') or self.reach_goal_timer is None:
             self.reach_goal_timer = torch.zeros(num_envs, dtype=torch.float, device=device)
        if not hasattr(self, 'cur_goals') or self.cur_goals is None:
            self.cur_goals = torch.zeros(num_envs, 2, device=device, requires_grad=False)
        if not hasattr(self, 'next_goals') or self.next_goals is None:
            self.next_goals = torch.zeros(num_envs, 2, device=device, requires_grad=False)
        if not hasattr(self, 'env_goals') or self.env_goals is None:
             self.env_goals = torch.zeros(num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=device, requires_grad=False)
        
        # Terrain info (used in some observations/logic)
        if not hasattr(self, 'env_class') or self.env_class is None:
            self.env_class = torch.zeros(num_envs, device=device, requires_grad=False)
        if not hasattr(self, 'terrain_levels') or self.terrain_levels is None:
            self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
        if not hasattr(self, 'terrain_types') or self.terrain_types is None:
            self.terrain_types = torch.zeros(num_envs, dtype=torch.long, device=device)
        if not hasattr(self, 'terrain_goals') or self.terrain_goals is None:
            self.terrain_goals = torch.zeros(num_envs, self.cfg.terrain.num_goals, 2, device=device, requires_grad=False)

LeggedRobot._get_env_origins = _patched_get_env_origins
# Also patch _draw_goals globally to avoid crash during __init__
LeggedRobot._draw_goals = lambda self: None
# --- Monkey Patch End ---

def debug_joints(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 1. Simplify Environment
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.fix_base_link = True # Fix robot base in air
    env_cfg.init_state.pos = [0.0, 0.0, 1.0] # 1m height
    
    # Force use of hustdog2 urdf file for debugging
    env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hustdog2/urdf/hustdog2.urdf'
    env_cfg.asset.collapse_fixed_joints = False # CRITICAL: hustdog2 has fixed feet, do not collapse them!
    env_cfg.asset.flip_visual_attachments = False # Isaac Gym defaults this to True, which rotates STLs by 90 deg. If your STLs are Z-up, set this to False.
    
    # 2. Disable Rewards to avoid dependencies on Terrain
    class EmptyScales: pass
    env_cfg.rewards.scales = EmptyScales()

    # 3. Create Env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # --- DEBUG PRINT START ---
    print("\n" + "="*30)
    print("DEBUG: Loaded Default Joint Angles:")
    for i, name in enumerate(env.dof_names):
        angle = env.default_dof_pos[0, i].item()
        print(f"  {name}: {angle:.4f} rad")
    print("="*30 + "\n")
    # --- DEBUG PRINT END ---

    # 4. Disable Internal Debug Viz (causes crashes with missing terrain)
    env.debug_viz = False
    # Also disable viewer sync specifically for debug viz parts inside post_physics_step
    # Note: env.debug_viz = False should be enough if checked in post_physics_step, 
    # but let's make sure by mocking _draw_goals if necessary.
    env._draw_goals = lambda: None
    
    if env.viewer is None:
        print("No viewer found. Use GUI mode.")
        return

    # 5. Setup Interactive Loop
    dof_names = env.dof_names
    default_angles = env.default_dof_pos[0, :].cpu().numpy()
    target_angles = default_angles.copy()
    num_dofs = len(dof_names)
    selected_dof_idx = 1
    action_scale = env.cfg.control.action_scale

    gym = env.gym
    viewer = env.viewer
    
    # Subscribe to keys
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "DOWN")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "PREV")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "NEXT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "RESET")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "QUIT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "CAM_FRONT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "CAM_SIDE")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "CAM_TOP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "CAM_ISO")

    print(f"\n{'='*50}")
    print("Debug Joints (Interactive)")
    print("Mouse: Move Camera")
    print("Keys:")
    print("  LEFT/RIGHT : Select Joint")
    print("  UP/DOWN    : +/- Angle (0.05 rad)")
    print("  R          : Reset Joint")
    print("  1-4        : Switch Camera View (Front, Side, Top, Iso)")
    print("  Q          : Quit")
    print(f"{'='*50}\n")
    
    last_update = 0
    
    while not gym.query_viewer_has_closed(viewer):
        # Handle input
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0: # Key Press
                cmd = evt.action
                if cmd == "QUIT": return
                elif cmd == "NEXT": selected_dof_idx = (selected_dof_idx + 1) % num_dofs
                elif cmd == "PREV": selected_dof_idx = (selected_dof_idx - 1) % num_dofs
                elif cmd == "UP": target_angles[selected_dof_idx] += 0.05
                elif cmd == "DOWN": target_angles[selected_dof_idx] -= 0.05
                elif cmd == "RESET": target_angles[selected_dof_idx] = default_angles[selected_dof_idx]
                elif cmd == "CAM_FRONT": env.set_camera([1.5, 0.0, 1.0], [0.0, 0.0, 0.5])
                elif cmd == "CAM_SIDE": env.set_camera([0.0, 1.5, 1.0], [0.0, 0.0, 0.5])
                elif cmd == "CAM_TOP": env.set_camera([0.1, 0.0, 2.5], [0.0, 0.0, 0.0]) # Slight offset to avoid gimbal lock
                elif cmd == "CAM_ISO": env.set_camera([1.2, 1.2, 1.0], [0.0, 0.0, 0.5])

        # Apply Actions
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        t_tensor = torch.from_numpy(target_angles).to(env.device).float()
        d_tensor = env.default_dof_pos[0, :]
        actions[0, :] = (t_tensor - d_tensor) / action_scale
        
        env.step(actions)
        
        # Render
        if hasattr(env, 'render'): env.render()
        else:
            gym.draw_viewer(viewer, env.sim, True)
            gym.sync_frame_time(env.sim)
            
        # Status Update
        if time.time() - last_update > 0.1:
            name = dof_names[selected_dof_idx]
            target = target_angles[selected_dof_idx]
            # Fetch actual current position from env
            current = env.dof_pos[0, selected_dof_idx].item()
            print(f"\rIdx: {selected_dof_idx:2d} | {name:20s} | Tgt: {target:6.3f} | Act: {current:6.3f} rad    ", end="")
            last_update = time.time()

    print("\nExiting...")

if __name__ == '__main__':
    args = get_args()
    debug_joints(args)
