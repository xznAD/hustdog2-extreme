"""
测试单个关节控制
让机器人保持默认姿态，只对指定关节施加动作
"""
import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import  get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import torch

def test_single_joint():
    args = get_args()
    
    # 强制使用hustdog2任务
    args.task = 'hustdog2'
    args.headless = False  # 开启渲染
    args.num_envs = 1  # 只用一个环境方便观察
    
    # 强制使用CPU运行，避免CUDA错误
    args.device = 'cpu'
    args.sim_device = 'cpu'
    args.use_gpu = False
    args.use_gpu_pipeline = False
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 准备环境
    env_cfg.env.num_envs = args.num_envs
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.terrain_length = 4.0
    env_cfg.terrain.terrain_width = 4.0
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # 固定基座，让机器人悬浮在空中
    env_cfg.asset.fix_base_link = True
    
    # 关闭重力
    env_cfg.sim.disable_gravity = True
    
    # 初始位置设置在原点，便于相机找到
    env_cfg.init_state.pos = [0.0, 0.0, 0.8]  # 0.8米高
    
    # 不设置相机位置，环境创建后根据实际位置设置
    # env_cfg.viewer.pos = [3, 0, 1.0]
    # env_cfg.viewer.lookat = [0, 0, 0.8]
    
    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 重置环境获取初始观测
    obs = env.reset()
    
    print("=" * 60)
    print("环境信息:")
    print(f"  动作维度: {env.num_actions}")
    print(f"  观测维度: {env.num_obs}")
    print(f"  环境数量: {env.num_envs}")
    
    # 打印机器人初始位置
    print(f"\n机器人初始位置:")
    for i in range(env.num_envs):
        pos = env.root_states[i, :3].cpu().numpy()
        print(f"  机器人 {i}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
    print(f"  fix_base_link: {env_cfg.asset.fix_base_link}")
    print(f"  disable_gravity: {env_cfg.sim.disable_gravity}")
    print(f"  关节名称: {env.dof_names}")
    print("=" * 60)
    
    # 找到FR_hip_joint的索引
    try:
        fr_hip_idx = env.dof_names.index('FR_hip_joint')
        print(f"\nFR_hip_joint 在动作空间中的索引: {fr_hip_idx}")
    except ValueError:
        print("\n错误: 找不到 FR_hip_joint!")
        print(f"可用的关节名称: {env.dof_names}")
        return
    
    # 创建全零动作(保持default_joint_angles)
    actions = torch.zeros(args.num_envs, env.num_actions, device=env.device)
    
    # 只施加+1的动作
    actions[:, fr_hip_idx] = 0.5
    #小腿加为伸直，减为抬起；大腿加为往后，减为往前；髋关节加为往外，减为往内。
    
    # 设置相机位置根据实际机器人位置
    if hasattr(env, 'set_camera'):
        robot_pos = env.root_states[0, :3].cpu().numpy()
        cam_pos = robot_pos + [3, 0, 0.5]  # 相机在机器人侧面3米处
        env.set_camera(cam_pos, robot_pos)
        print(f"\n相机设置:")
        print(f"  相机位置: {cam_pos}")
        print(f"  看向: {robot_pos}")
    
    print(f"\n动作向量:")
    print(f"  所有关节初始动作: {actions[0].cpu().numpy()}")
    print(f"  FR_hip_joint (索引{fr_hip_idx}): +1")
    print(f"  其他关节: 0.0 (保持默认角度)")
    print(f"\n提示: 在渲染窗口中，可以使用鼠标拖拽旋转视角，滚轮缩放")
    
    print("\n" + "=" * 60)
    print("应用静态姿态...")
    print("按 Ctrl+C 停止观察")
    print("=" * 60)
    
    # 先执行几步让关节到达目标位置
    for i in range(50):
        obs, _, rewards, dones, infos, _, _ = env.step(actions)
    
    # 显示当前姿态信息
    current_dof_pos = env.dof_pos[0].cpu().numpy()
    print(f"\n当前姿态:")
    print(f"  基座高度: {env.root_states[0, 2].cpu().item():.4f} m")
    print(f"  基座位置: x={env.root_states[0, 0].cpu().item():.4f}, y={env.root_states[0, 1].cpu().item():.4f}, z={env.root_states[0, 2].cpu().item():.4f}")
    print(f"\n  所有关节角度:")
    for j, name in enumerate(env.dof_names):
        default_angle = env.default_dof_pos[0, j].cpu().item()
        current_angle = current_dof_pos[j]
        action = actions[0, j].cpu().item()
        if abs(action) > 0.01:
            print(f"    {name:20s}: {current_angle:7.4f} rad (默认: {default_angle:7.4f}, 动作: +{action:.2f}) <-- 目标关节")
        else:
            print(f"    {name:20s}: {current_angle:7.4f} rad (默认: {default_angle:7.4f})")
    
    # 保持静态姿态，持续渲染
    print("\n机器人将保持当前姿态...")
    try:
        while True:
            obs, _, rewards, dones, infos, _, _ = env.step(actions)
    except KeyboardInterrupt:
        print("\n用户中断")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == '__main__':
    test_single_joint()
