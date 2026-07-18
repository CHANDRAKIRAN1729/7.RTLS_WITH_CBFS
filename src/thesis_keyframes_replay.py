#!/usr/bin/env python3
"""
Thesis Keyframe Replay — Visualize safe vs unsafe trajectories in RViz.

Loads pre-selected scenes, plans trajectories, and replays them with:
  - Full EE path visible from the first frame (LINE_STRIP marker)
  - Moving circle marker showing current robot position on path
  - Red marker when the robot is colliding with the obstacle
  - Green tick at goal for safe, red cross for unsafe

Prerequisites:
  1. Terminal 1:  roslaunch panda_moveit_config demo.launch
  2. Terminal 2:  python thesis_keyframes_replay.py

Uses the same parameters as run_simulation.sh (Phase 2 optimized).
"""

import argparse
import json
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.optim as optim

# ML imports
from vae import VAE
from vae_obs import VAEObstacleBCE
from robot_state_dataset import RobotStateDataset
from robot_obs_dataset import RobotObstacleDataset
from sim.panda import Panda
from sim.robot3d import Robo3D

# ROS imports
try:
    import rospy
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    ROS_AVAILABLE = True
except ImportError:
    print("ERROR: ROS not available. Source /opt/ros/noetic/setup.bash first.")
    sys.exit(1)

from simulate_in_moveit import (
    MoveItCollisionOracle,
    PANDA_ALL_JOINT_NAMES,
    GRIPPER_CLOSED,
    setup_logging,
)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', force=True)

# Scene type mapping: scenario_id → safe/unsafe
SAFE_SCENARIO_ID = 5
UNSAFE_SCENARIO_ID = 10


# =============================================================================
# Marker helpers
# =============================================================================
def publish_full_path(marker_pub, ee_positions, color, ns, marker_id=0,
                      line_width=0.015):
    """Publish the ENTIRE EE path as a persistent LINE_STRIP."""
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = line_width
    marker.color = ColorRGBA(*color)
    for pos in ee_positions:
        p = Point(x=pos[0], y=pos[1], z=pos[2])
        marker.points.append(p)
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)
    rospy.sleep(0.05)


def publish_current_marker(marker_pub, pos, color, ns="current_pos",
                            marker_id=50, size=0.03):
    """Publish a sphere at current EE position (replaces previous)."""
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]
    marker.pose.orientation.w = 1.0
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color = ColorRGBA(*color)
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)
    rospy.sleep(0.02)


def delete_marker(marker_pub, ns, marker_id):
    """Delete a specific marker."""
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.action = Marker.DELETE
    marker_pub.publish(marker)
    rospy.sleep(0.02)


def publish_goal_star(marker_pub, pos, ns="goal_star", marker_id=98,
                      size=0.05):
    """Publish a yellow sphere marker at the goal position."""
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]
    marker.pose.orientation.w = 1.0
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color = ColorRGBA(1.0, 0.9, 0.0, 1.0)  # yellow
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)
    rospy.sleep(0.05)


def publish_mid_result_marker(marker_pub, pos, is_safe, ns="mid_result",
                              marker_id=99):
    """
    Publish a green tick (✓) or red cross (✗) at mid-keyframe position.
    - Safe: green tick where robot avoids obstacle
    - Unsafe: red cross where robot collides with obstacle
    """
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2] + 0.30
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.10
    if is_safe:
        marker.text = "✓ SAFE"
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
    else:
        marker.text = "✗ COLLISION"
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)
    rospy.sleep(0.05)


def publish_collision_flash(marker_pub, pos, ns="collision_flash",
                             marker_id=80, size=0.05):
    """Publish a red semi-transparent sphere to indicate collision zone."""
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]
    marker.pose.orientation.w = 1.0
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.5)
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)
    rospy.sleep(0.02)


def clear_all_markers(marker_pub):
    """Clear all visualization markers."""
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.action = Marker.DELETEALL
    marker_pub.publish(marker)
    rospy.sleep(0.2)


# =============================================================================
# Plan trajectory (same as simulate_in_moveit.py)
# =============================================================================
def plan_trajectory(model, model_obs, robot, mean_train, std_train,
                    mean_obs, std_obs, q_start_np, e_start_np, e_target_np,
                    obstacles, device, args):
    """
    Plan a trajectory using the baseline planner (same params as run_simulation.sh).
    Returns: list of (q_rad, ee_pos) at each step.
    """
    mean_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train, dtype=torch.float32).to(device)
    mean_obs_t = torch.tensor(mean_obs, dtype=torch.float32).to(device)
    std_obs_t = torch.tensor(std_obs, dtype=torch.float32).to(device)

    q_start = torch.tensor([q_start_np], dtype=torch.float32).to(device)
    e_start = torch.tensor([e_start_np], dtype=torch.float32).to(device)
    e_target = torch.tensor([e_target_np], dtype=torch.float32).to(device)

    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_t[:, :10]) / std_t[:, :10]

    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=args.planning_lr)

    # Normalize obstacles for classifier
    obs_tensors = []
    for obs in obstacles:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        obs_normalized = (obs_tensor - mean_obs_t) / std_obs_t
        obs_tensors.append(obs_normalized.unsqueeze(0))

    # GECO state
    lambda_prior = args.lambda_prior
    lambda_collision = args.lambda_collision
    C_prior_ma = None
    C_collision_ma = None

    waypoints = []  # list of {'q': np.array, 'ee': np.array}

    # Save z_init for interpolation
    z_start = z_init.detach()

    for step in range(args.max_steps):
        optimizer.zero_grad()

        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_t[:, :10] + mean_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)

        L_collision = torch.tensor(0.0, device=device)
        if model_obs is not None and len(obstacles) > 0:
            for obs_tensor in obs_tensors:
                logit = model_obs.obstacle_collision_classifier(z, obs_tensor)
                p_collision = torch.sigmoid(logit / args.temperature)
                L_collision = L_collision + (-torch.log(1 - p_collision + 1e-8))
            L_collision = L_collision / len(obstacles)

        # GECO
        if args.use_geco:
            C_prior = L_prior.item() - args.tau_prior_goal
            C_collision = L_collision.item() - args.tau_obs_goal
            if step == 0:
                C_prior_ma = C_prior
                C_collision_ma = C_collision
            else:
                C_prior_ma = args.alpha_ma_prior * C_prior_ma + (1 - args.alpha_ma_prior) * C_prior
                C_collision_ma = args.alpha_ma_obs * C_collision_ma + (1 - args.alpha_ma_obs) * C_collision
            kappa_prior = np.exp(args.alpha_geco * C_prior_ma)
            kappa_collision = np.exp(args.alpha_geco * C_collision_ma)
            lambda_prior = np.clip(kappa_prior * lambda_prior, 1e-6, 1000.0)
            lambda_collision = np.clip(kappa_collision * lambda_collision, 1e-6, 1000.0)

        L_total = L_goal + lambda_prior * L_prior + lambda_collision * L_collision

        if L_goal.item() < args.success_threshold:
            break

        L_total.backward()
        optimizer.step()

    # z_goal after optimization
    z_goal = z.detach()

    # ---- Smooth interpolation in latent space (same as simulate_in_moveit.py) ----
    # Instead of raw optimization zigzag, linearly interpolate z_start → z_goal
    # and decode each point. This is what run_simulation.sh shows in RViz.
    num_interp = 50
    waypoints = []
    with torch.no_grad():
        for i in range(num_interp + 1):
            alpha = i / num_interp
            z_interp = (1 - alpha) * z_start + alpha * z_goal
            x_decoded_norm = model.decoder(z_interp)
            x_decoded = x_decoded_norm * std_t[:, :10] + mean_t[:, :10]
            q_interp = x_decoded[:, :7].cpu().numpy()[0]
            ee_interp = x_decoded[:, 7:10].cpu().numpy()[0]
            waypoints.append({'q': q_interp, 'ee': ee_interp})

    return waypoints


# =============================================================================
# Replay a trajectory in RViz with path markers
# =============================================================================
def replay_trajectory(oracle, marker_pub, waypoints, obstacles, robo3d,
                      is_safe, step_duration=0.08, pause_at_keyframes=5.0):
    """
    Replay a planned trajectory in RViz with full path visibility.

    Args:
        oracle: MoveItCollisionOracle
        marker_pub: ROS marker publisher
        waypoints: list of {'q': np.array, 'ee': np.array}
        obstacles: list of [x,y,h,r]
        robo3d: Robo3D for collision checking
        is_safe: True for safe scenario, False for unsafe
        step_duration: seconds per waypoint
        pause_at_keyframes: seconds to pause at start, mid, end
    """
    n = len(waypoints)
    obstacles_xyhr = [obs for obs in obstacles]

    # Color scheme
    if is_safe:
        path_color = (0.0, 0.85, 0.2, 0.9)   # green
        marker_color = (0.0, 1.0, 0.3, 1.0)   # bright green
        label = "SAFE"
        ns_prefix = "safe"
    else:
        path_color = (0.9, 0.15, 0.0, 0.9)    # red
        marker_color = (1.0, 0.3, 0.0, 1.0)   # orange-red
        label = "UNSAFE"
        ns_prefix = "unsafe"

    collision_color = (1.0, 0.0, 0.0, 1.0)  # pure red for collision

    # Extract all EE positions for the full path
    all_ee = [wp['ee'] for wp in waypoints]

    # --- 1. Publish FULL path from the start ---
    logging.info(f"Publishing full {label} path ({n} waypoints)...")
    publish_full_path(marker_pub, all_ee, path_color,
                      ns=f"{ns_prefix}_path", marker_id=0, line_width=0.015)

    # Check collision for each waypoint
    collision_flags = []
    for wp in waypoints:
        q_deg = np.degrees(wp['q']).tolist()
        collides = robo3d.check_for_collision(q_deg, obstacles_xyhr)
        collision_flags.append(collides)

    n_collisions = sum(collision_flags)
    logging.info(f"  Collision waypoints: {n_collisions}/{n}")

    # Keyframe indices: start, mid, end
    keyframe_indices = [0, n // 2, n - 1]
    keyframe_names = ['Start', 'Mid-trajectory', 'Goal']

    # --- 2. Animate through trajectory ---
    for i, wp in enumerate(waypoints):
        # Move robot to this configuration
        oracle.publish_joint_state(wp['q'], duration=0.01)

        # Publish current position marker
        if collision_flags[i]:
            # RED marker when colliding
            publish_current_marker(marker_pub, wp['ee'], collision_color,
                                    ns=f"{ns_prefix}_current", marker_id=50,
                                    size=0.035)
            # Collision flash effect
            publish_collision_flash(marker_pub, wp['ee'],
                                    ns=f"{ns_prefix}_flash", marker_id=80)
        else:
            # Normal color marker
            publish_current_marker(marker_pub, wp['ee'], marker_color,
                                    ns=f"{ns_prefix}_current", marker_id=50,
                                    size=0.025)
            # Remove collision flash if it was shown
            delete_marker(marker_pub, f"{ns_prefix}_flash", 80)

        # Pause at keyframes for screenshot
        if i in keyframe_indices:
            ki = keyframe_indices.index(i)
            logging.info(f"\n  >>> KEYFRAME {ki+1}/3: {keyframe_names[ki]} "
                         f"(step {i}/{n}) <<<")
            logging.info(f"  EE: ({wp['ee'][0]:.3f}, {wp['ee'][1]:.3f}, "
                         f"{wp['ee'][2]:.3f})")
            logging.info(f"  Collision: {'YES' if collision_flags[i] else 'NO'}")

            # Mid-keyframe (ki==1): show green tick (safe) or red cross (unsafe)
            if ki == 1:
                publish_mid_result_marker(
                    marker_pub, wp['ee'], is_safe,
                    ns=f"{ns_prefix}_mid_result", marker_id=99)

            logging.info(f"  >>> TAKE SCREENSHOT NOW! "
                         f"Pausing {pause_at_keyframes}s <<<")
            logging.info(f"  Save as: keyframe_{ns_prefix}_{ki+1}.png")
            rospy.sleep(pause_at_keyframes)

            # Remove mid-result marker after screenshot
            if ki == 1:
                delete_marker(marker_pub, f"{ns_prefix}_mid_result", 99)
        else:
            rospy.sleep(step_duration)

    # --- 3. Remove the current position marker (final state) ---
    delete_marker(marker_pub, f"{ns_prefix}_current", 50)
    delete_marker(marker_pub, f"{ns_prefix}_flash", 80)

    logging.info(f"\n  {label} trajectory complete!")
    logging.info(f"  Total waypoints: {n}, Collisions: {n_collisions}")
    logging.info(f"  Final hold — take last screenshot if needed")
    rospy.sleep(pause_at_keyframes)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Thesis keyframe replay in RViz')

    parser.add_argument('--scenes_file', type=str,
                        default='../model_params/panda_10k/pick_scenes.json')
    parser.add_argument('--data_path', type=str, default='../data')

    # Planning params (same as run_simulation.sh Phase 2)
    parser.add_argument('--planning_lr', type=float, default=0.15)
    parser.add_argument('--lambda_prior', type=float, default=0.7)
    parser.add_argument('--lambda_collision', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--success_threshold', type=float, default=0.01)
    parser.add_argument('--use_geco', action='store_true', default=True)
    parser.add_argument('--alpha_geco', type=float, default=0.008)
    parser.add_argument('--tau_prior_goal', type=float, default=6.0)
    parser.add_argument('--tau_obs_goal', type=float, default=2.0)
    parser.add_argument('--alpha_ma_prior', type=float, default=0.8)
    parser.add_argument('--alpha_ma_obs', type=float, default=0.8)

    # Replay params
    parser.add_argument('--step_duration', type=float, default=0.08,
                        help='Seconds per waypoint during animation')
    parser.add_argument('--pause_time', type=float, default=8.0,
                        help='Seconds to pause at each keyframe')
    parser.add_argument('--safe_id', type=int, default=SAFE_SCENARIO_ID,
                        help='Scenario ID that is safe')
    parser.add_argument('--unsafe_id', type=int, default=UNSAFE_SCENARIO_ID,
                        help='Scenario ID that is unsafe')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load scene data
    with open(args.scenes_file, 'r') as f:
        scene_data = json.load(f)

    scenarios = scene_data['scenarios']
    logging.info(f"Loaded {len(scenarios)} scenarios from {args.scenes_file}")

    # Find safe and unsafe scenes
    safe_scene = None
    unsafe_scene = None
    for s in scenarios:
        if s['scenario_id'] == args.safe_id:
            safe_scene = s
        if s['scenario_id'] == args.unsafe_id:
            unsafe_scene = s

    if safe_scene is None:
        logging.error(f"Safe scenario (id={args.safe_id}) not found!")
        return
    if unsafe_scene is None:
        logging.error(f"Unsafe scenario (id={args.unsafe_id}) not found!")
        return

    logging.info(f"Safe scene: scenario_id={safe_scene['scenario_id']}")
    logging.info(f"Unsafe scene: scenario_id={unsafe_scene['scenario_id']}")

    # Load model config
    import cbf_config as cfg
    with open(cfg.VAE_CONFIG, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']

    # Normalization
    dataset = RobotStateDataset(args.data_path, train=0,
                                 train_data_name='free_space_100k_train.dat')
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()

    obs_dataset = RobotObstacleDataset(
        args.data_path, train=0,
        train_data_name='collision_100k_train.dat',
        test_data_name='collision_10k_test.dat',
        free_space_train_name='free_space_100k_train.dat',
        free_space_test_name='free_space_10k_test.dat')
    mean_obs = obs_dataset.get_mean_train()[0, 10:14]
    std_obs = obs_dataset.get_std_train()[0, 10:14]

    # Robot
    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    # Load models
    model = VAE(config['input_dim'], config['latent_dim'],
                config['units_per_layer'], config['num_hidden_layers'])
    ckpt = torch.load(cfg.VAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    model_obs = VAEObstacleBCE(config['input_dim'], config['latent_dim'],
                                config['units_per_layer'], config['num_hidden_layers'])
    ckpt_obs = torch.load(cfg.CLASSIFIER_CHECKPOINT, map_location=device,
                          weights_only=False)
    model_obs.load_state_dict(ckpt_obs['model_state_dict'])
    model_obs.to(device).eval()

    logging.info("Models loaded successfully")

    # =========================================================================
    # Initialize ROS + MoveIt
    # =========================================================================
    rospy.init_node('thesis_keyframe_replay', anonymous=True, disable_signals=True)
    setup_logging()

    oracle = MoveItCollisionOracle(group_name='panda_arm',
                                    collision_padding=0.001)
    marker_pub = oracle.marker_pub

    logging.info("\n" + "=" * 60)
    logging.info("THESIS KEYFRAME REPLAY — Safe vs Unsafe")
    logging.info("=" * 60)

    # =========================================================================
    # Plan and replay: SAFE trajectory (scenario 5)
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info(f"PLANNING SAFE TRAJECTORY (scenario {args.safe_id})")
    logging.info("=" * 60)

    safe_waypoints = plan_trajectory(
        model, model_obs, robot, mean_train, std_train, mean_obs, std_obs,
        safe_scene['q_start'], safe_scene['e_start'],
        safe_scene['e_target'], safe_scene['obstacles'],
        device, args)
    logging.info(f"Safe path: {len(safe_waypoints)} waypoints")

    # Setup scene in RViz
    clear_all_markers(marker_pub)
    oracle.clear_all_obstacles()
    oracle.clear_trajectory_display()
    oracle.add_table(height=0.0)
    oracle.add_obstacles_from_array(safe_scene['obstacles'])
    rospy.sleep(0.5)

    # Start/goal markers
    e_s = safe_scene['e_start']
    e_t = safe_scene['e_target']
    oracle.publish_start_marker(e_s[0], e_s[1], e_s[2])
    publish_goal_star(marker_pub, [e_t[0], e_t[1], e_t[2]])

    # Move to start position
    oracle.publish_joint_state(np.array(safe_scene['q_start']), duration=0.1)
    rospy.sleep(1.0)

    logging.info("Starting safe trajectory replay...")
    replay_trajectory(oracle, marker_pub, safe_waypoints,
                      safe_scene['obstacles'], robo3d,
                      is_safe=True, step_duration=args.step_duration,
                      pause_at_keyframes=args.pause_time)

    # =========================================================================
    # Plan and replay: UNSAFE trajectory (scenario 10)
    # =========================================================================
    input("\n>>> Press ENTER to continue to UNSAFE trajectory... <<<\n")

    logging.info("\n" + "=" * 60)
    logging.info(f"PLANNING UNSAFE TRAJECTORY (scenario {args.unsafe_id})")
    logging.info("=" * 60)

    unsafe_waypoints = plan_trajectory(
        model, model_obs, robot, mean_train, std_train, mean_obs, std_obs,
        unsafe_scene['q_start'], unsafe_scene['e_start'],
        unsafe_scene['e_target'], unsafe_scene['obstacles'],
        device, args)
    logging.info(f"Unsafe path: {len(unsafe_waypoints)} waypoints")

    # Setup new scene
    clear_all_markers(marker_pub)
    oracle.clear_all_obstacles()
    oracle.clear_trajectory_display()
    oracle.add_table(height=0.0)
    oracle.add_obstacles_from_array(unsafe_scene['obstacles'])
    rospy.sleep(0.5)

    e_s = unsafe_scene['e_start']
    e_t = unsafe_scene['e_target']
    oracle.publish_start_marker(e_s[0], e_s[1], e_s[2])
    publish_goal_star(marker_pub, [e_t[0], e_t[1], e_t[2]])

    oracle.publish_joint_state(np.array(unsafe_scene['q_start']), duration=0.1)
    rospy.sleep(1.0)

    logging.info("Starting unsafe trajectory replay...")
    replay_trajectory(oracle, marker_pub, unsafe_waypoints,
                      unsafe_scene['obstacles'], robo3d,
                      is_safe=False, step_duration=args.step_duration,
                      pause_at_keyframes=args.pause_time)

    # =========================================================================
    # Done
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("KEYFRAME CAPTURE COMPLETE")
    logging.info("=" * 60)
    logging.info("Screenshots needed:")
    logging.info("  Safe:   keyframe_safe_1.png (start)")
    logging.info("          keyframe_safe_2.png (mid)")
    logging.info("          keyframe_safe_3.png (goal)")
    logging.info("  Unsafe: keyframe_unsafe_1.png (start)")
    logging.info("          keyframe_unsafe_2.png (mid)")
    logging.info("          keyframe_unsafe_3.png (goal)")
    logging.info("")
    logging.info("Arrange as 2×3 grid in Google Draw:")
    logging.info("  Row 1: Safe    — green checkmarks ✓")
    logging.info("  Row 2: Unsafe  — red crosses ✗")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
