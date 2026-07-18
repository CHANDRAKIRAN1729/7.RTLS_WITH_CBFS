#!/usr/bin/env python3
"""
Thesis Keyframe Capture — Safe vs Unsafe Trajectory in RViz.

Captures 6 keyframe screenshots from RViz showing:
  Row 1: Safe trajectory (classifier-based) — 3 keyframes
  Row 2: Unsafe trajectory (nominal only) — 3 keyframes

Each frame shows:
  - The Panda arm at the current configuration
  - The full EE path drawn as a colored LINE_STRIP marker
  - The obstacle cylinder
  - Start (blue) and Goal (green) markers

Prerequisites:
  1. Terminal 1:  roslaunch panda_moveit_config demo.launch
  2. Terminal 2:  python thesis_keyframes_rviz.py

The script pauses at each keyframe so you can take a screenshot from RViz
(File → Save Screenshot, or use the screenshot tool).

Author: RTLS Project
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

# ROS imports
try:
    import rospy
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import Header, ColorRGBA
    ROS_AVAILABLE = True
except ImportError:
    print("ERROR: ROS not available. Source your ROS workspace first.")
    print("  source /opt/ros/noetic/setup.bash")
    sys.exit(1)

# Import MoveIt oracle from existing simulation script
from simulate_in_moveit import (
    MoveItCollisionOracle,
    PANDA_ALL_JOINT_NAMES,
    GRIPPER_CLOSED,
    setup_logging,
)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thesis_figures')


# =============================================================================
# Path Marker Publisher
# =============================================================================
class PathMarkerPublisher:
    """Publishes EE path as a LINE_STRIP marker in RViz."""

    def __init__(self, marker_pub):
        self.marker_pub = marker_pub

    def publish_path(self, ee_positions, color_rgba, ns="ee_path",
                     marker_id=0, line_width=0.008):
        """
        Publish end-effector path as a colored line in RViz.

        Args:
            ee_positions: list of (x, y, z) tuples — full EE path
            color_rgba: (r, g, b, a) tuple
            ns: namespace for the marker
            marker_id: unique marker ID
            line_width: line thickness in meters
        """
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = line_width  # line width

        marker.color.r = color_rgba[0]
        marker.color.g = color_rgba[1]
        marker.color.b = color_rgba[2]
        marker.color.a = color_rgba[3]

        for pos in ee_positions:
            p = Point()
            p.x, p.y, p.z = pos[0], pos[1], pos[2]
            marker.points.append(p)

        marker.lifetime = rospy.Duration(0)  # persistent
        self.marker_pub.publish(marker)
        rospy.sleep(0.1)

    def publish_current_position_marker(self, pos, color_rgba, ns="current_ee",
                                         marker_id=100, size=0.025):
        """Publish a sphere at current EE position."""
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

        marker.color.r = color_rgba[0]
        marker.color.g = color_rgba[1]
        marker.color.b = color_rgba[2]
        marker.color.a = color_rgba[3]

        marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(marker)
        rospy.sleep(0.05)

    def publish_label(self, text, pos, color_rgba, ns="label",
                      marker_id=200, size=0.04):
        """Publish a text label in RViz."""
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.w = 1.0

        marker.scale.z = size  # text height
        marker.color.r = color_rgba[0]
        marker.color.g = color_rgba[1]
        marker.color.b = color_rgba[2]
        marker.color.a = color_rgba[3]

        marker.text = text
        marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(marker)
        rospy.sleep(0.05)


# =============================================================================
# Planning Functions
# =============================================================================
def plan_trajectory(model, model_obs, robot, mean_train, std_train,
                    mean_obs, std_obs, q_start, e_start, e_target,
                    obstacles, device, use_classifier=True,
                    planning_lr=0.15, lambda_prior=0.7,
                    lambda_collision=0.5, temperature=3.0,
                    max_steps=300, success_threshold=0.01):
    """
    Plan a trajectory and return joint angles + EE positions at every step.

    Args:
        use_classifier: If True, uses collision avoidance (safe).
                       If False, uses only Goal+Prior (unsafe nominal).
    """
    mean_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train, dtype=torch.float32).to(device)
    mean_obs_t = torch.tensor(mean_obs, dtype=torch.float32).to(device)
    std_obs_t = torch.tensor(std_obs, dtype=torch.float32).to(device)

    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_t[:, :10]) / std_t[:, :10]

    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=planning_lr)

    # Normalize obstacles for classifier
    obs_tensors = []
    for obs in obstacles:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        obs_normalized = (obs_tensor - mean_obs_t) / std_obs_t
        obs_tensors.append(obs_normalized.unsqueeze(0))

    trajectory_q = []  # joint angles (radians)
    trajectory_ee = []  # EE positions (x,y,z)

    for step in range(max_steps):
        optimizer.zero_grad()

        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_t[:, :10] + mean_t[:, :10]
        q_decoded = x_decoded[:, :7]
        e_decoded = x_decoded[:, 7:10]

        # Goal loss
        L_goal = torch.norm(e_decoded - e_target)

        # Prior loss
        L_prior = 0.5 * torch.sum(z ** 2)

        # Collision loss (only if classifier enabled)
        L_collision = torch.tensor(0.0, device=device)
        if use_classifier and model_obs is not None and len(obstacles) > 0:
            for obs_tensor in obs_tensors:
                logit = model_obs.obstacle_collision_classifier(z, obs_tensor)
                p_collision = torch.sigmoid(logit / temperature)
                L_collision = L_collision + (-torch.log(1 - p_collision + 1e-8))
            L_collision = L_collision / len(obstacles)

        # Total loss
        if use_classifier:
            L_total = L_goal + lambda_prior * L_prior + lambda_collision * L_collision
        else:
            L_total = L_goal + lambda_prior * L_prior

        # Record
        with torch.no_grad():
            trajectory_q.append(q_decoded.cpu().numpy()[0].copy())
            trajectory_ee.append(e_decoded.cpu().numpy()[0].copy())

        # Goal reached?
        if L_goal.item() < success_threshold:
            break

        L_total.backward()
        optimizer.step()

    # Final state
    with torch.no_grad():
        x_final_norm = model.decoder(z)
        x_final = x_final_norm * std_t[:, :10] + mean_t[:, :10]
        trajectory_q.append(x_final[:, :7].cpu().numpy()[0])
        trajectory_ee.append(x_final[:, 7:10].cpu().numpy()[0])

    return trajectory_q, trajectory_ee


# =============================================================================
# Find a scenario where nominal collides but classifier avoids
# =============================================================================
def find_good_scenario(model, model_obs, robot, mean_train, std_train,
                       mean_obs, std_obs, device, robo3d, max_attempts=100):
    """
    Search for a scenario where:
      - Nominal (no classifier) path COLLIDES with obstacle
      - Classifier path AVOIDS the obstacle

    Returns: (q_start, e_start, e_target, obstacles) or None
    """
    from sim.robot3d import Robo3D

    q_min = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    for attempt in range(max_attempts):
        # Random start and goal
        q_start = torch.rand(1, 7, device=device) * (q_max - q_min) + q_min
        e_start = robot.FK(q_start.clone(), device, rad=True)
        q_target = torch.rand(1, 7, device=device) * (q_max - q_min) + q_min
        e_target = robot.FK(q_target.clone(), device, rad=True)

        # Place obstacle between start and goal EE positions
        e_s = e_start.cpu().numpy()[0]
        e_t = e_target.cpu().numpy()[0]
        t = np.random.uniform(0.3, 0.7)
        obs_x = e_s[0] * (1-t) + e_t[0] * t + np.random.uniform(-0.05, 0.05)
        obs_y = e_s[1] * (1-t) + e_t[1] * t + np.random.uniform(-0.05, 0.05)
        obs_h = np.random.uniform(0.5, 1.0)
        obs_r = np.random.uniform(0.06, 0.12)

        # Validate obstacle position
        if obs_x < 0.2 or obs_x > 0.9:
            continue

        obstacles = [[obs_x, obs_y, obs_h, obs_r]]

        # Plan UNSAFE (nominal only)
        traj_q_unsafe, traj_ee_unsafe = plan_trajectory(
            model, model_obs, robot, mean_train, std_train,
            mean_obs, std_obs, q_start, e_start, e_target, obstacles,
            device, use_classifier=False,
            planning_lr=0.15, lambda_prior=0.01, max_steps=300)

        # Check if unsafe path actually collides
        unsafe_collides = False
        obstacles_xyhr = [obstacles[0]]
        for q_rad in traj_q_unsafe:
            q_deg = np.degrees(q_rad).tolist()
            if robo3d.check_for_collision(q_deg, obstacles_xyhr):
                unsafe_collides = True
                break

        if not unsafe_collides:
            continue

        # Plan SAFE (with classifier)
        traj_q_safe, traj_ee_safe = plan_trajectory(
            model, model_obs, robot, mean_train, std_train,
            mean_obs, std_obs, q_start, e_start, e_target, obstacles,
            device, use_classifier=True,
            planning_lr=0.15, lambda_prior=0.01,
            lambda_collision=5.0, temperature=3.0, max_steps=300)

        # Check if safe path is actually collision-free
        safe_collides = False
        for q_rad in traj_q_safe:
            q_deg = np.degrees(q_rad).tolist()
            if robo3d.check_for_collision(q_deg, obstacles_xyhr):
                safe_collides = True
                break

        if not safe_collides:
            logging.info(f"Found good scenario at attempt {attempt+1}!")
            logging.info(f"  Obstacle: x={obs_x:.3f}, y={obs_y:.3f}, h={obs_h:.3f}, r={obs_r:.3f}")
            logging.info(f"  Unsafe path: {len(traj_q_unsafe)} steps, COLLIDES ✗")
            logging.info(f"  Safe path: {len(traj_q_safe)} steps, AVOIDS ✓")
            return (q_start, e_start, e_target, obstacles,
                    traj_q_unsafe, traj_ee_unsafe,
                    traj_q_safe, traj_ee_safe)

    return None


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Thesis keyframe capture in RViz')

    # Model paths
    import cbf_config as cfg
    parser.add_argument('--checkpoint', type=str,
                        default=cfg.VAE_CHECKPOINT)
    parser.add_argument('--checkpoint_obs', type=str,
                        default=cfg.CLASSIFIER_CHECKPOINT)
    parser.add_argument('--config', type=str,
                        default=cfg.VAE_CONFIG)
    parser.add_argument('--data_path', type=str, default='../data')

    # Scenario
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_search_attempts', type=int, default=200,
                        help='Max attempts to find safe/unsafe scenario pair')
    parser.add_argument('--pause_time', type=float, default=8.0,
                        help='Seconds to pause at each keyframe for screenshot')

    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']

    # Normalization stats
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
    from sim.robot3d import Robo3D
    robo3d = Robo3D(Panda())

    # Load VAE
    model = VAE(config['input_dim'], config['latent_dim'],
                config['units_per_layer'], config['num_hidden_layers'])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    logging.info(f"VAE loaded from {args.checkpoint}")

    # Load classifier
    model_obs = VAEObstacleBCE(config['input_dim'], config['latent_dim'],
                                config['units_per_layer'], config['num_hidden_layers'])
    ckpt_obs = torch.load(args.checkpoint_obs, map_location=device, weights_only=False)
    model_obs.load_state_dict(ckpt_obs['model_state_dict'])
    model_obs.to(device).eval()
    logging.info(f"Classifier loaded from {args.checkpoint_obs}")

    # =========================================================================
    # Initialize ROS and MoveIt
    # =========================================================================
    rospy.init_node('thesis_keyframe_capture', anonymous=True, disable_signals=True)
    setup_logging()

    collision_oracle = MoveItCollisionOracle(
        group_name='panda_arm',
        collision_padding=0.005
    )
    path_publisher = PathMarkerPublisher(collision_oracle.marker_pub)

    logging.info("\n" + "=" * 60)
    logging.info("THESIS KEYFRAME CAPTURE")
    logging.info("=" * 60)

    # =========================================================================
    # Find a scenario where nominal collides, classifier avoids
    # =========================================================================
    logging.info("\nSearching for a good safe/unsafe scenario pair...")
    result = find_good_scenario(
        model, model_obs, robot, mean_train, std_train,
        mean_obs, std_obs, device, robo3d,
        max_attempts=args.max_search_attempts)

    if result is None:
        logging.error("Could not find a suitable scenario! Try different seed.")
        return

    (q_start, e_start, e_target, obstacles,
     traj_q_unsafe, traj_ee_unsafe,
     traj_q_safe, traj_ee_safe) = result

    e_s = e_start.cpu().numpy()[0]
    e_t = e_target.cpu().numpy()[0]

    # Pick 3 keyframe indices (start, mid, end)
    def get_keyframe_indices(trajectory):
        n = len(trajectory)
        return [0, n // 2, n - 1]

    safe_indices = get_keyframe_indices(traj_q_safe)
    unsafe_indices = get_keyframe_indices(traj_q_unsafe)

    # =========================================================================
    # Display keyframes: SAFE trajectory (green path)
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("SAFE TRAJECTORY KEYFRAMES (with collision avoidance)")
    logging.info("=" * 60)

    frame_names = ['Start', 'Mid-trajectory', 'Goal reached']

    for ki, idx in enumerate(safe_indices):
        logging.info(f"\n--- Safe Keyframe {ki+1}/3: {frame_names[ki]} (step {idx}) ---")

        # Clear previous visualization
        collision_oracle.clear_markers()
        collision_oracle.clear_trajectory_display()

        # Setup scene
        collision_oracle.clear_all_obstacles()
        collision_oracle.add_table(height=0.0)
        collision_oracle.add_obstacles_from_array(obstacles)
        rospy.sleep(0.5)

        # Publish start and goal markers
        collision_oracle.publish_start_marker(e_s[0], e_s[1], e_s[2])
        collision_oracle.publish_goal_marker(e_t[0], e_t[1], e_t[2])

        # Publish FULL path as green line (up to current step)
        ee_path_so_far = traj_ee_safe[:idx+1]
        path_publisher.publish_path(
            ee_path_so_far,
            color_rgba=(0.0, 0.9, 0.2, 0.9),  # green
            ns="safe_path", marker_id=0, line_width=0.006)

        # Also publish full path as faint line (shows planned trajectory)
        path_publisher.publish_path(
            traj_ee_safe,
            color_rgba=(0.0, 0.6, 0.1, 0.3),  # faint green
            ns="safe_path_full", marker_id=1, line_width=0.003)

        # Current EE position marker
        path_publisher.publish_current_position_marker(
            traj_ee_safe[idx],
            color_rgba=(0.0, 1.0, 0.0, 1.0),
            ns="safe_current", marker_id=100, size=0.03)

        # Label
        path_publisher.publish_label(
            f"SAFE - {frame_names[ki]}",
            [0.0, 0.0, 1.3],
            color_rgba=(0.0, 0.8, 0.0, 1.0),
            ns="label", marker_id=200, size=0.05)

        # Move robot to this configuration
        q_rad = traj_q_safe[idx]
        collision_oracle.publish_joint_state(q_rad, duration=0.1)
        rospy.sleep(0.5)

        logging.info(f"  EE position: ({traj_ee_safe[idx][0]:.3f}, "
                     f"{traj_ee_safe[idx][1]:.3f}, {traj_ee_safe[idx][2]:.3f})")
        logging.info(f"  >>> SCREENSHOT NOW! Pausing for {args.pause_time}s <<<")
        logging.info(f"  Save as: keyframe_safe_{ki+1}.png")

        rospy.sleep(args.pause_time)

    # =========================================================================
    # Display keyframes: UNSAFE trajectory (red path)
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("UNSAFE TRAJECTORY KEYFRAMES (nominal only, no collision avoidance)")
    logging.info("=" * 60)

    for ki, idx in enumerate(unsafe_indices):
        logging.info(f"\n--- Unsafe Keyframe {ki+1}/3: {frame_names[ki]} (step {idx}) ---")

        # Clear previous
        collision_oracle.clear_markers()
        collision_oracle.clear_trajectory_display()

        # Setup scene
        collision_oracle.clear_all_obstacles()
        collision_oracle.add_table(height=0.0)
        collision_oracle.add_obstacles_from_array(obstacles)
        rospy.sleep(0.5)

        # Start and goal markers
        collision_oracle.publish_start_marker(e_s[0], e_s[1], e_s[2])
        collision_oracle.publish_goal_marker(e_t[0], e_t[1], e_t[2])

        # Publish FULL path as red line (up to current step)
        ee_path_so_far = traj_ee_unsafe[:idx+1]
        path_publisher.publish_path(
            ee_path_so_far,
            color_rgba=(0.9, 0.1, 0.0, 0.9),  # red
            ns="unsafe_path", marker_id=0, line_width=0.006)

        # Full planned path as faint red
        path_publisher.publish_path(
            traj_ee_unsafe,
            color_rgba=(0.7, 0.0, 0.0, 0.3),  # faint red
            ns="unsafe_path_full", marker_id=1, line_width=0.003)

        # Current EE position marker
        path_publisher.publish_current_position_marker(
            traj_ee_unsafe[idx],
            color_rgba=(1.0, 0.0, 0.0, 1.0),
            ns="unsafe_current", marker_id=100, size=0.03)

        # Label
        label_text = f"UNSAFE - {frame_names[ki]}"
        if ki == 1:
            label_text += " (COLLISION!)"
        path_publisher.publish_label(
            label_text,
            [0.0, 0.0, 1.3],
            color_rgba=(0.9, 0.0, 0.0, 1.0),
            ns="label", marker_id=200, size=0.05)

        # Move robot
        q_rad = traj_q_unsafe[idx]
        collision_oracle.publish_joint_state(q_rad, duration=0.1)
        rospy.sleep(0.5)

        logging.info(f"  EE position: ({traj_ee_unsafe[idx][0]:.3f}, "
                     f"{traj_ee_unsafe[idx][1]:.3f}, {traj_ee_unsafe[idx][2]:.3f})")
        logging.info(f"  >>> SCREENSHOT NOW! Pausing for {args.pause_time}s <<<")
        logging.info(f"  Save as: keyframe_unsafe_{ki+1}.png")

        rospy.sleep(args.pause_time)

    # =========================================================================
    # Summary
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("KEYFRAME CAPTURE COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Save 6 screenshots to: {OUTPUT_DIR}")
    logging.info("  keyframe_safe_1.png, keyframe_safe_2.png, keyframe_safe_3.png")
    logging.info("  keyframe_unsafe_1.png, keyframe_unsafe_2.png, keyframe_unsafe_3.png")
    logging.info("\nArrange in Google Draw as 2×3 grid:")
    logging.info("  Row 1: Safe trajectory (green checkmarks)")
    logging.info("  Row 2: Unsafe trajectory (red crosses)")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
