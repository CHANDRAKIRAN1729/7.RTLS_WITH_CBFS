"""
CBF v2 Data Generation — Unified pipeline with signed distance targets.

Follows the v1 pattern:
    1. Run nominal planner (Goal + Prior ONLY) on random scenarios
    2. Record transitions (z_k, z_nom, obs, sdf_k, sdf_nom) at each step
    3. Extract state-label data FROM the same transitions
    4. Split by scenario to prevent data leakage

Key improvement over v1: computes continuous signed distance (not binary labels)
using Robo3D.dist_jpos_to_obstacles for gap distance, and radius-shrinking
for penetration depth estimation.

Usage:
    python cbf_v2_generate_data.py
    python cbf_v2_generate_data.py --num_scenarios 35000
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import time
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
from evaluate_planning import ObstacleScenarioGenerator
import cbf_config as cfg
import cbf_v2_config as v2cfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# Helpers
# =============================================================================
def load_frozen_vae(device):
    with open(cfg.VAE_CONFIG, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']
    model = VAE(config['input_dim'], config['latent_dim'],
                config['units_per_layer'], config['num_hidden_layers'])
    ckpt = torch.load(cfg.VAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    logging.info(f"Frozen VAE loaded from {cfg.VAE_CHECKPOINT}")
    return model


def get_normalization_stats():
    dataset = RobotStateDataset(
        cfg.DATA_PATH, train=0, train_data_name='free_space_100k_train.dat')
    return dataset.get_mean_train(), dataset.get_std_train()


def compute_signed_distance(robo3d, q_deg, obstacle_xyhr, n_radius_steps=10):
    """
    Compute signed distance from robot to obstacle.

    Uses Robo3D.dist_jpos_to_obstacles:
        > 0 → safe (gap distance)
        = 0 → collision → estimate penetration by shrinking obstacle radius

    Returns:
        float: positive if safe, negative if colliding
    """
    dist = robo3d.dist_jpos_to_obstacles(q_deg, [obstacle_xyhr])

    if dist > 0:
        return dist
    else:
        # Collision — estimate penetration depth
        x, y, h, r = obstacle_xyhr
        penetration = 0.0
        for step in range(1, n_radius_steps + 1):
            shrink = r * step / n_radius_steps
            r_test = r - shrink
            if r_test <= 0.001:
                penetration = r
                break
            test_dist = robo3d.dist_jpos_to_obstacles(q_deg, [[x, y, h, r_test]])
            if test_dist > 0:
                penetration = shrink
                break
            penetration = shrink
        return -penetration


# =============================================================================
# Run nominal planner and record transitions with signed distances
# =============================================================================
def run_nominal_planner_and_record_transitions(
        model, robot, robo3d,
        q_start, e_start, e_target, obstacles_raw,
        mean_train_t, std_train_t,
        device, planning_lr, lambda_prior, max_steps):
    """
    Run Goal+Prior-only planner and record (z_k, z_nom, obs, sdf_k, sdf_nom).

    This is the NOMINAL planner — no collision loss. Transitions capture
    how the planner moves through latent space if unconstrained by safety.
    """
    x_start = torch.cat([q_start, e_start], dim=1)
    x_start_norm = (x_start - mean_train_t[:, :10]) / std_train_t[:, :10]

    with torch.no_grad():
        z_init = model.encoder(x_start_norm)[0]

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=planning_lr)

    transitions = []
    obstacles_xyhr = [obs.tolist() for obs in obstacles_raw]
    obs_for_record = obstacles_raw[0]  # first obstacle

    for step in range(max_steps):
        z_before = z.data.clone()

        optimizer.zero_grad()
        x_decoded_norm = model.decoder(z)
        x_decoded = x_decoded_norm * std_train_t[:, :10] + mean_train_t[:, :10]
        e_decoded = x_decoded[:, 7:10]

        L_goal = torch.norm(e_decoded - e_target)
        L_prior = 0.5 * torch.sum(z ** 2)
        L_nominal = L_goal + lambda_prior * L_prior

        if L_goal.item() < cfg.SUCCESS_THRESHOLD:
            break

        L_nominal.backward()
        optimizer.step()

        z_after = z.data.clone()

        # Decode and compute signed distances for z_before and z_after
        with torch.no_grad():
            x_before = model.decoder(z_before) * std_train_t[:, :10] + mean_train_t[:, :10]
            q_before = x_before[0, :7].cpu().numpy()
            q_before_deg = np.degrees(q_before).tolist()

            x_after = model.decoder(z_after) * std_train_t[:, :10] + mean_train_t[:, :10]
            q_after = x_after[0, :7].cpu().numpy()
            q_after_deg = np.degrees(q_after).tolist()

        sdf_k = compute_signed_distance(robo3d, q_before_deg, obstacles_xyhr[0])
        sdf_nom = compute_signed_distance(robo3d, q_after_deg, obstacles_xyhr[0])

        transitions.append({
            'z_k': z_before.cpu().squeeze(0),
            'z_nom': z_after.cpu().squeeze(0),
            'obs': torch.tensor(obs_for_record, dtype=torch.float32),
            'sdf_k': sdf_k,
            'sdf_nom': sdf_nom,
        })

    return transitions


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='CBF v2 unified data generation')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=v2cfg.SEED)
    parser.add_argument('--num_scenarios', type=int, default=v2cfg.NUM_TRANSITION_SCENARIOS)
    parser.add_argument('--max_steps', type=int, default=cfg.TRANSITION_MAX_STEPS)
    parser.add_argument('--planning_lr', type=float, default=cfg.TRANSITION_PLANNING_LR)
    parser.add_argument('--lambda_prior', type=float, default=cfg.TRANSITION_LAMBDA_PRIOR)
    parser.add_argument('--num_obstacles', type=int, default=1)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(v2cfg.DATA_DIR, exist_ok=True)

    model = load_frozen_vae(device)
    mean_train, std_train = get_normalization_stats()
    mean_train_t = torch.tensor(mean_train, dtype=torch.float32).to(device)
    std_train_t = torch.tensor(std_train, dtype=torch.float32).to(device)

    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())
    scenario_gen = ObstacleScenarioGenerator(robot)

    q_min_rad = robot.joint_min_limits_tensor * (torch.pi / 180.0)
    q_max_rad = robot.joint_max_limits_tensor * (torch.pi / 180.0)

    # =========================================================================
    # Phase 1: Generate transitions from nominal planner
    # =========================================================================
    logging.info(f"\n{'=' * 60}")
    logging.info(f"CBF v2 DATA GENERATION (unified, with signed distance)")
    logging.info(f"  Scenarios: {args.num_scenarios}")
    logging.info(f"  Planner: Goal + Prior only (lr={args.planning_lr}, "
                 f"λ_prior={args.lambda_prior})")
    logging.info(f"{'=' * 60}\n")

    all_transitions = []
    scenario_boundaries = [0]
    start_time = time.time()

    for scenario_id in range(args.num_scenarios):
        q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_start = robot.FK(q_start.clone(), device, rad=True)
        q_target = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
        e_target = robot.FK(q_target.clone(), device, rad=True)

        obstacles_raw = scenario_gen.generate_scenario(
            q_start.cpu().numpy()[0],
            e_start.cpu().numpy()[0],
            e_target.cpu().numpy()[0],
            num_obstacles=args.num_obstacles
        )

        if len(obstacles_raw) == 0:
            continue

        transitions = run_nominal_planner_and_record_transitions(
            model, robot, robo3d,
            q_start, e_start, e_target, obstacles_raw,
            mean_train_t, std_train_t,
            device, args.planning_lr, args.lambda_prior, args.max_steps
        )

        all_transitions.extend(transitions)
        scenario_boundaries.append(len(all_transitions))

        if (scenario_id + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (scenario_id + 1) / elapsed
            n_col = sum(1 for t in all_transitions if t['sdf_k'] <= 0)
            logging.info(
                f"Progress: {scenario_id+1}/{args.num_scenarios} "
                f"({rate:.1f} scen/s) | "
                f"Transitions: {len(all_transitions)} | "
                f"Collisions: {n_col} ({n_col/max(len(all_transitions),1)*100:.1f}%)"
            )

    elapsed = time.time() - start_time
    logging.info(f"\nGenerated {len(all_transitions)} transitions from "
                 f"{args.num_scenarios} scenarios in {elapsed:.1f}s")

    if len(all_transitions) == 0:
        logging.error("No transitions generated!")
        return

    # =========================================================================
    # Phase 2: Collate into tensors
    # =========================================================================
    z_k_all = torch.stack([t['z_k'] for t in all_transitions])
    z_nom_all = torch.stack([t['z_nom'] for t in all_transitions])
    obs_all = torch.stack([t['obs'] for t in all_transitions])
    sdf_k_all = torch.tensor([t['sdf_k'] for t in all_transitions], dtype=torch.float32)
    sdf_nom_all = torch.tensor([t['sdf_nom'] for t in all_transitions], dtype=torch.float32)

    # Clip SDF values
    sdf_k_clipped = torch.clamp(sdf_k_all * v2cfg.SDF_SCALE, -v2cfg.SDF_CLIP, v2cfg.SDF_CLIP)
    sdf_nom_clipped = torch.clamp(sdf_nom_all * v2cfg.SDF_SCALE, -v2cfg.SDF_CLIP, v2cfg.SDF_CLIP)

    # Binary labels (derived from SDF)
    safe_k_all = (sdf_k_all > 0).float()
    safe_nom_all = (sdf_nom_all > 0).float()

    # Statistics
    total = len(all_transitions)
    logging.info(f"\nTransition statistics:")
    logging.info(f"  z_k:   safe={int(safe_k_all.sum())} "
                 f"({safe_k_all.mean()*100:.1f}%)")
    logging.info(f"  z_nom: safe={int(safe_nom_all.sum())} "
                 f"({safe_nom_all.mean()*100:.1f}%)")
    logging.info(f"  SDF_k range: [{sdf_k_all.min():.4f}, {sdf_k_all.max():.4f}]")
    logging.info(f"  SDF_nom range: [{sdf_nom_all.min():.4f}, {sdf_nom_all.max():.4f}]")

    # =========================================================================
    # Phase 3: Split by scenario (prevent data leakage)
    # =========================================================================
    num_scenarios = len(scenario_boundaries) - 1
    n_val_scenarios = max(1, int(num_scenarios * args.val_split))
    n_train_scenarios = num_scenarios - n_val_scenarios

    scenario_perm = torch.randperm(num_scenarios).tolist()
    train_scenarios = scenario_perm[:n_train_scenarios]
    val_scenarios = scenario_perm[n_train_scenarios:]

    def gather_scenario_indices(scenario_ids):
        indices = []
        for s in scenario_ids:
            start = scenario_boundaries[s]
            end = scenario_boundaries[s + 1]
            indices.extend(range(start, end))
        return indices

    train_idx = gather_scenario_indices(train_scenarios)
    val_idx = gather_scenario_indices(val_scenarios)

    # =========================================================================
    # Phase 4: Save transition data
    # =========================================================================
    train_data = {
        'z_k': z_k_all[train_idx], 'z_nom': z_nom_all[train_idx],
        'obs': obs_all[train_idx],
        'sdf_k': sdf_k_clipped[train_idx], 'sdf_nom': sdf_nom_clipped[train_idx],
        'safe_k': safe_k_all[train_idx], 'safe_nom': safe_nom_all[train_idx],
    }
    val_data = {
        'z_k': z_k_all[val_idx], 'z_nom': z_nom_all[val_idx],
        'obs': obs_all[val_idx],
        'sdf_k': sdf_k_clipped[val_idx], 'sdf_nom': sdf_nom_clipped[val_idx],
        'safe_k': safe_k_all[val_idx], 'safe_nom': safe_nom_all[val_idx],
    }

    torch.save(train_data, v2cfg.TRANSITION_TRAIN)
    logging.info(f"Saved training transitions: {len(train_idx)} → {v2cfg.TRANSITION_TRAIN}")
    torch.save(val_data, v2cfg.TRANSITION_VAL)
    logging.info(f"Saved validation transitions: {len(val_idx)} → {v2cfg.TRANSITION_VAL}")

    # =========================================================================
    # Phase 5: Extract state-label data FROM the SAME transitions
    #
    # Each transition (z_k, z_nom) gives two state-label samples:
    #   (z_k, obs, label_k, sdf_k)  and  (z_nom, obs, label_nom, sdf_nom)
    # =========================================================================
    def extract_state_labels(indices):
        z_k_sub = z_k_all[indices]
        z_nom_sub = z_nom_all[indices]
        obs_sub = obs_all[indices]
        sdf_k_sub = sdf_k_clipped[indices]
        sdf_nom_sub = sdf_nom_clipped[indices]
        safe_k_sub = safe_k_all[indices]
        safe_nom_sub = safe_nom_all[indices]

        z_states = torch.cat([z_k_sub, z_nom_sub], dim=0)
        obs_states = torch.cat([obs_sub, obs_sub], dim=0)
        sdf_states = torch.cat([sdf_k_sub, sdf_nom_sub], dim=0)
        # label: safe=0, unsafe=1
        labels = torch.cat([1.0 - safe_k_sub, 1.0 - safe_nom_sub], dim=0)

        return {'z': z_states, 'obs': obs_states, 'label': labels, 'sdf': sdf_states}

    train_state = extract_state_labels(train_idx)
    val_state = extract_state_labels(val_idx)

    torch.save(train_state, v2cfg.STATE_LABELS_TRAIN)
    n_safe = (train_state['label'] == 0).sum().item()
    n_unsafe = (train_state['label'] == 1).sum().item()
    logging.info(f"Saved training state-labels: {len(train_state['z'])} "
                 f"(safe={n_safe}, unsafe={n_unsafe}) → {v2cfg.STATE_LABELS_TRAIN}")

    torch.save(val_state, v2cfg.STATE_LABELS_VAL)
    n_safe_v = (val_state['label'] == 0).sum().item()
    n_unsafe_v = (val_state['label'] == 1).sum().item()
    logging.info(f"Saved validation state-labels: {len(val_state['z'])} "
                 f"(safe={n_safe_v}, unsafe={n_unsafe_v}) → {v2cfg.STATE_LABELS_VAL}")

    logging.info(f"\n{'=' * 60}")
    logging.info("Data generation complete!")
    logging.info(f"  Transitions:  train={len(train_idx)}, val={len(val_idx)}")
    logging.info(f"  State-labels: train={len(train_state['z'])}, "
                 f"val={len(val_state['z'])}")
    logging.info(f"  SDF range: [{sdf_k_all.min():.4f}, {sdf_k_all.max():.4f}]")
    logging.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
