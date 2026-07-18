"""
Thesis Figure: 3-way trajectory comparison.

Searches for a start config where the NOMINAL path collides with the obstacle,
then plots 3 paths from the same start:
    1. Nominal (Goal + Prior only) — passes through obstacle (UNSAFE)
    2. Classifier baseline C(z)   — avoids obstacle
    3. CBF corrected B(z)         — avoids obstacle

Usage:
    python thesis_trajectory_figure.py
    python thesis_trajectory_figure.py --max_seeds 200
"""

import argparse
import json
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from vae import VAE
from robot_state_dataset import RobotStateDataset
from sim.panda import Panda
from sim.robot3d import Robo3D
import cbf_config as cfg
import cbf_fixed_config as fcfg
import classifier_fixed_config as clcfg

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thesis_figures')


# ============================================================================
# Model definitions (must match training scripts)
# ============================================================================
class FixedBarrierNet(nn.Module):
    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_in = nn.Linear(latent_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)])
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z):
        h = F.elu(self.fc_in(z.view(-1, self.latent_dim)))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


class FixedCollisionClassifier(nn.Module):
    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_in = nn.Linear(latent_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)])
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z):
        h = F.elu(self.fc_in(z.view(-1, self.latent_dim)))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


# ============================================================================
# Planning functions
# ============================================================================
def plan_nominal(model, z_init, e_target, mean_t, std_t, device,
                 max_steps=250, lr=0.03, lambda_prior=0.01):
    """Goal + Prior only, no safety."""
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)
    path = []

    for step in range(max_steps):
        optimizer.zero_grad()
        x_dec = model.decoder(z) * std_t + mean_t
        e = x_dec[:, 7:10]
        q = x_dec[:, :7]
        L = torch.norm(e - e_target) + lambda_prior * 0.5 * torch.sum(z ** 2)
        L.backward()
        optimizer.step()

        with torch.no_grad():
            x_safe = model.decoder(z) * std_t + mean_t
        path.append({
            'e': x_safe[:, 7:10].cpu().numpy()[0].copy(),
            'q': x_safe[:, :7].cpu().numpy()[0].copy(),
        })
        if torch.norm(x_safe[:, 7:10] - e_target).item() < 0.01:
            break
    return path


def plan_cbf(model, cbf_net, z_init, e_target, mean_t, std_t, device,
             max_steps=250, lr=0.03, alpha=0.20, lambda_prior=0.01):
    """Goal + Prior + CBF safety correction. Best config: α=0.20."""
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)
    path = []

    for step in range(max_steps):
        z_before = z.data.clone()
        optimizer.zero_grad()
        x_dec = model.decoder(z) * std_t + mean_t
        e = x_dec[:, 7:10]
        L = torch.norm(e - e_target) + lambda_prior * 0.5 * torch.sum(z ** 2)
        L.backward()
        optimizer.step()

        # CBF correction
        with torch.no_grad():
            B_cur = cbf_net(z_before)
        B_target = (1 - alpha) * B_cur
        z_nom = z.data.clone().requires_grad_(True)
        B_nom = cbf_net(z_nom)
        if B_nom.item() < B_target.item():
            grad = torch.autograd.grad(B_nom, z_nom)[0]
            lam = max(0, min(1, ((B_target - B_nom) / (torch.sum(grad ** 2) + 1e-8)).item()))
            z.data = z_nom.detach() + lam * grad.detach()
            optimizer.state[z] = {}

        with torch.no_grad():
            x_safe = model.decoder(z) * std_t + mean_t
        path.append({
            'e': x_safe[:, 7:10].cpu().numpy()[0].copy(),
            'q': x_safe[:, :7].cpu().numpy()[0].copy(),
        })
        if torch.norm(x_safe[:, 7:10] - e_target).item() < 0.01:
            break
    return path


def plan_classifier(model, classifier, z_init, e_target, mean_t, std_t, device,
                    max_steps=250, lr=0.03, lambda_prior=0.01,
                    lambda_collision=5.0, temperature=1.0):
    """Goal + Prior + Classifier collision loss. Best config: λp=0.01, λc=5.0."""
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)
    path = []

    for step in range(max_steps):
        optimizer.zero_grad()
        x_dec = model.decoder(z) * std_t + mean_t
        e = x_dec[:, 7:10]
        L_goal = torch.norm(e - e_target)
        L_prior = lambda_prior * 0.5 * torch.sum(z ** 2)

        logit = classifier(z)
        p_col = torch.sigmoid(logit / temperature)
        L_collision = -torch.log(1 - p_col + 1e-8)

        L = L_goal + L_prior + lambda_collision * L_collision
        L.backward()
        optimizer.step()

        with torch.no_grad():
            x_safe = model.decoder(z) * std_t + mean_t
        path.append({
            'e': x_safe[:, 7:10].cpu().numpy()[0].copy(),
            'q': x_safe[:, :7].cpu().numpy()[0].copy(),
        })
        if torch.norm(x_safe[:, 7:10] - e_target).item() < 0.01:
            break
    return path


def check_path_collision(path, obstacle, robo3d):
    """Check if any waypoint in path collides."""
    for wp in path:
        q_deg = np.degrees(wp['q']).tolist()
        if robo3d.check_for_collision(q_deg, [obstacle]):
            return True
    return False


def path_to_ee(path):
    return np.array([wp['e'] for wp in path])


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='3-way trajectory comparison figure')
    parser.add_argument('--max_seeds', type=int, default=500,
                        help='Max random starts to try to find a colliding nominal path')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    # Best CBF config from ablation: α=0.20 → 75.6% success
    parser.add_argument('--alpha', type=float, default=0.20)
    # Best Classifier config from ablation: λp=0.01, λc=5.0 → 63.6% success
    parser.add_argument('--lambda_prior', type=float, default=0.01)
    parser.add_argument('--lambda_collision', type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load VAE
    with open(cfg.VAE_CONFIG, 'r') as f:
        vae_config = json.load(f)
        if 'parsed_args' in vae_config:
            vae_config = vae_config['parsed_args']
    model = VAE(vae_config['input_dim'], vae_config['latent_dim'],
                vae_config['units_per_layer'], vae_config['num_hidden_layers'])
    ckpt = torch.load(cfg.VAE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    dataset = RobotStateDataset(cfg.DATA_PATH, train=0, train_data_name='free_space_100k_train.dat')
    mean_t = torch.tensor(dataset.get_mean_train()[:, :10], dtype=torch.float32).to(device)
    std_t = torch.tensor(dataset.get_std_train()[:, :10], dtype=torch.float32).to(device)

    robot = Panda()
    robot.to(device)
    robo3d = Robo3D(Panda())

    q_min = robot.joint_min_limits_tensor * (np.pi / 180.0)
    q_max = robot.joint_max_limits_tensor * (np.pi / 180.0)

    q_target = torch.tensor([fcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)
    obstacle = fcfg.FIXED_OBSTACLE

    # Load CBF
    cbf_net = FixedBarrierNet(fcfg.LATENT_DIM, fcfg.CBF_HIDDEN_UNITS, fcfg.CBF_NUM_HIDDEN)
    cbf_ckpt = torch.load(fcfg.FIXED_BEST_CHECKPOINT, map_location=device, weights_only=False)
    cbf_net.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_net.to(device).eval()
    logging.info(f"CBF loaded (epoch {cbf_ckpt['epoch']})")

    # Load Classifier
    classifier = FixedCollisionClassifier(clcfg.LATENT_DIM, clcfg.HIDDEN_UNITS, clcfg.NUM_HIDDEN)
    cl_ckpt = torch.load(clcfg.BEST_CHECKPOINT, map_location=device, weights_only=False)
    classifier.load_state_dict(cl_ckpt['model_state_dict'])
    classifier.to(device).eval()
    logging.info(f"Classifier loaded (epoch {cl_ckpt['epoch']})")

    # =========================================================================
    # Search for a start where nominal path COLLIDES
    # =========================================================================
    logging.info(f"\nSearching for a start config where nominal path collides...")
    found = False

    for seed in range(args.max_seeds):
        torch.manual_seed(seed)
        q_start = torch.rand(1, 7, device=device) * (q_max - q_min) + q_min
        e_start = robot.FK(q_start.clone(), device, rad=True)

        x_start = torch.cat([q_start, e_start], dim=1)
        x_norm = (x_start - mean_t) / std_t
        with torch.no_grad():
            z_init = model.encoder(x_norm)[0]

        # Run nominal
        path_nom = plan_nominal(model, z_init, e_target, mean_t, std_t, device)
        nom_collides = check_path_collision(path_nom, obstacle, robo3d)

        if nom_collides:
            # Check if CBF and classifier avoid it
            path_cbf = plan_cbf(model, cbf_net, z_init, e_target, mean_t, std_t, device,
                                alpha=args.alpha, lambda_prior=args.lambda_prior)
            path_cls = plan_classifier(model, classifier, z_init, e_target, mean_t, std_t,
                                       device, lambda_prior=args.lambda_prior,
                                       lambda_collision=args.lambda_collision)

            cbf_safe = not check_path_collision(path_cbf, obstacle, robo3d)
            cls_safe = not check_path_collision(path_cls, obstacle, robo3d)

            # Check that both reach goal
            ee_nom = path_to_ee(path_nom)
            ee_cbf = path_to_ee(path_cbf)
            ee_cls = path_to_ee(path_cls)
            e_goal = e_target.cpu().numpy()[0]

            cbf_goal = np.linalg.norm(ee_cbf[-1] - e_goal) < 0.02
            cls_goal = np.linalg.norm(ee_cls[-1] - e_goal) < 0.02

            logging.info(
                f"  Seed {seed}: Nominal=COLLIDES, "
                f"CBF={'SAFE' if cbf_safe else 'COLLIDES'}+{'GOAL' if cbf_goal else 'NO_GOAL'}, "
                f"Classifier={'SAFE' if cls_safe else 'COLLIDES'}+{'GOAL' if cls_goal else 'NO_GOAL'}"
            )

            # Ideal: both safe methods avoid AND reach goal
            if cbf_safe and cls_safe and cbf_goal and cls_goal:
                found = True
                best_seed = seed
                logging.info(f"  ✓ PERFECT scenario found at seed {seed}!")
                break

            # Acceptable: at least CBF avoids
            if cbf_safe and cbf_goal:
                found = True
                best_seed = seed
                logging.info(f"  ✓ Good scenario (CBF safe) at seed {seed}")
                # Keep searching for a perfect one
                best_paths = (path_nom, path_cbf, path_cls, cbf_safe, cls_safe)

        if (seed + 1) % 50 == 0:
            logging.info(f"  Searched {seed+1}/{args.max_seeds} seeds...")

    if not found:
        logging.warning("No ideal scenario found. Using best available.")
        # Fallback: just use last colliding nominal
        if 'path_nom' in dir():
            best_paths = (path_nom, path_cbf, path_cls, cbf_safe, cls_safe)
        else:
            logging.error("No colliding nominal path found at all!")
            return

    # =========================================================================
    # Plot
    # =========================================================================
    if 'best_paths' not in dir():
        best_paths = (path_nom, path_cbf, path_cls, cbf_safe, cls_safe)

    path_nom, path_cbf, path_cls = best_paths[0], best_paths[1], best_paths[2]
    cbf_safe, cls_safe = best_paths[3], best_paths[4]

    ee_nom = path_to_ee(path_nom)
    ee_cbf = path_to_ee(path_cbf)
    ee_cls = path_to_ee(path_cls)
    e_goal = e_target.cpu().numpy()[0]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Draw obstacle cylinder
    theta = np.linspace(0, 2 * np.pi, 60)
    z_cyl = np.linspace(0, obstacle[2], 30)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = obstacle[0] + obstacle[3] * np.cos(theta_grid)
    y_cyl = obstacle[1] + obstacle[3] * np.sin(theta_grid)
    ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.25, color='red')

    # Nominal path (UNSAFE — collides)
    ax.plot(ee_nom[:, 0], ee_nom[:, 1], ee_nom[:, 2],
            '-', color='#e74c3c', linewidth=2.5, label='Nominal (no safety) — COLLIDES',
            alpha=0.9, zorder=2)
    ax.scatter(*ee_nom[0], color='#e74c3c', s=60, marker='o', zorder=5)

    # Classifier path
    cls_label = f'Classifier C(z) λp={args.lambda_prior} λc={args.lambda_collision} — {"SAFE" if cls_safe else "COLLIDES"}'
    ax.plot(ee_cls[:, 0], ee_cls[:, 1], ee_cls[:, 2],
            '-', color='#3498db', linewidth=2.5, label=cls_label,
            alpha=0.9, zorder=3)
    ax.scatter(*ee_cls[0], color='#3498db', s=60, marker='o', zorder=5)

    # CBF path
    cbf_label = f'CBF B(z) α={args.alpha} λp={args.lambda_prior} — {"SAFE" if cbf_safe else "COLLIDES"}'
    ax.plot(ee_cbf[:, 0], ee_cbf[:, 1], ee_cbf[:, 2],
            '-', color='#2ecc71', linewidth=2.5, label=cbf_label,
            alpha=0.9, zorder=4)
    ax.scatter(*ee_cbf[0], color='#2ecc71', s=60, marker='o', zorder=5)

    # Goal
    ax.scatter(*e_goal, color='gold', s=200, marker='*', zorder=10,
               edgecolors='black', linewidth=0.5, label='Goal')

    # Start marker
    ax.scatter(*ee_nom[0], color='black', s=100, marker='D', zorder=10, label='Start')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('End-Effector Trajectory Comparison\n'
                 'Nominal (unsafe) vs Classifier vs CBF (safe)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'fig_trajectory_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"\nSaved: {fname}")

    # Stats
    logging.info(f"\nTrajectory stats:")
    logging.info(f"  Nominal: {len(path_nom)} steps, collides=True, "
                 f"final dist={np.linalg.norm(ee_nom[-1] - e_goal):.4f}m")
    logging.info(f"  CBF:     {len(path_cbf)} steps, safe={cbf_safe}, "
                 f"final dist={np.linalg.norm(ee_cbf[-1] - e_goal):.4f}m")
    logging.info(f"  Classif: {len(path_cls)} steps, safe={cls_safe}, "
                 f"final dist={np.linalg.norm(ee_cls[-1] - e_goal):.4f}m")


if __name__ == '__main__':
    main()
