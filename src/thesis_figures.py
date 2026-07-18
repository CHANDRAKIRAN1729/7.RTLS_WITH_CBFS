"""
Thesis Figure Generator — All publication-quality figures.

Generates:
    Fig 1: α tradeoff curves (3 CBF models)
    Fig 2: Classifier ablation heatmap
    Fig 3: Covariate shift histogram (B values)
    Fig 4: 3D trajectory comparison

Usage:
    python thesis_figures.py --figure all
    python thesis_figures.py --figure alpha
    python thesis_figures.py --figure heatmap
    python thesis_figures.py --figure histogram
    python thesis_figures.py --figure trajectory
"""

import argparse
import logging
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thesis_figures')

# ============================================================================
# DATA — from ablation experiments
# ============================================================================
ALPHAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

CBF_FIXED = {
    'success': [46.8,74.4,74.4,75.0,75.6,74.0,71.0,71.4,71.8,68.6,68.6,67.0,64.8,63.0,62.4,62.2,59.6,57.6,59.0,59.2,58.0],
    'goal':    [56.4,86.2,88.2,90.0,91.4,92.2,92.2,94.6,95.6,95.0,95.6,96.6,97.4,97.8,98.0,98.0,98.2,98.0,98.8,98.6,99.0],
    'cf':      [88.2,87.0,85.0,83.2,82.0,79.2,75.8,74.4,73.6,70.6,70.0,68.4,65.8,63.6,62.8,62.8,60.0,58.2,59.6,59.6,58.4],
}

CBF_MULTIPOS = {
    'success': [36.6,79.4,83.8,83.0,81.6,79.6,79.6,77.4,75.0,71.6,69.4,68.2,67.2,64.8,64.4,62.8,61.2,61.0,59.8,56.6,56.6],
    'goal':    [45.2,88.6,93.6,94.8,95.0,95.8,96.0,96.6,96.2,96.6,96.6,96.6,96.6,96.2,97.6,97.6,97.2,98.0,97.6,97.6,98.4],
    'cf':      [90.8,90.2,89.8,87.6,85.2,82.8,82.4,79.6,77.2,73.6,71.6,70.4,68.8,66.6,65.0,63.6,62.8,62.0,60.8,57.8,57.4],
}

CBF_PLANNING = {
    'success': [19.2,33.0,30.4,30.2,28.8,28.6,28.2,28.8,28.6,29.0,29.2,29.2,29.4,30.0,29.6,30.4,30.4,30.6,31.0,30.8,30.8],
    'goal':    [40.4,69.4,71.4,73.6,74.0,74.0,74.2,74.8,74.2,74.2,74.0,74.6,74.8,75.2,74.4,75.2,74.4,74.8,74.8,74.8,75.8],
    'cf':      [58.4,39.8,34.6,33.2,30.2,30.4,29.6,30.2,29.8,30.0,30.2,30.2,30.4,31.0,30.6,31.4,31.4,31.6,32.0,31.8,31.8],
}

# Classifier ablation: paste your CSV data here
# Format: CLASSIFIER_GRID[lambda_prior][lambda_collision] = (success, goal, cf)
# If you haven't run it yet, set to None
CLASSIFIER_GRID = None  # Will be loaded from CSV if available
CLASSIFIER_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'classifier_fixed_ablation_results',
                                    'ablation_summary.csv')


def load_classifier_grid():
    """Load classifier ablation from CSV if available."""
    if not os.path.exists(CLASSIFIER_CSV_PATH):
        logging.warning(f"Classifier ablation CSV not found: {CLASSIFIER_CSV_PATH}")
        return None
    grid = {}
    with open(CLASSIFIER_CSV_PATH) as f:
        header = f.readline()  # skip
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            lp, lc = float(parts[0]), float(parts[1])
            s = float(parts[2]) if parts[2] else 0
            g = float(parts[3]) if parts[3] else 0
            c = float(parts[4]) if parts[4] else 0
            if lp not in grid:
                grid[lp] = {}
            grid[lp][lc] = (s, g, c)
    return grid


# ============================================================================
# Figure 1: α Tradeoff Curves
# ============================================================================
def plot_alpha_tradeoff():
    """Three-panel figure: α vs metrics for Fixed, MultiPos, Planning."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    datasets = [
        ('FixedBarrierNet B(z)', CBF_FIXED),
        ('MultiPosBarrierNet B(z,x,y)', CBF_MULTIPOS),
        ('BarrierNet B(z,o)', CBF_PLANNING),
    ]

    for ax, (title, data) in zip(axes, datasets):
        ax.plot(ALPHAS, data['success'], 'o-', color='#2ecc71', linewidth=2,
                markersize=4, label='Success Rate', zorder=3)
        ax.plot(ALPHAS, data['goal'], 's-', color='#3498db', linewidth=2,
                markersize=4, label='Goal Reached', zorder=3)
        ax.plot(ALPHAS, data['cf'], '^-', color='#e74c3c', linewidth=2,
                markersize=4, label='Collision-Free', zorder=3)

        # Mark best success
        best_idx = np.argmax(data['success'])
        ax.axvline(x=ALPHAS[best_idx], color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'Best α={ALPHAS[best_idx]:.2f}\n({data["success"][best_idx]:.1f}%)',
                    xy=(ALPHAS[best_idx], data['success'][best_idx]),
                    xytext=(ALPHAS[best_idx]+0.15, data['success'][best_idx]-10),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('α (CBF decay rate)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    axes[0].set_ylabel('Rate (%)', fontsize=12)

    plt.suptitle('CBF α Parameter Ablation — Safety-Performance Tradeoff',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = os.path.join(OUTPUT_DIR, 'fig_alpha_tradeoff.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fname}")

    # Also save individual panels
    for i, (title, data) in enumerate(datasets):
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(ALPHAS, data['success'], 'o-', color='#2ecc71', linewidth=2.5,
                 markersize=5, label='Success Rate')
        ax2.plot(ALPHAS, data['goal'], 's-', color='#3498db', linewidth=2.5,
                 markersize=5, label='Goal Reached')
        ax2.plot(ALPHAS, data['cf'], '^-', color='#e74c3c', linewidth=2.5,
                 markersize=5, label='Collision-Free')
        ax2.set_xlabel('α (CBF decay rate)', fontsize=13)
        ax2.set_ylabel('Rate (%)', fontsize=13)
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        plt.tight_layout()
        names = ['fixed', 'multipos', 'planning']
        fname2 = os.path.join(OUTPUT_DIR, f'fig_alpha_{names[i]}.png')
        plt.savefig(fname2, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {fname2}")


# ============================================================================
# Figure 2: Classifier Ablation Heatmap
# ============================================================================
def plot_classifier_heatmap():
    """Heatmap of success rate over λ_prior × λ_collision grid."""
    grid = load_classifier_grid()
    if grid is None:
        logging.warning("Skipping heatmap — no classifier ablation data found.")
        logging.warning(f"Run classifier_fixed_ablation.sh first, then re-run this script.")
        return

    lp_vals = sorted(grid.keys())
    lc_vals = sorted(list(grid[lp_vals[0]].keys()))

    metrics = {'Success Rate': 0, 'Goal Rate': 1, 'Collision-Free Rate': 2}

    for metric_name, idx in metrics.items():
        matrix = np.zeros((len(lp_vals), len(lc_vals)))
        for i, lp in enumerate(lp_vals):
            for j, lc in enumerate(lc_vals):
                matrix[i, j] = grid[lp][lc][idx]

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                       vmin=matrix.min() - 5, vmax=matrix.max() + 5)

        ax.set_xticks(range(len(lc_vals)))
        ax.set_xticklabels([f'{v}' for v in lc_vals])
        ax.set_yticks(range(len(lp_vals)))
        ax.set_yticklabels([f'{v}' for v in lp_vals])
        ax.set_xlabel('λ_collision', fontsize=13)
        ax.set_ylabel('λ_prior', fontsize=13)

        # Annotate cells
        for i in range(len(lp_vals)):
            for j in range(len(lc_vals)):
                val = matrix[i, j]
                color = 'white' if val < (matrix.max() + matrix.min()) / 2 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

        # Mark best
        best = np.unravel_index(np.argmax(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle((best[1]-0.5, best[0]-0.5), 1, 1,
                                   fill=False, edgecolor='blue', linewidth=3))

        ax.set_title(f'Classifier Fixed-Obstacle: {metric_name} (%)',
                     fontsize=14, fontweight='bold')
        fig.colorbar(im, label=f'{metric_name} (%)')
        plt.tight_layout()

        safe_name = metric_name.lower().replace(' ', '_').replace('-', '_')
        fname = os.path.join(OUTPUT_DIR, f'fig_classifier_heatmap_{safe_name}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {fname}")


# ============================================================================
# Figure 3: Covariate Shift Histogram (B values from training data)
# ============================================================================
def _plot_single_histogram(B_values, labels, model_name, input_label, subtitle, fname):
    """Helper to plot one histogram."""
    safe = B_values[labels == 0]
    unsafe = B_values[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(safe, bins=100, alpha=0.7, color='#2ecc71',
            label=f'Safe ({len(safe)})', density=True)
    ax.hist(unsafe, bins=100, alpha=0.7, color='#e74c3c',
            label=f'Unsafe ({len(unsafe)})', density=True)
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--', label='B=0 boundary')
    ax.set_xlabel(input_label, fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'{model_name}\n{subtitle}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(-8, 8)
    ax.grid(True, alpha=0.3)

    # Add stats text box
    safe_acc = (safe >= 0).mean() * 100
    unsafe_acc = (unsafe < 0).mean() * 100
    stats_text = (f'Safe acc (B≥0): {safe_acc:.1f}%\n'
                  f'Unsafe acc (B<0): {unsafe_acc:.1f}%\n'
                  f'Safe mean: {safe.mean():.2f}\n'
                  f'Unsafe mean: {unsafe.mean():.2f}')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fname}")
    logging.info(f"  Safe: mean={safe.mean():.3f}, std={safe.std():.3f}, acc={safe_acc:.1f}%")
    logging.info(f"  Unsafe: mean={unsafe.mean():.3f}, std={unsafe.std():.3f}, acc={unsafe_acc:.1f}%")


def plot_barrier_histogram():
    """Histogram of B(z) values on training data — separate plot per model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import cbf_config as cfg
    import cbf_fixed_config as fcfg
    import cbf_multipos_config as mcfg

    device = torch.device("cpu")
    logging.info("Using CPU for histogram (avoids CUDA OOM issues)")

    # --- Fixed model: B(z) ---
    try:
        from cbf_fixed_train import FixedBarrierNet
        ckpt = torch.load(fcfg.FIXED_BEST_CHECKPOINT, map_location=device, weights_only=False)
        net = FixedBarrierNet(fcfg.LATENT_DIM, fcfg.CBF_HIDDEN_UNITS, fcfg.CBF_NUM_HIDDEN)
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(device).eval()

        data = torch.load(fcfg.FIXED_STATE_LABELS_TRAIN, weights_only=False)
        z, labels = data['z'].float(), data['label'].float().numpy()
        with torch.no_grad():
            B = net(z).numpy()

        logging.info(f"\n--- FixedBarrierNet B(z): {len(B)} samples ---")
        _plot_single_histogram(B, labels, 'FixedBarrierNet B(z)', 'B(z)',
                               f'Obstacle: {fcfg.FIXED_OBSTACLE} (epoch {ckpt["epoch"]})',
                               os.path.join(OUTPUT_DIR, 'fig_histogram_fixed.png'))
        del net, ckpt, data, z, B
    except Exception as e:
        logging.warning(f"Could not load fixed model: {e}")

    # --- MultiPos model: B(z, x, y) ---
    try:
        from cbf_multipos_train import MultiPosBarrierNet
        ckpt = torch.load(mcfg.MULTIPOS_BEST_CHECKPOINT, map_location=device, weights_only=False)
        net = MultiPosBarrierNet(mcfg.LATENT_DIM, mcfg.OBS_XY_DIM,
                                 mcfg.CBF_HIDDEN_UNITS, mcfg.CBF_NUM_HIDDEN)
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(device).eval()

        data = torch.load(mcfg.MULTIPOS_STATE_LABELS_TRAIN, weights_only=False)
        z = data['z'].float()
        obs_xy = data['obs_xy'].float()
        labels = data['label'].float().numpy()
        with torch.no_grad():
            B = net(z, obs_xy).numpy()

        logging.info(f"\n--- MultiPosBarrierNet B(z,x,y): {len(B)} samples ---")
        _plot_single_histogram(B, labels, 'MultiPosBarrierNet B(z, x, y)', 'B(z, x, y)',
                               f'4 positions (epoch {ckpt["epoch"]})',
                               os.path.join(OUTPUT_DIR, 'fig_histogram_multipos.png'))
        del net, ckpt, data, z, obs_xy, B
    except Exception as e:
        logging.warning(f"Could not load multipos model: {e}")

    # --- Planning model: B(z, o) ---
    try:
        from cbf_model import BarrierNet
        ckpt = torch.load(cfg.CBF_BEST_CHECKPOINT, map_location=device, weights_only=False)
        net = BarrierNet(cfg.LATENT_DIM, cfg.OBS_DIM, cfg.CBF_HIDDEN_UNITS, cfg.CBF_NUM_HIDDEN)
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(device).eval()

        data = torch.load(cfg.STATE_LABELS_TRAIN, weights_only=False)
        z = data['z'].float()
        obs = data['obs'].float()
        labels = data['label'].float().numpy()
        with torch.no_grad():
            B = net(z, obs).numpy()

        logging.info(f"\n--- BarrierNet B(z,o): {len(B)} samples ---")
        _plot_single_histogram(B, labels, 'BarrierNet B(z, o)', 'B(z, o)',
                               f'Random obstacles (epoch {ckpt["epoch"]})',
                               os.path.join(OUTPUT_DIR, 'fig_histogram_planning.png'))
        del net, ckpt, data, z, obs, B
    except Exception as e:
        logging.warning(f"Could not load planning model: {e}")


# ============================================================================
# Figure 4: Qualitative Trajectory (3D end-effector path)
# ============================================================================
def plot_trajectory_comparison():
    """3D plot of end-effector trajectories from saved evaluation data."""
    import torch
    import json

    # We need to run a few planning scenarios and save trajectories
    # This function generates them on-the-fly if models are available
    try:
        from vae import VAE
        from sim.panda import Panda
        from sim.robot3d import Robo3D
        from robot_state_dataset import RobotStateDataset
        import cbf_config as cfg
        import cbf_fixed_config as fcfg
    except ImportError as e:
        logging.warning(f"Cannot generate trajectories: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    mean_train = dataset.get_mean_train()
    std_train = dataset.get_std_train()
    mean_t = torch.tensor(mean_train[:, :10], dtype=torch.float32).to(device)
    std_t = torch.tensor(std_train[:, :10], dtype=torch.float32).to(device)

    robot = Panda()
    robot.to(device)

    q_min = robot.joint_min_limits_tensor * (np.pi / 180.0)
    q_max = robot.joint_max_limits_tensor * (np.pi / 180.0)

    q_target = torch.tensor([fcfg.FIXED_GOAL_Q_RAD], dtype=torch.float32, device=device)
    e_target = robot.FK(q_target.clone(), device, rad=True)

    obstacle = fcfg.FIXED_OBSTACLE  # [x, y, h, r]

    # Find a start that produces a collision-relevant trajectory
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate a nominal (no safety) trajectory
    def run_trajectory(use_cbf=False, cbf_net=None, alpha=0.3):
        q_start = torch.tensor([[0.0, 0.5, 0.0, -1.5, 0.0, 2.0, 0.0]],
                               dtype=torch.float32, device=device)
        e_start = robot.FK(q_start.clone(), device, rad=True)
        x_start = torch.cat([q_start, e_start], dim=1)
        x_norm = (x_start - mean_t) / std_t
        with torch.no_grad():
            z_init = model.encoder(x_norm)[0]

        z = z_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=0.03)
        path_ee = [e_start.cpu().numpy()[0].copy()]

        for step in range(200):
            z_before = z.data.clone()
            optimizer.zero_grad()
            x_dec = model.decoder(z) * std_t + mean_t
            e_dec = x_dec[:, 7:10]
            L = torch.norm(e_dec - e_target) + 0.01 * 0.5 * torch.sum(z**2)
            L.backward()
            optimizer.step()

            if use_cbf and cbf_net is not None:
                with torch.no_grad():
                    B_cur = cbf_net(z_before)
                B_target = (1 - alpha) * B_cur
                z_nom = z.data.clone().requires_grad_(True)
                B_nom = cbf_net(z_nom)
                if B_nom.item() < B_target.item():
                    grad = torch.autograd.grad(B_nom, z_nom)[0]
                    lam = max(0, min(1, ((B_target - B_nom) / (torch.sum(grad**2) + 1e-8)).item()))
                    z.data = z_nom.detach() + lam * grad.detach()
                    optimizer.state[z] = {}

            with torch.no_grad():
                x_safe = model.decoder(z) * std_t + mean_t
                e_safe = x_safe[:, 7:10]
            path_ee.append(e_safe.cpu().numpy()[0].copy())

            if torch.norm(e_safe - e_target).item() < 0.01:
                break

        return np.array(path_ee)

    # Nominal (no safety)
    path_nominal = run_trajectory(use_cbf=False)

    # CBF path
    try:
        from cbf_fixed_train import FixedBarrierNet
        ckpt = torch.load(fcfg.FIXED_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net = FixedBarrierNet(fcfg.LATENT_DIM, fcfg.CBF_HIDDEN_UNITS, fcfg.CBF_NUM_HIDDEN)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        cbf_net.to(device).eval()
        path_cbf = run_trajectory(use_cbf=True, cbf_net=cbf_net, alpha=0.2)
        has_cbf = True
    except Exception as e:
        logging.warning(f"Could not load CBF for trajectory: {e}")
        has_cbf = False
        path_cbf = None

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw obstacle cylinder
    theta = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(0, obstacle[2], 20)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = obstacle[0] + obstacle[3] * np.cos(theta_grid)
    y_cyl = obstacle[1] + obstacle[3] * np.sin(theta_grid)
    ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.3, color='red', label='Obstacle')

    # Nominal path
    ax.plot(path_nominal[:, 0], path_nominal[:, 1], path_nominal[:, 2],
            'b-', linewidth=2, label='Nominal (no safety)', alpha=0.8)
    ax.scatter(*path_nominal[0], color='blue', s=80, marker='o', zorder=5)

    # CBF path
    if has_cbf and path_cbf is not None:
        ax.plot(path_cbf[:, 0], path_cbf[:, 1], path_cbf[:, 2],
                'g-', linewidth=2, label='CBF corrected (α=0.2)', alpha=0.8)
        ax.scatter(*path_cbf[0], color='green', s=80, marker='o', zorder=5)

    # Goal
    e_goal = e_target.cpu().numpy()[0]
    ax.scatter(*e_goal, color='gold', s=150, marker='*', zorder=5, label='Goal')

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title('End-Effector Trajectory: Nominal vs CBF-Corrected', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'fig_trajectory_3d.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fname}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate thesis figures')
    parser.add_argument('--figure', type=str, default='all',
                        choices=['all', 'alpha', 'heatmap', 'histogram', 'trajectory'])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    if args.figure in ('all', 'alpha'):
        logging.info("\n=== Generating α tradeoff curves ===")
        plot_alpha_tradeoff()

    if args.figure in ('all', 'heatmap'):
        logging.info("\n=== Generating classifier heatmap ===")
        plot_classifier_heatmap()

    if args.figure in ('all', 'histogram'):
        logging.info("\n=== Generating barrier histogram ===")
        plot_barrier_histogram()

    if args.figure in ('all', 'trajectory'):
        logging.info("\n=== Generating trajectory comparison ===")
        plot_trajectory_comparison()

    logging.info(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
