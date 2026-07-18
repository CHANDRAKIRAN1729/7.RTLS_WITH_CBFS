"""
CBF Visualization — 3D surface plots of the learned barrier function.

Projects the 7D latent space onto 2D using PCA of the training data,
then evaluates B(z) over a grid to create:
    1. 3D surface plot of B values
    2. Contour plot with B=0 boundary
    3. Scatter of training data colored by safe/unsafe

Works with:
    --model fixed     → FixedBarrierNet B(z)
    --model multipos  → MultiPosBarrierNet B(z, x, y)
    --model planning  → BarrierNet B(z, obs) where obs=[x,y,h,r]
"""

import argparse
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import cbf_config as cfg
import cbf_fixed_config as fcfg
import cbf_multipos_config as mcfg
from cbf_model import BarrierNet

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# Model definitions (must match training scripts)
# =============================================================================
class FixedBarrierNet(nn.Module):
    def __init__(self, latent_dim=7, hidden_units=2048, num_hidden=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_in = nn.Linear(latent_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z):
        h = F.elu(self.fc_in(z.view(-1, self.latent_dim)))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


class MultiPosBarrierNet(nn.Module):
    def __init__(self, latent_dim=7, obs_xy_dim=2, hidden_units=2048, num_hidden=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_xy_dim = obs_xy_dim
        self.fc_in = nn.Linear(latent_dim + obs_xy_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z, obs_xy):
        x = torch.cat([z.view(-1, self.latent_dim), obs_xy.view(-1, self.obs_xy_dim)], dim=-1)
        h = F.elu(self.fc_in(x))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


def load_model_and_data(model_type, device):
    """Load the appropriate model and training data."""
    if model_type == 'fixed':
        cbf_net = FixedBarrierNet(
            latent_dim=fcfg.LATENT_DIM,
            hidden_units=fcfg.CBF_HIDDEN_UNITS,
            num_hidden=fcfg.CBF_NUM_HIDDEN
        )
        ckpt = torch.load(fcfg.FIXED_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        logging.info(f"Loaded FixedBarrierNet (epoch {ckpt['epoch']})")

        data = torch.load(fcfg.FIXED_STATE_LABELS_TRAIN, weights_only=False)
        z_data = data['z']
        labels = data['label']
        obs_xy = None

    elif model_type == 'multipos':
        cbf_net = MultiPosBarrierNet(
            latent_dim=mcfg.LATENT_DIM,
            obs_xy_dim=mcfg.OBS_XY_DIM,
            hidden_units=mcfg.CBF_HIDDEN_UNITS,
            num_hidden=mcfg.CBF_NUM_HIDDEN
        )
        ckpt = torch.load(mcfg.MULTIPOS_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        logging.info(f"Loaded MultiPosBarrierNet (epoch {ckpt['epoch']})")

        data = torch.load(mcfg.MULTIPOS_STATE_LABELS_TRAIN, weights_only=False)
        z_data = data['z']
        labels = data['label']
        obs_xy = data['obs_xy']

    elif model_type == 'planning':
        cbf_net = BarrierNet(
            latent_dim=cfg.LATENT_DIM,
            obs_dim=cfg.OBS_DIM,
            hidden_units=cfg.CBF_HIDDEN_UNITS,
            num_hidden=cfg.CBF_NUM_HIDDEN
        )
        ckpt = torch.load(cfg.CBF_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        logging.info(f"Loaded BarrierNet B(z,obs) (epoch {ckpt['epoch']})")

        data = torch.load(cfg.STATE_LABELS_TRAIN, weights_only=False)
        z_data = data['z']
        labels = data['label']
        obs_xy = data['obs']  # full [x,y,h,r] stored as 'obs'

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cbf_net.to(device)
    cbf_net.eval()
    return cbf_net, z_data, labels, obs_xy


def evaluate_B_on_grid(cbf_net, pca, grid_pc1, grid_pc2, z_mean, device,
                       model_type='fixed', obs_xy_val=None):
    """Evaluate B(z) over a 2D PCA grid."""
    B_grid = np.zeros_like(grid_pc1)
    n_rows, n_cols = grid_pc1.shape

    with torch.no_grad():
        for i in range(n_rows):
            # Build batch of z's for this row
            pc_coords = np.stack([grid_pc1[i, :], grid_pc2[i, :]], axis=1)  # (n_cols, 2)
            # Reconstruct full 7D z from PCA 2D coordinates
            z_7d = pca.inverse_transform(pc_coords)  # (n_cols, 7)
            z_tensor = torch.tensor(z_7d, dtype=torch.float32).to(device)

            if model_type == 'fixed':
                B_vals = cbf_net(z_tensor)
            else:
                obs_tensor = torch.tensor(obs_xy_val, dtype=torch.float32).unsqueeze(0)
                obs_tensor = obs_tensor.expand(z_tensor.shape[0], -1).to(device)
                B_vals = cbf_net(z_tensor, obs_tensor)

            B_grid[i, :] = B_vals.cpu().numpy()

    return B_grid


def create_visualizations(cbf_net, z_data, labels, obs_xy, model_type, device, output_dir,
                          obs_xy_val=None, max_samples=10000):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Subsample for speed
    n = len(z_data)
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        z_sub = z_data[idx].numpy()
        labels_sub = labels[idx].numpy()
    else:
        z_sub = z_data.numpy()
        labels_sub = labels.numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_sub)
    logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Evaluate B on data points
    with torch.no_grad():
        z_tensor = torch.tensor(z_sub, dtype=torch.float32).to(device)
        if model_type == 'fixed':
            B_data = cbf_net(z_tensor).cpu().numpy()
        else:
            if obs_xy_val is not None:
                obs_t = torch.tensor(obs_xy_val, dtype=torch.float32).unsqueeze(0)
                obs_t = obs_t.expand(z_tensor.shape[0], -1).to(device)
            else:
                obs_t = torch.tensor(obs_xy[idx] if n > max_samples else obs_xy,
                                     dtype=torch.float32).to(device)
            B_data = cbf_net(z_tensor, obs_t).cpu().numpy()

    # Create grid for surface plot
    margin = 0.5
    pc1_min, pc1_max = z_pca[:, 0].min() - margin, z_pca[:, 0].max() + margin
    pc2_min, pc2_max = z_pca[:, 1].min() - margin, z_pca[:, 1].max() + margin
    grid_res = 100
    pc1_range = np.linspace(pc1_min, pc1_max, grid_res)
    pc2_range = np.linspace(pc2_min, pc2_max, grid_res)
    grid_pc1, grid_pc2 = np.meshgrid(pc1_range, pc2_range)

    logging.info("Evaluating B on grid...")
    B_grid = evaluate_B_on_grid(
        cbf_net, pca, grid_pc1, grid_pc2, z_sub.mean(axis=0), device,
        model_type=model_type, obs_xy_val=obs_xy_val
    )

    # Clamp for visualization
    B_clamp = np.clip(B_grid, -5, 5)
    B_data_clamp = np.clip(B_data, -5, 5)

    title_suffix = ""
    if model_type == 'multipos' and obs_xy_val is not None:
        title_suffix = f" | obs=({obs_xy_val[0]:.1f}, {obs_xy_val[1]:.1f})"

    # =====================================================================
    # Plot 1: 3D Surface
    # =====================================================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_pc1, grid_pc2, B_clamp,
                           cmap='RdYlGn', alpha=0.7,
                           edgecolor='none')
    # Add B=0 plane
    ax.plot_surface(grid_pc1, grid_pc2, np.zeros_like(grid_pc1),
                    alpha=0.2, color='gray')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('B(z)', fontsize=12)
    ax.set_title(f'Learned CBF Surface — B(z){title_suffix}', fontsize=14)
    fig.colorbar(surf, shrink=0.5, label='B value')
    plt.tight_layout()
    fname = os.path.join(output_dir, f'cbf_3d_surface_{model_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fname}")

    # =====================================================================
    # Plot 2: Contour plot with B=0 boundary
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    contour = ax.contourf(grid_pc1, grid_pc2, B_clamp, levels=50,
                          cmap='RdYlGn', alpha=0.8)
    # B=0 contour (the safety boundary)
    ax.contour(grid_pc1, grid_pc2, B_grid, levels=[0], colors='black',
               linewidths=2, linestyles='solid')

    # Overlay training data
    safe_mask = labels_sub == 0
    unsafe_mask = labels_sub == 1
    ax.scatter(z_pca[safe_mask, 0], z_pca[safe_mask, 1],
               c='green', s=3, alpha=0.15, label=f'Safe ({safe_mask.sum()})')
    ax.scatter(z_pca[unsafe_mask, 0], z_pca[unsafe_mask, 1],
               c='red', s=3, alpha=0.15, label=f'Unsafe ({unsafe_mask.sum()})')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'CBF Contour Map — B=0 boundary (black line){title_suffix}', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    fig.colorbar(contour, label='B value')
    plt.tight_layout()
    fname = os.path.join(output_dir, f'cbf_contour_{model_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fname}")

    # =====================================================================
    # Plot 3: B value distribution (histogram)
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    B_safe = B_data[labels_sub == 0]
    B_unsafe = B_data[labels_sub == 1]

    axes[0].hist(B_safe, bins=100, color='green', alpha=0.7, label='Safe')
    axes[0].hist(B_unsafe, bins=100, color='red', alpha=0.7, label='Unsafe')
    axes[0].axvline(x=0, color='black', linewidth=2, linestyle='--', label='B=0')
    axes[0].set_xlabel('B(z)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('B Value Distribution', fontsize=14)
    axes[0].legend()

    # Zoomed view near boundary
    near_boundary = (B_data > -3) & (B_data < 3)
    axes[1].hist(B_safe[B_safe > -3][B_safe[B_safe > -3] < 3],
                 bins=80, color='green', alpha=0.7, label='Safe')
    axes[1].hist(B_unsafe[B_unsafe > -3][B_unsafe[B_unsafe > -3] < 3],
                 bins=80, color='red', alpha=0.7, label='Unsafe')
    axes[1].axvline(x=0, color='black', linewidth=2, linestyle='--', label='B=0')
    axes[1].set_xlabel('B(z)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('B Value Distribution (zoomed near B=0)', fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    fname = os.path.join(output_dir, f'cbf_histogram_{model_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fname}")

    # Log statistics
    logging.info(f"\nB value statistics:")
    logging.info(f"  Safe:   mean={B_safe.mean():.3f}, std={B_safe.std():.3f}, "
                 f"min={B_safe.min():.3f}, max={B_safe.max():.3f}")
    logging.info(f"  Unsafe: mean={B_unsafe.mean():.3f}, std={B_unsafe.std():.3f}, "
                 f"min={B_unsafe.min():.3f}, max={B_unsafe.max():.3f}")
    logging.info(f"  Safe correctly classified (B≥0): "
                 f"{(B_safe >= 0).sum()}/{len(B_safe)} ({(B_safe >= 0).mean()*100:.1f}%)")
    logging.info(f"  Unsafe correctly classified (B<0): "
                 f"{(B_unsafe < 0).sum()}/{len(B_unsafe)} ({(B_unsafe < 0).mean()*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Visualize learned CBF')
    parser.add_argument('--model', type=str, default='fixed',
                        choices=['fixed', 'multipos', 'planning'],
                        help='Which model to visualize')
    parser.add_argument('--obs_xy', type=float, nargs=2, default=None,
                        help='For multipos: obstacle (x, y) to visualize. '
                             'If not specified, generates plots for all positions.')
    parser.add_argument('--obs_xyhr', type=float, nargs=4, default=None,
                        help='For planning: obstacle [x, y, h, r] to visualize. '
                             'Default: [0.5, 0.0, 0.75, 0.1]')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    if args.output_dir is None:
        args.output_dir = os.path.join(cfg.PROJECT_ROOT, f'cbf_visualizations_{args.model}')

    cbf_net, z_data, labels, obs_xy = load_model_and_data(args.model, device)

    if args.model == 'fixed':
        create_visualizations(
            cbf_net, z_data, labels, obs_xy, 'fixed', device,
            args.output_dir, max_samples=args.max_samples
        )

    elif args.model == 'multipos':
        if args.obs_xy is not None:
            # Single position
            create_visualizations(
                cbf_net, z_data, labels, obs_xy, 'multipos', device,
                args.output_dir, obs_xy_val=args.obs_xy,
                max_samples=args.max_samples
            )
        else:
            # All positions
            for pos in mcfg.OBSTACLE_POSITIONS:
                logging.info(f"\n{'='*50}")
                logging.info(f"Visualizing position: {pos}")
                logging.info(f"{'='*50}")
                sub_dir = os.path.join(args.output_dir, f'obs_{pos[0]:.1f}_{pos[1]:.1f}')
                create_visualizations(
                    cbf_net, z_data, labels, obs_xy, 'multipos', device,
                    sub_dir, obs_xy_val=pos, max_samples=args.max_samples
                )

    elif args.model == 'planning':
        obs_val = args.obs_xyhr if args.obs_xyhr else [0.5, 0.0, 0.75, 0.1]
        logging.info(f"Visualizing planning model with obs={obs_val}")
        create_visualizations(
            cbf_net, z_data, labels, obs_xy, 'planning', device,
            args.output_dir, obs_xy_val=obs_val,
            max_samples=args.max_samples
        )

    logging.info(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
