"""
Interactive 3D CBF Visualization using Plotly.

Generates an interactive HTML file with:
    1. 3D surface of B(z) over PCA-projected latent space
    2. Scatter overlay of training data (safe=green, unsafe=red)
    3. B=0 boundary surface (gray plane)

Open the HTML in a browser to rotate, zoom, and hover.

Usage:
    python cbf_visualize_interactive.py --model planning
    python cbf_visualize_interactive.py --model fixed
    python cbf_visualize_interactive.py --model multipos --obs_xy 0.5 0.0
"""

import argparse
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Model definitions
# =============================================================================
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


class MultiPosBarrierNet(nn.Module):
    def __init__(self, latent_dim=7, obs_xy_dim=2, hidden_units=2048, num_hidden=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_xy_dim = obs_xy_dim
        self.fc_in = nn.Linear(latent_dim + obs_xy_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)])
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z, obs_xy):
        x = torch.cat([z.view(-1, self.latent_dim), obs_xy.view(-1, self.obs_xy_dim)], dim=-1)
        h = F.elu(self.fc_in(x))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


def load_model_and_data(model_type, device):
    if model_type == 'fixed':
        cbf_net = FixedBarrierNet(fcfg.LATENT_DIM, fcfg.CBF_HIDDEN_UNITS, fcfg.CBF_NUM_HIDDEN)
        ckpt = torch.load(fcfg.FIXED_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        data = torch.load(fcfg.FIXED_STATE_LABELS_TRAIN, weights_only=False)
        obs_data = None
        logging.info(f"Loaded FixedBarrierNet (epoch {ckpt['epoch']})")

    elif model_type == 'multipos':
        cbf_net = MultiPosBarrierNet(mcfg.LATENT_DIM, mcfg.OBS_XY_DIM,
                                     mcfg.CBF_HIDDEN_UNITS, mcfg.CBF_NUM_HIDDEN)
        ckpt = torch.load(mcfg.MULTIPOS_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        data = torch.load(mcfg.MULTIPOS_STATE_LABELS_TRAIN, weights_only=False)
        obs_data = data.get('obs_xy')
        logging.info(f"Loaded MultiPosBarrierNet (epoch {ckpt['epoch']})")

    elif model_type == 'planning':
        cbf_net = BarrierNet(cfg.LATENT_DIM, cfg.OBS_DIM,
                             cfg.CBF_HIDDEN_UNITS, cfg.CBF_NUM_HIDDEN)
        ckpt = torch.load(cfg.CBF_BEST_CHECKPOINT, map_location=device, weights_only=False)
        cbf_net.load_state_dict(ckpt['model_state_dict'])
        data = torch.load(cfg.STATE_LABELS_TRAIN, weights_only=False)
        obs_data = data.get('obs')
        logging.info(f"Loaded BarrierNet B(z,obs) (epoch {ckpt['epoch']})")

    else:
        raise ValueError(f"Unknown: {model_type}")

    cbf_net.to(device)
    cbf_net.eval()
    return cbf_net, data['z'], data['label'], obs_data


def main():
    parser = argparse.ArgumentParser(description='Interactive 3D CBF Visualization')
    parser.add_argument('--model', type=str, default='planning',
                        choices=['fixed', 'multipos', 'planning'])
    parser.add_argument('--obs_xy', type=float, nargs=2, default=None)
    parser.add_argument('--obs_xyhr', type=float, nargs=4, default=None)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--grid_res', type=int, default=80,
                        help='Grid resolution for surface (higher = smoother)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Lazy import plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logging.error("plotly not installed. Run: pip install plotly")
        return

    # Load model and data
    cbf_net, z_data, labels, obs_data = load_model_and_data(args.model, device)

    # Determine obs value for conditioned models
    obs_val = None
    if args.model == 'multipos':
        obs_val = args.obs_xy if args.obs_xy else [0.5, 0.0]
    elif args.model == 'planning':
        obs_val = args.obs_xyhr if args.obs_xyhr else [0.5, 0.0, 0.75, 0.1]

    # Subsample
    n = len(z_data)
    if n > args.max_samples:
        idx = np.random.choice(n, args.max_samples, replace=False)
        z_sub = z_data[idx].numpy()
        labels_sub = labels[idx].numpy()
    else:
        z_sub = z_data.numpy()
        labels_sub = labels.numpy()
        idx = np.arange(n)

    # PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_sub)
    logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
    logging.info(f"  Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Evaluate B on data points
    with torch.no_grad():
        z_tensor = torch.tensor(z_sub, dtype=torch.float32).to(device)
        if args.model == 'fixed':
            B_data = cbf_net(z_tensor).cpu().numpy()
        else:
            obs_t = torch.tensor(obs_val, dtype=torch.float32).unsqueeze(0)
            obs_t = obs_t.expand(z_tensor.shape[0], -1).to(device)
            B_data = cbf_net(z_tensor, obs_t).cpu().numpy()

    # Build grid
    margin = 0.5
    pc1_range = np.linspace(z_pca[:, 0].min() - margin, z_pca[:, 0].max() + margin, args.grid_res)
    pc2_range = np.linspace(z_pca[:, 1].min() - margin, z_pca[:, 1].max() + margin, args.grid_res)
    grid_pc1, grid_pc2 = np.meshgrid(pc1_range, pc2_range)

    logging.info(f"Evaluating B on {args.grid_res}x{args.grid_res} grid...")
    B_grid = np.zeros_like(grid_pc1)
    with torch.no_grad():
        for i in range(args.grid_res):
            pc_coords = np.stack([grid_pc1[i, :], grid_pc2[i, :]], axis=1)
            z_7d = pca.inverse_transform(pc_coords)
            z_t = torch.tensor(z_7d, dtype=torch.float32).to(device)
            if args.model == 'fixed':
                B_vals = cbf_net(z_t)
            else:
                obs_t = torch.tensor(obs_val, dtype=torch.float32).unsqueeze(0)
                obs_t = obs_t.expand(z_t.shape[0], -1).to(device)
                B_vals = cbf_net(z_t, obs_t)
            B_grid[i, :] = B_vals.cpu().numpy()

    B_clamp = np.clip(B_grid, -5, 5)

    # =========================================================================
    # Build interactive Plotly figure
    # =========================================================================
    logging.info("Building interactive plot...")

    fig = go.Figure()

    # 1. CBF Surface
    fig.add_trace(go.Surface(
        x=grid_pc1, y=grid_pc2, z=B_clamp,
        colorscale='RdYlGn',
        opacity=0.75,
        name='B(z) Surface',
        colorbar=dict(title='B value', x=1.02),
        showlegend=True,
    ))

    # 2. B=0 plane (safety boundary)
    fig.add_trace(go.Surface(
        x=grid_pc1, y=grid_pc2,
        z=np.zeros_like(grid_pc1),
        colorscale=[[0, 'rgba(128,128,128,0.3)'], [1, 'rgba(128,128,128,0.3)']],
        showscale=False,
        name='B=0 Boundary',
        showlegend=True,
    ))

    # 3. Safe data points
    safe_mask = labels_sub == 0
    B_safe_clamp = np.clip(B_data[safe_mask], -5, 5)
    fig.add_trace(go.Scatter3d(
        x=z_pca[safe_mask, 0],
        y=z_pca[safe_mask, 1],
        z=B_safe_clamp,
        mode='markers',
        marker=dict(size=1.5, color='green', opacity=0.3),
        name=f'Safe ({safe_mask.sum()})',
    ))

    # 4. Unsafe data points
    unsafe_mask = labels_sub == 1
    B_unsafe_clamp = np.clip(B_data[unsafe_mask], -5, 5)
    fig.add_trace(go.Scatter3d(
        x=z_pca[unsafe_mask, 0],
        y=z_pca[unsafe_mask, 1],
        z=B_unsafe_clamp,
        mode='markers',
        marker=dict(size=1.5, color='red', opacity=0.3),
        name=f'Unsafe ({unsafe_mask.sum()})',
    ))

    # Layout
    obs_str = ""
    if obs_val:
        obs_str = f" | obs={obs_val}"

    fig.update_layout(
        title=dict(
            text=f'Learned CBF: {args.model.upper()} model{obs_str}',
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='B(z)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        width=1200,
        height=800,
        legend=dict(x=0.01, y=0.99),
    )

    # Save
    if args.output is None:
        out_dir = os.path.join(cfg.PROJECT_ROOT, f'cbf_visualizations_{args.model}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'cbf_interactive_3d_{args.model}.html')
    else:
        out_path = args.output

    fig.write_html(out_path)
    logging.info(f"\nInteractive plot saved to: {out_path}")
    logging.info("Opening interactive window...")
    fig.show()

    # Also print B statistics
    B_safe = B_data[safe_mask]
    B_unsafe = B_data[unsafe_mask]
    logging.info(f"\nB statistics:")
    logging.info(f"  Safe:   mean={B_safe.mean():.3f}, std={B_safe.std():.3f}")
    logging.info(f"  Unsafe: mean={B_unsafe.mean():.3f}, std={B_unsafe.std():.3f}")
    logging.info(f"  Safe acc (B≥0):   {(B_safe >= 0).mean()*100:.1f}%")
    logging.info(f"  Unsafe acc (B<0): {(B_unsafe < 0).mean()*100:.1f}%")


if __name__ == '__main__':
    main()
