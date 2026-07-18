"""
CBF v2 Training — Improved loss functions for better barrier landscapes.

Three loss modes:
    1. 'sdf'              — MSE regression on signed distance (continuous B)
    2. 'margin_quadratic'  — Original margin + quadratic penalty (keeps pushing)
    3. 'margin'            — Original margin only (v1 baseline)

All modes include the decrease condition on transition data.

Usage:
    python cbf_v2_train.py --loss_type sdf
    python cbf_v2_train.py --loss_type margin_quadratic
    python cbf_v2_train.py --loss_type margin
"""

from __future__ import print_function
import argparse
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import cbf_v2_config as v2cfg

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# =============================================================================
# BarrierNet B(z, o) — same architecture as v1
# =============================================================================
class BarrierNetV2(nn.Module):
    """B(z, o) → scalar ∈ (-∞, ∞). Input: [z(7), obs(4)] = 11D."""

    def __init__(self, latent_dim=7, obs_dim=4, hidden_units=2048, num_hidden=4):
        super().__init__()
        self.input_dim = latent_dim + obs_dim
        self.fc_in = nn.Linear(self.input_dim, hidden_units)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)])
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z, obs):
        x = torch.cat([z, obs], dim=-1)
        h = F.elu(self.fc_in(x))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


# =============================================================================
# Datasets
# =============================================================================
class StateLabelDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.z = data['z'].float()
        self.obs = data['obs'].float()
        self.label = data['label'].float()
        self.sdf = data['sdf'].float()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.obs[idx], self.label[idx], self.sdf[idx]


class TransitionDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.z_k = data['z_k'].float()
        self.z_nom = data['z_nom'].float()
        self.obs = data['obs'].float()
        self.label_k = data['label_k'].float()
        self.label_nom = data['label_nom'].float()
        self.sdf_k = data['sdf_k'].float()
        self.sdf_nom = data['sdf_nom'].float()

    def __len__(self):
        return len(self.z_k)

    def __getitem__(self, idx):
        return (self.z_k[idx], self.z_nom[idx], self.obs[idx],
                self.label_k[idx], self.label_nom[idx],
                self.sdf_k[idx], self.sdf_nom[idx])


# =============================================================================
# Loss Functions
# =============================================================================
def compute_sdf_loss(B, sdf, label):
    """
    SDF Regression: train B to match the signed distance.
    B should approximate sdf everywhere, giving smooth gradients.
    """
    loss_mse = F.mse_loss(B, sdf)

    # Additional: ensure correct sign (classification accuracy)
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)

    sign_violations = 0.0
    if safe_mask.sum() > 0:
        sign_violations += F.relu(-B[safe_mask]).mean()  # B should be ≥ 0
    if unsafe_mask.sum() > 0:
        sign_violations += F.relu(B[unsafe_mask]).mean()  # B should be < 0

    return loss_mse + 0.5 * sign_violations


def compute_margin_quadratic_loss(B, label, gamma):
    """
    Improved margin loss: relu + quadratic penalty.
    The quadratic term keeps pushing B away from the boundary
    even after passing the margin threshold.
    """
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)
    loss = torch.tensor(0.0, device=B.device)

    if safe_mask.sum() > 0:
        B_safe = B[safe_mask]
        # Standard margin + gentle quadratic pull
        loss_s = F.relu(-B_safe + gamma).mean()
        loss_s += 0.05 * (F.relu(-B_safe + gamma) ** 2).mean()
        loss = loss + loss_s

    if unsafe_mask.sum() > 0:
        B_unsafe = B[unsafe_mask]
        # Standard margin + STRONG quadratic push to go more negative
        loss_u = F.relu(B_unsafe + gamma).mean()
        loss_u += 0.2 * ((B_unsafe + gamma) ** 2).mean()  # always active, not just relu
        loss = loss + loss_u

    return loss


def compute_margin_loss(B, label, gamma):
    """Original v1 margin loss (baseline for comparison)."""
    safe_mask = (label == 0)
    unsafe_mask = (label == 1)
    loss = torch.tensor(0.0, device=B.device)

    if safe_mask.sum() > 0:
        loss = loss + F.relu(-B[safe_mask] + gamma).mean()
    if unsafe_mask.sum() > 0:
        loss = loss + F.relu(B[unsafe_mask] + gamma).mean()

    return loss


def compute_decrease_loss(net, z_k, z_nom, obs, alpha, dt):
    """
    Transition decrease condition: B(z_nom) ≥ (1 - α·dt) · B(z_k).
    Applies only when z_k is in the safe set (B(z_k) > 0).
    """
    B_k = net(z_k, obs)
    B_nom = net(z_nom, obs)
    B_target = (1 - alpha * dt) * B_k

    # Only enforce when z_k is safe
    safe_mask = (B_k.detach() > 0)
    if safe_mask.sum() == 0:
        return torch.tensor(0.0, device=z_k.device)

    violation = F.relu(B_target[safe_mask] - B_nom[safe_mask])
    return violation.mean()


# =============================================================================
# Training
# =============================================================================
def train_epoch(net, optimizer, state_loader, trans_loader, device, args):
    net.train()
    total_loss = 0.0
    total_spatial = 0.0
    total_decrease = 0.0
    n_batches = 0

    # Interleave state and transition batches
    trans_iter = iter(trans_loader) if trans_loader else None

    for state_batch in state_loader:
        z, obs, label, sdf = [x.to(device) for x in state_batch]

        B = net(z, obs)

        # Spatial loss
        if args.loss_type == 'sdf':
            L_spatial = compute_sdf_loss(B, sdf, label)
        elif args.loss_type == 'margin_quadratic':
            L_spatial = compute_margin_quadratic_loss(B, label, args.gamma)
        else:
            L_spatial = compute_margin_loss(B, label, args.gamma)

        # Decrease loss from transitions
        L_decrease = torch.tensor(0.0, device=device)
        if trans_iter is not None:
            try:
                trans_batch = next(trans_iter)
            except StopIteration:
                trans_iter = iter(trans_loader)
                trans_batch = next(trans_iter)
            z_k, z_nom, obs_t, _, _, _, _ = [x.to(device) for x in trans_batch]
            L_decrease = compute_decrease_loss(
                net, z_k, z_nom, obs_t, args.alpha, args.dt)

        L_total = L_spatial + args.lambda_decrease * L_decrease

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        total_loss += L_total.item()
        total_spatial += L_spatial.item()
        total_decrease += L_decrease.item()
        n_batches += 1

    return {
        'total': total_loss / n_batches,
        'spatial': total_spatial / n_batches,
        'decrease': total_decrease / n_batches,
    }


def validate(net, loader, device, args):
    net.eval()
    correct = 0
    total = 0
    safe_correct = 0
    safe_total = 0
    unsafe_correct = 0
    unsafe_total = 0
    sdf_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for z, obs, label, sdf in loader:
            z, obs, label, sdf = z.to(device), obs.to(device), label.to(device), sdf.to(device)
            B = net(z, obs)

            pred = (B >= 0).float()
            gt = (label == 0).float()  # safe=1, unsafe=0
            correct += (pred == gt).sum().item()
            total += len(z)

            safe_mask = (label == 0)
            unsafe_mask = (label == 1)
            if safe_mask.sum() > 0:
                safe_correct += (B[safe_mask] >= 0).sum().item()
                safe_total += safe_mask.sum().item()
            if unsafe_mask.sum() > 0:
                unsafe_correct += (B[unsafe_mask] < 0).sum().item()
                unsafe_total += unsafe_mask.sum().item()

            sdf_mse += F.mse_loss(B, sdf).item()
            n_batches += 1

    return {
        'accuracy': correct / max(total, 1),
        'safe_acc': safe_correct / max(safe_total, 1),
        'unsafe_acc': unsafe_correct / max(unsafe_total, 1),
        'sdf_mse': sdf_mse / max(n_batches, 1),
        'unsafe_mean_B': 0.0,  # computed below if needed
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='CBF v2 Training')
    parser.add_argument('--loss_type', type=str, default=v2cfg.DEFAULT_LOSS_TYPE,
                        choices=['sdf', 'margin_quadratic', 'margin'])
    parser.add_argument('--epochs', type=int, default=v2cfg.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=v2cfg.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=v2cfg.LR)
    parser.add_argument('--gamma', type=float, default=v2cfg.SAFETY_MARGIN)
    parser.add_argument('--alpha', type=float, default=v2cfg.CBF_ALPHA)
    parser.add_argument('--dt', type=float, default=v2cfg.CBF_DT)
    parser.add_argument('--lambda_decrease', type=float, default=1.0)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=v2cfg.SEED)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Loss-type-specific save paths so all 3 can train in parallel
    snapshot_dir = os.path.join(v2cfg.SNAPSHOT_DIR, args.loss_type)
    best_checkpoint = os.path.join(snapshot_dir, 'barrier_net_best.pt')
    os.makedirs(snapshot_dir, exist_ok=True)
    logging.info(f"Checkpoints → {snapshot_dir}")

    # Load data
    logging.info("Loading datasets...")
    train_state = StateLabelDataset(v2cfg.STATE_LABELS_TRAIN)
    val_state = StateLabelDataset(v2cfg.STATE_LABELS_VAL)
    state_loader = DataLoader(train_state, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_state, batch_size=args.batch_size, shuffle=False)

    # Transition data (optional)
    try:
        train_trans = TransitionDataset(v2cfg.TRANSITION_TRAIN)
        trans_loader = DataLoader(train_trans, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
        logging.info(f"Transition data: {len(train_trans)} samples")
    except Exception as e:
        logging.warning(f"No transition data: {e}")
        trans_loader = None

    n_safe = (train_state.label == 0).sum().item()
    n_unsafe = (train_state.label == 1).sum().item()
    logging.info(f"State data: {len(train_state)} train, {len(val_state)} val")
    logging.info(f"  Safe: {n_safe} | Unsafe: {n_unsafe}")
    logging.info(f"  SDF range: [{train_state.sdf.min():.3f}, {train_state.sdf.max():.3f}]")

    # Model
    net = BarrierNetV2(v2cfg.LATENT_DIM, v2cfg.OBS_DIM,
                       v2cfg.CBF_HIDDEN_UNITS, v2cfg.CBF_NUM_HIDDEN)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    param_count = sum(p.numel() for p in net.parameters())
    logging.info(f"\nBarrierNetV2: {param_count:,} parameters")
    logging.info(f"Loss type: {args.loss_type}")
    logging.info(f"γ={args.gamma}, α={args.alpha}, λ_decrease={args.lambda_decrease}")

    best_acc = 0.0
    best_epoch = 0

    logging.info(f"\n{'=' * 70}")
    logging.info(f"Starting CBF v2 training [{args.loss_type}] for {args.epochs} epochs")
    logging.info(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(net, optimizer, state_loader, trans_loader,
                                     device, args)
        scheduler.step()

        if epoch % args.log_interval == 0:
            val_metrics = validate(net, val_loader, device, args)

            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_type': args.loss_type,
                    'val_accuracy': val_metrics['accuracy'],
                    'sdf_mse': val_metrics['sdf_mse'],
                }, best_checkpoint)

            logging.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Loss={train_metrics['total']:.4f} "
                f"(spatial={train_metrics['spatial']:.4f} "
                f"decrease={train_metrics['decrease']:.4f}) | "
                f"Val Acc={val_metrics['accuracy']*100:.1f}% "
                f"(safe={val_metrics['safe_acc']*100:.1f}% "
                f"unsafe={val_metrics['unsafe_acc']*100:.1f}%) "
                f"SDF_MSE={val_metrics['sdf_mse']:.4f}"
            )

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'loss_type': args.loss_type,
            }, os.path.join(snapshot_dir, f'barrier_epoch{epoch}.pt'))

    logging.info(f"\n{'=' * 70}")
    logging.info(f"Training complete! Best: epoch {best_epoch}, "
                 f"val_acc={best_acc*100:.1f}%")
    logging.info(f"Loss type: {args.loss_type}")
    logging.info(f"Saved to: {best_checkpoint}")
    logging.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
