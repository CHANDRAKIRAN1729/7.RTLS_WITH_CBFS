"""
CBF Model — Neural Control Barrier Function B_θ(z, o).

Defines the barrier network and the CBF2 safety correction function.

References:
    - CBF1.pdf Eqs 1-3: CBF definition (B ≥ 0 safe, B < 0 unsafe)
    - CBF2.pdf Eqs 3-10: Safe latent update derivation
    - TrainingCBF.pdf Section 4: Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BarrierNet(nn.Module):
    """
    Neural Control Barrier Function B_θ(z, o).

    Input:  concat(z, o) where z is the latent code and o is the obstacle descriptor.
    Output: Scalar barrier value B (signed, unbounded).
            B(z, o) ≥ 0  →  Safe
            B(z, o) < 0   →  Unsafe

    Architecture mirrors the classifier head in vae_obs.py (fc32 → fc_obs → fc42)
    but is a standalone module with no sigmoid — output is the raw signed barrier value.
    """

    def __init__(self, latent_dim=7, obs_dim=4, hidden_units=2048, num_hidden=4):
        super(BarrierNet, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Input layer: concat(z, o) → hidden
        self.fc_in = nn.Linear(latent_dim + obs_dim, hidden_units)

        # Hidden layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(num_hidden - 1)]
        )

        # Output layer: hidden → scalar barrier value
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, z, obs):
        """
        Compute barrier value B(z, o).

        Args:
            z:   (batch, latent_dim) latent codes
            obs: (batch, obs_dim) obstacle descriptors [x, y, h, r]

        Returns:
            B: (batch,) scalar barrier values
        """
        x = torch.cat([z.view(-1, self.latent_dim), obs.view(-1, self.obs_dim)], dim=-1)
        h = F.elu(self.fc_in(x))
        for fc in self.fc_hidden:
            h = F.elu(fc(h))
        return self.fc_out(h).view(-1)


def cbf_safety_correction(cbf_net, z_current, z_nominal, obs, alpha, delta_t,
                          lambda_max=1.0, safe_threshold=None, max_iters=5):
    """
    CBF2.pdf: Compute safe latent update via ITERATIVE gradient projection.

    The closed-form correction (CBF2 Eqs 4,9,10) uses a first-order Taylor
    approximation of B. For neural network B, this linearization is inexact.
    We iterate the correction and VERIFY B(z_safe) after each step.

    Algorithm:
        1. Compute target = (1-αΔ)·B(z_k)
        2. Set z_candidate = z_nom
        3. For each iteration:
            a. Evaluate B(z_candidate) and ∇B(z_candidate)
            b. If B(z_candidate) ≥ target → ACCEPT (constraint satisfied)
            c. Else: compute λ = (target - B_candidate) / ||∇B||²
            d. Update: z_candidate += λ·∇B
        4. If all iterations fail → REJECT: z_safe = z_k (stay in place)
        5. Hard safety guard: if B(z_safe) < 0 → REJECT (never enter unsafe set)

    Args:
        cbf_net:    Trained BarrierNet
        z_current:  (1, latent_dim) tensor — z_k
        z_nominal:  (1, latent_dim) tensor — z_{k+1}^nom (detached)
        obs:        (1, obs_dim) tensor — obstacle parameters
        alpha:      float — barrier decay rate
        delta_t:    float — time step
        lambda_max: float — max λ per iteration (caps correction magnitude)
        safe_threshold: float or None — skip correction if B(z_nom) > threshold
        max_iters:  int — maximum correction iterations (default 5)

    Returns:
        z_safe:     (1, latent_dim) tensor — z_{k+1}^safe
        info:       dict with correction diagnostics
    """
    # Evaluate B at current state (no grad needed)
    with torch.no_grad():
        B_current = cbf_net(z_current.detach(), obs)    # B(z_k)

    # Target barrier value — CBF2 Eq 1
    B_target = (1.0 - alpha * delta_t) * B_current      # (1-αΔ)·B(z_k)

    # Quick check: is the nominal step already safe enough?
    with torch.no_grad():
        B_nom_check = cbf_net(z_nominal.detach(), obs)

    if safe_threshold is not None and B_nom_check.item() >= safe_threshold:
        return z_nominal.detach().clone(), {
            'lambda_val': 0.0, 'B_current': B_current.item(),
            'B_nominal': B_nom_check.item(), 'B_target': B_target.item(),
            'B_safe': B_nom_check.item(), 'correction_applied': False,
            'iters_used': 0, 'constraint_satisfied': True, 'rejected': False,
            'grad_norm': 0.0,
        }

    # --- Iterative correction ---
    z_candidate = z_nominal.detach().clone()
    total_lambda = 0.0
    iters_used = 0
    constraint_satisfied = False
    B_nom_val = B_nom_check.item()
    grad_norm_val = 0.0

    for i in range(max_iters):
        # Evaluate B and ∇B at current candidate
        z_c = z_candidate.detach().clone().requires_grad_(True)
        B_c = cbf_net(z_c, obs)

        # Check: is the constraint satisfied?
        if B_c.item() >= B_target.item():
            z_candidate = z_c.detach()
            constraint_satisfied = True
            iters_used = i
            break

        # Not satisfied — compute gradient correction (CBF2 Eqs 3,9,10)
        grad_B = torch.autograd.grad(
            B_c, z_c, grad_outputs=torch.ones_like(B_c),
            create_graph=False, retain_graph=False
        )[0]

        d_norm_sq = torch.sum(grad_B ** 2) + 1e-8
        grad_norm_val = torch.sqrt(d_norm_sq).item()
        lambda_linear = ((B_target - B_c) / d_norm_sq).item()
        lambda_val = min(max(0.0, lambda_linear), lambda_max)

        if lambda_val == 0.0:
            # Can't improve — gradient points wrong way
            iters_used = i + 1
            break

        # Apply correction
        z_candidate = z_c.detach() + lambda_val * grad_B.detach()
        total_lambda += lambda_val
        iters_used = i + 1

    # --- Hard safety guard ---
    # If B(z_safe) < 0 but B(z_k) >= 0, the step would enter the unsafe set.
    # Reject the step entirely — stay at z_k to maintain safe set invariance.
    rejected = False
    with torch.no_grad():
        B_safe_final = cbf_net(z_candidate.detach(), obs)

    if B_safe_final.item() < 0.0 and B_current.item() >= 0.0:
        # Current state is safe but corrected state is unsafe → REJECT
        z_candidate = z_current.detach().clone()
        rejected = True
        B_safe_final = B_current

    info = {
        'lambda_val': total_lambda,
        'B_current': B_current.item(),
        'B_nominal': B_nom_val,
        'B_target': B_target.item(),
        'B_safe': B_safe_final.item(),
        'correction_applied': total_lambda > 0.0 or rejected,
        'iters_used': iters_used,
        'constraint_satisfied': constraint_satisfied,
        'rejected': rejected,
        'grad_norm': grad_norm_val,
    }

    return z_candidate.detach(), info

