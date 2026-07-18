"""
CBF v2 Configuration — Improved training with signed distance targets.
"""
import os
import cbf_config as cfg

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = os.path.join(cfg.PROJECT_ROOT, 'cbf_v2_data')
STATE_LABELS_TRAIN = os.path.join(DATA_DIR, 'state_labels_train.pt')
STATE_LABELS_VAL = os.path.join(DATA_DIR, 'state_labels_val.pt')
TRANSITION_TRAIN = os.path.join(DATA_DIR, 'transition_train.pt')
TRANSITION_VAL = os.path.join(DATA_DIR, 'transition_val.pt')

SNAPSHOT_DIR = os.path.join(cfg.PROJECT_ROOT, 'model_params/panda_10k/cbf_v2_snapshots')
BEST_CHECKPOINT = os.path.join(SNAPSHOT_DIR, 'barrier_net_best.pt')

# =============================================================================
# Architecture (same as original)
# =============================================================================
LATENT_DIM = cfg.LATENT_DIM      # 7
OBS_DIM = cfg.OBS_DIM            # 4
CBF_HIDDEN_UNITS = cfg.CBF_HIDDEN_UNITS  # 2048
CBF_NUM_HIDDEN = cfg.CBF_NUM_HIDDEN      # 4

# =============================================================================
# Data Generation
# =============================================================================
NUM_STATE_SCENARIOS = 15000
NUM_TRANSITION_SCENARIOS = 15000
VAL_SPLIT = 0.1

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE = 4096
EPOCHS = 3000
LR = 1e-4
SAFETY_MARGIN = 1.0            # γ for margin losses
CBF_ALPHA = 0.1                # decay rate
CBF_DT = 1.0

# Loss type: 'sdf', 'margin_quadratic', 'margin' (original)
DEFAULT_LOSS_TYPE = 'sdf'

# SDF regression parameters
SDF_CLIP = 5.0                 # clip signed distance to [-SDF_CLIP, SDF_CLIP]
SDF_SCALE = 1.0                # scale factor for SDF targets

SEED = cfg.SEED
