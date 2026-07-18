#!/bin/bash
# ============================================================================
# CBF Comprehensive Parameter Ablation Study
# ============================================================================
#
# Multi-stage ablation covering all tunable parameters in the CBF pipeline.
# Each experiment logs results to a unique JSON file for later analysis.
#
# Structure:
#   Stage 1: Inference-only sweeps (no retraining) — uses current checkpoint
#   Stage 2: Training parameter sweeps (retrain + evaluate each)
#   Stage 3: Grid search of top parameters from Stages 1 & 2
#
# Usage:
#   chmod +x cbf_ablation_study.sh
#   ./cbf_ablation_study.sh [STAGE]
#
#   STAGE: 1, 2, 3, or "all" (default: all)
#
# Total runtime estimates:
#   Stage 1: ~6-8 hours  (36 inference-only runs × ~15 min each)
#   Stage 2: ~40-60 hours (8 retrain runs × 3-5 hrs + eval)
#   Stage 3: ~10-15 hours (depends on Stage 1&2 results)
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SRC_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/ablation_results"
CHECKPOINT_DIR="${PROJECT_ROOT}/model_params/panda_10k/cbf_ablation"
TEST_SCENES="${PROJECT_ROOT}/model_params/panda_10k/test_scenes.json"
CURRENT_CHECKPOINT="${PROJECT_ROOT}/model_params/panda_10k/cbf_snapshots/barrier_net_best.pt"

# Reduced scenario count for ablation (200 for speed, final eval uses 1000)
ABLATION_PROBLEMS=200
FINAL_PROBLEMS=1000

# Defaults (baseline for single-variable sweeps)
DEFAULT_PLANNING_LR=0.03
DEFAULT_LAMBDA_PRIOR=0.01
DEFAULT_LAMBDA_MAX=1.0
DEFAULT_CBF_ALPHA=0.1
DEFAULT_CBF_DELTA_T=1.0
DEFAULT_CBF_MAX_ITERS=5
DEFAULT_MAX_STEPS=300

# Training defaults
DEFAULT_TRAIN_LR=1e-4
DEFAULT_SAFETY_MARGIN=1.0
DEFAULT_LAMBDA_SAFE=1.0
DEFAULT_LAMBDA_UNSAFE=1.0
DEFAULT_LAMBDA_DECREASE=1.0
DEFAULT_TRAIN_ALPHA=0.1
DEFAULT_TRAIN_DELTA_T=1.0
DEFAULT_EPOCHS=2000

STAGE="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_eval() {
    local exp_name="$1"
    local checkpoint="$2"
    shift 2
    local extra_args=("$@")

    local output_file="${RESULTS_DIR}/${exp_name}.json"

    if [[ -f "$output_file" ]]; then
        log "SKIP $exp_name (already exists)"
        return 0
    fi

    log "RUN  $exp_name"
    log "  Args: ${extra_args[*]}"

    cd "$SRC_DIR"
    python cbf_evaluate_planning.py \
        --load_scenes "$TEST_SCENES" \
        --cbf_checkpoint "$checkpoint" \
        --num_problems "$ABLATION_PROBLEMS" \
        --output "$output_file" \
        "${extra_args[@]}" \
        2>&1 | tee "${RESULTS_DIR}/logs/${exp_name}.log"

    log "DONE $exp_name → $output_file"
}

run_train() {
    local exp_name="$1"
    shift
    local extra_args=("$@")

    local save_dir="${CHECKPOINT_DIR}/${exp_name}"
    local checkpoint="${save_dir}/barrier_net_best.pt"

    if [[ -f "$checkpoint" ]]; then
        log "SKIP training $exp_name (checkpoint exists)" >&2
        echo "$checkpoint"
        return 0
    fi

    log "TRAIN $exp_name" >&2
    log "  Args: ${extra_args[*]}" >&2

    cd "$SRC_DIR"
    python cbf_train.py \
        --save_dir "$save_dir" \
        --save_every 100 \
        --log_interval 10 \
        "${extra_args[@]}" \
        2>&1 | tee "${RESULTS_DIR}/logs/train_${exp_name}.log" >&2

    log "TRAIN DONE $exp_name → $checkpoint" >&2
    echo "$checkpoint"
}

# ============================================================================
# Setup
# ============================================================================
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$CHECKPOINT_DIR"

log "============================================================"
log "CBF ABLATION STUDY — $TIMESTAMP"
log "============================================================"
log "Project:    $PROJECT_ROOT"
log "Results:    $RESULTS_DIR"
log "Checkpoints: $CHECKPOINT_DIR"
log "Test scenes: $TEST_SCENES"
log "Current CBF: $CURRENT_CHECKPOINT"
log "Ablation problems: $ABLATION_PROBLEMS"
log "Stage: $STAGE"
log "============================================================"

# ============================================================================
# STAGE 1: Inference-only parameter sweeps
# ============================================================================
# Uses the CURRENT trained checkpoint. No retraining needed.
# Each sweep changes ONE parameter while keeping others at default.
# ============================================================================

run_stage_1() {
    log ""
    log "################################################################"
    log "# STAGE 1: Inference-Only Parameter Sweeps"
    log "################################################################"
    log ""

    local ckpt="$CURRENT_CHECKPOINT"

    # -----------------------------------------------------------------------
    # 1A: Planning Learning Rate sweep
    # -----------------------------------------------------------------------
    log "--- 1A: Planning LR ---"
    for lr in 0.005 0.01 0.02 0.03 0.05 0.08 0.1; do
        run_eval "s1_plr_${lr}" "$ckpt" \
            --planning_lr "$lr" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 1B: Lambda Prior sweep
    # -----------------------------------------------------------------------
    log "--- 1B: Lambda Prior ---"
    for lp in 0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1.0; do
        run_eval "s1_lprior_${lp}" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$lp" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 1C: Lambda Max (CBF correction cap) sweep
    # -----------------------------------------------------------------------
    log "--- 1C: Lambda Max ---"
    for lm in 0.1 0.5 1.0 2.0 5.0 10.0 50.0; do
        run_eval "s1_lmax_${lm}" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$lm" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 1D: CBF Alpha (barrier decay rate) sweep
    # -----------------------------------------------------------------------
    log "--- 1D: CBF Alpha ---"
    for alpha in 0.001 0.01 0.05 0.1 0.2 0.5 0.9; do
        run_eval "s1_alpha_${alpha}" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$alpha" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 1E: CBF Delta T sweep
    # -----------------------------------------------------------------------
    log "--- 1E: CBF Delta T ---"
    for dt in 0.01 0.1 0.5 1.0 2.0 5.0; do
        run_eval "s1_dt_${dt}" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$dt" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 1F: CBF Max Iterations sweep
    # -----------------------------------------------------------------------
    log "--- 1F: CBF Max Iters ---"
    for iters in 1 2 3 5 10 20; do
        run_eval "s1_iters_${iters}" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$iters" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 1G: Max Planning Steps sweep
    # -----------------------------------------------------------------------
    log "--- 1G: Max Steps ---"
    for steps in 100 200 300 500 800; do
        run_eval "s1_steps_${steps}" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$steps"
    done

    log ""
    log "STAGE 1 COMPLETE — $(ls "${RESULTS_DIR}"/s1_*.json 2>/dev/null | wc -l) experiments done"
}


# ============================================================================
# STAGE 2: Training Parameter Sweeps
# ============================================================================
# Each experiment retrains the CBF with different hyperparameters,
# then evaluates using the default inference parameters.
# ============================================================================

run_stage_2() {
    log ""
    log "################################################################"
    log "# STAGE 2: Training Parameter Sweeps (Retrain + Evaluate)"
    log "################################################################"
    log ""

    # -----------------------------------------------------------------------
    # 2A: Safety Margin sweep (most impactful training param)
    # -----------------------------------------------------------------------
    log "--- 2A: Safety Margin ---"
    for margin in 0.1 0.5 1.0 2.0 5.0 10.0; do
        local exp_name="s2_margin_${margin}"
        local ckpt
        ckpt=$(run_train "$exp_name" \
            --epochs "$DEFAULT_EPOCHS" \
            --lr "$DEFAULT_TRAIN_LR" \
            --safety_margin "$margin" \
            --alpha "$DEFAULT_TRAIN_ALPHA" \
            --delta_t "$DEFAULT_TRAIN_DELTA_T" \
            --lambda_safe "$DEFAULT_LAMBDA_SAFE" \
            --lambda_unsafe "$DEFAULT_LAMBDA_UNSAFE" \
            --lambda_decrease "$DEFAULT_LAMBDA_DECREASE")

        run_eval "$exp_name" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 2B: Training Alpha sweep (affects decrease condition)
    # -----------------------------------------------------------------------
    log "--- 2B: Training Alpha ---"
    for alpha in 0.01 0.05 0.1 0.5; do
        local exp_name="s2_alpha_${alpha}"
        local ckpt
        ckpt=$(run_train "$exp_name" \
            --epochs "$DEFAULT_EPOCHS" \
            --lr "$DEFAULT_TRAIN_LR" \
            --safety_margin "$DEFAULT_SAFETY_MARGIN" \
            --alpha "$alpha" \
            --delta_t "$DEFAULT_TRAIN_DELTA_T" \
            --lambda_safe "$DEFAULT_LAMBDA_SAFE" \
            --lambda_unsafe "$DEFAULT_LAMBDA_UNSAFE" \
            --lambda_decrease "$DEFAULT_LAMBDA_DECREASE")

        # Evaluate with MATCHING alpha (train alpha = eval alpha)
        run_eval "$exp_name" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$alpha" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 2C: Training LR sweep
    # -----------------------------------------------------------------------
    log "--- 2C: Training LR ---"
    for tlr in 5e-5 1e-4 5e-4 1e-3; do
        local exp_name="s2_tlr_${tlr}"
        local ckpt
        ckpt=$(run_train "$exp_name" \
            --epochs "$DEFAULT_EPOCHS" \
            --lr "$tlr" \
            --safety_margin "$DEFAULT_SAFETY_MARGIN" \
            --alpha "$DEFAULT_TRAIN_ALPHA" \
            --delta_t "$DEFAULT_TRAIN_DELTA_T" \
            --lambda_safe "$DEFAULT_LAMBDA_SAFE" \
            --lambda_unsafe "$DEFAULT_LAMBDA_UNSAFE" \
            --lambda_decrease "$DEFAULT_LAMBDA_DECREASE")

        run_eval "$exp_name" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 2D: Loss Weight Balance sweep
    # -----------------------------------------------------------------------
    log "--- 2D: Loss Weights ---"
    # Format: safe_unsafe_decrease
    for weights in "1.0 1.0 0.1" "1.0 1.0 0.5" "1.0 1.0 1.0" "1.0 1.0 5.0" \
                   "2.0 1.0 1.0" "1.0 2.0 1.0" "5.0 1.0 1.0" "1.0 5.0 1.0"; do
        read -r ls lu ld <<< "$weights"
        local exp_name="s2_wt_${ls}_${lu}_${ld}"
        local ckpt
        ckpt=$(run_train "$exp_name" \
            --epochs "$DEFAULT_EPOCHS" \
            --lr "$DEFAULT_TRAIN_LR" \
            --safety_margin "$DEFAULT_SAFETY_MARGIN" \
            --alpha "$DEFAULT_TRAIN_ALPHA" \
            --delta_t "$DEFAULT_TRAIN_DELTA_T" \
            --lambda_safe "$ls" \
            --lambda_unsafe "$lu" \
            --lambda_decrease "$ld")

        run_eval "$exp_name" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    # -----------------------------------------------------------------------
    # 2E: Epoch Count sweep
    # -----------------------------------------------------------------------
    log "--- 2E: Epochs ---"
    for ep in 500 1000 2000 5000; do
        local exp_name="s2_ep_${ep}"
        local ckpt
        ckpt=$(run_train "$exp_name" \
            --epochs "$ep" \
            --lr "$DEFAULT_TRAIN_LR" \
            --safety_margin "$DEFAULT_SAFETY_MARGIN" \
            --alpha "$DEFAULT_TRAIN_ALPHA" \
            --delta_t "$DEFAULT_TRAIN_DELTA_T" \
            --lambda_safe "$DEFAULT_LAMBDA_SAFE" \
            --lambda_unsafe "$DEFAULT_LAMBDA_UNSAFE" \
            --lambda_decrease "$DEFAULT_LAMBDA_DECREASE")

        run_eval "$exp_name" "$ckpt" \
            --planning_lr "$DEFAULT_PLANNING_LR" \
            --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
            --lambda_max "$DEFAULT_LAMBDA_MAX" \
            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
            --max_steps "$DEFAULT_MAX_STEPS"
    done

    log ""
    log "STAGE 2 COMPLETE — $(ls "${RESULTS_DIR}"/s2_*.json 2>/dev/null | wc -l) experiments done"
}


# ============================================================================
# STAGE 3: Grid Search of Top Parameters
# ============================================================================
# Combine the best parameters from Stages 1 & 2 into focused grid searches.
# This stage requires running the analysis script first to pick top params.
# ============================================================================

run_stage_3() {
    log ""
    log "################################################################"
    log "# STAGE 3: Grid Search of Top Training × Inference Combos"
    log "################################################################"
    log ""

    # -----------------------------------------------------------------------
    # 3A: Margin × Alpha grid (training side)
    # -----------------------------------------------------------------------
    log "--- 3A: Margin × Alpha Grid ---"
    for margin in 0.5 1.0 2.0 5.0; do
        for alpha in 0.01 0.1 0.5; do
            local exp_name="s3_m${margin}_a${alpha}"
            local ckpt
            ckpt=$(run_train "$exp_name" \
                --epochs "$DEFAULT_EPOCHS" \
                --lr "$DEFAULT_TRAIN_LR" \
                --safety_margin "$margin" \
                --alpha "$alpha" \
                --delta_t "$DEFAULT_TRAIN_DELTA_T" \
                --lambda_safe "$DEFAULT_LAMBDA_SAFE" \
                --lambda_unsafe "$DEFAULT_LAMBDA_UNSAFE" \
                --lambda_decrease "$DEFAULT_LAMBDA_DECREASE")

            # Evaluate with matching alpha
            run_eval "$exp_name" "$ckpt" \
                --planning_lr "$DEFAULT_PLANNING_LR" \
                --lambda_prior "$DEFAULT_LAMBDA_PRIOR" \
                --lambda_max "$DEFAULT_LAMBDA_MAX" \
                --cbf_alpha "$alpha" \
                --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
                --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
                --max_steps "$DEFAULT_MAX_STEPS"
        done
    done

    # -----------------------------------------------------------------------
    # 3B: Best training model × inference param grid
    # -----------------------------------------------------------------------
    log "--- 3B: Best Model × Inference Sweep ---"
    # After 3A, find the best checkpoint and sweep inference params on it
    local best_3a_json
    best_3a_json=$(cd "$RESULTS_DIR" && python3 -c "
import json, glob
best_score = -1
best_file = ''
for f in glob.glob('s3_m*_a*.json'):
    with open(f) as fh:
        d = json.load(fh)
    sr = d.get('success_rate_percent', 0)
    cf = d.get('collision_free_rate_percent', 0)
    score = sr + cf
    if score > best_score:
        best_score = score
        best_file = f
print(best_file)
" 2>/dev/null || echo "")

    if [[ -n "$best_3a_json" && -f "${RESULTS_DIR}/${best_3a_json}" ]]; then
        # Extract checkpoint path from the best model name
        local best_3a_name="${best_3a_json%.json}"
        local best_3a_ckpt="${CHECKPOINT_DIR}/${best_3a_name}/barrier_net_best.pt"

        if [[ -f "$best_3a_ckpt" ]]; then
            log "Best 3A model: $best_3a_name"

            # Sweep inference params on this model
            for plr in 0.01 0.03 0.05; do
                for lp in 0.001 0.01 0.1; do
                    for lm in 0.5 1.0 5.0; do
                        local exp_name="s3_best_plr${plr}_lp${lp}_lm${lm}"
                        run_eval "$exp_name" "$best_3a_ckpt" \
                            --planning_lr "$plr" \
                            --lambda_prior "$lp" \
                            --lambda_max "$lm" \
                            --cbf_alpha "$DEFAULT_CBF_ALPHA" \
                            --cbf_delta_t "$DEFAULT_CBF_DELTA_T" \
                            --cbf_max_iters "$DEFAULT_CBF_MAX_ITERS" \
                            --max_steps "$DEFAULT_MAX_STEPS"
                    done
                done
            done
        fi
    else
        log "WARNING: No Stage 3A results found — skipping 3B"
    fi

    log ""
    log "STAGE 3 COMPLETE — $(ls "${RESULTS_DIR}"/s3_*.json 2>/dev/null | wc -l) experiments done"
}


# ============================================================================
# Results Analysis (runs after each stage)
# ============================================================================

analyze_results() {
    log ""
    log "################################################################"
    log "# RESULTS ANALYSIS"
    log "################################################################"
    log ""

    cd "$SRC_DIR"
    python3 << 'PYEOF'
import json, glob, os, sys

results_dir = os.environ.get('RESULTS_DIR', '')
if not results_dir:
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ablation_results')

files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
if not files:
    print("No results found")
    sys.exit(0)

print(f"\n{'Experiment':<45} {'Success%':>9} {'Goal%':>8} {'CollFree%':>10} {'AvgTime':>9}")
print("-" * 85)

rows = []
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        name = os.path.basename(f).replace('.json', '')
        sr = d.get('success_rate_percent', 0)
        gr = d.get('goal_reached_rate_percent', 0)
        cf = d.get('collision_free_rate_percent', 0)
        at = d.get('avg_planning_time_ms', 0)
        rows.append((name, sr, gr, cf, at))
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Sort by success rate descending
rows.sort(key=lambda x: x[1], reverse=True)

for name, sr, gr, cf, at in rows:
    marker = " ★" if sr > 50 else ""
    print(f"{name:<45} {sr:>8.1f}% {gr:>7.1f}% {cf:>9.1f}% {at:>8.1f}ms{marker}")

print(f"\nTotal experiments: {len(rows)}")
if rows:
    best = rows[0]
    print(f"\nBest experiment: {best[0]}")
    print(f"  Success: {best[1]:.1f}%, Goal: {best[2]:.1f}%, CollFree: {best[3]:.1f}%")
PYEOF
}


# ============================================================================
# Main Execution
# ============================================================================

export RESULTS_DIR

case "$STAGE" in
    1)
        run_stage_1
        analyze_results
        ;;
    2)
        run_stage_2
        analyze_results
        ;;
    3)
        run_stage_3
        analyze_results
        ;;
    all)
        run_stage_1
        analyze_results
        log ""
        log "Stage 1 complete. Review results above before starting Stage 2."
        log "To continue: $0 2"
        log ""
        read -p "Press Enter to continue to Stage 2 (or Ctrl+C to stop)... "
        run_stage_2
        analyze_results
        log ""
        log "Stage 2 complete. Review results above before starting Stage 3."
        log "To continue: $0 3"
        log ""
        read -p "Press Enter to continue to Stage 3 (or Ctrl+C to stop)... "
        run_stage_3
        analyze_results
        ;;
    analyze)
        analyze_results
        ;;
    *)
        echo "Usage: $0 [1|2|3|all|analyze]"
        exit 1
        ;;
esac

log ""
log "============================================================"
log "ABLATION STUDY COMPLETE"
log "Results: $RESULTS_DIR"
log "============================================================"
