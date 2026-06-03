#!/usr/bin/env bash
set -euo pipefail

# run_next_experiments.sh
# Set-and-forget GRPO experiment queue for /home/cang688/Unsloth-RL-2.
# Run with:
#   cd /home/cang688/Unsloth-RL-2
#   chmod +x run_next_experiments.sh
#   nohup ./run_next_experiments.sh > experiment_queue_launcher.log 2>&1 &
#   disown

PROJECT_ROOT="/home/cang688/Unsloth-RL-2"
RUNS_ROOT="$PROJECT_ROOT/grpo_runs"
DASHBOARD_ROOT="$PROJECT_ROOT/dashboards"
LOG_ROOT="$PROJECT_ROOT/results/worklogs"

CODELLAMA_ENV="unsloth_codelama"
QWEN_ENV="qwen35_unsloth"

mkdir -p "$RUNS_ROOT" "$DASHBOARD_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_in_conda_env() {
  local env_name="$1"
  shift
  local status=0

  log "Activating conda env: $env_name"

  # This form works in non-interactive nohup scripts.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$env_name"

  log "Python: $(which python)"
  log "Command: $*"
  "$@" || status=$?

  conda deactivate
  return "$status"
}

make_dashboard() {
  local env_name="$1"
  shift
  local run_dir="$1"
  local output_html="$2"

  if [[ -f "$PROJECT_ROOT/plot_training_conrad_toggle.py" ]]; then
    log "Generating dashboard: $output_html"
    if ! run_in_conda_env "$env_name" python "$PROJECT_ROOT/plot_training_conrad_toggle.py" \
      --run-dir "$run_dir" \
      --output "$output_html"; then
      log "Dashboard failed for $run_dir; continuing."
    fi
  else
    log "plot_training_conrad_toggle.py not found; skipping dashboard."
  fi
}

log "Starting GRPO experiment queue."
log "Project root: $PROJECT_ROOT"
log "Runs root: $RUNS_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CodeLlama sanity check
# ─────────────────────────────────────────────────────────────────────────────
CODELLAMA_SANITY="$RUNS_ROOT/E0_codelama_sanity_reward"
log "E0: CodeLlama reward sanity check"
run_in_conda_env "$CODELLAMA_ENV" env \
  TEST_MODE=sanity_reward \
  MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
  RUN_ROOT="$CODELLAMA_SANITY" \
  LOAD_IN_4BIT=true \
  LOAD_IN_16BIT=false \
  FAST_INFERENCE=false \
  python train_conrad.py

# ─────────────────────────────────────────────────────────────────────────────
# 2. CodeLlama base holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────
CODELLAMA_BASE_HOLDOUT="$RUNS_ROOT/E1_codelama_base_holdout"
log "E1: CodeLlama base holdout eval"
run_in_conda_env "$CODELLAMA_ENV" env \
  TEST_MODE=eval_base_holdout \
  MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
  RUN_ROOT="$CODELLAMA_BASE_HOLDOUT" \
  LOAD_IN_4BIT=true \
  LOAD_IN_16BIT=false \
  FAST_INFERENCE=false \
  EVAL_NUM_GENERATIONS=4 \
  python train_conrad.py
make_dashboard "$CODELLAMA_ENV" "$CODELLAMA_BASE_HOLDOUT" "$DASHBOARD_ROOT/E1_codelama_base_holdout.html"

# ─────────────────────────────────────────────────────────────────────────────
# 3. CodeLlama single-GPU GRPO training
#    Single GPU is used because your previous dual-GPU run hit DDP/model.config.
# ─────────────────────────────────────────────────────────────────────────────
CODELLAMA_TRAIN="$RUNS_ROOT/E3_codelama_single3090_grpo"
log "E3: CodeLlama single-GPU GRPO training"
run_in_conda_env "$CODELLAMA_ENV" env \
  CUDA_VISIBLE_DEVICES=0 \
  TEST_MODE=train \
  MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
  RUN_ROOT="$CODELLAMA_TRAIN" \
  LOAD_IN_4BIT=true \
  LOAD_IN_16BIT=false \
  FAST_INFERENCE=false \
  CLEAN_OUTPUT_DIRS=true \
  MAX_STEPS=100 \
  SAVE_STEPS=50 \
  LOGGING_STEPS=1 \
  MAX_SEQ_LENGTH=2048 \
  MAX_PROMPT_LENGTH=1024 \
  MAX_COMPLETION_LENGTH=384 \
  PER_DEVICE_TRAIN_BATCH_SIZE=4 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  NUM_GENERATIONS=4 \
  LEARNING_RATE=5e-6 \
  python train_conrad.py
make_dashboard "$CODELLAMA_ENV" "$CODELLAMA_TRAIN" "$DASHBOARD_ROOT/E3_codelama_single3090_grpo.html"

# ─────────────────────────────────────────────────────────────────────────────
# 4. CodeLlama trained LoRA holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────
CODELLAMA_LORA_HOLDOUT="$RUNS_ROOT/E4_codelama_lora_holdout"
log "E4: CodeLlama trained LoRA holdout eval"
run_in_conda_env "$CODELLAMA_ENV" env \
  TEST_MODE=eval_lora_holdout \
  MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
  RUN_ROOT="$CODELLAMA_LORA_HOLDOUT" \
  LORA_PATH="$CODELLAMA_TRAIN/lora" \
  LOAD_IN_4BIT=true \
  LOAD_IN_16BIT=false \
  FAST_INFERENCE=false \
  EVAL_NUM_GENERATIONS=4 \
  python train_conrad.py
make_dashboard "$CODELLAMA_ENV" "$CODELLAMA_LORA_HOLDOUT" "$DASHBOARD_ROOT/E4_codelama_lora_holdout.html"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Qwen3.5-9B base holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────
QWEN_BASE_HOLDOUT="$RUNS_ROOT/E5_qwen35_9b_base_holdout"
log "E5: Qwen3.5-9B base holdout eval"
run_in_conda_env "$QWEN_ENV" env \
  CUDA_VISIBLE_DEVICES=0 \
  TEST_MODE=eval_base_holdout \
  MODEL_NAME=unsloth/Qwen3.5-9B \
  RUN_ROOT="$QWEN_BASE_HOLDOUT" \
  LOAD_IN_4BIT=false \
  LOAD_IN_16BIT=true \
  FAST_INFERENCE=false \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=4 \
  NUM_GENERATIONS=4 \
  EVAL_NUM_GENERATIONS=4 \
  python train_conrad.py
make_dashboard "$QWEN_ENV" "$QWEN_BASE_HOLDOUT" "$DASHBOARD_ROOT/E5_qwen35_9b_base_holdout.html"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Qwen3.5-9B single-GPU GRPO training
#    If this OOMs, rerun with MODEL_NAME=unsloth/Qwen3.5-4B.
# ─────────────────────────────────────────────────────────────────────────────
QWEN_TRAIN="$RUNS_ROOT/E6_qwen35_9b_single3090_grpo"
log "E6: Qwen3.5-9B single-GPU GRPO training"
run_in_conda_env "$QWEN_ENV" env \
  CUDA_VISIBLE_DEVICES=0 \
  TEST_MODE=train \
  MODEL_NAME=unsloth/Qwen3.5-9B \
  RUN_ROOT="$QWEN_TRAIN" \
  LOAD_IN_4BIT=false \
  LOAD_IN_16BIT=true \
  FAST_INFERENCE=false \
  CLEAN_OUTPUT_DIRS=true \
  MAX_STEPS=100 \
  SAVE_STEPS=50 \
  LOGGING_STEPS=1 \
  MAX_SEQ_LENGTH=2048 \
  MAX_PROMPT_LENGTH=1024 \
  MAX_COMPLETION_LENGTH=384 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=4 \
  NUM_GENERATIONS=4 \
  LEARNING_RATE=5e-6 \
  python train_conrad.py
make_dashboard "$QWEN_ENV" "$QWEN_TRAIN" "$DASHBOARD_ROOT/E6_qwen35_9b_single3090_grpo.html"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Qwen3.5-9B trained LoRA holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────
QWEN_LORA_HOLDOUT="$RUNS_ROOT/E7_qwen35_9b_lora_holdout"
log "E7: Qwen3.5-9B trained LoRA holdout eval"
run_in_conda_env "$QWEN_ENV" env \
  CUDA_VISIBLE_DEVICES=0 \
  TEST_MODE=eval_lora_holdout \
  MODEL_NAME=unsloth/Qwen3.5-9B \
  RUN_ROOT="$QWEN_LORA_HOLDOUT" \
  LORA_PATH="$QWEN_TRAIN/lora" \
  LOAD_IN_4BIT=false \
  LOAD_IN_16BIT=true \
  FAST_INFERENCE=false \
  EVAL_NUM_GENERATIONS=4 \
  python train_conrad.py
make_dashboard "$QWEN_ENV" "$QWEN_LORA_HOLDOUT" "$DASHBOARD_ROOT/E7_qwen35_9b_lora_holdout.html"

log "Experiment queue complete."
log "Dashboards saved to: $DASHBOARD_ROOT"
log "Runs saved to: $RUNS_ROOT"
