#!/usr/bin/env bash
set -euo pipefail

# run_next_experiments.sh
# Set-and-forget GRPO experiment queue for /home/cang688/Unsloth-RL-2.
#
# Examples:
#   ./run_next_experiments.sh
#   ./run_next_experiments.sh E6
#   ./run_next_experiments.sh E5 E6 E7
#   ./run_next_experiments.sh qwen

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

usage() {
  cat <<'EOF'
Usage:
  ./run_next_experiments.sh [job ...]

Jobs:
  E0  CodeLlama sanity reward check
  E1  CodeLlama base holdout evaluation
  E3  CodeLlama dual-GPU GRPO training
  E4  CodeLlama trained LoRA holdout evaluation
  E5  Qwen3.5-9B base holdout evaluation
  E6  Qwen3.5-9B dual-GPU GRPO training
  E7  Qwen3.5-9B trained LoRA holdout evaluation

Groups:
  all    Run the full queue in order
  code   Run E0, E1, E3, E4
  qwen   Run E5, E6, E7

Examples:
  ./run_next_experiments.sh
  ./run_next_experiments.sh E6
  ./run_next_experiments.sh E5 E6 E7
  ./run_next_experiments.sh qwen
EOF
}

job_e0() {
  local code_llama_sanity="$RUNS_ROOT/E0_codelama_sanity_reward"
  log "E0: CodeLlama reward sanity check"
  run_in_conda_env "$CODELLAMA_ENV" env \
    TEST_MODE=sanity_reward \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_sanity" \
    LOAD_IN_4BIT=true \
    LOAD_IN_16BIT=false \
    FAST_INFERENCE=false \
    python train_conrad.py
}

job_e1() {
  local code_llama_base_holdout="$RUNS_ROOT/E1_codelama_base_holdout"
  log "E1: CodeLlama base holdout eval"
  run_in_conda_env "$CODELLAMA_ENV" env \
    TEST_MODE=eval_base_holdout \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_base_holdout" \
    LOAD_IN_4BIT=true \
    LOAD_IN_16BIT=false \
    FAST_INFERENCE=false \
    EVAL_NUM_GENERATIONS=4 \
    python train_conrad.py
  make_dashboard "$CODELLAMA_ENV" "$code_llama_base_holdout" "$DASHBOARD_ROOT/E1_codelama_base_holdout.html"
}

job_e3() {
  local code_llama_train="$RUNS_ROOT/E3_codelama_single3090_grpo"
  log "E3: CodeLlama dual-GPU GRPO training"
  run_in_conda_env "$CODELLAMA_ENV" env \
    CUDA_VISIBLE_DEVICES=0,1 \
    WORLD_SIZE=2 \
    TEST_MODE=train \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_train" \
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
    accelerate launch --num_processes 2 train_conrad.py
  make_dashboard "$CODELLAMA_ENV" "$code_llama_train" "$DASHBOARD_ROOT/E3_codelama_single3090_grpo.html"
}

job_e4() {
  local code_llama_lora_holdout="$RUNS_ROOT/E4_codelama_lora_holdout"
  log "E4: CodeLlama trained LoRA holdout eval"
  run_in_conda_env "$CODELLAMA_ENV" env \
    TEST_MODE=eval_lora_holdout \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_lora_holdout" \
    LORA_PATH="$CODELLAMA_TRAIN/lora" \
    LOAD_IN_4BIT=true \
    LOAD_IN_16BIT=false \
    FAST_INFERENCE=false \
    EVAL_NUM_GENERATIONS=4 \
    python train_conrad.py
  make_dashboard "$CODELLAMA_ENV" "$code_llama_lora_holdout" "$DASHBOARD_ROOT/E4_codelama_lora_holdout.html"
}

job_e5() {
  local qwen_base_holdout="$RUNS_ROOT/E5_qwen35_9b_base_holdout"
  log "E5: Qwen3.5-9B base holdout eval"
  run_in_conda_env "$QWEN_ENV" env \
    CUDA_VISIBLE_DEVICES=0 \
    TEST_MODE=eval_base_holdout \
    MODEL_NAME=unsloth/Qwen3.5-9B \
    RUN_ROOT="$qwen_base_holdout" \
    LOAD_IN_4BIT=false \
    LOAD_IN_16BIT=true \
    FAST_INFERENCE=false \
    PER_DEVICE_TRAIN_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    NUM_GENERATIONS=4 \
    EVAL_NUM_GENERATIONS=4 \
    python train_conrad.py
  make_dashboard "$QWEN_ENV" "$qwen_base_holdout" "$DASHBOARD_ROOT/E5_qwen35_9b_base_holdout.html"
}

job_e6() {
  local qwen_train="$RUNS_ROOT/E6_qwen35_9b_single3090_grpo"
  log "E6: Qwen3.5-9B dual-GPU GRPO training"
  run_in_conda_env "$QWEN_ENV" env \
    CUDA_VISIBLE_DEVICES=0,1 \
    WORLD_SIZE=2 \
    TEST_MODE=train \
    MODEL_NAME=unsloth/Qwen3.5-9B \
    RUN_ROOT="$qwen_train" \
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
    accelerate launch --num_processes 2 train_conrad.py
  make_dashboard "$QWEN_ENV" "$qwen_train" "$DASHBOARD_ROOT/E6_qwen35_9b_single3090_grpo.html"
}

job_e7() {
  local qwen_lora_holdout="$RUNS_ROOT/E7_qwen35_9b_lora_holdout"
  log "E7: Qwen3.5-9B trained LoRA holdout eval"
  run_in_conda_env "$QWEN_ENV" env \
    CUDA_VISIBLE_DEVICES=0 \
    TEST_MODE=eval_lora_holdout \
    MODEL_NAME=unsloth/Qwen3.5-9B \
    RUN_ROOT="$qwen_lora_holdout" \
    LORA_PATH="$QWEN_TRAIN/lora" \
    LOAD_IN_4BIT=false \
    LOAD_IN_16BIT=true \
    FAST_INFERENCE=false \
    EVAL_NUM_GENERATIONS=4 \
    python train_conrad.py
  make_dashboard "$QWEN_ENV" "$qwen_lora_holdout" "$DASHBOARD_ROOT/E7_qwen35_9b_lora_holdout.html"
}

main() {
  local selectors=("$@")
  local expanded_jobs=()
  local selector
  local job
  local normalized

  if [[ ${#selectors[@]} -eq 0 ]]; then
    selectors=(all)
  fi

  for selector in "${selectors[@]}"; do
    normalized="$(printf '%s' "$selector" | tr '[:lower:]' '[:upper:]')"
    case "$normalized" in
      -H|--HELP|HELP)
        usage
        return 0
        ;;
      LIST|--LIST)
        printf '%s\n' E0 E1 E3 E4 E5 E6 E7
        return 0
        ;;
      ALL)
        expanded_jobs+=(E0 E1 E3 E4 E5 E6 E7)
        ;;
      CODE|CODELLAMA|CODELAMA)
        expanded_jobs+=(E0 E1 E3 E4)
        ;;
      QWEN|QWEN35)
        expanded_jobs+=(E5 E6 E7)
        ;;
      E0|E1|E3|E4|E5|E6|E7)
        expanded_jobs+=("$normalized")
        ;;
      *)
        log "Unknown selector: $selector"
        usage
        return 1
        ;;
    esac
  done

  for job in "${expanded_jobs[@]}"; do
    case "$job" in
      E0) job_e0 ;;
      E1) job_e1 ;;
      E3) job_e3 ;;
      E4) job_e4 ;;
      E5) job_e5 ;;
      E6) job_e6 ;;
      E7) job_e7 ;;
    esac
  done

  log "Experiment queue complete."
  log "Dashboards saved to: $DASHBOARD_ROOT"
  log "Runs saved to: $RUNS_ROOT"
}

main "$@"
