#!/usr/bin/env bash
set -euo pipefail

# run_next_experiments.sh
# Set-and-forget GRPO experiment queue for /home/cang688/Unsloth-RL-2.
#
# Examples:
#   ./run_next_experiments.sh
#   ./run_next_experiments.sh E6
#   ./run_next_experiments.sh E5 E6 E7 E8 E9
#   ./run_next_experiments.sh qwen

PROJECT_ROOT="/home/cang688/Unsloth-RL-2"
RUNS_ROOT="$PROJECT_ROOT/grpo_runs"
DASHBOARD_ROOT="$PROJECT_ROOT/dashboards"
LOG_ROOT="$PROJECT_ROOT/results/worklogs"

CODELLAMA_ENV="unsloth_codelama"
QWEN_ENV="qwen35_unsloth"
QWEN_MODEL="unsloth/Qwen3.5-4B"
DEFAULT_REWARD_PROFILE="${REWARD_PROFILE:-combined}"

mkdir -p "$RUNS_ROOT" "$DASHBOARD_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

qwen_train_root() {
  printf '%s/E8_qwen35_4b_%s_dual3090_grpo' "$RUNS_ROOT" "$1"
}

qwen_lora_holdout_root() {
  printf '%s/E9_qwen35_4b_%s_lora_holdout_8gen' "$RUNS_ROOT" "$1"
}

codellama_train_root() {
  printf '%s/E3_codelama_%s_dual3090_grpo' "$RUNS_ROOT" "$1"
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
  E5  Qwen3.5 sanity reward check
  E6  Qwen3.5 tiny overfit train (50 steps)
  E7  Qwen3.5-4B base holdout evaluation
  E8  Qwen3.5-4B dual-GPU GRPO training
  E9  Qwen3.5-4B trained LoRA holdout evaluation
  E8-CORRECTNESS / E9-CORRECTNESS  Correctness-only train/eval
  E8-STYLE / E9-STYLE              Style-only train/eval
  E8-COMBINED / E9-COMBINED        Combined correctness/style train/eval

Groups:
  all    Run the full queue in order
  code   Run E0, E1, E3, E4
  qwen   Run E5, E6, E7, E8, E9
  reward-profiles  Run correctness, style, and combined E8/E9 comparisons

Examples:
  ./run_next_experiments.sh
  ./run_next_experiments.sh E6
  ./run_next_experiments.sh E5 E6 E7 E8 E9
  ./run_next_experiments.sh qwen
  REWARD_PROFILE=style ./run_next_experiments.sh E6 E8 E9
  ./run_next_experiments.sh reward-profiles
EOF
}

job_e0() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local code_llama_sanity="$RUNS_ROOT/E0_codelama_${profile}_sanity_reward"
  log "E0: CodeLlama reward sanity check, reward profile=$profile"
  run_in_conda_env "$CODELLAMA_ENV" env \
    TEST_MODE=sanity_reward \
    REWARD_PROFILE="$profile" \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_sanity" \
    LOAD_IN_4BIT=true \
    LOAD_IN_16BIT=false \
    FAST_INFERENCE=false \
    python train_conrad.py
}

job_e1() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local code_llama_base_holdout="$RUNS_ROOT/E1_codelama_${profile}_base_holdout"
  log "E1: CodeLlama base holdout eval, reward profile=$profile"
  run_in_conda_env "$CODELLAMA_ENV" env \
    TEST_MODE=eval_base_holdout \
    REWARD_PROFILE="$profile" \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_base_holdout" \
    LOAD_IN_4BIT=true \
    LOAD_IN_16BIT=false \
    FAST_INFERENCE=false \
    EVAL_NUM_GENERATIONS=4 \
    python train_conrad.py
  make_dashboard "$CODELLAMA_ENV" "$code_llama_base_holdout" "$DASHBOARD_ROOT/E1_codelama_${profile}_base_holdout.html"
}

job_e3() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local code_llama_train
  code_llama_train="$(codellama_train_root "$profile")"
  log "E3: CodeLlama dual-GPU GRPO training, reward profile=$profile"
  run_in_conda_env "$CODELLAMA_ENV" env \
    CUDA_VISIBLE_DEVICES=0,1 \
    WORLD_SIZE=2 \
    TEST_MODE=train \
    REWARD_PROFILE="$profile" \
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
  make_dashboard "$CODELLAMA_ENV" "$code_llama_train" "$DASHBOARD_ROOT/E3_codelama_${profile}_dual3090_grpo.html"
}

job_e4() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local code_llama_train
  local code_llama_lora_holdout="$RUNS_ROOT/E4_codelama_${profile}_lora_holdout"
  code_llama_train="$(codellama_train_root "$profile")"
  log "E4: CodeLlama trained LoRA holdout eval, reward profile=$profile"
  run_in_conda_env "$CODELLAMA_ENV" env \
    TEST_MODE=eval_lora_holdout \
    REWARD_PROFILE="$profile" \
    MODEL_NAME=codellama/CodeLlama-7b-Python-hf \
    RUN_ROOT="$code_llama_lora_holdout" \
    LORA_PATH="$code_llama_train/lora" \
    LOAD_IN_4BIT=true \
    LOAD_IN_16BIT=false \
    FAST_INFERENCE=false \
    EVAL_NUM_GENERATIONS=4 \
    python train_conrad.py
  make_dashboard "$CODELLAMA_ENV" "$code_llama_lora_holdout" "$DASHBOARD_ROOT/E4_codelama_${profile}_lora_holdout.html"
}

job_e5() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local run_root="$RUNS_ROOT/E5_qwen35_${profile}_sanity_reward"
  log "E5: Qwen3.5 sanity reward check, reward profile=$profile"
  run_in_conda_env "$QWEN_ENV" env \
    UNSLOTH_COMPILE_DISABLE=1 \
    TEST_MODE=sanity_reward \
    REWARD_PROFILE="$profile" \
    MODEL_NAME="$QWEN_MODEL" \
    RUN_ROOT="$run_root" \
    LOAD_IN_4BIT=false \
    LOAD_IN_16BIT=true \
    FAST_INFERENCE=false \
    python train_conrad.py
}

job_e6() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local run_root="$RUNS_ROOT/E6_qwen35_${profile}_tiny_overfit_50"
  log "E6: Qwen3.5 tiny overfit train (50 steps), reward profile=$profile"
  run_in_conda_env "$QWEN_ENV" env \
    UNSLOTH_COMPILE_DISABLE=1 \
    CUDA_VISIBLE_DEVICES=0,1 \
    WORLD_SIZE=2 \
    TEST_MODE=tiny_overfit_train \
    REWARD_PROFILE="$profile" \
    TINY_MAX_STEPS=50 \
    MODEL_NAME="$QWEN_MODEL" \
    RUN_ROOT="$run_root" \
    LOAD_IN_4BIT=false \
    LOAD_IN_16BIT=true \
    FAST_INFERENCE=false \
    CLEAN_OUTPUT_DIRS=true \
    MAX_STEPS=50 \
    SAVE_STEPS=25 \
    LOGGING_STEPS=1 \
    MAX_SEQ_LENGTH=2048 \
    MAX_PROMPT_LENGTH=1024 \
    MAX_COMPLETION_LENGTH=384 \
    PER_DEVICE_TRAIN_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    NUM_GENERATIONS=4 \
    LEARNING_RATE=5e-6 \
    accelerate launch --num_processes 2 train_conrad.py
  make_dashboard "$QWEN_ENV" "$run_root" "$DASHBOARD_ROOT/E6_qwen35_${profile}_tiny_overfit_50.html"
}

job_e7() {
  local profile="$DEFAULT_REWARD_PROFILE"
  local run_root="$RUNS_ROOT/E7_qwen35_4b_${profile}_base_holdout_8gen"
  log "E7: Qwen3.5-4B base holdout eval, reward profile=$profile"
  run_in_conda_env "$QWEN_ENV" env \
    UNSLOTH_COMPILE_DISABLE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TEST_MODE=eval_base_holdout \
    REWARD_PROFILE="$profile" \
    MODEL_NAME="$QWEN_MODEL" \
    RUN_ROOT="$run_root" \
    LOAD_IN_4BIT=false \
    LOAD_IN_16BIT=true \
    FAST_INFERENCE=false \
    PER_DEVICE_TRAIN_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    NUM_GENERATIONS=4 \
    EVAL_NUM_GENERATIONS=8 \
    python train_conrad.py
  make_dashboard "$QWEN_ENV" "$run_root" "$DASHBOARD_ROOT/E7_qwen35_4b_${profile}_base_holdout_8gen.html"
}

job_e8() {
  local profile="${1:-$DEFAULT_REWARD_PROFILE}"
  local train_root
  train_root="$(qwen_train_root "$profile")"

  log "E8: Qwen3.5-4B dual-GPU GRPO training, reward profile=$profile"
  run_in_conda_env "$QWEN_ENV" env \
    UNSLOTH_COMPILE_DISABLE=1 \
    CUDA_VISIBLE_DEVICES=0,1 \
    WORLD_SIZE=2 \
    TEST_MODE=train \
    REWARD_PROFILE="$profile" \
    MODEL_NAME="$QWEN_MODEL" \
    RUN_ROOT="$train_root" \
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
  make_dashboard "$QWEN_ENV" "$train_root" "$DASHBOARD_ROOT/E8_qwen35_4b_${profile}_dual3090_grpo.html"
}

job_e9() {
  local profile="${1:-$DEFAULT_REWARD_PROFILE}"
  local train_root
  local eval_root
  train_root="$(qwen_train_root "$profile")"
  eval_root="$(qwen_lora_holdout_root "$profile")"

  log "E9: Qwen3.5-4B trained LoRA holdout eval, reward profile=$profile"
  run_in_conda_env "$QWEN_ENV" env \
    UNSLOTH_COMPILE_DISABLE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TEST_MODE=eval_lora_holdout \
    REWARD_PROFILE="$profile" \
    MODEL_NAME="$QWEN_MODEL" \
    RUN_ROOT="$eval_root" \
    LORA_PATH="$train_root/lora" \
    LOAD_IN_4BIT=false \
    LOAD_IN_16BIT=true \
    FAST_INFERENCE=false \
    EVAL_NUM_GENERATIONS=8 \
    python train_conrad.py
  make_dashboard "$QWEN_ENV" "$eval_root" "$DASHBOARD_ROOT/E9_qwen35_4b_${profile}_lora_holdout_8gen.html"
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
        printf '%s\n' E0 E1 E3 E4 E5 E6 E7 E8 E9 \
          E8-CORRECTNESS E9-CORRECTNESS E8-STYLE E9-STYLE E8-COMBINED E9-COMBINED
        return 0
        ;;
      ALL)
        expanded_jobs+=(E0 E1 E3 E4 E5 E6 E7 E8 E9)
        ;;
      CODE|CODELLAMA|CODELAMA)
        expanded_jobs+=(E0 E1 E3 E4)
        ;;
      QWEN|QWEN35)
        expanded_jobs+=(E5 E6 E7 E8 E9)
        ;;
      REWARD-PROFILES|PROFILES)
        expanded_jobs+=(
          E8-CORRECTNESS E9-CORRECTNESS
          E8-STYLE E9-STYLE
          E8-COMBINED E9-COMBINED
        )
        ;;
      E0|E1|E3|E4|E5|E6|E7|E8|E9|E8-CORRECTNESS|E9-CORRECTNESS|E8-STYLE|E9-STYLE|E8-COMBINED|E9-COMBINED)
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
      E8) job_e8 ;;
      E9) job_e9 ;;
      E8-CORRECTNESS) job_e8 correctness ;;
      E9-CORRECTNESS) job_e9 correctness ;;
      E8-STYLE) job_e8 style ;;
      E9-STYLE) job_e9 style ;;
      E8-COMBINED) job_e8 combined ;;
      E9-COMBINED) job_e9 combined ;;
    esac
  done

  log "Experiment queue complete."
  log "Dashboards saved to: $DASHBOARD_ROOT"
  log "Runs saved to: $RUNS_ROOT"
}

main "$@"
