# UnSloth RL Experimental

This repository contains a GRPO-based coding RL workflow built around `train_conrad.py` and a set-and-forget launcher, `run_next_experiments.sh`.

The project trains and evaluates code models on HumanEval using [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), PEFT LoRA fine-tuning, and a reward function that checks formatting, compilation, runtime success, and unit tests.

The repo currently supports two model families:

- CodeLlama
- Qwen3.5

The Qwen path is the one you should use for the dual RTX 3090 setup.

## What This Project Does

At a high level, the workflow is:

1. Load a model with Unsloth.
2. Format HumanEval prompts into the expected chat or instruction format.
3. Generate candidate completions with GRPO.
4. Score completions with a reward function.
5. Train LoRA adapters from those rewards.
6. Evaluate base and LoRA checkpoints on train and holdout slices.
7. Build HTML dashboards from the saved logs.

The training and evaluation logic lives in `train_conrad.py`.
The queue/orchestration logic lives in `run_next_experiments.sh`.

## How It Works

### Data split

The script loads `openai/openai_humaneval` and uses the dataset's `test` split as the source corpus. It then creates three index-based slices:

- Train slice: tasks `0` through `130`
- Holdout slice: tasks `131` through `163`
- Tiny slice: tasks `0` through `4`

These slices are used by the different `TEST_MODE` values.

### Prompt formatting

The model is prompted using either:

- Qwen chat template formatting, or
- a plain instruction-style fallback

The prompt expects the model to return a complete Python solution and then close with `</code>`.

### Reward function

The current reward is designed around code correctness and output discipline:

- `0.2` for formatting
- `0.8` for code that compiles and runs successfully
- `5.0` for passing the test
- penalty for extra text after `</code>`

The total reward is:

```text
0.2 * reward_format + 0.8 * reward_compile + 5.0 * reward_tests - extra_text_penalty
```

Important note:

- Style is tracked in the logs and dashboard.
- Style is not part of the current reward formula in `train_conrad.py`.

### Execution and safety

The script:

- extracts code only from the text before `</code>`
- rejects dangerous patterns such as `os.system`, `subprocess`, `eval`, and `exec`
- runs generated code in a temporary directory
- optionally applies CPU/memory limits
- times out execution if code takes too long

### Training

Training uses TRL `GRPOTrainer` with LoRA adapters. Key training behavior includes:

- multiple generations per prompt for ranking
- short-run logging and saved checkpoints
- periodic saving
- optional cleaning of output directories for training runs

### Outputs

Each run writes to a `RUN_ROOT` directory with:

- `logs/` for CSV and JSONL logs
- `output/` for trainer outputs
- `lora/` for the saved LoRA adapter

The launcher also writes HTML dashboards under `dashboards/`.

## Repository Layout

- `train_conrad.py` - main training and evaluation script
- `run_next_experiments.sh` - experiment queue launcher
- `plot_training_conrad_toggle.py` - dashboard builder for logs
- `envs/` - conda environment files
- `configs/env_matrix.json` - model-to-environment and launch recommendations

## Setup

### 1. Create the environment

For CodeLlama:

```bash
conda env create -f envs/codellama-unsloth.yml
```

For Qwen3.5:

```bash
conda env create -f envs/qwen35-unsloth.yml
```

### 2. Activate the environment

```bash
conda activate qwen35_unsloth
```

### 3. Verify GPU visibility

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

For the dual 3090 setup, you want:

- `torch.version.cuda` not `None`
- `torch.cuda.is_available()` = `True`
- `torch.cuda.device_count()` = `2`

## How To Run

The launcher is designed for the remote machine path hardcoded in the script:

```text
/home/cang688/Unsloth-RL-2
```

### Full queue

Run everything in order:

```bash
cd /home/cang688/Unsloth-RL-2
chmod +x run_next_experiments.sh
nohup ./run_next_experiments.sh > experiment_queue_launcher.log 2>&1 &
disown
```

### Qwen-only queue

Run the staged Qwen workflow:

```bash
./run_next_experiments.sh qwen
```

This runs:

- `E5` sanity reward check
- `E6` tiny overfit train
- `E7` base holdout eval
- `E8` dual-GPU training
- `E9` LoRA holdout eval

### Individual jobs

Run a single job:

```bash
./run_next_experiments.sh E8
```

Run a custom sequence:

```bash
./run_next_experiments.sh E5 E6 E7 E8 E9
```

### Direct training script usage

You can also bypass the launcher and run `train_conrad.py` directly.

Sanity check:

```bash
TEST_MODE=sanity_reward \
MODEL_NAME=unsloth/Qwen3.5-9B \
RUN_ROOT=/home/cang688/Unsloth-RL-2/grpo_runs/qwen_sanity_reward \
python train_conrad.py
```

Tiny overfit:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WORLD_SIZE=2 \
TEST_MODE=tiny_overfit_train \
TINY_MAX_STEPS=50 \
MODEL_NAME=unsloth/Qwen3.5-9B \
RUN_ROOT=/home/cang688/Unsloth-RL-2/grpo_runs/qwen_tiny_overfit_50 \
LOAD_IN_4BIT=false \
LOAD_IN_16BIT=true \
FAST_INFERENCE=false \
accelerate launch --num_processes 2 train_conrad.py
```

Base holdout eval:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEST_MODE=eval_base_holdout \
EVAL_NUM_GENERATIONS=8 \
MODEL_NAME=unsloth/Qwen3.5-9B \
LOAD_IN_4BIT=false \
LOAD_IN_16BIT=true \
FAST_INFERENCE=false \
RUN_ROOT=/home/cang688/Unsloth-RL-2/grpo_runs/qwen_eval_base_holdout_8gen \
python train_conrad.py
```

LoRA holdout eval:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEST_MODE=eval_lora_holdout \
EVAL_NUM_GENERATIONS=8 \
MODEL_NAME=unsloth/Qwen3.5-9B \
LORA_PATH=/home/cang688/Unsloth-RL-2/grpo_runs/E8_qwen35_9b_single3090_grpo/lora \
LOAD_IN_4BIT=false \
LOAD_IN_16BIT=true \
FAST_INFERENCE=false \
RUN_ROOT=/home/cang688/Unsloth-RL-2/grpo_runs/qwen_eval_lora_holdout_8gen \
python train_conrad.py
```

## Important Commands

### Launcher commands

- `./run_next_experiments.sh`
  - Runs the full queue
- `./run_next_experiments.sh qwen`
  - Runs the Qwen-only staged workflow
- `./run_next_experiments.sh code`
  - Runs the CodeLlama workflow
- `./run_next_experiments.sh E5 E6 E7 E8 E9`
  - Runs a custom Qwen sequence
- `./run_next_experiments.sh E8`
  - Runs a single job

### Python commands

- `python train_conrad.py`
  - Runs training or evaluation based on `TEST_MODE`
- `python plot_training_conrad_toggle.py --run-dir ... --output ...`
  - Builds a dashboard from a run directory

### Environment checks

- `nvidia-smi`
  - Confirms the system can see the GPUs
- `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"`
  - Confirms the Python env can actually use CUDA

## Job Order

The current Qwen order is:

1. `E5` - sanity reward check
2. `E6` - tiny overfit train, 50 steps
3. `E7` - base holdout eval with 8 generations
4. `E8` - dual-GPU Qwen3.5-9B training
5. `E9` - LoRA holdout eval with 8 generations

This order makes sense because it checks the reward path first, then validates learning on a tiny slice, then checks base performance, then spends compute on full training, then evaluates the trained LoRA.

## Key Variables

The table below explains the main variables, what they do, and what they influence.

| Variable | What it does | Depends on / influences |
| --- | --- | --- |
| `TEST_MODE` | Chooses training or eval behavior | Controls dataset slice, reward path, and whether training or eval runs |
| `MODEL_NAME` | Selects the base model to load | Affects tokenizer, memory use, Qwen vs CodeLlama behavior, and LoRA compatibility |
| `RUN_ROOT` | Root directory for a run | Controls where logs, outputs, and LoRA artifacts are written |
| `LORA_PATH` | Path to a saved LoRA adapter for eval | Used by LoRA eval modes; must point at a trained adapter directory |
| `LOAD_IN_4BIT` | Loads the base model in 4-bit | Strongly affects VRAM use; if enabled, 16-bit is disabled |
| `LOAD_IN_16BIT` | Loads the model in 16-bit/bf16-style mode | Used for Qwen by default; affects VRAM use and stability |
| `LOAD_IN_8BIT` | Loads the model in 8-bit | Alternative low-memory loading path |
| `FAST_INFERENCE` | Disables or enables fast inference mode | Affects generation behavior and memory use |
| `MAX_SEQ_LENGTH` | Maximum total sequence length | Influences memory use and prompt/completion capacity |
| `MAX_PROMPT_LENGTH` | Maximum prompt length | Affects how much context is kept before generation |
| `MAX_COMPLETION_LENGTH` | Maximum generated completion length | Affects runtime, memory use, and rambling behavior |
| `NUM_GENERATIONS` | Number of completions per prompt during GRPO | Influences reward ranking signal, memory use, and training throughput |
| `EVAL_NUM_GENERATIONS` | Number of completions per prompt during eval | Influences eval time and the stability of reported metrics |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | Per-GPU training batch size | Influences VRAM use and effective batch size |
| `GRADIENT_ACCUMULATION_STEPS` | Accumulates gradients over multiple microbatches | Influences effective batch size and training stability |
| `MAX_STEPS` | Total training steps for normal training | Directly controls training length and total compute |
| `TINY_MAX_STEPS` | Total training steps for tiny overfit mode | Controls the short diagnostic overfit run |
| `LEARNING_RATE` | Optimizer learning rate | Influences stability, convergence, and collapse risk |
| `WARMUP_RATIO` | Learning-rate warmup fraction | Influences early training stability |
| `WEIGHT_DECAY` | Weight decay regularization | Influences generalization and parameter drift |
| `LORA_R` | LoRA rank | Influences adapter capacity, memory, and trainability |
| `LORA_ALPHA` | LoRA scaling factor | Influences LoRA update strength |
| `LORA_DROPOUT` | LoRA dropout | Influences regularization |
| `LORA_TARGET_MODULES` | Which modules receive LoRA adapters | Influences what parts of the model can adapt |
| `USE_CHAT_TEMPLATE` | Enables chat-style prompting | Influences prompt format and model behavior |
| `QWEN_ENABLE_THINKING` | Enables Qwen thinking mode if supported | Influences Qwen prompt formatting and reasoning behavior |
| `TRAIN_TASK_START` | Start index for the training slice | Influences which HumanEval tasks are used for training |
| `TRAIN_TASK_END` | End index for the training slice | Influences which HumanEval tasks are used for training |
| `HOLDOUT_TASK_START` | Start index for the holdout slice | Influences which HumanEval tasks are used for evaluation |
| `HOLDOUT_TASK_END` | End index for the holdout slice | Influences which HumanEval tasks are used for evaluation |
| `TINY_TASK_START` | Start index for the tiny slice | Influences the overfit diagnostic set |
| `TINY_TASK_END` | End index for the tiny slice | Influences the overfit diagnostic set |
| `RUN_TIMEOUT_SECONDS` | Per-execution timeout | Influences how long generated code is allowed to run |
| `RUN_CPU_SECONDS` | CPU time limit for subprocess execution | Influences execution sandbox strictness |
| `RUN_MEMORY_MB` | Memory limit for subprocess execution | Influences execution sandbox strictness |
| `USE_RESOURCE_LIMITS` | Enables subprocess resource limits | Works with `RUN_CPU_SECONDS` and `RUN_MEMORY_MB` |
| `STRICT_AFTER_CODE` | Treats any text after `</code>` as bad | Influences formatting reward and penalties |
| `EXTRA_TEXT_PENALTY` | Penalty for extra text after `</code>` | Influences reward total and brevity behavior |
| `EVAL_TEMPERATURE` | Temperature used in eval generation | Influences eval output diversity |
| `EVAL_TOP_P` | Top-p used in eval generation | Influences eval output diversity |
| `WORLD_SIZE` | Number of distributed workers | Used with `accelerate launch`; influences distributed behavior |
| `CUDA_VISIBLE_DEVICES` | Which GPUs are visible to the job | Directly controls which GPUs the job can use |
| `CLEAN_OUTPUT_DIRS` | Deletes output dirs before training | Helpful for fresh training runs, dangerous for reuse if misunderstood |

## Practical Notes

- The queue launcher is designed for the remote path `/home/cang688/Unsloth-RL-2`.
- Qwen training uses both RTX 3090 GPUs.
- Base and LoRA evals are kept separate from the training jobs.
- The script writes dashboards after evals and training runs.
- If PyTorch cannot see CUDA, `unsloth` will fail immediately during import.

## Troubleshooting

If a Qwen run fails with a GPU error, check:

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If `torch.version.cuda` is `None`, the env is using a CPU-only PyTorch build and must be fixed before training will run.

If a LoRA eval says the path is missing, make sure `LORA_PATH` points to the `lora/` directory inside a completed training run.

## Suggested Workflow

For Qwen, the recommended progression is:

1. `E5` sanity check
2. `E6` tiny overfit
3. `E7` base holdout eval
4. `E8` full dual-GPU training
5. `E9` trained LoRA holdout eval

That gives you a fast correctness check before you spend longer compute on the full run.
