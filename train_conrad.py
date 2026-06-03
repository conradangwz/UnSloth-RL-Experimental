"""
train_conrad.py

GRPO training for Qwen3.6 with Unsloth.

Main modes:
- train
- eval_base
- eval_base_holdout
- eval_lora
- eval_lora_holdout
- sanity_reward
- tiny_overfit_train
- eval_lora_tiny

Example usage:

1. Check reward/evaluation code:
   TEST_MODE=sanity_reward python train_conrad.py

2. Evaluate base model on training tasks 0-49:
   TEST_MODE=eval_base python train_conrad.py

3. Train normally:
   TEST_MODE=train python train_conrad.py

4. Evaluate trained LoRA on training tasks 0-49:
   TEST_MODE=eval_lora LORA_PATH=grpo_mvp_lora python train_conrad.py

5. Evaluate trained LoRA on unseen holdout tasks 50-99:
   TEST_MODE=eval_lora_holdout LORA_PATH=grpo_mvp_lora python train_conrad.py

6. Tiny overfit test:
   TEST_MODE=tiny_overfit_train python train_conrad.py

7. Evaluate tiny overfit LoRA:
   TEST_MODE=eval_lora_tiny LORA_PATH=grpo_mvp_lora python train_conrad.py
"""

import re
import shutil
import subprocess
import sys
import tempfile
import random
from datetime import datetime
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

import sys
import os
import shutil
import csv
import json
import tempfile
from datetime import datetime

from unsloth import FastModel
import torch

try:
    import transformers.utils.hub as transformers_hub
    from huggingface_hub.constants import HF_HUB_CACHE

    if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
        transformers_hub.TRANSFORMERS_CACHE = str(HF_HUB_CACHE)
except Exception:
    pass

from trl import GRPOConfig, GRPOTrainer


# ── 0. Config helpers ──────────────────────────────────────────────────────────

def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_main_process() -> bool:
    return env_int("LOCAL_RANK", 0) == 0


def print_main(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 0. Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3.6-27B")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "1") == "1"
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "0") == "1"
LOAD_IN_16BIT = os.getenv("LOAD_IN_16BIT", "0") == "1"
FULL_FINETUNING = False

if LOAD_IN_4BIT or LOAD_IN_8BIT:
    LOAD_IN_16BIT = False

LORA_RANK = int(os.getenv("LORA_RANK", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_RANK)))
LORA_TARGET_MODULES = os.getenv("LORA_TARGET_MODULES", "all-linear")
USE_CHAT_TEMPLATE = os.getenv("USE_CHAT_TEMPLATE", "1") == "1"
QWEN_ENABLE_THINKING = os.getenv("QWEN_ENABLE_THINKING", "0") == "1"

TEST_MODE = env_str("TEST_MODE", "train")
VALID_TEST_MODES = {
    "train",
    "eval_base",
    "eval_base_holdout",
    "eval_lora",
    "eval_lora_holdout",
    "sanity_reward",
    "tiny_overfit_train",
    "eval_lora_tiny",
    "eval_all",
}

if TEST_MODE not in VALID_TEST_MODES:
    raise ValueError(f"Unknown TEST_MODE={TEST_MODE}. Valid modes: {sorted(VALID_TEST_MODES)}")

MODEL_NAME = env_str("MODEL_NAME", "codellama/CodeLlama-7b-Python-hf")
RUN_NAME = env_str("RUN_NAME", f"{TEST_MODE}_{MODEL_NAME.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
RUN_ROOT = Path(env_str("RUN_ROOT", str(Path("grpo_runs") / RUN_NAME)))
LOG_DIR = Path(env_str("LOG_DIR", str(RUN_ROOT / "logs")))
OUTPUT_DIR = Path(env_str("OUTPUT_DIR", str(RUN_ROOT / "output")))
LORA_OUTPUT_DIR = Path(env_str("LORA_OUTPUT_DIR", str(RUN_ROOT / "lora")))
LORA_PATH = Path(env_str("LORA_PATH", str(LORA_OUTPUT_DIR)))

CLEAN_OUTPUT_DIRS = env_bool("CLEAN_OUTPUT_DIRS", False)

SEED = env_int("SEED", 3407)
set_seed(SEED)

WORLD_SIZE = env_int("WORLD_SIZE", 1)

NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
EVAL_NUM_GENERATIONS = int(os.getenv("EVAL_NUM_GENERATIONS", "2"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", str(NUM_GENERATIONS))
)
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))

MAX_STEPS = int(os.getenv("MAX_STEPS", "500"))
TINY_MAX_STEPS = int(os.getenv("TINY_MAX_STEPS", "200"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "50"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "1"))

MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1024"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "384"))

LEARNING_RATE = env_float("LEARNING_RATE", 5e-6)
WARMUP_RATIO = env_float("WARMUP_RATIO", 0.03)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.0)

LOAD_IN_4BIT = env_bool("LOAD_IN_4BIT", True)

LORA_R = env_int("LORA_R", 16)
LORA_ALPHA = env_int("LORA_ALPHA", 16)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.0)

# Use fuller target modules by default. This helps Qwen/CodeLlama more than q/v only.
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
TARGET_MODULES = [
    module.strip()
    for module in env_str("LORA_TARGET_MODULES", DEFAULT_TARGET_MODULES).split(",")
    if module.strip()
]

TRAIN_TASK_START = env_int("TRAIN_TASK_START", 0)
TRAIN_TASK_END = env_int("TRAIN_TASK_END", 131)

HOLDOUT_TASK_START = env_int("HOLDOUT_TASK_START", 131)
HOLDOUT_TASK_END = env_int("HOLDOUT_TASK_END", 164)

TINY_TASK_START = env_int("TINY_TASK_START", 0)
TINY_TASK_END = env_int("TINY_TASK_END", 5)

RUN_TIMEOUT_SECONDS = env_int("CODE_RUN_TIMEOUT_SECONDS", 8)
RUN_CPU_SECONDS = env_int("CODE_RUN_CPU_SECONDS", 8)
RUN_MEMORY_MB = env_int("CODE_RUN_MEMORY_MB", 0)
USE_RESOURCE_LIMITS = env_bool("CODE_RUN_USE_RESOURCE_LIMITS", False)

STRICT_AFTER_CODE = env_bool("STRICT_AFTER_CODE", True)
EXTRA_TEXT_PENALTY = env_float("EXTRA_TEXT_PENALTY", 0.2)

EVAL_TEMPERATURE = env_float("EVAL_TEMPERATURE", 0.2)
EVAL_TOP_P = env_float("EVAL_TOP_P", 0.95)
TRAIN_TEMPERATURE_NOTE = env_str("TRAIN_TEMPERATURE_NOTE", "GRPOTrainer controls generation internally.")

LOG_COMPLETIONS_ON_ALL_RANKS = env_bool("LOG_COMPLETIONS_ON_ALL_RANKS", False)


SAFE_EXEC_PREAMBLE = """from typing import *
import functools
from functools import reduce
import math
import re
import itertools
import collections
import heapq
import bisect
import string
"""


# ── 2. Run folders / logging ───────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LORA_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

if CLEAN_OUTPUT_DIRS and TEST_MODE in {"train", "tiny_overfit_train"} and is_main_process():
    for path in [OUTPUT_DIR, LORA_OUTPUT_DIR]:
        if path.exists():
            shutil.rmtree(path)
            print_main(f"Cleaned {path}/")
        path.mkdir(parents=True, exist_ok=True)

JSONL_LOG_PATH = LOG_DIR / "samples.jsonl"
CSV_LOG_PATH = LOG_DIR / "samples.csv"
RUN_CONFIG_PATH = RUN_ROOT / "run_config.json"

LOG_FIELDS = [
    "run_name",
    "run_root",
    "test_mode",
    "model_name",
    "seed",
    "world_size",
    "step",
    "completion_index",
    "task_id",
    "prompt",
    "raw_completion",
    "extracted_code",
    "format_status",
    "compile_status",
    "runtime_status",
    "failure_reason",
    "error_message",
    "stderr_preview",
    "test_status",
    "test_error_message",
    "test_stderr_preview",
    "reward_format",
    "reward_compile",
    "reward_tests",
    "reward_extra_text_penalty",
    "reward_total",
    "prompt_token_length",
    "completion_token_length",
]

LOG_STATE = {"rows_written": 0}
DIAGNOSTIC_CACHE = {}

if is_main_process():
    with CSV_LOG_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=LOG_FIELDS).writeheader()

print(f"TEST_MODE={TEST_MODE}")
print(f"MODEL_NAME={MODEL_NAME}")
print(
    "Model loading: "
    f"max_seq_length={MAX_SEQ_LENGTH}, "
    f"load_in_4bit={LOAD_IN_4BIT}, "
    f"load_in_8bit={LOAD_IN_8BIT}, "
    f"load_in_16bit={LOAD_IN_16BIT}, "
    f"full_finetuning={FULL_FINETUNING}, "
    "fast_inference=False"
)
print(
    "LoRA: "
    f"rank={LORA_RANK}, "
    f"alpha={LORA_ALPHA}, "
    f"target_modules={LORA_TARGET_MODULES}"
)
print(
    "Prompting: "
    f"use_chat_template={USE_CHAT_TEMPLATE}, "
    f"qwen_enable_thinking={QWEN_ENABLE_THINKING}"
)
print(
    "Run sizing: "
    f"num_generations={NUM_GENERATIONS}, "
    f"eval_num_generations={EVAL_NUM_GENERATIONS}, "
    f"per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE}, "
    f"gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS}, "
    f"max_prompt_length={MAX_PROMPT_LENGTH}, "
    f"max_completion_length={MAX_COMPLETION_LENGTH}, "
    f"max_steps={MAX_STEPS}, "
    f"tiny_max_steps={TINY_MAX_STEPS}"
)
print(f"Logging completions to {JSONL_LOG_PATH} and {CSV_LOG_PATH}")


# ── 2. Clean outputs only when training ────────────────────────────────────────
# Important: do NOT delete grpo_mvp_lora when evaluating LoRA.

if TEST_MODE in ["train", "tiny_overfit_train"]:
    for d in ["grpo_mvp_output", "grpo_mvp_lora"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleaned {d}/")


# ── 3. Model loading ───────────────────────────────────────────────────────────

def parse_lora_target_modules(value: str):
    if value in {"all-linear", "all_linear", "all"}:
        return "all-linear"
    return [module.strip() for module in value.split(",") if module.strip()]


def ensure_warnings_issued_attr(loaded_model, max_depth: int = 5):
    seen = set()
    queue = [(loaded_model, 0)]

    while queue:
        candidate, depth = queue.pop(0)
        if candidate is None:
            continue

        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        try:
            if not hasattr(candidate, "warnings_issued"):
                candidate.warnings_issued = {}
        except Exception:
            pass

        if depth >= max_depth:
            continue

        for attr_name in ["base_model", "model", "module"]:
            try:
                child = getattr(candidate, attr_name, None)
            except Exception:
                child = None
            if child is not None:
                queue.append((child, depth + 1))


model = None
tokenizer = None

if TEST_MODE != "sanity_reward":
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        load_in_16bit=LOAD_IN_16BIT,
        full_finetuning=FULL_FINETUNING,
        fast_inference=False,
    )

    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    model = FastModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=parse_lora_target_modules(LORA_TARGET_MODULES),
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    ensure_warnings_issued_attr(model)


# ── 4. Prompt format ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Python coding assistant.
Return only a complete valid Python solution, then immediately close with </code>.
Do not explain.
Do not include markdown.
Do not include text after </code>.
Do not include examples after the solution."""


def apply_qwen_chat_template(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=QWEN_ENABLE_THINKING,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{prompt_text}<code>\n"


def format_prompt(example):
    user_prompt = f"### Instruction:\n{example['prompt']}\n\n### Response:"

    if USE_CHAT_TEMPLATE:
        prompt_text = apply_qwen_chat_template(user_prompt)
    else:
        prompt_text = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n<code>\n"

    return {"prompt": prompt_text}


# ── 5. Dataset splits ──────────────────────────────────────────────────────────

if TEST_MODE == "sanity_reward":
    raw_dataset = None
    train_dataset = None
    holdout_dataset = None
    tiny_dataset = None
    dataset = None
else:
    raw_dataset = load_dataset("openai/openai_humaneval", split="test")

    train_raw = raw_dataset.select(range(TRAIN_TASK_START, TRAIN_TASK_END))
    holdout_raw = raw_dataset.select(range(HOLDOUT_TASK_START, HOLDOUT_TASK_END))
    tiny_raw = raw_dataset.select(range(TINY_TASK_START, TINY_TASK_END))

    train_dataset = train_raw.map(format_prompt)
    holdout_dataset = holdout_raw.map(format_prompt)
    tiny_dataset = tiny_raw.map(format_prompt)

    if TEST_MODE in ["tiny_overfit_train", "eval_lora_tiny"]:
        dataset = tiny_dataset
    elif TEST_MODE in ["eval_base_holdout", "eval_lora_holdout"]:
        dataset = holdout_dataset
    else:
        dataset = train_dataset

    print("\n" + "=" * 60)
    print("DEBUG - raw prompt the model will receive:")
    print("=" * 60)
    print(dataset[0]["prompt"])
    print("=" * 60 + "\n")


# ── 6. Code extraction / format diagnostics ────────────────────────────────────

BAD_AFTER_CODE_PATTERNS = [
    r"```",
    r"###",
    r"<\/?p>",
    r"<\/?details>",
    r"<\/?summary>",
    r"<\/?textarea>",
    r"<script",
    r"Explanation:",
    r"Solution:",
    r"Score:",
    r"Test:",
]


def normalize_completion(completion) -> str:
    if isinstance(completion, list):
        return " ".join(
            message["content"] if isinstance(message, dict) and "content" in message else str(message)
            for message in completion
        )
    return str(completion)


def extract_code(text: str) -> str | None:
    if isinstance(text, list):
        text = normalize_completion(text)

    if "</code>" not in text:
        return None

    before_close = text.split("</code>", 1)[0].strip()

    if "<code>" in before_close:
        before_close = before_close.split("<code>")[-1].strip()

    return before_close if len(before_close.strip()) >= 5 else None


def text_after_code(text: str) -> str:
    if "</code>" not in text:
        return ""
    return text.split("</code>", 1)[1].strip()


def has_bad_after_code_text(text: str) -> bool:
    after = text_after_code(text)
    if not after:
        return False

    if STRICT_AFTER_CODE:
        return True

    for pattern in BAD_AFTER_CODE_PATTERNS:
        if re.search(pattern, after, flags=re.IGNORECASE):
            return True
    return False


# ── 7. Safety/blocklist ────────────────────────────────────────────────────────

BLOCKED_PATTERNS = [
    r"\bos\s*\.\s*system\b",
    r"\bsubprocess\b",
    r"\bshutil\s*\.\s*rmtree\b",
    r"\bos\s*\.\s*remove\b",
    r"\bos\s*\.\s*unlink\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bopen\s*\(.*['\"]w['\"]",
    r"while\s+True\s*:",
    r"\bsocket\b",
    r"\brequests\b",
    r"\burllib\b",
]


def matched_blocked_pattern(code: str) -> str | None:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return pattern
    return None


def set_subprocess_limits():
    if resource is None or not USE_RESOURCE_LIMITS:
        return

    resource.setrlimit(resource.RLIMIT_CPU, (RUN_CPU_SECONDS, RUN_CPU_SECONDS))

    if RUN_MEMORY_MB > 0:
        memory_bytes = RUN_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def run_python_script(script: str) -> subprocess.CompletedProcess:
    preexec_fn = set_subprocess_limits if resource is not None and USE_RESOURCE_LIMITS else None

    with tempfile.TemporaryDirectory(prefix="code_reward_") as tmpdir:
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmpdir,
            timeout=RUN_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            preexec_fn=preexec_fn,
        )


# ── 8. Token helpers ───────────────────────────────────────────────────────────

def token_count(text: str) -> int:
    try:
        encoder = tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer
        return len(encoder.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def value_at(kwargs: dict, key: str, index: int, default: str = ""):
    value = kwargs.get(key, default)

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        if index < len(value):
            return value[index]
        return value[index % len(value)]

    return value


# ── 9. Completion analysis / reward diagnostics ───────────────────────────────

def analyze_completion(completion, test: str = "", entry_point: str = "") -> dict:
    text = normalize_completion(completion)
    format_ok = "</code>" in text
    code = extract_code(text)

    extra_text_penalty = EXTRA_TEXT_PENALTY if has_bad_after_code_text(text) else 0.0

    diagnostic = {
        "raw_completion": text,
        "extracted_code": code or "",
        "format_status": "ok" if format_ok else "missing_closing_tag",
        "compile_status": "not_checked",
        "runtime_status": "not_run",
        "failure_reason": "",
        "error_message": "",
        "stderr_preview": "",
        "test_status": "not_run",
        "test_error_message": "",
        "test_stderr_preview": "",
        "reward_format": 1.0 if format_ok else 0.0,
        "reward_compile": 0.0,
        "reward_tests": 0.0,
        "reward_extra_text_penalty": extra_text_penalty,
    }

    if extra_text_penalty > 0:
        diagnostic["format_status"] = "extra_text_after_code"

    if code is None:
        diagnostic["failure_reason"] = "no_code_extracted"
        return diagnostic

    blocked_pattern = matched_blocked_pattern(code)
    if blocked_pattern:
        diagnostic["compile_status"] = "blocked"
        diagnostic["failure_reason"] = "blocked_pattern"
        diagnostic["error_message"] = blocked_pattern
        return diagnostic

    try:
        compile(code, "<string>", "exec")
        diagnostic["compile_status"] = "ok"
    except SyntaxError as exc:
        diagnostic["compile_status"] = "syntax_error"
        diagnostic["failure_reason"] = "syntax_error"
        diagnostic["error_message"] = str(exc)
        return diagnostic

    try:
        result = run_python_script(f"{SAFE_EXEC_PREAMBLE}\n{code}")

        if result.returncode == 0:
            diagnostic["runtime_status"] = "ok"
            diagnostic["reward_compile"] = 1.0
        else:
            diagnostic["runtime_status"] = "runtime_error"
            diagnostic["failure_reason"] = "runtime_error"
            diagnostic["error_message"] = f"returncode={result.returncode}"
            diagnostic["stderr_preview"] = result.stderr[:500]
            diagnostic["reward_compile"] = 0.5

    except subprocess.TimeoutExpired:
        diagnostic["runtime_status"] = "timeout"
        diagnostic["failure_reason"] = "timeout"

    except Exception as exc:
        diagnostic["runtime_status"] = "exception"
        diagnostic["failure_reason"] = "exception"
        diagnostic["error_message"] = str(exc)

    if diagnostic["runtime_status"] != "ok" or not test or not entry_point:
        return diagnostic

    test_script = f"{SAFE_EXEC_PREAMBLE}\n{code}\n\n{test}\n\ncheck({entry_point})"

    try:
        result = run_python_script(test_script)

        if result.returncode == 0:
            diagnostic["test_status"] = "passed"
            diagnostic["reward_tests"] = 1.0
        else:
            diagnostic["test_status"] = "failed"
            diagnostic["test_error_message"] = f"returncode={result.returncode}"
            diagnostic["test_stderr_preview"] = result.stderr[:500]

    except subprocess.TimeoutExpired:
        diagnostic["test_status"] = "timeout"
        diagnostic["test_error_message"] = "timeout"

    except Exception as exc:
        diagnostic["test_status"] = "exception"
        diagnostic["test_error_message"] = str(exc)

    return diagnostic


def total_reward_from_diagnostic(diagnostic: dict) -> float:
    total = (
        0.2 * diagnostic["reward_format"]
        + 0.8 * diagnostic["reward_compile"]
        + 5.0 * diagnostic["reward_tests"]
        - diagnostic.get("reward_extra_text_penalty", 0.0)
    )
    return max(0.0, float(total))


def get_completion_diagnostic(index: int, completion, kwargs: dict) -> dict:
    text = normalize_completion(completion)
    test = value_at(kwargs, "test", index, "")
    entry_point = value_at(kwargs, "entry_point", index, "")
    cache_key = (text, test, entry_point)

    if cache_key not in DIAGNOSTIC_CACHE:
        DIAGNOSTIC_CACHE[cache_key] = analyze_completion(
            completion,
            test=test,
            entry_point=entry_point,
        )

    return DIAGNOSTIC_CACHE[cache_key]


# ── 10. Logging ────────────────────────────────────────────────────────────────

def write_sample_log(row: dict) -> None:
    if not is_main_process() and not LOG_COMPLETIONS_ON_ALL_RANKS:
        return

    with JSONL_LOG_PATH.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with CSV_LOG_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def log_completion(index: int, completion, diagnostic: dict, kwargs: dict) -> None:
    row_number = LOG_STATE["rows_written"]
    prompt = value_at(kwargs, "prompts", index, value_at(kwargs, "prompt", index, ""))
    reward_total = total_reward_from_diagnostic(diagnostic)

    row = {
        "run_name": RUN_NAME,
        "run_root": str(RUN_ROOT),
        "test_mode": TEST_MODE,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "world_size": WORLD_SIZE,
        "step": row_number // NUM_GENERATIONS,
        "completion_index": row_number % NUM_GENERATIONS,
        "task_id": value_at(kwargs, "task_id", index, ""),
        "prompt": prompt,
        "raw_completion": diagnostic["raw_completion"],
        "extracted_code": diagnostic["extracted_code"],
        "format_status": diagnostic["format_status"],
        "compile_status": diagnostic["compile_status"],
        "runtime_status": diagnostic["runtime_status"],
        "failure_reason": diagnostic["failure_reason"],
        "error_message": diagnostic["error_message"],
        "stderr_preview": diagnostic["stderr_preview"],
        "test_status": diagnostic["test_status"],
        "test_error_message": diagnostic["test_error_message"],
        "test_stderr_preview": diagnostic["test_stderr_preview"],
        "reward_format": diagnostic["reward_format"],
        "reward_compile": diagnostic["reward_compile"],
        "reward_tests": diagnostic["reward_tests"],
        "reward_extra_text_penalty": diagnostic.get("reward_extra_text_penalty", 0.0),
        "reward_total": reward_total,
        "prompt_token_length": token_count(prompt),
        "completion_token_length": token_count(normalize_completion(completion)),
    }

    write_sample_log(row)
    LOG_STATE["rows_written"] += 1


# ── 11. Reward function for GRPO ───────────────────────────────────────────────

def reward_combined(completions, **kwargs) -> list[float]:
    scores = []

    for index, completion in enumerate(completions):
        diagnostic = get_completion_diagnostic(index, completion, kwargs)
        score = total_reward_from_diagnostic(diagnostic)
        scores.append(score)

        if is_main_process():
            preview = diagnostic["raw_completion"][:220].replace("\n", "|")
            print(f"  [OUTPUT] {preview}")
            print(
                "  [COMBINED] "
                f"score={score} "
                f"format={diagnostic['reward_format']} "
                f"compile={diagnostic['reward_compile']} "
                f"tests={diagnostic['reward_tests']} "
                f"extra_penalty={diagnostic.get('reward_extra_text_penalty', 0.0)} "
                f"compile_status={diagnostic['compile_status']} "
                f"runtime_status={diagnostic['runtime_status']} "
                f"test_status={diagnostic['test_status']} "
                f"reason={diagnostic['failure_reason'] or 'none'}"
            )

        log_completion(index, completion, diagnostic, kwargs)

    return scores


# ── 12. Training config ────────────────────────────────────────────────────────

def make_training_args(mode: str) -> GRPOConfig:
    max_steps = TINY_MAX_STEPS if mode == "tiny_overfit_train" else MAX_STEPS

    return GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        report_to="none",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
    )


training_args = make_training_args(TEST_MODE)


# ── 13. Evaluation generation ─────────────────────────────────────────────────

def generate_completion(prompt: str, max_new_tokens: int = MAX_COMPLETION_LENGTH) -> str:
    try:
        FastModel.for_inference(model)
    except Exception:
        pass

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    outputs = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    completion_ids = outputs[0][input_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def evaluate_model(eval_dataset, label: str, output_path: Path):
    rows = []

    print_main("\n" + "=" * 60)
    print_main(f"Starting evaluation: {label}")
    print_main("=" * 60)

    for item_idx, example in enumerate(eval_dataset):
        prompt = example["prompt"]
        test = example["test"]
        entry_point = example["entry_point"]
        task_id = example.get("task_id", f"task_{item_idx}")

        for gen_idx in range(EVAL_NUM_GENERATIONS):
            completion = generate_completion(prompt, max_new_tokens=MAX_COMPLETION_LENGTH)

            diagnostic = analyze_completion(
                completion,
                test=test,
                entry_point=entry_point,
            )

            reward_total = total_reward_from_diagnostic(diagnostic)

            row = {
                "label": label,
                "run_name": RUN_NAME,
                "run_root": str(RUN_ROOT),
                "model_name": MODEL_NAME,
                "lora_path": str(LORA_PATH) if "lora" in label else "",
                "task_id": task_id,
                "item_idx": item_idx,
                "generation_idx": gen_idx,
                "raw_completion": diagnostic["raw_completion"],
                "extracted_code": diagnostic["extracted_code"],
                "format_status": diagnostic["format_status"],
                "compile_status": diagnostic["compile_status"],
                "runtime_status": diagnostic["runtime_status"],
                "failure_reason": diagnostic["failure_reason"],
                "error_message": diagnostic["error_message"],
                "stderr_preview": diagnostic["stderr_preview"],
                "test_status": diagnostic["test_status"],
                "test_error_message": diagnostic["test_error_message"],
                "test_stderr_preview": diagnostic["test_stderr_preview"],
                "reward_format": diagnostic["reward_format"],
                "reward_compile": diagnostic["reward_compile"],
                "reward_tests": diagnostic["reward_tests"],
                "reward_extra_text_penalty": diagnostic.get("reward_extra_text_penalty", 0.0),
                "reward_total": reward_total,
                "completion_token_length": token_count(completion),
            }

            rows.append(row)

            print_main(
                f"[{label}] "
                f"task={task_id} "
                f"gen={gen_idx} "
                f"reward={reward_total} "
                f"format={diagnostic['format_status']} "
                f"compile={diagnostic['compile_status']} "
                f"runtime={diagnostic['runtime_status']} "
                f"test={diagnostic['test_status']} "
                f"reason={diagnostic['failure_reason'] or 'none'}"
            )

    if not is_main_process():
        return

    if not rows:
        print("No rows generated.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["test_status"] == "passed")
    format_ok = sum(1 for row in rows if row["format_status"] == "ok")
    compile_ok = sum(1 for row in rows if row["compile_status"] == "ok")
    avg_reward = sum(float(row["reward_total"]) for row in rows) / len(rows)
    avg_len = sum(int(row["completion_token_length"]) for row in rows) / len(rows)

    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)

    pass_at_any = sum(
        any(row["test_status"] == "passed" for row in task_rows)
        for task_rows in by_task.values()
    )

    summary = {
        "label": label,
        "rows": len(rows),
        "tasks": len(by_task),
        "avg_reward": avg_reward,
        "test_pass_rate_generation_level": passed / len(rows),
        "pass_at_any_generation_task_level": pass_at_any / len(by_task) if by_task else 0.0,
        "format_ok_rate": format_ok / len(rows),
        "compile_ok_rate": compile_ok / len(rows),
        "avg_completion_length": avg_len,
        "output_csv": str(output_path),
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Evaluation summary: {label}")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved to: {output_path}")
    print(f"Saved summary to: {summary_path}")


# ── 14. Reward/evaluator sanity check ──────────────────────────────────────────

def sanity_check_reward_code():
    raw = load_dataset("openai/openai_humaneval", split="test")
    example = raw[0]

    correct_code = example["prompt"] + example["canonical_solution"]

    correct_completion = correct_code + "\n</code>"
    syntax_error_completion = "def broken(:\n    pass\n</code>"
    missing_tag_completion = "def whatever():\n    return 1"

    cases = [
        ("correct_canonical_solution", correct_completion),
        ("syntax_error", syntax_error_completion),
        ("missing_closing_tag", missing_tag_completion),
    ]

    for name, completion in cases:
        diagnostic = analyze_completion(
            completion,
            test=example["test"],
            entry_point=example["entry_point"],
        )

        reward_total = total_reward_from_diagnostic(diagnostic)

        print("\n" + "-" * 60)
        print(name)
        print("-" * 60)
        print(f"format_status: {diagnostic['format_status']}")
        print(f"compile_status: {diagnostic['compile_status']}")
        print(f"runtime_status: {diagnostic['runtime_status']}")
        print(f"test_status: {diagnostic['test_status']}")
        print(f"failure_reason: {diagnostic['failure_reason'] or 'none'}")
        print(f"reward_format: {diagnostic['reward_format']}")
        print(f"reward_compile: {diagnostic['reward_compile']}")
        print(f"reward_tests: {diagnostic['reward_tests']}")
        print(f"reward_extra_text_penalty: {diagnostic.get('reward_extra_text_penalty', 0.0)}")
        print(f"reward_total: {reward_total}")

    print("\nExpected:")
    print("correct_canonical_solution should get reward_total = 6.0")
    print("syntax_error should not pass compile/tests")
    print("missing_closing_tag should get no_code_extracted")


# ── 15. LoRA loading for eval modes ────────────────────────────────────────────

def load_lora_for_eval():
    global model

    if not LORA_PATH.exists():
        raise FileNotFoundError(
            f"LoRA path not found: {LORA_PATH}. "
            f"Train first or pass LORA_PATH=/path/to/lora."
        )

    from peft import PeftModel

    print(f"Loading LoRA from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    ensure_warnings_issued_attr(model)
    model.eval()

    try:
        FastModel.for_inference(model)
    except Exception:
        pass


# ── 16. Main switch ────────────────────────────────────────────────────────────

if TEST_MODE == "sanity_reward":
    sanity_check_reward_code()
    sys.exit(0)


if TEST_MODE == "eval_base":
    evaluate_model(
        train_dataset,
        label=f"base_train_tasks_{TRAIN_TASK_START}_{TRAIN_TASK_END - 1}",
        output_path=LOG_DIR / "eval_base_train_tasks.csv",
    )
    sys.exit(0)


if TEST_MODE == "eval_base_holdout":
    evaluate_model(
        holdout_dataset,
        label=f"base_holdout_tasks_{HOLDOUT_TASK_START}_{HOLDOUT_TASK_END - 1}",
        output_path=LOG_DIR / "eval_base_holdout_tasks.csv",
    )
    sys.exit(0)


if TEST_MODE == "eval_lora":
    load_lora_for_eval()
    evaluate_model(
        train_dataset,
        label=f"lora_train_tasks_{TRAIN_TASK_START}_{TRAIN_TASK_END - 1}",
        output_path=LOG_DIR / "eval_lora_train_tasks.csv",
    )
    sys.exit(0)


if TEST_MODE == "eval_lora_holdout":
    load_lora_for_eval()
    evaluate_model(
        holdout_dataset,
        label=f"lora_holdout_tasks_{HOLDOUT_TASK_START}_{HOLDOUT_TASK_END - 1}",
        output_path=LOG_DIR / "eval_lora_holdout_tasks.csv",
    )
    sys.exit(0)


if TEST_MODE == "eval_lora_tiny":
    load_lora_for_eval()
    evaluate_model(
        tiny_dataset,
        label=f"lora_tiny_tasks_{TINY_TASK_START}_{TINY_TASK_END - 1}",
        output_path=LOG_DIR / "eval_lora_tiny_tasks.csv",
    )
    sys.exit(0)


if TEST_MODE == "eval_all":
    evaluate_model(
        train_dataset,
        label=f"base_train_tasks_{TRAIN_TASK_START}_{TRAIN_TASK_END - 1}",
        output_path=LOG_DIR / "eval_base_train_tasks.csv",
    )
    evaluate_model(
        holdout_dataset,
        label=f"base_holdout_tasks_{HOLDOUT_TASK_START}_{HOLDOUT_TASK_END - 1}",
        output_path=LOG_DIR / "eval_base_holdout_tasks.csv",
    )
    if LORA_PATH.exists():
        load_lora_for_eval()
        evaluate_model(
            train_dataset,
            label=f"lora_train_tasks_{TRAIN_TASK_START}_{TRAIN_TASK_END - 1}",
            output_path=LOG_DIR / "eval_lora_train_tasks.csv",
        )
        evaluate_model(
            holdout_dataset,
            label=f"lora_holdout_tasks_{HOLDOUT_TASK_START}_{HOLDOUT_TASK_END - 1}",
            output_path=LOG_DIR / "eval_lora_holdout_tasks.csv",
        )
    sys.exit(0)


if TEST_MODE in {"train", "tiny_overfit_train"}:
    if TEST_MODE == "tiny_overfit_train":
        print_main("Running tiny overfit training.")
        print_main(f"TINY_MAX_STEPS={TINY_MAX_STEPS}")
    else:
        print_main(f"Running normal training on HumanEval tasks {TRAIN_TASK_START}-{TRAIN_TASK_END - 1}.")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[reward_combined],
        train_dataset=dataset,
    )

    print_main("Starting training.")
    print_main("Good sign: more test_status=passed and score=6.0 over time.")
    print_main(f"Full completion diagnostics: {JSONL_LOG_PATH}")
    print_main(f"Trainer output: {OUTPUT_DIR}")

    trainer.train()

    if is_main_process():
        LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(LORA_OUTPUT_DIR))
        tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))

        print("Done.")
        print(f"Saved LoRA to {LORA_OUTPUT_DIR}")
        print(f"Run root: {RUN_ROOT}")


raise ValueError(f"Unknown TEST_MODE: {TEST_MODE}")
