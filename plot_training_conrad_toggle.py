import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from html import escape


STATUS_COLUMNS = [
    "format_status",
    "compile_status",
    "runtime_status",
    "test_status",
]

REWARD_COLUMNS = [
    "reward_format",
    "reward_syntax",
    "reward_style",
    "reward_compile",
    "reward_tests",
    "reward_extra_text_penalty",
    "reward_total",
]

TOKEN_COLUMNS = [
    "prompt_token_length",
    "completion_token_length",
]


IMPORTANT_METRIC_DESCRIPTIONS = [
    ("reward_total / trainer reward", "Overall reward signal. In this setup the maximum is normally 6.0: 0.2 format, 0.8 compile/runtime, and 5.0 tests, minus any extra-text penalty."),
    ("reward_format", "Small reward for following the requested output format, especially ending with </code>."),
    ("reward_compile", "Reward for code that can be extracted, parsed, compiled, and run without immediate failure."),
    ("reward_tests", "Main quality reward. This is awarded when the generated solution passes HumanEval tests."),
    ("reward_syntax", "Small reward for parseable Python in style-RL runs."),
    ("reward_style", "Main style-RL reward, computed from Ruff E/W rule compliance."),
    ("style_score", "Normalised style score from Ruff violations. Higher means cleaner conventional Python style."),
    ("style_violation_count", "Number of Ruff style violations found in the extracted code."),
    ("reward_extra_text_penalty", "Penalty for unwanted text after </code>, markdown, HTML, explanations, or other rambling."),
    ("format_ok_rate", "Share of completions where the model followed the format cleanly."),
    ("compile_ok_rate", "Share of completions that compile successfully."),
    ("runtime_ok_rate", "Share of completions that run without crashing."),
    ("test_pass_rate", "Share of completions that pass tests. This is the most important training outcome metric."),
    ("score_6_rate", "Share of completions receiving perfect reward. This should roughly match test_pass_rate."),
    ("no_code_rate", "Share of completions where no valid code could be extracted."),
    ("syntax_error_rate", "Share of completions that failed with Python syntax errors."),
    ("timeout_rate", "Share of completions that timed out during execution/testing."),
    ("extra_text_rate", "Share of completions with extra text after </code>. This should go down."),
    ("loss", "GRPO optimisation loss. Useful for stability, but not the main success metric."),
    ("kl", "How far the policy is moving from the reference/base model."),
    ("grad_norm", "Gradient magnitude. Spikes can indicate unstable updates."),
    ("learning_rate", "Current learning rate from the schedule."),
    ("completion_length / mean_length", "Average generated answer length. Very high values usually mean rambling or poor stop behaviour."),
    ("clipped_ratio", "Fraction of completions cut off at max length. This should usually be low."),
    ("reward_std / reward_total_std", "Reward variation between generations. GRPO needs variation to rank completions."),
    ("frac_reward_zero_std", "Fraction of batches where all completions got the same reward. High values mean weak ranking signal."),
    ("eval summaries", "Base/LoRA/holdout evaluation results. These are the best proof of generalisation."),
]


METRIC_CATEGORIES = {
    "Reward components": [
        "mean reward_total",
        "format reward",
        "syntax reward",
        "style reward",
        "compile reward",
        "tests reward",
        "extra text penalty",
        "reward_combined",
        "logged reward_total std",
    ],
    "Core success rates": [
        "format ok",
        "syntax ok",
        "style clean",
        "compile ok",
        "runtime ok",
        "tests passed",
        "perfect 6.0",
    ],
    "Failure/debug rates": [
        "no code extracted",
        "syntax error",
        "style violations",
        "style status",
        "style codes",
        "timeout",
        "extra text after code",
        "failure reasons",
        "test status",
        "compile status",
    ],
    "GRPO/trainer signals": [
        "trainer reward",
        "loss",
        "kl",
        "grad_norm",
        "learning_rate",
        "reward_std",
        "frac_reward_zero_std",
        "clipped_ratio",
    ],
    "Length/stop behaviour": [
        "completion_length",
        "mean_length",
        "max_length",
        "logged completion tokens",
        "logged prompt tokens",
    ],
    "Summary/evaluation tables": [
        "Training summary",
        "Evaluation summaries",
        "trace",
    ],
}


def metric_category(trace_name: str) -> str:
    name = trace_name.lower()
    for category, keywords in METRIC_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in name:
                return category
    return "Other"


def metric_descriptions_html() -> str:
    rows = []
    for metric, description in IMPORTANT_METRIC_DESCRIPTIONS:
        rows.append(
            "<tr>"
            f"<td><code>{escape(metric)}</code></td>"
            f"<td>{escape(description)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def wrap_dashboard_html(fig, output: Path) -> None:
    plot_div_id = "training-dashboard-plot"
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=plot_div_id,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )

    by_category: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for index, trace in enumerate(fig.data):
        name = getattr(trace, "name", None) or f"trace {index}"
        trace_type = getattr(trace, "type", "trace")
        category = metric_category(str(name))
        by_category[category].append((index, str(trace_type), str(name)))

    category_order = [
        "Reward components",
        "Core success rates",
        "Failure/debug rates",
        "GRPO/trainer signals",
        "Length/stop behaviour",
        "Summary/evaluation tables",
        "Other",
    ]

    sections = []
    for category in category_order:
        items = by_category.get(category, [])
        if not items:
            continue

        labels = []
        for index, trace_type, name in items:
            labels.append(
                f'<label class="metric-toggle-label" data-category="{escape(category)}" data-name="{escape(name.lower())}">'
                f'<input class="metric-toggle" type="checkbox" data-trace-index="{index}" data-category="{escape(category)}" checked> '
                f'<span class="trace-type">{escape(trace_type)}</span> {escape(name)}'
                '</label>'
            )

        sections.append(
            f"""
            <details class="metric-section" data-category="{escape(category)}" open>
                <summary>
                    <span>
                        <strong>{escape(category)}</strong>
                        <small>{len(items)} metrics</small>
                    </span>
                    <span class="section-actions">
                        <button type="button" class="section-show" data-category="{escape(category)}">Show section</button>
                        <button type="button" class="section-hide" data-category="{escape(category)}">Hide section</button>
                    </span>
                </summary>
                <div class="metric-toggle-grid">
                    {''.join(labels)}
                </div>
            </details>
            """
        )

    section_options = ['<option value="all">All sections</option>']
    for category in category_order:
        if by_category.get(category):
            section_options.append(f'<option value="{escape(category)}">{escape(category)}</option>')

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conrad GRPO Training Dashboard</title>
<style>
    body {{
        margin: 0;
        padding: 24px;
        font-family: Arial, Helvetica, sans-serif;
        background: #f7f9fc;
        color: #17233c;
    }}

    .dashboard-header {{
        max-width: 1500px;
        margin: 0 auto 16px auto;
    }}

    .dashboard-header h1 {{
        margin: 0 0 8px 0;
        font-size: 24px;
    }}

    .dashboard-header p {{
        margin: 0;
        color: #4a5875;
    }}

    .metric-controls {{
        max-width: 1500px;
        margin: 18px auto 24px auto;
        padding: 16px;
        background: white;
        border: 1px solid #d9e2f1;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20, 38, 70, 0.06);
    }}

    .metric-controls h2 {{
        margin: 0 0 8px 0;
        font-size: 18px;
    }}

    .controls-help {{
        margin: 0 0 12px 0;
        color: #4a5875;
        font-size: 14px;
    }}

    .control-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 12px;
    }}

    .control-row button,
    .control-row input,
    .control-row select,
    .section-actions button {{
        padding: 8px 10px;
        border: 1px solid #c7d3e6;
        border-radius: 8px;
        background: white;
        font-size: 14px;
    }}

    .control-row button,
    .section-actions button {{
        cursor: pointer;
    }}

    .metric-sections {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
    }}

    .metric-section {{
        border: 1px solid #e2e9f4;
        border-radius: 12px;
        background: #fbfdff;
        overflow: hidden;
    }}

    .metric-section[hidden] {{
        display: none;
    }}

    .metric-section summary {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        background: #eef4fc;
        cursor: pointer;
        list-style: none;
    }}

    .metric-section summary::-webkit-details-marker {{
        display: none;
    }}

    .metric-section summary strong {{
        display: block;
        font-size: 15px;
    }}

    .metric-section summary small {{
        color: #64748b;
        font-size: 12px;
    }}

    .section-actions {{
        display: flex;
        gap: 6px;
        flex-shrink: 0;
    }}

    .section-actions button {{
        padding: 6px 8px;
        font-size: 12px;
    }}

    .metric-toggle-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 8px 12px;
        max-height: 230px;
        overflow-y: auto;
        padding: 12px 14px 14px 14px;
    }}

    .metric-toggle-label {{
        display: block;
        font-size: 13px;
        line-height: 1.35;
        cursor: pointer;
        user-select: none;
        padding: 4px 6px;
        border-radius: 6px;
    }}

    .metric-toggle-label:hover {{
        background: #eef5ff;
    }}

    .metric-toggle-label[hidden] {{
        display: none;
    }}

    .trace-type {{
        display: inline-block;
        min-width: 42px;
        margin-right: 4px;
        color: #697891;
        font-size: 11px;
        text-transform: uppercase;
    }}

    .plot-wrap {{
        max-width: 1500px;
        margin: 0 auto;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20, 38, 70, 0.06);
    }}

    .metric-glossary {{
        max-width: 1500px;
        margin: 180px auto 80px auto;
        padding: 24px;
        background: white;
        border: 1px solid #d9e2f1;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20, 38, 70, 0.06);
    }}

    .metric-glossary h2 {{
        margin-top: 0;
        font-size: 22px;
    }}

    .metric-glossary table {{
        width: 100%;
        border-collapse: collapse;
    }}

    .metric-glossary th,
    .metric-glossary td {{
        text-align: left;
        vertical-align: top;
        padding: 10px 12px;
        border-bottom: 1px solid #edf1f7;
    }}

    .metric-glossary th {{
        background: #f1f5fb;
    }}

    code {{
        background: #eef2f8;
        padding: 2px 4px;
        border-radius: 4px;
    }}
</style>
</head>
<body>
<section class="dashboard-header">
    <h1>Conrad GRPO Training Dashboard</h1>
    <p>Use the sectioned controls to isolate rewards, success rates, failure modes, length behaviour, GRPO trainer signals, and eval comparisons.</p>
</section>

<section class="metric-controls">
    <h2>Sectioned metric toggles</h2>
    <p class="controls-help">Start with <strong>Core success rates</strong> and <strong>Reward components</strong>. Use failure, length, and trainer sections to debug why training is or is not improving.</p>
    <div class="control-row">
        <button id="show-all-metrics" type="button">Show all</button>
        <button id="hide-all-metrics" type="button">Hide all</button>
        <select id="section-filter" aria-label="Filter by section">
            {''.join(section_options)}
        </select>
        <input id="metric-filter" type="search" placeholder="Search metric name..." aria-label="Search metric name">
    </div>
    <div class="metric-sections" id="metric-sections">
        {''.join(sections)}
    </div>
</section>

<section class="plot-wrap">
    {plot_html}
</section>

<section class="metric-glossary">
    <h2>Important metric descriptions</h2>
    <p>These are the main metrics to use when deciding whether the run is improving.</p>
    <table>
        <thead><tr><th>Metric</th><th>What it means</th></tr></thead>
        <tbody>
            {metric_descriptions_html()}
        </tbody>
    </table>
</section>

<script>
(function () {{
    const plot = document.getElementById('{plot_div_id}');
    const checkboxes = Array.from(document.querySelectorAll('.metric-toggle'));
    const labels = Array.from(document.querySelectorAll('.metric-toggle-label'));
    const metricFilter = document.getElementById('metric-filter');
    const sectionFilter = document.getElementById('section-filter');
    const sections = Array.from(document.querySelectorAll('.metric-section'));

    function setTraceVisible(index, visible) {{
        if (!plot || !window.Plotly) return;
        window.Plotly.restyle(plot, {{ visible: visible ? true : 'legendonly' }}, [index]);
    }}

    function applyFilters() {{
        const query = metricFilter.value.trim().toLowerCase();
        const selectedSection = sectionFilter.value;

        sections.forEach((section) => {{
            const sectionName = section.dataset.category;
            const showSection = selectedSection === 'all' || selectedSection === sectionName;
            section.hidden = !showSection;
        }});

        labels.forEach((label) => {{
            const matchesQuery = label.dataset.name.includes(query);
            label.hidden = !matchesQuery;
        }});
    }}

    checkboxes.forEach((checkbox) => {{
        checkbox.addEventListener('change', () => {{
            setTraceVisible(Number(checkbox.dataset.traceIndex), checkbox.checked);
        }});
    }});

    document.getElementById('show-all-metrics').addEventListener('click', () => {{
        checkboxes.forEach((checkbox) => {{
            checkbox.checked = true;
            setTraceVisible(Number(checkbox.dataset.traceIndex), true);
        }});
    }});

    document.getElementById('hide-all-metrics').addEventListener('click', () => {{
        checkboxes.forEach((checkbox) => {{
            checkbox.checked = false;
            setTraceVisible(Number(checkbox.dataset.traceIndex), false);
        }});
    }});

    document.querySelectorAll('.section-show').forEach((button) => {{
        button.addEventListener('click', (event) => {{
            event.preventDefault();
            const category = button.dataset.category;
            checkboxes.filter(cb => cb.dataset.category === category).forEach((checkbox) => {{
                checkbox.checked = true;
                setTraceVisible(Number(checkbox.dataset.traceIndex), true);
            }});
        }});
    }});

    document.querySelectorAll('.section-hide').forEach((button) => {{
        button.addEventListener('click', (event) => {{
            event.preventDefault();
            const category = button.dataset.category;
            checkboxes.filter(cb => cb.dataset.category === category).forEach((checkbox) => {{
                checkbox.checked = false;
                setTraceVisible(Number(checkbox.dataset.traceIndex), false);
            }});
        }});
    }});

    metricFilter.addEventListener('input', applyFilters);
    sectionFilter.addEventListener('change', applyFilters);
}})();
</script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


# ── Loading helpers ────────────────────────────────────────────────────────────

def load_trainer_history(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    return state.get("log_history", [])


def load_csv(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def numeric(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_file(pattern: str, root: Path = Path(".")) -> Path | None:
    files = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.glob("*") if path.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def discover_trainer_state(run_dir: Path | None) -> Path | None:
    if run_dir:
        found = latest_file("output/**/trainer_state.json", run_dir)
        if found:
            return found
        found = latest_file("**/trainer_state.json", run_dir)
        if found:
            return found

    return (
        latest_file("grpo_runs/*/output/**/trainer_state.json")
        or latest_file("grpo_mvp_output/**/trainer_state.json")
    )


def discover_samples(run_dir: Path | None) -> Path | None:
    if run_dir:
        found = latest_file("logs/samples.csv", run_dir)
        if found:
            return found
        found = latest_file("**/samples.csv", run_dir)
        if found:
            return found

    return (
        latest_file("grpo_runs/*/logs/samples.csv")
        or latest_file("grpo_mvp_logs/*/samples.csv")
    )


def discover_eval_files(run_dir: Path | None) -> list[Path]:
    files: list[Path] = []

    if run_dir and run_dir.exists():
        files.extend(sorted(run_dir.glob("logs/eval_*.csv")))
        files.extend(sorted(run_dir.glob("**/eval_*.csv")))

    files.extend(sorted(Path(".").glob("grpo_runs/*/logs/eval_*.csv")))
    files.extend(sorted(Path(".").glob("grpo_mvp_logs/*/eval_*.csv")))

    seen = set()
    unique = []
    for file in files:
        resolved = file.resolve()
        if resolved not in seen and file.exists():
            unique.append(file)
            seen.add(resolved)

    return unique


# ── Pandas windowing ───────────────────────────────────────────────────────────

def window_xy(
    xs: Iterable[float],
    ys: Iterable[float],
    rolling_window: int = 1,
    recent_steps: int | None = None,
) -> tuple[list[float], list[float]]:
    xs = list(xs)
    ys = list(ys)

    if not xs or not ys:
        return xs, ys

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is not installed. Install it with: pip install pandas") from exc

    frame = pd.DataFrame({"step": xs, "value": ys}).dropna().sort_values("step")

    if recent_steps is not None and recent_steps > 0 and not frame.empty:
        latest_step = frame["step"].max()
        frame = frame[frame["step"] >= latest_step - recent_steps]

    if rolling_window > 1 and not frame.empty:
        frame["value"] = frame["value"].rolling(window=rolling_window, min_periods=1).mean()

    return frame["step"].tolist(), frame["value"].tolist()


# ── Trainer history ────────────────────────────────────────────────────────────

def history_series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []

    for row in history:
        x = numeric(row.get("step"))
        y = numeric(row.get(key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)

    return xs, ys


def available_reward_keys(history: list[dict]) -> list[str]:
    keys = set()

    for row in history:
        for key in row:
            if key.startswith("rewards/") and key.endswith("/mean"):
                keys.add(key)

    return sorted(keys)


# ── Sample log aggregations ────────────────────────────────────────────────────

def group_by_step(samples: list[dict]) -> dict[int, list[dict]]:
    by_step: dict[int, list[dict]] = defaultdict(list)

    for row in samples:
        step = numeric(row.get("step"))
        if step is not None:
            by_step[int(step)].append(row)

    return by_step


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def sample_step_stats(samples: list[dict]) -> dict[str, dict[int, float]]:
    by_step = group_by_step(samples)
    stats: dict[str, dict[int, float]] = defaultdict(dict)

    for step, rows in by_step.items():
        count = len(rows)
        if count == 0:
            continue

        stats["format_ok_rate"][step] = sum(row.get("format_status") == "ok" for row in rows) / count
        stats["extra_text_rate"][step] = sum(row.get("format_status") == "extra_text_after_code" for row in rows) / count
        stats["syntax_ok_rate"][step] = sum(row.get("syntax_status") == "ok" for row in rows) / count
        stats["style_clean_rate"][step] = sum(
            row.get("style_status") == "ok" or numeric(row.get("style_violation_count")) == 0
            for row in rows
        ) / count
        stats["style_violation_rate"][step] = sum(row.get("style_status") == "violations" for row in rows) / count
        stats["compile_ok_rate"][step] = sum(row.get("compile_status") == "ok" for row in rows) / count
        stats["runtime_ok_rate"][step] = sum(row.get("runtime_status") == "ok" for row in rows) / count
        stats["test_pass_rate"][step] = sum(row.get("test_status") == "passed" for row in rows) / count
        stats["score_6_rate"][step] = sum(numeric(row.get("reward_total")) == 6.0 for row in rows) / count
        stats["no_code_rate"][step] = sum(row.get("failure_reason") == "no_code_extracted" for row in rows) / count
        stats["syntax_error_rate"][step] = sum(row.get("failure_reason") == "syntax_error" for row in rows) / count
        stats["timeout_rate"][step] = sum(
            row.get("failure_reason") == "timeout" or row.get("test_status") == "timeout"
            for row in rows
        ) / count

        for column in REWARD_COLUMNS:
            value = mean([numeric(row.get(column)) for row in rows])
            if value is not None:
                stats[f"mean_{column}"][step] = value

        for column in TOKEN_COLUMNS:
            value = mean([numeric(row.get(column)) for row in rows])
            if value is not None:
                stats[f"mean_{column}"][step] = value

        for column in ["style_score", "style_penalty", "style_violation_count"]:
            value = mean([numeric(row.get(column)) for row in rows])
            if value is not None:
                stats[f"mean_{column}"][step] = value

        totals = [numeric(row.get("reward_total")) for row in rows]
        totals = [value for value in totals if value is not None]

        if len(totals) >= 2:
            avg = sum(totals) / len(totals)
            variance = sum((value - avg) ** 2 for value in totals) / len(totals)
            stats["reward_total_std_by_step"][step] = variance ** 0.5

    return stats


def value_counts(rows: list[dict], column: str) -> Counter:
    counts = Counter()

    for row in rows:
        value = row.get(column) or "none"
        counts[value] += 1

    return counts


def split_value_counts(rows: list[dict], column: str, separator: str = ",") -> Counter:
    counts = Counter()

    for row in rows:
        value = row.get(column) or ""
        parts = [part.strip() for part in value.split(separator) if part.strip()]
        if not parts:
            counts["none"] += 1
            continue
        for part in parts:
            counts[part] += 1

    return counts


def final_summary(samples: list[dict]) -> dict[str, float | int | str]:
    if not samples:
        return {}

    total = len(samples)
    reward_values = [numeric(row.get("reward_total")) for row in samples]
    reward_values = [value for value in reward_values if value is not None]

    return {
        "samples": total,
        "avg_reward_total": round(sum(reward_values) / len(reward_values), 4) if reward_values else 0,
        "format_ok_rate": round(sum(row.get("format_status") == "ok" for row in samples) / total, 4),
        "syntax_ok_rate": round(sum(row.get("syntax_status") == "ok" for row in samples) / total, 4),
        "style_clean_rate": round(
            sum(row.get("style_status") == "ok" or numeric(row.get("style_violation_count")) == 0 for row in samples)
            / total,
            4,
        ),
        "avg_style_score": round(
            mean([numeric(row.get("style_score")) for row in samples]) or 0,
            4,
        ),
        "avg_style_violations": round(
            mean([numeric(row.get("style_violation_count")) for row in samples]) or 0,
            4,
        ),
        "extra_text_rate": round(sum(row.get("format_status") == "extra_text_after_code" for row in samples) / total, 4),
        "compile_ok_rate": round(sum(row.get("compile_status") == "ok" for row in samples) / total, 4),
        "runtime_ok_rate": round(sum(row.get("runtime_status") == "ok" for row in samples) / total, 4),
        "test_pass_rate": round(sum(row.get("test_status") == "passed" for row in samples) / total, 4),
        "score_6_rate": round(sum(numeric(row.get("reward_total")) == 6.0 for row in samples) / total, 4),
    }


# ── Eval aggregations ──────────────────────────────────────────────────────────

def load_eval_rows(eval_files: list[Path]) -> list[dict]:
    rows: list[dict] = []

    for file in eval_files:
        for row in load_csv(file):
            row["eval_file"] = str(file)
            row["eval_name"] = row.get("label") or file.stem
            rows.append(row)

    return rows


def eval_summary_rows(eval_rows: list[dict]) -> list[dict]:
    by_name: dict[str, list[dict]] = defaultdict(list)

    for row in eval_rows:
        by_name[row.get("eval_name", "eval")].append(row)

    summaries = []

    for name, rows in sorted(by_name.items()):
        total = len(rows)
        if total == 0:
            continue

        rewards = [numeric(row.get("reward_total")) for row in rows]
        rewards = [value for value in rewards if value is not None]

        by_task = defaultdict(list)
        for row in rows:
            by_task[row.get("task_id", row.get("item_idx", "unknown"))].append(row)

        pass_at_any = sum(
            any(row.get("test_status") == "passed" for row in task_rows)
            for task_rows in by_task.values()
        )

        summaries.append({
            "name": name,
            "rows": total,
            "tasks": len(by_task),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "format_ok_rate": sum(row.get("format_status") == "ok" for row in rows) / total,
            "syntax_ok_rate": sum(row.get("syntax_status") == "ok" for row in rows) / total,
            "style_clean_rate": sum(
                row.get("style_status") == "ok" or numeric(row.get("style_violation_count")) == 0
                for row in rows
            ) / total,
            "avg_style_score": mean([numeric(row.get("style_score")) for row in rows]) or 0,
            "compile_ok_rate": sum(row.get("compile_status") == "ok" for row in rows) / total,
            "runtime_ok_rate": sum(row.get("runtime_status") == "ok" for row in rows) / total,
            "test_pass_rate": sum(row.get("test_status") == "passed" for row in rows) / total,
            "pass_at_any_rate": pass_at_any / len(by_task) if by_task else 0,
        })

    return summaries


# ── Plot helpers ───────────────────────────────────────────────────────────────

def add_xy_line(fig, row: int, col: int, xs, ys, name: str, rolling_window: int, recent_steps: int | None) -> None:
    if not xs or not ys:
        return

    import plotly.graph_objects as go

    xs, ys = window_xy(xs, ys, rolling_window, recent_steps)

    if not xs or not ys:
        return

    suffix = f" ({rolling_window}-pt avg)" if rolling_window > 1 else ""
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=f"{name}{suffix}",
        ),
        row=row,
        col=col,
    )


def add_history_line(
    fig,
    row: int,
    col: int,
    history: list[dict],
    key: str,
    name: str,
    rolling_window: int,
    recent_steps: int | None,
) -> None:
    xs, ys = history_series(history, key)
    add_xy_line(fig, row, col, xs, ys, name, rolling_window, recent_steps)


def add_stat_line(
    fig,
    row: int,
    col: int,
    stats: dict[str, dict[int, float]],
    key: str,
    name: str,
    rolling_window: int,
    recent_steps: int | None,
) -> None:
    values = stats.get(key, {})

    if not values:
        return

    xs = sorted(values)
    ys = [values[x] for x in xs]
    add_xy_line(fig, row, col, xs, ys, name, rolling_window, recent_steps)


def add_bar(fig, row: int, col: int, counts: Counter, name: str, top_n: int = 12) -> None:
    if not counts:
        return

    import plotly.graph_objects as go

    items = counts.most_common(top_n)
    fig.add_trace(
        go.Bar(
            x=[item[0] for item in items],
            y=[item[1] for item in items],
            name=name,
        ),
        row=row,
        col=col,
    )


# ── Dashboard ─────────────────────────────────────────────────────────────────

def build_dashboard(
    history: list[dict],
    samples: list[dict],
    eval_rows: list[dict],
    output: Path,
    rolling_window: int = 10,
    recent_steps: int | None = None,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit("Plotly is not installed. Install it with: pip install plotly") from exc

    stats = sample_step_stats(samples)
    eval_summaries = eval_summary_rows(eval_rows)

    fig = make_subplots(
        rows=5,
        cols=2,
        specs=[
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{"type": "table"}, {"type": "table"}],
        ],
        subplot_titles=[
            "Reward components",
            "Core success rates",
            "Failure rates",
            "Trainer GRPO signals",
            "Completion/token lengths",
            "Reward spread / clipping",
            "Failure reasons",
            "Test/compile status counts",
            "Training summary",
            "Evaluation summaries",
        ],
    )

    add_stat_line(fig, 1, 1, stats, "mean_reward_total", "mean reward_total", rolling_window, recent_steps)
    add_stat_line(fig, 1, 1, stats, "mean_reward_format", "format reward", rolling_window, recent_steps)
    add_stat_line(fig, 1, 1, stats, "mean_reward_syntax", "syntax reward", rolling_window, recent_steps)
    add_stat_line(fig, 1, 1, stats, "mean_reward_style", "style reward", rolling_window, recent_steps)
    add_stat_line(fig, 1, 1, stats, "mean_reward_compile", "compile reward", rolling_window, recent_steps)
    add_stat_line(fig, 1, 1, stats, "mean_reward_tests", "tests reward", rolling_window, recent_steps)
    add_stat_line(fig, 1, 1, stats, "mean_reward_extra_text_penalty", "extra text penalty", rolling_window, recent_steps)

    add_stat_line(fig, 1, 2, stats, "format_ok_rate", "format ok", rolling_window, recent_steps)
    add_stat_line(fig, 1, 2, stats, "syntax_ok_rate", "syntax ok", rolling_window, recent_steps)
    add_stat_line(fig, 1, 2, stats, "style_clean_rate", "style clean", rolling_window, recent_steps)
    add_stat_line(fig, 1, 2, stats, "compile_ok_rate", "compile ok", rolling_window, recent_steps)
    add_stat_line(fig, 1, 2, stats, "runtime_ok_rate", "runtime ok", rolling_window, recent_steps)
    add_stat_line(fig, 1, 2, stats, "test_pass_rate", "tests passed", rolling_window, recent_steps)
    add_stat_line(fig, 1, 2, stats, "score_6_rate", "perfect 6.0", rolling_window, recent_steps)

    add_stat_line(fig, 2, 1, stats, "no_code_rate", "no code extracted", rolling_window, recent_steps)
    add_stat_line(fig, 2, 1, stats, "syntax_error_rate", "syntax error", rolling_window, recent_steps)
    add_stat_line(fig, 2, 1, stats, "style_violation_rate", "style violations", rolling_window, recent_steps)
    add_stat_line(fig, 2, 1, stats, "timeout_rate", "timeout", rolling_window, recent_steps)
    add_stat_line(fig, 2, 1, stats, "extra_text_rate", "extra text after code", rolling_window, recent_steps)

    add_history_line(fig, 2, 2, history, "reward", "trainer reward", rolling_window, recent_steps)
    add_history_line(fig, 2, 2, history, "loss", "loss", rolling_window, recent_steps)
    add_history_line(fig, 2, 2, history, "kl", "kl", rolling_window, recent_steps)
    add_history_line(fig, 2, 2, history, "grad_norm", "grad_norm", rolling_window, recent_steps)
    add_history_line(fig, 2, 2, history, "learning_rate", "learning_rate", rolling_window, recent_steps)

    for key in available_reward_keys(history):
        add_history_line(
            fig,
            2,
            2,
            history,
            key,
            key.removeprefix("rewards/").removesuffix("/mean"),
            rolling_window,
            recent_steps,
        )

    add_history_line(fig, 3, 1, history, "completion_length", "trainer completion_length", rolling_window, recent_steps)
    add_history_line(fig, 3, 1, history, "completions/mean_length", "trainer mean_length", rolling_window, recent_steps)
    add_history_line(fig, 3, 1, history, "completions/max_length", "trainer max_length", rolling_window, recent_steps)
    add_stat_line(fig, 3, 1, stats, "mean_completion_token_length", "logged completion tokens", rolling_window, recent_steps)
    add_stat_line(fig, 3, 1, stats, "mean_prompt_token_length", "logged prompt tokens", rolling_window, recent_steps)
    add_stat_line(fig, 3, 1, stats, "mean_style_score", "style_score", rolling_window, recent_steps)
    add_stat_line(fig, 3, 1, stats, "mean_style_violation_count", "style_violation_count", rolling_window, recent_steps)

    add_stat_line(fig, 3, 2, stats, "reward_total_std_by_step", "logged reward_total std", rolling_window, recent_steps)
    add_history_line(fig, 3, 2, history, "reward_std", "trainer reward_std", rolling_window, recent_steps)
    add_history_line(fig, 3, 2, history, "frac_reward_zero_std", "frac_reward_zero_std", rolling_window, recent_steps)
    add_history_line(fig, 3, 2, history, "completions/clipped_ratio", "clipped_ratio", rolling_window, recent_steps)

    add_bar(fig, 4, 1, value_counts(samples, "failure_reason"), "failure reasons")
    add_bar(fig, 4, 1, split_value_counts(samples, "style_codes"), "style codes")
    add_bar(fig, 4, 2, value_counts(samples, "test_status"), "test status")
    add_bar(fig, 4, 2, value_counts(samples, "compile_status"), "compile status")
    add_bar(fig, 4, 2, value_counts(samples, "style_status"), "style status")

    summary = final_summary(samples)

    if summary:
        fig.add_trace(
            go.Table(
                header=dict(values=["metric", "value"]),
                cells=dict(values=[list(summary.keys()), list(summary.values())]),
                name="Training summary",
            ),
            row=5,
            col=1,
        )

    if eval_summaries:
        fig.add_trace(
            go.Table(
                header=dict(values=[
                    "eval",
                    "rows",
                    "tasks",
                    "avg_reward",
                    "format_ok",
                    "syntax_ok",
                    "style_clean",
                    "avg_style",
                    "compile_ok",
                    "runtime_ok",
                    "test_pass",
                    "pass@any",
                ]),
                cells=dict(values=[
                    [row["name"] for row in eval_summaries],
                    [row["rows"] for row in eval_summaries],
                    [row["tasks"] for row in eval_summaries],
                    [round(row["avg_reward"], 4) for row in eval_summaries],
                    [round(row["format_ok_rate"], 4) for row in eval_summaries],
                    [round(row["syntax_ok_rate"], 4) for row in eval_summaries],
                    [round(row["style_clean_rate"], 4) for row in eval_summaries],
                    [round(row["avg_style_score"], 4) for row in eval_summaries],
                    [round(row["compile_ok_rate"], 4) for row in eval_summaries],
                    [round(row["runtime_ok_rate"], 4) for row in eval_summaries],
                    [round(row["test_pass_rate"], 4) for row in eval_summaries],
                    [round(row["pass_at_any_rate"], 4) for row in eval_summaries],
                ]),
                name="Evaluation summaries",
            ),
            row=5,
            col=2,
        )

    fig.update_layout(
        title=(
            "Conrad GRPO Coding Training Dashboard "
            f"| rolling_window={rolling_window}, recent_steps={recent_steps or 'all'}"
        ),
        height=1750,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.10,
            xanchor="left",
            x=0,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )

    fig.update_xaxes(title_text="step")
    wrap_dashboard_html(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Plotly dashboard for train_conrad.py GRPO coding logs."
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory, e.g. grpo_runs/E1_codelama_100_steps. If omitted, latest grpo_runs/* is used.",
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("grpo_runs"),
        help="Root folder containing run folders.",
    )

    parser.add_argument(
        "--trainer-state",
        type=Path,
        default=None,
        help="Path to trainer_state.json. Defaults to run-dir/output/**/trainer_state.json.",
    )

    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Path to samples.csv. Defaults to run-dir/logs/samples.csv.",
    )

    parser.add_argument(
        "--eval-file",
        action="append",
        type=Path,
        default=[],
        help="Optional eval CSV file. Can be passed multiple times.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conrad_training_dashboard.html"),
        help="Output HTML file.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Pandas trailing rolling mean window size. Use 1 for raw values.",
    )

    parser.add_argument(
        "--recent-steps",
        type=int,
        default=None,
        help="Only plot the most recent N training steps. Omit to show all steps.",
    )

    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir(args.runs_root)
    trainer_state = args.trainer_state or discover_trainer_state(run_dir)
    samples = args.samples or discover_samples(run_dir)
    eval_files = args.eval_file or discover_eval_files(run_dir)

    history = load_trainer_history(trainer_state)
    sample_rows = load_csv(samples)
    eval_rows = load_eval_rows(eval_files)

    print(f"Run dir: {run_dir if run_dir else 'not found'}")
    print(f"Trainer state: {trainer_state if trainer_state else 'not found'}")
    print(f"Sample diagnostics: {samples if samples else 'not found'}")
    print(f"Eval files: {', '.join(str(file) for file in eval_files) if eval_files else 'none found'}")
    print(f"Trainer log rows: {len(history)}")
    print(f"Sample rows: {len(sample_rows)}")
    print(f"Eval rows: {len(eval_rows)}")

    build_dashboard(
        history=history,
        samples=sample_rows,
        eval_rows=eval_rows,
        output=args.output,
        rolling_window=max(1, args.rolling_window),
        recent_steps=args.recent_steps,
    )

    print(f"Wrote {args.output}")
    print("Key panels to watch:")
    print("- test_pass_rate and reward_tests should trend upward")
    print("- no_code_extracted, syntax_error, timeout, and extra_text_after_code should trend downward")
    print("- reward_total_std should not collapse to zero too early")
    print("- holdout eval test_pass_rate matters more than train-only reward")


if __name__ == "__main__":
    main()