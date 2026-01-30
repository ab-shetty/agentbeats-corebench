#!/usr/bin/env python3
"""
LLM Judge Consistency Test

Validates that the LLM-as-judge produces consistent Task Adherence scores
across repeated evaluations of the same agent traces.

Usage:
    # Test with default settings (10 runs, temperature 1.0)
    python run_batch_consistency.py path/to/traces/

    # Test at multiple temperatures (if model supports it)
    python run_batch_consistency.py path/to/traces/ --model gpt-4o --temps 0.0 0.3 0.7

    # Quick validation (fewer runs)
    python run_batch_consistency.py path/to/traces/ --runs 5 --max-traces 4

Output:
    - CSV files with per-temperature results
    - Summary CSV comparing temperatures
    - Markdown report with recommendations
"""

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

# Add project root for metrics import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scenarios" / "corebench"))

from metrics.metrics import _calculate_score_from_components


# --- Trace loading and judge calling ---

def load_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    """Load all events from a trace file (JSONL or pretty-printed JSON)."""
    text = trace_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    events: list[dict[str, Any]] = []
    length = len(text)

    while True:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(text, idx)
        idx = end
        if isinstance(obj, list):
            events.extend(obj)
        else:
            events.append(obj)

    return events


def extract_judge_context(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the judge prompt and metadata from trace events."""
    judge_input = next((e for e in events if e.get("type") == "llm_judge_input"), None)
    if not judge_input:
        raise ValueError("No llm_judge_input event found in trace")

    prompt = judge_input.get("prompt")
    if not prompt:
        raise ValueError("llm_judge_input event has no 'prompt' field")

    task_start = next((e for e in events if e.get("type") == "task_start"), None)

    return {
        "prompt": prompt,
        "model": judge_input.get("model", "gpt-4o"),
        "domain": judge_input.get("domain", "unknown"),
        "task_prompt": judge_input.get("task_prompt", ""),
        "questions": judge_input.get("questions", []),
        "tool_calls_count": judge_input.get("tool_calls_count", 0),
        "task_id": task_start.get("task_id") if task_start else "unknown",
    }


async def call_judge_once(
    prompt: str,
    model: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call the LLM judge once and parse the response."""
    api_base = (os.environ.get("COREBENCH_TEXT_API_BASE") or "").strip()
    api_key = (os.environ.get("COREBENCH_TEXT_API_KEY") or "").strip()

    if api_base:
        model_name = model if model.startswith("openai/") else f"openai/{model}"
        completion_kwargs = {
            "model": model_name,
            "api_base": api_base,
            "api_key": api_key or "dummy",
        }
    else:
        completion_kwargs = {"model": model}

    response = await litellm.acompletion(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
        **completion_kwargs,
    )

    raw_content = response.choices[0].message.content

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            raise ValueError(f"Could not parse judge response as JSON: {raw_content[:200]}")

    component_scores = result.get("component_scores", {})
    llm_score = float(result.get("score", 0.0))
    calculated_score = _calculate_score_from_components(component_scores, llm_score)

    return {
        "score": calculated_score,
        "llm_raw_score": llm_score,
        "component_scores": component_scores,
        "reasoning": result.get("reasoning", ""),
        "strengths": result.get("strengths", []),
        "weaknesses": result.get("weaknesses", []),
        "status": "success",
    }


# --- Batch testing ---

def grade_stdev(stdev: float) -> str:
    """Grade consistency based on standard deviation."""
    if stdev < 0.05:
        return "EXCELLENT"
    elif stdev < 0.10:
        return "GOOD"
    elif stdev < 0.15:
        return "FAIR"
    else:
        return "POOR"


def find_traces_with_judge_input(trace_dir: Path, max_traces: int = 10) -> list[Path]:
    """Find trace files that have llm_judge_input events."""
    traces_found = []

    for trace_file in sorted(trace_dir.glob("*.jsonl")):
        try:
            events = load_trace_events(trace_file)
            has_judge_input = any(e.get("type") == "llm_judge_input" for e in events)
            if has_judge_input:
                traces_found.append(trace_file)
                if len(traces_found) >= max_traces:
                    break
        except Exception as e:
            print(f"  Skipping {trace_file.name}: {e}")
            continue

    return traces_found


async def run_judge_for_trace(
    trace_path: Path,
    n_runs: int,
    temperature: float,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Run judge N times for a single trace at given temperature."""
    events = load_trace_events(trace_path)
    context = extract_judge_context(events)
    model = model_override if model_override else context["model"]

    capsule_id = trace_path.stem.split("capsule-")[-1] if "capsule-" in trace_path.stem else trace_path.stem

    scores = []
    full_results = []
    errors = []

    for run_num in range(n_runs):
        try:
            result = await call_judge_once(
                prompt=context["prompt"],
                model=model,
                temperature=temperature,
            )
            scores.append(result["score"])
            full_results.append({
                "run": run_num + 1,
                "score": result["score"],
                "component_scores": result.get("component_scores", {}),
                "status": "success",
            })
        except Exception as e:
            errors.append(str(e))
            scores.append(None)
            full_results.append({
                "run": run_num + 1,
                "score": None,
                "error": str(e),
                "status": "error",
            })

    valid_scores = [s for s in scores if s is not None]

    if len(valid_scores) >= 2:
        mean = statistics.mean(valid_scores)
        stdev = statistics.stdev(valid_scores)
    elif len(valid_scores) == 1:
        mean = valid_scores[0]
        stdev = 0.0
    else:
        mean = 0.0
        stdev = 0.0

    return {
        "capsule_id": capsule_id,
        "trace_file": trace_path.name,
        "temperature": temperature,
        "n_runs": n_runs,
        "n_valid": len(valid_scores),
        "n_errors": len(errors),
        "scores": scores,
        "valid_scores": valid_scores,
        "mean": mean,
        "stdev": stdev,
        "min": min(valid_scores) if valid_scores else 0.0,
        "max": max(valid_scores) if valid_scores else 0.0,
        "range": (max(valid_scores) - min(valid_scores)) if valid_scores else 0.0,
        "grade": grade_stdev(stdev),
        "errors": errors,
        "full_results": full_results,
    }


# --- Output formatting ---

def print_results_table(results: list[dict[str, Any]], temperature: float, n_runs: int):
    """Print formatted results table."""
    print(f"\n{'=' * 80}")
    print(f"Temperature = {temperature}")
    print(f"{'=' * 80}")

    run_headers = [f"R{i+1}" for i in range(n_runs)]
    header = f"{'CAPSULE':12} | " + " | ".join(f"{h:5}" for h in run_headers) + f" | {'MEAN':5} | {'STDEV':5} | {'GRADE':9}"
    print(header)
    print("-" * len(header))

    for r in results:
        capsule = r["capsule_id"][:12]
        scores_str = ["  ERR" if s is None else f"{s:5.2f}" for s in r["scores"]]
        row = f"{capsule:12} | " + " | ".join(scores_str) + f" | {r['mean']:5.2f} | {r['stdev']:5.3f} | {r['grade']:9}"
        print(row)

    if results:
        avg_mean = statistics.mean([r["mean"] for r in results])
        avg_stdev = statistics.mean([r["stdev"] for r in results])
        grades = [r["grade"] for r in results]
        print("-" * 80)
        print(f"{'AVERAGE':12} |" + " " * (len(run_headers) * 8) + f" | {avg_mean:5.2f} | {avg_stdev:5.3f} |")
        print(f"\nGrades: EXCELLENT={grades.count('EXCELLENT')}, GOOD={grades.count('GOOD')}, "
              f"FAIR={grades.count('FAIR')}, POOR={grades.count('POOR')}")


def print_comparison(all_results: dict[float, list[dict]]):
    """Print comparison across temperatures."""
    temps = sorted(all_results.keys())
    if len(temps) <= 1:
        return

    print(f"\n{'=' * 80}")
    print("TEMPERATURE COMPARISON")
    print(f"{'=' * 80}")

    avg_stdevs = {t: statistics.mean([r["stdev"] for r in all_results[t]]) for t in temps}
    best_temp = min(avg_stdevs, key=avg_stdevs.get)

    print("\nAverage stdev by temperature:")
    for t in temps:
        marker = " <-- BEST" if t == best_temp else ""
        print(f"  temp={t}: {avg_stdevs[t]:.4f}{marker}")

    print(f"\nRECOMMENDATION: Use temperature={best_temp}")


def save_csv(results: list[dict], output_path: Path, n_runs: int):
    """Save results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["capsule_id", "temperature"] + [f"run_{i+1}" for i in range(n_runs)] + ["mean", "stdev", "grade"]
        writer.writerow(header)

        for r in results:
            row = [r["capsule_id"], r["temperature"]]
            row += [f"{s:.4f}" if s is not None else "ERROR" for s in r["scores"]]
            row += [f"{r['mean']:.4f}", f"{r['stdev']:.4f}", r["grade"]]
            writer.writerow(row)


def save_summary_csv(all_results: dict[float, list[dict]], output_path: Path):
    """Save combined summary CSV."""
    temps = sorted(all_results.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["capsule_id"]
        for t in temps:
            t_str = str(t).replace(".", "_")
            header += [f"mean_t{t_str}", f"stdev_t{t_str}", f"grade_t{t_str}"]
        writer.writerow(header)

        capsule_ids = [r["capsule_id"] for r in all_results[temps[0]]]
        results_by_temp = {t: {r["capsule_id"]: r for r in all_results[t]} for t in temps}

        for capsule_id in capsule_ids:
            row = [capsule_id]
            for t in temps:
                r = results_by_temp[t].get(capsule_id)
                if r:
                    row += [f"{r['mean']:.4f}", f"{r['stdev']:.4f}", r["grade"]]
                else:
                    row += ["N/A", "N/A", "N/A"]
            writer.writerow(row)


def generate_report(
    all_results: dict[float, list[dict]],
    output_path: Path,
    trace_dir: str,
    n_runs: int,
    model: str | None,
):
    """Generate markdown report."""
    temps = sorted(all_results.keys())
    n_traces = len(all_results[temps[0]])
    total_calls = n_traces * n_runs * len(temps)

    stats_by_temp = {}
    for t in temps:
        grades = [r["grade"] for r in all_results[t]]
        stats_by_temp[t] = {
            "avg_stdev": statistics.mean([r["stdev"] for r in all_results[t]]),
            "avg_mean": statistics.mean([r["mean"] for r in all_results[t]]),
            "excellent": grades.count("EXCELLENT"),
            "good": grades.count("GOOD"),
            "fair": grades.count("FAIR"),
            "poor": grades.count("POOR"),
        }

    best_temp = min(temps, key=lambda t: stats_by_temp[t]["avg_stdev"])

    report = f"""# LLM Judge Consistency Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Configuration

| Setting | Value |
|---------|-------|
| Model | {model or "from trace"} |
| Traces tested | {n_traces} |
| Runs per trace | {n_runs} |
| Temperatures | {", ".join(str(t) for t in temps)} |
| Total LLM calls | {total_calls} |

## Grading Criteria

| Grade | Stdev Threshold | Interpretation |
|-------|-----------------|----------------|
| EXCELLENT | < 0.05 | Highly consistent (~5% variance) |
| GOOD | < 0.10 | Acceptable (~10% variance) |
| FAIR | < 0.15 | Moderate variance (~15%) |
| POOR | >= 0.15 | Unreliable |

## Results Summary

| Metric | """ + " | ".join(f"T={t}" for t in temps) + """ |
|--------|""" + "|".join("------|" for _ in temps) + """
| Avg Stdev | """ + " | ".join(f"{stats_by_temp[t]['avg_stdev']:.4f}" for t in temps) + """ |
| EXCELLENT | """ + " | ".join(f"{stats_by_temp[t]['excellent']}/{n_traces}" for t in temps) + """ |
| GOOD | """ + " | ".join(f"{stats_by_temp[t]['good']}/{n_traces}" for t in temps) + """ |
| FAIR | """ + " | ".join(f"{stats_by_temp[t]['fair']}/{n_traces}" for t in temps) + """ |
| POOR | """ + " | ".join(f"{stats_by_temp[t]['poor']}/{n_traces}" for t in temps) + """ |

## Recommendation

**Use temperature={best_temp}** (avg stdev: {stats_by_temp[best_temp]['avg_stdev']:.4f})

## Per-Capsule Results
"""

    for t in temps:
        report += f"\n### Temperature = {t}\n\n"
        report += "| Capsule | Mean | Stdev | Grade |\n"
        report += "|---------|------|-------|-------|\n"
        for r in all_results[t]:
            report += f"| {r['capsule_id']} | {r['mean']:.3f} | {r['stdev']:.3f} | {r['grade']} |\n"

    output_path.write_text(report)


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(
        description="Validate LLM judge consistency across repeated evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace_dir", help="Directory containing trace files (.jsonl)")
    parser.add_argument("--runs", "-n", type=int, default=10, help="Runs per trace (default: 10)")
    parser.add_argument("--max-traces", "-m", type=int, default=10, help="Max traces to test (default: 10)")
    parser.add_argument("--temps", "-t", type=float, nargs="+", default=[1.0], help="Temperatures to test (default: 1.0)")
    parser.add_argument("--model", type=str, default=None, help="Override model (e.g., gpt-4o, claude-3-sonnet)")
    parser.add_argument("--output-dir", "-o", default="consistency-results", help="Output directory")

    args = parser.parse_args()
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    temps = sorted(args.temps)

    if not trace_dir.exists():
        sys.exit(f"Trace directory not found: {trace_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {trace_dir} for traces...")
    traces = find_traces_with_judge_input(trace_dir, args.max_traces)

    if not traces:
        sys.exit("No traces found with llm_judge_input events")

    print(f"Found {len(traces)} traces")
    if args.model:
        print(f"Model: {args.model}")

    total_calls = len(traces) * args.runs * len(temps)
    print(f"Total LLM calls: {total_calls}")
    print()

    all_results: dict[float, list[dict]] = {t: [] for t in temps}

    for i, trace_path in enumerate(traces):
        capsule_id = trace_path.stem.split("capsule-")[-1] if "capsule-" in trace_path.stem else trace_path.stem
        print(f"[{i+1}/{len(traces)}] {capsule_id}")

        for temp in temps:
            print(f"  T={temp}: ", end="", flush=True)
            result = await run_judge_for_trace(trace_path, args.runs, temp, args.model)
            print(f"mean={result['mean']:.3f}, stdev={result['stdev']:.3f} [{result['grade']}]")
            all_results[temp].append(result)

    # Print results
    for temp in temps:
        print_results_table(all_results[temp], temp, args.runs)
    print_comparison(all_results)

    # Save outputs
    for temp in temps:
        t_str = str(temp).replace(".", "_")
        save_csv(all_results[temp], output_dir / f"results_temp_{t_str}.csv", args.runs)

    save_summary_csv(all_results, output_dir / "summary.csv")
    generate_report(all_results, output_dir / "CONSISTENCY_REPORT.md", str(trace_dir), args.runs, args.model)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
