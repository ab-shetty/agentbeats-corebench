#!/usr/bin/env python3
"""
Compare two CoreBench benchmark runs.

Usage:
    python compare_runs.py <old_log> <new_log>
    python compare_runs.py corebench_output_easy.txt corebench_output_llama_easy.txt
"""

import json
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskResult:
    success: bool
    accuracy: float
    faithfulness: float
    adherence: float


def extract_task_results(filename: str) -> tuple[dict[str, TaskResult], dict]:
    """
    Extract task results from a CoreBench log file.
    Returns (task_results dict, summary_metrics dict)
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find the JSON data block in DataPart - look for the task_results JSON structure
    # Pattern to match the data= dictionary in DataPart
    pattern = r"data=\{['\"]domain['\"].*?['\"]task_results['\"]:\s*(\{.*?\})\}"
    
    # Alternative: Look for the cleaner JSON block that appears earlier in the log
    # The log contains a formatted JSON with "capsule-XXXXXX": { ... } blocks
    json_pattern = r'"(capsule-\d+)":\s*\{\s*"success":\s*(true|false),\s*"accuracy":\s*([\d.]+),\s*([\d.]+),\s*"adherence":\s*([\d.]+)\s*\}'
    
    task_results = {}
    for match in re.finditer(json_pattern, content, re.IGNORECASE):
        capsule_id = match.group(1)
        success = match.group(2).lower() == 'true'
        accuracy = float(match.group(3))
        adherence = float(match.group(5))
        task_results[capsule_id] = TaskResult(
            success=success,
            accuracy=accuracy,
            adherence=adherence
        )
    
    # Extract summary metrics from the human-readable output
    summary = {}
    
    # Match: "Tasks: 18/45 passed (40.0%)"
    pass_match = re.search(r'Tasks:\s*(\d+)/(\d+)\s*passed\s*\(([\d.]+)%\)', content)
    if pass_match:
        summary['passed'] = int(pass_match.group(1))
        summary['total'] = int(pass_match.group(2))
        summary['pass_rate'] = float(pass_match.group(3))
    
    # Match: "Accuracy: 46.7%"
    acc_match = re.search(r'Accuracy:\s*([\d.]+)%', content)
    if acc_match:
        summary['mean_accuracy'] = float(acc_match.group(1))
    
    
    # Match: "Total Time: 3901.5s"
    time_match = re.search(r'Total Time:\s*([\d.]+)s', content)
    if time_match:
        summary['total_time'] = float(time_match.group(1))
    
    # Match model name from log header
    model_match = re.search(r'Text model:\s*(.+)', content)
    if model_match:
        summary['model'] = model_match.group(1).strip()
    
    return task_results, summary


def compare_logs(old_file: str, new_file: str):
    """Compare two CoreBench benchmark log files."""
    
    print(f"\n{'='*80}")
    print("📊 COREBENCH COMPARISON REPORT")
    print(f"{'='*80}\n")
    
    old_results, old_summary = extract_task_results(old_file)
    new_results, new_summary = extract_task_results(new_file)
    
    if not old_results:
        print(f"⚠️  Warning: No task results found in {old_file}")
        return
    if not new_results:
        print(f"⚠️  Warning: No task results found in {new_file}")
        return
    
    # Print model info
    old_model = old_summary.get('model', 'Unknown')
    new_model = new_summary.get('model', 'Unknown')
    print(f"📁 Old: {old_file}")
    print(f"   Model: {old_model}")
    print(f"📁 New: {new_file}")
    print(f"   Model: {new_model}")
    print()
    
    # Summary comparison
    print(f"{'─'*80}")
    print("📈 SUMMARY COMPARISON")
    print(f"{'─'*80}")
    
    metrics = [
        ('Pass Rate', 'pass_rate', '%', 1),
        ('Mean Accuracy', 'mean_accuracy', '%', 1),
        ('Total Time', 'total_time', 's', 1),
    ]
    
    print(f"{'Metric':<20} | {'Old':>12} | {'New':>12} | {'Δ Change':>12}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    
    for name, key, unit, decimals in metrics:
        old_val = old_summary.get(key)
        new_val = new_summary.get(key)
        
        if old_val is not None and new_val is not None:
            delta = new_val - old_val
            delta_str = f"{delta:+.{decimals}f}{unit}"
            # For time, lower is better; for others, higher is better
            if key == 'total_time':
                indicator = "🟢" if delta < 0 else ("🔴" if delta > 0 else "⚪")
            else:
                indicator = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
            print(f"{name:<20} | {old_val:>11.{decimals}f}{unit} | {new_val:>11.{decimals}f}{unit} | {indicator} {delta_str:>9}")
        else:
            old_str = f"{old_val:.{decimals}f}{unit}" if old_val else "N/A"
            new_str = f"{new_val:.{decimals}f}{unit}" if new_val else "N/A"
            print(f"{name:<20} | {old_str:>12} | {new_str:>12} | {'N/A':>12}")
    
    print()
    
    # Per-task comparison
    print(f"{'─'*80}")
    print("📋 PER-TASK COMPARISON")
    print(f"{'─'*80}")
    
    all_tasks = sorted(set(old_results.keys()) | set(new_results.keys()))
    
    improved = []
    regressed = []
    unchanged_pass = []
    unchanged_fail = []
    acc_improved = []
    acc_regressed = []
    
    print(f"{'Capsule':<20} | {'Old':^6} | {'New':^6} | {'Old Acc':>7} | {'New Acc':>7} | {'Status'}")
    print(f"{'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")
    
    for task in all_tasks:
        old_task = old_results.get(task)
        new_task = new_results.get(task)
        
        old_status = "✅" if old_task and old_task.success else "❌"
        new_status = "✅" if new_task and new_task.success else "❌"
        old_acc = f"{old_task.accuracy*100:.0f}%" if old_task else "N/A"
        new_acc = f"{new_task.accuracy*100:.0f}%" if new_task else "N/A"
        
        # Determine change
        change = ""
        if old_task and new_task:
            if not old_task.success and new_task.success:
                change = "🟢 FIXED"
                improved.append(task)
            elif old_task.success and not new_task.success:
                change = "🔴 BROKE"
                regressed.append(task)
            elif old_task.success and new_task.success:
                unchanged_pass.append(task)
                if new_task.accuracy > old_task.accuracy:
                    change = "⬆️  acc+"
                    acc_improved.append(task)
                elif new_task.accuracy < old_task.accuracy:
                    change = "⬇️  acc-"
                    acc_regressed.append(task)
            else:
                unchanged_fail.append(task)
                if new_task.accuracy > old_task.accuracy:
                    change = "⬆️  acc+"
                    acc_improved.append(task)
                elif new_task.accuracy < old_task.accuracy:
                    change = "⬇️  acc-"
                    acc_regressed.append(task)
        elif not old_task:
            change = "🆕 NEW"
        elif not new_task:
            change = "⚠️  MISSING"
        
        print(f"{task:<20} | {old_status:^6} | {new_status:^6} | {old_acc:>7} | {new_acc:>7} | {change}")
    
    # Final summary
    print()
    print(f"{'='*80}")
    print("📊 CHANGE SUMMARY")
    print(f"{'='*80}")
    
    total = len(all_tasks)
    old_pass = len([t for t in all_tasks if old_results.get(t) and old_results[t].success])
    new_pass = len([t for t in all_tasks if new_results.get(t) and new_results[t].success])
    
    print(f"\n🎯 Pass/Fail Changes:")
    print(f"   Old: {old_pass}/{total} passed ({100*old_pass/total:.1f}%)")
    print(f"   New: {new_pass}/{total} passed ({100*new_pass/total:.1f}%)")
    print(f"   Net: {new_pass - old_pass:+d} tasks")
    
    print(f"\n📈 Improvements ({len(improved)}):")
    if improved:
        for task in improved:
            old_t = old_results[task]
            new_t = new_results[task]
            print(f"   🟢 {task}: acc {old_t.accuracy*100:.0f}%→{new_t.accuracy*100:.0f}%")
    else:
        print("   (none)")
    
    print(f"\n📉 Regressions ({len(regressed)}):")
    if regressed:
        for task in regressed:
            old_t = old_results[task]
            new_t = new_results[task]
            print(f"   🔴 {task}: acc {old_t.accuracy*100:.0f}%→{new_t.accuracy*100:.0f}%")
    else:
        print("   (none)")
    
    print(f"\n⚖️  Unchanged: {len(unchanged_pass)} still passing, {len(unchanged_fail)} still failing")
    
    if acc_improved:
        print(f"\n📈 Accuracy improved (same pass/fail): {len(acc_improved)} tasks")
    if acc_regressed:
        print(f"📉 Accuracy regressed (same pass/fail): {len(acc_regressed)} tasks")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <old_log> <new_log>")
        print("\nExample:")
        print("  python compare_runs.py corebench_output_easy.txt corebench_output_llama_easy.txt")
        sys.exit(1)
    
    compare_logs(sys.argv[1], sys.argv[2])
