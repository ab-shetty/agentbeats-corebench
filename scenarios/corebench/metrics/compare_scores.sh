#!/bin/bash

# Compare two CoreBench trace directories and show per-capsule deltas (accuracy, task adherence, methodology).
# How to run
# ----------
# From repo root:
#   bash scenarios/corebench/metrics/compare_scores.sh <OLD_TRACE_DIR> <NEW_TRACE_DIR>
# Example:
#   bash scenarios/corebench/metrics/compare_scores.sh \
#     logs/traces/20260129_c6e93559_corebench_hard \
#     logs/traces/20260130_63cd6611_corebench_hard
# Requires: jq, *.jsonl trace files in each dir.


set -euo pipefail

usage() {
  echo "Usage: $0 <OLD_RUN_DIR> <NEW_RUN_DIR>"
  echo "Example: $0 logs/traces/20260129_c6e93559_corebench_hard logs/traces/20260130_63cd6611_corebench_hard"
  exit 1
}

if [ $# -lt 2 ]; then
  usage
fi

OLD_DIR="$1"
NEW_DIR="$2"

if [ ! -d "$OLD_DIR" ] || [ ! -d "$NEW_DIR" ]; then
  echo "One of the directories does not exist:"
  echo "  OLD_DIR=$OLD_DIR"
  echo "  NEW_DIR=$NEW_DIR"
  exit 1
fi

tmp_old=$(mktemp)
tmp_new=$(mktemp)

# Extract metrics: capsule, accuracy, written_acc, vision_acc, adherence, methodology, steps, tools, time_sec, success
extract() {
  local dir="$1" out="$2"
  grep -h '"type": "evaluation"' "$dir"/*.jsonl | \
    jq -r '[.evaluation.task_id,
            .evaluation.accuracy.accuracy,
            (.evaluation.accuracy.written_accuracy // 0),
            (.evaluation.accuracy.vision_accuracy // 0),
            .evaluation.task_adherence.score,
            (.evaluation.methodology_metrics.methodology_score // 0),
            (.evaluation.efficiency.steps_used // 0),
            (.evaluation.efficiency.tool_calls // .evaluation.efficiency.tool_calls_count // 0),
            (.evaluation.efficiency.time_seconds // 0),
            (.evaluation.success // false)
           ] | @csv' | sort > "$out"
}

extract "$OLD_DIR" "$tmp_old"
extract "$NEW_DIR" "$tmp_new"

echo "Comparing:"
echo "  OLD: $OLD_DIR"
echo "  NEW: $NEW_DIR"
echo ""
echo "CAPSULE                        ACC_O  ADH_O  METH_O  →  ACC_N  ADH_N  METH_N  ΔADH  ΔACC  ΔMETH"
echo "================================================================================================"

join -t',' "$tmp_old" "$tmp_new" | \
  awk -F',' '{
    capsule = substr($1, 2, length($1)-2);
    acc_o=$2; adh_o=$5; meth_o=$6;
    acc_n=$10; adh_n=$13; meth_n=$14;
    printf "%-30s %.2f  %.2f  %.2f  →  %.2f  %.2f  %.2f  %+.2f  %+.2f  %+.2f\n",
      capsule, acc_o, adh_o, meth_o, acc_n, adh_n, meth_n, adh_n-adh_o, acc_n-acc_o, meth_n-meth_o;
  }'

echo ""
echo "SUMMARY"
echo "-------"
join -t',' "$tmp_old" "$tmp_new" | \
  awk -F',' '{d_adh+=($13-$5); d_acc+=($10-$2); d_meth+=($14-$6); c++}
             END {if(c==0){print "No overlapping capsules."} else {
               printf "Capsules compared: %d\nAvg ΔAdh: %+.3f\nAvg ΔAcc: %+.3f\nAvg ΔMeth: %+.3f\n", c, d_adh/c, d_acc/c, d_meth/c
             }}'

rm -f "$tmp_old" "$tmp_new"
