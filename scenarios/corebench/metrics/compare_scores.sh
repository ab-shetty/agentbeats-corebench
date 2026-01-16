#!/bin/bash

# Quick Comparison Command for the terminal - or run the script below
# Extract old scores (before new prompt)
# echo "=== OLD SCORES (20260115_20dfdec7) ===" && \
# grep -h '"type": "evaluation"' logs/traces/20260115_20dfdec7_corebench_hard/*.jsonl | \
# jq -r '[.evaluation.task_id, .evaluation.accuracy.accuracy, .evaluation.task_adherence.score] | @tsv' | \
# awk 'BEGIN {print "CAPSULE\t\t\tACCURACY\tTASK_ADH"} {printf "%-30s\t%.2f\t\t%.2f\n", $1, $2, $3}' | \
# column -t

# # After rerun, extract new scores
# echo -e "\n=== NEW SCORES (your new run) ===" && \
# grep -h '"type": "evaluation"' logs/traces/NEWRUN_*/corebench_hard/*.jsonl | \
# jq -r '[.evaluation.task_id, .evaluation.accuracy.accuracy, .evaluation.task_adherence.score] | @tsv' | \
# awk 'BEGIN {print "CAPSULE\t\t\tACCURACY\tTASK_ADH"} {printf "%-30s\t%.2f\t\t%.2f\n", $1, $2, $3}' | \
# column -t

OLD_DIR="logs/traces/20260115_20dfdec7_corebench_hard"
NEW_DIR="${1:-logs/traces/20260115_27461789_corebench_hard}"  # Pass new run dir as arg

echo "Comparing: $OLD_DIR vs $NEW_DIR"
echo ""

# Extract old scores to temp file
grep -h '"type": "evaluation"' "$OLD_DIR"/*.jsonl | \
  jq -r '[.evaluation.task_id, .evaluation.accuracy.accuracy, .evaluation.task_adherence.score] | @csv' | \
  sort > /tmp/old_scores.csv

# Extract new scores to temp file  
grep -h '"type": "evaluation"' "$NEW_DIR"/*.jsonl | \
  jq -r '[.evaluation.task_id, .evaluation.accuracy.accuracy, .evaluation.task_adherence.score] | @csv' | \
  sort > /tmp/new_scores.csv

# Compare side by side
echo "CAPSULE                        ACC_OLD  ADH_OLD  →  ACC_NEW  ADH_NEW  DELTA"
echo "=================================================================================="

join -t',' /tmp/old_scores.csv /tmp/new_scores.csv | \
  awk -F',' '{
    capsule = substr($1, 2, length($1)-2);  # Remove quotes
    acc_old = $2;
    adh_old = $3;
    acc_new = $5;
    adh_new = $6;
    delta = adh_new - adh_old;
    
    printf "%-30s %.2f    %.2f   →  %.2f    %.2f    %+.2f\n", 
           capsule, acc_old, adh_old, acc_new, adh_new, delta;
  }'

# Summary statistics
echo ""
echo "SUMMARY:"
echo "--------"
join -t',' /tmp/old_scores.csv /tmp/new_scores.csv | \
  awk -F',' '{delta += ($6 - $3); count++} 
             END {printf "Average delta: %+.3f\nCapsules compared: %d\n", delta/count, count}'

rm /tmp/old_scores.csv /tmp/new_scores.csv