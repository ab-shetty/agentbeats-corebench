# LLM Judge Consistency Validation

This document describes the consistency testing performed on the LLM-as-judge used for Task Adherence evaluation in CoreBench.

## Why Consistency Testing?

The Task Adherence metric uses an LLM to evaluate agent execution quality. Since LLMs can produce different outputs for the same input, we need to validate that:

1. Repeated evaluations of the same trace produce consistent scores
2. The chosen temperature setting minimizes score variance
3. The variance is acceptable for benchmark reporting
4. The model itself is suitable for judge tasks

## Test Methodology

### Experimental Setup
- **Traces tested**: 10 agent execution traces from `corebench_hard`
- **Runs per configuration**: 10 repeated judge calls
- **Total evaluations**: 400 LLM calls across all experiments

### Models Tested
| Model               | Temperatures Tested  | Total Calls |
| ------------------- | -------------------- | ----------- |
| OpenAi gpt-oss:120b | 0.0, 0.3, 0.7        | 300         |
| OpenAI GPT-5-mini   | 1.0 (only supported) | 100         |

### Grading Criteria
| Grade     | Stdev Threshold | Interpretation                            |
| --------- | --------------- | ----------------------------------------- |
| EXCELLENT | < 0.05          | Highly consistent, scores vary by ~5% max |
| GOOD      | < 0.10          | Acceptably consistent, ~10% variance      |
| FAIR      | < 0.15          | Moderate variance, ~15%                   |
| POOR      | >= 0.15         | Unreliable, too much variance             |

---

## Results: Model Comparison

### Executive Summary

| Metric               | Ollama (T=0.0) | GPT-5-mini (T=1.0) | Winner            |
| -------------------- | -------------- | ------------------ | ----------------- |
| **Mean Score**       | 0.582          | **0.713**          | GPT-5-mini (+22%) |
| **Avg Std Dev**      | 0.062          | **0.027**          | GPT-5-mini (-56%) |
| **EXCELLENT grades** | 4/10           | **10/10**          | GPT-5-mini        |

**Conclusion: GPT-5-mini is the recommended model** - it produces both higher scores AND more consistent results.

---

## Detailed Results: gpt-oss-120b (gpt-oss:120b)

### Temperature Comparison

| Metric           | T=0.0     | T=0.3     | T=0.7 |
| ---------------- | --------- | --------- | ----- |
| **Overall Mean** | 0.582     | **0.594** | 0.582 |
| **Avg Std Dev**  | **0.062** | 0.068     | 0.071 |
| EXCELLENT grades | **4**     | 2         | 2     |
| GOOD grades      | 4         | 6         | 6     |
| FAIR grades      | 2         | 2         | 2     |
| POOR grades      | 0         | 0         | 0     |

**Finding**: Temperature 0.0 produces the most consistent results (lowest stdev, most EXCELLENT grades).

### Per-Capsule Results at T=0.0

| Capsule | Mean  | Stdev | Grade     |
| ------- | ----- | ----- | --------- |
| 8536428 | 0.843 | 0.044 | EXCELLENT |
| 3449234 | 0.743 | 0.033 | EXCELLENT |
| 9670283 | 0.673 | 0.021 | EXCELLENT |
| 8807709 | 0.690 | 0.046 | EXCELLENT |
| 5507257 | 0.615 | 0.100 | FAIR      |
| 9641396 | 0.626 | 0.091 | GOOD      |
| 8234136 | 0.468 | 0.071 | GOOD      |
| 6003668 | 0.429 | 0.055 | GOOD      |
| 9660931 | 0.417 | 0.053 | GOOD      |
| 6049678 | 0.313 | 0.102 | FAIR      |

---

## Detailed Results: GPT-5-mini

### Per-Capsule Results at T=1.0

| Capsule | Mean      | Stdev | Grade     |
| ------- | --------- | ----- | --------- |
| 8536428 | **0.995** | 0.005 | EXCELLENT |
| 8807709 | 0.824     | 0.016 | EXCELLENT |
| 3449234 | 0.796     | 0.028 | EXCELLENT |
| 9641396 | 0.791     | 0.026 | EXCELLENT |
| 5507257 | 0.766     | 0.035 | EXCELLENT |
| 9670283 | 0.753     | 0.022 | EXCELLENT |
| 8234136 | 0.698     | 0.044 | EXCELLENT |
| 6003668 | 0.563     | 0.030 | EXCELLENT |
| 6049678 | 0.484     | 0.028 | EXCELLENT |
| 9660931 | 0.460     | 0.033 | EXCELLENT |

**Key observation**: GPT-5-mini achieves EXCELLENT consistency on ALL 10 capsules, even at T=1.0.

---

## Head-to-Head: Per-Capsule Comparison

| Capsule | gpt-oss-120b Mean | GPT-5 Mean | gpt-oss-120b σ | GPT-5 σ | More Consistent |
| ------- | ----------------- | ---------- | -------------- | ------- | --------------- |
| 5507257 | 0.615             | 0.766      | 0.100          | 0.035   | GPT-5           |
| 3449234 | 0.743             | 0.796      | 0.033          | 0.028   | GPT-5           |
| 9670283 | 0.673             | 0.753      | 0.021          | 0.022   | gpt-oss-120b    |
| 8536428 | 0.843             | 0.995      | 0.044          | 0.005   | GPT-5           |
| 8807709 | 0.690             | 0.824      | 0.046          | 0.016   | GPT-5           |
| 6049678 | 0.313             | 0.484      | 0.102          | 0.028   | GPT-5           |
| 9641396 | 0.626             | 0.791      | 0.091          | 0.026   | GPT-5           |
| 6003668 | 0.429             | 0.563      | 0.055          | 0.030   | GPT-5           |
| 8234136 | 0.468             | 0.698      | 0.071          | 0.044   | GPT-5           |
| 9660931 | 0.417             | 0.460      | 0.053          | 0.033   | GPT-5           |

**Result**: GPT-5-mini is more consistent on 9/10 capsules.

---

## Component Score Analysis

The judge scores four components. Here's how variance breaks down:

| Component       | Max Points | gpt-oss-120b Avg σ | GPT-5 Avg σ |
| --------------- | ---------- | ------------------ | ----------- |
| Core Process    | 50         | ~5.1 pts           | ~2.3 pts    |
| Problem Solving | 25         | ~1.6 pts           | ~1.1 pts    |
| Discovery       | 15         | ~1.0 pts           | ~0.7 pts    |
| Technical       | 10         | ~0.5 pts           | ~0.4 pts    |

**Finding**: Core Process has the highest variance for both models, but GPT-5-mini reduces it by ~55%.

---

## Conclusions & Recommendations

### Key Findings

1. **GPT-5-mini is significantly better** than gpt-oss-120b gpt-oss:120b for judge tasks
   - 22% higher mean scores
   - 56% lower variance
   - 10/10 EXCELLENT vs 4/10 EXCELLENT

2. **Temperature 0.0 is best for gpt-oss-120b** when GPT-5-mini is unavailable
   - More EXCELLENT grades than T=0.3 or T=0.7
   - The +2% mean score at T=0.3 isn't worth the consistency loss

3. **GPT-5-mini works well at T=1.0** (its only supported temperature)
   - Achieves EXCELLENT consistency despite non-deterministic setting
   - Suggests the model has strong internal consistency

### Production Recommendation

```python
# Preferred: Use GPT-5-mini
model = "gpt-5-mini"
temperature = 1.0  # Only supported value

# Fallback: Use gpt-oss-120b with T=0.0
model = "openai/gpt-oss:120b"
temperature = 0.0
```

---

## Reproducing These Tests

### Full Model Comparison
```bash
# Test gpt-oss-120b at multiple temperatures (300 calls, ~30 min)
uv run python consistency-tests/run_batch_consistency.py \
    logs/traces/20260129_c6e93559_corebench_hard/ \
    --runs 10 --temps 0.0 0.3 0.7 \
    --output-dir consistency-tests/

# Test GPT-5-mini (100 calls, ~40 min)
uv run python consistency-tests/run_batch_consistency.py \
    logs/traces/20260129_c6e93559_corebench_hard/ \
    --model gpt-5-mini --runs 10 --temps 1.0 \
    --output-dir consistency-tests/gpt5mini_results
```

### Quick Validation
```bash
# Quick test (40 calls, ~5 min)
uv run python consistency-tests/run_batch_consistency.py \
    logs/traces/20260129_c6e93559_corebench_hard/ \
    --runs 5 --max-traces 4
```

---

## Data Files

All raw data is available in the `consistency-tests/` directory:

| File                                       | Description                                                |
| ------------------------------------------ | ---------------------------------------------------------- |
| `batch_full_results.json`                  | gpt-oss-120b results (300 evaluations) with full reasoning |
| `gpt5mini_results/batch_full_results.json` | GPT-5-mini results (100 evaluations)                       |
| `final_results.csv`                        | Tabular format with all component scores                   |
| `figures/`                                 | Visualizations (box plots, heatmaps, etc.)                 |

### Sample Data Structure
```json
{
  "capsule_id": "8536428",
  "mean": 0.995,
  "stdev": 0.005,
  "grade": "EXCELLENT",
  "runs": [
    {
      "run": 1,
      "score": 0.99,
      "component_scores": {
        "core_process": "48/50",
        "problem_solving": "24/25",
        "discovery": "14/15",
        "technical": "10/10"
      },
      "reasoning": "The agent successfully...",
      "strengths": ["..."],
      "weaknesses": ["..."]
    }
  ]
}
```


## References

- Test script: `consistency-tests/run_batch_consistency.py`
- Single-trace test: `consistency-tests/judge_consistency_test.py`
- Visualization script: `consistency-tests/generate_visualizations.py`
- Temperature analysis report: `consistency-tests/LLM_JUDGE_TEMPERATURE_REPORT.md`
