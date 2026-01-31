# LLM Judge Consistency Validation

Validates that Task Adherence scores are reproducible across repeated evaluations.

**Our Results**: GPT-5-mini achieved EXCELLENT consistency on all test cases.

| Model        | Temperature | Avg Stdev | EXCELLENT |
| ------------ | ----------- | --------- | --------- |
| GPT-5-mini   | 1.0         | 0.027     | 10/10     |
| gpt-oss:120b | 0.0         | 0.062     | 4/10      |



### Analysis: LLM Judge Consistency Thresholds

**Metric:** Standard Deviation ($\sigma$) on a normalized 0.0–1.0 scale.
**Objective:** Evaluation noise must be lower than the signal of model improvement.

| Grade         | Range ($\sigma$) | Interpretation & Source                                                                                                                                                                                                                         |
| :------------ | :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EXCELLENT** | **< 0.05**       | **Production Grade.** Measurement noise is negligible (<5%), allowing for reliable regression testing. This meets strict [G-Eval standards](https://arxiv.org/abs/2303.16634) for detecting even micro-improvements.                            |
| **GOOD**      | **< 0.10**       | **Stable.** Acceptable for tracking aggregate trends. While generally reliable, it may occasionally "flip" on borderline cases, aligning with [DeepEval Self-Consistency](https://docs.confident-ai.com/docs/metrics-self-consistency) targets. |
| **FAIR**      | **< 0.15**       | **High Variance.** The judge disagrees with itself as often as humans do (approaching [Human Inter-Rater Agreement](https://arxiv.org/abs/2306.05685) error rates). Usable only with "Majority Voting" (averaging 3+ runs).                     |
| **POOR**      | **≥ 0.15**       | **Unusable.** The signal-to-noise ratio is too low. It is impossible to distinguish whether a score change is due to agent performance or evaluator randomness.                                                                                 |


## How to Run Consistency Tests
```bash
# 4 traces, 5 runs each
python run_batch_consistency.py path/to/traces/ --runs 5 --max-traces 4
# pass model and temperatures
python run_batch_consistency.py path/to/traces/ --model gpt-oss-120b --runs 10 --temps 0.0 0.3 0.7
```

**Grading**

| Grade     | Stdev   | Interpretation   |
| --------- | ------- | ---------------- |
| EXCELLENT | < 0.05  | ~5% variance max |
| GOOD      | < 0.10  | ~10% variance    |
| FAIR      | < 0.15  | ~15% variance    |
| POOR      | >= 0.15 | Unreliable       |

**Output**

Results are saved to `consistency-results/` (or `--output-dir`):
- `results_temp_X.csv` per temperature
- `summary.csv` combined comparison
- `CONSISTENCY_REPORT.md` with recommendations
