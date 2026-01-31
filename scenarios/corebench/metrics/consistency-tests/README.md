# LLM Judge Consistency Validation

Validates that Task Adherence scores are reproducible across repeated evaluations.

**Our Results**: GPT-5-mini achieved EXCELLENT consistency on all test cases.

| Model        | Temperature | Avg Stdev | EXCELLENT |
| ------------ | ----------- | --------- | --------- |
| GPT-5-mini   | 1.0         | 0.027     | 10/10     |
| gpt-oss:120b | 0.0         | 0.062     | 4/10      |


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
