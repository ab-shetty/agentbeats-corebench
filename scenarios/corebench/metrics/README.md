# CoreBench Evaluation Metrics

As a Green Agent for AgentBeats, CoreBench evaluates Purple Agents on their ability to **reproduce results from published computational research**. Where the original benchmark measured only accuracy, we add **process-aware evaluation** because *process is the product*. We distinguish agents that genuinely reproduce experiments from those taking shortcuts.

## Evaluation Dimensions

| Metric             | Type          | What It Measures                                             |
| ------------------ | ------------- | ------------------------------------------------------------ |
| **Accuracy**       | Deterministic | Did the agent get the correct answer?                        |
| **Methodology**    | Deterministic | Did the agent follow proper scientific reproduction process? |
| **Task Adherence** | LLM Judge     | How well did the agent execute the task? (qualitative)       |
| **Efficiency**     | Deterministic | How resource-efficient was the agent?                        |

This multi-dimensional approach captures agents that get right answers through wrong methods (high accuracy, low methodology) and agents that follow correct processes but hit environment issues (low accuracy, high methodology).

---

## Accuracy

Measures answer correctness against ground truth from published research.

**Numeric values**: Uses 95% prediction intervals to account for stochastic experiments (e.g., ML training with different random seeds):

```
Interval = mean ± t(0.975, n-1) × std × sqrt(1 + 1/n)
```

A submitted value is correct if it falls within this interval. For single ground truth values, exact match is used. Scale mismatches (e.g., 0.96 vs 96%) are detected and logged for debugging.

---

## Methodology Score

Deterministic scoring based on observable trace events. Answers: *Did the agent follow the scientific method for reproduction?*

### Hard Mode Scoring (Primary)

| Component             | Weight | Criteria                                           |
| --------------------- | ------ | -------------------------------------------------- |
| Documentation Reading | 10%    | Read README.md or relevant docs                    |
| Script Reading        | 20%    | Inspected target script before execution           |
| Execution Coverage    | 30%    | Ran expected scripts (partial credit for attempts) |
| Successful Execution  | 30%    | At least one script completed successfully         |
| Error Recovery        | 10%    | Persisted through errors vs. giving up             |

**Penalty**: −5% if agent failed execution without attempting dependency installation.

### Anti-Pattern Detection

We detect "shortcut" behaviors that circumvent genuine reproduction:

- **Reading pre-existing results**: Agent accessed `results/` directory before running any code (looking up answers instead of computing them)

### Why Both Methodology AND Task Adherence?

These metrics are complementary:

| Methodology (Deterministic) | Task Adherence (LLM Judge)        |
| --------------------------- | --------------------------------- |
| *What* did the agent do?    | *How well* did it do it?          |
| Binary/quantitative signals | Qualitative assessment            |
| Read docs? (yes/no)         | Discovery efficiency              |
| Execution coverage (%)      | Problem-solving quality           |
| Reproducible scoring        | Actionable feedback for debugging |

---

## Task Adherence (LLM-as-Judge)

Qualitative assessment of execution quality. Provides scores, reasoning, and specific feedback.

### Scoring Rubric

| Component       | Weight | Criteria                                     |
| --------------- | ------ | -------------------------------------------- |
| Core Process    | 50%    | Executed code AND extracted results          |
| Problem Solving | 25%    | Debugged errors, persisted through obstacles |
| Discovery       | 15%    | Found information efficiently                |
| Technical       | 10%    | Command correctness, avoiding redundancy     |

### Example Output

```json
{
  "score": 0.68,
  "reasoning": "Agent located script, installed dependencies, but TensorFlow import failed on platform...",
  "strengths": ["Located script quickly", "Identified missing dependencies", "Persisted through errors"],
  "weaknesses": ["Did not resolve platform compatibility", "Redundant install attempts"]
}
```

### LLM Judge Consistency

Following best practices for LLM-as-judge reliability, we validated consistency across repeated evaluations.

**Test methodology**: 10 traces × 10 runs per configuration = 100+ evaluations per model.

| Model        | Temperature | Avg Std Dev | Consistency                  |
| ------------ | ----------- | ----------- | ---------------------------- |
| GPT-5-mini   | 1.0         | 0.027       | **EXCELLENT** (10/10 traces) |
| gpt-oss:120b | 0.0         | 0.062       | GOOD (4/10 EXCELLENT)        |

**Recommended configuration**: GPT-5-mini achieves EXCELLENT consistency (σ < 0.05) on all test capsules, with 56% lower variance than alternatives.

**Consistency grades**:
- EXCELLENT: σ < 0.05 (scores vary ~5% max)
- GOOD: σ < 0.10 (~10% variance)
- FAIR: σ < 0.15 (~15% variance)
- POOR: σ ≥ 0.15 (unreliable)

---

## Efficiency

Resource usage during task execution:

| Metric             | Description              |
| ------------------ | ------------------------ |
| `steps_used`       | Interaction turns taken  |
| `tool_calls_count` | Tool invocations         |
| `time_seconds`     | Total execution time     |
| `protocol_errors`  | JSON/format errors       |
| `command_timeouts` | Commands hitting timeout |
| `token_cost`       | Estimated API cost (USD) |

Token costs are computed from input/output token counts using a model price dictionary, enabling cost-aware evaluation across different LLM backends.

---

## Interpreting Results

| Pattern                         | Interpretation                                       |
| ------------------------------- | ---------------------------------------------------- |
| High Accuracy + Low Methodology | Agent found shortcuts (e.g., reading cached results) |
| Low Accuracy + High Methodology | Environment issues despite correct process           |
| All metrics aligned             | Agent succeeded or failed consistently               |

---

## Reproducibility

All evaluation metrics are designed for reproducibility:

1. **Accuracy**: Deterministic comparison against ground truth
2. **Methodology**: Computed from trace events deterministically
3. **Task Adherence**: LLM judge with validated consistency (see above)
4. **Efficiency**: Direct measurement from execution logs

To reproduce LLM judge consistency tests:
```bash
uv run python consistency-tests/run_batch_consistency.py \
    logs/traces/<run_dir>/ --model gpt-5-mini --runs 10
```