# CoreBench Scenario

> 🔬 Agentified version of [CoreBench](https://github.com/siegelz/core-bench) for the AgentBeats platform.

CoreBench is a computational reproducibility benchmark that tests whether AI agents 
can execute code from scientific papers and reproduce their reported results.

## Files
| `corebench_agent.py`     | Purple agent - LLM reasoning, emits tool intents         |
| `corebench_evaluator.py` | Green agent - orchestration, task loading, evaluation    |
| `mcp_server.py`          | MCP tool server (bash execution, file ops)               |
| `metrics.py`             | Evaluation metrics module (accuracy, faithfulness, etc.) |
| `scenario.toml`          | Configuration (endpoints, domain, task count)            |
| `core_test.json`         | Task definitions with ground truth answers               |

## Evaluation Metrics

| Metric              | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| **Accuracy**        | Answer correctness using 95% prediction intervals for numeric values  |
| **Reproducibility** | File restoration quality - files must have content (medium/hard only) |
| **Faithfulness**    | LLM-as-judge: are answers grounded in tool execution evidence?        |
| **Task Adherence**  | LLM-as-judge: did agent follow instructions properly?                 |
| **Efficiency**      | Steps used, tool calls made, execution time                           |

## Difficulty Levels

Configure via `domain` in `scenario.toml`:

| Domain             | What's Removed                              | Challenge                                |
| ------------------ | ------------------------------------------- | ---------------------------------------- |
| `corebench_easy`   | Nothing                                     | Execute existing code, extract results   |
| `corebench_medium` | `results/` folder                           | Re-run experiments to regenerate results |
| `corebench_hard`   | `results/` + `REPRODUCING.md` + run scripts | Figure out how to run from scratch       |

## Configuration

Key settings in `scenario.toml`:

```toml
[config]
domain = "corebench_medium"   # Difficulty level
num_tasks = 1                 # Number of tasks to run
use_mcp = true                # Enable MCP tool server
keep_traces = true            # Save execution traces
```

See the main [README](../../README.md) for full configuration options.