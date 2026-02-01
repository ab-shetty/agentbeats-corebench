# CORE-Bench for AgentBeats

> 🔬 Agentified version of [CoreBench](https://github.com/siegelz/core-bench) for the AgentBeats platform.

Repo structure:
```
scenarios/corebench/
├── scenario.toml                     # scenario configuration (domain, tasks, models, caching)
├── corebench_agent.py                # Purple agent
├── corebench_evaluator.py            # Green agent orchestrator
├── mcp_server.py                     # MCP tools
├── mdconvert.py                      # markdown → HTML/PNG helper for vision questions
├── planning_prompts.yaml             # ReAct-style planning prompts
├── model_prices.py                   # token pricing for cost reporting
├── shared_logging.py                 # shared logging helpers
├── Dockerfile.corebench-agent        # container for Purple agent
├── Dockerfile.corebench-evaluator    # container for Green agent
├── metrics/    
│   ├── consistency-tests/            # LLM judge consistency tests                   
│   ├── metrics.py                    # evaluation metrics
│   ├── models.py                     # dataclasses
│   └── internal/                     # metric deep dives
├── capsules/                         # cache capsules
├── workspace/                        # Purple agent sandbox (code/data/environment)
└── test_mcp_tools_jsonrpc_full.py    
```