# CORE-Bench for AgentBeats

> 🔬 Agentified version of [CoreBench](https://github.com/siegelz/core-bench) for the AgentBeats platform.


## MCP Tools

The MCP server provides these sandboxed tools:

| Tool                          | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `execute_bash`                | Run shell commands (sandboxed to workspace)      |
| `inspect_file_as_text`        | Read files as markdown text                      |
| `query_vision_language_model` | Analyze images with a vision model               |
| `file_content_search`         | Search file contents with regex patterns         |
| `edit_file`                   | Modify files (view, str_replace, create, insert) |
| `web_search`                  | Search the web via DuckDuckGo                    |
| `visit_webpage`               | Fetch and extract text from webpages             |
| `python_interpreter`          | Execute Python code snippets                     |

<details>
<summary><strong>Testing the MCP Server</strong></summary>
To test the MCP server functionality using an interactive, web-based MCP inspector:

1. Navigate to `scenarios/corebench` and run:
```bash
uv run mcp dev mcp_server.py
```

2. Click **Connect** > **Tools** > **List Tools** > Select tool to test

![MCP Inspector](../../assets/image.png)

3. Alternatively, run the Python test harness (starts MCP server and communicates via JSON-RPC):
```bash
uv run python test_mcp_tools_jsonrpc_full.py
```
</details>

## Repo structure:
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