# Task Adherence (LLM-as-Judge)

Qualitative assessment of agent execution quality using an LLM judge.

## What This Does

Unlike `methodology_score` (deterministic), task adherence uses an LLM to provide
qualitative assessment of how well the agent executed the task. It answers questions like:
- Did the agent figure out how to run the code?
- How well did the agent handle obstacles and errors?
- How efficiently did the agent work?

The judge provides a score (0.0-1.0), reasoning, strengths, and weaknesses for improving purple agent performance.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TASK ADHERENCE EVALUATION FLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │    domain    │  │ task_prompt  │  │  questions   │  │ tool_calls/  │
    │              │  │              │  │              │  │   results    │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │                 │
           │                 └────────┬────────┴─────────────────┘
           │                          │
           ▼                          ▼
    ┌─────────────────┐    ┌─────────────────────────────────┐
    │ DOMAIN_CRITERIA │    │     Build Judge Context         │
    │ [domain]        │───▶│  - Domain-specific criteria     │
    └─────────────────┘    │  - Capsule docs (README files)  │
                           │  - Tool interactions (all)      │
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │       LLM Judge Prompt          │
                           │   (TASK_ADHERENCE_PROMPT)       │
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │     Parse JSON Response         │
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                                 ┌─────────────────────┐
                                 │ TaskAdherenceMetrics│
                                 └─────────────────────┘
```

## Judge Model

The recommended judge model is `gpt-5-mini` based on consistency testing (see `LLM_JUDGE_CONSISTENCY.md`):
- 22% higher mean scores than gpt-oss-120b
- 56% lower variance
- EXCELLENT consistency on all test capsules

## What the Judge Receives

```
┌─────────────────────────────────────────────────────────────────┐
│ JUDGE INPUT CONTEXT                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Task Instruction                                            │
│     - e.g., "Run main.py and report the accuracy"               │
│                                                                 │
│  2. Questions to Answer                                         │
│     - List of expected questions from ground truth              │
│     - e.g., ["Report the accuracy", "Report the F1 score"]      │
│                                                                 │
│  3. Capsule Docs                                                │
│     - First README found in root or code/ directory             │
│     - Case-insensitive match (README.md, readme.txt, etc.)      │
│     - Max 10KB per file                                         │
│                                                                 │
│  4. Domain-Specific Criteria                                    │
│     - Injected from DOMAIN_CRITERIA[domain]                     │
│                                                                 │
│  5. Execution Stats                                             │
│     - tool_calls_count: Number of tool calls made               │
│                                                                 │
│  6. Tool Interactions (see format below)                        │
│     - All tool call/result pairs shown (system max: 40)         │
│     - Arguments truncated at 1200 chars if needed               │
│     - Results not truncated (MCP tools handle truncation)       │
│                                                                 │
│  7. Final Answer Status                                         │
│     - Whether agent provided a final answer (yes/no)            │
│                                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Tool Interaction Format

Each tool call/result pair is formatted as:

```
[Turn N] tool_name
Arguments:
{
  "command": "python script.py",
  "other_arg": "value"
}
Exit code: 1                    <- only shown if non-zero (error)
⚠️ Command timed out            <- only shown if timeout occurred
Output:
<tool output here>
Hint: <evaluator hint>          <- only shown if hint was provided
```

- **Arguments**: JSON with indent=2, truncated at 1200 chars if needed
- **Exit code**: Only shown for errors (non-zero), success is implicit
- **Timeout**: Only shown when a command timed out
- **Output**: Full tool result (MCP tools handle their own truncation)
- **Hint**: Evaluator guidance shown to agent (e.g., "File read error...")

## Domain-Specific Criteria (Hard Mode)

Hard mode is the primary focus. The criteria injected for hard mode:

```
┌─────────────────────────────────────────────────────────────────┐
│ HARD MODE CRITERIA (injected via DOMAIN_CRITERIA)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GOAL:                                                          │
│    Figure out how to run the code and reproduce results         │
│                                                                 │
│  POSSIBLE APPROACHES (suggestions, not requirements):           │
│    - Reading README.md or other documentation                   │
│    - Inspecting scripts to understand dependencies              │
│    - Exploring the directory structure                          │
│    - Trial and error with debugging                             │
│                                                                 │
│  GOLD STANDARD WORKFLOW:                                        │
│    Understand codebase → Execute code → Debug errors →          │
│    Extract results                                              │
│                                                                 │
│  CORE PROCESS SCORING GUIDE (50%):                              │
│    45-50: Successfully executed code and extracted results      │
│    30-44: Executed but didn't extract, or partial execution     │
│    15-29: Explored codebase but never executed                  │
│     0-14: No meaningful attempt                                 │
│                                                                 │
│  RED FLAGS:                                                     │
│    - Never attempted to run code                                │
│    - Gave up after first error without trying to debug          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scoring Rubric

```
┌─────────────────────────────────────────────────────────────────┐
│ COMPONENT WEIGHTS (100% total)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Core Process (50%)                                             │
│   40-50: Executed code AND extracted results (full success)     │
│   25-39: Executed code but results incomplete/missing           │
│   10-24: Attempted execution but script failed/crashed          │ 
│   0-9: Never attempted to run code                              │
│                                                                 │
│  Problem Solving (25%)                                          │
│    20-25: Debugged errors and persisted                         │
│    10-19: Tried to fix but gave up                              │
│     0-9:  No error handling                                     │
│                                                                 │
│  Discovery (15%)                                                │
│    12-15: Direct path to needed information                     │
│     6-11: Some wandering                                        │
│     0-5:  Lost                                                  │
│                                                                 │
│  Technical (10%)                                                │
│    Command correctness, avoiding redundant operations           │
│                                                                 │
│  Final score = (Core + Problem + Discovery + Technical) / 100   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Output Format

```
┌─────────────────────────────────────────────────────────────────┐
│ TaskAdherenceMetrics                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  score: float                 # 0.0 - 1.0                       │
│                                                                 │
│  reasoning: str               # Component breakdown +           │
│                               # observations from judge         │
│                                                                 │
│  strengths: list[str]         # Specific good behaviors         │
│                               # e.g., "Located relevant script" │
│                                                                 │
│  weaknesses: list[str]        # Areas for improvement           │
│                               # e.g., "Gave up after error"     │
│                                                                 │
│  status: str                  # "success" or "error"            │
│                                                                 │
│  error_message: str | None    # If judge API call failed        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Example Judge Output

Real example from capsule-5507257 (hard mode):

```json
{
  "score": 0.68,
  "reasoning": "The agent located the relevant script, inspected the README, and attempted to run the testing code. It identified missing dependencies, installed TensorFlow and Keras, and corrected data path strings. However, the script never completed successfully because TensorFlow could not be imported on the current platform, so no accuracy numbers were obtained.
  **Core Process (32/50)** – Explored codebase, installed packages, edited file paths and launched script, but execution ended with import error.
  **Problem Solving (18/25)** – Multiple attempts to resolve TensorFlow import issue demonstrate persistence, though final problem remained unsolved.**Discovery (11/15)** – README and script found quickly; also discovered data directory and necessary packages.
  **Technical (7/10)** – Commands generally correct. Redundant install attempts slightly lower score.",
  "component_scores": {
    "core_process": "32/50",
    "problem_solving": "18/25",
    "discovery": "11/15",
    "technical": "7/10"
  },
  "strengths": [
    "Located and inspected the relevant script and README quickly",
    "Identified missing dependencies and attempted to install them",
    "Corrected file path issues to point at the data directory",
    "Persisted through several error messages and tried alternative package versions"
  ],
  "weaknesses": [
    "Did not resolve the TensorFlow import failure on the aarch64 platform",
    "Repeated redundant pip install commands without checking compatibility",
    "Did not create or use a virtual environment to isolate package versions",
    "Failed to produce the required accuracy metric"
  ]
}
```
