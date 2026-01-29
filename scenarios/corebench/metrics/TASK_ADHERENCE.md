# Task Adherence (LLM-as-Judge)

Qualitative assessment of agent execution quality using an LLM judge.

## What This Does

Unlike `methodology_score` (deterministic), task adherence uses an LLM to provide
qualitative assessment of how well the agent executed the task. It answers questions like:
- Did the agent figure out how to run the code?
- How well did the agent handle obstacles and errors?
- How efficiently did the agent work?

The judge provides a score (0.0-1.0), reasoning, strengths, and weaknesses for debugging.

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
                           │  - Tool interactions (24 max)   │
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
│     - README files from workspace/environment/                  │
│     - Up to 4 files, max 6KB each, 12KB total                   │
│     - Shows what documentation was available                    │
│                                                                 │
│  4. Domain-Specific Criteria                                    │
│     - Injected from DOMAIN_CRITERIA[domain]                     │
│     - Contains: workflow, scoring guide, red flags              │
│                                                                 │
│  5. Execution Stats                                             │
│     - tool_calls_count: Number of tool calls made               │
│     - command_timeouts: Commands that hit timeout               │
│                                                                 │
│  6. Tool Interactions                                           │
│     - Up to 24 tool call/result pairs                           │
│     - Truncated if more (8 head + 16 tail)                      │
│     - Shows arguments, exit codes, summaries                    │
│                                                                 │
│  7. Final Answer Status                                         │
│     - Whether agent provided a final answer (yes/no)            │
│                                                                 │
│  NOT included (intentionally):                                  │
│     - Accuracy/correctness (avoids anchoring bias)              │
│     - Info about "deleted" files (judge can't verify)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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
│    45-50: Successfully executed code and extracted results      │
│    30-44: Executed but didn't extract, or partial execution     │
│    15-29: Explored codebase but never executed                  │
│     0-14: No meaningful attempt                                 │
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

```json
{
  "score": 0.6,
  "reasoning": "The agent managed to locate the notebook and execute it
    (Core Process 30/50), but never extracted the required values. It
    attempted dependency installation but gave up when pip failed
    (Problem Solving 12/25). Discovery was efficient - quickly found
    the main script (Discovery 12/15). Technical execution was reasonable
    but incomplete (Technical 6/10).",
  "component_scores": {
    "core_process": "30/50",
    "problem_solving": "12/25",
    "discovery": "12/15",
    "technical": "6/10"
  },
  "strengths": [
    "Quickly identified the entry point script",
    "Used appropriate execution command",
    "Read README to understand project structure"
  ],
  "weaknesses": [
    "Did not extract or report the requested values",
    "Gave up after pip install failed instead of debugging"
  ]
}
```
