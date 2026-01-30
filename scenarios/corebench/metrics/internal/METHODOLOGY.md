# Methodology Metrics

Deterministic scoring of agent behavior based on observable trace events.

## What This Does

This module analyzes execution traces to score how well an agent followed the correct methodology for reproducing scientific results. It answers questions like:
- Did the agent read the documentation before running code?
- Did the agent run the correct script?
- Did the agent recover from errors?

The final `methodology_score` (0.0–1.0) reflects process quality, independent of whether the agent got the right answer.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METHODOLOGY SCORING FLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  tool_calls  │      │ tool_results │      │ task_prompt  │
    │    (trace)   │      │    (trace)   │      │   (string)   │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │
           └──────────┬──────────┴─────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ extract_methodology_   │
         │       metrics()        │
         └───────────┬────────────┘
                     │
        ┌────────────┼────────────┬────────────┬────────────┐
        ▼            ▼            ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Discovery│ │Execution│ │ Script  │ │  Error  │ │  Anti-  │
   │  Phase  │ │  Phase  │ │Coverage │ │Recovery │ │Patterns │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │           │
        └───────────┴───────────┴─────┬─────┴───────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │  _compute_methodology_score()   │
                    │         (domain-specific)       │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  MethodologyMetrics │
                          │    (final output)   │
                          └─────────────────────┘
```

## Scoring by Domain

### Hard Mode (corebench_hard)

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARD MODE SCORING (100%)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ Doc Reading 10% │  Read README.md?                           │
│  └────────┬────────┘  YES → +0.10 | NO → +0.00                  │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │Script Reading20%│  Read target script?                       │
│  └────────┬────────┘  YES → +0.20 | NO → +0.00                  │
│           │                                                     │
│  ┌────────▼────────┐  Did it execute scripts?                   │
│  │Exec Coverage 30%│  YES → +0.30 × (executed / expected)       │
│  └────────┬────────┘  NO but attempted? → +0.15 (partial)       │
│           │           NO attempt → +0.00                        │
│           │                                                     │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │Successful Ex 30%│  At least one script succeeded?            │
│  └────────┬────────┘  YES → +0.30 | NO → +0.00                  │
│           │                                                     │
│  ┌────────▼────────┐  persistence_score × 0.10                  │
│  │Error Recovery10%│  NO errors → 1.0 (full points)             │
│  └────────┬────────┘  Errors recovered → higher score           │
│           │           Max failures streak → lower score         │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │    Penalties    │  No deps install + failed exec? → -0.05    │
│  └────────┬────────┘  Otherwise → +0.00                         │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  FINAL SCORE    │  Sum all components, clamp to [0.0, 1.0]   │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Medium Mode (corebench_medium)

In medium mode, REPRODUCING.md is available but results/ is deleted.
Agent must follow instructions and re-run code.

```
┌─────────────────────────────────────────────────────────────────┐
│                   MEDIUM MODE SCORING (100%)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ Doc Reading 25% │  Read REPRODUCING.md?                      │
│  └────────┬────────┘  YES → +0.25 | NO → +0.00                  │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │Exec Coverage 35%│  (coverage = executed/expected, 0.0-1.0)   │
│  └────────┬────────┘  YES → +0.35 × coverage | NO → +0.00       │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │Successful Ex 25%│  At least one script succeeded?            │
│  └────────┬────────┘  YES → +0.25 | NO → +0.00                  │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │Error Recovery15%│  persistence_score × 0.15                  │
│  └────────┬────────┘  (see formula in Error Recovery section)   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  FINAL SCORE    │  Sum all components, clamp to [0.0, 1.0]   │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Easy Mode (corebench_easy)

In easy mode, results/ directory exists with pre-computed outputs.
Agent should read results, NOT execute code.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EASY MODE SCORING (100%)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │Doc Reading 100% │  Read results/ files?                      │
│  └────────┬────────┘  YES → +1.00 | NO → +0.00                  │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │    Penalties    │  Tried to execute code?                    │
│  └────────┬────────┘  YES → -0.70 | NO → +0.00                  │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  FINAL SCORE    │  Clamp to [0.0, 1.0]                       │
│  └─────────────────┘                                            │
│                                                                 │
│  NOTE: Running code in easy mode is an anti-pattern because     │
│        results already exist - the agent should just read them. │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Execution Status Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION STATUS LOGIC                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │  expected_scripts   │
                    │   from task_prompt  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Is list empty?     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │ YES                             │ NO
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  Vague prompt   │               │ Compare with    │
    │ (no script name)│               │ executed_scripts│
    └────────┬────────┘               └────────┬────────┘
             │                                 │
    ┌────────▼────────┐          ┌─────────────┴─────────────┐
    │successful_exec? │          │                           │
    └────────┬────────┘          ▼                           ▼
             │            ┌────────────┐              ┌────────────┐
    YES: "success"        │ Ran right  │              │ Ran wrong  │
    NO but tried:"failed" │  script?   │              │  script?   │
    NO never tried:       └─────┬──────┘              └─────┬──────┘
      "no_attempt"              │                           │
                    ┌───────────┴───────────┐               │
                    ▼                       ▼               ▼
              ┌──────────┐           ┌──────────┐    ┌──────────────┐
              │ SUCCESS  │           │ PARTIAL  │    │ WRONG_SCRIPT │
              │ (all ran)│           │(some ran)│    │              │
              └──────────┘           └──────────┘    └──────────────┘

              If right script attempted but failed:
              ┌──────────────┐
              │TARGET_FAILED │
              └──────────────┘

              If nothing attempted:
              ┌──────────────┐
              │  NO_ATTEMPT  │
              └──────────────┘

    NOTE: Vague prompts get same partial credit (0.15) as non-vague
          when attempted but failed.
```

## Key Metrics Extracted

### Discovery Phase
```
┌─────────────────────────────────────────────────────────────────┐
│ DISCOVERY METRICS                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  read_documentation: bool    Did agent read README.md?          │
│  docs_read: list[str]        Which doc files were read          │
│                                                                 │
│  read_target_script: bool    Did agent read the script to run?  │
│  scripts_read: list[str]     Which .py/.R/.sh files were read   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                    DISCOVERY DETECTION FLOW

    ┌───────────────────────────────────────────────────┐
    │  Two detection paths for script/doc reading:      │
    └───────────────────────────────────────────────────┘

    PATH 1: inspect_file_as_text
    ─────────────────────────────
         ┌─────────────────────────────────────┐
         │  For each inspect_file_as_text call │
         └──────────────────┬──────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Path has    │   │ Path ends   │   │ Path matches│
   │ "README" or │   │ in .py/.R/  │   │ expected    │
   │"REPRODUCING"│   │ .sh/.jl?    │   │ script name?│
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                 │                 │
          ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Add to      │   │ Add to      │   │ Set         │
   │ docs_read[] │   │ scripts_    │   │read_target_ │
   │ Set read_   │   │ read[]      │   │script = true│
   │documentation│   │             │   │             │
   │ = true      │   │             │   │             │
   └─────────────┘   └─────────────┘   └─────────────┘

    PATH 2: execute_bash (file reading commands)
    ─────────────────────────────────────────────
         ┌─────────────────────────────────────┐
         │  For each execute_bash call         │
         └──────────────────┬──────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Is command a file read?│
              │  cat, head, tail, sed,  │
              │  awk, grep, less, more  │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │ YES                     │ NO
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │ Extract script  │       │ Skip            │
    │ paths from args │       │                 │
    │ (e.g., .py/.R)  │       │                 │
    └────────┬────────┘       └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ exit_code == 0? │
    └────────┬────────┘
             │ YES
             ▼
    ┌─────────────────┐
    │ Add to scripts_ │
    │ read[], set     │
    │ read_target_    │
    │ script = true   │
    └─────────────────┘
```

### Execution Phase
```
┌─────────────────────────────────────────────────────────────────┐
│ EXECUTION METRICS                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  attempted_execution: bool      Did agent try to run scripts?   │
│  execution_attempts: int        How many times?                 │
│  successful_execution: bool     Did at least one succeed?       │
│                                                                 │
│  expected_scripts: list[str]    Scripts parsed from task_prompt │
│  executed_scripts: list[str]    Scripts that ran successfully   │
│  attempted_failed_scripts: []   Scripts that failed             │
│  execution_coverage: float      Fraction of expected that ran   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                    EXECUTION DETECTION FLOW

    ┌───────────────────────────────────────────────────┐
    │  Parse task_prompt for script names               │
    │  (regex: Run '(.+\.py)' or execute (.+\.R) etc.)  │
    │  → expected_scripts[]                             │
    └─────────────────────────┬─────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────┐
    │  For each execute_bash tool call                  │
    └─────────────────────────┬─────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Does command contain         │
              │  "python", "Rscript", "bash", │
              │  "julia", or "./" ?           │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ YES                           │ NO
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ attempted_      │             │ Skip (not a     │
    │ execution = true│             │ script run)     │
    │ execution_      │             │                 │
    │ attempts++      │             │ e.g. ls, pip,   │
    │                 │             │ cat, etc.       │
    │ Extract script  │             └─────────────────┘
    │ name from cmd   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  Check tool_result      │
    │  exit_code == 0?        │
    └────────────┬────────────┘
                 │
      ┌──────────┴──────────┐
      │ YES                 │ NO
      ▼                     ▼
┌───────────────┐    ┌───────────────┐
│ Add to        │    │ Add to        │
│ executed_     │    │ attempted_    │
│ scripts[]     │    │ failed_       │
│               │    │ scripts[]     │
│ successful_   │    │               │
│ execution=true│    │               │
└───────────────┘    └───────────────┘

                    COVERAGE CALCULATION

    ┌───────────────────────────────────────────────────┐
    │  execution_coverage =                             │
    │    len(executed ∩ expected) / len(expected)       │
    │                                                   │
    │  If expected is empty: coverage = 1.0 if any      │
    │  successful execution, else 0.0                   │
    └───────────────────────────────────────────────────┘
```

### Error Recovery
```
┌─────────────────────────────────────────────────────────────────┐
│ ERROR RECOVERY METRICS                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  total_errors: int              Errors encountered              │
│  errors_recovered: int          Errors followed by success      │
│  recovery_rate: float           recovered / total               │
│  consecutive_failures: int      Max streak of failures          │
│  persistence_score: float       Combined recovery metric        │
│                                                                 │
│  error_types: dict              Breakdown by category:          │
│    - import_error               Missing dependencies            │
│    - file_not_found             Wrong paths                     │
│    - permission_denied          Access issues                   │
│    - timeout                    Command too slow                │
│    - syntax_error               Code bugs                       │
│    - other                      Uncategorized                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                    ERROR DETECTION FLOW

    ┌───────────────────────────────────────────────────┐
    │  For each execute_bash call                       │
    └─────────────────────────┬─────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │  Is this an execution?  │
                │  (python, Rscript, bash │
                │   script, ./run, etc.)  │
                └────────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ YES                         │ NO
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ Check result:   │           │ SKIP entirely   │
    │ - exit_code != 0│           │ (ls, cat, pip,  │
    │ - timed_out     │           │  etc. filtered) │
    │ - error pattern │           │                 │
    └────────┬────────┘           └─────────────────┘
             │
    ┌────────┴────────┐
    │ ERROR?          │
    └────────┬────────┘
             │
    ┌────────┴────────┐           ┌─────────────────┐
    │ YES             │           │ NO              │
    ▼                 │           ▼                 │
┌─────────────┐       │     ┌─────────────┐         │
│total_errors+│       │     │Reset streak │         │
│Classify type│       │     │to 0         │         │
│streak++     │       │     └─────────────┘         │
│             │       │                             │
│Check if NEXT│       │                             │
│EXECUTION    │       │                             │
│succeeds:    │       │                             │
│YES=recovered│       │                             │
└─────────────┘       └─────────────────────────────┘

    NOTE: Only EXECUTION attempts are tracked.
          ls, cat, pip install, etc. are filtered out.
          Recovery = exec error → exec success (ignoring non-exec commands)

                    PERSISTENCE SCORE CALCULATION

    ┌───────────────────────────────────────────────────┐
    │  total_errors == 0?                               │
    └─────────────────────────┬─────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ YES                           │ NO
              ▼                               ▼
    ┌─────────────────┐           ┌─────────────────────────┐
    │ persistence_    │           │ recovery_rate =         │
    │ score = 1.0     │           │   recovered / total     │
    │                 │           │                         │
    │ (no errors =    │           │ streak_penalty =        │
    │  full credit)   │           │   min(1.0, streak / 5)  │
    └─────────────────┘           │   (5+ failures = 1.0)   │
                                  │                         │
                                  │ persistence_score =     │
                                  │   rate * (1 - penalty   │
                                  │           * 0.5)        │
                                  │   clamped to [0, 1]     │
                                  └─────────────────────────┘

Example:
  - 5 errors, 4 recovered, max 2 consecutive failures
  - recovery_rate = 4/5 = 0.8
  - streak_penalty = min(1.0, 2/5) = 0.4
  - persistence_score = 0.8 * (1 - 0.4 * 0.5) = 0.8 * 0.8 = 0.64
```

### Anti-Patterns
```
┌─────────────────────────────────────────────────────────────────┐
│ ANTI-PATTERN DETECTION                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  read_preexisting_results: bool                                 │
│    Agent read from results/ directory before executing          │
│    This is "cheating" - reading answers instead of computing    │
│                                                                 │
│  violations: list[str]                                          │
│    Descriptions of detected anti-patterns                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                    ANTI-PATTERN DETECTION FLOW

    ┌───────────────────────────────────────────────────┐
    │  Only applies to corebench_hard mode              │
    └─────────────────────────┬─────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Track first SUCCESSFUL       │
              │  execute_bash (exit_code=0)   │
              │  that runs a script           │
              │  (first_successful_exec_turn) │
              └───────────────┬───────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────┐
    │  For each of these BEFORE first_successful_exec:  │
    │  - inspect_file_as_text                           │
    │  - query_vision_language_model                    │
    └─────────────────────────┬─────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │  Does path contain      │
                │  "results/" OR          │
                │  "/results" OR          │
                │  "result_" ?            │
                └────────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ YES                         │ NO
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ read_preexisting│           │ No violation    │
    │ _results = true │           │ (normal file    │
    │                 │           │  reading)       │
    │ Add to          │           │                 │
    │ violations[]:   │           │                 │
    │ - "Read pre-    │           │                 │
    │   existing      │           │                 │
    │   results: path"│           │                 │
    │ - "Queried pre- │           │                 │
    │   existing      │           │                 │
    │   results image"│           │                 │
    └─────────────────┘           └─────────────────┘

    NOTE: Reading results/ AFTER successful execution is fine.
          Reading results/ BEFORE any successful execution is cheating.
          Also catches vision model queries on result images.
```

## Example Scores

```
┌─────────────────────────────────────────────────────────────────┐
│ EXAMPLE: Agent runs wrong script (hard mode)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Task: Run testing.py                                           │
│  Agent: Ran training.py (failed)                                │
│                                                                 │
│  read_documentation:    true   → +0.10                          │
│  read_target_script:    true   → +0.20                          │
│  execution_coverage:    0.0    → +0.15 (attempted, partial)     │
│  successful_execution:  false  → +0.00                          │
│  error_recovery:        0.2    → +0.02                          │
│  penalty:               none   → +0.00                          │
│                                 ─────                           │
│  TOTAL:                          0.47                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ EXAMPLE: Agent succeeds (hard mode)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Task: Run analysis.py                                          │
│  Agent: Read docs, read script, ran analysis.py successfully    │
│                                                                 │
│  read_documentation:    true   → +0.10                          │
│  read_target_script:    true   → +0.20                          │
│  execution_coverage:    1.0    → +0.30                          │
│  successful_execution:  true   → +0.30                          │
│  error_recovery:        0.5    → +0.05                          │
│                                 ─────                           │
│  TOTAL:                          0.95                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```