# Capsule File Layout

Base directory: `scenarios/corebench/capsules/`

Capsule directory pattern:
```
├── capsule-<ID>/
│   ├── code/
│   │   ├── README.md
│   │   └── ... (.py, .ipynb, .xlsx, .R, .sh)
│   ├── data/
│   │   ├── ...(.csv, .txt, .json, .zip, .png files ...)
│   │   └── LICENSE
│   ├── environment/
│   │   └── Dockerfile
│   ├── metadata/
│   │   └── metadata.yml
│   ├── downloads/
│   ├── results/
│   │   └── ... (pre-populated output files)
│   ├── REPRODUCING.md
│   └── README.md
```

## CoreBench Hardmode

Hardmode removes: 
- results/ directory
- REPRODUCING.md at root level
- environment/ directory
- environment/Dockerfile
- code/run.sh and code/run scripts

Workspace directory (hardmode): `workspace/environment/`
```
workspace/
├── environment/
│   └── code/
│   │   ├── LICENSE
│   │   ├── (README.md)
│   │   └── (code files...)
│   ├── data/
│   │   ├── LICENSE
│   │   └── (data files...)
│   ├── metadata/
│   ├── downloads/
│   │   └── (empty)
│   └── (README.md)
```

## Results Reporting and Analysis

After a benchmark run, CoreBench reports results in two places:

Per-task traces + a run summary JSON

```text
logs/traces/<run_dir>_corebench_hard/
├── corebench_hard_20260124_005637_capsule-9670283.jsonl
└── run_summary_6697b77e.json
```

Example (`run_summary_6697b77e.json`):

```json
{
  "run_id": "6697b77e",
  "domain": "corebench_hard",
  "completed_at": "2026-01-24T00:59:24.083267+00:00",
  "num_tasks": 1,
  "task_ids": ["capsule-9670283"],
  "model_used": "openai/gpt-oss:120b",
  "aggregate_metrics": {
    "num_tasks": 1,
    "num_successful": 1,
    "pass_rate": 1.0,
    "mean_accuracy": 1.0,
    "mean_written_accuracy": 0.0,
    "mean_vision_accuracy": 1.0,
    "mean_adherence": 0.22,
    "mean_methodology_score": 0.4,
    "doc_read_rate": 1.0,
    "execution_attempt_rate": 0.0,
    "successful_execution_rate": 0.0,
    "mean_error_recovery_rate": 1.0,
    "mean_steps": 4.0,
    "mean_tool_calls": 3.0,
    "mean_time": 135.71241307258606,
    "task_results": {
      "capsule-9670283": {
        "success": true,
        "accuracy": 1.0,
        "adherence": 0.22,
        "methodology_score": 0.4
      }
    }
  }
}
```

### Log summary (what you see after run)

At the end of a benchmark session, the evaluator prints a single consolidated summary block.

```text
⭐ CoreBench Benchmark Results ⭐
Domain: corebench_hard
Tasks: 1/1 passed (100.0%)

📊 Accuracy Metrics:
  Accuracy: 100.0%
  Written: 0.0%, Vision: 100.0%

🔧 Methodology Metrics (Deterministic):
  Methodology Score: 0.40/1.0
  Doc Read Rate: 100.0%
  Execution Attempt Rate: 0.0%
  Successful Execution Rate: 0.0%
  Error Recovery Rate: 100.0%

📋 Task Adherence (LLM Judge): 0.22/1.0

⚡ Efficiency:
  Avg Steps: 4.0
  Avg Tool Calls: 3.0
  Total Time: 135.7s

📋 Task Results:
  capsule-9670283: ✅ (acc=100.0%, process=0.40)
```

### Task Prompt Script References (`core_test.json`)

The CoreBench task list (`scenarios/corebench/core_test.json`) contains 66 task entries total: 45 "base" tasks and 21 "extension" tasks listed in `scenarios/corebench/capsule_extension.json`.

Within the 45 base tasks:
- 41/45 `task_prompt`s explicitly name a script/notebook filename to run (e.g. `*.py`, `*.R`, `*.Rmd`, `*.ipynb`, `*.sh`).
- The 4/45 that do not name a specific script/notebook filename are:
  - `capsule-8536428` — "Run the python files in the folder and its subdirectories. If there are multiple python files in the same directory or subdirectory, only run the train files, not the test files."
  - `capsule-3593259` — "Run 'physalia_automators.reports' as a python module with /results as the output directory."
  - `capsule-2345790` — "Set up the following subfolders in the ../results directory: intermediates, figures, stats_figures_markdowns. Run all the .Rmd files using Rscript and render them as html. Store the output files in ../results/stats_figures_markdowns."
  - `capsule-5136217` — "Make the following subfolders in the ../results directory: tables, figures, for_publication/tables, for_publication/figures. Run all the .R scripts in the ../code folder using Rscript with 'source' and set echo to 'TRUE'."

## README Schemas

Note: Schema lists are not mutually exclusive (capsules with multiple `readme*` files appear in multiple lists).

## README in Capsule Code Dir

`workspace/environment/code/README.md` (16)

- `capsule-0504157`
- `capsule-0851068`
- `capsule-1624349`
- `capsule-2804717`
- `capsule-2816027`
- `capsule-3301293`
- `capsule-3449234`
- `capsule-3593259`
- `capsule-3821950`
- `capsule-5507257`
- `capsule-6003668`
- `capsule-8234136`
- `capsule-8536428`
- `capsule-9052293`
- `capsule-9641396`
- `capsule-9660931`

### Code Dir Edge Cases (3)

- `capsule-2414499`
  - `capsule-ID/code/readme.md` (lowercase)
- `capsule-4299879`
  - `capsule-ID/code/readme.pdf` (lowercase, PDF format)
- `capsule-6049678`
  - `capsule-ID/code/README.txt`
- `README.Rmd`

---

## README in Capsule Root Dir

`workspace/environment/README.md` (7)

- `capsule-1394704`
- `capsule-3262218`
- `capsule-3418007`
- `capsule-8807709`
- `capsule-9054015`
- `capsule-9670283`
- `capsule-9832712`

### Root Dir Edge Cases (1)

- `capsule-1724988`
  - `workspace/environment/readme.txt` (lowercase, TXT format)

---

## README in Environment Dir

`workspace/environment/environment/README.md` (1)

- `capsule-9052293` (also has `code/README.md`)

---

## Edge Cases

### Capsules With Multiple `readme*` Files (2)

- `capsule-3593259`
  - `code/README.md`
  - `code/experiments_part_2/README.md`
  - note that this capsule doesn't have a root README
- `capsule-9052293`
  - `code/README.md`
  - `environment/README.md`
    - This second README will be deleted during codebench hardmode, however the one in the code directory will remain.

---

## REPRODUCING.md Pattern

All 27 capsules contain a `REPRODUCING.md` file in the capsule root directory.

`workspace/environment/REPRODUCING.md` (27)

- `capsule-0504157`
- `capsule-0851068`
- `capsule-1394704`
- `capsule-1624349`
- `capsule-1724988`
- `capsule-2414499`
- `capsule-2804717`
- `capsule-2816027`
- `capsule-3262218`
- `capsule-3301293`
- `capsule-3418007`
- `capsule-3449234`
- `capsule-3593259`
- `capsule-3821950`
- `capsule-4299879`
- `capsule-5507257`
- `capsule-6003668`
- `capsule-6049678`
- `capsule-8234136`
- `capsule-8536428`
- `capsule-8807709`
- `capsule-9052293`
- `capsule-9054015`
- `capsule-9641396`
- `capsule-9660931`
- `capsule-9670283`
- `capsule-9832712`

---

## Summary

| Location                | Count | Format       |
| ----------------------- | ----- | ------------ |
| `code/README.md`        | 16    | Markdown     |
| `code/` (edge cases)    | 3     | md, pdf, txt |
| Root `README.md`        | 7     | Markdown     |
| Root (edge cases)       | 1     | txt          |
| `environment/README.md` | 1     | Markdown     |
| `REPRODUCING.md` (root) | 27    | Markdown     |

---

### Capsules that needs tensorflow
- capsule-5507257
- capsule-3449234
- capsule-6003668
- capsule-9660931
- capsule-8536428 contains TF/Keras code (under Neural_Network_based_Methods), but its traditional-ML scripts don’t need TF.
