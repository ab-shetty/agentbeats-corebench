#!/usr/bin/env python3
"""
Prettify a JSONL trace or an entire trace folder and drop noisy fields.

1) Parses each JSON object safely.
2) Removes selected keys:
   - run_id (always)
   - hint (when null)
   - timed_out (when false)
   - exit_code (when null or 0)
3) Structures the "summary" string into a dict (exit_code/stdout/stderr)
   when it matches the common trace format.
4) Writes a pretty-printed JSONL output.

Usage:
    python trace_prettify.py path/to/file.jsonl
    python trace_prettify.py path/to/file.jsonl -o path/to/file.pretty.cleaned.jsonl
    python trace_prettify.py path/to/file.jsonl --in-place
    # Batch mode (skip already-pretty, skip existing outputs, require evaluation):
    python trace_prettify.py --folder logs/traces/20260129_9cf36493_corebench_hard --pattern "*.jsonl" --require-evaluation
    # Recursive batch:
    python trace_prettify.py --folder logs/traces --recursive --pattern "*capsule-*.jsonl"
    # Overwrite outputs:
    python trace_prettify.py --folder logs/traces --pattern "*.jsonl" --no-skip-existing

Remove old files after batch processing
    find logs/traces/20260129_9cf36493_corebench_hard -type f ! -name '*pretty*' -print
    find logs/traces/20260129_9cf36493_corebench_hard -type f ! -name '*pretty*' -delete

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import tempfile
import os
import re
import json.decoder


def _structure_summary(summary: Any) -> Any:
    """
    If summary matches the standard tool result blob, turn it into an easy to read dict.
    For stdout/stderr, split into lines for readability. Otherwise, if it's a multiline
    string, return a list of lines.
    """
    if not isinstance(summary, str):
        return summary

    # Match patterns like:
    # Exit Code: 1\nStdout:\n... \n\nStderr:\n...
    m = re.match(
        r"^Exit Code:\s*(?P<code>-?\d+)\nStdout:\n(?P<stdout>.*?)(?:\nStderr:\n(?P<stderr>.*))?$",
        summary,
        flags=re.DOTALL,
    )
    if m:
        stdout_txt = m.group("stdout").rstrip()
        stderr_txt = (m.group("stderr") or "").rstrip()
        structured = {
            "exit_code": int(m.group("code")),
            "stdout": stdout_txt.splitlines() if "\n" in stdout_txt else stdout_txt,
            "stderr": stderr_txt.splitlines() if "\n" in stderr_txt else stderr_txt,
        }
        return structured

    # Otherwise, if it contains newlines, keep as list of lines for readability
    if "\n" in summary:
        return summary.splitlines()

    return summary


def clean_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Remove noisy fields according to the requested rules."""
    event.pop("run_id", None)

    if event.get("hint") is None:
        event.pop("hint", None)

    if event.get("timed_out") is False:
        event.pop("timed_out", None)

    if event.get("exit_code") in (None, 0):
        event.pop("exit_code", None)

    if "summary" in event:
        event["summary"] = _structure_summary(event["summary"])

    # Normalize plan content into a list of lines for readability
    if event.get("type") == "plan" and isinstance(event.get("content"), str):
        event["content"] = event["content"].splitlines()

    # For llm_judge_input, break tool_interactions into lines
    if event.get("type") == "llm_judge_input" and isinstance(event.get("tool_interactions"), str):
        event["tool_interactions"] = event["tool_interactions"].splitlines()

    # If summary has stdout/stderr strings, split them to lines as well
    if isinstance(event.get("summary"), dict):
        for key in ("stdout", "stderr"):
            val = event["summary"].get(key)
            if isinstance(val, str) and "\n" in val:
                event["summary"][key] = val.splitlines()

    return event


def _load_events(in_path: Path) -> List[Dict[str, Any]]:
    """
    Load one or more JSON objects from a file.
    Supports:
      - standard JSONL (one object per line)
      - pretty-printed stream (objects separated by whitespace/newlines)
      - a single JSON array of objects
    """
    text = in_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    events: List[Dict[str, Any]] = []
    length = len(text)

    while True:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(text, idx)
        idx = end
        if isinstance(obj, list):
            events.extend(obj)
        else:
            events.append(obj)

    if not events:
        raise ValueError(f"No JSON objects found in {in_path}")
    return events


def prettify_jsonl(
    in_path: Path,
    out_path: Path,
    *,
    preload_events: List[Dict[str, Any]] | None = None,
) -> None:
    """Read JSONL, clean fields, and write pretty-printed JSONL safely."""
    # If writing in place, use a temp file to avoid truncating the input.
    writing_in_place = in_path.resolve() == out_path.resolve()
    if writing_in_place:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="trace_prettify_", suffix=".jsonl", dir=str(in_path.parent)
        )
        os.close(tmp_fd)
        out_path = Path(tmp_path)

    if preload_events is None:
        try:
            events = _load_events(in_path)
        except Exception as exc:
            sys.exit(f"Failed to parse JSON in {in_path}: {exc}")
    else:
        events = preload_events

    with out_path.open("w", encoding="utf-8") as fout:
        for event in events:
            event = clean_event(event)
            fout.write(json.dumps(event, ensure_ascii=False, indent=2))
            fout.write("\n")

    if writing_in_place:
        out_path.replace(in_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prettify a JSONL trace and remove selected fields."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a JSONL file to prettify (omit when using --folder).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: <input>.pretty.cleaned.jsonl)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new one.",
    )
    parser.add_argument(
        "--folder",
        help="Batch mode: process all matching files in this folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when using --folder.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern for --folder mode (default: *.jsonl).",
    )
    parser.add_argument(
        "--process-pretty",
        action="store_true",
        help="Also process files whose name already contains .pretty or .cleaned.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing output files instead of skipping.",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--require-evaluation",
        action="store_true",
        help="Skip files that do not contain an event with type == 'evaluation'.",
    )

    args = parser.parse_args()
    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            sys.exit(f"Folder not found: {folder}")

        glob_pattern = "**/" + args.pattern if args.recursive else args.pattern
        candidates = list(folder.glob(glob_pattern))
        processed = 0
        skipped = 0

        for in_path in candidates:
            if in_path.is_dir():
                continue
            name_lower = in_path.name.lower()
            if not args.process_pretty and (".pretty" in name_lower or ".cleaned" in name_lower):
                skipped += 1
                continue

            if args.in_place:
                out_path = in_path
            elif args.output:
                out_path = Path(args.output)
            else:
                out_path = in_path.parent / f"{in_path.name}.pretty.cleaned.jsonl"

            if args.skip_existing and out_path.exists():
                skipped += 1
                continue

            # Optionally require the trace to have an evaluation event
            events = None
            if args.require_evaluation:
                try:
                    events = _load_events(in_path)
                except Exception:
                    skipped += 1
                    continue
                if not any(e.get("type") == "evaluation" for e in events):
                    skipped += 1
                    continue

            prettify_jsonl(
                in_path,
                out_path,
                preload_events=events,
            )
            processed += 1

        print(f"Batch complete. Processed: {processed}, skipped: {skipped}")
        return

    # single-file mode
    if not args.input:
        sys.exit("No input file provided (and --folder not set).")

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"Input file not found: {in_path}")

    if args.in_place:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.parent / f"{in_path.name}.pretty.cleaned.jsonl"

    prettify_jsonl(in_path, out_path)
    print(f"Wrote cleaned file to {out_path}")


if __name__ == "__main__":
    main()
