"""
Compute the average time taken per task across one or more CSV files.
Each CSV must have columns: started_at, finished_at
Timestamp format example: 2025-09-30T02:58:16.303002+00:00 (ISO 8601 with tz)

Usage:
  python avg_task_time.py file1.csv file2.csv --unit seconds --per-file
"""

from __future__ import annotations
import argparse
import csv
import math
import sys
from datetime import datetime
from typing import List, Tuple

def parse_iso(ts: str) -> datetime:
    """
    Parse an ISO 8601 timestamp with timezone (e.g., 2025-09-30T02:58:16.303002+00:00).
    Relies on Python 3.11+'s datetime.fromisoformat handling of offsets.
    """
    # Trim whitespace just in case
    ts = ts.strip()
    return datetime.fromisoformat(ts)

def durations_from_csv(path: str,
                       start_field: str = "started_at",
                       end_field: str = "finished_at") -> Tuple[List[float], int, int]:
    """
    Returns:
      (durations_in_seconds, rows_ok, rows_skipped)
    Rows are skipped if fields are missing, parsing fails, or duration is negative.
    """
    durations: List[float] = []
    rows_ok = 0
    rows_skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate header presence
        if start_field not in reader.fieldnames or end_field not in reader.fieldnames:
            raise ValueError(
                f"{path}: missing required columns '{start_field}' and/or '{end_field}'. "
                f"Found: {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=2):  # start=2 accounts for header line as row 1
            s = row.get(start_field, "")
            e = row.get(end_field, "")
            if not s or not e:
                rows_skipped += 1
                continue
            try:
                ds = parse_iso(s)
                de = parse_iso(e)
                delta = (de - ds).total_seconds()
                if math.isfinite(delta) and delta >= 0:
                    durations.append(delta)
                    rows_ok += 1
                else:
                    rows_skipped += 1
            except Exception:
                rows_skipped += 1

    return durations, rows_ok, rows_skipped

def convert_unit(seconds: float, unit: str) -> float:
    if unit == "seconds":
        return seconds
    if unit == "milliseconds":
        return seconds * 1000.0
    if unit == "minutes":
        return seconds / 60.0
    if unit == "hours":
        return seconds / 3600.0
    raise ValueError(f"Unknown unit {unit}")

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Compute average task time from CSVs.")
    p.add_argument("files", nargs="+", help="CSV files to process")
    p.add_argument("--unit",
                   choices=["seconds", "milliseconds", "minutes", "hours"],
                   default="seconds",
                   help="Output unit (default: seconds)")
    p.add_argument("--per-file", action="store_true",
                   help="Also print per-file averages")
    p.add_argument("--start-field", default="started_at",
                   help="Column name for task start (default: started_at)")
    p.add_argument("--end-field", default="finished_at",
                   help="Column name for task end (default: finished_at)")
    args = p.parse_args(argv)

    all_durations_sec: List[float] = []
    overall_ok = 0
    overall_skip = 0

    for path in args.files:
        try:
            durs, ok, skip = durations_from_csv(
                path, start_field=args.start_field, end_field=args.end_field
            )
        except Exception as ex:
            print(f"[ERROR] {path}: {ex}", file=sys.stderr)
            continue

        all_durations_sec.extend(durs)
        overall_ok += ok
        overall_skip += skip

        if args.per_file:
            avg = (sum(durs) / len(durs)) if durs else float("nan")
            avg_conv = convert_unit(avg, args.unit) if durs else float("nan")
            unit_label = args.unit
            print(f"{path}:")
            print(f"  rows_ok={ok}, rows_skipped={skip}")
            if durs:
                print(f"  avg_time={avg_conv:.6f} {unit_label} "
                      f"(from {len(durs)} tasks)")
            else:
                print("  avg_time=NaN (no valid tasks found)")

    # Overall
    unit_label = args.unit
    if all_durations_sec:
        overall_avg_sec = sum(all_durations_sec) / len(all_durations_sec)
        overall_avg = convert_unit(overall_avg_sec, args.unit)
        print("\nOverall:")
        print(f"  rows_ok={overall_ok}, rows_skipped={overall_skip}")
        print(f"  tasks_count={len(all_durations_sec)}")
        print(f"  avg_time={overall_avg:.6f} {unit_label}")
    else:
        print("No valid tasks found across provided files.", file=sys.stderr)
        return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
