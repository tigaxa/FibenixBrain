#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert price TXT (CSV) to JSON or JSONL without loading all data into memory."
    )
    parser.add_argument("input", type=Path, help="Path to input TXT file (CSV format).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON/JSONL path. Defaults to <input>.json or <input>.jsonl.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSON Lines (one JSON object per line) instead of a JSON array.",
    )
    parser.add_argument(
        "--no-cast",
        action="store_true",
        help="Keep all values as strings (no float/int casting).",
    )
    return parser.parse_args()


def cast_row(row: dict, no_cast: bool) -> dict:
    if no_cast:
        return row
    out = dict(row)
    for key in ("OPEN", "HIGH", "LOW", "CLOSE"):
        if key in out and out[key] != "":
            try:
                out[key] = float(out[key])
            except ValueError:
                pass
    if "VOL" in out and out["VOL"] != "":
        try:
            out["VOL"] = int(out["VOL"])
        except ValueError:
            pass
    return out


def convert(input_path: Path, output_path: Path, jsonl: bool, no_cast: bool) -> None:
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as f_in:
        reader = csv.DictReader(f_in)
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            if jsonl:
                for row in reader:
                    obj = cast_row(row, no_cast)
                    f_out.write(json.dumps(obj, ensure_ascii=False))
                    f_out.write("\n")
            else:
                first = True
                f_out.write("[\n")
                for row in reader:
                    obj = cast_row(row, no_cast)
                    if not first:
                        f_out.write(",\n")
                    f_out.write(json.dumps(obj, ensure_ascii=False))
                    first = False
                f_out.write("\n]\n")


def main() -> None:
    args = parse_args()
    input_path = args.input
    if args.output is None:
        suffix = ".jsonl" if args.jsonl else ".json"
        output_path = input_path.with_suffix(suffix)
    else:
        output_path = args.output
    convert(input_path, output_path, args.jsonl, args.no_cast)


if __name__ == "__main__":
    main()
