#!/usr/bin/env python3
"""Convert a markdown table into a CSV file.

Usage
-----
python scripts/md_to_csv.py input.md output.csv

If *output.csv* is omitted, the CSV is written next to the input file with a .csv extension.
"""

import argparse
import csv
import pathlib
import re
import sys
from typing import List, Tuple


def _parse_table(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Extracts header and rows from a markdown table.

    Assumes that the table follows the GitHub-flavoured markdown syntax:

    | Header1 | Header2 |
    |---------|---------|
    | cell11  | cell12  |
    | cell21  | cell22  |

    The function is tolerant of leading/trailing whitespace.
    """
    header: List[str] = []
    rows: List[List[str]] = []
    capturing = False

    pipe_pat = re.compile(r"^\|.*\|$")

    for raw_line in lines:
        line = raw_line.strip()
        if not pipe_pat.match(line):
            # Skip non-table lines
            continue

        # Split the row into cells, trimming each cell
        cells = [c.strip() for c in line.strip("|").split("|")]

        if not capturing:
            header = cells
            capturing = True
            continue  # Next line should be the delimiter row

        # Skip the delimiter row consisting solely of dashes/colons
        if all(re.fullmatch(r"[:\-]+", c) for c in cells):
            continue

        rows.append(cells)

    if not header:
        raise ValueError("No markdown table found in the provided file.")
    return header, rows


def markdown_to_csv(md_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    """Read *md_path*, parse the first markdown table, and write it to *csv_path*."""
    with md_path.open(encoding="utf-8") as f:
        header, rows = _parse_table(f.readlines())

    # Ensure all rows have the same number of columns as the header
    sanitized_rows = []
    for idx, row in enumerate(rows, start=1):
        if len(row) != len(header):
            raise ValueError(
                f"Row {idx} has {len(row)} columns; expected {len(header)} columns."
            )
        sanitized_rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sanitized_rows)

    # Report the written file path; fall back gracefully if relative conversion fails
    try:
        displayed_path = csv_path.resolve().relative_to(pathlib.Path.cwd())
    except ValueError:
        displayed_path = csv_path

    print(f"Wrote {len(rows)} rows to {displayed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a markdown table to CSV.")
    parser.add_argument("input_md", type=pathlib.Path, help="Path to the markdown file")
    parser.add_argument(
        "output_csv",
        type=pathlib.Path,
        nargs="?",
        help="Destination CSV file (default: replace .md with .csv)",
    )
    args = parser.parse_args()

    input_md: pathlib.Path = args.input_md
    output_csv: pathlib.Path = (
        args.output_csv if args.output_csv else input_md.with_suffix(".csv")
    )

    if not input_md.is_file():
        print(f"Error: {input_md} does not exist or is not a file.", file=sys.stderr)
        sys.exit(1)

    try:
        markdown_to_csv(input_md, output_csv)
    except Exception as exc:
        print(f"Failed to convert markdown to CSV: {exc}", file=sys.stderr)
        sys.exit(1)