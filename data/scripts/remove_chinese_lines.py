#!/usr/bin/env python3
"""remove_chinese_lines.py

Delete any line that contains at least one Chinese (Han) character from
text files. Operates *in-place* by default (creates a ``.bak`` backup).  If
no files are provided, reads from STDIN and writes filtered text to STDOUT.

Chinese-character detection:
  * Basic CJK Unified Ideographs (4E00–9FFF)
  * Extension-A       (3400–4DBF)
  * Compatibility     (F900–FAFF)
  * CJK punctuation   (3000–303F)
  * Full-width ASCII  (FF01–FF60)
  * Supplementary Ideographs (20000–2EBEF)

Requires Python 3.7+.  No third-party dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from typing import Iterable, List

# Regex covering the ranges above.  Python's re supports \Uxxxxxxxx for >\uFFFF
HAN_RE = re.compile(
    r"[\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF01-\uFF60\U00020000-\U0002EBEF]"
)

###############################################################################
# Helpers
###############################################################################

def contains_han(text: str) -> bool:
    """Return True if *text* contains any Han (Chinese) characters."""
    return bool(HAN_RE.search(text))


def filter_lines(lines: Iterable[str]) -> Iterable[str]:
    """Yield only lines that **do not** contain Han characters."""
    for line in lines:
        if not contains_han(line):
            yield line

###############################################################################
# Main CLI
###############################################################################

def process_file(path: Path, backup_suffix: str = ".bak") -> None:
    """Filter *path* in-place (backup to ``path + backup_suffix``)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with path.open("r", encoding="utf-8", errors="replace") as f_in, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        for line in filter_lines(f_in):
            f_out.write(line)

    backup_path = path.with_suffix(path.suffix + backup_suffix)
    shutil.move(path, backup_path)
    shutil.move(tmp_path, path)
    print(f"Filtered {path}  (backup saved to {backup_path})")


DEFAULT_FILE = Path("couplet_first_lines.txt")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete lines containing Chinese characters from text files. If no FILE is given, "
            f"defaults to {DEFAULT_FILE}. A .bak backup is kept next to each file."
        )
    )
    parser.add_argument("files", metavar="FILE", nargs="*", type=Path, help="File(s) to clean.")
    args = parser.parse_args()

    targets: List[Path] = [DEFAULT_FILE] if not args.files else list(args.files)

    for path in targets:
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        process_file(path)  # backup suffix defaults to .bak


if __name__ == "__main__":
    main() 