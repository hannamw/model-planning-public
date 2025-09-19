#!/usr/bin/env python3
#%%
"""
Parse the Qwen3-14B couplet log and load it into a pandas DataFrame.

DataFrame columns
-----------------
1. first_sentence         – the full “FIRST LINE:” text
2. second_sentence        – the full candidate line (001:, 002:, …)
3. first_last_word        – last word (lower-cased, punctuation–stripped) of the first sentence
4. second_last_word       – last word (lower-cased, punctuation–stripped) of the second sentence
"""
from pathlib import Path
import re
import pandas as pd

# ----------------------------------------------------------------------
# 1.  Point to the file you want to parse
# ----------------------------------------------------------------------
LOG_FILE = Path("Qwen3-14B_completions.log")

# ----------------------------------------------------------------------
# 2.  Regular expressions for the two kinds of lines we care about
# ----------------------------------------------------------------------
FIRST_RE  = re.compile(r"^FIRST LINE:\s*(.*\S)\s*$")
SECOND_RE = re.compile(r"^\s*\d{3}:\s*(.*\S)\s*$")   # handles leading spaces

# ----------------------------------------------------------------------
# 3.  Helpers
# ----------------------------------------------------------------------
def last_word(sentence: str) -> str:
    """Return the last word, lower-cased, stripped of punctuation."""
    # Keep alphanumerics and apostrophes inside words, drop everything else
    cleaned = re.sub(r"[^\w']+", " ", sentence).strip()
    return cleaned.split()[-1].lower() if cleaned else ""

# ----------------------------------------------------------------------
# 4.  Main parse loop
# ----------------------------------------------------------------------
rows = []
current_first = None

with LOG_FILE.open(encoding="utf-8") as fh:
    for raw in fh:
        raw = raw.rstrip("\n")

        # a) FIRST LINE …
        m_first = FIRST_RE.match(raw)
        if m_first:
            current_first = m_first.group(1).strip()
            continue

        # b) numbered second line
        m_second = SECOND_RE.match(raw)
        if m_second and current_first:
            second = m_second.group(1).strip()

            rows.append(
                {
                    "first_sentence":  current_first,
                    "second_sentence": second,
                    "first_last_word":  last_word(current_first),
                    "second_last_word": last_word(second),
                }
            )

# ----------------------------------------------------------------------
# 5.  Build the DataFrame
# ----------------------------------------------------------------------
df = pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 6.  Example usage
# ----------------------------------------------------------------------
print(df.head())            # quick sanity check
df.to_csv("Qwen3-14B.csv", index=False)   # optional persistence
# %%
