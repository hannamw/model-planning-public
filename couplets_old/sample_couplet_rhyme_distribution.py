#!/usr/bin/env python3
"""sample_couplet_rhyme_distribution.py

Given a text file containing the first line of rhyming couplets (one per line),
this script:

1. Prompts a local ReplacementModel (from *circuit_tracer*) to complete each
   couplet, generating *N* independent samples (default 200).
2. Extracts the last word of each generated line and computes its frequency
   distribution.
3. Uses the Datamuse API to decide whether the generated last word rhymes with
   the last word of the prompt line, tallying rhyme success across samples.
4. Writes, for each input line, a JSON object containing the first line, the
   distribution over last words, the rhyme-success count, and the total number
   of samples.

The final output is a JSON list written to *--output* (default
``results/couplet_sampling.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set

import requests
import torch

# circuit_tracer utilities ----------------------------------------------------
try:
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.utils.intervention_utils import chattify
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "This script requires the *circuit_tracer* package.\n"
        f"Import error: {exc}"
    )

# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------
DATAMUSE_URL = "https://api.datamuse.com/words"
LAST_WORD_RE = re.compile(r"([A-Za-z']+)[^A-Za-z']*$")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def last_word(text: str) -> str:
    """Return the final word (letters and apostrophes) in *text*, lower-cased."""
    match = LAST_WORD_RE.search(text.strip())
    return match.group(1).lower() if match else ""


def fetch_rhymes(word: str, cache: Dict[str, Set[str]]) -> Set[str]:
    """Return a set of words that rhyme with *word* (cached via *cache*)."""
    word = word.lower()
    if word not in cache:
        try:
            resp = requests.get(DATAMUSE_URL, params={"rel_rhy": word, "max": 1000}, timeout=20)
            resp.raise_for_status()
            cache[word] = {item["word"].lower() for item in resp.json()}
        except Exception:  # noqa: BLE001 (network issues)
            cache[word] = set()
    return cache[word]


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def sample_completions(
    first_line: str,
    model: "ReplacementModel",
    num_samples: int,
    max_tokens: int,
    rhyme_cache: Dict[str, Set[str]],
) -> dict:
    """Generate *num_samples* completions and compute rhyme statistics."""

    prompt_user = f"/no_think A rhyming couplet:\n {first_line.strip()}"
    # Chat format expects [user, assistant] messages, with the assistant start empty.
    chat_seq = chattify([prompt_user, ""], model.tokenizer)

    last_word_prompt = last_word(first_line)
    distribution: Counter[str] = Counter()
    rhyme_hits = 0

    # Ensure deterministic variety by advancing RNG per iteration (seed optional).
    for i in range(num_samples):
        torch.manual_seed(i)  # gives reproducibility with some variety
        completion: str = model.generate(
            chat_seq,
            use_past_kv_cache=False,
            do_sample=True,
            max_new_tokens=max_tokens,
        )
        lw = last_word(completion)
        distribution[lw] += 1
        if lw and lw in fetch_rhymes(last_word_prompt, rhyme_cache):
            rhyme_hits += 1

    return {
        "first_line": first_line,
        "last_word": last_word_prompt,
        "distribution": dict(distribution),
        "rhyme_successes": rhyme_hits,
        "num_samples": num_samples,
        "rhyme_ratio": rhyme_hits / num_samples if num_samples else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Sample rhyming couplet completions.")
    parser.add_argument("--input", type=Path, required=True, help="Text file with first lines of couplets.")
    parser.add_argument("--output", type=Path, default=Path("results/couplet_sampling.json"), help="Where to write JSON output.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-14B", help="HF model identifier.")
    parser.add_argument("--model-config", required=True, help="ReplacementModel config YAML path.")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples per first line.")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Maximum tokens to generate per sample.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type used when loading the model.",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"Loading model {args.model_name} …", file=sys.stderr, flush=True)
    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.model_config,
        transcoders_offload="disk",
        dtype=dtype_map[args.dtype],
    )

    # ------------------------------------------------------------------
    # Read input lines
    # ------------------------------------------------------------------
    first_lines: List[str] = [ln.strip() for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not first_lines:
        sys.exit("Input file is empty or contains only blank lines.")

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------
    rhyme_cache: Dict[str, Set[str]] = {}
    results: List[dict] = []
    for idx, line in enumerate(first_lines, start=1):
        print(f"[{idx}/{len(first_lines)}] Sampling completions …", file=sys.stderr, flush=True)
        res = sample_completions(
            line,
            model,
            args.num_samples,
            args.max_new_tokens,
            rhyme_cache,
        )
        results.append(res)

    # ------------------------------------------------------------------
    # Write output JSON
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done ✓  Wrote results for {len(first_lines)} lines to {args.output}.", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main() 