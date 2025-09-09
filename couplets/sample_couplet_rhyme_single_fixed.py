#!/usr/bin/env python3
"""sample_couplet_rhyme_single.py

Generate single greedy completions for rhyming couplets using **HuggingFace Transformers**.

For each first line in the provided text file, the script:

1. Generates a single greedy completion asking for ONLY the second line of the couplet.
2. Extracts the last word of the completion.
3. Checks whether the generated last word rhymes with the prompt's last word
   (Datamuse API).
4. Records results in a DataFrame with columns for first line, raw/clean completions,
   last words, and rhyme success.

Uses HuggingFace Transformers for direct model loading and generation.
"""

from __future__ import annotations

from functools import lru_cache
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATAMUSE_URL = "https://api.datamuse.com/words"

THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
LAST_WORD_RE = re.compile(r"([A-Za-z']+)[^A-Za-z']*$")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _echo(msg: str) -> None:  # print to stderr & flush
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def last_word(text: str) -> str:
    """Return final alphabetic word (lower-cased)."""
    match = LAST_WORD_RE.search(text.strip())
    return match.group(1).lower() if match else ""

@lru_cache(maxsize=1000)
def fetch_rhymes(word: str) -> frozenset[str]:
    """Datamuse rhymes with caching."""
    word = word.lower()
    try:
        resp = requests.get(DATAMUSE_URL, params={"rel_rhy": word, "max": 1000}, timeout=20)
        resp.raise_for_status()
        s = {*(item["word"].lower() for item in resp.json()), word}
        return frozenset(s)
    except Exception as e:  # noqa: BLE001
        _echo(f"Datamuse query failed for '{word}'. Reason: {e}.")
        return frozenset()


def get_or_create_rhyme_group(word: str, rhyme_groups: Dict[str, Set[str]]) -> str:
    """Get the rhyme group name for a word, or create a new one if needed."""
    word = word.lower()
    if not word:
        return "<empty>"
    
    # Check if word is already in any existing rhyme group
    for group_name, group_words in rhyme_groups.items():
        if word in group_words:
            return group_name
    
    # Word not in any existing group, create new group
    rhyming_words = fetch_rhymes(word)
    # Include the word itself in its rhyme group
    rhyme_groups[word] = rhyming_words
    
    return word


def sanitize_second_line(first_line: str, raw_second: str) -> str:
    """Port of *evaluate_couplet_rhymes.sanitize_second_line*."""
    # 1. Drop common prefixes like "Second line:" etc.
    cleaned = re.sub(r"^\s*(?:the\s+)?second\s+line\s*[:\-–]?\s*", "", raw_second, flags=re.I)
    # 2. Remove /no_think or variants
    cleaned = re.sub(r"/\s*no?_?think\b.*", "", cleaned, flags=re.I)
    # 3. If first line repeated, drop it
    first_norm = first_line.strip().lower()
    if cleaned.lower().startswith(first_norm):
        cleaned = cleaned[len(first_line):].lstrip(" \t\n,/:-—")
    # 4. If multi-line output, pick segment not identical to first line
    parts = re.split(r"[\n/]+", cleaned)
    if len(parts) > 1:
        for part in parts:
            p = part.strip(" \t-—")
            if p and p.lower() != first_norm:
                cleaned = p
                break
    # 5. Collapse spaces
    return " ".join(cleaned.split()).strip()


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

def generate_second_line(
    first_line: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> tuple[str, str]:
    """Generate completion using HuggingFace model & return (raw, cleaned) completion."""
    prompt = f"/no_think Write only the next line of this rhyming couplet: {first_line.strip()}"
    
    chattified_prompt = chattify([prompt, ""], tokenizer)
    # Tokenize input
    inputs = tokenizer(chattified_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Return both raw and cleaned versions
    cleaned_text = sanitize_second_line(first_line, THINK_RE.sub("", raw_text).strip())
    return raw_text, cleaned_text


# ---------------------------------------------------------------------------
# Sampling routine
# ---------------------------------------------------------------------------

def generate_single_completion(
    first_line: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    rhyme_groups: Dict[str, Set[str]],
) -> dict:
    """Generate a single greedy completion and return result data."""
    last1 = last_word(first_line)
    rhyme_group = get_or_create_rhyme_group(last1, rhyme_groups)
    
    try:
        raw_second, cleaned_second = generate_second_line(first_line, model, tokenizer, max_tokens)
    except Exception as exc:  # noqa: BLE001
        _echo(f"Generation failed for '{first_line}': {exc}")
        raw_second = ""
        cleaned_second = ""
    
    last2 = last_word(cleaned_second)
    rhyme_success = bool(last1 and last2 and last2 in fetch_rhymes(last1) and last1 != last2)
    
    return {
        "first_line": first_line,
        "first_last_word": last1,
        "second_last_word": last2,
        "rhyme_group": rhyme_group,
        "rhyme_success": rhyme_success,
        "clean_completion": cleaned_second,
        "raw_completion": raw_second,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Generate greedy rhyming couplets via HuggingFace Transformers.")
    parser.add_argument("--input", type=Path, required=True, help="Text file with first lines (one per line).")
    parser.add_argument("--output-dir", type=Path, default="results/couplets-fixed", help="Output directory to save CSV results.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens per completion.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model data type.")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load model and tokenizer
    # ------------------------------------------------------------------
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    models_and_transcoders = {
        'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
        'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
        'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
        'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
        'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0",
        'Qwen/Qwen3-32B':"mwhanna/qwen3-14b-transcoders-lowl0"
    }
    
    for model_name in models_and_transcoders.keys():
        _echo(f"Loading model {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ------------------------------------------------------------------
        # Read first lines
        # ------------------------------------------------------------------
        first_lines: List[str] = [ln.strip() for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not first_lines:
            sys.exit("Input file is empty or only blank lines.")

        rhyme_groups: Dict[str, Set[str]] = {}
        results: List[dict] = []

        interrupted = False
        try:
            for idx, fl in enumerate(first_lines, start=1):
                _echo(f"[{idx}/{len(first_lines)}] Generating greedy completion …")
                res = generate_single_completion(
                    fl,
                    model,
                    tokenizer,
                    args.max_new_tokens,
                    rhyme_groups,
                )
                results.append(res)
        except KeyboardInterrupt:
            interrupted = True
            _echo("Interrupted by user. Saving partial results …")

        # ------------------------------------------------------------------
        # Create DataFrame and save results
        # ------------------------------------------------------------------
        df = pd.DataFrame(results)

        # Generate output filename based on model
        model_slug = model_name.split("/")[-1]
        output_path = args.output_dir / f"{model_slug}.csv"

        # Save to CSV
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        if interrupted:
            _echo(f"Partial results ({len(results)} prompts) written to {output_path}.")
        else:
            _echo(f"Done ✓  Wrote {len(results)} entries to {output_path}.")


if __name__ == "__main__":  # pragma: no cover
    main() 