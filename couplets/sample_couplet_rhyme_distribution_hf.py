#!/usr/bin/env python3
"""sample_couplet_rhyme_distribution_hf.py

Sample multiple completions for rhyming couplets using **HuggingFace Transformers**.

For each first line in the provided text file, the script:

1. Generates completions asking for ONLY the second line of the couplet,
   sampling *N* times (default 200).
2. Extracts the last word of every completion and builds its frequency
   distribution.
3. Checks whether each generated last word rhymes with the prompt's last word
   (Datamuse API).
4. Records per-prompt metrics in a JSON list: distribution, rhyme counts &
   ratio, original prompt, etc.

Uses HuggingFace Transformers for direct model loading and generation.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 1.3
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


def fetch_rhymes(word: str, cache: Dict[str, Set[str]]) -> Set[str]:
    """Datamuse rhymes with caching."""
    word = word.lower()
    if word not in cache:
        try:
            resp = requests.get(DATAMUSE_URL, params={"rel_rhy": word, "max": 1000}, timeout=20)
            resp.raise_for_status()
            cache[word] = {item["word"].lower() for item in resp.json()}
        except Exception:  # noqa: BLE001
            _echo(f"Datamuse query failed for '{word}'.")
            cache[word] = set()
    return cache[word]


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
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    """Generate completion using HuggingFace model & return (raw, cleaned) completion."""
    prompt = f"/no_think Write only the next line of this rhyming couplet: {first_line.strip()},"
    
    # Tokenize input
    chattified_prompt = chattify([prompt, ""], tokenizer)
    inputs = tokenizer(chattified_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
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

def sample_completions(
    first_line: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    rhyme_cache: Dict[str, Set[str]],
) -> tuple[dict, list[str], list[str]]:
    last1 = last_word(first_line)
    distribution: Counter[str] = Counter()
    rhyme_hits = 0
    completions: list[str] = []
    raw_completions: list[str] = []

    for _ in range(num_samples):
        try:
            raw_second, cleaned_second = generate_second_line(first_line, model, tokenizer, temperature, max_tokens)
        except Exception as exc:  # noqa: BLE001
            _echo(f"Generation failed for '{first_line}': {exc}")
            raw_second = ""
            cleaned_second = ""
        last2 = last_word(cleaned_second)
        distribution[last2] += 1
        if last1 and last2 and last2 in fetch_rhymes(last1, rhyme_cache) and last1 != last2:
            rhyme_hits += 1
        completions.append(cleaned_second)
        raw_completions.append(raw_second)

    result = {
        "first_line": first_line,
        "last_word_1": last1,
        "distribution": dict(distribution),
        "rhyme_successes": rhyme_hits,
        "num_samples": num_samples,
        "rhyme_ratio": rhyme_hits / num_samples if num_samples else 0.0,
    }
    return result, completions, raw_completions


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Sample rhyming couplets via HuggingFace Transformers.")
    parser.add_argument("--input", type=Path, required=True, help="Text file with first lines (one per line).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/couplet_samples"), help="Directory to write results; filename will be <model>.json.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples per first line.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens per completion.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model data type.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory to write a verbose completions log (<model>.log).")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load model and tokenizer
    # ------------------------------------------------------------------
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    _echo(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
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

    rhyme_cache: Dict[str, Set[str]] = {}
    results: List[dict] = []
    log_fh = None
    raw_log_fh = None
    if args.log_dir is not None:
        model_slug = args.model.split("/")[-1]
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{model_slug}_completions.log"
        raw_log_path = args.log_dir / f"{model_slug}_raw_completions.log"
        log_fh = log_path.open("w", encoding="utf-8")
        raw_log_fh = raw_log_path.open("w", encoding="utf-8")

    interrupted = False
    try:
        for idx, fl in enumerate(first_lines, start=1):
            _echo(f"[{idx}/{len(first_lines)}] Sampling {args.num_samples} completions …")
            res, compls, raw_compls = sample_completions(
                fl,
                model,
                tokenizer,
                args.temperature,
                args.num_samples,
                args.max_new_tokens,
                rhyme_cache,
            )
            # Add raw completions to the result
            res["raw_completions"] = raw_compls
            res["cleaned_completions"] = compls
            results.append(res)

            # Write verbose log if enabled
            if log_fh is not None:
                print(f"FIRST LINE: {fl}", file=log_fh)
                for j, c in enumerate(compls, 1):
                    print(f"  {j:03d}: {c}", file=log_fh)
                print("", file=log_fh)
            
            # Write raw completions log if enabled
            if raw_log_fh is not None:
                print(f"FIRST LINE: {fl}", file=raw_log_fh)
                for j, c in enumerate(raw_compls, 1):
                    print(f"  {j:03d}: {c}", file=raw_log_fh)
                print("", file=raw_log_fh)
    except KeyboardInterrupt:
        interrupted = True
        _echo("Interrupted by user. Saving partial results …")

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    output_dir: Path = args.output_dir
    model_slug = args.model.split("/")[-1]
    output_path = output_dir / f"{model_slug}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if interrupted:
        _echo(f"Partial results ({len(results)} prompts) written to {output_path}.")
    else:
        _echo(f"Done ✓  Wrote {len(results)} entries to {output_path}.")

    if log_fh is not None:
        log_fh.close()
    if raw_log_fh is not None:
        raw_log_fh.close()


if __name__ == "__main__":  # pragma: no cover
    main() 