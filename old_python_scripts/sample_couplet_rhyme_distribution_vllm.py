#!/usr/bin/env python3
"""sample_couplet_rhyme_distribution_vllm.py

Sample multiple completions for rhyming couplets using a **vLLM** OpenAI-compatible
server (local).

For each first line in the provided text file, the script:

1. Sends a chat request asking for ONLY the second line of the couplet,
   sampling *N* times (default 200).
2. Extracts the last word of every completion and builds its frequency
   distribution.
3. Checks whether each generated last word rhymes with the prompt's last word
   (Datamuse API).
4. Records per-prompt metrics in a JSON list: distribution, rhyme counts &
   ratio, original prompt, etc.

The script can optionally launch the vLLM server itself (``--start-server``),
mirroring the behaviour of *evaluate_couplet_rhymes.py*.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set

import requests

# ---------------------------------------------------------------------------
# Configuration (adapted from evaluate_couplet_rhymes)
# ---------------------------------------------------------------------------
DEFAULT_PORT = 8000
DEFAULT_TEMPERATURE = 1.3
DATAMUSE_URL = "https://api.datamuse.com/words"
DEFAULT_MODEL = "placeholder-model"

VLLM_BASE_CMD: list[str] = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    DEFAULT_MODEL,
    "--dtype",
    "bfloat16",
    "--port",
    str(DEFAULT_PORT),
    "--trust-remote-code",
    "--max-model-len",
    "1000",
]

THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
LAST_WORD_RE = re.compile(r"([A-Za-z']+)[^A-Za-z']*$")

# Endpoint readiness poll (vLLM exposes /v1/models)
def wait_for_model_ready(port: int, model_id: str, timeout: float = 600.0) -> None:
    """Block until *model_id* appears in `/v1/models` list or *timeout* seconds."""
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/v1/models"
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if any(m.get("id") == model_id for m in data):
                    return
        except requests.RequestException:
            pass
        _echo("Waiting for model to finish loading …")
        time.sleep(10)
    raise RuntimeError("Timed out waiting for model to load in vLLM server.")

# ---------------------------------------------------------------------------
# Helper utilities (waiting for port, echo, etc.)
# ---------------------------------------------------------------------------

def _echo(msg: str) -> None:  # print to stderr & flush
    print(msg, file=sys.stderr, flush=True)


def wait_for_port(host: str, port: int, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            _echo("Waiting for server to become ready …")
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for server on {host}:{port}.")


def launch_vllm_server(model: str, port: int, gpus: int | None) -> subprocess.Popen[bytes]:
    cmd = VLLM_BASE_CMD.copy()
    cmd[cmd.index("--model") + 1] = model
    cmd[cmd.index("--port") + 1] = str(port)
    if gpus is not None:
        cmd.extend(["--tensor-parallel-size", str(gpus)])
    _echo("Starting vLLM server:\n  " + " ".join(cmd))
    return subprocess.Popen(cmd)


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

def generate_second_line(
    first_line: str,
    model: str,
    port: int,
    temperature: float,
    max_tokens: int,
) -> str:
    """Query local OpenAI endpoint & return cleaned completion."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"/no_think A rhyming couplet: {first_line.strip()},"
                ),
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    raw_text: str = resp.json()["choices"][0]["message"]["content"].strip()
    return sanitize_second_line(first_line, THINK_RE.sub("", raw_text).strip())


# ---------------------------------------------------------------------------
# Sampling routine
# ---------------------------------------------------------------------------

def sample_completions(
    first_line: str,
    model_id: str,
    port: int,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    rhyme_cache: Dict[str, Set[str]],
) -> tuple[dict, list[str]]:
    last1 = last_word(first_line)
    distribution: Counter[str] = Counter()
    rhyme_hits = 0
    completions: list[str] = []

    for _ in range(num_samples):
        try:
            second = generate_second_line(first_line, model_id, port, temperature, max_tokens)
        except Exception as exc:  # noqa: BLE001
            _echo(f"Generation failed for '{first_line}': {exc}")
            second = ""
        last2 = last_word(second)
        distribution[last2] += 1
        if last1 and last2 and last2 in fetch_rhymes(last1, rhyme_cache):
            rhyme_hits += 1
        completions.append(second)

    result = {
        "first_line": first_line,
        "last_word_1": last1,
        "distribution": dict(distribution),
        "rhyme_successes": rhyme_hits,
        "num_samples": num_samples,
        "rhyme_ratio": rhyme_hits / num_samples if num_samples else 0.0,
    }
    return result, completions


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Sample rhyming couplets via vLLM.")
    parser.add_argument("--input", type=Path, required=True, help="Text file with first lines (one per line).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/couplet_samples"), help="Directory to write results; filename will be <model>.json.")
    parser.add_argument("--model", required=True, help="HF model ID served by vLLM.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port where the server listens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples per first line.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens per completion.")
    parser.add_argument("--start-server", action="store_true", help="Launch vLLM server automatically.")
    parser.add_argument("--gpus", type=int, default=None, help="Tensor parallel size for vLLM.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory to write a verbose completions log (<model>.log).")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Optionally launch server
    # ------------------------------------------------------------------
    server_proc: subprocess.Popen[bytes] | None = None
    if args.start_server:
        server_proc = launch_vllm_server(args.model, args.port, args.gpus)

    try:
        wait_for_port("localhost", args.port)
        wait_for_model_ready(args.port, args.model)
    except RuntimeError as exc:
        if not args.start_server:
            cmd = VLLM_BASE_CMD.copy()
            cmd[cmd.index("--model") + 1] = args.model
            cmd[cmd.index("--port") + 1] = str(args.port)
            if args.gpus is not None:
                cmd.extend(["--tensor-parallel-size", str(args.gpus)])
            _echo("vLLM server not found. Launch it with:\n  " + " ".join(cmd))
            sys.exit(1)
        raise exc

    # ------------------------------------------------------------------
    # Read first lines
    # ------------------------------------------------------------------
    first_lines: List[str] = [ln.strip() for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not first_lines:
        sys.exit("Input file is empty or only blank lines.")

    rhyme_cache: Dict[str, Set[str]] = {}
    results: List[dict] = []
    log_fh = None
    if args.log_dir is not None:
        model_slug = args.model.split("/")[-1]
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{model_slug}_completions.log"
        log_fh = log_path.open("w", encoding="utf-8")

    interrupted = False
    try:
        for idx, fl in enumerate(first_lines, start=1):
            _echo(f"[{idx}/{len(first_lines)}] Sampling {args.num_samples} completions …")
            res, compls = sample_completions(
                fl,
                args.model,
                args.port,
                args.temperature,
                args.num_samples,
                args.max_new_tokens,
                rhyme_cache,
            )
            results.append(res)

            # Write verbose log if enabled
            if log_fh is not None:
                print(f"FIRST LINE: {fl}", file=log_fh)
                for j, c in enumerate(compls, 1):
                    print(f"  {j:03d}: {c}", file=log_fh)
                print("", file=log_fh)
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

    if server_proc:
        _echo("Stopping vLLM server …")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":  # pragma: no cover
    main() 