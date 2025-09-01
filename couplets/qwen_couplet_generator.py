#!/usr/bin/env python3
"""qwen_couplet_generator.py

This script
1. Launches a local vLLM OpenAI-compatible server serving the Qwen3-32B model
   in bfloat16 precision.
2. Waits until the server is ready to accept requests.
3. Queries the server <N> times (default 50) at a moderately high temperature
   to generate the FIRST line of a rhyming couplet.
4. Writes each generated line to ``couplet_first_lines.txt`` one per line.

Requirements (install manually if not present):
  pip install vllm requests

Ensure you have sufficient GPU memory (≈ 70 GB+) and driver support for
bfloat16 before running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, TextIO

import requests
import re
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "Qwen/Qwen3-32B"
DEFAULT_PORT: int = 8000
DEFAULT_LINES: int = 1000
DEFAULT_TEMPERATURE: float = 1.7  # "moderately high"
OUTPUT_FILE: Path = Path("data/couplet_first_lines.txt")
DEFAULT_TOPICS_FILE: Path = Path("data/poem_topics.txt")

VLLM_SERVER_CMD = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    DEFAULT_MODEL,
    "--dtype",
    "bfloat16",
    "--port",
    str(DEFAULT_PORT),
    "--trust-remote-code",  # required for Qwen models
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _print(msg: str, /) -> None:
    """Print to stderr so stdout can be redirected if desired."""
    print(msg, file=sys.stderr, flush=True)


def start_vllm_server(env: dict | None = None) -> subprocess.Popen[bytes]:
    """Start the vLLM OpenAI-compatible server and return the process handle."""
    _print(f"Starting vLLM server: {' '.join(VLLM_SERVER_CMD)}")
    return subprocess.Popen(VLLM_SERVER_CMD, env=env)


def wait_for_port(host: str, port: int, timeout: float = 600.0) -> None:
    """Block until *host:port* is accepting TCP connections or *timeout* seconds elapsed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                _print("vLLM server is ready ✔️")
                return
        except OSError:
            _print("Waiting for server to become ready…")
            time.sleep(2)
    raise RuntimeError("Timed out waiting for the vLLM server to start.")


def generate_lines(
    model: str,
    num_lines: int,
    temperature: float,
    topics: List[str],
    port: int = DEFAULT_PORT,
    output_handle: TextIO | None = None,
) -> List[str]:
    """Query the local vLLM server ``num_lines`` times and return the generated lines."""

    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    lines: List[str] = []

    for idx in range(num_lines):
        topic = random.choice(topics)

        system_prompt = (
            "You are a creative poet. Produce ONLY the first line of a rhyming "
            f"couplet about the topic: '{topic}'. Return a single poetic line and nothing else. /nothink"
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 32,
        }

        _print(f"Request {idx + 1}/{num_lines} (topic: {topic}) → temperature={temperature}")
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        raw_text: str = r.json()["choices"][0]["message"]["content"].strip()

        # Remove "<think>...</think>" sections entirely (case-insensitive, multiline).
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()

        _print(f"← {cleaned}")
        lines.append(cleaned)

        # Stream-write to file if handle provided.
        if output_handle:
            output_handle.write(cleaned + "\n")
            output_handle.flush()

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate first lines of rhyming couplets using Qwen3-32B via vLLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model hub ID (default: %(default)s)")
    parser.add_argument("--lines", type=int, default=DEFAULT_LINES, help="Number of lines to generate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the local OpenAI server (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output text file (default: %(default)s)")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use with vLLM (overrides default of using all visible GPUs)")
    parser.add_argument("--topics-file", type=Path, default=DEFAULT_TOPICS_FILE, help="Text file with poem topics (one per line)")
    args = parser.parse_args()

    # Read topics list
    try:
        topics_list = [line.strip() for line in args.topics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError as exc:
        raise SystemExit(f"Topics file not found: {args.topics_file}") from exc

    # Update global command list with custom args if needed.
    vllm_cmd = VLLM_SERVER_CMD.copy()
    vllm_cmd[vllm_cmd.index("--model") + 1] = args.model
    vllm_cmd[vllm_cmd.index("--port") + 1] = str(args.port)

    # Respect user-specified GPU count by adding the tensor parallel flag.
    if args.gpus is not None:
        vllm_cmd.extend(["--tensor-parallel-size", str(args.gpus)])

    # Inform user how to launch server manually
    _print("\nIf the vLLM server is not already running, launch it with:\n\n" + " ".join(vllm_cmd) + "\n")

    # Wait for server availability (either already up or started externally)
    wait_for_port("localhost", args.port)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out_f:
        lines = generate_lines(
            model=args.model,
            num_lines=args.lines,
            temperature=args.temperature,
            topics=topics_list,
            port=args.port,
            output_handle=out_f,
        )

    _print(f"Saved {len(lines)} lines total to {args.output.resolve()}")

    # Done


if __name__ == "__main__":
    main() 