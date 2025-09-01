#!/usr/bin/env python3
"""evaluate_couplet_rhymes.py

Complete the second line of rhyming couplets with a local vLLM-powered model
and evaluate whether the two lines rhyme (using Datamuse).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Set

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PORT: int = 8000
DEFAULT_TEMPERATURE: float = 1.3
DATAMUSE_URL: str = "https://api.datamuse.com/words"
DEFAULT_MODEL: str = "placeholder-model"

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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _echo(message: str) -> None:
    """Print *message* to stderr and flush immediately."""
    print(message, file=sys.stderr, flush=True)


def wait_for_port(host: str, port: int, timeout: float = 600.0) -> None:
    """Block until *host:port* accepts TCP connections or *timeout* seconds elapse."""
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
    """Launch a vLLM OpenAI-compatible server and return the subprocess handle."""
    cmd = VLLM_BASE_CMD.copy()
    cmd[cmd.index("--model") + 1] = model
    cmd[cmd.index("--port") + 1] = str(port)

    if gpus is not None:
        cmd.extend(["--tensor-parallel-size", str(gpus)])

    _echo("Starting vLLM server:\n  " + " ".join(cmd))
    return subprocess.Popen(cmd)


def last_word(line: str) -> str:
    """Return the final word (letters and apostrophes) in *line*, lower-cased."""
    match = LAST_WORD_RE.search(line.strip())
    return match.group(1).lower() if match else ""


def fetch_rhymes(word: str, cache: Dict[str, Set[str]]) -> Set[str]:
    """Return a set of rhyming words for *word* (cached)."""
    word = word.lower()
    if word not in cache:
        try:
            resp = requests.get(DATAMUSE_URL, params={"rel_rhy": word, "max": 1000}, timeout=20)
            resp.raise_for_status()
            cache[word] = {item["word"].lower() for item in resp.json()}
        except Exception as exc:  # noqa: BLE001  (broad ≈ network)
            _echo(f"Datamuse query failed for '{word}': {exc}")
            cache[word] = set()
    return cache[word]


def generate_second_line(
    first_line: str,
    model: str,
    port: int,
    temperature: float,
) -> str:
    """Query the local OpenAI endpoint for the second line of the couplet."""

    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are a creative poet"
            #     ),
            # },
            {"role": "user", "content": f"Provide ONLY the second line completing "
                    "this rhyming couplet. Do NOT repeat the first line. Return one poetic "
                    f"line and nothing else: {first_line.strip()} /no_think"},
        ],
        "temperature": temperature,
        "max_tokens": 64,
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    raw_text: str = resp.json()["choices"][0]["message"]["content"].strip()
    return THINK_RE.sub("", raw_text).strip()


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def sanitize_second_line(first_line: str, raw_second: str) -> str:
    """Heuristically clean *raw_second* produced by the model.

    1. Remove common prefaces like "Second line:" / "The second line:".
    2. Drop trailing markers such as "/no_think" or similar.
    3. If the model repeated *first_line*, trim that portion.
    4. Collapse multi-line output to a single line, choosing the portion that
       is *not* an exact repetition of *first_line*.
    """

    # Remove common explanatory prefixes.
    cleaned = re.sub(r"^\s*(?:the\s+)?second\s+line\s*[:\-–]?\s*", "", raw_second, flags=re.I)

    # Remove no_think or similar markers.
    cleaned = re.sub(r"/\s*no?_?think\b.*", "", cleaned, flags=re.I)

    # If the first line appears verbatim at the start, drop it.
    first_norm = first_line.strip().lower()
    if cleaned.lower().startswith(first_norm):
        cleaned = cleaned[len(first_line):].lstrip(" \t\n,/:-—")

    # If the model separated lines by newline or slash, pick the part that
    # differs from the first line.
    delimit_split = re.split(r"[\n/]+", cleaned)
    if len(delimit_split) > 1:
        for part in delimit_split:
            part_stripped = part.strip(" \t-—")
            if part_stripped and part_stripped.lower() != first_norm:
                cleaned = part_stripped
                break

    # Collapse any remaining newlines/extra spaces.
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def evaluate_couplets(
    first_lines_file: Path,
    output_csv: Path,
    model: str,
    port: int,
    temperature: float,
    *,
    sonnet_file: Path | None = Path("data/sonnet_couplets.csv"),
    start_server: bool = False,
    gpus: int | None = None,
) -> None:
    """Complete couplets and evaluate rhymes.

    first_lines_file : Path
        Text file containing the first line of each couplet (one per line).
    sonnet_file : Path | None, optional
        Optional CSV file (with *line_1* column) containing Shakespeare
        sonnet couplets.  If provided, these will be evaluated in addition to
        *first_lines_file* and identified with the "Source" column in the
        output.
    output_csv : Path
        Where to write the evaluation CSV.
    model : str
        HF model ID served by the local vLLM server.
    port : int
        Port of the local OpenAI-compatible endpoint.
    temperature : float
        Sampling temperature for generation.
    start_server : bool, optional
        If true, this function will launch a vLLM server (and shut it down
        afterwards).  If false, it assumes one is already running.
    gpus : int | None, optional
        Number of GPUs (tensor-parallel size) to pass to vLLM when launching.
    """

    def _load_first_lines(path: Path, *, is_csv: bool) -> list[str]:
        """Return a list of first lines from *path* (text or CSV)."""
        if is_csv:
            lines: list[str] = []
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames and "line_1" in reader.fieldnames:
                    for row in reader:
                        ln = row.get("line_1", "").strip()
                        if ln:
                            lines.append(ln)
                else:
                    # Fallback to positional index if header absent or named differently
                    fh.seek(0)
                    reader2 = csv.reader(fh)
                    # Skip header if present (heuristic: contains non-letter characters like '1' or 'line')
                    for idx, row in enumerate(reader2):
                        if idx == 0 and any("line" in cell.lower() for cell in row):
                            continue  # header row
                        if row:
                            ln = row[0].strip()
                            if ln:
                                lines.append(ln)
            return lines
        else:
            return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # ------------------------------------------------------------------
    # Optionally launch a vLLM server
    # ------------------------------------------------------------------
    server_proc: subprocess.Popen[bytes] | None = None
    if start_server:
        server_proc = launch_vllm_server(model, port, gpus)
        # Ensure server terminates gracefully on SIGINT
        signal.signal(signal.SIGINT, lambda *_: server_proc.terminate() if server_proc else None)

    try:
        try:
            wait_for_port("localhost", port)
        except RuntimeError as exc:
            # If the user opted not to start the server, guide them.
            if not start_server:
                server_cmd = VLLM_BASE_CMD.copy()
                server_cmd[server_cmd.index("--model") + 1] = model
                server_cmd[server_cmd.index("--port") + 1] = str(port)
                if gpus is not None:
                    server_cmd.extend(["--tensor-parallel-size", str(gpus)])

                _echo(
                    "vLLM server does not appear to be running. Launch it with:\n\n"
                    + " ".join(server_cmd)
                    + "\n"
                )
                return  # Exit gracefully so the user can start the server.
            else:
                # We attempted to start the server but still failed – re-raise.
                raise exc

        # ------------------------------------------------------------------
        # Read first lines from datasets
        # ------------------------------------------------------------------

        datasets: list[tuple[str, str]] = []  # (source, first_line)

        # Primary dataset (plain text)
        primary_lines = _load_first_lines(first_lines_file, is_csv=False)
        if not primary_lines:
            raise SystemExit("Primary input file is empty or contains only blank lines.")
        datasets.extend([(first_lines_file.stem, ln) for ln in primary_lines])

        # Optional sonnet dataset (CSV)
        if sonnet_file is not None:
            sonnet_lines = _load_first_lines(sonnet_file, is_csv=True)
            if sonnet_lines:
                datasets.extend([(sonnet_file.stem, ln) for ln in sonnet_lines])

        if not datasets:
            raise SystemExit("No input lines found across provided datasets.")

        # Prepare rhyme cache and CSV output
        rhyme_cache: Dict[str, Set[str]] = {}
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "Source",
                "Dataset",
                "Rhymes",
                "Last_Word_1",
                "Last_Word_2",
                "First_Line",
                "Second_Line",
                "Raw_Second",
            ])

            rhymed = 0
            for idx, (src, first) in enumerate(datasets, start=1):
                dataset_label = (
                    "sonnet" if (sonnet_file is not None and src == sonnet_file.stem) else "qwen_poems"
                )
                _echo(f"[{idx}/{len(datasets)}] Generating completion … (source={src})")

                try:
                    raw_second = generate_second_line(first, model, port, temperature)
                    second = sanitize_second_line(first, raw_second)
                except Exception as exc:  # noqa: BLE001
                    _echo(f"Generation failed for '{first}': {exc}")
                    second = ""

                last1, last2 = last_word(first), last_word(second)
                rhymes = bool(last1 and last2 and last2 in fetch_rhymes(last1, rhyme_cache))
                rhymed += int(rhymes)

                writer.writerow([src, dataset_label, rhymes, last1, last2, first, second, raw_second])

        ratio = rhymed / len(datasets)
        _echo(f"Done ✓  {rhymed}/{len(datasets)} lines rhymed ({ratio:.1%}).")

    finally:
        if server_proc:
            _echo("Stopping vLLM server …")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete rhyming couplets with a vLLM model and judge rhyme accuracy.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/couplet_first_lines.txt"),
        help="Text file containing FIRST lines (one per line).",
    )
    parser.add_argument(
        "--sonnets",
        type=Path,
        default=Path("data/sonnet_couplets.csv"),
        help="CSV file containing Shakespeare sonnet couplets (with line_1 column).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV file to write results to.  If not provided, defaults to "
            "results/couplets/{model_name}.csv where {model_name} is the "
            "portion after the final slash in --model."
        ),
    )
    parser.add_argument("--model", required=True, help="HF model ID for the vLLM server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="OpenAI-compatible server port.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Launch a vLLM server automatically (otherwise expects one already running).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Tensor parallel size (number of GPUs) to pass to vLLM when launching.",
    )

    args = parser.parse_args()

    # Derive default output path if none supplied.
    output_path: Path
    if args.output is None:
        model_slug = args.model.split("/")[-1]
        output_path = Path("results/couplets") / f"{model_slug}.csv"
    else:
        output_path = args.output

    evaluate_couplets(
        first_lines_file=args.input,
        output_csv=output_path,
        model=args.model,
        port=args.port,
        temperature=args.temperature,
        sonnet_file=args.sonnets,
        start_server=args.start_server,
        gpus=args.gpus,
    ) 