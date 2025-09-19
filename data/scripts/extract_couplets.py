import csv
import re
from pathlib import Path


def extract_couplets(input_path: Path) -> list[tuple[str, str]]:
    """Read the sonnets file and return a list of (penultimate_line, last_line) tuples."""
    text = input_path.read_text(encoding="utf-8")

    # Split sonnets on blank lines (one or more).
    raw_sonnets = re.split(r"\n\s*\n", text.strip())

    couplets: list[tuple[str, str]] = []
    for sonnet in raw_sonnets:
        # Clean up and split into individual non-empty lines
        lines = [line.strip() for line in sonnet.splitlines() if line.strip()]
        if len(lines) < 2:
            # Skip malformed sonnets
            continue
        couplets.append((lines[-2], lines[-1]))

    return couplets


def write_couplets_to_csv(couplets: list[tuple[str, str]], output_path: Path) -> None:
    """Write the couplets to a CSV file with two columns."""
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["line_1", "line_2"])
        writer.writerows(couplets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract the final couplet from each Shakespeare sonnet and store them in a CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shakespeare_sonnets_unformatted.txt"),
        help="Path to the text file containing Shakespeare's sonnets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sonnet_couplets.csv"),
        help="Destination CSV file path.",
    )

    args = parser.parse_args()

    couplets = extract_couplets(args.input)
    write_couplets_to_csv(couplets, args.output)

    print(f"Extracted {len(couplets)} couplets and wrote them to {args.output}") 