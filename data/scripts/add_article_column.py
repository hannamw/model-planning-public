import argparse
import csv
import pathlib
import sys

VOWELS = set("aeiouAEIOU")

def infer_article(word: str) -> str:
    """Return the appropriate indefinite article ("a" or "an") for *word*.

    The heuristic is based on the first letter. This covers most cases reliably
    for standard profession names present in the dataset.
    """
    if not word:
        return "a"
    first_char = word[0]
    return "an" if first_char in VOWELS else "a"


def add_article_column(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Read *src*, prepend an 'Article' column with inferred values, and write to *dst*."""
    with src.open(newline="", encoding="utf-8") as f_in, dst.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        try:
            header = next(reader)
        except StopIteration:
            print(f"Error: {src} appears to be empty.", file=sys.stderr)
            sys.exit(1)

        if not header:
            print(f"Error: {src} header row is empty.", file=sys.stderr)
            sys.exit(1)

        # Insert the Article column after the Profession column, which we assume is the first.
        new_header = [header[0], "Article", *header[1:]]
        writer.writerow(new_header)

        for row in reader:
            if not row:
                continue  # Skip blank lines
            profession = row[0].strip()
            article = infer_article(profession)
            new_row = [profession, article, *row[1:]]
            writer.writerow(new_row)

    print(f"Wrote updated CSV with articles to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add an 'Article' column to a professions CSV.")
    parser.add_argument("input_csv", type=pathlib.Path, help="Path to the original CSV file")
    parser.add_argument(
        "output_csv",
        type=pathlib.Path,
        nargs="?",
        help="Destination CSV (default: append _with_articles before the extension)",
    )
    args = parser.parse_args()

    input_csv: pathlib.Path = args.input_csv
    output_csv: pathlib.Path = (
        args.output_csv
        if args.output_csv
        else input_csv.with_name(f"{input_csv.stem}_with_articles{input_csv.suffix}")
    )

    if not input_csv.is_file():
        print(f"Error: {input_csv} does not exist or is not a file.", file=sys.stderr)
        sys.exit(1)

    add_article_column(input_csv, output_csv)