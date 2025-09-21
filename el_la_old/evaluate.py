import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd  # type: ignore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

# Default Qwen3 models to evaluate
DEFAULT_QWEN3_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B", 
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and return a namespace."""
    parser = argparse.ArgumentParser(
        description="Evaluate a causal language model on the Spanish nouns dataset with article + noun predictions."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="HF Transformers model names or local paths (causal LM). If not specified, uses default Qwen3 models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/behavioral"),
        help="Directory where to write the per-example predictions CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting in-context examples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cpu or cuda).",
    )
    parser.add_argument(
        "--ic-article-filter",
        type=str,
        choices=["el", "la"],
        default=None,
        help="Filter in-context examples by article type ('el' or 'la'). If not specified, uses all examples.",
    )
    return parser.parse_args()


def pick_in_context_index(
    cur_idx: int, 
    df: pd.DataFrame, 
    rng: random.Random, 
    article_filter: str | None = None
) -> int:
    """Return an index different from cur_idx, optionally filtered by article type."""
    valid_indices = [
        idx for idx, row in df.iterrows() 
        if idx != cur_idx and (
            article_filter is None or 
            row["Article"].strip().lower() == article_filter.lower()
        )
    ]
    
    if not valid_indices:
        filter_msg = f" with article '{article_filter}'" if article_filter else ""
        raise ValueError(f"No examples found{filter_msg} (excluding current index {cur_idx})")
    
    return rng.choice(valid_indices)


def predict_next_token(prompt: str, tokenizer, model, device: str) -> Tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_id = probs.argmax().item()
    token_str = tokenizer.decode([top_id])
    return token_str, probs[top_id].item()


DATASET_PATH = Path("data/spanish-data.csv")

def evaluate_nouns(
    model_name: str,
    output_path: Path | str,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    device: str | None = None,
    ic_article_filter: str | None = None,
) -> pd.DataFrame:
    """Run evaluation and return a DataFrame with predictions.

    Args:
        model_name: HuggingFace model identifier or local path.
        output_path: Where to save the per-example CSV.
        dtype: PyTorch dtype for model weights.
        seed: RNG seed for selecting in-context examples.
        device: 'cpu' or 'cuda'. If None, infer automatically.
        ic_article_filter: Filter in-context examples by article ('a', 'an', or None for no filtering).
    Returns:
        pandas DataFrame of per-example predictions.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATASET_PATH)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,).to(device=device, dtype=dtype)
    model.eval()

    rng = random.Random(seed)

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        spanish_noun: str = row["Spanish"].strip()
        english_noun: str = row["English"].strip()
        article: str = row["Article"].strip()
        description: str = row["Example"].strip()

        context_before_article = description
        context_with_article = f"{context_before_article} {article}"

        # choose in-context example
        ic_idx = pick_in_context_index(idx, df, rng, ic_article_filter)
        ic_row = df.iloc[ic_idx]
        ic_example = ic_row["Example"].strip()
        ic_article = ic_row['Article']
        ic_spanish = ic_row['Spanish'].strip()
        ic_description = f"{ic_example} {ic_article} {ic_spanish}"

        # final prompts as specified: [IC example]. [orig trimmed example]
        prompt_before_article = f"{ic_description}. {context_before_article}".strip()
        prompt_with_article = f"{ic_description}. {context_with_article}".strip()

        # predict article
        pred_article_tok, prob_article = predict_next_token(
            prompt_before_article, tokenizer, model, device
        )
        pred_article = pred_article_tok.strip().lower()
        # if tokenizer includes following space, take until next whitespace for multi-token predictions
        if " " in pred_article:
            pred_article = pred_article.split(" ")[0]

        # now predict noun given gold article
        pred_noun_tok, prob_noun = predict_next_token(
            prompt_with_article, tokenizer, model, device
        )
        pred_noun = pred_noun_tok.strip().lower()
        if " " in pred_noun:
            pred_noun = pred_noun.split(" ")[0]

        # predict noun given wrong article
        wrong_article_mapping = {
            "el": "la",
            "la": "el",
            "los": "las",
            "las": "los",
        }
        wrong_article = wrong_article_mapping[article]
        context_with_wrong_article = f"{context_before_article} {wrong_article}"
        prompt_with_wrong_article = f"{ic_description}. {context_with_wrong_article}".strip()
        
        pred_noun_wrong_tok, prob_noun_wrong = predict_next_token(
            prompt_with_wrong_article, tokenizer, model, device
        )
        pred_noun_wrong = pred_noun_wrong_tok.strip().lower()
        if " " in pred_noun_wrong:
            pred_noun_wrong = pred_noun_wrong.split(" ")[0]

        records.append(
            {
                "Spanish_Noun": spanish_noun,
                "English_Noun": english_noun,
                "Article": article,
                "Predicted_Article": pred_article,
                "Article_Correct": pred_article == article.lower(),
                "Predicted_Noun": pred_noun,
                "Noun_Correct": spanish_noun.lower().startswith(pred_noun),
                "Predicted_Noun_Wrong_Article": pred_noun_wrong,
                "Noun_Wrong_Article_Correct": spanish_noun.lower().startswith(pred_noun_wrong),
                "Prob_Article": prob_article,
                "Prob_Noun": prob_noun,
                "Prob_Noun_Wrong_Article": prob_noun_wrong,
                "IC_Spanish_Noun": ic_row["Spanish"].strip(),
                "IC_English_Noun": ic_row["English"].strip(),
                "IC_Article": ic_row["Article"].strip(),
                "IC_Description": ic_description,
                "Description": prompt_before_article
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)

    # Compute metrics
    article_acc = out_df["Article_Correct"].mean()
    noun_acc = out_df["Noun_Correct"].mean()
    noun_wrong_acc = out_df["Noun_Wrong_Article_Correct"].mean()

    per_article = (
        out_df.groupby("Article")["Article_Correct"].mean().to_dict()
    )

    print(f"==== {model_name} Accuracy ====")
    print(f"Article accuracy: {article_acc:.3%}")
    for art, acc in per_article.items():
        print(f"  {art}: {acc:.3%}")
    print(f"Noun accuracy (correct article): {noun_acc:.3%}")
    print(f"Noun accuracy (wrong article): {noun_wrong_acc:.3%}")

    return out_df


if __name__ == "__main__":
    cli_args = parse_args()
    
    # Use provided models or default to Qwen3 models
    models_to_evaluate = cli_args.models if cli_args.models else DEFAULT_QWEN3_MODELS
    
    # Ensure output directory exists
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating {models_to_evaluate}")
    
    for model_name in models_to_evaluate:
        print(f"Starting evaluation for: {model_name}")
        
        # Create output filename based on model name
        model_short_name = model_name.split("/")[-1]
        output_file = cli_args.output_dir / f"{model_short_name}.csv"
        
        evaluate_nouns(
            model_name=model_name,
            output_path=output_file,
            seed=cli_args.seed,
            device=cli_args.device,
            ic_article_filter=cli_args.ic_article_filter,
        )
        print(f"Results saved to: {output_file}")

    
    print(f"Results saved in: {cli_args.output_dir}")
