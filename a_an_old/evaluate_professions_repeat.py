import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd  # type: ignore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and return a namespace."""
    parser = argparse.ArgumentParser(
        description="Evaluate a causal language model on the professions dataset with article + profession predictions."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF Transformers model name or local path (causal LM)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the per-example predictions CSV.",
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
    return parser.parse_args()


def pick_in_context_index(cur_idx: int, num_rows: int, rng: random.Random) -> int:
    """Return an index different from cur_idx."""
    idx = cur_idx
    while idx == cur_idx:
        idx = rng.randint(0, num_rows - 1)
    return idx


def predict_next_token(prompt: str, tokenizer, model, device: str) -> Tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_id = probs.argmax().item()
    token_str = tokenizer.decode([top_id])
    return token_str, probs[top_id].item()


DATASET_PATH = Path("data/professions_dataset_with_articles.csv")


def evaluate_professions(
    model_name: str,
    output_path: Path | str,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    device: str | None = None,
) -> pd.DataFrame:
    """Run evaluation and return a DataFrame with predictions.

    Args:
        model_name: HuggingFace model identifier or local path.
        output_path: Where to save the per-example CSV.
        seed: RNG seed for selecting in-context examples.
        device: 'cpu' or 'cuda'. If None, infer automatically.
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
        profession: str = row["Profession"].strip()
        article: str = row["Article"].strip()
        description: str = row["Description"].strip()

        # build trimmed contexts
        tokens = description.split()
        # Expect last two tokens are article and profession
        if len(tokens) < 2 or tokens[-1].lower() != profession.lower():
            raise ValueError(f"Description/token mismatch at index {idx}")
        if tokens[-2].lower() != article.lower():
            raise ValueError(f"Article mismatch at index {idx}")

        context_before_article = " ".join(tokens[:-2])  # up to 'is'

        # choose in-context example
        ic_idx = pick_in_context_index(idx, len(df), rng)
        ic_row = df.iloc[ic_idx]
        ic_description = ic_row["Description"].strip()

        sentence_before_article = f'{ic_description} {context_before_article}'.strip()
        sentence_with_article = f'{ic_description} {context_before_article} {article}'.strip()

        def gen_prefill(s):
            instruction: str = f"/no_think Repeat this sentence and complete it. {s}"

            messages = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]

            # Convert messages to Qwen3 chat format using tokenizer
            formatted_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # Prefilled response (what the model should start generating after)
            prefill: str = f"<think>\n\n</think> {s}"

            # Combine the formatted input with prefilled response
            # The model will continue generating after the prefilled content
            return formatted_input + prefill

        # final prompts as specified: [IC example]. [orig trimmed example]
        prompt_before_article = gen_prefill(sentence_before_article)
        prompt_with_article = gen_prefill(sentence_with_article)

        # predict article
        pred_article_tok, prob_article = predict_next_token(
            prompt_before_article, tokenizer, model, device
        )
        pred_article = pred_article_tok.strip().lower()
        # if tokenizer includes following space, take until next whitespace for multi-token predictions
        if " " in pred_article:
            pred_article = pred_article.split(" ")[0]

        # now predict profession given gold article
        pred_prof_tok, prob_prof = predict_next_token(
            prompt_with_article, tokenizer, model, device
        )
        pred_profession = pred_prof_tok.strip().lower()
        if " " in pred_profession:
            pred_profession = pred_profession.split(" ")[0]

        records.append(
            {
                "Profession": profession,
                "Article": article,
                "Predicted_Article": pred_article,
                "Article_Correct": pred_article == article.lower(),
                "Predicted_Profession": pred_profession,
                "Profession_Correct": profession.lower().startswith(pred_profession),
                "IC_Index": ic_idx,
                "Prob_Article": prob_article,
                "Prob_Profession": prob_prof,
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)

    # Compute metrics
    article_acc = out_df["Article_Correct"].mean()
    profession_acc = out_df["Profession_Correct"].mean()

    per_article = (
        out_df.groupby("Article")["Article_Correct"].mean().to_dict()
    )

    print(f"==== {model_name} Accuracy ====")
    print(f"Article accuracy: {article_acc:.3%}")
    for art, acc in per_article.items():
        print(f"  {art}: {acc:.3%}")
    print(f"Profession accuracy: {profession_acc:.3%}")

    return out_df


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate_professions(
        model_name=cli_args.model,
        output_path=cli_args.output,
        seed=cli_args.seed,
        device=cli_args.device,
    ) 