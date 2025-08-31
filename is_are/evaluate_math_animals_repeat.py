import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd  # type: ignore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

# Path to the math animals dataset (CSV with columns: prompt, animal, number, answer)
DATASET_PATH = Path("data/animals_dataset.csv")

# Simple mapping from English number words (lower-case) to integers
WORD2NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a causal LM on the math-animals dataset by first predicting the verb "
            "('is' / 'are') and then predicting the following number, using the model's own "
            "most-likely verb as context."
        )
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
        help="Where to write the per-example predictions CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cpu or cuda)",
    )
    return parser.parse_args()


def predict_next_token(prompt: str, tokenizer, model, device: str):
    """Return the softmax probabilities over the vocabulary for the next token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # shape: [vocab]
    probs = torch.softmax(logits, dim=-1)
    return probs


def most_likely_between(probs: torch.Tensor, tokenizer, candidates: List[str]):
    """Return the most-likely token among *candidates*.

    All *candidates* must correspond to exactly one token under the provided
    tokenizer.  If any candidate is not a single token, a ValueError is raised
    immediately – this guards against accidental misuse where multi-token
    strings would yield misleading probabilities.
    """

    best_tok: str | None = None
    best_prob: float = -1.0

    assert len(candidates)

    for cand in candidates:
        token_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Candidate '{cand}' is not a single token for this tokenizer; got {len(token_ids)} tokens."
            )
        prob_val = probs[token_ids[0]].item()
        if prob_val > best_prob:
            best_prob = prob_val
            best_tok = cand

    return best_tok, best_prob


def token_to_int(tok: str) -> int | None:
    """Convert a token representing a number to int, if possible."""
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return WORD2NUM.get(tok)


def evaluate_math_animals(
    model_name: str,
    output_path: Path | str,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> pd.DataFrame:
    """Run evaluation and return per-example predictions DataFrame."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and model
    df = pd.read_csv(DATASET_PATH)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device, dtype=dtype)
    model.eval()

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sentence: str = row["prompt"].strip()

        instruction: str = f"Repeat this sentence and complete it. {sentence}"

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
        prefill: str = f"<think>\n\n</think> {sentence}"

        # Combine the formatted input with prefilled response
        # The model will continue generating after the prefilled content
        prompt_base = formatted_input + prefill

        gold_verb: str = row["answer"].strip().lower()  # 'is' / 'are'
        gold_number: int = int(row["number"])

        # Step 1: predict verb (is / are)
        probs = predict_next_token(prompt_base, tokenizer, model, device)
        pred_verb, prob_verb = most_likely_between(probs, tokenizer, ["is", "are"])
        verb_correct = pred_verb == gold_verb

        # Build new prompt including predicted verb followed by a space to mimic natural continuation
        prompt_with_verb = f"{prompt_base} {pred_verb} "

        # Step 2: predict number
        probs_num = predict_next_token(prompt_with_verb, tokenizer, model, device)
        top_id = probs_num.argmax().item()
        pred_number_tok = tokenizer.decode([top_id]).strip()
        prob_number = probs_num[top_id].item()

        pred_number_int = token_to_int(pred_number_tok)
        number_correct = pred_number_int == gold_number

        records.append(
            {
                "Prompt": sentence,
                "Animal": row["animal"],
                "Gold_Verb": gold_verb,
                "Predicted_Verb": pred_verb,
                "Verb_Correct": verb_correct,
                "Gold_Number": gold_number,
                "Predicted_Number_Token": pred_number_tok,
                "Predicted_Number_Int": pred_number_int,
                "Number_Correct": number_correct,
                "Prob_Verb": prob_verb,
                "Prob_Number": prob_number,
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)

    # Aggregate metrics
    verb_acc = out_df["Verb_Correct"].mean()
    num_acc = out_df["Number_Correct"].mean()

    per_verb = (
        out_df.groupby("Gold_Verb")["Verb_Correct"].mean().to_dict()
    )

    print(f"==== {model_name} on Math-Animals ====")
    print(f"Overall verb accuracy: {verb_acc:.3%}")
    for v, acc in per_verb.items():
        print(f"  {v}: {acc:.3%}")
    print(f"Number accuracy: {num_acc:.3%}")

    return out_df


if __name__ == "__main__":
    args = parse_args()
    evaluate_math_animals(
        model_name=args.model,
        output_path=args.output,
        dtype=torch.float32,
        device=args.device,
    ) 