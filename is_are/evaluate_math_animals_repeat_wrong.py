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
            "('is' / 'are') and then predicting the following number twice: once using the "
            "correct verb and once using the wrong verb as context."
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

        # Step 1: predict verb (is / are) for reference
        probs = predict_next_token(prompt_base, tokenizer, model, device)
        pred_verb, prob_verb = most_likely_between(probs, tokenizer, ["is", "are"])
        verb_correct = pred_verb == gold_verb

        # Determine the wrong verb
        wrong_verb = "are" if gold_verb == "is" else "is"

        # Predict number with correct verb
        prompt_with_correct_verb = f"{prompt_base} {gold_verb} "
        probs_num_correct = predict_next_token(prompt_with_correct_verb, tokenizer, model, device)
        top_id_correct = probs_num_correct.argmax().item()
        pred_number_tok_correct = tokenizer.decode([top_id_correct]).strip()
        prob_number_correct = probs_num_correct[top_id_correct].item()
        pred_number_int_correct = token_to_int(pred_number_tok_correct)
        number_correct_with_correct_verb = pred_number_int_correct == gold_number

        # Predict number with wrong verb
        prompt_with_wrong_verb = f"{prompt_base} {wrong_verb} "
        probs_num_wrong = predict_next_token(prompt_with_wrong_verb, tokenizer, model, device)
        top_id_wrong = probs_num_wrong.argmax().item()
        pred_number_tok_wrong = tokenizer.decode([top_id_wrong]).strip()
        prob_number_wrong = probs_num_wrong[top_id_wrong].item()
        pred_number_int_wrong = token_to_int(pred_number_tok_wrong)
        number_correct_with_wrong_verb = pred_number_int_wrong == gold_number

        records.append(
            {
                "Prompt": sentence,
                "Animal": row["animal"],
                "Gold_Verb": gold_verb,
                "Wrong_Verb": wrong_verb,
                "Predicted_Verb": pred_verb,
                "Verb_Correct": verb_correct,
                "Gold_Number": gold_number,
                "Predicted_Number_Token": pred_number_tok_correct,
                "Predicted_Number_Int": pred_number_int_correct,
                "Number_Correct": number_correct_with_correct_verb,
                "Predicted_Number_Token_Wrong_Verb": pred_number_tok_wrong,
                "Predicted_Number_Int_Wrong_Verb": pred_number_int_wrong,
                "Number_Correct_With_Wrong_Verb": number_correct_with_wrong_verb,
                "Prob_Verb": prob_verb,
                "Prob_Number": prob_number_correct,
                "Prob_Number_Wrong_Verb": prob_number_wrong,
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)

    # Aggregate metrics
    verb_acc = out_df["Verb_Correct"].mean()
    
    # Number accuracy metrics
    num_acc_correct = out_df["Number_Correct"].mean()
    num_acc_wrong = out_df["Number_Correct_With_Wrong_Verb"].mean()

    per_verb = (
        out_df.groupby("Gold_Verb")["Verb_Correct"].mean().to_dict()
    )

    print(f"==== {model_name} on Math-Animals ====")
    print(f"Overall verb accuracy: {verb_acc:.3%}")
    for v, acc in per_verb.items():
        print(f"  {v}: {acc:.3%}")
    print(f"Number accuracy with CORRECT verb: {num_acc_correct:.3%}")
    print(f"Number accuracy with WRONG verb: {num_acc_wrong:.3%}")
    
    # Per-verb breakdown for number accuracy
    print("\nNumber accuracy by gold verb:")
    for verb in ["is", "are"]:
        verb_subset = out_df[out_df["Gold_Verb"] == verb]
        if len(verb_subset) > 0:
            correct_acc = verb_subset["Number_Correct"].mean()
            wrong_acc = verb_subset["Number_Correct_With_Wrong_Verb"].mean()
            print(f"  {verb} - correct verb: {correct_acc:.3%}")
            print(f"  {verb} - wrong verb: {wrong_acc:.3%}")

    return out_df


if __name__ == "__main__":
    args = parse_args()
    evaluate_math_animals(
        model_name=args.model,
        output_path=args.output,
        dtype=torch.float32,
        device=args.device,
    ) 