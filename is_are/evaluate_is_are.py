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
                "prompt": sentence,
                "animal": row["animal"],
                "gold_verb": gold_verb,
                "wrong_verb": wrong_verb,
                "predicted_verb": pred_verb,
                "verb_correct": verb_correct,
                "gold_number": gold_number,
                "predicted_number_token": pred_number_tok_correct,
                "predicted_number_int": pred_number_int_correct,
                "number_correct": number_correct_with_correct_verb,
                "predicted_number_token_wrong_verb": pred_number_tok_wrong,
                "predicted_number_int_wrong_verb": pred_number_int_wrong,
                "number_correct_with_wrong_verb": number_correct_with_wrong_verb,
                "prob_verb": prob_verb,
                "prob_number": prob_number_correct,
                "prob_number_wrong_verb": prob_number_wrong,
                "prompt_base": prompt_base,
                "prompt_with_correct_verb": prompt_with_correct_verb,
                "prompt_with_wrong_verb": prompt_with_wrong_verb,
            }
        )

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Aggregate metrics
    verb_acc = out_df["verb_correct"].mean()
    
    # Number accuracy metrics
    num_acc_correct = out_df["number_correct"].mean()
    num_acc_wrong = out_df["number_correct_with_wrong_verb"].mean()

    per_verb = (
        out_df.groupby("gold_verb")["verb_correct"].mean().to_dict()
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
        verb_subset = out_df[out_df["gold_verb"] == verb]
        if len(verb_subset) > 0:
            correct_acc = verb_subset["number_correct"].mean()
            wrong_acc = verb_subset["number_correct_with_wrong_verb"].mean()
            print(f"  {verb} - correct verb: {correct_acc:.3%}")
            print(f"  {verb} - wrong verb: {wrong_acc:.3%}")

    return out_df


if __name__ == "__main__":
    model_names = [f'Qwen/Qwen3-{size}B' for size in ['0.6', '1.7', '4', '8', '14', '32']]
    
    for model_name in model_names:
        safe_model_name = model_name.split('/')[-1]
        output_path = Path(f"results/behavioral/{safe_model_name}.csv")
        
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"Output path: {output_path}")
        print(f"{'='*60}")
        
        evaluate_math_animals(
            model_name=model_name,
            output_path=output_path,
        ) 