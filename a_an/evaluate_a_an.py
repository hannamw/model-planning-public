import random
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd  # type: ignore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

# Configuration
SEED = 42
IC_ARTICLE_FILTER = None  # 'a', 'an', or None for no filtering


def pick_in_context_index(cur_idx: int, num_rows: int, rng: random.Random) -> int:
    """Return an index different from cur_idx."""
    idx = cur_idx
    while idx == cur_idx:
        idx = rng.randint(0, num_rows - 1)
    return idx

def _chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified


def pick_filtered_in_context_index(
    cur_idx: int, 
    df: pd.DataFrame, 
    rng: random.Random, 
    article_filter: str | None = None
) -> int:
    """Return an index different from cur_idx, optionally filtered by article type.
    
    Args:
        cur_idx: Current example index to avoid
        df: The dataframe with professions data
        rng: Random number generator
        article_filter: 'a', 'an', or None for no filtering
    
    Returns:
        Valid index for in-context example
    """
    if article_filter is None:
        return pick_in_context_index(cur_idx, len(df), rng)
    
    # Filter indices by article type
    valid_indices = []
    for idx, row in df.iterrows():
        if idx != cur_idx and row["article"].strip().lower() == article_filter.lower():
            valid_indices.append(idx)
    
    if not valid_indices:
        raise ValueError(f"No examples found with article '{article_filter}' (excluding current index {cur_idx})")
    
    return rng.choice(valid_indices)


def predict_next_token(prompt: str, tokenizer, model) -> Tuple[str, float, float, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_id = probs.argmax().item()
    token_str = tokenizer.decode([top_id])
    
    # Get probabilities for "a" and "an" tokens
    a_token_id = tokenizer.encode(" a", add_special_tokens=False)[-1]  # Get last token in case of multiple
    an_token_id = tokenizer.encode(" an", add_special_tokens=False)[-1]
    
    prob_a = probs[a_token_id].item()
    prob_an = probs[an_token_id].item()
    
    return token_str, probs[top_id].item(), prob_a, prob_an


DATASET_PATH = Path("data/a_an_general.csv")

def evaluate_professions(
    model_name: str,
    output_path: Path | str,
    dtype: torch.dtype = torch.float32,
) -> pd.DataFrame:
    """Run evaluation and return a DataFrame with predictions.

    Args:
        model_name: HuggingFace model identifier or local path.
        output_path: Where to save the per-example CSV.
        dtype: PyTorch dtype for model weights.
    Returns:
        pandas DataFrame of per-example predictions.
    """

    df = pd.read_csv(DATASET_PATH)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chattify = partial(_chattify, tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name,).to(device="cuda", dtype=dtype)
    model.eval()

    rng = random.Random(SEED)

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        profession: str = row["word"].strip()
        article: str = row["article"].strip()
        description: str = row["sentence"].strip()

        context_before_article = description
        context_with_article = f"{context_before_article} {article}"

        # choose in-context example
        ic_idx = pick_filtered_in_context_index(idx, df, rng, IC_ARTICLE_FILTER)
        ic_row = df.iloc[ic_idx]
        ic_description = f'{ic_row["sentence"].strip()} {ic_row["article"].strip()} {ic_row["word"].strip()}'

        # final prompts as specified: [IC example]. [orig trimmed example]
        prompt_before_article = f"{ic_description}. {context_before_article}".strip()
        prompt_with_article = f"{ic_description}. {context_with_article}".strip()

        # predict article
        chattified_prompt_before_article = chattify([prompt_before_article])
        pred_article_tok, prob_article, prob_a, prob_an = predict_next_token(
            chattified_prompt_before_article, tokenizer, model
        )
        pred_article = pred_article_tok.strip().lower()
        # if tokenizer includes following space, take until next whitespace for multi-token predictions
        if " " in pred_article:
            pred_article = pred_article.split(" ")[0]

        # now predict profession given gold article
        chattified_prompt_with_article = chattify([prompt_with_article])
        pred_prof_tok, prob_prof, _, _ = predict_next_token(
            chattified_prompt_with_article, tokenizer, model
        )
        pred_profession = pred_prof_tok.strip().lower()
        if " " in pred_profession:
            pred_profession = pred_profession.split(" ")[0]

        # predict profession given wrong article
        wrong_article = "an" if article.lower() == "a" else "a"
        context_with_wrong_article = f"{context_before_article} {wrong_article}"
        prompt_with_wrong_article = f"{ic_description}. {context_with_wrong_article}".strip()
        
        chattified_prompt_with_wrong_article = chattify([prompt_with_wrong_article])
        pred_prof_wrong_tok, prob_prof_wrong, _, _ = predict_next_token(
            chattified_prompt_with_wrong_article, tokenizer, model
        )
        pred_profession_wrong = pred_prof_wrong_tok.strip().lower()
        if " " in pred_profession_wrong:
            pred_profession_wrong = pred_profession_wrong.split(" ")[0]

        # Determine if correct article has higher probability
        correct_article_prob = prob_a if article.lower() == "a" else prob_an
        incorrect_article_prob = prob_an if article.lower() == "a" else prob_a
        correct_article_higher = correct_article_prob > incorrect_article_prob

        records.append(
            {
                "planned": profession,
                "article": article,
                "predicted_article": pred_article,
                "article_correct": pred_article == article.lower(),
                "predicted_planned": pred_profession,
                "planned_correct": profession.lower().startswith(pred_profession),
                "predicted_planned_wrong_article": pred_profession_wrong,
                "planned_wrong_article_correct": profession.lower().startswith(pred_profession_wrong),
                "prob_article": prob_article,
                "prob_a": prob_a,
                "prob_an": prob_an,
                "correct_article_higher": correct_article_higher,
                "prob_planned": prob_prof,
                "prob_planned_wrong_article": prob_prof_wrong,
                "ic_planned": ic_row["word"].strip(),
                "ic_article": ic_row["article"].strip(),
                "ic_description": ic_description,
                "prompt_before_article": chattified_prompt_before_article,
                "prompt_with_article": chattified_prompt_with_article,
                "prompt_with_wrong_article": chattified_prompt_with_wrong_article,
            }
        )

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Compute metrics
    article_acc = out_df["article_correct"].mean()
    profession_acc = out_df["planned_correct"].mean()
    profession_wrong_acc = out_df["planned_wrong_article_correct"].mean()
    correct_article_higher_acc = out_df["correct_article_higher"].mean()

    per_article = (
        out_df.groupby("article")["article_correct"].mean().to_dict()
    )

    print(f"==== {model_name} Accuracy ====")
    print(f"Article accuracy: {article_acc:.3%}")
    for art, acc in per_article.items():
        print(f"  {art}: {acc:.3%}")
    print(f"Correct article higher probability: {correct_article_higher_acc:.3%}")
    print(f"Planned accuracy (correct article): {profession_acc:.3%}")
    print(f"Planned accuracy (wrong article): {profession_wrong_acc:.3%}")

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
        
        evaluate_professions(
            model_name=model_name,
            output_path=output_path,
        ) 