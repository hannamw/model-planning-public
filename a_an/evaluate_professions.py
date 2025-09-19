import random
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
        if idx != cur_idx and row["Article"].strip().lower() == article_filter.lower():
            valid_indices.append(idx)
    
    if not valid_indices:
        raise ValueError(f"No examples found with article '{article_filter}' (excluding current index {cur_idx})")
    
    return rng.choice(valid_indices)


def predict_next_token(prompt: str, tokenizer, model) -> Tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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
    model = AutoModelForCausalLM.from_pretrained(model_name,).to(device="cuda", dtype=dtype)
    model.eval()

    rng = random.Random(SEED)

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
        context_with_article = f"{context_before_article} {article}"

        # choose in-context example
        ic_idx = pick_filtered_in_context_index(idx, df, rng, IC_ARTICLE_FILTER)
        ic_row = df.iloc[ic_idx]
        ic_description = ic_row["Description"].strip()

        # final prompts as specified: [IC example]. [orig trimmed example]
        prompt_before_article = f"{ic_description}. {context_before_article}".strip()
        prompt_with_article = f"{ic_description}. {context_with_article}".strip()

        # predict article
        pred_article_tok, prob_article = predict_next_token(
            prompt_before_article, tokenizer, model
        )
        pred_article = pred_article_tok.strip().lower()
        # if tokenizer includes following space, take until next whitespace for multi-token predictions
        if " " in pred_article:
            pred_article = pred_article.split(" ")[0]

        # now predict profession given gold article
        pred_prof_tok, prob_prof = predict_next_token(
            prompt_with_article, tokenizer, model
        )
        pred_profession = pred_prof_tok.strip().lower()
        if " " in pred_profession:
            pred_profession = pred_profession.split(" ")[0]

        # predict profession given wrong article
        wrong_article = "an" if article.lower() == "a" else "a"
        context_with_wrong_article = f"{context_before_article} {wrong_article}"
        prompt_with_wrong_article = f"{ic_description}. {context_with_wrong_article}".strip()
        
        pred_prof_wrong_tok, prob_prof_wrong = predict_next_token(
            prompt_with_wrong_article, tokenizer, model
        )
        pred_profession_wrong = pred_prof_wrong_tok.strip().lower()
        if " " in pred_profession_wrong:
            pred_profession_wrong = pred_profession_wrong.split(" ")[0]

        records.append(
            {
                "Profession": profession,
                "Article": article,
                "Predicted_Article": pred_article,
                "Article_Correct": pred_article == article.lower(),
                "Predicted_Profession": pred_profession,
                "Profession_Correct": profession.lower().startswith(pred_profession),
                "Predicted_Profession_Wrong_Article": pred_profession_wrong,
                "Profession_Wrong_Article_Correct": profession.lower().startswith(pred_profession_wrong),
                "Prob_Article": prob_article,
                "Prob_Profession": prob_prof,
                "Prob_Profession_Wrong_Article": prob_prof_wrong,
                "IC_Profession": ic_row["Profession"].strip(),
                "IC_Article": ic_row["Article"].strip(),
                "IC_Description": ic_description,
            }
        )

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Compute metrics
    article_acc = out_df["Article_Correct"].mean()
    profession_acc = out_df["Profession_Correct"].mean()
    profession_wrong_acc = out_df["Profession_Wrong_Article_Correct"].mean()

    per_article = (
        out_df.groupby("Article")["Article_Correct"].mean().to_dict()
    )

    print(f"==== {model_name} Accuracy ====")
    print(f"Article accuracy: {article_acc:.3%}")
    for art, acc in per_article.items():
        print(f"  {art}: {acc:.3%}")
    print(f"Profession accuracy (correct article): {profession_acc:.3%}")
    print(f"Profession accuracy (wrong article): {profession_wrong_acc:.3%}")

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