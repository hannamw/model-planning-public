import random

import pandas as pd

def pick_in_context_index(cur_idx: int, num_rows: int, rng: random.Random) -> int:
    """Return an index different from cur_idx."""
    idx = cur_idx
    while idx == cur_idx:
        idx = rng.randint(0, num_rows - 1)
    return idx

def create_dataset_examples(df: pd.DataFrame, seed: int = 42):
    """
    Create prompts from the professions dataset and add them as a new column.
    
    For each profession, creates a prompt with:
    1. An in-context example (full description from another profession)
    2. The current profession's description with article + profession stripped off
    
    Returns:
        Modified dataframe with a new 'Prompt' column
    """
    rng = random.Random(seed)
    
    # Create a copy of the dataframe to avoid modifying the original
    df_with_prompts = df.copy()
    prompts = []
    
    for idx, row in df.iterrows():
        profession: str = row["Profession"].strip()
        article: str = row["Article"].strip()
        description: str = row["Description"].strip()
        
        # Build trimmed context (remove article + profession)
        tokens = description.split()
        
        # Expect last two tokens are article and profession
        if len(tokens) < 2 or tokens[-1].lower() != profession.lower():
            raise ValueError(f"Description/token mismatch at index {idx}")
        if tokens[-2].lower() != article.lower():
            raise ValueError(f"Article mismatch at index {idx}")
        
        # Context before article (up to 'is')
        context_before_article = " ".join(tokens[:-2])
        
        # Choose in-context example (different from current)
        ic_idx = pick_in_context_index(idx, len(df), rng)
        ic_row = df.iloc[ic_idx]
        ic_description = ic_row["Description"].strip()
        
        # Create final prompt: [IC example]. [context before article]
        prompt = f"{ic_description}. {context_before_article}"
        prompts.append(prompt)
    
    # Add the prompts as a new column
    df_with_prompts['Prompt'] = prompts
    
    return df_with_prompts