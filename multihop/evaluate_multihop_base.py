#%%
import argparse 
from typing import List, Optional
from tqdm import tqdm
import unicodedata
import re
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset



def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

class MultihopDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, prompt_template: str):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = self.prompt_template.format(question=row['question'])
        return {
            'prompt': prompt,
            'answer': row['answer'],
            'intermediate': row['intermediate'],
            'prompt_type': row['prompt_type'],
            'question': row['question']
        }


def collate_fn(batch):
    return batch


def evaluate_batch(model: AutoModelForCausalLM, tokenizer, batch, device, max_new_tokens=50):
    prompts = [item['prompt'] for item in batch]
    
    # Tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated_texts = []
    for i, output in enumerate(outputs):
        input_length = inputs['input_ids'][i].shape[0]
        generated = output[input_length:]
        generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        generated_texts.append(generated_text)
    
    return generated_texts


def normalize_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove periods, newlines, and extra spaces
    text = re.sub(r'[.\n]', '', text)
    # Remove accents and normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def exact_match_score(predicted: str, target: str) -> bool:
    return normalize_text(predicted) == normalize_text(target)

def contains_answer_score(predicted: str, target: str) -> bool:
    return normalize_text(target) in normalize_text(predicted)


def load_dataset(dataset_path: str, prompt_type_filter: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    
    if prompt_type_filter:
        df = df[df['prompt_type'] == prompt_type_filter]
        print(f"Filtered dataset to {len(df)} examples with prompt_type='{prompt_type_filter}'")
    
    return df


def evaluate_model(model_name: str, dataset_path: str, prompt_template: str, batch_size: int = 8, 
                  prompt_type_filter: Optional[str] = None, max_new_tokens: int = 50):
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.generation_config.temperature = None
    model.generation_config.top_p = None  
    model.generation_config.top_k = None

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = next(model.parameters()).device
    
    # Load dataset
    df = load_dataset(dataset_path, prompt_type_filter)
    dataset = MultihopDataset(df, tokenizer, chattify(prompt_template, tokenizer))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Evaluation
    all_results = []
    
    print(f"Evaluating on {len(dataset)} examples...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        generated_texts = evaluate_batch(model, tokenizer, batch, device, max_new_tokens)
        
        for item, generated in zip(batch, generated_texts):
            exact_match = exact_match_score(generated, item['answer'])
            contains_answer = contains_answer_score(generated, item['answer'])
            
            result = {
                'question': item['question'],
                'generated': generated,
                'answer': item['answer'],
                'intermediate': item['intermediate'],
                'prompt_type': item['prompt_type'],
                'exact_match': exact_match,
                'contains_answer': contains_answer
            }
            all_results.append(result)
    
    return pd.DataFrame(all_results)


def display_performance(results_df: pd.DataFrame):
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE")
    print("="*50)
    
    exact_match_rate = results_df['exact_match'].mean()
    contains_answer_rate = results_df['contains_answer'].mean()
    
    print(f"Total examples: {len(results_df)}")
    print(f"Exact match accuracy: {exact_match_rate:.3f} ({exact_match_rate*100:.1f}%)")
    print(f"Contains answer accuracy: {contains_answer_rate:.3f} ({contains_answer_rate*100:.1f}%)")
    
    print("\n" + "="*50)
    print("PERFORMANCE BY PROMPT TYPE")
    print("="*50)
    
    for prompt_type in results_df['prompt_type'].unique():
        subset = results_df[results_df['prompt_type'] == prompt_type]
        exact_match = subset['exact_match'].mean()
        contains_answer = subset['contains_answer'].mean()
        
        print(f"\nPrompt type: {prompt_type}")
        print(f"  Examples: {len(subset)}")
        print(f"  Exact match: {exact_match:.3f} ({exact_match*100:.1f}%)")
        print(f"  Contains answer: {contains_answer:.3f} ({contains_answer*100:.1f}%)")


if __name__ == "__main__":
    # model = "Qwen/Qwen3-4B"
    # dataset_path = "/root/model-planning/multihop/data/combined_multihop_dataset.csv"
    # batch_size=32
    # prompt_type_filter='country_capital'
    # max_new_tokens = 5
    # output_dir = "results/multihop_results.csv"

    parser = argparse.ArgumentParser(description="Evaluate Qwen3 model on multihop dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="data/combined_multihop_dataset.csv", 
                       help="Path to dataset CSV")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--prompt_type_filter", type=str, help="Filter dataset by prompt type")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output_dir", type=str, default='results/behavioral', help="Output directory for results CSV")
    
    args = parser.parse_args()

    prompt_template = ["/no_think Answer the following question in one word. Q: {question}", "<think>\n\n</think>\n\nA:"]
    
    # Evaluate model
    # results_df = evaluate_model(
    #     model_name=model,
    #     dataset_path=dataset_path,
    #     prompt_template=prompt_template,
    #     batch_size=batch_size,
    #     prompt_type_filter=prompt_type_filter,
    #     max_new_tokens=max_new_tokens,
    # )

    model_name = args.model.split('/')[-1] if '/' in args.model else args.model

    # Evaluate model
    results_df = evaluate_model(
        model_name=args.model,
        dataset_path=args.dataset,
        prompt_template=prompt_template,
        batch_size=args.batch_size,
        prompt_type_filter=args.prompt_type_filter,
        max_new_tokens=args.max_new_tokens,
    )

    output_dir = args.output_dir
    
    # Display performance
    display_performance(results_df)
    
    # Save results if output directory is specified
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

# %%
