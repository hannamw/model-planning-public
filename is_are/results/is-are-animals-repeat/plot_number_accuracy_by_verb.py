#%%
#!/usr/bin/env python3
"""
Plot number accuracy when using correct vs wrong verb.
Shows how verb correctness affects number prediction accuracy across different models.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Find all CSV files in the current directory
    csv_paths = glob.glob("*.csv")
    
    if not csv_paths:
        print("No CSV files found in current directory")
        return
    
    # Storage for results
    results = []
    
    for path in csv_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        
        # Calculate number accuracy for correct and wrong verb usage
        # Now both predictions are in the same row
        num_acc_correct = df["Number_Correct"].mean()
        num_acc_wrong = df["Number_Correct_With_Wrong_Verb"].mean()
        
        results.append({
            "model": model_name,
            "correct_verb": num_acc_correct,
            "wrong_verb": num_acc_wrong
        })
    
    # Convert to DataFrame for easier plotting
    results_df = pd.DataFrame(results)
    
    # Sort by model size (extract size from model name)
    def extract_size(model_name):
        # Extract the size part (e.g., "0.6B" from "Qwen3-0.6B")
        if "-" in model_name:
            size_str = model_name.split("-")[1]
            # Convert to float for proper sorting (e.g., "0.6B" -> 0.6)
            if size_str.endswith("B"):
                return float(size_str[:-1])
        return 0
    
    results_df["size"] = results_df["model"].apply(extract_size)
    results_df = results_df.sort_values("size").drop("size", axis=1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, results_df["correct_verb"], width, 
                   label="Correct Verb", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df["wrong_verb"], width,
                   label="Wrong Verb", color="crimson", alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Number Prediction Accuracy", fontsize=14)
    ax.set_title("Number Prediction Accuracy: Correct vs Wrong Verb Usage", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(bbox_to_anchor=(1.0, -0.15), ncol=2, loc='lower right', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('number_accuracy_by_verb.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NUMBER PREDICTION ACCURACY BY VERB CORRECTNESS")
    print("="*60)
    
    for _, row in results_df.iterrows():
        model = row["model"]
        correct = row["correct_verb"]
        wrong = row["wrong_verb"]
        diff = correct - wrong
        print(f"{model:15} | Correct: {correct:6.2%} | Wrong: {wrong:6.2%} | Diff: {diff:+6.2%}")
    
    # Overall statistics
    avg_correct = results_df["correct_verb"].mean()
    avg_wrong = results_df["wrong_verb"].mean()
    avg_diff = avg_correct - avg_wrong
    
    print("-" * 60)
    print(f"{'Average':15} | Correct: {avg_correct:6.2%} | Wrong: {avg_wrong:6.2%} | Diff: {avg_diff:+6.2%}")

if __name__ == "__main__":
    main()
# %%
