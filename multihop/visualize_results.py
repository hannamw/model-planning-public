#%%
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Base directory for input/output
base_dir = "results/behavioral"

def load_and_process_results(results_dir):
    results = []
    model_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    # Dictionary to store counts
    prompt_type_counts = {}
    
    for file_path in model_files:
        model_name = os.path.basename(file_path).replace(".csv", "")
        df = pd.read_csv(file_path)
        
        # Get counts for each prompt type if not already calculated
        if not prompt_type_counts:
            prompt_type_counts = df.groupby("prompt_type").size().to_dict()
        
        # Group by prompt_type and calculate accuracies
        grouped = df.groupby("prompt_type").agg({
            "exact_match": "mean",
            "contains_answer": "mean"
        }).reset_index()
        
        for _, row in grouped.iterrows():
            results.append({
                "Model": model_name,
                "Prompt Type": row["prompt_type"],
                "Exact Match": row["exact_match"] * 100,
                "Contains Answer": row["contains_answer"] * 100,
                "Count": prompt_type_counts[row["prompt_type"]]
            })
    
    return pd.DataFrame(results)

def plot_accuracies(df, output_dir):
    # Sort models by size - specifically for Qwen3-{size}B format
    df = df.copy()  # Create a copy to avoid modifying the original
    df["Size"] = df["Model"].str.extract(r"Qwen3-(\d+\.?\d*)B").astype(float)
    df = df.sort_values(["Size", "Prompt Type"])
    
    # Get unique categories
    prompt_types = df["Prompt Type"].unique()
    models = df.sort_values("Size")["Model"].unique()
    
    # Plot settings
    width = 0.8 / len(prompt_types)
    x = range(len(models))
    
    # Plot Exact Match
    plt.figure(figsize=(12, 6))
    for i, ptype in enumerate(prompt_types):
        mask = df["Prompt Type"] == ptype
        count = df[mask]["Count"].iloc[0]  # Get count for this prompt type
        plt.bar([j + (i - len(prompt_types)/2 + 0.5) * width for j in x], 
                df[mask]["Exact Match"], width, 
                label=f"{ptype} (n={count})")
    
    plt.ylabel("Exact Match Accuracy (%)")
    plt.title("Model Performance - Exact Match by Category")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend(title="Prompt Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exact_match_accuracies.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot Contains Answer
    plt.figure(figsize=(12, 6))
    for i, ptype in enumerate(prompt_types):
        mask = df["Prompt Type"] == ptype
        count = df[mask]["Count"].iloc[0]  # Get count for this prompt type
        plt.bar([j + (i - len(prompt_types)/2 + 0.5) * width for j in x], 
                df[mask]["Contains Answer"], width, 
                label=f"{ptype} (n={count})")
    
    plt.xlabel("Model")
    plt.ylabel("Contains Answer Accuracy (%)")
    plt.title("Model Performance - Contains Answer by Category")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend(title="Prompt Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contains_answer_accuracies.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_accuracies_line(df, output_dir):
    # Sort models by size - specifically for Qwen3-{size}B format
    df = df.copy()  # Create a copy to avoid modifying the original
    df["Size"] = df["Model"].str.extract(r"Qwen3-(\d+\.?\d*)B").astype(float)
    df = df.sort_values(["Size", "Prompt Type"])
    
    # Get unique categories
    prompt_types = df["Prompt Type"].unique()
    models = df.sort_values("Size")["Model"].unique()
    
    # Plot Exact Match
    plt.figure(figsize=(12, 6))
    for ptype in prompt_types:
        mask = df["Prompt Type"] == ptype
        count = df[mask]["Count"].iloc[0]
        plt.plot(models, df[mask]["Exact Match"], marker='o', label=f"{ptype} (n={count})")
    
    plt.ylabel("Exact Match Accuracy (%)")
    plt.title("Model Performance - Exact Match by Category")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prompt Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exact_match_accuracies_line.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot Contains Answer
    plt.figure(figsize=(12, 6))
    for ptype in prompt_types:
        mask = df["Prompt Type"] == ptype
        count = df[mask]["Count"].iloc[0]
        plt.plot(models, df[mask]["Contains Answer"], marker='o', label=f"{ptype} (n={count})")
    
    plt.xlabel("Model")
    plt.ylabel("Contains Answer Accuracy (%)")
    plt.title("Model Performance - Contains Answer by Category")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prompt Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contains_answer_accuracies_line.png"), dpi=300, bbox_inches="tight")
    plt.close()

# Use base_dir for both input and output
df = load_and_process_results(base_dir)
plot_accuracies(df, base_dir)
plot_accuracies_line(df, base_dir)
print(f"\nResults saved in {base_dir}/:")
print("- exact_match_accuracies.png")
print("- contains_answer_accuracies.png")
print("- exact_match_accuracies_line.png")
print("- contains_answer_accuracies_line.png")
# %%
