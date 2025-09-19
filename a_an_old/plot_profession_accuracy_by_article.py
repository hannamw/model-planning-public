
#%%
# #!/usr/bin/env python3

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_analyze_data(data_dir: str = "results/a-an-IC-random"):
    """Load CSV files and calculate profession accuracy metrics."""
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    
    results = {
        "model": [],
        "mean_profession_correct": [],
        "mean_profession_wrong_article_correct": []
    }
    
    # Model names in the order we want them displayed (from the run script)
    model_order = [f'Qwen3-{size}B' for size in ['0.6', '1.7', '4', '8', '14', '32']]
    
    for path in csv_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        
        # Calculate mean of both metrics
        mean_prof_correct = df["Profession_Correct"].mean()
        mean_prof_wrong_article_correct = df["Profession_Wrong_Article_Correct"].mean()
        
        results["model"].append(model_name)
        results["mean_profession_correct"].append(mean_prof_correct)
        results["mean_profession_wrong_article_correct"].append(mean_prof_wrong_article_correct)
    
    # Convert to DataFrame and sort by model order
    df_results = pd.DataFrame(results)
    
    # Create a mapping for sorting
    model_order_map = {model: i for i, model in enumerate(model_order)}
    df_results['sort_order'] = df_results['model'].map(model_order_map)
    df_results = df_results.sort_values('sort_order').drop('sort_order', axis=1)
    
    return df_results

def create_profession_accuracy_plot(df_results: pd.DataFrame, output_path: str = "results/a-an-IC-random/profession_accuracy_by_article.pdf"):
    """Create a grouped bar plot showing mean profession accuracy metrics."""
    
    models = df_results['model'].tolist()
    prof_correct = df_results['mean_profession_correct'].tolist()
    prof_wrong_article_correct = df_results['mean_profession_wrong_article_correct'].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, prof_correct, width, 
                   label='Profession Accuracy (Correct Article)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, prof_wrong_article_correct, width,
                   label='Profession Accuracy (Inorrect Article)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Profession Accuracy', fontsize=14)
    ax.set_title('Profession Accuracy by Model and Article Correctness', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 1.05)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_article_specific_plot(data_dir: str = "results/a-an-IC-random"):
    """Create merged plot for 'a' and 'an' cases using Profession_Wrong_Article_Correct."""
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Model names in the order we want them displayed
    model_order = [f'Qwen3-{size}B' for size in ['0.6', '1.7', '4', '8', '14', '32']]
    
    # Combined results
    results = {
        "model": [],
        "a_mean": [],
        "an_mean": []
    }
    
    for path in csv_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        
        # Filter for 'a' cases and calculate mean
        a_df = df[df['Article'] == 'a']
        a_mean = a_df['Profession_Wrong_Article_Correct'].mean() if len(a_df) > 0 else 0
        
        # Filter for 'an' cases and calculate mean
        an_df = df[df['Article'] == 'an']
        an_mean = an_df['Profession_Wrong_Article_Correct'].mean() if len(an_df) > 0 else 0
        
        results["model"].append(model_name)
        results["a_mean"].append(a_mean)
        results["an_mean"].append(an_mean)
    
    # Convert to DataFrame and sort
    df_results = pd.DataFrame(results)
    model_order_map = {model: i for i, model in enumerate(model_order)}
    df_results['sort_order'] = df_results['model'].map(model_order_map)
    df_results = df_results.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Create merged plot
    create_merged_article_plot(df_results, "results/a-an-IC-random/profession_wrong_article_correct_by_article.pdf")
    
    return df_results

def create_merged_article_plot(df_results: pd.DataFrame, output_path: str):
    """Create a grouped bar plot for 'a' vs 'an' cases."""
    
    models = df_results['model'].tolist()
    a_values = df_results['a_mean'].tolist()
    an_values = df_results['an_mean'].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use different colors from main plot (main uses steelblue and coral)
    bars1 = ax.bar(x - width/2, a_values, width, 
                   label="Correct article = a", color='forestgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, an_values, width,
                   label="Correct article = an", color='darkorange', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Mean Profession Accuracy', fontsize=14)
    ax.set_title('Mean Profession Accuracy by Correct Article\nGiven Incorrect Article', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 1.05)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_summary_stats(df_results: pd.DataFrame):
    """Print summary statistics."""
    print("=== Profession Accuracy Summary ===")
    print(f"{'Model':<12} {'Prof_Correct':<15} {'Prof_Wrong_Art_Correct':<20} {'Difference':<10}")
    print("-" * 65)
    
    for _, row in df_results.iterrows():
        model = row['model']
        prof_correct = row['mean_profession_correct']
        prof_wrong_art_correct = row['mean_profession_wrong_article_correct']
        diff = prof_correct - prof_wrong_art_correct
        print(f"{model:<12} {prof_correct:<15.2%} {prof_wrong_art_correct:<20.2%} {diff:<10.2%}")
    
    # Overall averages
    avg_prof_correct = df_results['mean_profession_correct'].mean()
    avg_prof_wrong_art_correct = df_results['mean_profession_wrong_article_correct'].mean()
    avg_diff = avg_prof_correct - avg_prof_wrong_art_correct
    
    print("-" * 65)
    print(f"{'Average':<12} {avg_prof_correct:<15.2%} {avg_prof_wrong_art_correct:<20.2%} {avg_diff:<10.2%}")

if __name__ == "__main__":
    # Load and analyze data
    df_results = load_and_analyze_data()
    
    # Print summary statistics
    print_summary_stats(df_results)
    
    # Create original plot
    fig = create_profession_accuracy_plot(df_results)
    
    # Create merged article-specific plot
    article_df_results = create_article_specific_plot()
    

# %%
