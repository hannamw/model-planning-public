#%%
#!/usr/bin/env python3
"""
Plot intervention results for TinyStories models.

This script analyzes the output CSV files from intervention experiments and computes:
1. Exact match success rate: detected_animal == target_animal
2. Change success rate: detected_animal != source_animal

Creates visualizations showing success rates per model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def load_intervention_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load specific intervention result CSV files in the correct order."""
    # Define the specific models we want in the correct order
    model_files = [
        "Qwen3-0.6B.csv",
        "Qwen3-1.7B.csv", 
        "Qwen3-4B.csv",
        "Qwen3-8B.csv",
        "Qwen3-14B.csv",
        "Meta-Llama-3-8B-Instruct.csv"
    ]
    
    results = {}
    
    for csv_filename in model_files:
        csv_file = results_dir / csv_filename
        if not csv_file.exists():
            print(f"Warning: {csv_filename} not found, skipping...")
            continue
            
        model_name = csv_file.stem
        # Clean up the model name for display
        display_name = model_name.replace("Meta-", "") if "Meta-" in model_name else model_name
        
        print(f"Loading results for {display_name}...")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"  Loaded {len(df)} intervention pairs")
            results[display_name] = df
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
            continue
    
    return results


def compute_success_rates(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute success rates for each model.
    
    Returns DataFrame with columns:
    - model: model name
    - total_pairs: total number of intervention pairs
    - exact_match_success: detected_animal == target_animal
    - exact_match_rate: exact match success rate
    - change_success: detected_animal != source_animal
    - change_rate: change success rate
    """
    summary_data = []
    
    for model_name, df in results.items():
        # Clean data - remove rows with missing detected_animal
        df_clean = df.dropna(subset=['detected_animal']).copy()
        
        # Convert to lowercase for comparison
        df_clean['detected_animal_lower'] = df_clean['detected_animal'].str.lower()
        df_clean['source_animal_lower'] = df_clean['source_animal'].str.lower()
        df_clean['target_animal_lower'] = df_clean['target_animal'].str.lower()
        
        total_pairs = len(df_clean)
        
        # Exact match success: detected_animal == target_animal
        exact_matches = (df_clean['detected_animal_lower'] == df_clean['target_animal_lower']).sum()
        exact_match_rate = exact_matches / total_pairs if total_pairs > 0 else 0
        
        # Change success: detected_animal != source_animal
        changes = (df_clean['detected_animal_lower'] != df_clean['source_animal_lower']).sum()
        change_rate = changes / total_pairs if total_pairs > 0 else 0
        
        summary_data.append({
            'model': model_name,
            'total_pairs': total_pairs,
            'exact_match_success': exact_matches,
            'exact_match_rate': exact_match_rate,
            'change_success': changes,
            'change_rate': change_rate
        })
    
    return pd.DataFrame(summary_data)





def create_comparison_plot(summary_df: pd.DataFrame, output_dir: Path):
    """Create a comparison plot showing both success metrics side by side."""
    # Keep the original order from the data loading (don't sort)
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create grouped bar chart
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, summary_df['exact_match_rate'], width, 
                   label='Exact Match (detected == target)', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, summary_df['change_rate'], width,
                   label='Any Change (detected ≠ source)', color='darkorange', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=16)
    ax.set_ylabel('Success Rate', fontsize=16)
    ax.set_title('Intervention Success Rates by Model', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['model'], rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=14)
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "intervention_success_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    
    plt.show()


def print_summary_statistics(summary_df: pd.DataFrame):
    """Print detailed summary statistics."""
    print("\n" + "="*80)
    print("INTERVENTION RESULTS SUMMARY")
    print("="*80)
    
    print("\nOverall Success Rates by Model:")
    print("-" * 50)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<25} | "
              f"Exact: {row['exact_match_rate']:.1%} ({row['exact_match_success']:,}/{row['total_pairs']:,}) | "
              f"Change: {row['change_rate']:.1%} ({row['change_success']:,}/{row['total_pairs']:,})")
    
    print(f"\nAverages across models:")
    print(f"  Exact Match: {summary_df['exact_match_rate'].mean():.1%}")
    print(f"  Any Change:  {summary_df['change_rate'].mean():.1%}")
    
    print(f"\nBest performing models:")
    best_exact = summary_df.loc[summary_df['exact_match_rate'].idxmax()]
    best_change = summary_df.loc[summary_df['change_rate'].idxmax()]
    print(f"  Exact Match: {best_exact['model']} ({best_exact['exact_match_rate']:.1%})")
    print(f"  Any Change:  {best_change['model']} ({best_change['change_rate']:.1%})")
    



#%%
results_dir = Path(".")
output_dir = Path(".")

print("Loading intervention results...")
results = load_intervention_results(results_dir)

if not results:
    print("No result files found!")

print(f"\nLoaded results for {len(results)} models")

# Compute success rates
print("\nComputing success rates...")
summary_df = compute_success_rates(results)

# Create plot
print("\nCreating visualization...")
create_comparison_plot(summary_df, output_dir)

# Print summary statistics
print_summary_statistics(summary_df)

# Save summary data
summary_file = output_dir / "intervention_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\nSummary data saved to: {summary_file}")

print("\nAnalysis complete!")
