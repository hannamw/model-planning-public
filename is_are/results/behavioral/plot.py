#%%
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_recall_data(data_dir, models, gold_col, correct_col, label1, label2):
    """Load and process recall data for binary classification tasks."""
    csv_paths = [os.path.join(data_dir, model + '.csv') for model in models]
    recalls = {"model": [], f"{label1}_recall": [], f"{label2}_recall": []}
    
    for path in csv_paths:
        model = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        
        mask1 = df[gold_col].str.lower() == label1
        mask2 = df[gold_col].str.lower() == label2
        
        recall1 = df.loc[mask1, correct_col].mean()
        recall2 = df.loc[mask2, correct_col].mean()
        
        recalls["model"].append(model)
        recalls[f"{label1}_recall"].append(recall1)
        recalls[f"{label2}_recall"].append(recall2)
    
    return pd.DataFrame(recalls).set_index("model")

models = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']

# Load is/are verb data from current directory
verb_data = load_recall_data(
    ".", models,
    "gold_verb", "verb_correct", "is", "are"
)

# Create single plot for is/are data
plt.figure(figsize=(12, 6))

x = range(len(models))

plt.plot(x, verb_data["is_recall"], marker='o', linewidth=2, markersize=8,
         label='Gold verb = "is"', color='#d62728')
plt.plot(x, verb_data["are_recall"], marker='s', linewidth=2, markersize=8,
         label='Gold verb = "are"', color='#ff6b6b')

plt.xticks(x, models, rotation=45, ha="right", fontsize=16)
plt.ylabel("Recall (accuracy)", fontsize=18)
plt.ylim(0, 1.05)
plt.yticks(fontsize=14)
plt.title("Verb Recall (is/are task)", fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("is_are_behavioral_recall.pdf", dpi=300, bbox_inches='tight')
plt.show()
# %%
