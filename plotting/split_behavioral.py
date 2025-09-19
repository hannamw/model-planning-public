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

# Load both datasets
article_data = load_recall_data(
    "../a_an/results/a-an-IC-random", models, 
    "Article", "Article_Correct", "a", "an"
)

verb_data = load_recall_data(
    "../is_are/results/is-are-animals-repeat", models,
    "Gold_Verb", "Verb_Correct", "is", "are"
)

x = range(len(models))

#%%
# Plot 1: Article recall (a/an) only
plt.figure(figsize=(12, 5))

plt.plot(x, article_data["a_recall"], marker='o', linewidth=2, markersize=8,
         label='Article: "a"', color='#1f77b4', linestyle='-')
plt.plot(x, article_data["an_recall"], marker='o', linewidth=2, markersize=8,
         label='Article: "an"', color='#4d94d4', linestyle='--')

plt.xticks(x, models, rotation=0, ha="center", fontsize=16)
plt.ylabel("Recall", fontsize=18)
plt.ylim(0, 1.05)
plt.yticks(fontsize=14)
plt.title("Article Recall Across Model Sizes (a/an)", fontsize=20)
plt.legend(fontsize=14, numpoints=1, handlelength=4, handletextpad=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("article_behavioral_recall.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot 2: Verb recall (is/are) only
plt.figure(figsize=(12, 5))

plt.plot(x, verb_data["is_recall"], marker='o', linewidth=2, markersize=8,
         label='Verb: "is"', color='#d62728', linestyle='--')
plt.plot(x, verb_data["are_recall"], marker='o', linewidth=2, markersize=8,
         label='Verb: "are"', color='#ff6b6b', linestyle='-')

plt.xticks(x, models, rotation=0, ha="center", fontsize=16)
plt.ylabel("Recall", fontsize=18)
plt.ylim(0, 1.05)
plt.yticks(fontsize=14)
plt.title("Verb Recall Across Model Sizes (is/are)", fontsize=20)
plt.legend(fontsize=14, numpoints=1, handlelength=4, handletextpad=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("verb_behavioral_recall.pdf", dpi=300, bbox_inches='tight')
plt.show()
# %%