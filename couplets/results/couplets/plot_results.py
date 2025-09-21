#%%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Folder that holds all model-specific CSVs
DATA_DIR = "."

accuracies = {}

for csv_path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    model_name = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)

    # Make sure the Rhymes column is boolean
    rhymes_col = df["rhyme_success"]
    if rhymes_col.dtype != bool:
        rhymes_col = rhymes_col.astype(str).str.lower() == "true"

    accuracies[model_name] = rhymes_col.mean()

# --- Plotting ---------------------------------------------------------------
models   = list(accuracies.keys())
scores   = [accuracies[m] for m in models]

plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 14})
bars = plt.bar(models, scores, color="steelblue", edgecolor="black")

# Add value labels above the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.2%}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

plt.ylabel("Rhyming Accuracy", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.title("Poem Rhyming Accuracy per Model", fontsize=18)
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("rhyming_accuracy.pdf")
plt.show()
# %%
