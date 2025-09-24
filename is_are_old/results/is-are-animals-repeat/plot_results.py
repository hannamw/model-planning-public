#%%
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "."
csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))

verb_recalls = {"model": [], "is_recall": [], "are_recall": []}
num_acc      = {}

for path in csv_paths:
    model = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)

    # Boolean masks for the gold verb
    is_mask  = df["Gold_Verb"].str.lower() == "is"
    are_mask = df["Gold_Verb"].str.lower() == "are"

    # Recall = correct / all with that gold answer
    is_recall  = df.loc[is_mask,  "Verb_Correct"].mean()
    are_recall = df.loc[are_mask, "Verb_Correct"].mean()

    verb_recalls["model"].append(model)
    verb_recalls["is_recall"].append(is_recall)
    verb_recalls["are_recall"].append(are_recall)

    # Number accuracy (regardless of verb)
    num_acc[model] = df["Number_Correct"].mean()

# ---------------- PLOT 1 : verb recall -----------------
df_v = pd.DataFrame(verb_recalls).set_index("model")
x = range(len(df_v))

bar_w = 0.35
plt.figure(figsize=(10,5))
plt.bar([i - bar_w/2 for i in x], df_v["is_recall"],  width=bar_w, label="Gold verb = is")
plt.bar([i + bar_w/2 for i in x], df_v["are_recall"], width=bar_w, label="Gold verb = are")

plt.xticks(x, df_v.index, rotation=45, ha="right")
plt.ylabel("Recall (accuracy)")
plt.ylim(0,1.05)
plt.title("Verb recall on is/are-animals task")
plt.legend()
plt.tight_layout()
plt.savefig('is_are_recall.png')
plt.show()

# ---------------- PLOT 2 : number accuracy -------------
plt.figure(figsize=(10,4))
plt.bar(num_acc.keys(), num_acc.values(), color="steelblue", edgecolor="black")
for x0, v in enumerate(num_acc.values()):
    plt.text(x0, v+0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=9)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.ylim(0,1.05)
plt.title("Number-prediction accuracy")
plt.tight_layout()
plt.savefig('number_accuracy.png')
plt.show()
# %%
