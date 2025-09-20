#%%
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "."
csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# Hold per-model results
article_recalls = {"model": [], "a_recall": [], "an_recall": []}
prof_acc        = {}

for path in csv_paths:
    model = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)

    # Masks for each gold article
    a_mask  = df["Article"].str.lower() == "a"
    an_mask = df["Article"].str.lower() == "an"

    # Recall = #correct / #rows with that gold answer
    a_recall  = df.loc[a_mask,  "Article_Correct"].mean()
    an_recall = df.loc[an_mask, "Article_Correct"].mean()

    article_recalls["model"].append(model)
    article_recalls["a_recall"].append(a_recall)
    article_recalls["an_recall"].append(an_recall)

    # Profession accuracy
    prof_acc[model] = df["Profession_Correct"].mean()

# --------------- PLOT 1 : article recall -----------------
df_art = pd.DataFrame(article_recalls).set_index("model")
x = range(len(df_art))
w = 0.35

plt.figure(figsize=(10,5))
plt.bar([i - w/2 for i in x], df_art["a_recall"],  width=w, label='Gold article = "a"')
plt.bar([i + w/2 for i in x], df_art["an_recall"], width=w, label='Gold article = "an"')

plt.xticks(x, df_art.index, rotation=45, ha="right")
plt.ylabel("Recall (accuracy)")
plt.ylim(0, 1.05)
plt.title("Article recall on a/an task")
plt.legend()
plt.tight_layout()
plt.savefig("a_an_recall.png")
plt.show()

# --------------- PLOT 2 : profession accuracy ------------
plt.figure(figsize=(10,4))
plt.bar(prof_acc.keys(), prof_acc.values(), color="steelblue", edgecolor="black")

for xi, v in enumerate(prof_acc.values()):
    plt.text(xi, v + 0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=9)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Profession-name accuracy")
plt.tight_layout()
plt.savefig("profession_recall.png")
plt.show()

# %%
