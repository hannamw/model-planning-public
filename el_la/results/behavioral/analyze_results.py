#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14,32]]
article_correct = []
noun_correct = []

for model in models:
    df = pd.read_csv(f'{model}.csv')
    article_correct.append(df['article_correct'].mean())
    noun_correct.append(df['noun_correct'].mean())
    print(model, df['article_correct'].mean(), df['noun_correct'].mean())

#%%
plt.figure(figsize=(10, 6))
plt.plot(models, article_correct, color='blue', linewidth=2, label='Article Correct')
plt.plot(models, noun_correct, color='red', linewidth=2, label='Noun Correct')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Article and Noun Prediction Accuracy Across Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

