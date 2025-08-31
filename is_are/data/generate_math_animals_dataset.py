#%%
#!/usr/bin/env python3
"""
Create a toy subtraction dataset for LLM evaluation (Pandas version).

Columns produced
----------------
prompt : str      – "At first … Now, there"
animal : str      – the plural animal name used in the prompt
original : int    – the original number of animals (i)
subtracted : int  – the number of animals that went away (j)
number : int      – i − j (2-9 minus 1 ... i-1)
answer : str      – 'is'  if number == 1
                    'are' otherwise

Example usage
-------------
python generate_animals_math_dataset_pandas.py                # → stdout
python generate_animals_math_dataset_pandas.py -o data.csv    # → file
python generate_animals_math_dataset_pandas.py --seed 123     # → different animal allocation
"""

import argparse
import pandas as pd

ANIMALS = [
    "cats", "dogs", "rabbits", "elephants", "lions",
    "tigers", "giraffes", "cows", "horses", "sheep",
]


def build_dataset() -> pd.DataFrame:
    rows = []

    for animal in ANIMALS:
        for i in range(2, 10):          # i = 2 … 9
            for j in range(1, i):       # j = 1 … i-1
                number = i - j
                rows.append(
                    {
                        "prompt": (
                            f"At first there were {i} {animal}. "
                            f"Then, {j} went away. Now, there"
                        ),
                        "animal": animal,
                        "original": i,
                        "subtracted": j,
                        "number": number,
                        "answer": "is" if number == 1 else "are",
                    }
                )

    return pd.DataFrame(rows)

output = 'animals_dataset.csv'
df = build_dataset()
df.to_csv(output, index=False)

# %%
