#%%
import random

import pandas as pd

results = {k:[] for k in ['n1', 'n2', 'intermediate', 'answer', 'eqfull']}
for n1 in range(10, 100):
    for n2 in range(10, 100):
        eqfull = f'{n1} + {n2} ='
        intermediate = (n1 % 10) + (n2 % 10)
        answer = n1 + n2
        results['n1'].append(n1)
        results['n2'].append(n2)
        results['intermediate'].append(intermediate)
        results['answer'].append(answer)
        results['eqfull'].append(eqfull)
 
df = pd.DataFrame(results)
df.to_csv('pre-data/two_digit_addition_data.csv', index=False)
# %%
