#%%
from collections import defaultdict
import random

import pandas as pd

counter = 0
eqs = set()
results = defaultdict(list)
operators = ['+', '-', '*', '//']
while counter < 10000:
    op1 = random.choice(operators)
    op2 = random.choice(operators)

    n1 = random.randint(1, 9)
    n2 = random.randint(1, 9)
    n3 = random.randint(1, 9)

    intermediate_str = f'({n1}{op1}{n2})'
    intermediate = eval(intermediate_str)
    answer_str = f'{intermediate}{op2}{n3}'
    answer = eval(answer_str)


    eqfull = f'{intermediate_str}{op2}{n3}='
    if not eqfull in eqs:
        eqs.add(eqfull)
        results['op1'].append(op1)
        results['op2'].append(op2)
        results['n1'].append(n1)
        results['n2'].append(n2)
        results['n3'].append(n3)
        results['intermediate'].append(intermediate)
        results['answer'].append(answer)
        results['eq1'].append(intermediate_str)
        results['eqfull'].append(eqfull)
        counter += 1

df = pd.DataFrame(results)
df.to_csv('pre-data/math_data_novar.csv', index=False)
# %%
