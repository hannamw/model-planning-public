#%%
import requests

# Get words that rhyme with "hello" using Datamuse API
url = "https://api.datamuse.com/words"

#%%
word = 'embrace'
max_results = 1000
params = {'rel_rhy': word, 'max': max_results}

response = requests.get(url, params=params)
rhymes = response.json()

print("Words that rhyme with 'hello':")
for rhyme in rhymes:
    print(rhyme['word'])
# %%
from feature_utils import get_feature
#%%
feature = get_feature(transcoder_id='Qwen3-1.7B/Qwen3-1.7b-relu-lowl0-5', feature_idx=22688)
# %%
'accountant' in ''.join(feature['examples_quantiles'][0]['examples'][0]['tokens']).lower()
# %%
