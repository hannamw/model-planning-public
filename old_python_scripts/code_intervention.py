#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk

# Helper function to strip the model's internal "thinking" preamble before displaying.

def strip_think(text: str) -> str:
    marker = "</think>\n"
    idx = text.find(marker)
    if idx != -1:
        stripped = text[idx + len(marker):]
        return stripped
    return text

#%%
model_name = 'Qwen/Qwen3-14B' 
model_config = 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                        model_config, 
                                        transcoders_offload='cpu', 
                                        dtype=torch.bfloat16)


# %%
code_prompt1 = """/no_think Generate a function body that chooses one random x. Put each assignment on its own line.
from torch import randperm

def f(xs: list[int]) -> int:"""

code_response1 = """<think>

</think>

```python
   """

raw_sentences = {
    'torch': [code_prompt1, code_response1],
}

urls = {
    'torch': 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-code-0&clerps=%5B%5B%22160196%22%2C%22choose%22%5D%2C%5B%2212859%22%2C%22random%22%5D%2C%5B%22143774%22%2C%22list%22%5D%2C%5B%2275008%22%2C%22random%22%5D%2C%5B%2247896%22%2C%22random+numbers%22%5D%2C%5B%2270136%22%2C%22random+sampling%22%5D%2C%5B%2294334%22%2C%22random+generation%22%5D%2C%5B%22129078%22%2C%22array+%2F+vector+%2F+list%22%5D%2C%5B%22108516%22%2C%22range%22%5D%2C%5B%2213953%22%2C%22%28draw%2Fcast%29+lots%22%5D%2C%5B%2268566%22%2C%22index%22%5D%2C%5B%22119570%22%2C%22random+generation%22%5D%2C%5B%22145484%22%2C%22random+generation%22%5D%2C%5B%2256239%22%2C%22random+drawing%22%5D%2C%5B%2292946%22%2C%22random+choice%22%5D%2C%5B%2286712%22%2C%22random+generation%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%2244443%22%2C%22function+definition%22%5D%2C%5B%2268996%22%2C%22predict+verb%3F%22%5D%2C%5B%22123648%22%2C%22random%22%5D%2C%5B%22105439%22%2C%22colon+%2F+newline+before+%28pseudo-%29code%22%5D%2C%5B%22115876%22%2C%22random+choice%22%5D%2C%5B%2277739%22%2C%22function+definition%22%5D%2C%5B%2274645%22%2C%22variable+declaration%22%5D%2C%5B%22142459%22%2C%22ranges%22%5D%2C%5B%2279731%22%2C%22list%22%5D%2C%5B%2290448%22%2C%22selecting+%2F+indexing%22%5D%2C%5B%22120829%22%2C%22variable+declaration%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2262982%22%2C%22-VCs-%22%5D%2C%5B%22152234%22%2C%22variable+declaration%22%5D%2C%5B%22144420%22%2C%22function+definition%22%5D%2C%5B%2283476%22%2C%22say+len%22%5D%2C%5B%2290831%22%2C%22variable+declaration%22%5D%2C%5B%2232560%22%2C%22variable+declaration+%28counter%29%22%5D%2C%5B%2272594%22%2C%22%28say%29+index%22%5D%2C%5B%22103128%22%2C%22say+len%22%5D%2C%5B%2253313%22%2C%22say+index%22%5D%2C%5B%2220881%22%2C%22variable+declaration%22%5D%2C%5B%22159114%22%2C%22indexing%22%5D%2C%5B%22102294%22%2C%22indices+%2F+dimensions+%2F+axes%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2262923%22%2C%22say+length%22%5D%2C%5B%2290517%22%2C%22slicing+%2F+indexing%22%5D%2C%5B%2217497%22%2C%22say+index%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2277627%22%2C%22say+index%22%5D%2C%5B%22148880%22%2C%22say+random%22%5D%2C%5B%2252721%22%2C%22variable+declaration%22%5D%2C%5B%2298148%22%2C%22say+length%22%5D%2C%5B%2223398%22%2C%22say+index%22%5D%2C%5B%2263349%22%2C%22variable+declaration%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2261440%22%2C%22say+index%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2259743%22%2C%22say+index%22%5D%2C%5B%22101871%22%2C%22don%27t+say+index%22%5D%2C%5B%2212565%22%2C%22variable+declaration%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F%22%5D%2C%5B%22125009%22%2C%22variable+declaration%22%5D%2C%5B%2211135%22%2C%22don%27t+say+in%28dex%29%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%2219679%22%2C%22variable+declaration%22%5D%2C%5B%22102658%22%2C%22don%27t+say+index%22%5D%2C%5B%2266873%22%2C%22variable+declaration%22%5D%5D&pinnedIds=41_14937_52%2C39_66873_52%2C34_23398_52%2C29_53313_52%2C26_120829_52%2C25_74645_52%2C25_77739_52%2C28_90831_52%2C28_32560_52%2C35_61440_42%2C36_101871_42%2C36_59743_42%2C37_11135_42%2C39_102658_52%2C36_59743_52%2C36_101871_52%2C29_53313_42%2C34_23398_42%2C31_77627_42%2C28_72594_42%2C26_90448_42%2C0_160196_11%2C13_145484_15%2C25_115876_42%2C23_123648_42%2C11_13953_13%2C13_145484_42%2C8_70136_28%2C1_75008_13%2C0_12859_13%2C8_70136_13%2C4_47896_13%2CE_4194_13%2CE_39911_11%2CE_19813_6%2C38_19679_52%2C37_125009_52%2C36_12565_52%2C33_52721_52%2C34_63349_52%2C29_20881_52%2C27_152234_52%2C27_144420_52%2C21_44443_52%2C35_61440_52%2C13_145484_28%2C9_94334_28%2C12_119570_28%2C17_86712_28%2C35_61440_28%2C29_53313_28%2C31_77627_36%2C29_53313_36%2C31_77627_52%2C8_70136_15%2C11_13953_15%2C13_145484_13%2C28_72594_52%2C29_159114_42%2C30_17497_42%2C30_90517_42%2C29_102294_28&pruningThreshold=0.63&supernodes=%5B%5B%22variable+declaration%22%2C%2225_74645_52%22%2C%2228_90831_52%22%2C%2227_152234_52%22%2C%2226_120829_52%22%2C%2228_32560_52%22%2C%2239_66873_52%22%2C%2237_125009_52%22%2C%2236_12565_52%22%2C%2229_20881_52%22%2C%2233_52721_52%22%2C%2234_63349_52%22%2C%2238_19679_52%22%5D%2C%5B%22function+definition%22%2C%2227_144420_52%22%2C%2225_77739_52%22%2C%2221_44443_52%22%5D%2C%5B%22don%27t+say+in%28dex%29%22%2C%2237_11135_42%22%2C%2236_101871_42%22%5D%2C%5B%22don%27t+say+index%22%2C%2236_101871_52%22%2C%2239_102658_52%22%5D%2C%5B%22random%22%2C%224_47896_13%22%2C%221_75008_13%22%2C%220_12859_13%22%5D%2C%5B%22say+index%22%2C%2229_53313_36%22%2C%2231_77627_36%22%5D%2C%5B%22random+generation%22%2C%228_70136_28%22%2C%229_94334_28%22%2C%2212_119570_28%22%2C%2213_145484_28%22%2C%2217_86712_28%22%5D%2C%5B%22random+generation+%28im_start%29%22%2C%2223_123648_42%22%2C%2225_115876_42%22%2C%2213_145484_42%22%2C%2226_90448_42%22%5D%2C%5B%22random+sampling+%28.%29%22%2C%228_70136_15%22%2C%2211_13953_15%22%2C%2213_145484_15%22%5D%2C%5B%22random+sampling+%28random%29%22%2C%2213_145484_13%22%2C%228_70136_13%22%2C%2211_13953_13%22%5D%2C%5B%22say+index+%28final%29%22%2C%2228_72594_52%22%2C%2231_77627_52%22%2C%2235_61440_52%22%2C%2236_59743_52%22%2C%2234_23398_52%22%2C%2229_53313_52%22%5D%2C%5B%22%28say%29+index+%28im_start%29%22%2C%2230_90517_42%22%2C%2230_17497_42%22%2C%2229_159114_42%22%2C%2228_72594_42%22%2C%2229_53313_42%22%2C%2231_77627_42%22%2C%2234_23398_42%22%2C%2236_59743_42%22%2C%2235_61440_42%22%5D%2C%5B%22index+%28perm%29%22%2C%2229_102294_28%22%2C%2235_61440_28%22%2C%2229_53313_28%22%5D%5D&clickedId=29_102294_28',
}

supernodes = {
    'torch': ["(say) index (im_start)", "don't say in(dex)"],
}
features = {name: decode_url_features(url)[0] for name, url in urls.items()}

# %%
sentences = {
    name: chattify(lines, model.tokenizer)
    for name, lines in raw_sentences.items()
}

sentences_gen = {
    name: chattify([raw_sentences[name][0], ""], model.tokenizer)
    for name in raw_sentences
}

orig_logits, orig_acts = {}, {}
for name, s in sentences.items():
    logits, acts = model.get_activations(s, zero_bos=True, sparse=True)
    orig_logits[name] = logits
    orig_acts[name] = acts
#%%
zero_interventions = {name: [(*feat,-2*orig_acts[name][feat]) for feat in [f for supernode in supernodes[name] 
                                                                                    for f in feats[supernode]]] 
                                                                for name, feats in features.items()}
boost_interventions = {name: [(*feat,5*orig_acts[name][feat]) for feat in [f for supernode in supernodes[name] 
                                                                                    for f in feats[supernode]]]
                                                                for name, feats in features.items()}
#%%
print(model.generate(sentences['torch'], max_new_tokens=100))
print(model.feature_intervention_generation(sentences['torch'], zero_interventions['torch'], max_new_tokens=100))

#%%
zero_logits, zero_acts = {}, {}
for name, s in sentences.items():
    logits, acts = model.feature_intervention(s, zero_interventions[name], zero_bos=True)
    zero_logits[name] = logits
    zero_acts[name] = acts

#%%
for name, s in sentences.items():
    print(name, s)
    print(get_topk(orig_logits[name], model.tokenizer))
    print(get_topk(zero_logits[name], model.tokenizer))
    orig_generations = []
    for _ in range(5):
            orig_generations.append(strip_think(model.generate(sentences_gen[name], do_sample=True, max_new_tokens=30)))
    print("orig generations")
    for i in range(5):
        print(orig_generations[i])
    zero_generations = []
    for _ in range(5):
        zero_generations.append(strip_think(model.feature_intervention_generation(sentences_gen[name], zero_interventions[name], do_sample=True, max_new_tokens=30)))
    print("zero generations")
    for i in range(5):
        print(zero_generations[i])
    for name2 in sentences.keys():
        if name == name2:
            continue
        
        new_generations = []
        for i in range(5):
            new_generations.append(strip_think(model.feature_intervention_generation(sentences_gen[name], 
                                                        zero_interventions[name] + boost_interventions[name2], 
                                                        do_sample=True, 
                                                        max_new_tokens=30)))
        print(f"Changing {name} to {name2}")
        for i in range(5):
            print(new_generations[i])
# %%
name = 'delight'
name2 = 'delight'
new_generations = []
for i in range(5):
    new_generations.append(strip_think(model.feature_intervention_generation(sentences_gen[name], 
                                                zero_interventions[name] + boost_interventions[name2], 
                                                do_sample=True, 
                                                max_new_tokens=30,
                                                temperature=0.5)))
print(f"Changing {name} to {name2}")
for i in range(5):
    print(new_generations[i])
# %%
