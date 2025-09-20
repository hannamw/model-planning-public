#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
from functools import partial
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.decode_url_features import decode_url_features, get_sentence_from_url

def _chattify(inputs: list[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

# Helper function to strip the model's internal "thinking" preamble before displaying.
def strip_think(text: str) -> str:
    return re.sub('</think>\n\n', '', text, flags=re.DOTALL)

def _get_topk(logits, tokenizer):
    logits = logits.squeeze(0)[-1]
    probs = torch.softmax(logits, -1)
    values, indices = torch.topk(probs, k=5)
    return [(tokenizer.decode(index), value.item()) for value, index in zip(values, indices)]


#%%
model_name = 'Qwen/Qwen3-14B' 
model_config = "mwhanna/qwen3-14b-transcoders-lowl0"

model = ReplacementModel.from_pretrained(model_name, 
                                        model_config, 
                                        cpu_encoder=True, 
                                        dtype=torch.bfloat16)
#%%
chattify = partial(_chattify, tokenizer=model.tokenizer)
get_topk = partial(_get_topk, tokenizer=model.tokenizer)
#%%
own_url = 'http://localhost:8046/index.html?slug=Qwen3-14B-93-own&clerps=%5B%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%22125080805%22%2C%22than+%28should+be+that%29%22%5D%2C%5B%225121883830%22%2C%22*n%22%5D%2C%5B%223536110626%22%2C%22*n%22%5D%2C%5B%228423176283%22%2C%22*N%22%5D%2C%5B%227876066244%22%2C%22*N%22%5D%2C%5B%22999424967%22%2C%22ends+of+lines%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%225613389926%22%2C%22*n%22%5D%2C%5B%2210385503357%22%2C%22*n*%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%22544120533%22%2C%22near+ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%228755283601%22%2C%22say+*n%22%5D%2C%5B%221712148865%22%2C%22*N%22%5D%2C%5B%221061061177%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%22389163124%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%226553978%22%2C%22near+ends+of+lines%22%5D%2C%5B%222260911364%22%2C%22near+ends+of+lines%22%5D%2C%5B%2212280500812%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211292264592%22%2C%22near+ends+of+lines%22%5D%2C%5B%221940987640%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210913184429%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210589563188%22%2C%22near+ends+of+lines%22%5D%2C%5B%2213374055896%22%2C%22near+ends+of+lines%22%5D%2C%5B%221096641491%22%2C%22very+same%22%5D%2C%5B%222392694042%22%2C%22own%22%5D%2C%5B%226801569988%22%2C%22own%22%5D%2C%5B%2213203856228%22%2C%22own%22%5D%2C%5B%221389989140%22%2C%22*o%22%5D%2C%5B%227372925997%22%2C%22*o%22%5D%2C%5B%2236273877%22%2C%22*o%22%5D%2C%5B%223932115513%22%2C%22*o%22%5D%2C%5B%222336852408%22%2C%22o*%22%5D%2C%5B%22131438771%22%2C%22o*%22%5D%2C%5B%2217949017%22%2C%22o*%22%5D%2C%5B%222862179939%22%2C%22*o%22%5D%2C%5B%22968682092%22%2C%22o*%22%5D%2C%5B%221789605021%22%2C%22w%22%5D%2C%5B%224147009024%22%2C%22own%22%5D%2C%5B%2233944651%22%2C%22own%22%5D%2C%5B%224026588900%22%2C%22own%22%5D%2C%5B%223075357918%22%2C%22own%22%5D%2C%5B%22503888353%22%2C%22own%22%5D%2C%5B%224759220667%22%2C%22own%22%5D%2C%5B%225489476557%22%2C%22own%22%5D%2C%5B%223616538593%22%2C%22own%22%5D%2C%5B%22659516685%22%2C%22own%22%5D%2C%5B%2229918943%22%2C%22own%22%5D%2C%5B%22531885382%22%2C%22own%22%5D%2C%5B%222978028863%22%2C%22own%22%5D%2C%5B%225637467798%22%2C%22own%22%5D%5D&clickedId=37_106145_44&pinnedIds=41_1828_45%2C33_69142_45%2C33_69142_44%2C39_116592_45%2C31_162472_44%2C31_91039_44%2C30_121401_44%2C28_8210_44%2C26_132300_44%2C29_89709_44%2C32_78393_44%2C31_31713_44%2C31_91039_45%2C31_31713_45%2C28_8210_45%2C35_97526_45%2C32_104747_45%2C34_85012_45%2C35_36282_45%2C31_162472_45%2C36_7698_45%2C36_77138_45%2C37_32577_45%2C35_36282_44%2C37_106145_44&pruningThreshold=0.59&supernodes=%5B%5B%22own%22%2C%2231_162472_44%22%2C%2231_31713_44%22%2C%2228_8210_44%22%2C%2229_89709_44%22%2C%2231_91039_44%22%2C%2232_78393_44%22%2C%2233_69142_44%22%2C%2235_36282_44%22%2C%2237_106145_44%22%5D%5D&sentence=%3C%7Cim_start%7C%3Euser%E2%8F%8E%2Fno_think+Write+only+the+next+line+of+this+rhyming+couplet%3A+In+laughter%E2%80%99s+glow%2C+where+trust+is+sown%2C%3C%7Cim_end%7C%3E%E2%8F%8E%3C%7Cim_start%7C%3Eassistant%E2%8F%8E%3Cthink%3E%E2%8F%8E%E2%8F%8E%3C%2Fthink%3E%E2%8F%8E%E2%8F%8EWe+find+the+strength+to+call+our'
#%%
def get_interventions(url, node_name):
    sentence = get_sentence_from_url(url)
    _, acts = model.get_activations(sentence, sparse=True)
    nodes, _ = decode_url_features(url)
    features = nodes[node_name]
    return [(l, slice(-1, None), f, acts[l,p,f].item()) for l,p,f in features]
#%%
own_interventions = get_interventions(own_url, 'own')
#%%
strong_own_interventions = [(l,p,f, 3*a) for l,p,f,a in own_interventions]

# %%
generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, do_sample=False)
print(generation)
# %%
get_topk(model(chattify(["Yesterday I saw a dog"]))[0])
# %%
int_logits, _= model.feature_intervention(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, constrained_layers=range(0, model.cfg.n_layers))
get_topk(int_logits)
# %%
int_logits, _= model.feature_intervention(chattify(["Yesterday I saw a dog that"]), 
                        strong_own_interventions, return_activations=False)
get_topk(int_logits)
# %%
int_logits, _= model.feature_intervention(chattify(["Yesterday I saw a dog that was"]), 
                        strong_own_interventions, return_activations=False)
get_topk(int_logits)
# %%
int_logits, _= model.feature_intervention(chattify(["Yesterday I saw a dog that was"]), 
                        [(l,slice(-3, None),f, a) for l,_, f, a in strong_own_interventions], return_activations=False, freeze_attention=False)
get_topk(int_logits)
# %%
# 1. Define the set of word-features
# 2. Define the dataset over which I will steer with them
# 3. Compute steering accuracy
# 4. Some sort of qualitative analysis?