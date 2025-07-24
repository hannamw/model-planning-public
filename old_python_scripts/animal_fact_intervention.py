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
raw_sentences = {
    'largest':["/no_think Write a sentence about the largest animal in the world.", 
        "<think>\n\n</think>\n\nThe blue whale is the largest animal in the world, growing up to 100 feet long and weighing as much as 200 tons"],
    'largest-no-name':["/no_think Write a sentence about the largest animal in the world. Don't say the animal's name until the end.", 
        "<think>\n\n</think>\n\nThe largest animal in the world is a gentle giant that roams the oceans, feeding mostly on tiny plankton and growing up to nearly 100 feet in length"],
    'fastest':["/no_think Write a sentence about the fastest land animal in the world.", 
        "<think>\n\n</think>\n\nThe cheetah is the fastest land animal in the world, capable of reaching speeds up to 70 miles per hour"],
    'fastest-no-name':["/no_think Write a sentence about the fastest land animal in the world. Don't say the animal's name until the end.", 
        "<think>\n\n</think>\n\nIt can sprint at speeds exceeding 70 miles per hour, making it the fastest land animal in the world"],
}

urls = {
    'largest-no-name': 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-blue-whale-and&clerps=%5B%5B%2258486%22%2C%22is%22%5D%2C%5B%2247669%22%2C%22writing%22%5D%2C%5B%22102140%22%2C%22end%22%5D%2C%5B%228468%22%2C%22until%22%5D%2C%5B%22135877%22%2C%22end%22%5D%2C%5B%2244726%22%2C%22end%22%5D%2C%5B%2233820%22%2C%22until%22%5D%2C%5B%2261515%22%2C%22final+touches+%2F+steps%22%5D%2C%5B%222038%22%2C%22dinosaurs%22%5D%2C%5B%2227662%22%2C%22late+to+x%22%5D%2C%5B%22127278%22%2C%22throughout+%2F+start+to+finish%22%5D%2C%5B%22107404%22%2C%22save+%2F+reserve+x+for+y%22%5D%2C%5B%2241802%22%2C%22end%22%5D%2C%5B%2238975%22%2C%22large+size%22%5D%2C%5B%22151234%22%2C%22planning+%2F+spoiling+the+ending%22%5D%2C%5B%2212251%22%2C%22outcome+%2F+planning+%2F+spoiling%22%5D%2C%5B%22138317%22%2C%22open+%2F+answer+%2F+reveal%22%5D%2C%5B%2247314%22%2C%22descriptive+text%22%5D%2C%5B%2228774%22%2C%22final+%2F+conclude%22%5D%2C%5B%22138003%22%2C%22descriptions%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%22107324%22%2C%22buildup%22%5D%2C%5B%2253722%22%2C%22before+we+get+to+X%22%5D%2C%5B%2263309%22%2C%22revealing+things%22%5D%2C%5B%22149743%22%2C%22revealing+things%22%5D%2C%5B%22125631%22%2C%22a+certain+%5Bperson%5D%22%5D%2C%5B%229050%22%2C%22revealing+things%22%5D%2C%5B%22137605%22%2C%22you+guessed+it+%2F+rhetorical+questions%22%5D%2C%5B%2244676%22%2C%22naming+things%22%5D%2C%5B%2210151%22%2C%22you+guessed+it+%2F+rhetorical+questions%22%5D%2C%5B%2289099%22%2C%22you+guessed+it%22%5D%2C%5B%22120485%22%2C%22called+X%22%5D%2C%5B%22139%22%2C%22known+as%22%5D%2C%5B%22160644%22%2C%22it%27s+called%22%5D%2C%5B%2271316%22%2C%22known+as%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2269012%22%2C%22by+the+name+of+%2F+is+called%22%5D%2C%5B%22103465%22%2C%22completed+%2F+finished%22%5D%2C%5B%2299583%22%2C%22I+am...%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2216629%22%2C%22x%27s+name+is%22%5D%2C%5B%2251634%22%2C%22%2C+%28say+and%29%22%5D%2C%5B%2231143%22%2C%22safari%22%5D%2C%5B%2282816%22%2C%22go+by%22%5D%2C%5B%2245525%22%2C%22it+%28in+sentences+about+animals%29%22%5D%2C%5B%2220229%22%2C%22whaling%22%5D%2C%5B%22118982%22%2C%22tails%22%5D%2C%5B%2287367%22%2C%22elephants%22%5D%2C%5B%22157037%22%2C%22%2C+%28say+and%29%22%5D%2C%5B%2217199%22%2C%22whales%22%5D%2C%5B%22118745%22%2C%22also+known+as%22%5D%2C%5B%2261946%22%2C%22large+animals%22%5D%2C%5B%2248298%22%2C%22blue+whales%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%22127878%22%2C%22whales%22%5D%2C%5B%2216087%22%2C%22known+as+%2F+referred+to+as%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%5D&pinnedIds=41_323_71%2C38_7190_71%2C21_120876_36%2C38_7190_35%2C21_120876_35%2C21_120876_34%2C21_120876_48%2C21_120876_37%2C21_120876_52%2C21_120876_71%2C24_115231_71%2C22_53722_26%2C22_63309_26%2C24_137605_71%2C27_89099_71%2C26_10151_71%2C34_48298_71%2C31_31143_71%2C32_20229_71%2C33_17199_71%2C32_87367_71%2C11_2038_39%2C34_61946_71%2C32_45525_71%2C32_118982_71%2C36_127878_71%2C32_82816_71%2C29_71316_71%2C26_44676_71%2C31_16629_71%2C33_157037_71%2C31_51634_71%2C20_28774_26%2C13_41802_26%2C9_44726_26%2C5_135877_26%2C0_8468_24%2C0_102140_26%2CE_3080_24%2CE_835_26%2C13_38975_39%2C11_2038_12%2C28_139_71%2C36_16087_71%2C33_118745_71%2C23_9050_26%2C15_138317_26%2C13_151234_26%2C14_12251_26%2C12_107404_26%2C9_33820_24%2C11_127278_26%2C11_27662_26%2C10_61515_26%2C22_149743_26%2C21_107324_26%2C27_120485_71%2C30_69012_71%2C30_99583_71%2C28_160644_71%2C22_125631_71&pruningThreshold=0.7&supernodes=%5B%5B%22%28dense+feature%29%22%2C%2238_7190_35%22%2C%2238_7190_71%22%5D%2C%5B%22end%22%2C%225_135877_26%22%2C%220_102140_26%22%2C%229_44726_26%22%2C%2213_41802_26%22%5D%2C%5B%22blue+whales%22%2C%2234_48298_71%22%2C%2236_127878_71%22%5D%2C%5B%22whales%22%2C%2233_17199_71%22%2C%2232_20229_71%22%5D%2C%5B%22you+guessed+it+%2F+rhetorical+questions%22%2C%2227_89099_71%22%2C%2224_137605_71%22%2C%2226_10151_71%22%5D%2C%5B%22%3F%3F%3F+often-relevant+often-on+feature%22%2C%2221_120876_48%22%2C%2221_120876_34%22%2C%2221_120876_35%22%2C%2221_120876_36%22%2C%2221_120876_71%22%2C%2221_120876_52%22%5D%2C%5B%22until%22%2C%220_8468_24%22%2C%229_33820_24%22%5D%2C%5B%22%2C+%28say+and%29%22%2C%2231_51634_71%22%2C%2233_157037_71%22%5D%2C%5B%22outcome+%2F+planning+%2F+spoiling%22%2C%2214_12251_26%22%2C%2213_151234_26%22%2C%2212_107404_26%22%5D%2C%5B%22revealing+things%22%2C%2221_107324_26%22%2C%2222_149743_26%22%2C%2215_138317_26%22%2C%2223_9050_26%22%2C%2222_63309_26%22%5D%2C%5B%22known+as+%2F+referred+to+as%22%2C%2230_69012_71%22%2C%2227_120485_71%22%2C%2224_115231_71%22%2C%2226_44676_71%22%2C%2233_118745_71%22%2C%2229_71316_71%22%2C%2228_139_71%22%2C%2236_16087_71%22%2C%2231_16629_71%22%2C%2232_82816_71%22%5D%2C%5B%22final+%2F+conclude%22%2C%2220_28774_26%22%2C%2210_61515_26%22%5D%5D&clickedId=20_28774_26',
    'fastest-no-name': 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-poem-books-fun&clerps=%5B%5B%2292656%22%2C%22say+*Vn-%22%5D%2C%5B%22148103%22%2C%22rhetoric%22%5D%2C%5B%2245888%22%2C%22stopping%22%5D%2C%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22142705%22%2C%22no%2Fnever+stopping%22%5D%2C%5B%2222243%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22145527%22%2C%22say+-%28C%29Vn-%22%5D%2C%5B%2229089%22%2C%22say+-*an-%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%2257079%22%2C%22positions+before+rhymes%22%5D%2C%5B%22102121%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22103817%22%2C%22say+-*an*-%22%5D%2C%5B%222185%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%22149793%22%2C%22enough+%2F+too%22%5D%2C%5B%2274039%22%2C%22say+-*%28V%29nC-%22%5D%2C%5B%2269832%22%2C%22positions+before+rhymes%22%5D%2C%5B%2298007%22%2C%22limit%22%5D%2C%5B%22125140%22%2C%22say+-*an-%22%5D%2C%5B%22138760%22%2C%22say+-an%28C%29-%22%5D%2C%5B%2218707%22%2C%22fun%22%5D%2C%5B%2268436%22%2C%22-unC-%22%5D%2C%5B%2234562%22%2C%22-%28V%29Cn-%22%5D%2C%5B%22104889%22%2C%22never%22%5D%2C%5B%2279437%22%2C%22fun%22%5D%2C%5B%22105934%22%2C%22fun%22%5D%2C%5B%22108626%22%2C%22-un-%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2231413%22%2C%22fun%22%5D%2C%5B%2265944%22%2C%22fun%22%5D%2C%5B%22136953%22%2C%22fun%22%5D%2C%5B%222686%22%2C%22fun-%22%5D%2C%5B%2279561%22%2C%22fun%22%5D%2C%5B%2284066%22%2C%22-%28V%2FC%29n-%22%5D%2C%5B%2270052%22%2C%22-%28V%29%28C%29n-%22%5D%2C%5B%22139555%22%2C%22never%22%5D%2C%5B%2289528%22%2C%22function%22%5D%2C%5B%2265455%22%2C%22fun%22%5D%2C%5B%22137084%22%2C%22say+%5C%22ah%5C%22+vowel%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2238973%22%2C%22-un-%22%5D%2C%5B%22122249%22%2C%22fun%22%5D%2C%5B%22103465%22%2C%22completed+%2F+finished%22%5D%2C%5B%2244466%22%2C%22parentheticals%3F%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2219483%22%2C%22fun%22%5D%2C%5B%2287410%22%2C%22nonstop%22%5D%2C%5B%22126114%22%2C%22f-%22%5D%2C%5B%22163128%22%2C%22finished+%2F+done%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%2282230%22%2C%22say+%5C%22end%5C%22+%28after+%5C%22put+an%5C%22%29%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%2297880%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%22155383%22%2C%22pronunciations%22%5D%2C%5B%22138631%22%2C%22say+-*n%28e%29-%22%5D%2C%5B%22128928%22%2C%22put+%28before+%5C%22down%5C%22%29%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2283012%22%2C%22say+d%28one%29%22%5D%2C%5B%2261457%22%2C%22not%22%5D%2C%5B%2238962%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%22113990%22%2C%22don%27t+say+done%22%5D%2C%5B%2267295%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%22145227%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%2281324%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22129755%22%2C%22don%27t+say+-CVn-%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22104119%22%2C%22say+-Vn%28g%29-%22%5D%5D&pinnedIds=41_2814_37%2C32_126114_20%2C31_19483_20%2C29_79561_20%2C30_122249_20%2C29_2686_20%2C30_38973_20%2C29_84066_20%2C27_126048_20%2C27_105934_20%2C26_34562_20%2C27_108626_20%2C29_136953_20%2C35_128928_37%2C37_145227_37%2C37_67295_37%2C37_129755_37%2C29_70052_20%2C28_31413_20%2C35_88319_37%2C39_104119_37%2C36_83012_37%2C37_113990_37%2C35_155383_37%2C33_82230_37%2C30_103465_37%2C32_163128_37%2C31_87410_37%2C26_104889_37%2C29_139555_37%2C22_98007_37%2C14_45888_37%2C21_149793_37%2C15_142705_37%2C30_137084_37%2C30_44466_37%2C36_38962_37%2C35_138631_37%2C37_81324_37%2C18_145527_20%2C24_138760_20%2C21_74039_20%2C23_125140_20%2C7_92656_20%2C20_103817_20%2C25_68436_20%2C19_29089_20%2C29_89528_20%2C28_65944_20%2C25_18707_20%2C26_79437_20%2C17_140798_21%2C19_102121_21%2C14_154449_21%2C15_22243_21%2C16_2405_21%2C19_102121_23&supernodes=%5B%5B%22finished+%2F+done%22%2C%2232_163128_37%22%2C%2230_103465_37%22%5D%2C%5B%22never%22%2C%2229_139555_37%22%2C%2226_104889_37%22%5D%2C%5B%22fun%22%2C%2226_79437_20%22%2C%2225_18707_20%22%2C%2228_65944_20%22%2C%2228_31413_20%22%2C%2231_19483_20%22%2C%2230_122249_20%22%2C%2229_79561_20%22%2C%2227_105934_20%22%2C%2229_2686_20%22%2C%2229_136953_20%22%5D%2C%5B%22pronunciation+%28final+position%29%22%2C%2230_137084_37%22%2C%2235_138631_37%22%2C%2239_104119_37%22%2C%2237_67295_37%22%2C%2237_145227_37%22%2C%2236_38962_37%22%2C%2237_81324_37%22%2C%2237_129755_37%22%5D%2C%5B%22pronunciation+%28fun+position%29%22%2C%227_92656_20%22%2C%2220_103817_20%22%2C%2221_74039_20%22%2C%2227_108626_20%22%2C%2230_38973_20%22%2C%2227_126048_20%22%2C%2226_34562_20%22%2C%2229_84066_20%22%2C%2229_70052_20%22%2C%2224_138760_20%22%2C%2218_145527_20%22%2C%2223_125140_20%22%2C%2219_29089_20%22%2C%2225_68436_20%22%5D%2C%5B%22ends+of+first+lines+of+poems+%2F+rhymes%22%2C%2215_22243_21%22%2C%2214_154449_21%22%2C%2216_2405_21%22%2C%2217_140798_21%22%2C%2219_102121_21%22%5D%5D&clickedId=19_102121_21',
}

supernodes = {
    'largest-no-name': ["*ight rhyming words", "pronunciation (delight position)"],
    'fun': ["pronunciation (fun position)"],
}
features = {name: decode_url_features(url)[0] for name, url in urls.items()}

# %%
sentences = {
    name: chattify(lines, model.tokenizer)
    for name, lines in raw_sentences.items()
}

sentences_tokens = {
    name: model.tokenizer.convert_ids_to_tokens(model.tokenizer(sentence).input_ids)
    for name, sentence in sentences.items()
}

sentences_gen = {
    name: chattify([raw_sentences[name][0], ""], model.tokenizer)
    for name in raw_sentences
}

#%%
orig_logits, orig_acts = {}, {}
for name, s in sentences.items():
    logits, acts = model.get_activations(s, zero_bos=True)
    orig_logits[name] = logits
    orig_acts[name] = acts

#%%
supernode_features = features['largest-no-name']['revealing things'] + features['largest-no-name']['outcome / planning / spoiling']
new_pos = 16
boost_interventions = [(feat.layer, new_pos, feat.feature_idx, 15*orig_acts['largest-no-name'][feat]) for feat in supernode_features]
#%%
model.feature_intervention_generation(sentences_gen['fastest'], boost_interventions, 
do_sample=True, max_new_tokens=40, freeze_attention=False)
#%%
supernode_features = features['largest-no-name']['revealing things'] + features['largest-no-name']['outcome / planning / spoiling']
boost_interventions = [(*feat, -15*orig_acts['largest-no-name'][feat]) for feat in supernode_features]

model.feature_intervention_generation(sentences_gen['largest-no-name'], boost_interventions, 
do_sample=True, max_new_tokens=40, freeze_attention=True)

#%%
zero_interventions = {name: [(*feat,-2*orig_acts[name][feat]) for feat in [f for supernode in supernodes[name] 
                                                                                    for f in feats[supernode]]] 
                                                                for name, feats in features.items()}
boost_interventions = {name: [(*feat,5*orig_acts[name][feat]) for feat in [f for supernode in supernodes[name] 
                                                                                    for f in feats[supernode]]]
                                                                for name, feats in features.items()}
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
