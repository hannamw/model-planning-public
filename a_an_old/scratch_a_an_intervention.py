#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel

#%%
model_name = 'Qwen/Qwen3-14B'
models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
transcoders = models_and_transcoders[model_name]

model = ReplacementModel.from_pretrained(model_name, 
                                            transcoders, 
                                            lazy_encoder=True, 
                                            dtype=torch.bfloat16)

# %%
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
# %%
url_bartender = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-a-bartender&clerps=%5B%5B%2220372%22%2C%22drink%22%5D%2C%5B%2221053%22%2C%22beverages%22%5D%2C%5B%2258486%22%2C%22is%22%5D%2C%5B%22162913%22%2C%22drink%22%5D%2C%5B%22122500%22%2C%22alcohol%22%5D%2C%5B%2220774%22%2C%22is%22%5D%2C%5B%22117371%22%2C%22drink%28ing%29%22%5D%2C%5B%2288830%22%2C%22is%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%22115000%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2266090%22%2C%22beer%22%5D%2C%5B%2219697%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2235919%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2221680%22%2C%22cigarettes%22%5D%2C%5B%2245780%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%22111814%22%2C%22cocktails%22%5D%2C%5B%2287234%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2290938%22%2C%22prohibition%22%5D%2C%5B%224018%22%2C%22bar%22%5D%2C%5B%2232724%22%2C%22cash+registers+%2F+POS%22%5D%2C%5B%22160499%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2287701%22%2C%22driving+%2F+say+%5C%22behind%5C%22%22%5D%2C%5B%22159184%22%2C%22say+%5C%22a%5C%22%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2276218%22%2C%22say+%5C%22b*%5C%22%22%5D%2C%5B%22140516%22%2C%22say+%5C%22a*%5C%22%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%2227113%22%2C%22say+%5C%22b*%5C%22%22%5D%2C%5B%2291098%22%2C%22cocktail%22%5D%2C%5B%22149685%22%2C%22say+%5C%22b*%5C%22%22%5D%2C%5B%2284175%22%2C%22say+%5C%22a%5C%22%22%5D%5D&pinnedIds=41_264_18%2C39_84175_18%2C36_76218_18%2C38_149685_18%2C37_27113_18%2C34_4018_18%2C32_111814_18%2C33_90938_18%2C30_66090_18%2C25_122500_17%2C27_117371_17%2C0_20372_17%2C6_162913_17%2C0_21053_17%2CE_20987_17%2C32_45780_18%2C35_160499_18%2C33_87234_18%2C30_19697_18%2C35_159184_18%2C29_115000_18%2C31_35919_18%2C36_140516_18%2CE_374_18%2C28_26828_18%2C26_20774_18%2C27_88830_18%2C0_58486_18%2C37_91098_18&pruningThreshold=0.5&supernodes=%5B%5B%22drink%28ing%29+alcohol%22%2C%2227_117371_17%22%2C%2225_122500_17%22%2C%220_20372_17%22%2C%226_162913_17%22%2C%220_21053_17%22%5D%2C%5B%22say+%5C%22b*%5C%22%22%2C%2236_76218_18%22%2C%2237_27113_18%22%2C%2238_149685_18%22%5D%2C%5B%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%2C%2233_87234_18%22%2C%2235_160499_18%22%2C%2232_45780_18%22%2C%2231_35919_18%22%2C%2229_115000_18%22%2C%2230_19697_18%22%5D%2C%5B%22say+%5C%22a%5C%22%22%2C%2235_159184_18%22%2C%2239_84175_18%22%5D%2C%5B%22is%22%2C%220_58486_18%22%2C%2227_88830_18%22%2C%2228_26828_18%22%2C%2226_20774_18%22%5D%2C%5B%22bar%22%2C%2234_4018_18%22%2C%2230_66090_18%22%2C%2232_111814_18%22%2C%2233_90938_18%22%2C%2237_91098_18%22%5D%5D&clickedId=alcohol'
url_accountant = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-an-accountant&clerps=%5B%5B%223374%22%2C%22financial%22%5D%2C%5B%2273538%22%2C%22financial%22%5D%2C%5B%2258486%22%2C%22is%22%5D%2C%5B%22155776%22%2C%22financial%22%5D%2C%5B%22116874%22%2C%22financial%22%5D%2C%5B%2282999%22%2C%22exchequer%22%5D%2C%5B%2225022%22%2C%22checkbook+%2F+expenses%22%5D%2C%5B%2210233%22%2C%22accounting+%2F+bookkeeping%22%5D%2C%5B%2247756%22%2C%22CPAs%22%5D%2C%5B%2243609%22%2C%22accounting%22%5D%2C%5B%2220774%22%2C%22is%22%5D%2C%5B%2288830%22%2C%22is%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%22115000%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2255618%22%2C%22banking%22%5D%2C%5B%2223813%22%2C%22finances%22%5D%2C%5B%2219697%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2235919%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%22111811%22%2C%22accounting%22%5D%2C%5B%2240107%22%2C%22economics+%2F+accounting%22%5D%2C%5B%2231593%22%2C%22finance%22%5D%2C%5B%22135260%22%2C%22accounting%22%5D%2C%5B%2245780%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2221020%22%2C%22accounting%22%5D%2C%5B%2287234%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2210752%22%2C%22accounting%22%5D%2C%5B%2272914%22%2C%22accounting%22%5D%2C%5B%22117738%22%2C%22audit%22%5D%2C%5B%2243347%22%2C%22finances%22%5D%2C%5B%22159252%22%2C%22say+%5C%22acc%5C%22+%28account%29%22%5D%2C%5B%2285195%22%2C%22say+%5C%22acc%5C%22%22%5D%2C%5B%22160499%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2244167%22%2C%22%28chartered%29+accountant%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%22159184%22%2C%22say+%5C%22a%5C%22%22%5D%2C%5B%2229458%22%2C%22%5C%22+ac%5C%22+%28don%27t+say+%5C%22+ac%5C%22%29%22%5D%2C%5B%2292068%22%2C%22accounts%22%5D%2C%5B%22100905%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%22140516%22%2C%22say+%5C%22a*%5C%22%22%5D%2C%5B%2276520%22%2C%22say+%5C%22+a*%5C%22%22%5D%2C%5B%2213446%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2232592%22%2C%22say+%5C%22a*%5C%22%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%2242344%22%2C%22say+%5C%22an%5C%22%22%5D%2C%5B%2252418%22%2C%22don%27t+say+a%2Fan%22%5D%2C%5B%2284175%22%2C%22say+%5C%22a%5C%22%22%5D%5D&pinnedIds=41_458_17%2C38_42344_17%2C35_159252_17%2C37_76520_17%2C39_52418_17%2C32_135260_17%2C34_72914_17%2C33_21020_17%2C36_29458_17%2C35_85195_17%2C36_92068_17%2C37_32592_17%2C33_87234_17%2C37_13446_17%2C35_160499_17%2C36_100905_17%2C32_45780_17%2C31_35919_17%2C29_115000_17%2C33_10752_17%2C35_44167_17%2C31_111811_17%2C34_43347_17%2C30_23813_17%2C24_43609_16%2C23_47756_16%2C8_10233_16%2C1_116874_15%2C0_3374_15%2C0_73538_15%2C1_155776_15%2CE_5896_15%2C31_40107_17%2C31_31593_17%2C37_103617_17%2C35_125138_17%2C36_103390_17&supernodes=%5B%5B%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%2C%2236_100905_17%22%2C%2231_35919_17%22%2C%2232_45780_17%22%2C%2229_115000_17%22%2C%2233_87234_17%22%2C%2235_160499_17%22%2C%2237_13446_17%22%5D%2C%5B%22say+%5C%22a*%5C%22%22%2C%2237_32592_17%22%2C%2237_76520_17%22%5D%2C%5B%22financial%22%2C%220_73538_15%22%2C%221_155776_15%22%2C%221_116874_15%22%2C%220_3374_15%22%5D%2C%5B%22finance%22%2C%2231_31593_17%22%2C%2230_23813_17%22%2C%2234_43347_17%22%5D%2C%5B%22say+%5C%22acc%5C%22%22%2C%2235_85195_17%22%2C%2235_159252_17%22%5D%2C%5B%22accounting%22%2C%2223_47756_16%22%2C%2224_43609_16%22%2C%228_10233_16%22%5D%2C%5B%22accounting%22%2C%2231_40107_17%22%2C%2231_111811_17%22%2C%2233_10752_17%22%2C%2234_72914_17%22%2C%2233_21020_17%22%2C%2232_135260_17%22%2C%2235_44167_17%22%5D%2C%5B%22%3F%3F%3F%22%2C%2236_103390_17%22%2C%2235_125138_17%22%5D%2C%5B%22say+%5C%22an%5C%22%22%2C%2237_103617_17%22%2C%2238_42344_17%22%5D%5D&clickedId=37_76520_17'

bartender_features, _ = decode_url_features(url_bartender)
accountant_features, _ = decode_url_features(url_accountant)


# %%
s_bartender = chattify(["Someone who studies living organisms is a biologist. Someone who mixes and serves drinks is"], model.tokenizer)
s_accountant = chattify(["Someone who studies living organisms is a biologist. Someone who manages financial records is"], model.tokenizer)

logits_bartender, acts_bartender = model.get_activations(s_bartender, zero_bos=True)
logits_accountant, acts_accountant = model.get_activations(s_accountant, zero_bos=True)
#%%
zero_interventions = [(*feat, 0.0) for feat in accountant_features['accounting']]
#%%
logits, acts = model.feature_intervention(s_accountant, zero_interventions)
#%%
print(get_topk(logits_accountant, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
print(get_topk(logits_bartender, model.tokenizer))
#%%
boost_interventions = [(feat.layer, -1, feat.feature_idx, acts_bartender[feat]*2) for feat in bartender_features['bar']]
#%%
logits, acts = model.feature_intervention(s_accountant, zero_interventions + boost_interventions)
#%%
print(get_topk(logits_accountant, model.tokenizer))
print(get_topk(logits, model.tokenizer))
#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel

#%%
model_name, model_config = 'Qwen/Qwen3-4B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-4b-relu.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)

# %%
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
# %%
url_bartender = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-4b-relu-a-bartender&clerps=%5B%5B%2218029%22%2C%22drink%28s%29%22%5D%2C%5B%2235023%22%2C%22alcohol%22%5D%2C%5B%22116532%22%2C%22cocktail%22%5D%2C%5B%2296140%22%2C%22bars%22%5D%2C%5B%2276647%22%2C%22bars%22%5D%2C%5B%2233979%22%2C%22shops+%2F+shopkeepers%22%5D%2C%5B%2271569%22%2C%22clubs+%2F+brothels%22%5D%2C%5B%22155785%22%2C%22alcohol%22%5D%2C%5B%22125675%22%2C%22alcohol%22%5D%2C%5B%22121844%22%2C%22liquor%22%5D%2C%5B%22109613%22%2C%22bars+%28alcohol%29%22%5D%2C%5B%2281931%22%2C%22say+%28b%29ar%22%5D%5D&clickedId=9_116532_17&pruningThreshold=0.59&pinnedIds=30_71569_18%2C10_96140_17%2C29_76647_18%2C9_116532_17%2C30_155785_18%2C7_35023_17%2C5_18029_17%2C3_156811_17%2C3_106724_17%2C3_21066_17%2C3_154760_17%2CE_20987_17%2C0_108354_17%2C30_125675_18%2C31_121844_18%2C32_109613_18%2C34_81931_18&supernodes=%5B%5B%22drink%28s%29%22%2C%225_18029_17%22%2C%223_156811_17%22%2C%223_106724_17%22%2C%223_21066_17%22%2C%223_154760_17%22%2C%220_108354_17%22%5D%2C%5B%22bars%22%2C%2230_155785_18%22%2C%2230_125675_18%22%2C%2229_76647_18%22%2C%2232_109613_18%22%2C%2231_121844_18%22%2C%2230_71569_18%22%5D%5D'
url_accountant = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-4b-relu-an-accountant&clerps=%5B%5B%22101010%22%2C%22finances%22%5D%2C%5B%2285644%22%2C%22financial%22%5D%2C%5B%2290421%22%2C%22keep+track+of%22%5D%2C%5B%2221819%22%2C%22records%22%5D%2C%5B%2257827%22%2C%22accounting%22%5D%2C%5B%22140276%22%2C%22financial%22%5D%2C%5B%22123868%22%2C%22finance%22%5D%2C%5B%22110404%22%2C%22accounting+%2F+bookkeeping%22%5D%2C%5B%22150784%22%2C%22accounting%22%5D%2C%5B%2296491%22%2C%22accounting%22%5D%2C%5B%22122958%22%2C%22accounting%22%5D%2C%5B%2228554%22%2C%22accounting%22%5D%2C%5B%2210928%22%2C%22accounting%22%5D%2C%5B%2286852%22%2C%22banking%22%5D%2C%5B%22105198%22%2C%22accounting+%2F+auditing%22%5D%5D&pinnedIds=33_62967_17%2C29_114321_17%2C34_145627_17%2C32_72206_17%2C31_121645_17%2C27_141272_17%2C37_458_17%2C27_52529_17%2C32_159668_17%2C34_60788_17%2C28_34381_17%2C33_6859_17%2C33_141689_17%2C34_67251_17%2C23_132264_9%2C19_22603_9%2C32_16678_17%2C29_162331_17%2C30_105511_17%2C27_81396_17%2C24_49173_15%2C22_47651_15%2C5_10966_15%2C19_42615_15%2C4_136906_15%2C0_136711_15%2CE_5896_15%2C29_159363_17%2C31_112979_17%2C33_8786_17%2C33_35155_17%2C29_150784_17%2C31_96491_17%2C32_28554_17%2C33_10928_17%2C34_105198_17%2C31_122958_17%2C28_110404_17%2C25_140276_15%2C9_57827_16%2C6_21819_16%2C6_90421_16%2C1_85644_15%2C0_101010_15%2CE_7424_16&supernodes=%5B%5B%22earlier+instance+of+%5C%22a%5C%22%22%2C%2219_22603_9%22%2C%2223_132264_9%22%5D%2C%5B%22say+an+%3E+a%22%2C%2231_121645_17%22%2C%2229_114321_17%22%2C%2227_141272_17%22%2C%2227_52529_17%22%5D%2C%5B%22accounting%22%2C%2233_141689_17%22%2C%2234_67251_17%22%2C%2234_60788_17%22%2C%2233_62967_17%22%2C%2232_159668_17%22%5D%2C%5B%22finance%22%2C%225_10966_15%22%2C%2224_49173_15%22%2C%2222_47651_15%22%2C%2219_42615_15%22%2C%220_136711_15%22%2C%224_136906_15%22%5D%2C%5B%22accounting%22%2C%2234_105198_17%22%2C%2233_10928_17%22%2C%2231_122958_17%22%2C%2228_110404_17%22%2C%2229_150784_17%22%2C%2231_96491_17%22%2C%2232_28554_17%22%5D%2C%5B%22financial%22%2C%2225_140276_15%22%2C%221_85644_15%22%2C%220_101010_15%22%5D%5D&pruningThreshold=0.56&clickedId=37_458_17'

bartender_features, _ = decode_url_features(url_bartender)
accountant_features, _ = decode_url_features(url_accountant)


# %%
s_bartender = chattify(["Someone who studies living organisms is a biologist. Someone who mixes and serves drinks is"], model.tokenizer)
s_accountant = chattify(["Someone who studies living organisms is a biologist. Someone who manages financial records is"], model.tokenizer)

logits_bartender, acts_bartender = model.get_activations(s_bartender, zero_bos=True)
logits_accountant, acts_accountant = model.get_activations(s_accountant, zero_bos=True)
#%%
zero_interventions = [(*feat, 3*acts_accountant[feat]) for feat in accountant_features['accounting (2)']]
#%%
logits, acts = model.feature_intervention(s_accountant, zero_interventions, direct_effects=True)
#%%
print(get_topk(logits_accountant, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
print(get_topk(logits_bartender, model.tokenizer))
#%%
boost_interventions = [(feat.layer, -1, feat.feature_idx, acts_bartender[feat]*2) for feat in bartender_features['bars']]
#%%
logits, acts = model.feature_intervention(s_accountant, zero_interventions + boost_interventions)
#%%
print(get_topk(logits_accountant, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
url_economist = "http://localhost:8002/?clerps=%5B%5B%222973%22%2C%22econ%22%5D%2C%5B%22129505%22%2C%22econ%22%5D%2C%5B%2240107%22%2C%22econ%22%5D%2C%5B%2216237%22%2C%22econ%22%5D%2C%5B%2252156%22%2C%22macro%22%5D%2C%5B%2240443%22%2C%22econ%22%5D%2C%5B%22119217%22%2C%22econ%22%5D%2C%5B%2287234%22%2C%22say+an%22%5D%2C%5B%2210497%22%2C%22economic+theory%22%5D%2C%5B%22160499%22%2C%22say+an%22%5D%2C%5B%2211542%22%2C%22say+e%22%5D%2C%5B%22100905%22%2C%22say+an%22%5D%2C%5B%2213446%22%2C%22say+an%22%5D%5D&pinnedIds=34_10497_20%2C32_52156_20%2C31_139793_20%2C30_129505_20%2C35_83107_20%2C31_40107_20%2C32_40443_20%2C34_130463_20%2C29_2973_20%2C36_81603_20%2C36_122151_20%2C32_16237_20%2C36_11542_20%2C30_16729_20%2C33_119217_20%2C41_458_20%2C35_160499_20%2C37_13446_20%2C33_87234_20%2C36_100905_20&pruningThreshold=0.4&clickedId=37_13446_20&supernodes=%5B%5B%22econ%22%2C%2230_129505_20%22%2C%2231_40107_20%22%2C%2232_16237_20%22%2C%2233_119217_20%22%2C%2232_40443_20%22%2C%2229_2973_20%22%5D%2C%5B%22say+an%22%2C%2233_87234_20%22%2C%2235_160499_20%22%2C%2236_100905_20%22%2C%2237_13446_20%22%5D%5D"
economist_features, _ = decode_url_features(url_economist)
# %%
s_economist = chattify(["Someone who studies living organisms is a biologist. Someone who studies financial systems and markets is"], model.tokenizer)

logits_economist, acts_economist = model.get_activations(s_economist)
print(get_topk(logits_economist, model.tokenizer))
# %%
interventions = [(layer, -1, idx, 5*acts_economist[layer, -1, idx]) for layer, pos, idx in economist_features['econ']]
logits, acts = model.feature_intervention(s_economist, interventions)
# %%
print(get_topk(logits, model.tokenizer))
# %%
