#%%
from pathlib import Path
from typing import List
from collections import namedtuple, Counter
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
#%%
model_name, model_config = 'Qwen/Qwen3-14B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)


# %%
url_night = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-poem&clerps=%5B%5B%2231375%22%2C%22moon%22%5D%2C%5B%2249906%22%2C%22moon%22%5D%2C%5B%2279011%22%2C%22*t%22%5D%2C%5B%222571%22%2C%22constellations%22%5D%2C%5B%22120328%22%2C%22PM%22%5D%2C%5B%2241589%22%2C%22night%22%5D%2C%5B%2231260%22%2C%22*t%22%5D%2C%5B%2214881%22%2C%22%28V%29t*%22%5D%2C%5B%2296953%22%2C%22evening%22%5D%2C%5B%22155381%22%2C%22%28C%29ot%22%5D%2C%5B%2298927%22%2C%22ain%28t%29%22%5D%2C%5B%22116397%22%2C%22%28C%29V%28C%29t%22%5D%2C%5B%227132%22%2C%22night%3F%22%5D%2C%5B%22113706%22%2C%22right%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2254879%22%2C%22%28C%29it%22%5D%2C%5B%22130088%22%2C%22%28V%29rt%22%5D%2C%5B%2237510%22%2C%22night%22%5D%2C%5B%22437%22%2C%22bright%22%5D%2C%5B%22100088%22%2C%22night%22%5D%2C%5B%2248866%22%2C%22night%22%5D%2C%5B%2243374%22%2C%22night%22%5D%2C%5B%22122077%22%2C%22nighttime%22%5D%2C%5B%2239481%22%2C%22night%22%5D%2C%5B%22163566%22%2C%22midnight%22%5D%2C%5B%22129887%22%2C%22say+*dt+%2F+positions+before+ight%22%5D%2C%5B%2241393%22%2C%22say+*I%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%22136212%22%2C%22night%22%5D%2C%5B%22129639%22%2C%22say+%28V%29t%2FT%28V%29%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2241599%22%2C%22night%22%5D%2C%5B%2288131%22%2C%22say+Cit%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%2228637%22%2C%22say+*ite%22%5D%2C%5B%22103126%22%2C%22don%27t+say+%5C%22night%5C%22%22%5D%2C%5B%224046%22%2C%22positions+before+*t%22%5D%5D&slug=qwen3-14b-lowl0-poem&pinnedIds=41_3729_46%2C35_136212_46%2C29_37510_18%2C28_113706_18%2C26_98927_18%2C26_155381_18%2C27_116397_18%2C26_90214_18%2C25_100813_18%2C22_41971_18%2C23_60055_18%2C20_43164_18%2C1_150636_18%2C1_38379_18%2CE_9906_18%2C0_132473_18%2C35_136212_24%2C27_27985_32%2C24_96953_12%2C14_41589_12%2C12_120328_12%2C11_106103_12%2C9_2571_12%2C0_31375_12%2CE_17788_12%2C37_95349_45%2C36_41599_45%2C36_53902_45%2C35_136212_45%2C36_142751_45%2C39_103126_46%2C36_41599_46%2C37_85066_46%2C38_81398_46%2C27_7132_45%2C27_7132_25%2C33_39481_46%2C30_43374_46%2C27_7132_46%2C24_96953_33%2C13_85565_46%2C5_49906_12%2C14_41589_33%2C12_120328_33%2C11_106103_33%2C13_85565_33%2C9_2571_33%2C5_49906_33%2C0_31375_33%2CE_17788_33%2C27_116397_39%2C26_155381_39&supernodes=%5B%5B%22bright%22%2C%2226_90214_18%22%2C%2225_100813_18%22%2C%2223_60055_18%22%2C%2222_41971_18%22%2C%2220_43164_18%22%2C%221_150636_18%22%2C%221_38379_18%22%2C%220_132473_18%22%5D%2C%5B%22moon%22%2C%220_31375_12%22%2C%225_49906_12%22%5D%2C%5B%22moon%22%2C%220_31375_33%22%2C%225_49906_33%22%5D%2C%5B%22early+morning+times%22%2C%2211_106103_12%22%2C%2212_120328_12%22%5D%2C%5B%22early+morning+times%22%2C%2211_106103_33%22%2C%2212_120328_33%22%5D%2C%5B%22night%22%2C%2224_96953_33%22%2C%2213_85565_33%22%2C%2213_85565_46%22%2C%2214_41589_33%22%5D%2C%5B%22nighttime%22%2C%2214_41589_12%22%2C%2224_96953_12%22%5D%2C%5B%22say+%5C%22night%5C%22%22%2C%2235_136212_46%22%2C%2236_41599_46%22%2C%2237_85066_46%22%2C%2238_81398_46%22%5D%2C%5B%22don%27t+say+night%22%2C%2236_53902_45%22%2C%2237_95349_45%22%5D%5D&clickedId=29_37510_18'
url_4_3 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-3-cats&clerps=%5B%5B%225117%22%2C%22objects+%2F+group%22%5D%2C%5B%22158206%22%2C%22math%22%5D%2C%5B%2242615%22%2C%22financial%22%5D%2C%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%22155947%22%2C%22one%22%5D%2C%5B%22152826%22%2C%22a+pair%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%22140901%22%2C%221%22%5D%2C%5B%22147235%22%2C%22remaining%22%5D%2C%5B%22129295%22%2C%221+%2F+single%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%2250828%22%2C%221%22%5D%2C%5B%22126791%22%2C%22say+1%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%22139674%22%2C%22one%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2268293%22%2C%22numbers%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28&pruningThreshold=0.7&supernodes=%5B%5B%221%22%2C%2223_155947_58%22%2C%2229_121081_58%22%2C%2230_139674_58%22%2C%2228_129295_58%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22say+%5C%22is%5C%22%22%2C%2228_132480_58%22%2C%2233_1511_58%22%2C%2235_65326_58%22%5D%2C%5B%22remaining%22%2C%2226_147235_58%22%2C%2228_69028_58%22%5D%5D&clickedId=34_68293_58'

night_features, _ = decode_url_features(url_night)
diff_4_3_features, _ = decode_url_features(url_4_3)


# %%
s_night = chattify(["Finish this rhyming couplet. The moon shone up above so bright, /nothink", 
"The moon shone up above so bright,\nA silver glow upon the"], model.tokenizer)
s_4_3 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
"At first there were 4 cats. Then, 3 went away. Now, there"], model.tokenizer)

logits_night, acts_night = model.get_activations(s_night, zero_bos=True)
logits_4_3, acts_4_3 = model.get_activations(s_4_3, zero_bos=True)
#%%
night_feature = (29,18, 37510)
zero_interventions = [(*night_feature, -10 * acts_night[night_feature])]
#%%
logits, acts = model.feature_intervention(s_night, zero_interventions)
#%%
print(get_topk(logits_night, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
s_night_gen = chattify(["Finish this rhyming couplet. The moon shone up above so bright, /nothink", 
"The moon shone up above so bright,"], model.tokenizer)
hooks, cache = model._get_feature_intervention_hooks(s_night_gen, zero_interventions)
with model.hooks(hooks):
    print(model.generate(s_night_gen, use_past_kv_cache=False, do_sample=False))

#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
#%%
model_name = 'Qwen/Qwen3-14B' 
model_config = 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                        model_config, 
                                        transcoders_offload='disk', 
                                        dtype=torch.bfloat16)


# %%
url_night = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-poem-boxes-of-books&clerps=%5B%5B%22117967%22%2C%22delight%22%5D%2C%5B%22133048%22%2C%22delight%22%5D%2C%5B%2246867%22%2C%22delight%22%5D%2C%5B%228737%22%2C%22delight%22%5D%2C%5B%2279011%22%2C%22*t%22%5D%2C%5B%2285259%22%2C%22delight%22%5D%2C%5B%22152462%22%2C%22*t%22%5D%2C%5B%22140453%22%2C%22%28C%29Vl%22%5D%2C%5B%2231260%22%2C%22*t%22%5D%2C%5B%2214881%22%2C%22%28V%29t*%22%5D%2C%5B%22155381%22%2C%22%28C%29ot%22%5D%2C%5B%2298927%22%2C%22ain%28t%29%22%5D%2C%5B%22116397%22%2C%22%28C%29V%28C%29t%22%5D%2C%5B%2220349%22%2C%22delight%22%5D%2C%5B%22113706%22%2C%22right%22%5D%2C%5B%22106122%22%2C%22Vl%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2254879%22%2C%22%28C%29it%22%5D%2C%5B%22130088%22%2C%22%28V%29rt%22%5D%2C%5B%2237510%22%2C%22night%22%5D%2C%5B%22437%22%2C%22bright%22%5D%2C%5B%22100088%22%2C%22night%22%5D%2C%5B%2285020%22%2C%22fright%22%5D%2C%5B%2272654%22%2C%22delight%22%5D%2C%5B%22129887%22%2C%22say+*dt+%2F+positions+before+ight%22%5D%2C%5B%2241393%22%2C%22say+*I%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%22136212%22%2C%22night%22%5D%2C%5B%22129639%22%2C%22say+%28V%29t%2FT%28V%29%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2241599%22%2C%22night%22%5D%2C%5B%2288131%22%2C%22say+Cit%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%2228637%22%2C%22say+*ite%22%5D%2C%5B%224046%22%2C%22positions+before+*t%22%5D%5D&pinnedIds=41_3270_36%2C27_116397_20%2C26_155381_20%2C29_437_20%2C29_37510_20%2C28_113706_20%2C28_57695_20%2C28_54879_20%2C28_130088_20%2C27_20349_20%2C30_85020_20%2C31_72654_20%2C39_4046_36%2C38_28637_36%2C33_129887_36%2C36_88131_36%2C35_129639_36%2C34_41393_36%2C18_31260_20%2C23_14881_20%2C9_152462_20%2C9_85259_20%2C7_79011_20%2C3_133048_20%2C3_46867_20%2C3_8737_20%2CE_17970_20%2C0_117967_20&pruningThreshold=0.7&supernodes=%5B%5B%22delight%22%2C%223_133048_20%22%2C%223_46867_20%22%2C%229_85259_20%22%2C%223_8737_20%22%2C%220_117967_20%22%5D%2C%5B%22delight%22%2C%2227_20349_20%22%2C%2231_72654_20%22%5D%2C%5B%22*ight+rhyming+words%22%2C%2229_37510_20%22%2C%2228_113706_20%22%2C%2229_437_20%22%2C%2230_85020_20%22%5D%2C%5B%22subword+tokens+containing+t%22%2C%2228_57695_20%22%2C%2228_130088_20%22%2C%2228_54879_20%22%2C%2227_116397_20%22%2C%2226_155381_20%22%2C%2223_14881_20%22%2C%229_152462_20%22%2C%2218_31260_20%22%2C%227_79011_20%22%5D%2C%5B%22positions+before+*t%22%2C%2233_129887_36%22%2C%2239_4046_36%22%2C%2234_41393_36%22%2C%2236_88131_36%22%2C%2235_129639_36%22%2C%2238_28637_36%22%5D%5D&clickedId=34_22652_20'
url_4_3 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-3-cats&clerps=%5B%5B%225117%22%2C%22objects+%2F+group%22%5D%2C%5B%22158206%22%2C%22math%22%5D%2C%5B%2242615%22%2C%22financial%22%5D%2C%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%22155947%22%2C%22one%22%5D%2C%5B%22152826%22%2C%22a+pair%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%22140901%22%2C%221%22%5D%2C%5B%22147235%22%2C%22remaining%22%5D%2C%5B%22129295%22%2C%221+%2F+single%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%2250828%22%2C%221%22%5D%2C%5B%22126791%22%2C%22say+1%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%22139674%22%2C%22one%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2268293%22%2C%22numbers%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28&pruningThreshold=0.7&supernodes=%5B%5B%221%22%2C%2223_155947_58%22%2C%2229_121081_58%22%2C%2230_139674_58%22%2C%2228_129295_58%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22say+%5C%22is%5C%22%22%2C%2228_132480_58%22%2C%2233_1511_58%22%2C%2235_65326_58%22%5D%2C%5B%22remaining%22%2C%2226_147235_58%22%2C%2228_69028_58%22%5D%5D&clickedId=34_68293_58'

night_features, _ = decode_url_features(url_night)
diff_4_3_features, _ = decode_url_features(url_4_3)

# %%
s_night = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
"Stacked high with stories to"], model.tokenizer)
s_4_3 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
"At first there were 4 cats. Then, 3 went away. Now, there"], model.tokenizer)

logits_night, acts_night = model.get_activations(s_night, zero_bos=True)
logits_4_3, acts_4_3 = model.get_activations(s_4_3, zero_bos=True)
#%%
zero_interventions = [(*night_feature,-5*acts_night[night_feature]) for night_feature in night_features["*ight rhyming words"]]
#%%
logits, acts = model.feature_intervention(s_night, zero_interventions)
#%%
print(get_topk(logits_night, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
s_night_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
""], model.tokenizer)
hooks, cache = model._get_feature_intervention_hooks(s_night_gen, zero_interventions)
with model.hooks(hooks):
    print(model.generate(s_night_gen, use_past_kv_cache=False, do_sample=True))
# %%
#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
#%%
model_name = 'Qwen/Qwen3-14B' 
model_config = 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                        model_config, 
                                        transcoders_offload='disk', 
                                        dtype=torch.bfloat16)


# %%
url_delight = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-poem-boxes-of-books&clerps=%5B%5B%22117967%22%2C%22delight%22%5D%2C%5B%22133048%22%2C%22delight%22%5D%2C%5B%2246867%22%2C%22delight%22%5D%2C%5B%228737%22%2C%22delight%22%5D%2C%5B%2279011%22%2C%22*t%22%5D%2C%5B%2285259%22%2C%22delight%22%5D%2C%5B%22152462%22%2C%22*t%22%5D%2C%5B%22148103%22%2C%22rhetoric%22%5D%2C%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2268490%22%2C%22rhyming+poetry%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140453%22%2C%22%28C%29Vl%22%5D%2C%5B%2231260%22%2C%22*t%22%5D%2C%5B%2259848%22%2C%22say+-VVC-%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%2257079%22%2C%22positions+before+rhymes%22%5D%2C%5B%22102121%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2260921%22%2C%22say+-i%28V%29%28C%29-%22%5D%2C%5B%222185%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%2269832%22%2C%22positions+before+rhymes%22%5D%2C%5B%2265862%22%2C%22say+-VV%28C%29-%22%5D%2C%5B%2214881%22%2C%22%28V%29t*%22%5D%2C%5B%2236412%22%2C%22say+-iC*-%22%5D%2C%5B%22155381%22%2C%22%28C%29ot%22%5D%2C%5B%2298927%22%2C%22ain%28t%29%22%5D%2C%5B%22116397%22%2C%22%28C%29V%28C%29t%22%5D%2C%5B%2220349%22%2C%22delight%22%5D%2C%5B%22113706%22%2C%22right%22%5D%2C%5B%22106122%22%2C%22Vl%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2254879%22%2C%22%28C%29it%22%5D%2C%5B%22130088%22%2C%22%28V%29rt%22%5D%2C%5B%2237510%22%2C%22night%22%5D%2C%5B%22437%22%2C%22bright%22%5D%2C%5B%22100088%22%2C%22night%22%5D%2C%5B%2285020%22%2C%22fright%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2272654%22%2C%22delight%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2245760%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%223588%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%22152499%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%22129887%22%2C%22say+*dt+%2F+positions+before+ight%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%2241393%22%2C%22say+*I%22%5D%2C%5B%2297880%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%22136212%22%2C%22night%22%5D%2C%5B%22129639%22%2C%22say+%28V%29t%2FT%28V%29%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%2236520%22%2C%22say+ite%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2241599%22%2C%22night%22%5D%2C%5B%2288131%22%2C%22say+Cit%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%2228637%22%2C%22say+*ite%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%224046%22%2C%22positions+before+*t%22%5D%5D&pinnedIds=41_3270_36%2C27_116397_20%2C26_155381_20%2C29_437_20%2C29_37510_20%2C28_113706_20%2C28_57695_20%2C28_54879_20%2C28_130088_20%2C27_20349_20%2C30_85020_20%2C31_72654_20%2C39_4046_36%2C38_28637_36%2C33_129887_36%2C36_88131_36%2C35_129639_36%2C34_41393_36%2C18_31260_20%2C23_14881_20%2C9_152462_20%2C9_85259_20%2C7_79011_20%2C3_133048_20%2C3_46867_20%2C3_8737_20%2CE_17970_20%2C0_117967_20%2C26_98927_20%2C19_60921_20%2C18_59848_20%2C22_65862_20%2C23_36412_20%2C35_36520_36%2C32_146050_36%2C34_97880_36%2C33_58511_36%2C32_152499_36%2C31_45760_36%2C31_3588_36%2C19_95229_36%2C21_69832_36%2C20_2185_36%2C19_57079_36%2C17_140798_21%2C19_102121_21%2C16_2405_21%2C14_154449_21%2C14_68490_21&pruningThreshold=0.7&supernodes=%5B%5B%22delight%22%2C%223_133048_20%22%2C%223_46867_20%22%2C%229_85259_20%22%2C%223_8737_20%22%2C%220_117967_20%22%5D%2C%5B%22delight%22%2C%2227_20349_20%22%2C%2231_72654_20%22%5D%2C%5B%22*ight+rhyming+words%22%2C%2229_37510_20%22%2C%2228_113706_20%22%2C%2229_437_20%22%2C%2230_85020_20%22%5D%2C%5B%22pronunciation+%28final+position%29%22%2C%2233_129887_36%22%2C%2239_4046_36%22%2C%2234_41393_36%22%2C%2236_88131_36%22%2C%2235_129639_36%22%2C%2238_28637_36%22%2C%2235_36520_36%22%5D%2C%5B%22positions+before+rhymes%22%2C%2219_57079_36%22%2C%2219_95229_36%22%2C%2220_2185_36%22%2C%2221_69832_36%22%5D%2C%5B%22ends+of+first+lines+of+poems+%2F+rhymes%22%2C%2217_140798_21%22%2C%2216_2405_21%22%2C%2219_102121_21%22%2C%2214_154449_21%22%5D%2C%5B%22old-fashioned+text+%2F+poetry%22%2C%2231_45760_36%22%2C%2232_152499_36%22%2C%2231_3588_36%22%2C%2232_146050_36%22%2C%2233_58511_36%22%2C%2234_97880_36%22%5D%2C%5B%22pronunciation+%28delight+position%29%22%2C%2226_98927_20%22%2C%2223_36412_20%22%2C%2222_65862_20%22%2C%2218_59848_20%22%2C%2219_60921_20%22%2C%2228_57695_20%22%2C%2228_130088_20%22%2C%2228_54879_20%22%2C%2227_116397_20%22%2C%2226_155381_20%22%2C%2223_14881_20%22%2C%229_152462_20%22%2C%2218_31260_20%22%2C%227_79011_20%22%5D%5D&clickedId=19_60921_20'
url_fun = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-poem-books-fun&clerps=%5B%5B%2292656%22%2C%22say+*Vn-%22%5D%2C%5B%22148103%22%2C%22rhetoric%22%5D%2C%5B%2245888%22%2C%22stopping%22%5D%2C%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22142705%22%2C%22no%2Fnever+stopping%22%5D%2C%5B%2222243%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22145527%22%2C%22say+-%28C%29Vn-%22%5D%2C%5B%2229089%22%2C%22say+-*an-%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%2257079%22%2C%22positions+before+rhymes%22%5D%2C%5B%22102121%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22103817%22%2C%22say+-*an*-%22%5D%2C%5B%222185%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%22149793%22%2C%22enough+%2F+too%22%5D%2C%5B%2274039%22%2C%22say+-*%28V%29nC-%22%5D%2C%5B%2269832%22%2C%22positions+before+rhymes%22%5D%2C%5B%2298007%22%2C%22limit%22%5D%2C%5B%22125140%22%2C%22say+-*an-%22%5D%2C%5B%22138760%22%2C%22say+-an%28C%29-%22%5D%2C%5B%2218707%22%2C%22fun%22%5D%2C%5B%2268436%22%2C%22-unC-%22%5D%2C%5B%2234562%22%2C%22-%28V%29Cn-%22%5D%2C%5B%22104889%22%2C%22never%22%5D%2C%5B%2279437%22%2C%22fun%22%5D%2C%5B%22105934%22%2C%22fun%22%5D%2C%5B%22108626%22%2C%22-un-%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2231413%22%2C%22fun%22%5D%2C%5B%2265944%22%2C%22fun%22%5D%2C%5B%22136953%22%2C%22fun%22%5D%2C%5B%222686%22%2C%22fun-%22%5D%2C%5B%2279561%22%2C%22fun%22%5D%2C%5B%2284066%22%2C%22-%28V%2FC%29n-%22%5D%2C%5B%2270052%22%2C%22-%28V%29%28C%29n-%22%5D%2C%5B%22139555%22%2C%22never%22%5D%2C%5B%2289528%22%2C%22function%22%5D%2C%5B%2265455%22%2C%22fun%22%5D%2C%5B%22137084%22%2C%22say+%5C%22ah%5C%22+vowel%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2238973%22%2C%22-un-%22%5D%2C%5B%22122249%22%2C%22fun%22%5D%2C%5B%22103465%22%2C%22completed+%2F+finished%22%5D%2C%5B%2244466%22%2C%22parentheticals%3F%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2219483%22%2C%22fun%22%5D%2C%5B%2287410%22%2C%22nonstop%22%5D%2C%5B%22126114%22%2C%22f-%22%5D%2C%5B%22163128%22%2C%22finished+%2F+done%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%2282230%22%2C%22say+%5C%22end%5C%22+%28after+%5C%22put+an%5C%22%29%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%2297880%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%22155383%22%2C%22pronunciations%22%5D%2C%5B%22138631%22%2C%22say+-*n%28e%29-%22%5D%2C%5B%22128928%22%2C%22put+%28before+%5C%22down%5C%22%29%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2283012%22%2C%22say+d%28one%29%22%5D%2C%5B%2261457%22%2C%22not%22%5D%2C%5B%2238962%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%22113990%22%2C%22don%27t+say+done%22%5D%2C%5B%2267295%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%22145227%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%2281324%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22129755%22%2C%22don%27t+say+-CVn-%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22104119%22%2C%22say+-Vn%28g%29-%22%5D%5D&pinnedIds=41_2814_37%2C32_126114_20%2C31_19483_20%2C29_79561_20%2C30_122249_20%2C29_2686_20%2C30_38973_20%2C29_84066_20%2C27_126048_20%2C27_105934_20%2C26_34562_20%2C27_108626_20%2C29_136953_20%2C35_128928_37%2C37_145227_37%2C37_67295_37%2C37_129755_37%2C29_70052_20%2C28_31413_20%2C35_88319_37%2C39_104119_37%2C36_83012_37%2C37_113990_37%2C35_155383_37%2C33_82230_37%2C30_103465_37%2C32_163128_37%2C31_87410_37%2C26_104889_37%2C29_139555_37%2C22_98007_37%2C14_45888_37%2C21_149793_37%2C15_142705_37%2C30_137084_37%2C30_44466_37%2C36_38962_37%2C35_138631_37%2C37_81324_37%2C18_145527_20%2C24_138760_20%2C21_74039_20%2C23_125140_20%2C7_92656_20%2C20_103817_20%2C25_68436_20%2C19_29089_20%2C29_89528_20%2C28_65944_20%2C25_18707_20%2C26_79437_20%2C17_140798_21%2C19_102121_21%2C14_154449_21%2C15_22243_21%2C16_2405_21%2C19_102121_23&supernodes=%5B%5B%22finished+%2F+done%22%2C%2232_163128_37%22%2C%2230_103465_37%22%5D%2C%5B%22never%22%2C%2229_139555_37%22%2C%2226_104889_37%22%5D%2C%5B%22fun%22%2C%2226_79437_20%22%2C%2225_18707_20%22%2C%2228_65944_20%22%2C%2228_31413_20%22%2C%2231_19483_20%22%2C%2230_122249_20%22%2C%2229_79561_20%22%2C%2227_105934_20%22%2C%2229_2686_20%22%2C%2229_136953_20%22%5D%2C%5B%22pronunciation+%28final+position%29%22%2C%2230_137084_37%22%2C%2235_138631_37%22%2C%2239_104119_37%22%2C%2237_67295_37%22%2C%2237_145227_37%22%2C%2236_38962_37%22%2C%2237_81324_37%22%2C%2237_129755_37%22%5D%2C%5B%22pronunciation+%28fun+position%29%22%2C%227_92656_20%22%2C%2220_103817_20%22%2C%2221_74039_20%22%2C%2227_108626_20%22%2C%2230_38973_20%22%2C%2227_126048_20%22%2C%2226_34562_20%22%2C%2229_84066_20%22%2C%2229_70052_20%22%2C%2224_138760_20%22%2C%2218_145527_20%22%2C%2223_125140_20%22%2C%2219_29089_20%22%2C%2225_68436_20%22%5D%2C%5B%22ends+of+first+lines+of+poems+%2F+rhymes%22%2C%2215_22243_21%22%2C%2214_154449_21%22%2C%2216_2405_21%22%2C%2217_140798_21%22%2C%2219_102121_21%22%5D%5D&clickedId=19_102121_21'

delight_features, _ = decode_url_features(url_delight)
fun_features, _ = decode_url_features(url_fun)

# %%
s_delight = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
"Stacked high with stories to"], model.tokenizer)
s_fun = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
"Stacked high, they're never"], model.tokenizer)

logits_delight, acts_delight = model.get_activations(s_delight, zero_bos=True)
logits_fun, acts_fun = model.get_activations(s_fun, zero_bos=True)
#%%
zero_interventions = [(*delight_feature,-5*acts_delight[delight_feature]) for delight_feature in delight_features["*ight rhyming words"] + delight_features["pronunciation (delight position)"]]
#%%
logits, acts = model.feature_intervention(s_delight, zero_interventions)
#%%
print(get_topk(logits_delight, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
s_delight_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
""], model.tokenizer)
hooks, cache = model._get_feature_intervention_hooks(s_delight_gen, zero_interventions)
with model.hooks(hooks):
    print(model.generate(s_delight_gen, use_past_kv_cache=False, do_sample=True, max_new_tokens=30))

#%%
zero_interventions = [(*fun_feature,-2*acts_fun[fun_feature]) for fun_feature in fun_features["pronunciation (fun position)"]]
#%%
logits, acts = model.feature_intervention(s_fun, zero_interventions)
#%%
print(get_topk(logits_fun, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
s_fun_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
""], model.tokenizer)
hooks, cache = model._get_feature_intervention_hooks(s_fun_gen, zero_interventions)
with model.hooks(hooks):
    print(model.generate(s_fun_gen, use_past_kv_cache=False, do_sample=True, max_new_tokens=30))
# %%
zero_interventions = [(*fun_feature,-2*acts_fun[fun_feature]) for fun_feature in fun_features["pronunciation (fun position)"]]
write_interventions = [(*delight_feature,5*acts_delight[delight_feature]) for delight_feature in delight_features["*ight rhyming words"] + delight_features["pronunciation (delight position)"]]
s_fun_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
""], model.tokenizer)
hooks, cache = model._get_feature_intervention_hooks(s_fun_gen, zero_interventions + write_interventions)
with model.hooks(hooks):
    print(model.generate(s_fun_gen, use_past_kv_cache=False, do_sample=True, max_new_tokens=30))
# %%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
#%%
model_name = 'Qwen/Qwen3-14B' 
model_config = 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                        model_config, 
                                        transcoders_offload='cpu', 
                                        dtype=torch.bfloat16)

# cpu: 6.38s/it
# %%
url_chore = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-poem-books-explore&clerps=%5B%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2268490%22%2C%22rhyming+poetry%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%2221201%22%2C%22%28C%29Vl%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2227454%22%2C%22one%22%5D%2C%5B%22137084%22%2C%22say+%5C%22ah%5C%22+vowel%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2287363%22%2C%22treasure+%2F+gem%22%5D%2C%5B%223588%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%2236520%22%2C%22say+ite%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22101487%22%2C%22rhyming+%2F+abbreviations%22%5D%5D&clickedId=31_74720_20&pinnedIds=41_96451_35%2C41_13186_35%2C30_125121_20%2C29_90324_20%2C29_103782_20%2C28_136719_20%2C28_92161_20%2C27_43987_20%2C27_99427_20%2C27_124515_20%2C27_154033_20%2C26_53355_20%2C26_25311_20%2C25_1252_20%2C25_11031_20%2C25_21201_20%2C24_96733_20%2C24_28942_20%2C23_61654_20%2C23_148979_20%2C23_98821_20%2C22_147686_20%2C21_57394_20%2C21_68342_20%2C31_74720_20&supernodes=%5B%5B%22pronunciation+%28chore+position%29%22%2C%2228_92161_20%22%2C%2227_124515_20%22%2C%2225_1252_20%22%2C%2221_68342_20%22%2C%2223_98821_20%22%2C%2222_147686_20%22%2C%2223_148979_20%22%2C%2224_28942_20%22%2C%2225_21201_20%22%2C%2229_90324_20%22%2C%2227_43987_20%22%2C%2226_25311_20%22%2C%2231_74720_20%22%2C%2230_125121_20%22%2C%2229_103782_20%22%2C%2227_154033_20%22%2C%2228_136719_20%22%2C%2227_99427_20%22%2C%2226_53355_20%22%2C%2225_11031_20%22%2C%2224_96733_20%22%2C%2223_61654_20%22%2C%2221_57394_20%22%5D%5D'
url_fun = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-poem-books-fun&clerps=%5B%5B%2292656%22%2C%22say+*Vn-%22%5D%2C%5B%22148103%22%2C%22rhetoric%22%5D%2C%5B%2245888%22%2C%22stopping%22%5D%2C%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22142705%22%2C%22no%2Fnever+stopping%22%5D%2C%5B%2222243%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22145527%22%2C%22say+-%28C%29Vn-%22%5D%2C%5B%2229089%22%2C%22say+-*an-%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%2257079%22%2C%22positions+before+rhymes%22%5D%2C%5B%22102121%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22103817%22%2C%22say+-*an*-%22%5D%2C%5B%222185%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%22149793%22%2C%22enough+%2F+too%22%5D%2C%5B%2274039%22%2C%22say+-*%28V%29nC-%22%5D%2C%5B%2269832%22%2C%22positions+before+rhymes%22%5D%2C%5B%2298007%22%2C%22limit%22%5D%2C%5B%22125140%22%2C%22say+-*an-%22%5D%2C%5B%22138760%22%2C%22say+-an%28C%29-%22%5D%2C%5B%2218707%22%2C%22fun%22%5D%2C%5B%2268436%22%2C%22-unC-%22%5D%2C%5B%2234562%22%2C%22-%28V%29Cn-%22%5D%2C%5B%22104889%22%2C%22never%22%5D%2C%5B%2279437%22%2C%22fun%22%5D%2C%5B%22105934%22%2C%22fun%22%5D%2C%5B%22108626%22%2C%22-un-%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2231413%22%2C%22fun%22%5D%2C%5B%2265944%22%2C%22fun%22%5D%2C%5B%22136953%22%2C%22fun%22%5D%2C%5B%222686%22%2C%22fun-%22%5D%2C%5B%2279561%22%2C%22fun%22%5D%2C%5B%2284066%22%2C%22-%28V%2FC%29n-%22%5D%2C%5B%2270052%22%2C%22-%28V%29%28C%29n-%22%5D%2C%5B%22139555%22%2C%22never%22%5D%2C%5B%2289528%22%2C%22function%22%5D%2C%5B%2265455%22%2C%22fun%22%5D%2C%5B%22137084%22%2C%22say+%5C%22ah%5C%22+vowel%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2238973%22%2C%22-un-%22%5D%2C%5B%22122249%22%2C%22fun%22%5D%2C%5B%22103465%22%2C%22completed+%2F+finished%22%5D%2C%5B%2244466%22%2C%22parentheticals%3F%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2219483%22%2C%22fun%22%5D%2C%5B%2287410%22%2C%22nonstop%22%5D%2C%5B%22126114%22%2C%22f-%22%5D%2C%5B%22163128%22%2C%22finished+%2F+done%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%2282230%22%2C%22say+%5C%22end%5C%22+%28after+%5C%22put+an%5C%22%29%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%2297880%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%22155383%22%2C%22pronunciations%22%5D%2C%5B%22138631%22%2C%22say+-*n%28e%29-%22%5D%2C%5B%22128928%22%2C%22put+%28before+%5C%22down%5C%22%29%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2283012%22%2C%22say+d%28one%29%22%5D%2C%5B%2261457%22%2C%22not%22%5D%2C%5B%2238962%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%22113990%22%2C%22don%27t+say+done%22%5D%2C%5B%2267295%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%22145227%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%2281324%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22129755%22%2C%22don%27t+say+-CVn-%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22104119%22%2C%22say+-Vn%28g%29-%22%5D%5D&pinnedIds=41_2814_37%2C32_126114_20%2C31_19483_20%2C29_79561_20%2C30_122249_20%2C29_2686_20%2C30_38973_20%2C29_84066_20%2C27_126048_20%2C27_105934_20%2C26_34562_20%2C27_108626_20%2C29_136953_20%2C35_128928_37%2C37_145227_37%2C37_67295_37%2C37_129755_37%2C29_70052_20%2C28_31413_20%2C35_88319_37%2C39_104119_37%2C36_83012_37%2C37_113990_37%2C35_155383_37%2C33_82230_37%2C30_103465_37%2C32_163128_37%2C31_87410_37%2C26_104889_37%2C29_139555_37%2C22_98007_37%2C14_45888_37%2C21_149793_37%2C15_142705_37%2C30_137084_37%2C30_44466_37%2C36_38962_37%2C35_138631_37%2C37_81324_37%2C18_145527_20%2C24_138760_20%2C21_74039_20%2C23_125140_20%2C7_92656_20%2C20_103817_20%2C25_68436_20%2C19_29089_20%2C29_89528_20%2C28_65944_20%2C25_18707_20%2C26_79437_20%2C17_140798_21%2C19_102121_21%2C14_154449_21%2C15_22243_21%2C16_2405_21%2C19_102121_23&supernodes=%5B%5B%22finished+%2F+done%22%2C%2232_163128_37%22%2C%2230_103465_37%22%5D%2C%5B%22never%22%2C%2229_139555_37%22%2C%2226_104889_37%22%5D%2C%5B%22fun%22%2C%2226_79437_20%22%2C%2225_18707_20%22%2C%2228_65944_20%22%2C%2228_31413_20%22%2C%2231_19483_20%22%2C%2230_122249_20%22%2C%2229_79561_20%22%2C%2227_105934_20%22%2C%2229_2686_20%22%2C%2229_136953_20%22%5D%2C%5B%22pronunciation+%28final+position%29%22%2C%2230_137084_37%22%2C%2235_138631_37%22%2C%2239_104119_37%22%2C%2237_67295_37%22%2C%2237_145227_37%22%2C%2236_38962_37%22%2C%2237_81324_37%22%2C%2237_129755_37%22%5D%2C%5B%22pronunciation+%28fun+position%29%22%2C%227_92656_20%22%2C%2220_103817_20%22%2C%2221_74039_20%22%2C%2227_108626_20%22%2C%2230_38973_20%22%2C%2227_126048_20%22%2C%2226_34562_20%22%2C%2229_84066_20%22%2C%2229_70052_20%22%2C%2224_138760_20%22%2C%2218_145527_20%22%2C%2223_125140_20%22%2C%2219_29089_20%22%2C%2225_68436_20%22%5D%2C%5B%22ends+of+first+lines+of+poems+%2F+rhymes%22%2C%2215_22243_21%22%2C%2214_154449_21%22%2C%2216_2405_21%22%2C%2217_140798_21%22%2C%2219_102121_21%22%5D%5D&clickedId=19_102121_21'

chore_features, _ = decode_url_features(url_chore)
fun_features, _ = decode_url_features(url_fun)

# %%
s_chore = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's chore,", 
"A treasure trove to"], model.tokenizer)
s_fun = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
"Stacked high, they're never"], model.tokenizer)

logits_chore, acts_chore = model.get_activations(s_chore, zero_bos=True)
logits_fun, acts_fun = model.get_activations(s_fun, zero_bos=True)
#%%
zero_interventions = [(*chore_feature,-5*acts_chore[chore_feature]) for chore_feature in chore_features["pronunciation (chore position)"]]
#%%
logits, acts = model.feature_intervention(s_chore, zero_interventions)
#%%
print(get_topk(logits_chore, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
s_chore_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's chore,", 
""], model.tokenizer)
print(model.feature_intervention_generation(s_chore_gen, zero_interventions, do_sample=True, max_new_tokens=30))
#%%
zero_interventions = [(*fun_feature,-2*acts_fun[fun_feature]) for fun_feature in fun_features["pronunciation (fun position)"]]

#%%
logits, acts = model.feature_intervention(s_fun, zero_interventions)
#%%
print(get_topk(logits_fun, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
s_fun_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
""], model.tokenizer)
#%%
print(model.feature_intervention_generation(s_fun_gen, zero_interventions, 
                                            do_sample=True, 
                                            max_new_tokens=30))
# %%
zero_interventions = [(*fun_feature,-3*acts_fun[fun_feature]) for fun_feature in fun_features["pronunciation (fun position)"]]
write_interventions = [(*chore_feature,5*acts_chore[chore_feature]) for chore_feature in chore_features["pronunciation (chore position)"]]
s_fun_gen = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
""], model.tokenizer)
#%%
print(model.feature_intervention_generation(s_fun_gen, zero_interventions + write_interventions, do_sample=True, max_new_tokens=30))

# %%
print(model.generate(s_fun_gen, use_past_kv_cache=False, do_sample=True, max_new_tokens=30))

# %%
from collections import Counter

num_samples = 200
last_word_counts = Counter()

for i in range(num_samples):
    # Set a deterministic seed for reproducibility while still getting variety
    torch.manual_seed(i)
    completion = model.generate(s_fun_gen, use_past_kv_cache=False, do_sample=True, max_new_tokens=30)
    print(f"{i+1:03d}: {completion}")
    # Extract the last word, stripping common punctuation and normalising case
    last_word = completion.strip().split()[-1].strip(".,!?;:'\\\"").lower()
    last_word_counts[last_word] += 1

print("\nLast word frequencies (out of 200):")
for word, count in last_word_counts.most_common():
    print(f"{word}: {count}")

# %%
from collections import Counter

num_samples = 200
last_word_counts = Counter()

for i in range(num_samples):
    # Set a deterministic seed for reproducibility while still getting variety
    torch.manual_seed(i)
    completion = model.generate(s_chore_gen, use_past_kv_cache=False, do_sample=True, max_new_tokens=30)
    print(f"{i+1:03d}: {completion}")
    # Extract the last word, stripping common punctuation and normalising case
    last_word = completion.strip().split()[-1].strip(".,!?;:'\\\"").lower()
    last_word_counts[last_word] += 1

print("\nLast word frequencies (out of 200):")
for word, count in last_word_counts.most_common():
    print(f"{word}: {count}")

# %%
