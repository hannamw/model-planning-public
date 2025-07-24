#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel

#%%
model_name, model_config = 'Qwen/Qwen3-8B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-8b-relu.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)

# %%
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
# %%
url_4_1 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-1-cats&clerps=%5B%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%2264861%22%2C%22two%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28%2C26_64861_58%2C32_61564_58%2C37_525_58%2C30_22708_58%2C35_113402_58%2C24_108032_58&pruningThreshold=0.57&supernodes=%5B%5B%221%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22one%22%2C%2230_139674_58%22%2C%2228_129295_58%22%5D%2C%5B%22two%22%2C%2226_64861_58%22%2C%2229_19933_58%22%5D%5D&clickedId=29_19933_58'
url_4_3 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-3-cats&clerps=%5B%5B%225117%22%2C%22objects+%2F+group%22%5D%2C%5B%22158206%22%2C%22math%22%5D%2C%5B%2242615%22%2C%22financial%22%5D%2C%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%22155947%22%2C%22one%22%5D%2C%5B%22152826%22%2C%22a+pair%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%22140901%22%2C%221%22%5D%2C%5B%22147235%22%2C%22remaining%22%5D%2C%5B%22129295%22%2C%221+%2F+single%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%2250828%22%2C%221%22%5D%2C%5B%22126791%22%2C%22say+1%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%22139674%22%2C%22one%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2268293%22%2C%22numbers%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28&pruningThreshold=0.7&supernodes=%5B%5B%221%22%2C%2223_155947_58%22%2C%2229_121081_58%22%2C%2230_139674_58%22%2C%2228_129295_58%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22say+%5C%22is%5C%22%22%2C%2228_132480_58%22%2C%2233_1511_58%22%2C%2235_65326_58%22%5D%2C%5B%22remaining%22%2C%2226_147235_58%22%2C%2228_69028_58%22%5D%5D&clickedId=34_68293_58'

diff_4_1_features, _ = decode_url_features(url_4_1)
diff_4_3_features, _ = decode_url_features(url_4_3)


# %%
s_4_1 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 1 went away. Now, there", 
"At first there were 4 cats. Then, 1 went away. Now, there"], model.tokenizer)
s_4_3 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
"At first there were 4 cats. Then, 3 went away. Now, there"], model.tokenizer)

logits_4_1, acts_4_1 = model.get_activations(s_4_1, zero_bos=True)
logits_4_3, acts_4_3 = model.get_activations(s_4_3, zero_bos=True)
#%%
zero_interventions = [(*feat, 0.0) for feat in diff_4_3_features['1']]
#%%
logits, acts = model.feature_intervention(s_4_3, zero_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
print(get_topk(logits_4_1, model.tokenizer))
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
url_4_1 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-1-cats&clerps=%5B%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%2264861%22%2C%22two%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28%2C26_64861_58%2C32_61564_58%2C37_525_58%2C30_22708_58%2C35_113402_58%2C24_108032_58&pruningThreshold=0.57&supernodes=%5B%5B%221%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22one%22%2C%2230_139674_58%22%2C%2228_129295_58%22%5D%2C%5B%22two%22%2C%2226_64861_58%22%2C%2229_19933_58%22%5D%5D&clickedId=29_19933_58'
url_4_3 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-4b-relu-is-4-3-cats&clerps=%5B%5B%22142554%22%2C%22one%22%5D%2C%5B%2267462%22%2C%22one%22%5D%2C%5B%2249340%22%2C%22one%22%5D%2C%5B%22133233%22%2C%22a+%2F+one%22%5D%2C%5B%2289913%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22149956%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22131930%22%2C%22there%27s%22%5D%2C%5B%2293902%22%2C%22*is%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C9_5117_28%2C4_148912_20%2C0_68363_19%2C5_35540_20%2CE_19_19%2C10_150538_28%2CE_18_25%2C4_65890_25%2C0_77451_25%2C4_95529_25%2C5_143320_25%2C23_152826_58%2C33_1511_31%2C30_161390_31%2C23_125177_31%2C34_102028_58%2C32_64452_58%2C22_138077_31%2C0_127450_31%2CE_1052_31%2CE_1052_58%2C0_127450_58%2C27_31117_58%2C25_38436_58%2C22_138077_58%2C23_125177_58%2C26_64861_58%2C30_156639_58%2C29_126791_58%2C26_23003_58%2C32_27710_58%2C32_49340_58%2C30_142554_58%2C30_67462_58%2C33_89913_58%2C32_133233_58%2C34_93902_58%2C34_131930_58%2C33_149956_58&pruningThreshold=0.49&supernodes=%5B%5B%22remaining%22%2C%2230_149579_57%22%2C%2223_106041_58%22%2C%2222_58589_58%22%2C%2226_147235_58%22%2C%2228_69028_58%22%5D%2C%5B%223%22%2C%225_143320_25%22%2C%224_95529_25%22%2C%224_65890_25%22%2C%220_77451_25%22%5D%2C%5B%22say+%5C%22is%5C%22%22%2C%2228_132480_58%22%2C%2233_1511_58%22%2C%2235_65326_58%22%2C%2234_102028_58%22%5D%2C%5B%22two+%2F+only%22%2C%2226_64861_58%22%2C%2229_19933_58%22%2C%2223_152826_58%22%5D%2C%5B%22math+lessons%22%2C%225_35540_20%22%2C%224_148912_20%22%2C%2210_150538_28%22%2C%2218_158206_52%22%5D%2C%5B%221%22%2C%2230_156639_58%22%2C%2229_126791_58%22%2C%2223_155947_58%22%2C%2229_121081_58%22%2C%2230_139674_58%22%2C%2228_129295_58%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22there+is+%2F+are%22%2C%2227_31117_58%22%2C%2226_23003_58%22%2C%2232_27710_58%22%2C%2222_138077_58%22%2C%2223_125177_58%22%2C%2225_38436_58%22%5D%2C%5B%22one%22%2C%2232_133233_58%22%2C%2230_67462_58%22%2C%2230_142554_58%22%2C%2232_49340_58%22%5D%5D&clickedId=26_158180_58'

diff_4_1_features, _ = decode_url_features(url_4_1)
diff_4_3_features, _ = decode_url_features(url_4_3)


# %%
s_4_1 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 1 went away. Now, there", 
"At first there were 4 cats. Then, 1 went away. Now, there"], model.tokenizer)
s_4_3 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
"At first there were 4 cats. Then, 3 went away. Now, there"], model.tokenizer)

logits_4_1, acts_4_1 = model.get_activations(s_4_1, zero_bos=True)
logits_4_3, acts_4_3 = model.get_activations(s_4_3, zero_bos=True)
#%%
zero_interventions = [(*feat, 0.0) for feat in diff_4_3_features['one']]
#%%
logits, acts = model.feature_intervention(s_4_3, zero_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
print(get_topk(logits_4_1, model.tokenizer))
# %%
#%%
boost_interventions = [(*feat, acts_4_3[feat]*2) for feat in diff_4_3_features['one']]
#%%
logits, acts = model.feature_intervention(s_4_3, boost_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel

#%%
model_name, model_config = 'Qwen/Qwen3-1.7B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-1.7b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)

# %%
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
# %%
url_4_1 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-1-cats&clerps=%5B%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%2264861%22%2C%22two%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28%2C26_64861_58%2C32_61564_58%2C37_525_58%2C30_22708_58%2C35_113402_58%2C24_108032_58&pruningThreshold=0.57&supernodes=%5B%5B%221%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22one%22%2C%2230_139674_58%22%2C%2228_129295_58%22%5D%2C%5B%22two%22%2C%2226_64861_58%22%2C%2229_19933_58%22%5D%5D&clickedId=29_19933_58'
url_4_3 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-1.7b-relu-lowl0-is-4-3-cats&clerps=%5B%5B%2283402%22%2C%22one%22%5D%2C%5B%2210910%22%2C%22one%22%5D%2C%5B%2266945%22%2C%22one%22%5D%5D&clickedId=22_83402_58&pinnedIds=22_67250_58%2C22_83402_58%2C24_66945_58%2C23_10910_58&supernodes=%5B%5B%22one%22%2C%2222_83402_58%22%2C%2223_10910_58%22%2C%2224_66945_58%22%5D%5D'

diff_4_1_features, _ = decode_url_features(url_4_1)
diff_4_3_features, _ = decode_url_features(url_4_3)


# %%
s_4_1 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 1 went away. Now, there", 
"At first there were 4 cats. Then, 1 went away. Now, there"], model.tokenizer)
s_4_3 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
"At first there were 4 cats. Then, 3 went away. Now, there"], model.tokenizer)

logits_4_1, acts_4_1 = model.get_activations(s_4_1, zero_bos=True)
logits_4_3, acts_4_3 = model.get_activations(s_4_3, zero_bos=True)
#%%
zero_interventions = [(*feat, 0.0) for feat in diff_4_3_features['one']]
#%%
logits, acts = model.feature_intervention(s_4_3, zero_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
print(get_topk(logits_4_1, model.tokenizer))
#%%
boost_interventions = [(*feat, acts_4_3[feat]*2) for feat in diff_4_3_features['one']]
#%%
logits, acts = model.feature_intervention(s_4_3, boost_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.replacement_model import ReplacementModel

#%%
model_name, model_config = 'Qwen/Qwen3-0.6B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-0.6b-relu-lowl0.yaml'

model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)

# %%
from circuit_tracer.utils.intervention_utils import decode_url_features, chattify, get_topk
# %%
url_4_1 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-8b-relu-is-4-1-cats&clerps=%5B%5B%22108951%22%2C%22%28numbers%29%22%5D%2C%5B%2258589%22%2C%22remaining+%2F+math%22%5D%2C%5B%22106041%22%2C%22remaining%22%5D%2C%5B%2221117%22%2C%22odd+%2F+trio%22%5D%2C%5B%2264861%22%2C%22two%22%5D%2C%5B%22132480%22%2C%22say+%28there%29+%5C%22is%5C%22%22%5D%2C%5B%2269028%22%2C%22remaining%22%5D%2C%5B%22121081%22%2C%22only+%2F+single%22%5D%2C%5B%2219933%22%2C%22two+%2F+only%22%5D%2C%5B%22105511%22%2C%22%3F%3F%3F%22%5D%2C%5B%2211717%22%2C%221%22%5D%2C%5B%2262455%22%2C%22numbers+-%3E+there%22%5D%2C%5B%22161390%22%2C%22is+%3E+are%22%5D%2C%5B%22149579%22%2C%22left+with%22%5D%2C%5B%227066%22%2C%221%22%5D%2C%5B%221511%22%2C%22say+%5C%22is%5C%22%22%5D%2C%5B%22145627%22%2C%22%3F%3F%3F%22%5D%2C%5B%2210398%22%2C%22number%22%5D%2C%5B%2265326%22%2C%22say+%5C%22is%5C%22%22%5D%5D&pinnedIds=37_374_58%2C35_65326_58%2C33_1511_58%2C30_161390_58%2C28_132480_58%2C34_10398_58%2C34_68293_58%2C33_7066_58%2C30_11717_58%2C29_50828_58%2C29_19933_58%2C29_121081_58%2C26_140901_58%2C28_129295_58%2C28_69028_58%2C26_147235_58%2C25_21117_58%2C23_106041_58%2C30_149579_57%2C30_62455_57%2C30_139674_58%2C23_155947_58%2C22_108951_58%2C18_158206_52%2C22_58589_58%2C23_152826_58%2C9_5117_28%2C26_64861_58%2C32_61564_58%2C37_525_58%2C30_22708_58%2C35_113402_58%2C24_108032_58&pruningThreshold=0.57&supernodes=%5B%5B%221%22%2C%2233_7066_58%22%2C%2230_11717_58%22%2C%2229_50828_58%22%2C%2226_140901_58%22%5D%2C%5B%22one%22%2C%2230_139674_58%22%2C%2228_129295_58%22%5D%2C%5B%22two%22%2C%2226_64861_58%22%2C%2229_19933_58%22%5D%5D&clickedId=29_19933_58'
url_4_3 = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-1.7b-relu-lowl0-is-4-3-cats&clerps=%5B%5B%2283402%22%2C%22one%22%5D%2C%5B%2210910%22%2C%22one%22%5D%2C%5B%2266945%22%2C%22one%22%5D%5D&clickedId=22_83402_58&pinnedIds=22_67250_58%2C22_83402_58%2C24_66945_58%2C23_10910_58&supernodes=%5B%5B%22one%22%2C%2222_83402_58%22%2C%2223_10910_58%22%2C%2224_66945_58%22%5D%5D'

diff_4_1_features, _ = decode_url_features(url_4_1)
diff_4_3_features, _ = decode_url_features(url_4_3)


# %%
s_4_1 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 1 went away. Now, there", 
"At first there were 4 cats. Then, 1 went away. Now, there"], model.tokenizer)
s_4_3 = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
"At first there were 4 cats. Then, 3 went away. Now, there"], model.tokenizer)

logits_4_1, acts_4_1 = model.get_activations(s_4_1, zero_bos=True)
logits_4_3, acts_4_3 = model.get_activations(s_4_3, zero_bos=True)
#%%
zero_interventions = [(*feat, 0.0) for feat in [(22, 58, 17610)]]
#%%
logits, acts = model.feature_intervention(s_4_3, zero_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
print(get_topk(logits_4_1, model.tokenizer))
#%%
boost_interventions = [(*feat, acts_4_3[feat]*2) for feat in [(22, 58, 17610)]]
#%%
logits, acts = model.feature_intervention(s_4_3, boost_interventions)
#%%
print(get_topk(logits_4_3, model.tokenizer))
print(get_topk(logits, model.tokenizer))
# %%
