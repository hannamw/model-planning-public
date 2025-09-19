#%%
from pathlib import Path
import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
#%%
model_name = 'Qwen/Qwen3-14B'
transcoder_name = "mwhanna/qwen3-14b-transcoders-lowl0"
model = ReplacementModel.from_pretrained(model_name, transcoder_name, dtype=torch.bfloat16, lazy_encoder=True)
#%%
def print_topk(logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

def chattify(inputs: list[str]):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = model.tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified
#%%
prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that shine through the night"])  
# prompt = chattify(["/no_think A rhyming couplet:\n The clouds are gray, the raindrops drop,", 
#      "<think>\n\n</think>\n\nA quiet hush upon the"])  
# prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
#      "<think>\n\n</think>\n\nSecrets and stories that sparkle"])  
# prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
#      "<think>\n\n</think>\n\nTreasure each page, let your imagination"])  
# prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
#      "<think>\n\n</think>\n\nWorlds to explore,"])  
# prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
#      "<think>\n\n</think>\n\nWorlds to explore in the night and"])  
prompt = "Para tomar el autobús, esperas en la"
prompt = "Para rezar, los musulmanes van a"
max_n_logits = 10   # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 8192  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size=256  # Batch size when attributing
offload='disk'
verbose = True  # Whether to display a tqdm progress bar and timing report
#%%
graph = attribute(
    prompt=prompt,
    model=model,
    max_n_logits=max_n_logits,
    desired_logit_prob=desired_logit_prob,
    batch_size=batch_size,
    max_feature_nodes=max_feature_nodes,
    offload=offload,
    verbose=verbose
)
#%%
slug = "la-mezquita"  # this is the name that you assign to the graph
graph_dir = 'graphs'
graph_name = f'{slug}.pt'
graph_dir = Path(graph_dir)
graph_dir.mkdir(exist_ok=True)
graph_path = graph_dir / graph_name

graph.to_pt(graph_path)

graph_file_dir = './graph_files'  # where to write the graph files. no need to make this one; create_graph_files does that for you
node_threshold=0.8  # keep only the minimum # of nodes whose cumulative influence is >= 0.8
edge_threshold=0.98  # keep only the minimum # of edges whose cumulative influence is >= 0.98

create_graph_files(
    graph_or_path=graph_path,  # the graph to create files for
    slug=slug,
    output_path=graph_file_dir,
    node_threshold=node_threshold,
    edge_threshold=edge_threshold
)
#%%
prompt = chattify(["/no_think A rhyming couplet:\n The diamond had a special gleam,", 
     "<think>\n\n</think>\n\n A light that seemed"])
#%%
prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that sparkle"])  
print(model.generate(prompt, temperature=1.0, max_new_tokens=20))
# %%
from circuit_tracer.utils.intervention_utils import decode_url_features
#%%
night_url = "http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%225007352669%22%2C%22in+%28light%29%22%5D%2C%5B%225011957109%22%2C%22during+%2F+through+the+night%22%5D%2C%5B%22350317180%22%2C%22through+%2F+in+the+night%22%5D%2C%5B%221624015498%22%2C%22say+%5C%22through%5C%22%22%5D%2C%5B%223796210509%22%2C%22say+%5C%22through%5C%22%22%5D%2C%5B%225557057141%22%2C%22during+%2F+first+thing+in+the+morning+%2F+evening%22%5D%2C%5B%221703090669%22%2C%22my%2Ftheir+whole%2Fentire+life%22%5D%2C%5B%2210467568%22%2C%22during+the+day+%2F+night+%28say+%5C%22night%5C%22%29%22%5D%2C%5B%224391016298%22%2C%22stay+up+late+%2F+all+night%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222848879356%22%2C%22shine+light%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%226178661834%22%2C%22shine%22%5D%2C%5B%228431095552%22%2C%22shine%22%5D%2C%5B%221658620776%22%2C%22shine%22%5D%2C%5B%2210256055775%22%2C%22shine+bright%22%5D%2C%5B%22220132617%22%2C%22shine+bright%22%5D%2C%5B%2213318343991%22%2C%22say+bright%22%5D%2C%5B%228214992073%22%2C%22shine+bright%22%5D%2C%5B%224218165292%22%2C%22after+%2F+evening%22%5D%2C%5B%222323335828%22%2C%22sunset+%2F+day%22%5D%2C%5B%227167576547%22%2C%22shine+brighter%22%5D%2C%5B%221968938092%22%2C%22throughout%22%5D%2C%5B%229728939252%22%2C%22dusk+moonlight%22%5D%5D&slug=delight-shine-through&pinnedIds=41_1526_36%2C35_87098_36%2C37_56953_36%2C33_58328_36%2C30_100088_36%2C28_113706_20%2C28_57695_20%2C27_116397_20%2C28_148162_20%2C29_93682_36%2C29_37510_20%2C31_4543_36%2C34_26434_36%2C34_105388_36%2C30_100088_35%2C30_91345_36%2C29_75453_36%2C36_128142_36%2C35_20946_36%2C33_57561_36%2C32_129821_36%2C31_100041_36%2C34_143185_36%2C31_111131_36%2C32_91816_36%2C32_68133_36%2C35_62716_36%2C33_139457_36%2C30_109896_36&pruningThreshold=0.47&supernodes=%5B%5B%22say+%5C%22through%5C%22%22%2C%2235_87098_36%22%2C%2237_56953_36%22%5D%2C%5B%22during+%2F+through+the+night%22%2C%2230_100088_36%22%2C%2231_4543_36%22%2C%2234_26434_36%22%5D%2C%5B%22*t%22%2C%2227_116397_20%22%2C%2228_57695_20%22%5D%2C%5B%22shine%22%2C%2233_57561_36%22%2C%2232_129821_36%22%2C%2231_111131_36%22%5D%2C%5B%22shine+bright%22%2C%2236_128142_36%22%2C%2235_20946_36%22%2C%2234_143185_36%22%5D%5D&clickedId=33_139457_36"
night_features, _ = decode_url_features(night_url)
#%%
night_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that shine"])
night_logits, night_acts = model.get_activations(night_prompt)
#%%
bright_url = "http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%225000249971%22%2C%22say+l*%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%225007352669%22%2C%22in+%28light%29%22%5D%2C%5B%225011957109%22%2C%22during+%2F+through+the+night%22%5D%2C%5B%22350317180%22%2C%22through+%2F+in+the+night%22%5D%2C%5B%221624015498%22%2C%22say+%5C%22through%5C%22%22%5D%2C%5B%223796210509%22%2C%22say+%5C%22through%5C%22%22%5D%2C%5B%225557057141%22%2C%22during+%2F+first+thing+in+the+morning+%2F+evening%22%5D%2C%5B%221703090669%22%2C%22my%2Ftheir+whole%2Fentire+life%22%5D%2C%5B%2210467568%22%2C%22during+the+day+%2F+night+%28say+%5C%22night%5C%22%29%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222848879356%22%2C%22shine+light%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%229281826840%22%2C%22at+%28night%29%22%5D%2C%5B%224549387540%22%2C%22overnight%22%5D%2C%5B%227439865121%22%2C%22dark%22%5D%2C%5B%226178661834%22%2C%22shine%22%5D%2C%5B%228431095552%22%2C%22shine%22%5D%2C%5B%221658620776%22%2C%22shine%22%5D%2C%5B%2210256055775%22%2C%22shine+bright%22%5D%2C%5B%22220132617%22%2C%22shine+bright%22%5D%2C%5B%2213387633859%22%2C%22say+br%22%5D%2C%5B%226198242092%22%2C%22bright+%2F+say+bl%22%5D%2C%5B%2213318343991%22%2C%22say+bright%22%5D%2C%5B%224218165292%22%2C%22after+%2F+evening%22%5D%2C%5B%222323335828%22%2C%22sunset+%2F+day%22%5D%2C%5B%227167576547%22%2C%22shine+brighter%22%5D%2C%5B%22368601938%22%2C%22say+b%22%5D%2C%5B%224701638895%22%2C%22don%27t+say+bright%22%5D%5D&slug=deslight-that-sparkle&pinnedIds=41_323_36%2C31_99970_20%2C30_85020_20%2C29_437_20%2C30_91345_36%2C30_109896_35%2C30_100088_36%2C30_91345_35%2C30_100088_35%2C28_113706_20%2C28_57695_20%2C28_148162_20%2C31_4543_36%2C33_58328_36%2C32_33055_36%2C32_37666_36%2C32_91816_36%2C31_54812_36%2C35_136212_36%2C37_95349_36%2C29_75453_36%2C31_111131_36%2C32_129821_36%2C33_57561_36%2C30_99868_36%2C28_5224_36%2C34_143185_36%2C35_20946_36%2C37_119691_36%2C36_11765_36%2C36_163594_36%2C37_111301_36%2C34_91936_36%2C36_163170_36%2C37_27113_36%2C39_96930_36&supernodes=%5B%5B%22say+bright%22%2C%2236_163594_36%22%2C%2237_111301_36%22%2C%2239_96930_36%22%2C%2237_27113_36%22%2C%2236_163170_36%22%5D%2C%5B%22shine+bright%22%2C%2234_143185_36%22%2C%2235_20946_36%22%2C%2237_119691_36%22%5D%2C%5B%22during+%2F+through+the+night%22%2C%2230_100088_36%22%2C%2231_4543_36%22%5D%5D&clickedId=37_95349_36&pruningThreshold=0.63"
bright_features, _ = decode_url_features(bright_url)
#%%
bright_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that sparkle"])
bright_logits, bright_acts = model.get_activations(bright_prompt)
#%%
one_url = "http://localhost:8046/index.html?clerps=%5B%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%224310050563%22%2C%22imagination+transporing+places%22%5D%2C%5B%223485248272%22%2C%22verse%22%5D%2C%5B%225042540070%22%2C%22rhyming+verse%22%5D%2C%5B%228878380856%22%2C%22rhyming+verse%22%5D%2C%5B%22152591688%22%2C%22rhyming+verse%22%5D%2C%5B%223644402589%22%2C%22one-to%2Fon-one%22%5D%2C%5B%228790518085%22%2C%22one%22%5D%2C%5B%223328260046%22%2C%22one%22%5D%2C%5B%223453594460%22%2C%22one%22%5D%2C%5B%229757835416%22%2C%22one+by+one%22%5D%2C%5B%221785240349%22%2C%22fun%22%5D%2C%5B%223167403406%22%2C%22fun%22%5D%2C%5B%225614025675%22%2C%22fun%22%5D%2C%5B%224676557084%22%2C%22X-to-X%22%5D%2C%5B%229679204512%22%2C%22X-by%2Ffor-X%22%5D%2C%5B%224962022356%22%2C%22X-by%2Ffor-X%22%5D%2C%5B%2210262215179%22%2C%22Article+I%2C+Section+X%22%5D%5D&slug=fun-worlds-to-explore&pinnedIds=41_11_34%2C31_59721_34%2C35_85338_34%2C34_139663_34%2C31_81555_34%2C35_132557_34%2C34_83074_34%2C29_79561_20%2C27_105934_20%2C31_96679_34%2C33_99585_34%2C32_139101_34%2C36_143226_34&pruningThreshold=0.64&clickedId=34_83074_34&supernodes=%5B%5B%22X-by%2Ffor-X%22%2C%2233_99585_34%22%2C%2232_139101_34%22%2C%2231_96679_34%22%5D%2C%5B%22one%22%2C%2231_81555_34%22%2C%2234_83074_34%22%2C%2235_132557_34%22%5D%2C%5B%22one+by+one%22%2C%2234_139663_34%22%2C%2235_85338_34%22%5D%5D"
one_features, _ = decode_url_features(one_url)
#%%
one_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore"])
one_logits, one_acts = model.get_activations(one_prompt)
#%%
upweight_night = [(l,p, f, 2 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
zero_bright = [(*feat, 0) for feat in bright_features['say bright']]
zero_bright = [(*feat, -2 * bright_acts[feat]) for feat in bright_features['shine bright']]
#%%
generation, logits, acts = model.feature_intervention_generate(bright_prompt, upweight_night + zero_bright)
print(generation)
# %%
upweight_bright = [(*feat, 2 * bright_acts[feat]) for feat in bright_features['say bright']]
upweight_bright = [(*feat, 2 * bright_acts[feat]) for feat in bright_features['shine bright']]
zero_night = [(*feat, -2 * night_acts[feat]) for feat in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(night_prompt, upweight_bright + zero_night)
print(generation)
# %%
upweight_one = [(l, slice(night_acts.size(1)-1,None), f, 6 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
zero_night = [(l,slice(p, None), f, -6 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(night_prompt, upweight_one + zero_night)
print(generation)
# %%
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_night = [(l, slice(34,None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt, upweight_night + zero_one)
print(generation)
# %%
# %%
upweight_one = [(l, slice(bright_acts.size(1)-1,None), f, 6 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
zero_bright = [(l, slice(p, None), f, -2 * bright_acts[l,p,f]) for l,p,f in bright_features['say bright']]
#%%
generation, logits, acts = model.feature_intervention_generate(bright_prompt, upweight_one + zero_bright)
print(generation)
# %%
# %%
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_bright = [(l, slice(34,None), f, 5 * bright_acts[l,p,f]) for l,p,f in bright_features['say bright']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt, upweight_bright + zero_one)
print(generation)
# %%
one_prompt_extended = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore, one by"])
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_night = [(l, slice(34,None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt_extended, upweight_night + zero_one)
print(generation)
# %%
one_prompt_extended = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore, one"])
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_night = [(l, slice(34,None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt_extended, upweight_night + zero_one)
print(generation)
# %%
one_prompt_extended = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore,"])
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_night = [(l, slice(34,None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt_extended, upweight_night + zero_one)
print(generation)
# %%
one_prompt_extended = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore"])
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_night = [(l, slice(34,None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt_extended, upweight_night + zero_one)
print(generation)
# %%
one_prompt_extended = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore,"])
zero_one = [(l, slice(p, None), f, -2 * one_acts[l,p,f]) for l,p,f in one_features['one'] + one_features['one by one'] + one_features['X-by/for-X']]
#upweight_night = [(l, -1, f, 10 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
upweight_bright = [(l, slice(34,None), f, 10 * bright_acts[l,p,f]) for l,p,f in bright_features['shine bright']]
#%%
generation, logits, acts = model.feature_intervention_generate(one_prompt_extended, upweight_bright + zero_one)
print(generation)
# %%
night_prompt_extended = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that"])
upweight_one = [(l, slice(night_acts.size(1)-2,None), f, 6 * one_acts[l,p,f]) for l,p,f in one_features['one']]# + one_features['one by one']]# + one_features['X-by/for-X']]
zero_night = [(l,slice(p-1, None), f, -2 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
#%%
generation, logits, acts = model.feature_intervention_generate(night_prompt_extended, upweight_one + zero_night)
print(generation)
# %%
near_ends_url = 'http://localhost:8046/index.html?clerps=%5B%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%224737099744%22%2C%22near+ends+of+lines%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%225042540070%22%2C%22near+ends+of+lines%22%5D%2C%5B%223167403406%22%2C%22fun%22%5D%2C%5B%225614025675%22%2C%22fun%22%5D%2C%5B%22125080805%22%2C%22than+%28should+be+that%29%22%5D%2C%5B%223689656%22%2C%22fun%22%5D%2C%5B%22247075299%22%2C%22un%22%5D%2C%5B%225121883830%22%2C%22*n%22%5D%2C%5B%223536110626%22%2C%22*n%22%5D%2C%5B%228423176283%22%2C%22*N%22%5D%2C%5B%227876066244%22%2C%22*N%22%5D%2C%5B%2213270404215%22%2C%22n%22%5D%2C%5B%22278987613%22%2C%22ends+of+lines%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%22868840416%22%2C%22before+comma%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%2212137521087%22%2C%22say+n%22%5D%2C%5B%225613389926%22%2C%22*n%22%5D%2C%5B%2210385503357%22%2C%22*n*%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%22544120533%22%2C%22near+ends+of+lines%22%5D%2C%5B%222631279666%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211326253752%22%2C%22ends+of+lines%22%5D%2C%5B%224493851775%22%2C%22near+ends+of+lines%22%5D%2C%5B%221269953974%22%2C%22near+ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%222829588358%22%2C%22ends+of+lines%22%5D%2C%5B%222742552869%22%2C%22*n*%22%5D%2C%5B%228755283601%22%2C%22say+*n%22%5D%2C%5B%2211546756568%22%2C%22say+%28n%22%5D%2C%5B%2210835465615%22%2C%22*n%22%5D%2C%5B%221712148865%22%2C%22*N%22%5D%2C%5B%222283359213%22%2C%22*n%22%5D%2C%5B%229105683742%22%2C%22night%22%5D%2C%5B%223174331648%22%2C%22night%22%5D%2C%5B%227670835553%22%2C%22night%22%5D%2C%5B%2211401934513%22%2C%22before+comma%22%5D%2C%5B%22869841158%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%224044646732%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%22206542611%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%221061061177%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%22389163124%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%22564463177%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211772300385%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2212434408432%22%2C%22verse+%2F+meter%3F%22%5D%5D&slug=fun-worlds-to-explore-in-the-night&pinnedIds=41_323_37%2C29_2686_20%2C31_59721_37%2C28_65944_20%2C27_105934_20%2C29_79561_20%2C30_15785_37%2C30_15785_34%2C33_39042_37%2C35_22193_37%2C32_32955_37%2C33_46032_37%2C36_118775_37%2C36_25940_37%2C35_101175_37%2C29_84066_20%2C37_129755_37%2C33_125473_37%2C25_162887_37%2C26_132300_37%2C26_151938_37%2C26_116980_37%2C22_155781_37%2C23_144097_37%2C19_105936_20%2C21_74039_20%2C24_61_37%2C37_58479_37%2C39_147170_37%2C39_86757_37%2C39_37427_37%2C39_67537_37%2C39_73586_37%2C37_89902_37%2C32_83456_37%2C38_20285_37%2C36_41672_37%2C41_11_37%2C38_41646_37%2C37_123823_37%2C29_37510_37%2C32_79645_37%2C32_134916_37%2C31_150977_37%2C35_97299_37%2C26_17442_37%2C28_67123_37%2C23_12385_37%2C20_81910_37%2C18_73542_36%2C12_61386_37%2C15_43513_37%2C18_44689_37%2C17_23603_37%2C16_6934_37%2C19_75207_37%2C20_60314_37%2C18_16919_37%2C29_100394_37%2C26_27871_37%2C29_72513_37%2C25_150481_37%2C26_52769_37%2C27_109248_37%2C29_116923_37%2C28_50368_37%2C30_94772_37%2C26_56839_37%2C15_144713_37%2C17_153424_37%2C18_157679_37%2C22_33576_37%2CE_3729_37%2C17_140798_21%2C6_76533_21&pruningThreshold=0.5&supernodes=%5B%5B%22ends+of+lines%22%2C%2215_43513_37%22%2C%2225_150481_37%22%2C%2220_60314_37%22%2C%2218_16919_37%22%2C%2217_23603_37%22%2C%2219_75207_37%22%5D%2C%5B%22fun%22%2C%2229_2686_20%22%2C%2229_79561_20%22%2C%2227_105934_20%22%5D%2C%5B%22night%22%2C%2232_134916_37%22%2C%2232_79645_37%22%2C%2237_123823_37%22%2C%2229_37510_37%22%5D%2C%5B%22before+comma%22%2C%2231_150977_37%22%2C%2238_41646_37%22%5D%2C%5B%22*N%22%2C%2225_162887_37%22%2C%2223_144097_37%22%2C%2226_132300_37%22%2C%2226_151938_37%22%2C%2222_155781_37%22%2C%2221_74039_20%22%2C%2219_105936_20%22%2C%2229_84066_20%22%2C%2233_125473_37%22%2C%2235_101175_37%22%2C%2237_129755_37%22%2C%2239_147170_37%22%2C%2237_58479_37%22%2C%2239_67537_37%22%2C%2230_15785_37%22%5D%2C%5B%22verse+%2F+meter%3F%22%2C%2215_144713_37%22%2C%2217_153424_37%22%2C%2218_157679_37%22%2C%2226_27871_37%22%2C%2238_20285_37%22%2C%2236_41672_37%22%2C%2237_89902_37%22%2C%2233_46032_37%22%5D%2C%5B%22near+ends+of+lines%22%2C%2229_100394_37%22%2C%2222_33576_37%22%2C%2230_94772_37%22%2C%2229_72513_37%22%2C%2235_97299_37%22%2C%2232_32955_37%22%2C%2232_83456_37%22%2C%2228_67123_37%22%2C%2223_12385_37%22%2C%2228_50368_37%22%2C%2226_56839_37%22%5D%5D&clickedId=39_73586_37'
near_ends_features, _ = decode_url_features(near_ends_url)
# %%
explore_in_the_night_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
     "<think>\n\n</think>\n\nWorlds to explore in the night"])
_, explore_in_the_night_acts = model.get_activations(explore_in_the_night_prompt)
print(model.generate(explore_in_the_night_prompt))
#%%
zero_neol = [(l, slice(-1, None), f, -3* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines']  + near_ends_features['verse / meter?']]
generation, logits, acts = model.feature_intervention_generate(explore_in_the_night_prompt, zero_neol, freeze_attention=False)
print(generation)

# %%
ends_url = 'http://localhost:8046/index.html?clerps=%5B%5B%22704644540%22%2C%22night%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%22999424967%22%2C%22ends+of+lines%22%5D%2C%5B%22278987613%22%2C%22ends+of+lines%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%2224161659%22%2C%22end+of+line%22%5D%2C%5B%223356385325%22%2C%22near+the+end+of+a+line%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%223044847626%22%2C%22say+.%22%5D%2C%5B%2287047377%22%2C%22say+.%22%5D%2C%5B%2210081360976%22%2C%22say+.%22%5D%2C%5B%225814571009%22%2C%22ends+of+lines%22%5D%2C%5B%226839060551%22%2C%22ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%221564222247%22%2C%22ends+of+lines%22%5D%2C%5B%2211200629255%22%2C%22ends+of+lines%22%5D%2C%5B%2212327718682%22%2C%22ends+of+lines%22%5D%2C%5B%226124643803%22%2C%22ends+of+lines%22%5D%2C%5B%226523161287%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%22855262740%22%2C%22ends+of+lines%22%5D%2C%5B%22305724606%22%2C%22ends+of+lines%22%5D%2C%5B%221089721236%22%2C%22ends+of+lines%22%5D%5D&slug=delight-shine-through-the-night-full&pinnedIds=41_13_39%2C19_75207_39%2C18_16919_39%2C18_44689_39%2C17_23603_39%2C15_43513_39%2C6_76533_21%2C5_80565_11%2C7_98349_21%2C5_105574_39%2C16_6934_39%2C6_94630_11%2C18_139838_39%2C17_153424_39%2C29_116923_39%2C27_156992_39%2C22_110653_39%2C22_114197_39%2C17_140798_21%2C17_140798_23%2C12_102602_12%2CE_3729_39%2CE_17970_20%2C0_70799_39%2C0_86856_39%2C19_102121_21%2C20_60314_39%2C20_41337_39%2C20_142282_39%2C21_24705_39%2C28_145505_39%2C29_149640_39%2C30_55901_39%2C31_107806_39%2C33_46650_39%2C39_77996_39%2C37_13156_39%2C33_141961_39&pruningThreshold=0.49&supernodes=%5B%5B%22end+of+line+%28delight%29%22%2C%2217_140798_23%22%2C%2217_140798_21%22%2C%2219_102121_21%22%5D%2C%5B%22say+.%22%2C%2233_141961_39%22%2C%2237_13156_39%22%2C%2239_77996_39%22%5D%2C%5B%22ends+of+lines%22%2C%2233_46650_39%22%2C%2221_24705_39%22%2C%2230_55901_39%22%2C%2229_149640_39%22%2C%2231_107806_39%22%2C%2229_116923_39%22%2C%2228_145505_39%22%2C%2222_110653_39%22%2C%2220_142282_39%22%2C%2227_156992_39%22%2C%2218_16919_39%22%2C%2220_41337_39%22%2C%2216_6934_39%22%2C%2217_23603_39%22%2C%2218_44689_39%22%2C%2220_60314_39%22%2C%2222_114197_39%22%2C%2215_43513_39%22%5D%5D&clickedId=18_44689_39'
ends_features, _ = decode_url_features(ends_url)
full_night_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that shine through the night"])
full_night_logits, full_night_acts = model.get_activations(full_night_prompt)
#%%
upweight_eol = [(l, -1, f, 2 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
# %%
generation, logits, acts = model.feature_intervention_generate(explore_in_the_night_prompt, zero_neol + upweight_eol, freeze_attention=False)
print(generation)
#%%
zero_neol = [(l, slice(-1, None), f, -2.5* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines']  + near_ends_features['verse / meter?']]
upweight_eol = [(l, -1, f, 2 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
generation, logits, acts = model.feature_intervention_generate(explore_in_the_night_prompt, upweight_eol, freeze_attention=False)
print(generation)
# %%
upweight_neol = [(l, slice(-1, None), f, 2.5* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines'] + near_ends_features['verse / meter?']]
zero_eol = [(l, slice(-1, None), f, -2 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
#%%
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, upweight_neol + zero_eol, freeze_attention=False)
print(generation)
# %%
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, zero_eol, freeze_attention=False)
print(generation)
#%%
zero_neol = [(l, slice(-1, None), f, -2* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines'] + near_ends_features['verse / meter?']]
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, zero_eol + zero_neol, freeze_attention=False)
print(generation)
#%%
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, upweight_neol, freeze_attention=False)
print(generation)
# %%
zero_eol = [(l, -1, f, -2 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, zero_eol, freeze_attention=False)
print(generation)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
print(model.generate(park_prompt, do_sample=False))
# %%
upweight_neol = [(l, slice(-1, None), f, 5* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines'] + near_ends_features['verse / meter?']]
generation, logits, acts = model.feature_intervention_generate(park_prompt, upweight_neol, freeze_attention=False)
print(generation)
# %%
park_prompt = chattify(["/no_think A rhyming couplet:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
print(model.generate(park_prompt, do_sample=False))
# %%
near_ends_url2 = 'http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%225000249971%22%2C%22say+l*%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%225007352669%22%2C%22in+%28light%29%22%5D%2C%5B%225011957109%22%2C%22during+%2F+through+the+night%22%5D%2C%5B%22350317180%22%2C%22through+%2F+in+the+night%22%5D%2C%5B%225557057141%22%2C%22during+%2F+first+thing+in+the+morning+%2F+evening%22%5D%2C%5B%2210467568%22%2C%22during+the+day+%2F+night+%28say+%5C%22night%5C%22%29%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%229281826840%22%2C%22say+night%22%5D%2C%5B%224549387540%22%2C%22overnight%22%5D%2C%5B%227439865121%22%2C%22dark%22%5D%2C%5B%224218165292%22%2C%22after+%2F+evening%22%5D%2C%5B%222323335828%22%2C%22sunset+%2F+day%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%228439798047%22%2C%22*it%22%5D%2C%5B%224737099744%22%2C%22near+ends+of+lines%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%22152591688%22%2C%22near+ends+of+lines%22%5D%2C%5B%228589606881%22%2C%22say+night%22%5D%2C%5B%2264974264%22%2C%22say+night%22%5D%2C%5B%2211788263797%22%2C%22say+night%22%5D%2C%5B%22866799029%22%2C%22night%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%2224161659%22%2C%22end+of+line%22%5D%2C%5B%223356385325%22%2C%22near+the+end+of+a+line%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211326253752%22%2C%22ends+of+lines%22%5D%2C%5B%221393735179%22%2C%22ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%226839060551%22%2C%22ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%22305724606%22%2C%22ends+of+lines%22%5D%2C%5B%222829588358%22%2C%22ends+of+lines%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2211772300385%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2212434408432%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%226553978%22%2C%22near+ends+of+lines%22%5D%2C%5B%22151336472%22%2C%22near+ends+of+lines%22%5D%2C%5B%222926239720%22%2C%22near+ends+of+lines%22%5D%2C%5B%223518556300%22%2C%22through%22%5D%2C%5B%222260911364%22%2C%22near+ends+of+lines%22%5D%2C%5B%2212280500812%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211292264592%22%2C%22near+ends+of+lines%22%5D%2C%5B%221940987640%22%2C%22near+ends+of+lines%22%5D%2C%5B%223498117522%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210913184429%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210589563188%22%2C%22near+ends+of+lines%22%5D%2C%5B%2213374055896%22%2C%22near+ends+of+lines%22%5D%5D&slug=delight-shine-through-the-night&pinnedIds=41_3729_38%2C35_136212_38%2C33_131035_38%2C30_100088_38%2C26_98927_20%2C27_83859_38%2C28_113706_20%2C28_57695_20%2C29_37510_20%2C27_83859_37%2C18_157679_36%2C30_100088_35%2C31_4543_36%2C30_100088_34%2C33_129887_38%2C31_4543_38%2C32_146050_38%2C34_26434_38%2C37_95349_38%2C35_11363_38%2C36_41599_38%2C33_153512_38%2C30_109896_38%2C27_116397_20%2C36_41599_37%2C31_4543_37%2C34_105388_37%2C26_155381_20%2C20_81910_38%2C23_12385_38%2C28_67123_38%2C23_83619_38%2C23_147713_38%2C24_62280_38%2C25_67218_38%2C28_150252_38%2C26_56839_38%2C26_145503_38%2C27_156691_38%2C29_163518_38%2C30_76470_38%2C31_3588_38%2C30_17366_38&pruningThreshold=0.64&supernodes=%5B%5B%22right%22%2C%2228_113706_20%22%2C%2227_116397_20%22%2C%2228_57695_20%22%5D%2C%5B%22through+%2F+in+the+night%22%2C%2234_26434_38%22%2C%2231_4543_38%22%2C%2230_100088_38%22%5D%2C%5B%22say+night%22%2C%2233_153512_38%22%2C%2235_11363_38%22%2C%2233_131035_38%22%2C%2235_136212_38%22%5D%2C%5B%22near+ends+of+lines%22%2C%2220_81910_38%22%2C%2223_147713_38%22%2C%2223_83619_38%22%2C%2231_3588_38%22%2C%2229_163518_38%22%2C%2230_17366_38%22%2C%2230_76470_38%22%2C%2228_150252_38%22%2C%2227_156691_38%22%2C%2225_67218_38%22%2C%2226_145503_38%22%2C%2224_62280_38%22%2C%2228_67123_38%22%2C%2226_56839_38%22%2C%2223_12385_38%22%5D%5D&clickedId=20_81910_38'
near_ends_features2, _ = decode_url_features(near_ends_url2)
one_off_night_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that shine through the"])
one_off_night_logits, one_off_night_acts = model.get_activations(one_off_night_prompt)
upweight_neol2 = [(l, slice(-1, None), f, 3 * one_off_night_acts[l,p,f]) for l,p,f in near_ends_features2['near ends of lines']]
#%%
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, upweight_neol2, freeze_attention=False)
print(generation)
#%%
early_rhyme_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets"])
generation, logits, acts = model.feature_intervention_generate(early_rhyme_prompt, upweight_neol2, freeze_attention=False)
print(generation)
# %%
zero_eol = [(l, slice(-1, None), f, -2 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
generation, logits, acts = model.feature_intervention_generate(full_night_prompt, zero_eol + upweight_neol2, freeze_attention=False)
print(generation)
# %%
for i, tok in enumerate(model.tokenizer.convert_ids_to_tokens(model.tokenizer(park_prompt).input_ids)):
     print(i, tok)
# %%
upweight_eol = [(l, 15, f, 4 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
upweight_neol = [(l, slice(-1, None), f, 5* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines'] + near_ends_features['verse / meter?']]
generation, logits, acts = model.feature_intervention_generate(park_prompt, upweight_eol + upweight_neol, freeze_attention=False)
print(generation)
# %%
zero_neol = [(l, slice(-1, None), f, -2* explore_in_the_night_acts[l,p,f]) for l,p,f in near_ends_features['near ends of lines'] + near_ends_features['verse / meter?']]
one_off_night_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that shine through the"])
generation, logits, acts = model.feature_intervention_generate(one_off_night_prompt,zero_neol, freeze_attention=False)
print(generation)
# %%
upweight_eol = [(l, -1, f, 4 * full_night_acts[l,p,f]) for l,p,f in ends_features['ends of lines']]
one_off_night_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nSecrets and stories that shine through the"])
generation, logits, acts = model.feature_intervention_generate(one_off_night_prompt,upweight_eol, freeze_attention=False)
print(generation)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
upweight_night = [(l, slice(-1,None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
generation, logits, acts = model.feature_intervention_generate(park_prompt,upweight_night, freeze_attention=False)
print(generation)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
upweight_night = [(l, -1, f, 20 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
generation, logits, acts = model.feature_intervention_generate(park_prompt,upweight_night, freeze_attention=False)
print(generation)
# %%
acct_prompt = chattify(["/no_think Someone who studies living organisms is a biologist. Someone who manages financial records is"])
upweight_night = [(l, slice(-1, None), f, 5 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
generation, logits, acts = model.feature_intervention_generate(acct_prompt,upweight_night, freeze_attention=False)
print(generation)
# %%
accountant_url = 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-an-accountant&clerps=%5B%5B%223374%22%2C%22financial%22%5D%2C%5B%2273538%22%2C%22financial%22%5D%2C%5B%2258486%22%2C%22is%22%5D%2C%5B%22155776%22%2C%22financial%22%5D%2C%5B%22116874%22%2C%22financial%22%5D%2C%5B%2282999%22%2C%22exchequer%22%5D%2C%5B%2225022%22%2C%22checkbook+%2F+expenses%22%5D%2C%5B%2210233%22%2C%22accounting+%2F+bookkeeping%22%5D%2C%5B%22121438%22%2C%22doctors+%2F+physicians%22%5D%2C%5B%2247756%22%2C%22CPAs%22%5D%2C%5B%2243609%22%2C%22accounting%22%5D%2C%5B%22115876%22%2C%22random+choice%22%5D%2C%5B%2220774%22%2C%22is%22%5D%2C%5B%2251498%22%2C%22careers%22%5D%2C%5B%2288830%22%2C%22is%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%2269329%22%2C%22%28dense+feature%29%22%5D%2C%5B%22115000%22%2C%22say+a+%3E+an%22%5D%2C%5B%2255618%22%2C%22banking%22%5D%2C%5B%2223813%22%2C%22finances%22%5D%2C%5B%2219697%22%2C%22say+a+%3E+an%22%5D%2C%5B%22121960%22%2C%22computational+tools%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2235919%22%2C%22say+a+%3E+an%22%5D%2C%5B%22111811%22%2C%22accounting%22%5D%2C%5B%2240107%22%2C%22economics+%2F+accounting%22%5D%2C%5B%2231593%22%2C%22finance%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%22136526%22%2C%22the+doctor%22%5D%2C%5B%2296539%22%2C%22as+a+%3Cprofession%3E%22%5D%2C%5B%2276404%22%2C%22careers%22%5D%2C%5B%22135260%22%2C%22accounting%22%5D%2C%5B%2245780%22%2C%22say+a+%3E+an%22%5D%2C%5B%22152499%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%2221020%22%2C%22accounting%22%5D%2C%5B%2287234%22%2C%22say+a+%3E+an%22%5D%2C%5B%2210752%22%2C%22accounting%22%5D%2C%5B%2272914%22%2C%22accounting%22%5D%2C%5B%22117738%22%2C%22audit%22%5D%2C%5B%2243347%22%2C%22finances%22%5D%2C%5B%22143636%22%2C%22say+a+%3E+an%22%5D%2C%5B%22159252%22%2C%22say+%5C%22acc%5C%22+%28account%29%22%5D%2C%5B%2285195%22%2C%22say+%5C%22acc%5C%22%22%5D%2C%5B%22160499%22%2C%22say+a+%3E+an+%22%5D%2C%5B%2244167%22%2C%22%28chartered%29+accountant%22%5D%2C%5B%22125138%22%2C%22dense+feature%22%5D%2C%5B%22159184%22%2C%22say+a+%3E+an%22%5D%2C%5B%2229458%22%2C%22%5C%22+ac%5C%22+%28don%27t+say+%5C%22+ac%5C%22%29%22%5D%2C%5B%2292068%22%2C%22accounts%22%5D%2C%5B%22100905%22%2C%22say+an+%3E+a%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%22140516%22%2C%22say+%5C%22a*%5C%22%22%5D%2C%5B%2276520%22%2C%22say+%5C%22+a*%5C%22%22%5D%2C%5B%2213446%22%2C%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%5D%2C%5B%2232592%22%2C%22say+%5C%22a*%5C%22%22%5D%2C%5B%22103617%22%2C%22dense+feature%22%5D%2C%5B%2242344%22%2C%22say+an+%3E+a+%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%2252418%22%2C%22don%27t+say+a%3E+an%22%5D%2C%5B%2284175%22%2C%22say+a+%3E+an%22%5D%2C%5B%22128296%22%2C%22dense+feature%22%5D%5D&pinnedIds=41_458_17%2C38_42344_17%2C35_159252_17%2C37_76520_17%2C39_52418_17%2C32_135260_17%2C34_72914_17%2C33_21020_17%2C36_29458_17%2C35_85195_17%2C36_92068_17%2C37_32592_17%2C33_87234_17%2C37_13446_17%2C35_160499_17%2C36_100905_17%2C32_45780_17%2C31_35919_17%2C29_115000_17%2C33_10752_17%2C35_44167_17%2C31_111811_17%2C34_43347_17%2C30_23813_17%2C24_43609_16%2C23_47756_16%2C8_10233_16%2C1_116874_15%2C0_3374_15%2C0_73538_15%2C1_155776_15%2CE_5896_15%2C31_40107_17%2C31_31593_17%2C37_103617_17%2C35_125138_17%2C36_103390_17&supernodes=%5B%5B%22say+%5C%22an%5C%22+%3E+%5C%22a%5C%22%22%2C%2236_100905_17%22%2C%2231_35919_17%22%2C%2232_45780_17%22%2C%2229_115000_17%22%2C%2233_87234_17%22%2C%2235_160499_17%22%2C%2237_13446_17%22%5D%2C%5B%22say+%5C%22a*%5C%22%22%2C%2237_32592_17%22%2C%2237_76520_17%22%5D%2C%5B%22financial%22%2C%220_73538_15%22%2C%221_155776_15%22%2C%221_116874_15%22%2C%220_3374_15%22%5D%2C%5B%22finance%22%2C%2231_31593_17%22%2C%2230_23813_17%22%2C%2234_43347_17%22%5D%2C%5B%22say+%5C%22acc%5C%22%22%2C%2235_85195_17%22%2C%2235_159252_17%22%5D%2C%5B%22accounting%22%2C%2223_47756_16%22%2C%2224_43609_16%22%2C%228_10233_16%22%5D%2C%5B%22accounting%22%2C%2231_40107_17%22%2C%2231_111811_17%22%2C%2233_10752_17%22%2C%2234_72914_17%22%2C%2233_21020_17%22%2C%2232_135260_17%22%2C%2235_44167_17%22%5D%2C%5B%22%3F%3F%3F%22%2C%2236_103390_17%22%2C%2235_125138_17%22%5D%2C%5B%22say+%5C%22an%5C%22%22%2C%2237_103617_17%22%2C%2238_42344_17%22%5D%5D&clickedId=31_111811_17'
accountant_features, _ = decode_url_features(accountant_url)
acct_prompt = chattify(["Someone who studies living organisms is a biologist. Someone who manages financial records is"])
acct_logits, acct_acts = model.get_activations(acct_prompt)
#%%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
     # 3x works well too
upweight_accountant = [(l, slice(-1, None), f, 1.5 * acct_acts[l,p,f]) for l,p,f in accountant_features['accounting (2)']]
generation, logits, acts = model.feature_intervention_generate(park_prompt,upweight_accountant, freeze_attention=False)
print(generation)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
upweight_accountant = [(l, -1, f, 6 * acct_acts[l,p,f]) for l,p,f in accountant_features['accounting (2)']]
generation, logits, acts = model.feature_intervention_generate(park_prompt,upweight_accountant, freeze_attention=False)
print(generation)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
upweight_accountant = [(l, slice(-1, None), f, 5 * acct_acts[l,p,f]) for l,p,f in accountant_features['accounting (2)']]
generation, logits, acts = model.feature_intervention_generate(park_prompt,upweight_accountant, freeze_attention=False)
print(generation)
# %%
model.generate("Para tomar el autobús, esperas en una nueva y", do_sample=False)
# %%
model.tokenizer.convert_ids_to_tokens(model.tokenizer('Para tomar el autobús, esperas en una parada de autobús. A esa par').input_ids)
# %%
model.generate("Para tomar el autobús, esperas en la", do_sample=False)

# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
upweight_accountant = [(l, slice(-1, None), f, 2 * acct_acts[l,p,f]) for l,p,f in accountant_features['accounting (2)']]
logits, acts = model.feature_intervention(park_prompt,upweight_accountant, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to chase"])
upweight_accountant = [(l, slice(-1, None), f, 3 * acct_acts[l,p,f]) for l,p,f in accountant_features['accounting (2)']]
logits, acts = model.feature_intervention(park_prompt,upweight_accountant, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that was playing"])
upweight_accountant = [(l, slice(-1, None), f, 1.5 * acct_acts[l,p,f]) for l,p,f in accountant_features['accounting (2)']]
logits, acts = model.feature_intervention(park_prompt,upweight_accountant, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
upweight_night = [(l,-1, f, 4 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
logits, acts = model.feature_intervention(park_prompt,upweight_night, freeze_attention=False)
print_topk(logits)
#%%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to run"])
upweight_night = [(l,-1, f, 4 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
logits, acts = model.feature_intervention(park_prompt,upweight_night, freeze_attention=False)
print_topk(logits)
#%%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to run at"])
upweight_night = [(l,-1, f, 4 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
logits, acts = model.feature_intervention(park_prompt,upweight_night, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that started to"])
logits, acts = model.feature_intervention(park_prompt,[], freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that was running"])
upweight_night = [(l,-1, f, 6 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
logits, acts = model.feature_intervention(park_prompt,upweight_night, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that was very"])
upweight_bright = [(l, -1, f, 7 * bright_acts[l,p,f]) for l,p,f in bright_features['say bright']]
#upweight_bright = [(l, -1, f, 6 * bright_acts[l,p,f]) for l,p,f in bright_features['shine bright']]
logits, acts = model.feature_intervention(park_prompt,upweight_bright, freeze_attention=False)
print_topk(logits)
# %%
cats_url = "http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-is-4-3-cats&clerps=%5B%5B%22102506%22%2C%22sentences%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%22113816%22%2C%22say+indexing+operation%22%5D%2C%5B%22122195%22%2C%22one+%28other+numbers+too%29%22%5D%2C%5B%22106343%22%2C%22one%22%5D%2C%5B%22137034%22%2C%22one%22%5D%2C%5B%2227454%22%2C%22one%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%22126846%22%2C%22don%27t+say+single%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2246815%22%2C%22is%22%5D%2C%5B%2260610%22%2C%22one%22%5D%2C%5B%2232177%22%2C%22Chicago+Style%22%5D%2C%5B%22153029%22%2C%22one%22%5D%2C%5B%2285338%22%2C%22one%22%5D%2C%5B%22125138%22%2C%22dense+feature%22%5D%2C%5B%22132557%22%2C%22one%22%5D%2C%5B%2286640%22%2C%22say+1%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2238531%22%2C%22one%22%5D%2C%5B%2267967%22%2C%22say+1%22%5D%2C%5B%22103617%22%2C%22dense+feature%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22129530%22%2C%22one%22%5D%5D&pinnedIds=41_374_58%2C39_101253_58%2C37_93033_58%2C35_55751_58%2C35_30068_58%2C32_149321_58%2C27_108259_58%2C25_148696_58%2C20_113026_58%2C39_19350_58%2C38_118668_39%2C39_120945_58%2C38_12959_58%2C32_46815_58%2C37_67967_58%2C36_86640_58%2C35_157415_58%2C35_85338_58%2C29_106343_58%2C33_60610_58%2C35_101974_58%2C30_137034_58%2C28_122195_58%2C31_126846_58%2C30_27454_58%2C34_153029_58%2C35_132557_58%2C36_38531_58%2C38_129530_58&pruningThreshold=0.53&supernodes=%5B%5B%22one%22%2C%2229_106343_58%22%2C%2230_137034_58%22%2C%2238_129530_58%22%2C%2236_38531_58%22%2C%2235_132557_58%22%2C%2234_153029_58%22%2C%2235_85338_58%22%2C%2233_60610_58%22%2C%2230_27454_58%22%2C%2237_67967_58%22%2C%2236_86640_58%22%5D%5D&clickedId=29_106343_58"
cats_features, _ = decode_url_features(cats_url)
cats_prompt = chattify(["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
     "<think>\n\n</think>\n\nAt first there were 4 cats. Then, 3 went away. Now, there"])
cat_logits, cat_acts = model.get_activations(cats_prompt)
upweight_one = [(l, -1, f, 7 * cat_acts[l,p,f]) for l,p,f in cats_features['one']]

# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that was 1"])
upweight_one = [(l, -1, f, 5 * cat_acts[l,p,f]) for l,p,f in cats_features['one']]
logits, acts = model.feature_intervention(park_prompt,upweight_one, freeze_attention=False)
print_topk(logits)
# %%
flight_url = 'http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%225000249971%22%2C%22say+l*%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%2213318343991%22%2C%22say+bright%22%5D%2C%5B%2211581126301%22%2C%22fly%2Fflight%22%5D%2C%5B%228138009216%22%2C%22say+fl%22%5D%2C%5B%221329861340%22%2C%22fl%22%5D%2C%5B%22220657491%22%2C%22fl%22%5D%2C%5B%225586509215%22%2C%22fly%22%5D%2C%5B%221236859680%22%2C%22wing%22%5D%2C%5B%2210153908731%22%2C%22float%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%221196363042%22%2C%22imagination+go+wild+%2F+sky%27s+the+limit%22%5D%2C%5B%228439798047%22%2C%22*it%22%5D%2C%5B%22881055216%22%2C%22say+flight%22%5D%2C%5B%224310050563%22%2C%22imagination+transporing+places%22%5D%2C%5B%2210551323245%22%2C%22wings+%2F+glide%22%5D%2C%5B%229712346840%22%2C%22fl%22%5D%2C%5B%2212620712238%22%2C%22imagination+%2F+creativity+run+wild%22%5D%2C%5B%226785600723%22%2C%22fly%22%5D%2C%5B%226459468253%22%2C%22take+off%22%5D%2C%5B%229619330421%22%2C%22take+off%22%5D%2C%5B%229527727826%22%2C%22take+off%22%5D%2C%5B%224737099744%22%2C%22near+ends+of+lines%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%22866799029%22%2C%22night%22%5D%2C%5B%224229378371%22%2C%22shine%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%2224161659%22%2C%22end+of+line%22%5D%2C%5B%223356385325%22%2C%22near+the+end+of+a+line%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211326253752%22%2C%22ends+of+lines%22%5D%2C%5B%221393735179%22%2C%22ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%226839060551%22%2C%22ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%222829588358%22%2C%22ends+of+lines%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%226553978%22%2C%22near+ends+of+lines%22%5D%2C%5B%222926239720%22%2C%22near+ends+of+lines%22%5D%2C%5B%222260911364%22%2C%22near+ends+of+lines%22%5D%2C%5B%2212280500812%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211292264592%22%2C%22near+ends+of+lines%22%5D%2C%5B%221940987640%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210913184429%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210589563188%22%2C%22near+ends+of+lines%22%5D%2C%5B%2213374055896%22%2C%22near+ends+of+lines%22%5D%2C%5B%221263410740%22%2C%22fly%22%5D%5D&slug=delight-imagination-take-flight&pinnedIds=41_10971_39%2C34_152156_39%2C37_50229_39%2C33_142471_39%2C32_145234_39%2C34_152156_38%2C33_142471_38%2C36_127540_38%2C37_51534_38%2C33_129887_38%2C37_105664_38%2C27_116397_20%2C28_113706_20%2C28_57695_20%2C35_49700_38%2C36_20970_38%2C36_41940_39%2C37_139334_39%2C35_49700_39%2C27_48887_38%2C26_92817_38%2C11_158863_38%2C36_20970_39%2C36_127540_39%2C37_51534_39&pruningThreshold=0.6&supernodes=%5B%5B%22fl%22%2C%2237_51534_39%22%2C%2236_20970_39%22%2C%2236_127540_39%22%2C%2237_139334_39%22%5D%2C%5B%22fly%2Fflight%22%2C%2234_152156_39%22%2C%2237_50229_39%22%2C%2236_41940_39%22%5D%5D&clickedId=34_152156_39'
flight_prompt = chattify(["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
     "<think>\n\n</think>\n\nTreasure each page, let your imagination take"])
flight_features, _ = decode_url_features(flight_url)
flight_logits, flight_acts = model.get_activations(flight_prompt)
upweight_flight = [(l, -1, f, 2 * flight_acts[l,p,f]) for l,p,f in flight_features['fly/flight'] + flight_features['fl']]
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nI saw a dog, that was"])
upweight_flight = [(l, -1, f, 2 * flight_acts[l,p,f]) for l,p,f in flight_features['fly/flight']]# + flight_features['fl']]
logits, acts = model.feature_intervention(park_prompt,upweight_flight, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nIt was a cold day, the kind where your breath flies away"])
upweight_flight = [(l, -1, f, 2 * flight_acts[l,p,f]) for l,p,f in flight_features['fly/flight']]# + flight_features['fl']]
logits, acts = model.feature_intervention(park_prompt,upweight_flight, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nIt was a cold day, but the kind of cold that made the world feel still and bright"])
upweight_bright = [(l, -1, f, 7 * bright_acts[l,p,f]) for l,p,f in bright_features['say bright']]
logits, acts = model.feature_intervention(park_prompt,upweight_bright, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nIt was a cold day, but the"])
upweight_bright = []#[(l, -1, f, 5 * bright_acts[l,p,f]) for l,p,f in bright_features['say bright']]
logits, acts = model.feature_intervention(park_prompt,upweight_bright, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nIt was a cold day, but the kind of cold that makes the world feel still and quiet"])
upweight_night = [(l,-1, f, 4 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
logits, acts = model.feature_intervention(park_prompt,upweight_night, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nIt was a cold day, one of those"])
upweight_one = [(l, -1, f, 5 * cat_acts[l,p,f]) for l,p,f in cats_features['one']]
logits, acts = model.feature_intervention(park_prompt,upweight_one, freeze_attention=False)
print_topk(logits)
# %%
park_prompt = chattify(["/no_think A short story:\n Yesterday I went to the park,", 
     "<think>\n\n</think>\n\nIt was a cold day"])
upweight_night = [(l,slice(-1, None), f, 4 * night_acts[l,p,f]) for l,p,f in night_features['during / through the night']]
generation, logits, acts = model.feature_intervention_generate(park_prompt,upweight_night, freeze_attention=False, do_sample=False)
print(generation)
# %%
g = model.generate('La estructura por donde sale el humo de la casa es la chimenea. El recipiente pequeño donde bebes café o té es', do_sample=False)
print(g)
# %%
