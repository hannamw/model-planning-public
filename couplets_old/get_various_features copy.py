#%%
from collections import Counter
from load_feature_from_binary import get_feature_top_acts, get_features_top_acts_from_list
from circuit_tracer.graph import Graph
from transformers import AutoTokenizer
from tqdm import tqdm
import re
#%%
url = 'http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%225000249971%22%2C%22say+l*%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%225007352669%22%2C%22in+%28light%29%22%5D%2C%5B%225011957109%22%2C%22during+%2F+through+the+night%22%5D%2C%5B%22350317180%22%2C%22through+%2F+in+the+night%22%5D%2C%5B%225557057141%22%2C%22during+%2F+first+thing+in+the+morning+%2F+evening%22%5D%2C%5B%2210467568%22%2C%22during+the+day+%2F+night+%28say+%5C%22night%5C%22%29%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%229281826840%22%2C%22say+night%22%5D%2C%5B%224549387540%22%2C%22overnight%22%5D%2C%5B%227439865121%22%2C%22dark%22%5D%2C%5B%224218165292%22%2C%22after+%2F+evening%22%5D%2C%5B%222323335828%22%2C%22sunset+%2F+day%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%228439798047%22%2C%22*it%22%5D%2C%5B%224737099744%22%2C%22near+ends+of+lines%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%22152591688%22%2C%22near+ends+of+lines%22%5D%2C%5B%228589606881%22%2C%22say+night%22%5D%2C%5B%2264974264%22%2C%22say+night%22%5D%2C%5B%2211788263797%22%2C%22say+night%22%5D%2C%5B%22866799029%22%2C%22night%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%2224161659%22%2C%22end+of+line%22%5D%2C%5B%223356385325%22%2C%22near+the+end+of+a+line%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211326253752%22%2C%22ends+of+lines%22%5D%2C%5B%221393735179%22%2C%22ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%226839060551%22%2C%22ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%22305724606%22%2C%22ends+of+lines%22%5D%2C%5B%222829588358%22%2C%22ends+of+lines%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2211772300385%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2212434408432%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%226553978%22%2C%22near+ends+of+lines%22%5D%2C%5B%22151336472%22%2C%22near+ends+of+lines%22%5D%2C%5B%222926239720%22%2C%22near+ends+of+lines%22%5D%2C%5B%223518556300%22%2C%22through%22%5D%2C%5B%222260911364%22%2C%22near+ends+of+lines%22%5D%2C%5B%2212280500812%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211292264592%22%2C%22near+ends+of+lines%22%5D%2C%5B%221940987640%22%2C%22near+ends+of+lines%22%5D%2C%5B%223498117522%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210913184429%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210589563188%22%2C%22near+ends+of+lines%22%5D%2C%5B%2213374055896%22%2C%22near+ends+of+lines%22%5D%5D&slug=delight-shine-through-the-night&pinnedIds=41_3729_38%2C35_136212_38%2C33_131035_38%2C30_100088_38%2C26_98927_20%2C27_83859_38%2C28_113706_20%2C28_57695_20%2C29_37510_20%2C27_83859_37%2C18_157679_36%2C30_100088_35%2C31_4543_36%2C30_100088_34%2C33_129887_38%2C31_4543_38%2C32_146050_38%2C34_26434_38%2C37_95349_38%2C35_11363_38%2C36_41599_38%2C33_153512_38%2C30_109896_38%2C27_116397_20%2C36_41599_37%2C31_4543_37%2C34_105388_37%2C26_155381_20%2C20_81910_38%2C23_12385_38%2C28_67123_38%2C23_83619_38%2C23_147713_38%2C24_62280_38%2C25_67218_38%2C28_150252_38%2C26_56839_38%2C26_145503_38%2C27_156691_38%2C29_163518_38%2C30_76470_38%2C31_3588_38%2C30_17366_38%2C20_60314_20%2C18_16919_20%2C21_24705_20%2C20_142282_20%2C16_6934_20%2C25_150481_20%2C15_43513_20%2C28_145505_20%2C29_116923_20%2C26_52769_20%2C25_133750_20&pruningThreshold=0.64&supernodes=%5B%5B%22right%22%2C%2228_113706_20%22%2C%2227_116397_20%22%2C%2228_57695_20%22%5D%2C%5B%22through+%2F+in+the+night%22%2C%2234_26434_38%22%2C%2231_4543_38%22%2C%2230_100088_38%22%5D%2C%5B%22say+night%22%2C%2233_153512_38%22%2C%2235_11363_38%22%2C%2233_131035_38%22%2C%2235_136212_38%22%5D%2C%5B%22near+ends+of+lines%22%2C%2220_81910_38%22%2C%2223_147713_38%22%2C%2223_83619_38%22%2C%2231_3588_38%22%2C%2229_163518_38%22%2C%2230_17366_38%22%2C%2230_76470_38%22%2C%2228_150252_38%22%2C%2227_156691_38%22%2C%2225_67218_38%22%2C%2226_145503_38%22%2C%2224_62280_38%22%2C%2228_67123_38%22%2C%2226_56839_38%22%2C%2223_12385_38%22%5D%2C%5B%22end+of+line%22%2C%2215_43513_20%22%2C%2216_6934_20%22%2C%2218_16919_20%22%2C%2220_142282_20%22%2C%2221_24705_20%22%2C%2220_60314_20%22%2C%2225_150481_20%22%2C%2228_145505_20%22%2C%2229_116923_20%22%5D%5D&clickedId=21_24705_20'
#%%
graph = Graph.from_pt('attribution_graphs/Qwen3-8B/6-climb.pt')
model_name = graph.cfg.tokenizer_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
#find EOL features
input_tokens = tokenizer.convert_ids_to_tokens(graph.input_tokens)
last_word = input_tokens.index(tokenizer.eos_token)
selected_features = graph.active_features[graph.selected_features]
last_word_features = graph.active_features[graph.active_features[:, 1] == last_word - 2]
model_name_noslash = model_name.split('/')[-1]

# now for the dreaded rhyme features
from transliterate_logits import transliterate_word
#%%
def rhyme_group_top_logits(term:str, top: list[str], bottom: list[str]):
    for logits in [top, bottom]:
        # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
        logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
        logits = [transliterate_word(logit) for logit in logits]
        if any(len(logit) == 0 for logit in logits):
            continue
        if all(logit[0] == logits[0][0] for logit in logits[1:]) or all(logit[-1] == logits[0][-1] for logit in logits[1:]):
            return True
    return False

def is_say_rhyme_feature(feature_info, word):
    return rhyme_group_top_logits(word, feature_info['top_logits'], feature_info['bottom_logits'])

# fires directly on it
def is_rhyme_feature(feature_info, word, exclude=None):
    first_chars, last_chars = Counter(), Counter()
    token_counts = Counter()
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        token = tokens[top_index]
        token = re.sub(r'^[^\w]+|[^\w]+$', '', token.strip()).lower()
        if len(token) >  4:
            return False
        if not token:
            continue
        first_chars[token[0]] += 1
        last_chars[token[-1]] += 1
        token_counts[token] += 1
    if len(first_chars) == 0:
        return False
    most_common_first_char, first_count = first_chars.most_common(1)[0]
    most_common_last_char, last_count = last_chars.most_common(1)[0]
    most_common_token, token_count = token_counts.most_common(1)[0]
    return (token_count <= 5) and (exclude is None or exclude not in most_common_token) and ((most_common_first_char in 'aeiou' and first_count >=7) or last_count >=7)


def is_EOL_feature(feature_info):
    EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
            EOL_count += 1
    return EOL_count >= 3 or any('⏎' in logit for logit in feature_info['top_logits'])

def is_near_EOL_feature(feature_info):
    near_EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        near_EOL_tokens = tokens[top_index + 2: top_index + 5]
        if any('⏎' in tok for tok in near_EOL_tokens):
            near_EOL_count += 1
    return near_EOL_count >= 3

def is_near_word_feature(feature_info, word, exclude=''):
    word_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        s = max(top_index-2, 0)
        e = min(top_index + 2, len(tokens) - 1)
        if word in ''.join(tokens[s:e]).lower():
            word_count += 1
    return word_count >= 3

def is_not_other_feature(layer, feat, feature_info, word, exclude=None):
    return (layer > 16) and not (is_EOL_feature(feature_info) or is_near_EOL_feature(feature_info) 
                                or is_near_word_feature(feature_info, word) or feature_info['frequency'] > 0.05)
    
# %%
input_tokens = tokenizer.convert_ids_to_tokens(graph.input_tokens)
last_word = input_tokens.index(tokenizer.eos_token)
selected_features = graph.active_features[graph.selected_features]
last_word_features =selected_features[selected_features[:, 1] == last_word - 2]
model_name_noslash = model_name.split('/')[-1]
feats = [(layer, feature) for layer, _, feature in last_word_features.tolist()]
batch_results = get_features_top_acts_from_list(model_name_noslash, feats)
#%%
#say_rhyme_features = {k:v for k,v in batch_results.items() if is_say_rhyme_feature(v, 'night')}
rhyme_features = {k:v for k,v in batch_results.items() if is_rhyme_feature(v, 'night', 'delight')}
#%%
pinned_ids = '%2C'.join([f'{layer}_{feat}_{last_word - 2}' for layer, feat in rhyme_features.keys()])
new_url = f'http://localhost:8046/index.html?slug=Qwen3-8B-6-climb&pinnedIds={pinned_ids}'
# %%
print(new_url)
#%%
#say_rhyme_features = {k:v for k,v in batch_results.items() if is_say_rhyme_feature(v, 'night')}
not_other_features = {k:v for k,v in batch_results.items() if is_not_other_feature(*k, v, 'time')}
pinned_ids = '%2C'.join([f'{layer}_{feat}_{last_word - 2}' for layer, feat in not_other_features.keys()])
new_url = f'http://localhost:8046/index.html?slug=Qwen3-8B-6-climb&pinnedIds={pinned_ids}'

print(new_url)
# %%
batch_results[(26, 155381)]
# %%
is_rhyme_feature(batch_results[(26, 155381)], 'night', 'delight')
