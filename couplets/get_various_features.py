#%%
from load_feature_from_binary import get_feature_top_acts, get_features_top_acts_from_list
from circuit_tracer.graph import Graph
from transformers import AutoTokenizer
from tqdm import tqdm
#%%
url = 'http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%225000249971%22%2C%22say+l*%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%225007352669%22%2C%22in+%28light%29%22%5D%2C%5B%225011957109%22%2C%22during+%2F+through+the+night%22%5D%2C%5B%22350317180%22%2C%22through+%2F+in+the+night%22%5D%2C%5B%225557057141%22%2C%22during+%2F+first+thing+in+the+morning+%2F+evening%22%5D%2C%5B%2210467568%22%2C%22during+the+day+%2F+night+%28say+%5C%22night%5C%22%29%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%229281826840%22%2C%22say+night%22%5D%2C%5B%224549387540%22%2C%22overnight%22%5D%2C%5B%227439865121%22%2C%22dark%22%5D%2C%5B%224218165292%22%2C%22after+%2F+evening%22%5D%2C%5B%222323335828%22%2C%22sunset+%2F+day%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%228439798047%22%2C%22*it%22%5D%2C%5B%224737099744%22%2C%22near+ends+of+lines%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%22152591688%22%2C%22near+ends+of+lines%22%5D%2C%5B%228589606881%22%2C%22say+night%22%5D%2C%5B%2264974264%22%2C%22say+night%22%5D%2C%5B%2211788263797%22%2C%22say+night%22%5D%2C%5B%22866799029%22%2C%22night%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%2224161659%22%2C%22end+of+line%22%5D%2C%5B%223356385325%22%2C%22near+the+end+of+a+line%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211326253752%22%2C%22ends+of+lines%22%5D%2C%5B%221393735179%22%2C%22ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%226839060551%22%2C%22ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%22305724606%22%2C%22ends+of+lines%22%5D%2C%5B%222829588358%22%2C%22ends+of+lines%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2211772300385%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2212434408432%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%226553978%22%2C%22near+ends+of+lines%22%5D%2C%5B%22151336472%22%2C%22near+ends+of+lines%22%5D%2C%5B%222926239720%22%2C%22near+ends+of+lines%22%5D%2C%5B%223518556300%22%2C%22through%22%5D%2C%5B%222260911364%22%2C%22near+ends+of+lines%22%5D%2C%5B%2212280500812%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211292264592%22%2C%22near+ends+of+lines%22%5D%2C%5B%221940987640%22%2C%22near+ends+of+lines%22%5D%2C%5B%223498117522%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210913184429%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210589563188%22%2C%22near+ends+of+lines%22%5D%2C%5B%2213374055896%22%2C%22near+ends+of+lines%22%5D%5D&slug=delight-shine-through-the-night&pinnedIds=41_3729_38%2C35_136212_38%2C33_131035_38%2C30_100088_38%2C26_98927_20%2C27_83859_38%2C28_113706_20%2C28_57695_20%2C29_37510_20%2C27_83859_37%2C18_157679_36%2C30_100088_35%2C31_4543_36%2C30_100088_34%2C33_129887_38%2C31_4543_38%2C32_146050_38%2C34_26434_38%2C37_95349_38%2C35_11363_38%2C36_41599_38%2C33_153512_38%2C30_109896_38%2C27_116397_20%2C36_41599_37%2C31_4543_37%2C34_105388_37%2C26_155381_20%2C20_81910_38%2C23_12385_38%2C28_67123_38%2C23_83619_38%2C23_147713_38%2C24_62280_38%2C25_67218_38%2C28_150252_38%2C26_56839_38%2C26_145503_38%2C27_156691_38%2C29_163518_38%2C30_76470_38%2C31_3588_38%2C30_17366_38%2C20_60314_20%2C18_16919_20%2C21_24705_20%2C20_142282_20%2C16_6934_20%2C25_150481_20%2C15_43513_20%2C28_145505_20%2C29_116923_20%2C26_52769_20%2C25_133750_20&pruningThreshold=0.64&supernodes=%5B%5B%22right%22%2C%2228_113706_20%22%2C%2227_116397_20%22%2C%2228_57695_20%22%5D%2C%5B%22through+%2F+in+the+night%22%2C%2234_26434_38%22%2C%2231_4543_38%22%2C%2230_100088_38%22%5D%2C%5B%22say+night%22%2C%2233_153512_38%22%2C%2235_11363_38%22%2C%2233_131035_38%22%2C%2235_136212_38%22%5D%2C%5B%22near+ends+of+lines%22%2C%2220_81910_38%22%2C%2223_147713_38%22%2C%2223_83619_38%22%2C%2231_3588_38%22%2C%2229_163518_38%22%2C%2230_17366_38%22%2C%2230_76470_38%22%2C%2228_150252_38%22%2C%2227_156691_38%22%2C%2225_67218_38%22%2C%2226_145503_38%22%2C%2224_62280_38%22%2C%2228_67123_38%22%2C%2226_56839_38%22%2C%2223_12385_38%22%5D%2C%5B%22end+of+line%22%2C%2215_43513_20%22%2C%2216_6934_20%22%2C%2218_16919_20%22%2C%2220_142282_20%22%2C%2221_24705_20%22%2C%2220_60314_20%22%2C%2225_150481_20%22%2C%2228_145505_20%22%2C%2229_116923_20%22%5D%5D&clickedId=21_24705_20'
#%%
graph = Graph.from_pt('../graphs/delight-shine-through-the-night.pt')
model_name = graph.cfg.tokenizer_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
#find EOL features
input_tokens = tokenizer.convert_ids_to_tokens(graph.input_tokens)
last_word = input_tokens.index(tokenizer.eos_token)
selected_features = graph.active_features[graph.selected_features]
last_word_features = graph.active_features[graph.active_features[:, 1] == last_word - 2]
model_name_noslash = model_name.split('/')[-1]

# Option 1: Original approach (slower)
# for layer, _, feature in tqdm(last_word_features.tolist()):
#     feature_info = get_feature_top_acts(model_name_noslash, layer, feature)
#%%
# Load all features for each layer in batches
feats = [(layer, feature) for layer, _, feature in last_word_features.tolist()]
batch_results = get_features_top_acts_from_list(model_name_noslash, feats)
# %%
def is_EOL_feature(feature_info):
    EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
            EOL_count += 1
    return EOL_count >= 7
# %%
eol_features = {k:v for k,v in batch_results.items() if is_EOL_feature(v)}
# %%
is_EOL_feature(batch_results[(21, 24705)])
# %%
len(eol_features)
# %%
feature_info = batch_results[(21, 24705)]
EOL_count = 0
for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
    if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
        EOL_count += 1
# %%
end_second_line_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
# %%
end_second_line_feats = [(layer, feature) for layer, _, feature in end_second_line_features.tolist()]
end_second_line_info = get_features_top_acts_from_list(model_name_noslash, end_second_line_feats)
# %%
info = end_second_line_info[(31, 3588)]

#%%
for tokens, idx in zip(info['tokens'], info['top_indices']):
    print(tokens[idx:])

# %%
def is_near_EOL_feature(feature_info):
    near_EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        near_EOL_tokens = tokens[top_index + 2: top_index + 5]
        if any('⏎' in tok for tok in near_EOL_tokens):
            near_EOL_count += 1
    return near_EOL_count >= 7
# %%
neol_feats = {k:v for k,v in end_second_line_info.items() if is_near_EOL_feature(v)}
# %%
neol_feats.keys()
# %%
pinned_ids = '%2C'.join([f'{layer}_{feat}_{graph.n_pos - 1}' for layer, feat in neol_feats.keys()])
new_url = f'http://localhost:8046/index.html?slug=delight-shine-through-the-night&pinnedIds={pinned_ids}'
# %%
print(new_url)
# http://localhost:8046/index.html?slug=delight-shine-through-the-night&pinnedIds="8_38_111444"%2C"10_38_15487"%2C"14_38_50139"%2C"15_38_33397"%2C"16_38_47110"%2C"17_38_38220"%2C"18_38_1855"%2C"18_38_3264"%2C"19_38_12200"%2C"19_38_57079"%2C"19_38_95229"%2C"19_38_147496"%2C"19_38_153214"%2C"20_38_2185"%2C"20_38_9477"%2C"20_38_60369"%2C"20_38_104147"%2C"20_38_106651"%2C"20_38_123357"%2C"21_38_40842"%2C"21_38_45274"%2C"21_38_62334"%2C"21_38_69832"%2C"21_38_96860"%2C"21_38_113072"%2C"22_38_101516"%2C"22_38_128655"%2C"23_38_12385"%2C"23_38_77205"%2C"23_38_78392"%2C"23_38_83619"%2C"23_38_117340"%2C"23_38_147713"%2C"24_38_2412"%2C"24_38_29927"%2C"24_38_30753"%2C"24_38_62280"%2C"24_38_122571"%2C"25_38_67218"%2C"25_38_145528"%2C"26_38_56839"%2C"26_38_60465"%2C"26_38_143121"%2C"26_38_145503"%2C"27_38_64138"%2C"27_38_156691"%2C"28_38_46073"%2C"28_38_150252"%2C"29_38_138040"%2C"29_38_163518"%2C"30_38_17366"%2C"30_38_41713"%2C"30_38_44497"%2C"30_38_76470"%2C"30_38_144811"%2C"31_38_3588"%2C"31_38_45760"%2C"31_38_152578"%2C"31_38_163270"%2C"32_38_133856"%2C"32_38_146050"%2C"32_38_152499"%2C"33_38_58511"%2C"34_38_97880"%2C"34_38_109594"%2C"34_38_141222"%2C"35_38_36520"%2C"35_38_97299"%2C"36_38_7981"%2C"37_38_6009"%2C"37_38_11056"%2C"38_38_8193"%2C"38_38_34152"%2C"38_38_47782"%2C"38_38_149039"%2C"38_38_155322"%2C"39_38_28482"%2C"39_38_77122"%2C"39_38_126430"%2C"39_38_137811"%2C"39_38_146415"
# %%
neol_feats[(27, 156691)]
# %%
good_url = 'http://localhost:8046/index.html?clerps=%5B%5B%2210980360307%22%2C%22light%22%5D%2C%5B%225000249971%22%2C%22say+l*%22%5D%2C%5B%226467881951%22%2C%22right%22%5D%2C%5B%226777448497%22%2C%22*t%22%5D%2C%5B%221666058921%22%2C%22*t%22%5D%2C%5B%224524905988%22%2C%22*iC%22%5D%2C%5B%226042027597%22%2C%22by+x-light%22%5D%2C%5B%225007352669%22%2C%22in+%28light%29%22%5D%2C%5B%225011957109%22%2C%22during+%2F+through+the+night%22%5D%2C%5B%22350317180%22%2C%22through+%2F+in+the+night%22%5D%2C%5B%225557057141%22%2C%22during+%2F+first+thing+in+the+morning+%2F+evening%22%5D%2C%5B%2210467568%22%2C%22during+the+day+%2F+night+%28say+%5C%22night%5C%22%29%22%5D%2C%5B%22704644540%22%2C%22night%22%5D%2C%5B%224174832345%22%2C%22right%22%5D%2C%5B%222254729099%22%2C%22near+ends+of+lines%22%5D%2C%5B%224478128196%22%2C%22couplet%22%5D%2C%5B%22109248%22%2C%22bright%22%5D%2C%5B%229281826840%22%2C%22say+night%22%5D%2C%5B%224549387540%22%2C%22overnight%22%5D%2C%5B%227439865121%22%2C%22dark%22%5D%2C%5B%224218165292%22%2C%22after+%2F+evening%22%5D%2C%5B%222323335828%22%2C%22sunset+%2F+day%22%5D%2C%5B%2210670194453%22%2C%22poetry%22%5D%2C%5B%228439798047%22%2C%22*it%22%5D%2C%5B%224737099744%22%2C%22near+ends+of+lines%22%5D%2C%5B%223485248272%22%2C%22near+ends+of+lines%3F%22%5D%2C%5B%22152591688%22%2C%22near+ends+of+lines%22%5D%2C%5B%228589606881%22%2C%22say+night%22%5D%2C%5B%2264974264%22%2C%22say+night%22%5D%2C%5B%2211788263797%22%2C%22say+night%22%5D%2C%5B%22866799029%22%2C%22night%22%5D%2C%5B%22947408669%22%2C%22end+of+line%22%5D%2C%5B%2224161659%22%2C%22end+of+line%22%5D%2C%5B%223356385325%22%2C%22near+the+end+of+a+line%22%5D%2C%5B%22143456372%22%2C%22ends+of+lines%22%5D%2C%5B%221820186259%22%2C%22ends+of+lines%22%5D%2C%5B%2276997821%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211326253752%22%2C%22ends+of+lines%22%5D%2C%5B%221393735179%22%2C%22ends+of+lines%22%5D%2C%5B%221616899384%22%2C%22near+ends+of+lines%22%5D%2C%5B%226839060551%22%2C%22ends+of+lines%22%5D%2C%5B%2210590145316%22%2C%22ends+of+lines%22%5D%2C%5B%2210125143035%22%2C%22ends+of+lines%22%5D%2C%5B%22305724606%22%2C%22ends+of+lines%22%5D%2C%5B%222829588358%22%2C%22ends+of+lines%22%5D%2C%5B%229914643318%22%2C%22poem+comma+%2F+newline%22%5D%2C%5B%2210473314069%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2211772300385%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%2212434408432%22%2C%22verse+%2F+meter%3F%22%5D%2C%5B%226553978%22%2C%22near+ends+of+lines%22%5D%2C%5B%22151336472%22%2C%22near+ends+of+lines%22%5D%2C%5B%222926239720%22%2C%22near+ends+of+lines%22%5D%2C%5B%223518556300%22%2C%22through%22%5D%2C%5B%222260911364%22%2C%22near+ends+of+lines%22%5D%2C%5B%2212280500812%22%2C%22near+ends+of+lines%22%5D%2C%5B%2211292264592%22%2C%22near+ends+of+lines%22%5D%2C%5B%221940987640%22%2C%22near+ends+of+lines%22%5D%2C%5B%223498117522%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210913184429%22%2C%22near+ends+of+lines%22%5D%2C%5B%2210589563188%22%2C%22near+ends+of+lines%22%5D%2C%5B%2213374055896%22%2C%22near+ends+of+lines%22%5D%5D&slug=delight-shine-through-the-night&pinnedIds=41_3729_38%2C35_136212_38%2C33_131035_38%2C30_100088_38%2C26_98927_20%2C27_83859_38%2C28_113706_20%2C28_57695_20%2C29_37510_20%2C27_83859_37%2C18_157679_36%2C30_100088_35%2C31_4543_36%2C30_100088_34%2C33_129887_38%2C31_4543_38%2C32_146050_38%2C34_26434_38%2C37_95349_38%2C35_11363_38%2C36_41599_38%2C33_153512_38%2C30_109896_38%2C27_116397_20%2C36_41599_37%2C31_4543_37%2C34_105388_37%2C26_155381_20%2C20_81910_38%2C23_12385_38%2C28_67123_38%2C23_83619_38%2C23_147713_38%2C24_62280_38%2C25_67218_38%2C28_150252_38%2C26_56839_38%2C26_145503_38%2C27_156691_38%2C29_163518_38%2C30_76470_38%2C31_3588_38%2C30_17366_38%2C20_60314_20%2C18_16919_20%2C21_24705_20%2C20_142282_20%2C16_6934_20%2C25_150481_20%2C15_43513_20%2C28_145505_20%2C29_116923_20%2C26_52769_20%2C25_133750_20&pruningThreshold=0.64&supernodes=%5B%5B%22right%22%2C%2228_113706_20%22%2C%2227_116397_20%22%2C%2228_57695_20%22%5D%2C%5B%22through+%2F+in+the+night%22%2C%2234_26434_38%22%2C%2231_4543_38%22%2C%2230_100088_38%22%5D%2C%5B%22say+night%22%2C%2233_153512_38%22%2C%2235_11363_38%22%2C%2233_131035_38%22%2C%2235_136212_38%22%5D%2C%5B%22near+ends+of+lines%22%2C%2220_81910_38%22%2C%2223_147713_38%22%2C%2223_83619_38%22%2C%2231_3588_38%22%2C%2229_163518_38%22%2C%2230_17366_38%22%2C%2230_76470_38%22%2C%2228_150252_38%22%2C%2227_156691_38%22%2C%2225_67218_38%22%2C%2226_145503_38%22%2C%2224_62280_38%22%2C%2228_67123_38%22%2C%2226_56839_38%22%2C%2223_12385_38%22%5D%2C%5B%22end+of+line%22%2C%2215_43513_20%22%2C%2216_6934_20%22%2C%2218_16919_20%22%2C%2220_142282_20%22%2C%2221_24705_20%22%2C%2220_60314_20%22%2C%2225_150481_20%22%2C%2228_145505_20%22%2C%2229_116923_20%22%5D%5D&clickedId=27_156691_38'
#%%
# in top/bot
def is_say_x_feature():
    return

# fires directly on it

# it's within +- 2 tokens

#%%
# the dreaded rhyme features, how to capture them