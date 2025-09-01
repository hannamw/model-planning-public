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
    'delight':["/no_think A rhyming couplet:\n Boxes of books, a reader's delight,", 
"Stacked high with stories to"],
    'fun':["/no_think A rhyming couplet:\n Boxes of books, a reader's fun,", 
"Stacked high, they're never"],
    'chore':["/no_think A rhyming couplet:\n Boxes of books, a reader's chore,", 
"A treasure trove to"],
}

urls = {
    'delight': 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-lowl0-poem-boxes-of-books&clerps=%5B%5B%22117967%22%2C%22delight%22%5D%2C%5B%22133048%22%2C%22delight%22%5D%2C%5B%2246867%22%2C%22delight%22%5D%2C%5B%228737%22%2C%22delight%22%5D%2C%5B%2279011%22%2C%22*t%22%5D%2C%5B%2285259%22%2C%22delight%22%5D%2C%5B%22152462%22%2C%22*t%22%5D%2C%5B%22148103%22%2C%22rhetoric%22%5D%2C%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2268490%22%2C%22rhyming+poetry%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140453%22%2C%22%28C%29Vl%22%5D%2C%5B%2231260%22%2C%22*t%22%5D%2C%5B%2259848%22%2C%22say+-VVC-%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%2257079%22%2C%22positions+before+rhymes%22%5D%2C%5B%22102121%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2260921%22%2C%22say+-i%28V%29%28C%29-%22%5D%2C%5B%222185%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%2269832%22%2C%22positions+before+rhymes%22%5D%2C%5B%2265862%22%2C%22say+-VV%28C%29-%22%5D%2C%5B%2214881%22%2C%22%28V%29t*%22%5D%2C%5B%2236412%22%2C%22say+-iC*-%22%5D%2C%5B%22155381%22%2C%22%28C%29ot%22%5D%2C%5B%2298927%22%2C%22ain%28t%29%22%5D%2C%5B%22116397%22%2C%22%28C%29V%28C%29t%22%5D%2C%5B%2220349%22%2C%22delight%22%5D%2C%5B%22113706%22%2C%22right%22%5D%2C%5B%22106122%22%2C%22Vl%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2254879%22%2C%22%28C%29it%22%5D%2C%5B%22130088%22%2C%22%28V%29rt%22%5D%2C%5B%2237510%22%2C%22night%22%5D%2C%5B%22437%22%2C%22bright%22%5D%2C%5B%22100088%22%2C%22night%22%5D%2C%5B%2285020%22%2C%22fright%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2272654%22%2C%22delight%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2245760%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%223588%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%22152499%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%22129887%22%2C%22say+*dt+%2F+positions+before+ight%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%2241393%22%2C%22say+*I%22%5D%2C%5B%2297880%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%22136212%22%2C%22night%22%5D%2C%5B%22129639%22%2C%22say+%28V%29t%2FT%28V%29%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%2236520%22%2C%22say+ite%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2241599%22%2C%22night%22%5D%2C%5B%2288131%22%2C%22say+Cit%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%2228637%22%2C%22say+*ite%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%224046%22%2C%22positions+before+*t%22%5D%5D&pinnedIds=41_3270_36%2C27_116397_20%2C26_155381_20%2C29_437_20%2C29_37510_20%2C28_113706_20%2C28_57695_20%2C28_54879_20%2C28_130088_20%2C27_20349_20%2C30_85020_20%2C31_72654_20%2C39_4046_36%2C38_28637_36%2C33_129887_36%2C36_88131_36%2C35_129639_36%2C34_41393_36%2C18_31260_20%2C23_14881_20%2C9_152462_20%2C9_85259_20%2C7_79011_20%2C3_133048_20%2C3_46867_20%2C3_8737_20%2CE_17970_20%2C0_117967_20%2C26_98927_20%2C19_60921_20%2C18_59848_20%2C22_65862_20%2C23_36412_20%2C35_36520_36%2C32_146050_36%2C34_97880_36%2C33_58511_36%2C32_152499_36%2C31_45760_36%2C31_3588_36%2C19_95229_36%2C21_69832_36%2C20_2185_36%2C19_57079_36%2C17_140798_21%2C19_102121_21%2C16_2405_21%2C14_154449_21%2C14_68490_21&pruningThreshold=0.7&supernodes=%5B%5B%22delight%22%2C%223_133048_20%22%2C%223_46867_20%22%2C%229_85259_20%22%2C%223_8737_20%22%2C%220_117967_20%22%5D%2C%5B%22delight%22%2C%2227_20349_20%22%2C%2231_72654_20%22%5D%2C%5B%22*ight+rhyming+words%22%2C%2229_37510_20%22%2C%2228_113706_20%22%2C%2229_437_20%22%2C%2230_85020_20%22%5D%2C%5B%22pronunciation+%28final+position%29%22%2C%2233_129887_36%22%2C%2239_4046_36%22%2C%2234_41393_36%22%2C%2236_88131_36%22%2C%2235_129639_36%22%2C%2238_28637_36%22%2C%2235_36520_36%22%5D%2C%5B%22positions+before+rhymes%22%2C%2219_57079_36%22%2C%2219_95229_36%22%2C%2220_2185_36%22%2C%2221_69832_36%22%5D%2C%5B%22ends+of+first+lines+of+poems+%2F+rhymes%22%2C%2217_140798_21%22%2C%2216_2405_21%22%2C%2219_102121_21%22%2C%2214_154449_21%22%5D%2C%5B%22old-fashioned+text+%2F+poetry%22%2C%2231_45760_36%22%2C%2232_152499_36%22%2C%2231_3588_36%22%2C%2232_146050_36%22%2C%2233_58511_36%22%2C%2234_97880_36%22%5D%2C%5B%22pronunciation+%28delight+position%29%22%2C%2226_98927_20%22%2C%2223_36412_20%22%2C%2222_65862_20%22%2C%2218_59848_20%22%2C%2219_60921_20%22%2C%2228_57695_20%22%2C%2228_130088_20%22%2C%2228_54879_20%22%2C%2227_116397_20%22%2C%2226_155381_20%22%2C%2223_14881_20%22%2C%229_152462_20%22%2C%2218_31260_20%22%2C%227_79011_20%22%5D%5D&clickedId=19_60921_20',
    'fun': 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-poem-books-fun&clerps=%5B%5B%2292656%22%2C%22say+*Vn-%22%5D%2C%5B%22148103%22%2C%22rhetoric%22%5D%2C%5B%2245888%22%2C%22stopping%22%5D%2C%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22142705%22%2C%22no%2Fnever+stopping%22%5D%2C%5B%2222243%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22145527%22%2C%22say+-%28C%29Vn-%22%5D%2C%5B%2229089%22%2C%22say+-*an-%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%2257079%22%2C%22positions+before+rhymes%22%5D%2C%5B%22102121%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22103817%22%2C%22say+-*an*-%22%5D%2C%5B%222185%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%22149793%22%2C%22enough+%2F+too%22%5D%2C%5B%2274039%22%2C%22say+-*%28V%29nC-%22%5D%2C%5B%2269832%22%2C%22positions+before+rhymes%22%5D%2C%5B%2298007%22%2C%22limit%22%5D%2C%5B%22125140%22%2C%22say+-*an-%22%5D%2C%5B%22138760%22%2C%22say+-an%28C%29-%22%5D%2C%5B%2218707%22%2C%22fun%22%5D%2C%5B%2268436%22%2C%22-unC-%22%5D%2C%5B%2234562%22%2C%22-%28V%29Cn-%22%5D%2C%5B%22104889%22%2C%22never%22%5D%2C%5B%2279437%22%2C%22fun%22%5D%2C%5B%22105934%22%2C%22fun%22%5D%2C%5B%22108626%22%2C%22-un-%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2226828%22%2C%22is%22%5D%2C%5B%2257695%22%2C%22t*%22%5D%2C%5B%2231413%22%2C%22fun%22%5D%2C%5B%2265944%22%2C%22fun%22%5D%2C%5B%22136953%22%2C%22fun%22%5D%2C%5B%222686%22%2C%22fun-%22%5D%2C%5B%2279561%22%2C%22fun%22%5D%2C%5B%2284066%22%2C%22-%28V%2FC%29n-%22%5D%2C%5B%2270052%22%2C%22-%28V%29%28C%29n-%22%5D%2C%5B%22139555%22%2C%22never%22%5D%2C%5B%2289528%22%2C%22function%22%5D%2C%5B%2265455%22%2C%22fun%22%5D%2C%5B%22137084%22%2C%22say+%5C%22ah%5C%22+vowel%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2238973%22%2C%22-un-%22%5D%2C%5B%22122249%22%2C%22fun%22%5D%2C%5B%22103465%22%2C%22completed+%2F+finished%22%5D%2C%5B%2244466%22%2C%22parentheticals%3F%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2219483%22%2C%22fun%22%5D%2C%5B%2287410%22%2C%22nonstop%22%5D%2C%5B%22126114%22%2C%22f-%22%5D%2C%5B%22163128%22%2C%22finished+%2F+done%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%2282230%22%2C%22say+%5C%22end%5C%22+%28after+%5C%22put+an%5C%22%29%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%2297880%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%22155383%22%2C%22pronunciations%22%5D%2C%5B%22138631%22%2C%22say+-*n%28e%29-%22%5D%2C%5B%22128928%22%2C%22put+%28before+%5C%22down%5C%22%29%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%2283012%22%2C%22say+d%28one%29%22%5D%2C%5B%2261457%22%2C%22not%22%5D%2C%5B%2238962%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%22113990%22%2C%22don%27t+say+done%22%5D%2C%5B%2267295%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%22145227%22%2C%22say+-*%28C%29on-%22%5D%2C%5B%2281324%22%2C%22say+-%28C%29en-%22%5D%2C%5B%22129755%22%2C%22don%27t+say+-CVn-%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22104119%22%2C%22say+-Vn%28g%29-%22%5D%5D&pinnedIds=41_2814_37%2C32_126114_20%2C31_19483_20%2C29_79561_20%2C30_122249_20%2C29_2686_20%2C30_38973_20%2C29_84066_20%2C27_126048_20%2C27_105934_20%2C26_34562_20%2C27_108626_20%2C29_136953_20%2C35_128928_37%2C37_145227_37%2C37_67295_37%2C37_129755_37%2C29_70052_20%2C28_31413_20%2C35_88319_37%2C39_104119_37%2C36_83012_37%2C37_113990_37%2C35_155383_37%2C33_82230_37%2C30_103465_37%2C32_163128_37%2C31_87410_37%2C26_104889_37%2C29_139555_37%2C22_98007_37%2C14_45888_37%2C21_149793_37%2C15_142705_37%2C30_137084_37%2C30_44466_37%2C36_38962_37%2C35_138631_37%2C37_81324_37%2C18_145527_20%2C24_138760_20%2C21_74039_20%2C23_125140_20%2C7_92656_20%2C20_103817_20%2C25_68436_20%2C19_29089_20%2C29_89528_20%2C28_65944_20%2C25_18707_20%2C26_79437_20%2C17_140798_21%2C19_102121_21%2C14_154449_21%2C15_22243_21%2C16_2405_21%2C19_102121_23&supernodes=%5B%5B%22finished+%2F+done%22%2C%2232_163128_37%22%2C%2230_103465_37%22%5D%2C%5B%22never%22%2C%2229_139555_37%22%2C%2226_104889_37%22%5D%2C%5B%22fun%22%2C%2226_79437_20%22%2C%2225_18707_20%22%2C%2228_65944_20%22%2C%2228_31413_20%22%2C%2231_19483_20%22%2C%2230_122249_20%22%2C%2229_79561_20%22%2C%2227_105934_20%22%2C%2229_2686_20%22%2C%2229_136953_20%22%5D%2C%5B%22pronunciation+%28final+position%29%22%2C%2230_137084_37%22%2C%2235_138631_37%22%2C%2239_104119_37%22%2C%2237_67295_37%22%2C%2237_145227_37%22%2C%2236_38962_37%22%2C%2237_81324_37%22%2C%2237_129755_37%22%5D%2C%5B%22pronunciation+%28fun+position%29%22%2C%227_92656_20%22%2C%2220_103817_20%22%2C%2221_74039_20%22%2C%2227_108626_20%22%2C%2230_38973_20%22%2C%2227_126048_20%22%2C%2226_34562_20%22%2C%2229_84066_20%22%2C%2229_70052_20%22%2C%2224_138760_20%22%2C%2218_145527_20%22%2C%2223_125140_20%22%2C%2219_29089_20%22%2C%2225_68436_20%22%5D%2C%5B%22ends+of+first+lines+of+poems+%2F+rhymes%22%2C%2215_22243_21%22%2C%2214_154449_21%22%2C%2216_2405_21%22%2C%2217_140798_21%22%2C%2219_102121_21%22%5D%5D&clickedId=19_102121_21',
    'chore': 'http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug=qwen3-14b-relu-lowl0-poem-books-explore&clerps=%5B%5B%22154449%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2268490%22%2C%22rhyming+poetry%22%5D%2C%5B%222405%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%22140798%22%2C%22ends+of+first+lines+of+poems+%2F+rhymes%22%5D%2C%5B%2295229%22%2C%22positions+before+rhymes%22%5D%2C%5B%22120876%22%2C%22%3F%3F%3F+often-relevant+often-on+feature%22%5D%2C%5B%2221201%22%2C%22%28C%29Vl%22%5D%2C%5B%22126048%22%2C%22-uC-%22%5D%2C%5B%2227454%22%2C%22one%22%5D%2C%5B%22137084%22%2C%22say+%5C%22ah%5C%22+vowel%22%5D%2C%5B%2219964%22%2C%22%28dense+feature%29%22%5D%2C%5B%2217578%22%2C%22%3F%3F%3F%22%5D%2C%5B%2287363%22%2C%22treasure+%2F+gem%22%5D%2C%5B%223588%22%2C%22old-fashioned+text-poetry%22%5D%2C%5B%22146050%22%2C%22poetry%22%5D%2C%5B%2258511%22%2C%22old-fashioned+text+%2F+poetry%22%5D%2C%5B%22125138%22%2C%22%3F%3F%3F%22%5D%2C%5B%2288319%22%2C%22say+done%22%5D%2C%5B%2236520%22%2C%22say+ite%22%5D%2C%5B%22103390%22%2C%22%3F%3F%3F%22%5D%2C%5B%22103617%22%2C%22%3F%3F%3F+%2F+say+%5C%22an%5C%22%22%5D%2C%5B%227190%22%2C%22%28dense+feature%29%22%5D%2C%5B%22101487%22%2C%22rhyming+%2F+abbreviations%22%5D%5D&clickedId=31_74720_20&pinnedIds=41_96451_35%2C41_13186_35%2C30_125121_20%2C29_90324_20%2C29_103782_20%2C28_136719_20%2C28_92161_20%2C27_43987_20%2C27_99427_20%2C27_124515_20%2C27_154033_20%2C26_53355_20%2C26_25311_20%2C25_1252_20%2C25_11031_20%2C25_21201_20%2C24_96733_20%2C24_28942_20%2C23_61654_20%2C23_148979_20%2C23_98821_20%2C22_147686_20%2C21_57394_20%2C21_68342_20%2C31_74720_20&supernodes=%5B%5B%22pronunciation+%28chore+position%29%22%2C%2228_92161_20%22%2C%2227_124515_20%22%2C%2225_1252_20%22%2C%2221_68342_20%22%2C%2223_98821_20%22%2C%2222_147686_20%22%2C%2223_148979_20%22%2C%2224_28942_20%22%2C%2225_21201_20%22%2C%2229_90324_20%22%2C%2227_43987_20%22%2C%2226_25311_20%22%2C%2231_74720_20%22%2C%2230_125121_20%22%2C%2229_103782_20%22%2C%2227_154033_20%22%2C%2228_136719_20%22%2C%2227_99427_20%22%2C%2226_53355_20%22%2C%2225_11031_20%22%2C%2224_96733_20%22%2C%2223_61654_20%22%2C%2221_57394_20%22%5D%5D',
}

supernodes = {
    'delight': ["*ight rhyming words", "pronunciation (delight position)"],
    'fun': ["pronunciation (fun position)"],
    'chore': ["pronunciation (chore position)"],
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
    logits, acts = model.get_activations(s, zero_bos=True)
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
