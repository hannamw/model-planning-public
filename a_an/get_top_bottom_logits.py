#%%
import argparse
import json
import os
import gc

import numpy as np
import torch

from tokenizers import Tokenizer
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from circuit_tracer.frontend.utils import process_token
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub


def compute_clean_vocab(tokenizer: Tokenizer, cache_file: str) -> np.array:
    """Precompute vocab to avoid calling decode

    Args:
        tokenizer (Tokenizer): A HuggingFace tokenizer

    Returns:
        np.array: an array mapping token_idxs to string tokens
    """
    if os.path.exists(cache_file):
        return np.load(cache_file)

    vocab = np.array([process_token(tokenizer.decode([i])) for i in range(len(tokenizer))])

    np.save(cache_file, vocab)
    return vocab

@torch.no_grad
def compute_top_bottom_tokens(
    ln_final,
    W_U,
    W_dec,
    vocab: np.array,
    cache_file: str,
    logit_batch_size: int = 2**14,
    batch_size: int = 256,
    k:int = 10
):
    #if os.path.exists(cache_file):
    #    with open(cache_file, "r") as f:
    #        return json.load(f)

    normalized_w_dec = ln_final(W_dec)    

    all_results = []
    for i in range(0, normalized_w_dec.shape[0], logit_batch_size):
        results = []
        # sometimes W_U is bigger than the vocab, in which case the last few tokens decode to nonsense anyway
        logit_scores = normalized_w_dec[i : i + logit_batch_size] @ W_U[:, :vocab.shape[0]]
        for j in range(0, logit_scores.shape[0], batch_size):
            scored_tokens = torch.sort(logit_scores[j : j + batch_size]).indices
            bottom_tokens = vocab[scored_tokens[:, :k].cpu().numpy()]
            top_tokens = vocab[scored_tokens[:, -k:].cpu().numpy()]
            results.append((bottom_tokens, top_tokens))
        all_results.append(tuple(np.concatenate(x) for x in zip(*results)))

    final_results = tuple(np.concatenate(x).tolist() for x in zip(*all_results))

    with open(cache_file, "w") as f:
        json.dump(final_results, f)

    return final_results


def load_compute_top_bottom_logits(model_name, transcoder_repo, output_dir, logit_batch_size=2**14, batch_size=256, k = 10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_cache_file = os.path.join(output_dir, "vocab.npy")
    vocab = compute_clean_vocab(tokenizer, vocab_cache_file)
    print("Precomputed vocab")

    # only load model if we end up computing any
    model = HookedTransformer.from_pretrained_no_processing(model_name)
    ln_final = model.ln_final
    W_U = model.unembed.W_U.clone()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    transcoders, _ = load_transcoder_from_hub(transcoder_repo, lazy_encoder=True)
    transcoder_name = transcoder_repo.split('/')[-1]

    tops = []
    for i, transcoder in enumerate(transcoders.transcoders):
        cache_file = f'{output_dir}/{transcoder_name}-{i}.json'
        top, _ = compute_top_bottom_tokens(ln_final, W_U, transcoder.W_dec, vocab, cache_file, logit_batch_size=logit_batch_size, 
                                           batch_size=batch_size)
        tops.append(top)
    return tops


#%%
logit_batch_size = 2**14
batch_size = 256

model_name = 'Qwen/Qwen3-0.6B'
models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"}

for model_name in models_and_transcoders.keys():
    tops = load_compute_top_bottom_logits(model_name, models_and_transcoders[model_name], '../cache/top_logits', 
            logit_batch_size=logit_batch_size, batch_size=batch_size, k=10)
