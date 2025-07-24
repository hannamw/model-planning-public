#%%
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

ckpt = "google/gemma-3-27b-it"
model = Gemma3ForCausalLM.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
#%%
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Finish this rhyming couplet; just tell me the next line:\n Boxes of books, a reader's delight,"},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

generation = model.generate(**inputs, max_new_tokens=40, do_sample=True)
generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
# %%
