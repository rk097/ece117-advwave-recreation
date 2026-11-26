#!/usr/bin/env python3

# for actual attack replication

import torch
import librosa

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration
)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map={"": 0}
)

model.eval()
model.requires_grad_(False)
model.config.use_cache = False

print("Model loaded")

audio, sr = librosa.load(
    "apple.wav",
    sr=16000,
    mono=True
)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": "apple.wav"},
            {
                "type": "text",
                "text": "What word is spoken in this recording?"
            }
        ]
    }
]

prompt = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False
)

inputs = processor(
    text=prompt,
    audio=[audio],
    sampling_rate=16000,
    return_tensors="pt"
).to("cuda")

features = (
    inputs.pop("input_features")
    .detach()
    .float()
    .requires_grad_(True)
)

torch.cuda.reset_peak_memory_stats()

outputs = model(
    **inputs,
    input_features=features.to(torch.bfloat16),
    use_cache=False
)

# A simple diagnostic objective
loss = outputs.logits[:, -1, :].float().square().mean()

grad, = torch.autograd.grad(loss, features)

print("Feature shape:", features.shape)
print("Gradient shape:", grad.shape)
print("Gradient norm:", grad.norm().item())

print(
    "Peak VRAM:",
    torch.cuda.max_memory_allocated() / 2**30,
    "GiB"
)

logits = outputs.logits[0, -1, :].detach().float()

probs = torch.softmax(logits, dim=-1)

values, indices = torch.topk(probs, k=10)

print("\nTop 10 next-token predictions:")

for probability, token_id in zip(values, indices):
    token = processor.tokenizer.decode([token_id.item()])

    print(
        repr(token),
        f"{probability.item():.6f}"
    )

print("\nLogits shape:", outputs.logits.shape)

print(
    "Actual audio frames:",
    inputs["feature_attention_mask"].sum().item()
)