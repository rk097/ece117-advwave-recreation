#!/usr/bin/env python3

# for actual attack replication.
# ------ load model

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
                "text":  (
                            "Transcribe the spoken word. "
                            "Return only the word in lowercase, "
                            "without explanation or punctuation."
                        )
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

##------------------ testing loss on apple and variants

import torch.nn.functional as F

correct_words = ["apple", "Apple", " apple", " Apple"]

correct_ids = []

for word in correct_words:
    ids = processor.tokenizer.encode(
        word, add_special_tokens=False
    )

    if len(ids) == 1:
        correct_ids.append(ids[0])

correct_ids = sorted(set(correct_ids))

logits = outputs.logits[:, -1, :].float()
log_probs = torch.log_softmax(logits, dim=-1)

correct_log_prob = torch.logsumexp(
    log_probs[:, correct_ids],
    dim=-1
)

loss = -correct_log_prob.mean()

grad, = torch.autograd.grad(loss, features)

print("Recognition loss:", loss.item())
print("Correct-word probability:",
      correct_log_prob.exp().item())
print("Gradient norm:", grad.norm().item())

## ------- experiment
@torch.no_grad()
def generate_response(audio_features):

    generated_ids = model.generate(
        **inputs,
        input_features=audio_features.to(torch.bfloat16),
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        use_cache=True
    )

    response_ids = generated_ids[
        :, inputs["input_ids"].shape[1]:
    ]

    return processor.batch_decode(
        response_ids,
        skip_special_tokens=True
    )[0]


## ----------- adding random perturbation and calculating loss

original = features.detach().clone()

mask = inputs["feature_attention_mask"]
mask = mask.unsqueeze(1).to(original.dtype)

torch.manual_seed(42)

print("\nRandom perturbation experiment:")

for epsilon in [0, 0.001, 0.01, 0.05, 0.1]:

    noise = torch.empty_like(original).uniform_(
        -epsilon, epsilon
    )

    modified = original + noise * mask

    with torch.no_grad():
        result = model(
            **inputs,
            input_features=modified.to(torch.bfloat16),
            use_cache=False
        )

        probs = result.logits[0, -1, :].float().softmax(-1)

        correct_prob = probs[correct_ids].sum().item()

        predicted_id = probs.argmax().item()
        predicted_token = processor.tokenizer.decode(
            [predicted_id]
        )

    print(
        f"Epsilon: {epsilon:.3f} | "
        f"Correct probability: {correct_prob:.6f} | "
        f"Prediction: {predicted_token!r}"
    )

    response = generate_response(modified)
    print(f"Full response: {response!r}")

    del result

## --------------- FGSM adversarial feature manipulation

print("\nFGSM word recognition experiment:")

for epsilon in [0, 0.001, 0.01, 0.05, 0.1]:

    perturbation = epsilon * grad.sign() * mask

    modified = original + perturbation

    with torch.no_grad():
        result = model(
            **inputs,
            input_features=modified.to(torch.bfloat16),
            use_cache=False
        )

        probs = result.logits[0, -1, :].float().softmax(-1)

        correct_prob = probs[correct_ids].sum().item()

        predicted_id = probs.argmax().item()

        predicted_token = processor.tokenizer.decode(
            [predicted_id]
        )

    print(
        f"Epsilon: {epsilon:.3f} | "
        f"Correct probability: {correct_prob:.6f} | "
        f"Prediction: {predicted_token!r}"
    )

    response = generate_response(modified)
    print(f"Full response: {response!r}")

    del result