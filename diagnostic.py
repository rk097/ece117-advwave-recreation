import torch
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration
)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

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

t = torch.arange(16000) / 16000
waveform = torch.sin(2 * torch.pi * 440 * t)
waveform.requires_grad_(True)

inputs = processor(
    text=prompt,
    audio=[waveform],
    sampling_rate=16000,
    return_tensors="pt"
)

features = inputs["input_features"]

print("Waveform requires_grad:", waveform.requires_grad)
print("Features requires_grad:", features.requires_grad)
print("Features grad_fn:", features.grad_fn)