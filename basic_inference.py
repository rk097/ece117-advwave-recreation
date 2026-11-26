#!/usr/bin/env python3
"""Run Qwen2-Audio-7B-Instruct on one or more local audio recordings.

Usage:
    python run_qwen.py default_A.wav silent_suffix_A.wav wind_suffix_A.wav
"""

import os
import sys
from pathlib import Path

# Keep downloaded model weights on RunPod's persistent /workspace volume.
# This must be set before importing Transformers / Hugging Face libraries.
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_qwen.py FILE.wav [FILE2.wav ...]")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Check that this RunPod Pod has a GPU.")

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model ready!", flush=True)

    sample_rate = processor.feature_extractor.sampling_rate
    for filename in sys.argv[1:]:
        if not Path(filename).is_file():
            print(f"\nFile: {filename}\nERROR: File not found", flush=True)
            continue

        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)

        # Same audio-analysis mode as the prior baseline experiment.
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": filename},
                {"type": "text", "text": "Answer the question spoken in this audio."},
            ],
        }]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = processor(
            text=prompt,
            audio=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        if "input_features" not in inputs:
            raise RuntimeError(
                "Audio features missing. Check the Transformers processor argument: audio=[audio]."
            )

        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )

        generated = generated[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"\nFile: {filename}")
        print(f"Response: {response}", flush=True)


if __name__ == "__main__":
    main()
