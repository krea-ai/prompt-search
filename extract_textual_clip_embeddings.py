import os
import csv

import numpy as np
import torch
import clip

USE_CACHE = False
BATCH_SIZE = 2048
OUTDIR = "embeddings"

os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    _headers = next(reader)

    prompt_data = set([(row[0], row[1]) for row in reader if row[1] != ''])

prompt_ids = [data[0] for data in prompt_data]
prompts = (data[1] for data in prompt_data)

prompt_ids_filename = os.path.join(OUTDIR, f"prompt_ids.npy")
np.save(prompt_ids_filename, prompt_ids)

text_embeddings = None
batched_prompts = []
for idx, prompt in enumerate(prompts):
    batched_prompts.append(prompt)

    if len(batched_prompts) % BATCH_SIZE == 0 or idx == len(prompt_ids) - 1:
        print(f"processing -- {idx + 1}")

        batch_text_embeddings_filename = os.path.join(
            OUTDIR, f"text_embeddings_{idx + 1}.npy")

        if os.path.exists(batch_text_embeddings_filename) and USE_CACHE:
            batch_text_embeddings = np.load(batch_text_embeddings_filename)

        else:
            with torch.no_grad():
                batched_text = clip.tokenize(
                    batched_prompts,
                    truncate=True,
                ).to(device)

                batch_text_embeddings = model.encode_text(batched_text, )
                batch_text_embeddings /= batch_text_embeddings.norm(
                    dim=-1, keepdim=True)

            batch_text_embeddings = batch_text_embeddings.cpu().numpy().astype(
                'float32')

            if USE_CACHE:
                np.save(batch_text_embeddings_filename, batch_text_embeddings)

        if text_embeddings is None:
            text_embeddings = batch_text_embeddings

        else:
            text_embeddings = np.concatenate(
                (text_embeddings, batch_text_embeddings))

        print(f"text embeddings shape -- {text_embeddings.shape}")
        print("\n")

        batched_prompts = []

print(f"{len(text_embeddings)} CLIP embeddings extracted!")
text_embeddings_filename = os.path.join(OUTDIR, f"text_embeddings.npy")
np.save(text_embeddings_filename, text_embeddings)