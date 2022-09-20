import os
import faiss
import numpy as np

import clip
import torch
from PIL import Image
from clip_onnx import clip_onnx

indices_folder = "knn_indices"

prompt_index_filename = os.path.join(indices_folder, "visual_prompts.index")
embeddings_dir = "visual_embeddings"

prompt_ids = np.load(os.path.join(embeddings_dir, "visual_ids.npy"))

loaded_index = faiss.read_index(
    prompt_index_filename,
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
)

img_path = "img.jpg"

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
onnx_model = clip_onnx(None)
onnx_model.load_onnx(
    visual_path="visual.onnx",
    textual_path="textual.onnx",
    logit_scale=100.0000,
)
onnx_model.start_sessions(providers=["CPUExecutionProvider"], )

img = Image.open(img_path)
processed_img = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    visual_embedding = model.encode_image(processed_img, )
    visual_embedding /= visual_embedding.norm(dim=-1, keepdim=True)

    visual_embedding = visual_embedding.cpu().numpy().astype('float32')

_, I = loaded_index.search(visual_embedding, 5)
print("CLIP RESULTS")
print([f"{str(prompt_ids[idx])}" for idx in I[0]])

# image = image.detach().cpu().numpy().astype(np.int64)
onnx_text_embedding = onnx_model.encode_image(processed_img, )
# onnx_text_embedding /= onnx_text_embedding.norm(dim=-1, keepdim=True)
onnx_text_embedding = np.around(onnx_text_embedding, decimals=4)

_, I = loaded_index.search(onnx_text_embedding, 5)
print("ONNX RESULTS")
print([f"{str(prompt_ids[idx])}" for idx in I[0]])