import os

import numpy as np
from autofaiss import build_index

indices_folder = "knn_indices"

embeddings = np.load("embeddings/text_embeddings.npy")
# embeddings = np.float32(np.random.rand(100, 512))

prompt_index_filename = os.path.join(indices_folder, "prompts.index")
infos_index_filename = os.path.join(indices_folder, "prompts-infos.index")
index, index_infos = build_index(
    embeddings,
    index_path=prompt_index_filename,
    index_infos_path=infos_index_filename,
)