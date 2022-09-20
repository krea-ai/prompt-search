import os

import numpy as np
from autofaiss import build_index

indices_folder = "knn_indices"

embeddings = np.load("embeddings/text_embeddings.npy")

prompt_index_filename = os.path.join(indices_folder, "prompts.index")
index, index_infos = build_index(
    embeddings,
    index_path=prompt_index_filename,
)