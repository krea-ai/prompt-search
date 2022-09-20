import os
import itertools
import glob

import numpy as np
from autofaiss import build_index

indices_folder = "knn_indices"

embeddings_dir = "visual_embeddings"
embeddings_path = os.path.join(embeddings_dir, "embeddings")
out_ids_path = os.path.join(embeddings_dir, "visual_ids.npy")

if not os.path.exists(out_ids_path):
    ids_path = os.path.join(embeddings_dir, "ids")

    ids_paths = glob.glob(f"{ids_path}/*")
    ids_paths.sort()
    ids = [[str(embedding_id) for embedding_id in np.load(path)]
           for path in ids_paths]
    ids = np.asarray(list(itertools.chain.from_iterable(ids)))

    np.save(out_ids_path, ids)

# embeddings = np.float32(np.random.rand(100, 512))

prompt_index_filename = os.path.join(indices_folder, "visual_prompts.index")
infos_index_filename = os.path.join(indices_folder,
                                    "visual-prompts-infos.index")
index, index_infos = build_index(
    embeddings_path,
    index_path=prompt_index_filename,
    index_infos_path=infos_index_filename,
)