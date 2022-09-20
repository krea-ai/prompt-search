import os
import itertools
import glob

import numpy as np
from autofaiss import build_index

INDICES_FOLDER = "knn_indices"
EMBEDDINGS_DIR = "visual_embeddings"

embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings")
out_ids_path = os.path.join(EMBEDDINGS_DIR, "visual_ids.npy")

if not os.path.exists(out_ids_path):
    ids_path = os.path.join(EMBEDDINGS_DIR, "ids")

    ids_paths = glob.glob(f"{ids_path}/*")
    ids_paths.sort()
    ids = [[str(embedding_id) for embedding_id in np.load(path)]
           for path in ids_paths]
    ids = np.asarray(list(itertools.chain.from_iterable(ids)))

    np.save(out_ids_path, ids)

prompt_index_filename = os.path.join(INDICES_FOLDER, "visual_prompts.index")
infos_index_filename = os.path.join(INDICES_FOLDER,
                                    "visual-prompts-infos.index")
index, index_infos = build_index(
    embeddings_path,
    index_path=prompt_index_filename,
    index_infos_path=infos_index_filename,
)