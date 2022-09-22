import os
import csv
import glob

import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC

except ImportError:
    BICUBIC = Image.BICUBIC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_CACHE = False
IMG_DIR = "./imgs/00000"
BATCH_SIZE = 128
NUM_WORKERS = 14
PERFETCH_FACTOR = 14
OUTDIR = "./visual_embeddings"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(OUTDIR + "/ids", exist_ok=True)
os.makedirs(OUTDIR + "/embeddings", exist_ok=True)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class CLIPImgDataset(Dataset):

    def __init__(
        self,
        img_dir: str,
    ):
        self.img_paths = glob.glob(f"{img_dir}/*.jpg", )

        self.transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(
        self,
        idx,
    ):
        img_path = self.img_paths[idx]

        generation_id = img_path.split("/")[-1].split(".")[0]

        img = Image.open(img_path)
        img = self.transform(img)

        return img, generation_id


def main():
    print("setting up dataloader...")
    model, _preprocess = clip.load(
        "ViT-B/32",
        device=DEVICE,
    )
    clip_img_dataset = CLIPImgDataset(img_dir=IMG_DIR, )
    clip_img_dataloader = DataLoader(
        clip_img_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PERFETCH_FACTOR,
        persistent_workers=True,
        # multiprocessing_context="spawn",
    )

    print("starting to process!")
    for idx, (batched_imgs,
              generation_ids) in enumerate(clip_img_dataloader):
        print(f"processing! -- {(idx + 1) * BATCH_SIZE} / {len(clip_img_dataset)}")

        with torch.no_grad():
            batched_img_embeddings = model.visual(
                batched_imgs.to(DEVICE, torch.float16), )

        batched_img_embeddings /= batched_img_embeddings.norm(dim=-1,
                                                              keepdim=True)
        batched_img_embeddings = batched_img_embeddings.cpu().numpy().astype(
            'float32')

        prompt_ids_filename = os.path.join(OUTDIR,
                                           f"ids/{str(idx).zfill(9)}.npy")
        np.save(prompt_ids_filename, np.asarray(generation_ids))

        img_embeddings_filename = os.path.join(
            OUTDIR, f"embeddings/{str(idx).zfill(9)}.npy")
        np.save(img_embeddings_filename, batched_img_embeddings)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()