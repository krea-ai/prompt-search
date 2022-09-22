<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/krea_ai/prompt-search">
    <img src="prompt-search.png" alt="Logo" width="auto" height="200">
  </a>

<h3 align="center">prompt search</h3>

  <p align="center">
    simple implementation of CLIP search with a prompt database.
    <br />
    <a href="https://krea.ai"><strong>explore prompts</strong></a>
    <br />
    <br />
    <a href="https://theprompter.substack.com/">newsletter</a>
    ·
    <a href="https://discord.gg/3mkFbvPYut">community</a>
    ·
    <a href="#contributing">contribute</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
# About

This repository contain the implementations we developed to build a semantic search engine with CLIP. Our code is heavily inspired by [clip-retrieval](https://github.com/rom1504/clip-retrieval/), [autofaiss](https://github.com/criteo/autofaiss), and [CLIP-ONNX](https://github.com/Lednik7/CLIP-ONNX). We keept our implementation simple, focused on working with data from [https://github.com/krea-ai/open-prompts](open-prompts), and prepared to run efficiently on a CPU.

# CLIP Search

CLIP will serve us to find generated images given an input that can be a prompt or another image. It could also be used to find other prompts given the same input.

## CLIP
If you are not familiar with CLIP, we would recommend starting with the [blog](https://openai.com/blog/clip/) that OpenAI wrote about it. 

CLIP is a multi-modal neural network that can encode both, images and text in a common feature space. This means that we can create vectors that contain semantic information extracted from a text or an image. We can use these semantic vectors to compute operations such as cosine similarity, which would give us a similarity score. 

As a high level example, when CLIP extracts features from an image with a red car, it produces a similar vector to the one that it creates when extracting the features from the text "a red car", or the image from another red car, since the semantics in all these elements are related.

So far, CLIP has been helpful for creating datasets like [LAION-5B](https://laion.ai/blog/laion-5b/), guiding generative models like VQ-GAN, for image classification tasks where there is not a lot of labeled data, or as a backbone for AI models like Stable Diffusion.

## Semantic Search
Given a piece of data such as an image or a text description, semantic search consists of finding similar items within a dataset by comparing feature vectors. These feature vectors are also known as embeddings, and they can be computed in different ways. CLIP is one of the most interesting models for extracting features for semantic search. 

The search process consists of encoding items as embeddings, indexing them, and using these indices for fast search in order to build semantic systems. Romain Beaumont wrote a great [medium post](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) about semantic search, we highly recommend reading it.

With the code in this project you can compute embeddings using CLIP, index them using K-Nearest Neighbors, and search for similarities efficiently given an input CLIP embedding. Note that for the input embedding we will be able to use a vector computed from a text or an image, and we can also index CLIP embeddings from both, images and texts.

# Environment

We used conda to set up our environment. You will basically need the following packages:

```
- autofaiss==2.15.3
- clip-by-openai==1.1
- torch==1.12.1
- torchvision==0.13.1
```

We used `python 3.7`.

Create a new conda environment with the following command:

`conda env create -f environment.yml`


# Data Preparation

We will use a dataset of prompts generated with stable diffusion from [open-prompts](https://github.com/krea-ai/open-prompts). `1.csv` contains a subset of the dataset with 1000 elements, we recommend using it the first time you run the code to confirm that everything worked well without having to wait for millions items to be downloaded and processed. You can get it from [here](https://github.com/krea-ai/open-prompts/blob/main/data/1k.csv). It has the same structure as `prompts.csv`, which contains the whole dataset. `prompts.csv` can be downloadedf from [here](https://drive.google.com/file/d/1c4WHxtlzvHYd0UY5WCMJNn2EO-Aiv2A0/view).

# CLIP Search

You will need a GPU for this one. We recommend using [https://lambdalabs.com/service/gpu-cloud](Lambda)—great pricing and easy to set up.

## Image Search
### Compute Visual CLIP Embeddings
The first step will consist of downloading the images from the `csv` file that we downloaded. Create a folder named `imgs` with all this data (it might take a while).

Once imgs is filled with images, you can run the following command to compute the visual CLIP embeddings from each of them:

`python extract_visual_clip_embeddings.py`

The following are the main parameters that you might need to change: 
```
IMG_DIR = "./imgs" #directory where all your images were downloaded
DATA_PATH = "./1k.csv" #path to the data that was used to download the images
BATCH_SIZE = 128 #number of CLIP embeddings computed at each iterations
NUM_WORKERS = 14 #number of workers that will run in parallel (recommended is number_of_cores - 2)
```

The default value for `DATA_PATH` is the small dataset. Make sure to change it for the [larger](https://drive.google.com/file/d/1c4WHxtlzvHYd0UY5WCMJNn2EO-Aiv2A0/view) one once you made sure that everything worked.

Note that the filename of the images in `imgs` and the `ID` from the `csv` with the data are correlated. By default, the dataset that we used is structured in this way, but you might want to make sure that this is consistent with your dataset if you use custom data.

Once the process is finished, you will see a new folder named `visual_embeddings`. This folder will contain two other folders named `ids` and `embeddings`. `ids` will contain `.npy` files with information of the `ids` of each generation computed at each batch. `embeddings` will contain `.npy` files with the resulting embeddings computed at each batch. This data will be useful for computing the KNN indices, since we need to have information about both, the CLIP embedding and the ID they represent.

### Compute Visual KNN indices
If you did not make any modifications in the default output structure from the previous step, this process should be as easy as running the following command:

`python create_visual_knn_indices.py`

This script will read all the ids 

### Search Images 
#### Search from Prompts
#### Search from Images


## Prompt Search
### Compute Textual CLIP Embeddings

### Text-to-image and Image-to-image CLIP-search 

### Search Prompts from Prompts

### Search Prompts 
#### Search from Prompts

#### Search from Images