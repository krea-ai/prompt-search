<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/krea_ai/prompt-search">
    <img src="prompt-search.png" alt="Logo" width="auto" height="200">
  </a>

<h3 align="center">prompt search</h3>

  <p align="center">
    simple implementation of CLIP search with a database of prompts.
    <br />
    <a href="https://krea.ai"><strong>explore prompts</strong></a>
    <br />
    <br />
    <a href="https://theprompter.substack.com/">newsletter</a>
    ·
    <a href="https://discord.gg/3mkFbvPYut">community</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
# About

This is the code that we used build our CLIP semantic search engine for [krea.ai](https://krea.ai). This work heavily inspired by [clip-retrieval](https://github.com/rom1504/clip-retrieval/), [autofaiss](https://github.com/criteo/autofaiss), and [CLIP-ONNX](https://github.com/Lednik7/CLIP-ONNX). We keept our implementation simple, focused on working with data from [open-prompts](https://github.com/krea-ai/open-prompts), and prepared to run efficiently on a CPU.

# CLIP Search

CLIP will serve us to find generated images given an input that can be a prompt or another image. It could also be used to find other prompts given the same input.

## CLIP
If you are not familiar with CLIP, we would recommend starting with the [blog](https://openai.com/blog/clip/) that OpenAI wrote about it. 

CLIP is a multi-modal neural network that can encode both, images and text in a common feature space. This means that we can create vectors that contain semantic information extracted from a text or an image. We can use these semantic vectors to compute operations such as cosine similarity, which would give us a similarity score. 

As a high level example, when CLIP extracts features from an image with a red car, it produces a similar vector to the one that it creates when it sees the text "a red car", or an image from another red car—since the semantics in all these elements are related.

So far, CLIP has been helpful for creating datasets like [LAION-5B](https://laion.ai/blog/laion-5b/), guiding generative models like VQ-GAN, for image classification tasks where there is not a lot of labeled data, or as a backbone for AI models like Stable Diffusion.

## Semantic Search
Semantic search consists of finding similar items within a dataset by comparing feature vectors. These feature vectors are also known as embeddings, and they can be computed in different ways. CLIP is one of the most interesting models for extracting features for semantic search. 

The search process consists of encoding items as embeddings, indexing them, and using these indices for fast search. Romain Beaumont wrote a great [medium post](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) about semantic search, we highly recommend reading it.

With this code, you will compute embeddings using CLIP, index them using K-Nearest Neighbors, and search for similarities efficiently given an input CLIP embedding.

# Environment

We used conda to set up our environment. You will basically need the following packages:

```
- autofaiss==2.15.3
- clip-by-openai==1.1
- onnxruntime==1.12.1
- onnx==1.11.0
```

We used `python 3.7`.

Create a new conda environment with the following command:

`conda env create -f environment.yml`


# Data Preparation

We will use a dataset of prompts generated with stable diffusion from [open-prompts](https://github.com/krea-ai/open-prompts). `1k.csv` is a subset from a [larger dataset](https://drive.google.com/file/d/1c4WHxtlzvHYd0UY5WCMJNn2EO-Aiv2A0/view) that you can find there—perfect for testing! 

# CLIP Search

You will need a GPU for this one. We recommend using [Lambda](https://lambdalabs.com/service/gpu-cloud)—great pricing and easy to set up.

## Set Up Lambda 
[Sign up](https://lambdalabs.com/cloud/entrance) to Lambda. Fill your information, complete the email verification and add a credit card.

Press the "Launch instance" button and introduce your public ssh key. Your public key should be on the folder in `~/.ssh/` in a file named `id_rsa.pub`. You can see its content with `cat ~/.ssh/id_rsa.pub`. Copy and paste the result to lambda and you'll be set. If you do not find this folder, check out [here](https://docs.oracle.com/en/cloud/cloud-at-customer/occ-get-started/generate-ssh-key-pair.html) how you can generate an ssh key, it's really straightforward.

For this project a *1x RTX 6000 (24GB)* should be enough. Launch the instance with the *Launch instance* button in the *Instances* page from the Lambda dasboard.

Once the instance is launched, wait for a minute or two until the Status of the machine says "Running". Then, copy the line under "SSH LOGIN", the one that looks like: `ssh ubuntu@<ip-address>`, where the `<ip-address>` will be a series of numbers in the form `123.456.789.012`. Paste it on your terminal, type "yes" to the prompt that will appear and you'll have accessed to your new machine with an GPU!
First, *sign in* or *sign up* to [Lambda](https://lambdalabs.com/cloud/entrance). 

## Search Images
### Download images
The first step will consist of downloading the images from the `csv` file that contains all the prompt data. To do so, we will leverage the [img2dataset](https://github.com/rom1504/img2dataset) package.

Execute the following command to create a new file with links from images:

```
python extract_img_links_from_csv.py
```

Note that by default, the process will create the links from `1k.csv`. Change the `CSV_FILE` variable in `extract_img_links_from_csv.py` if you want to use another data file as input.

```python
CSV_FILE = "./1k.csv"
OUTPUT_FILE = "./img_links.txt"
```

The results will be stored in `img_links.txt`. 

Run the following command to download images:

```bash
img2dataset --url_list img_links.txt --output_folder imgs --thread_count=64 --image_size=256
```

The output will be stored in a sub-folder named `00000` within `imgs`.

### Compute Visual CLIP Embeddings

Once the folder `imgs` is created and filled with generated images, you can run the following command to compute visual CLIP embeddings for each of them:

`python extract_visual_clip_embeddings.py`

The following are the main parameters that you might need to change from `extract_visual_clip_embeddings.py`:
```
IMG_DIR = "./imgs/00000" #directory where all your images were downloaded
BATCH_SIZE = 128 #number of CLIP embeddings computed at each iterations
NUM_WORKERS = 14 #number of workers that will run in parallel (recommended is number_of_cores - 2)
```

Once the process is finished, you will see a new folder named `visual_embeddings`. This folder will contain two other folders named `ids` and `embeddings`. `ids` will contain `.npy` files with information of the `ids` of each generation computed at each batch. `embeddings` will contain `.npy` files with the resulting embeddings computed at each batch. This data will be useful for computing the KNN indices.

### Compute Visual KNN indices
If you did not make any modifications in the default output structure from the previous step, this process should be as easy as running the following command:

`python create_visual_knn_indices.py`

Otherwise, you might want to modify the following variables from `create_visual_knn_indices.py`:

```python
INDICES_FOLDER = "knn_indices"
EMBEDDINGS_DIR = "visual_embeddings"
```

The result will be stored within a new folder named `knn_indices` in a file named `visual_prompts.index`.


### Search Images 

In order to search generated images more efficiently, we will use an ONNX version of CLIP. We will use the implementation from [`CLIP-ONNX`](https://github.com/Lednik7/CLIP-ONNX) for this.

Install the following package:
```bash
pip install git+https://github.com/Lednik7/CLIP-ONNX.git --no-deps
```

Once installed, download the ONNX CLIP models with the following commands:
```
wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx
wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx
```

Finally, execute the following line to perform the search with regular CLIP and ONNX CLIP:

```
python test_visual_knn_index.py
```

The result should be a list of image filenames that are the most similar to the prompt `"image of a blue robot with red background"` and the image `prompt-search.png`.

Change the following parameters in `test_visual_knn_index.py` to try out different input prompts and images:

```python
INPUT_IMG_PATH = "./prompt-search.png"
INPUT_PROMPT = "image of a blue robot with red background"
NUM_RESULTS = 5    
```

Have fun!

# Get in touch

- Follow and DM us on Twitter: [@krea_ai](https://twitter.com/krea_ai)
- Join [our Discord community](https://discord.gg/3mkFbvPYut)
- Email either `v` or `d` (`v` at `krea` dot `ai`; `d` at `krea` dot `ai` respectively)