# ControlNet Single-Image Relighting
This is the secondary repository of our work "A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis".
The main repository is available at (https://github.com/graphdeco-inria/generative-radiance-field-relighting); this repository contains code for the 2D single-image relighting network we describe in the paper. 
To use our method for relightable novel-view synthesis with Gaussian Splatting, you will first need to setup this codebase and use it to transform your single-illumination capture into a (generated) multi-illumination capture. 

## Installation
```
git clone https://gitlab.inria.fr/ypoirier/controlnet-diffusers-relighting.git
```
First clone this repository, which includes a modified copy of "A Dataset of Multi-Illumination Images in the Wild" (https://projects.csail.mit.edu/illumination/) in `multi_illumination/` and of "Diffusers" (https://github.com/huggingface/diffusers) in `diffusers/`. 

**Important** You will need to make these modules visible in your search path. This can be done with:
```bash 
export PYTHONPATH=$PYTHONPATH:multi_illumination:diffusers/src
```
Make sure you do **not** have a copy of `diffusers` installed using pip.

### Creating the environment
Then create a virtual environment on python `3.9.7` and install the requirements in `requirements.txt`:
```bash
conda create -n csir python==3.9.7
conda activate csir
pip install -r requirements.txt
```
Other versions likely work fine, but we have not tried them. We recommend using a separate environment as the one you use for Gaussian Splatting. 

### Downloading Pretrained weights
Pretrained ControlNet weights and decoder weights can be downloaded with: 
```bash
wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/content/weights/{controlnet,decoder}_1536x1024.safetensors -P weights/
```

Which will place them into `weights/`. Both sets of weights are required for inference. The weights for Stable Diffusion and the Marigold depth estimator (https://github.com/prs-eth/Marigold) are also needed, but will download automatically as the code runs and end up in `~/.cache/huggingface/`.
 
(Optional) In the paper, we also trained at a smaller resolution network for quantitative evaluation against other methods. To download the weights for these networks, simply replace `1536x1024` with `768x512` in the previous URL.

## Inference
Inference scripts can be run directly from these weights. You will need at least 20GB of GPU memory.



### Relighting single images
To try our networks on a individual images, use the `sample_single_image.py` script. Some example images are already provided in the `exemplars` directory. 

```bash
python sample_single_image.py --image_paths exemplars/*.png
```
Images will be saved into `samples/`. 

You can select which directions to relight to using `--dir_ids`. The directions are numbered using the convention from "A Dataset of Multi-Illumination Images in the Wild", and their exact coordinates are listed in `relighting/light_directions.py`.
On an A40 card inference takes 3-4 seconds per relighting.

### Relighting entire colmap captures
You can download our scenes at: https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/datasets/
If you wish to use your own, we expect capture images with the following structure:

```bash
colmap/**/<scene_name>/
    └── train/
        └── images/
            ├── 0000.png
            ├── 0001.png
            └── ...
```
Normally this will a Colmap output directory. For example, for our scene `colmap/real/garage_wall`, you can launch sampling with:

```bash
python sample_entire_capture.py --capture_paths colmap/real/garage_wall
```

Relit images will be saved into `colmap/real/garage_wall/relit_images/`.
On an A40 relighting a dataset takes approximatively 1 minute per image. So for a capture with 100 images you can expect about 1 hour and a half of processing to generate all relit images. 

### (Optional) Rendering light sweeps
We provide a small script that renders a short video where the light direction is interpolated between different values. 

```
python sample_and_render_light_sweep.py --image_paths exemplars/kettle.png
```

Producing the video requires having `ffmpeg` available on your system. The sampled images will be saved in `sweep_samples/` and the videos in `sweep_videos/`.

## Training
Coming soon. 

<!-- To train from scratch, you must first download the dataset or provide your own data in a similar format. The main ControlNet network and the conditional decoder use the same data but are trained compeltely separately.

### Downloading and preparing the datasets
We train on images from "A Dataset of Multi-Illumination Images in the Wild" (https://projects.csail.mit.edu/illumination/). These images are downloaded, tonemapped, and resized to the target resolution with:

```bash
python download_multilum_images.py
python predict_depths_for_multilum_images.py
```

The second command predicts depth maps using Marigold. Images are saved in `multilum_images/`; this directory will contain the original `.exr` images as well as the processed images at resolutions `1536x1024` and `768x512`. Expect the whole process to take several hours. After the depths are predicted you can delete directory `multilum_images/fullres_exr_images` to save space.

### Training ControlNet
Once the data is ready, you can train with:
 todo: verify that the script works
```bash
bash train_controlnet_1536x1024.sh
```
 todo: reivew xformers install

By default, this will automatically use all available GPUs. Training took us approximatively 3 days on 4 A6000 cards. Training for longer still reduces the training loss somewhat but seems to worsen generalization. It is normal if the ouputs have incorrect colors at the begining of training. You can train in half resolution for prototyping, which is about 4 times faster (see [below](#optional-working-at-half-resolution-and-evaluation-on-synthetic-scenes-halfres)). 


`aim` is used for logging training losses and can be opened with the `aim up` command. 

 todo: enable aim in the launch scripts
 todo: clean up validation images
 todo: weights will save in ... + save weights in the expected format
 todo: configure automatic precision
 todo: configure # of iterations
 todo: does float32 fit in memory?

Note that training loop looks into the file `relighting/training_pairs.jsonl` which contains a list of all $(\mathrm{input}, \mathrm{target})$ image path pairs. If you provide your own data, you will need to modify this file.

### Fine-tuning the conditional decoder 
The conditional Asymmetric VQGAN decoder from "Designing a Better Asymmetric VQGAN for StableDiffusion" must also be fine-tuned on our task from their pretrained weights. This training process is entirely independent of the main ControlNet; as long as you do not change the training data you can always reuse the weights we provide for any ControlNet training instead of running this step. To launch the fine-tuning use:

```bash
bash train_decoder_1536x1024.sh
```

The initial weights will download automatically the first time the script runs. Note we have not implemented code for training on multi-gpu, but convergence is very fast and should only take a few hours.
 todo: review the remove OAR test line 999 (always use aim?)
 todo: verify that it still trains
 todo: verify that the weights save at the end

### Using the weights for inference
Our inference scripts assume that the weights are placed in `weights/` and that the config is unchanged (see `relighting/inference_pipeline.py` if you want to change the config).
 todo: say how to use your weights

## (Optional) Working at half resolution
We ran quantiative evaluation against other methods in `768x512` instead of the full `1536x1024`. All preparation and training scripts in this README can be run by in half resolution by replacing `1536x1024` with `768x512`. For the sampling scripts, pass the flags `--width 768 --height 512`.
 todo: test single image
 todo: test generate dataset -->

## (Optional) Computing the illumination direction from light probes
We computed the directions of illumination in the `multilum` dataset using the diffuse light probes (gray spheres). You can reproduce this with:
```
python compute_light_directions.py
```

## Citing our work
Please cite us with:
```bibtext
@article{
      10.1111:cgf.15147,
      journal = {Computer Graphics Forum},
      title = {{A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis}},
      author = {Poirier-Ginter, Yohan and Gauthier, Alban and Philip, Julien and Lalonde, Jean-François and Drettakis, George},
      year = {2024},
      publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
      ISSN = {1467-8659},
      DOI = {10.1111/cgf.15147}
    }


 todo: have one last look for unused files + the ignore, make sure its all good online
