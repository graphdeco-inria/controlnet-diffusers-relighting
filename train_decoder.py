#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from torchvision.utils import save_image
import accelerate
import numpy as np
import functools
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import gc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from relighting.light_directions import get_light_dir_encoding
from aim import Run
from pprint import pprint
from lpips import LPIPS


import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline
from torchvision.utils import save_image
import torch 


def is_wandb_available():
    return False
# if is_wandb_available():
#     import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


@torch.no_grad()
def log_validation(vae, args, accelerator, weight_dtype, step, num_samples=None, save_grid=True, text_encoder=None, tokenizer=None):
    logger.info("Running validation... ")
    if num_samples is None:
        num_samples = args.num_validation_images

    vae = accelerator.unwrap_model(vae)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images_controlnet = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images_controlnet = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images_controlnet = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    val_index = 0
    if False:
        extra_validation_images_controlnet = []
    else:
        extra_validation_images_controlnet = [ x for x in os.listdir("extra_validation_images/controlnet") if "_gt." not in x ]
        extra_validation_images_decoder = [ x for x in os.listdir("extra_validation_images/decoder") if "_gt." not in x and not x.endswith("_cond.png") ]
        
    for val_im_name in extra_validation_images_controlnet + extra_validation_images_decoder:
        if val_im_name in extra_validation_images_controlnet:
            val_image_path = "extra_validation_images" + "/" + val_im_name
            validation_image = Image.open(val_image_path).convert("RGB")
            validation_image = validation_image.resize(image_size, Image.LANCZOS)
        elif val_im_name in extra_validation_images_decoder:
            val_image_path = "validation_decoder" + "/" + val_im_name
            validation_image = Image.open(val_image_path).convert("RGB")
            validation_image = validation_image.resize(image_size, Image.LANCZOS)
            cond_image = Image.open(val_image_path.replace(".png", "_cond.png")).convert("RGB")
            cond_image = cond_image.resize(image_size, Image.LANCZOS)
            cond_image = TF.to_tensor(cond_image)
        else:
            val_image_path = (args.images_dir + "/" + val_im_name).replace(".jpg", ".png")
            validation_image = Image.open(val_image_path).convert("RGB")
            image_size = validation_image.size
        validation_image = TF.to_tensor(validation_image)

        target_dir = random.randint(0, 24)
        filename = val_image_path.split("/")[-1]
        if val_im_name in extra_validation_images_controlnet:
            cond_image = validation_image.to(dtype=weight_dtype).cuda()
        elif val_im_name in extra_validation_images_decoder:
            cond_image = cond_image.to(dtype=weight_dtype).cuda()
        else:
            name, ext = filename.split(".")
            new_name = "_".join(name.split("_")[:-1]) + "_" + str(target_dir)
            cond_image_path = os.path.dirname(val_image_path) + "/" + new_name + "." + ext
            cond_image = TF.to_tensor(Image.open(cond_image_path).convert("RGB")).to(weight_dtype).cuda()

        images = []

        with torch.autocast("cuda", dtype=weight_dtype):
            vae_in = validation_image.to(dtype=weight_dtype)[None].cuda() * 2 - 1
            if args.black_and_white:
                from kornia.color import rgb_to_lab, lab_to_rgb
                lab = rgb_to_lab(vae_in / 2 + 0.5)
                lab[:, 1:] *= 0
                vae_in = lab_to_rgb(lab) * 2 - 1
            latents = vae.encode(vae_in).latent_dist.mean

            mask = torch.zeros_like(vae_in).mean(dim=1, keepdim=True).cuda()
            image = vae.decode(latents, cond_image[None] * 2 - 1, mask).sample.detach() / 2 + 0.5
   
        images.append(image)
        
        image_logs.append(
            {"validation_image": TF.to_pil_image(validation_image.float()), "images": [TF.to_pil_image(x[0].float()) for x in images], "validation_prompt": f"val_{val_index}", "gt": TF.to_pil_image(validation_image.float()), "extra": val_im_name in validation_images_controlnet}
        )

        val_grid_path =  os.path.join(args.output_dir, val_im_name.replace(".", f"_step_{step}.") if val_im_name in validation_images_controlnet or val_im_name in validation_images_decoder else f"val_{val_index}_step{step}.png")
        if save_grid:
            if args.direct_cond_residual:
                save_image(torch.stack([cond_image.cuda(), validation_image.cuda() - cond_image.cuda(), image[0].cuda(), image[0].cuda() + cond_image.cuda(), validation_image.cuda()]), val_grid_path, padding=0)
            else:
                save_image(torch.stack([cond_image.cuda(), image[0].cuda(), validation_image.cuda()]), val_grid_path, padding=0)

        val_index += 1

    return image_logs



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        default="unipc"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=25000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    ) 
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_old_dirs",
        action="store_true",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--thousand_iters_latent_training",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2_500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--use_flash_dir",
        action="store_true",
    )
    parser.add_argument(
        "--dir_sh",
        type=int, 
        default=-1
    )
    parser.add_argument(
        "--use_probes",
        action="store_true",
    )
    parser.add_argument(
        "--probes_patch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
    )
    parser.add_argument(
        "--disable_aim",
        action="store_true",
    )
    parser.add_argument(
        "--skip_custom",
        action="store_true",
    )
    parser.add_argument(
        "--skip_first_val",
        action="store_true",
    )
    parser.add_argument(
        "--sampling_cfg_scale",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.0
    )
    # parser.add_argument(
    #     "--hs_weight",
    #     type=float,
    #     default=0.0
    # )
    # parser.add_argument(
    #     "--augment",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--augment_strategy_2",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--augment_everything",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--sample_from_ckpt",
    #     type=str,
    #     default=None
    # )
    #!!!
    # parser.add_argument(
    #     "--augment_factor",
    #     type=float,
    #     default=1.0
    # )
    # parser.add_argument(
    #     "--color_dropout",
    #     type=float,
    #     default=0.0
    # )
    # parser.add_argument(
    #     "--black_and_white",
    #     action="store_true"
    # )
    # parser.add_argument(
    #     "--learn_residuals",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--L_target",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--hs_residual",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--direct_cond_residual",
    #     action="store_true",
    # )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
 
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # if args.validation_prompt is not None and args.validation_image is None:
    #     raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    # if args.validation_prompt is None and args.validation_image is not None:
    #     raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    # if args.resolution % 8 != 0:
    #     raise ValueError(
    #         "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
    #     )

    return args

import json
import torchvision.transforms.functional as TF

INPUT_IDS = torch.tensor([
        49406, 49407,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0
] # this is a tokenized version of the empty string
)

compute_embeddings_fn = None
from PIL import UnidentifiedImageError

class MyDataset:
    def __init__(self, test=False):
        self.json = [
            json.loads(line) for line in open("relighting/training_pairs.jsonl", "r").read().splitlines()
        ]

    def __len__(self):
        return len(self.json)
    
    def __getitem__(self, i):
        try:
            "Should return a tensor in range [-1, 1]"
            name = self.json[i]["image"].replace(".jpg", ".png")
            image_path = args.images_dir + "/" + name
            image = Image.open(image_path).convert("RGB")
            conditioning_image_path = args.images_dir + "/" + self.json[i]["conditioning_image"].replace(".jpg", ".png")
            conditioning_image = Image.open(conditioning_image_path).convert("RGB")
            k = int(image_path.split("_dir_")[-1].replace(".png", ""))
            target_dir = get_light_dir_encoding(k)

            result = {
                "text": "",
                "target_dir": target_dir,
                "pixel_values": TF.to_tensor(image) * 2 - 1,
                "conditioning_image": conditioning_image,
                "conditioning_pixel_values": TF.to_tensor(conditioning_image) * 2 - 1,
                "input_ids": INPUT_IDS
            }

        except UnidentifiedImageError:
            print(f"WARNING: invalid file for scene {image_path}, will return another random batch item")
            return super().__getitem__(random.choice(list(range(len(self)))))
        return result 


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if args.use_probes:
        probes = torch.stack([example["probe_image"] for example in examples])
        probes = probes.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])
    target_dir = torch.stack([example["target_dir"] for example in examples])

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    result = {
        "conditioning_pixel_values": conditioning_pixel_values,
        "target_dir": target_dir,
        "pixel_values": pixel_values,
    }

    result["input_ids"] = input_ids

    if args.use_probes:
        result["probes"] = probes
        
    return result


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        with open(args.output_dir + "/args.txt", "w") as f:
            pprint(vars(args), stream=f)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    lpips = LPIPS(net="alex").cuda()
    lpips_vgg = LPIPS(net="vgg").cuda()
    
    vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = vae.decoder.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    
    train_dataset = MyDataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    # vae.to(accelerator.device, dtype=weight_dtype)
    lpips.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("/")[-1].split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    def run_val(val_dir, num_samples, save_grid=True):
        image_logs = log_validation(
            vae,
            args,
            accelerator,
            weight_dtype,
            global_step,
            num_samples,
            save_grid=save_grid
        )
        os.makedirs(val_dir)

        target_resolutions = [256, 512, 1024, 1536]
        
        psnr_scores = {res:0.0 for res in target_resolutions}
        lpips_scores = {res:0.0 for res in target_resolutions}

        for i, log in enumerate(image_logs):
            log["validation_image"].save(f"{val_dir}/{i:04d}_input.png")
            log["images"][-1].save(f"{val_dir}/{i:04d}_pred.png")
            #
            if "gt" in log and log["gt"] is not None:
                log["gt"].save(f"{val_dir}/{i:04d}_target.png")
                total = len(image_logs) / len(log["images"])
                for res in target_resolutions:
                    if log["images"][0].size[0] < res:
                        continue
                    for pred in log["images"]:
                        pred = pred.resize((res, res//2*3))
                        gt = log["gt"].resize((res, res//2*3))
                        psnr_scores[res] += 10 * np.log10(255**2 / np.mean((np.array(pred) - np.array(gt))**2)) / total
                        with torch.autocast("cuda", dtype=weight_dtype):
                            lpips_scores[res] += lpips(TF.to_tensor(pred).cuda() * 2 - 1, TF.to_tensor(gt).cuda() * 2 - 1).item() / total
        
        if not args.disable_aim: 
            for res in target_resolutions:
                run.track(psnr_scores[res], name=f'psnr_{res}x{res//2*3}', step=global_step)
                run.track(lpips_scores[res], name=f'lpips_{res}x{res//2*3}', step=global_step)
                with open(f"{args.output_dir}/scores_{res}.csv", "a") as f:
                    print(round(psnr_scores[res],3), round(lpips_scores[res], 3), file=f)

    if not args.eval_only:
        # Initialize a new run
        if accelerator.is_main_process:
            experiment = args.output_dir.split("/")[-2]
            run = Run(experiment=experiment)
        
        for _ in range(first_epoch, args.num_train_epochs):
            for step, batch in tqdm(enumerate(train_dataloader), initial=initial_global_step, total=args.max_train_steps, disable=not accelerator.is_local_main_process):
                if step == 1:
                    import time 
                    start = time.time()

                with accelerator.accumulate(vae):
                    vae_in = batch["pixel_values"].to(dtype=weight_dtype)
                    with torch.no_grad():
                        latents = vae.encode(vae_in).latent_dist.mean

                    vae_in = batch["pixel_values"].to(dtype=weight_dtype)

                    with torch.no_grad():
                        latents = vae.encode(vae_in).latent_dist.mean

                    cond_pixels = batch["conditioning_pixel_values"].to(dtype=weight_dtype) 
                    mask = torch.zeros_like(cond_pixels).mean(dim=1, keepdim=True) 
                    pixels = batch["pixel_values"].to(dtype=weight_dtype)
                    
                    # On higher resolution, it crashes on the backward pass if the input size is not square
                    # So train on random square crops. This also makes it easier to fit in 40g memory.
                    if cond_pixels.shape[-1] == 1536:
                        l = random.randint(0, 512)
                        r = l + 768
                        t = random.randint(0, 512)
                        b = t + 512
                        cond_pixels = cond_pixels[:, :, t:b, l:r]
                        pixels = pixels[:, :, t:b, l:r]
                        mask = mask[:, :, t:b, l:r]
                        latents = latents[:, :, t//8:b//8, l//8:r//8]
                    
                    sampleasym = vae.decode(latents, cond_pixels, mask).sample
                    loss = lpips_vgg(sampleasym, pixels).mean() + args.mse_weight * F.mse_loss(sampleasym, pixels) # note: all images scaled to [-1, 1]
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = vae.decoder.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step() 
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if accelerator.is_main_process:
                        if "SKIP_VAL" not in os.environ and global_step % args.validation_steps == 0 and not (global_step == 0 and args.skip_first_val):
                            run_val(f"{args.output_dir}/val_results/{global_step // args.validation_steps:03d}", args.num_validation_images)

                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                
                if accelerator.is_main_process:
                    if not args.disable_aim:
                        run.track(loss.detach().item(), name='loss', step=global_step)
                        run.track(loss.detach().item(), name='loss_per_image', step=global_step*args.train_batch_size)
                        run.track(loss.detach().item(), name='loss_per_pixel', step=global_step*args.train_batch_size * batch["pixel_values"].shape[-1] * batch["pixel_values"].shape[-2])
                
                if global_step >= args.max_train_steps:
                    print("time elapsed", time.time() - start)
                    break
                
        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()

        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
