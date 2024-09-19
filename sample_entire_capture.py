import torch
from PIL import Image
from dataclasses import dataclass
from relighting.light_directions import BACKWARD_DIR_IDS
import glob 
from torchvision.utils import save_image 
from typing import *
import os 
import torch.utils.data
import numpy as np 
import tyro 
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
from itertools import islice
from relighting.inference_pipeline import InferencePipeline
from relighting.match_color import * 


@dataclass
class Conf:
    capture_paths: List[str] = tuple(["colmap/real/garagewall"])
    splits: List[Literal["train", "test"]] = tuple(["train"])
    
    dir_ids: List[int] = BACKWARD_DIR_IDS
    seed: int = 777
    
    width: int = 1536
    height: int = 1024
    
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    

conf = tyro.cli(tyro.conf.FlagConversionOff[Conf]) 
pipeline = InferencePipeline(
    f"weights/controlnet_{conf.width}x{conf.height}.safetensors",
    f"weights/decoder_{conf.width}x{conf.height}.safetensors",
    dtype=dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[conf.dtype]
)

torch.no_grad().__enter__()

for capture_path in conf.capture_paths:
    for split in conf.splits:
        all_source_paths = sorted(glob.glob(f"{capture_path}/{split}/images/*.png"))
        assert len(all_source_paths) > 0
        os.makedirs(f"{capture_path}/{split}/relit_images/", exist_ok=True)
        
        for source_path in all_source_paths:
            control_image = None

            target_paths = []
            unmatched_images = []

            for dir_id in conf.dir_ids:
                print("Relighting", split, os.path.basename(source_path), "to direction", "#" + str(dir_id))
                target_path = source_path.replace("/images/", f"/relit_images/").replace(".png", f"_dir_{dir_id:02d}.png")
                assert target_path != source_path
                if not os.path.exists(target_path):
                    if control_image is None:
                        control_image = pipeline.open_control_image(source_path, conf.width, conf.height)
                    pred = pipeline(control_image, dir_id, conf.seed)[0]
                    target_paths.append(target_path)
                    unmatched_images.append(pred)

            for target_path, matched_image in zip(target_paths, match_color(reference=control_image, images=unmatched_images)):
                matched_image.save(target_path)