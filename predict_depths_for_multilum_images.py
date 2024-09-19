"""
This script downloads the images from the multilum dataset in exr format, tone maps them, resizes them, and saves them to png.
"""

import os 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import *
from tqdm import tqdm 
from PIL import Image
import numpy as np
import sys
import multilum
import torch 
from diffusers import AsymmetricAutoencoderKL, MarigoldDepthPipeline
import torchvision.transforms.functional as TF

RESOLUTIONS = [(1536, 1024), (768, 512)]

depth_pipeline: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-lcm-v1-0", device="cuda")
depth_pipeline.dtype = torch.float32

target_path = "multilum_images/{height}x{width}/" 
for (height, width) in RESOLUTIONS:
    os.makedirs(target_path.format(height=height, width=width), exist_ok=True)

def prepare_scene(scene):
    for dir_id, pixels in enumerate(multilum.query_images(scene, mip=2, hdr=True)[0]):
        out_path = f"{target_path}/{scene.name}_dir_{dir_id}.png"
        
        def filmic(tex_color, exposure=1.0):
            x = (tex_color * exposure - 0.004).clip(0)
            return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        
        pixels = filmic(pixels)
        image = Image.fromarray((pixels.clip(0, 1) * 255).astype(np.uint8))
        if dir_id == 10:
            for (height, width) in RESOLUTIONS:       
                color_map = TF.to_tensor(image.resize((height, width), Image.LANCZOS))[None]
                with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                    depth_pred = depth_pipeline(color_map.cuda()).prediction
                depth_map = Image.fromarray((depth_pred[0, 0] * 255).astype(np.uint8))
                depth_map.save(out_path.format(height=height, width=width).replace("_dir_10.png", "_depth.png"))
        
scenes = multilum.train_scenes()
for scene in tqdm(scenes):
    prepare_scene(scene)


