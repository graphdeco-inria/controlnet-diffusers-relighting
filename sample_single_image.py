import torch
from dataclasses import dataclass
from relighting.light_directions import LEFT_DIR_ID, TOP_DIR_ID, RIGHT_DIR_ID, BACK_DIR_ID
from typing import *
import os 
import tyro 
from relighting.inference_pipeline import InferencePipeline
from relighting.match_color import match_color


@dataclass
class Conf:
    image_paths: List[str]
    
    dir_ids: List[int] = tuple([LEFT_DIR_ID, TOP_DIR_ID, RIGHT_DIR_ID, BACK_DIR_ID])
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

os.makedirs("samples", exist_ok=True)


with torch.no_grad():
    for im_i, source_path in enumerate(conf.image_paths):
        control_image = pipeline.open_control_image(source_path, conf.width, conf.height).cuda()

        images = []
        for dir_id in conf.dir_ids:
            images += pipeline(control_image, dir_id, conf.seed)
        
        for dir_id, pred_image in zip(conf.dir_ids, match_color(reference=control_image, images=images)):
            target_path_base = "samples/" + os.path.basename(source_path) 
            suffix = f"_dir_{dir_id:02d}.png"
            target_path = target_path_base.replace(".jpg", suffix).replace(".png", suffix)
            pred_image.save(target_path)
            print("Saved:", target_path)

