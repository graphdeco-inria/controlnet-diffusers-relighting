import torch

from PIL import Image
import datetime
from dataclasses import dataclass
import math 
from torchvision.utils import save_image 
from typing import *
import os 
import torch.nn.functional as F 
import torchvision.transforms.functional as TF 
import numpy as np 
import tyro 
from relighting.inference_pipeline import InferencePipeline
from relighting.gray_ball_renderer import GrayBallRenderer
from relighting.match_color import match_color


@dataclass
class Conf:
    image_paths: List[str] = tuple(["exemplars/kettle.png"])
    seed: int = 777
    
    num_frames: int = 270
    draw_gray_ball: bool = True

    width: int = 1536
    height: int = 1024

    dtype: Literal["bf16", "fp16", "fp32"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    
    theta_range: Tuple[float] = (0.4, 1.35)
    phi_range: Tuple[float] = (-2.3, -0.3)


conf = tyro.cli(tyro.conf.FlagConversionOff[Conf]) 
pipeline = InferencePipeline(
    f"weights/controlnet_{conf.width}x{conf.height}.safetensors",
    f"weights/decoder_{conf.width}x{conf.height}.safetensors",
    dtype=dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[conf.dtype]
)

timestamp = datetime.datetime.now().isoformat(timespec="seconds")
os.makedirs("sweep_videos/", exist_ok=True)


with torch.no_grad():
    for im_i, source_path in enumerate(conf.image_paths):
        source_ext = "." + source_path.split(".")[-1]

        sample_dir = "sweep_samples/" + os.path.basename(source_path).replace(source_ext, f'_{timestamp}.mp4') + "/"
        os.makedirs(sample_dir, exist_ok=True)

        control_image = pipeline.open_control_image(source_path, conf.width, conf.height).cuda()

        theta = math.pi / 2
        for i in range(conf.num_frames):
            if i < conf.num_frames // 2:
                # left-right
                t = (i % (conf.num_frames//2)) / (conf.num_frames//2 - 1)
                w = math.sin(math.pi * 2 * t) / 2 + 0.5
                phi = -math.pi * (1.0 - w) + 0 * w
                theta = math.pi/2
            else:
                # circles
                t = (i % (conf.num_frames//2)) / (conf.num_frames//2 - 1)
                t = (0.25 + t) % 1.0
                w = math.cos(math.pi * 2 * (1.0 - t)) / 2 + 0.5
                phi = 0 * (1.0 - w) + -math.pi * w
                w = -math.cos(math.pi * 2 * t*2) / 2 + 0.5
                theta = 0 * (1.0 - w) + math.pi/2 * w

            theta_init_range = (0.0, math.pi/2)
            theta_scale = (conf.theta_range[1] - conf.theta_range[0]) / (theta_init_range[1] - theta_init_range[0])
            theta_shift = conf.theta_range[0]
            theta = theta_scale * (theta - theta_init_range[0]) + theta_shift

            phi_init_range = (-math.pi, 0)
            phi_scale = (conf.phi_range[1] - conf.phi_range[0]) / (phi_init_range[1] - phi_init_range[0])
            phi_shift = conf.phi_range[0]
            phi = phi_scale * (phi - phi_init_range[0]) + phi_shift

            light_dir = torch.tensor([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)], device=control_image.device)

            pred = pipeline(control_image, light_dir[None], conf.seed)[0]

            if conf.draw_gray_ball:
                x_coord = light_dir[0].item()
                y_coord = light_dir[1].item()
                z_coord = light_dir[2].item()
                gray_ball_renderer = GrayBallRenderer(150)
                ball_dir = torch.tensor([x_coord, z_coord, -y_coord], device=control_image.device)
                pred = TF.to_pil_image(gray_ball_renderer.render_onto(TF.to_tensor(pred)[None].cpu(), ball_dir.cpu())[0].clamp(0, 1))
            
            target_path_base = sample_dir + os.path.basename(source_path)
            target_path = target_path_base.replace(source_ext, f"_{i:04d}.png")

            pred = match_color(references=control_image, images=[pred])[0]
            pred.save(target_path)

        input_path = target_path_base.replace('.png', f'_%04d.png')
        filename = os.path.basename(source_path).replace('.png', f'_{timestamp}.mp4')
        os.system(f"ffmpeg -framerate 60 -i {input_path} -pix_fmt yuv420p -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -y sweep_videos/{filename}")