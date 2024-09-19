from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
import torch 
from relighting.light_directions import get_light_dir_encoding
from diffusers import AsymmetricAutoencoderKL, MarigoldDepthPipeline
from typing import *
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random 
import numpy as np 
from relighting.sh_encoding import sh_encoding


class InferencePipeline:
    def __init__(
        self, 
        ckpt_path: str,
        decoder_ckpt_path: Optional[str] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.dtype = dtype 

        self.controlnet_pipeline: StableDiffusionControlNetPipeline = _load_controlnet_pipeline(ckpt_path)

        if decoder_ckpt_path is not None:
            self.controlnet_pipeline.vae: AsymmetricAutoencoderKL = _load_decoder_pipeline(decoder_ckpt_path)

        self.depth_pipeline: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-lcm-v1-0")
        self.depth_pipeline.dtype = torch.float32
    
    @torch.no_grad()
    def open_control_image(self, source_path: str, width: int, height: int):
        color_map = TF.to_tensor(Image.open(source_path))[None][:, :3].moveaxis(1, -1).moveaxis(-1, 1)
        with torch.autocast("cuda", self.dtype):
            depth_map = torch.from_numpy(self.depth_pipeline(color_map).prediction).cpu()
        fullres_image = torch.cat([color_map, depth_map], dim=1)
        control_image = F.interpolate(fullres_image, (height, width), mode="bicubic", antialias=True)
        return control_image 

    @torch.no_grad()
    def __call__(self, control_images: torch.Tensor, light_dirs_or_ids: Union[int, torch.Tensor], seed: Optional[int]): # todo what about list of ids
        """
        control_images: shape [batch_size, 4 (rgb + depth), height, width]
        light_dirs_or_ids: either floating point light direction of shape [batch_size, 3] or a long tensor of light ids of shape [batch_size]
        """
        if isinstance(light_dirs_or_ids, int):
            light_dirs_or_ids = torch.tensor([light_dirs_or_ids] * control_images.shape[0]).cuda()

        if light_dirs_or_ids.dtype.is_floating_point:
            encodings = sh_encoding(light_dirs_or_ids).cuda()
        else:
            encodings = get_light_dir_encoding(light_dirs_or_ids).cuda()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        with torch.autocast("cuda", self.dtype):
            return self.controlnet_pipeline(
                prompt=[""] * control_images.shape[0],
                image=control_images,
                num_inference_steps=5, 
                guidance_scale=1.0,
                controlnet_kwargs=dict(timestep_cond=encodings),
            ).images
        
        
def _load_controlnet_pipeline(ckpt_path: str):
    model_id = "stabilityai/stable-diffusion-2-1"

    def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")

    text_encoder_cls = import_model_class_from_model_name_or_path(model_id, None)
    text_encoder = text_encoder_cls.from_pretrained(
        model_id, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", revision=None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    text_encoder = text_encoder_cls.from_pretrained(model_id, subfolder="text_encoder", revision=None)

    @torch.no_grad()
    def modify_layers(controlnet):
        controlnet.time_embedding.cond_proj = torch.nn.Linear(get_light_dir_encoding(torch.tensor([0])).shape[1], controlnet.time_embedding.in_channels, bias=False)

    controlnet = ControlNetModel.from_config({
        "_class_name": "ControlNetModel",
        "_diffusers_version": "0.22.0.dev0",
        "act_fn": "silu",
        "addition_embed_type": None,
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": None,
        "attention_head_dim": [
            5,
            10,
            20,
            20
        ],
        "block_out_channels": [
            320,
            640,
            1280,
            1280
        ],
        "class_embed_type": None,
        "conditioning_channels": 4,
        "conditioning_embedding_out_channels": [
            16,
            32,
            96,
            256
        ],
        "controlnet_conditioning_channel_order": "rgb",
        "cross_attention_dim": 1024,
        "down_block_types": [
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ],
        "downsample_padding": 1,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "global_pool_conditions": False,
        "in_channels": 4,
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        "num_attention_heads": None,
        "num_class_embeds": None,
        "only_cross_attention": False,
        "projection_class_embeddings_input_dim": None,
        "resnet_time_scale_shift": "default",
        "transformer_layers_per_block": 1,
        "upcast_attention": True,
        "use_linear_projection": True
    }).cuda()
    modify_layers(controlnet)
    _load_weights(controlnet, ckpt_path)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=None,
    ).to("cuda")
    pipeline.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing", rescale_betas_zero_snr=True, clip_sample=False)

    return pipeline

def _load_decoder_pipeline(ckpt_path: str):
    vae = AsymmetricAutoencoderKL.from_config({
        'in_channels': 3, 
        'out_channels': 3, 
        'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'], 
        'down_block_out_channels': [128, 256, 512, 512], 
        'layers_per_down_block': 2, 
        'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], 
        'up_block_out_channels': [192, 384, 768, 768], 
        'layers_per_up_block': 3, 
        'act_fn': 'silu', 
        'latent_channels': 4, 
        'norm_num_groups': 32, 
        'sample_size': 256, 
        'scaling_factor': 0.18215, 
        'learn_residuals': False, 
        '_use_default_values': ['learn_residuals'], 
        '_class_name': 'AsymmetricAutoencoderKL', 
        '_diffusers_version': '0.19.0.dev0', 
        '_name_or_path': 'cross-attention/asymmetric-autoencoder-kl-x-1-5'
    }).cuda()
    _load_weights(vae, ckpt_path)
    return vae 

def _load_weights(model, ckpt_path):
    from safetensors import safe_open
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device=0) as f: 
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)