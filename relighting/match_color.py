import torch 
from kornia.color.lab import rgb_to_lab, lab_to_rgb
from PIL import Image
import torchvision.transforms.functional as TF
from typing import *


def match_color(reference: Union[torch.Tensor, List[Image.Image]], images: Union[torch.Tensor, List[Image.Image]]):
    convert_from_pil_image = isinstance(images, list)
    
    if convert_from_pil_image:
        images = torch.stack([TF.to_tensor(image).to(reference.device) for image in images])
    else:
        assert images.ndim == 4, "Expected 4D tensor (batch of images)"

    a_lab = rgb_to_lab(reference[:, :3])
    b_lab = rgb_to_lab(images[:, :3])

    c_lab_L = _match_channels(b_lab[:, 0:1], a_lab[:, 0:1])
    c_lab_A = _match_channels(b_lab[:, 1:2], a_lab[:, 1:2])
    c_lab_B = _match_channels(b_lab[:, 2:3], a_lab[:, 2:3])
    c_lab = torch.cat([c_lab_L, c_lab_A, c_lab_B], dim=1)

    result = lab_to_rgb(c_lab)

    if convert_from_pil_image:
        return [TF.to_pil_image(image) for image in result]
    else:
        return result


def _match_channels(image_channels, reference_channels):
    return (image_channels - image_channels.mean()) / image_channels.std() * reference_channels.std() + reference_channels.mean()