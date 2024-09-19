
import math, torch

metadata = [
  {
   "direction_id": 0,
   "brightness_normalization": 0.056951658241450806,
   "phi": -2.124371,
   "theta": 1.570796
  },
  {
   "direction_id": 1,
   "brightness_normalization": 0.11982915960252286,
   "phi": -1.017222,
   "theta": 1.570796
  },
  {
   "direction_id": 2,
   "brightness_normalization": 5.648036599159245,
   "phi": 2.124371,
   "theta": 1.570796
  },
  {
   "direction_id": 3,
   "brightness_normalization": 0.19548214599490193,
   "phi": 1.017222,
   "theta": 1.570796
  },
  {
   "direction_id": 4,
   "brightness_normalization": 0.1163266882300377,
   "phi": -1.570796,
   "theta": 0.553574
  },
  {
   "direction_id": 5,
   "brightness_normalization": 0.21102009490132334,
   "phi": 1.570796,
   "theta": 0.553574
  },
  {
   "direction_id": 6,
   "brightness_normalization": 0.0593403669074178,
   "phi": 0.0,
   "theta": 1.017222
  },
  {
   "direction_id": 7,
   "brightness_normalization": 0.07011793740093708,
   "phi": -3.141593,
   "theta": 1.017222
  },
  {
   "direction_id": 8,
   "brightness_normalization": 0.07160339914262295,
   "phi": -1.93566,
   "theta": 1.047198
  },
  {
   "direction_id": 9,
   "brightness_normalization": 0.09316171519458297,
   "phi": -1.205932,
   "theta": 1.047198
  },
  {
   "direction_id": 10,
   "brightness_normalization": 0.08272826597094536,
   "phi": -1.570796,
   "theta": 1.570796
  },
  {
   "direction_id": 11,
   "brightness_normalization": 0.04724164921790363,
   "phi": -2.588018,
   "theta": 1.256637
  },
  {
   "direction_id": 12,
   "brightness_normalization": 0.09292297773063184,
   "phi": -2.588018,
   "theta": 0.628319
  },
  {
   "direction_id": 13,
   "brightness_normalization": 0.1603627033531666,
   "phi": 2.588018,
   "theta": 0.628319
  },
  {
   "direction_id": 14,
   "brightness_normalization": 0.11537112928926946,
   "phi": 0.0,
   "theta": 0.0
  },
  {
   "direction_id": 15,
   "brightness_normalization": 0.09658141024410727,
   "phi": 0.553574,
   "theta": 0.628319
  },
  {
   "direction_id": 16,
   "brightness_normalization": 0.10929663665592672,
   "phi": -0.553574,
   "theta": 0.628319
  },
  {
   "direction_id": 17,
   "brightness_normalization": 0.08106742911040786,
   "phi": -0.553574,
   "theta": 1.256637
  },
  {
   "direction_id": 18,
   "brightness_normalization": 0.05088545549660923,
   "phi": 0.0,
   "theta": 1.570796
  },
  {
   "direction_id": 19,
   "brightness_normalization": 0.04553791042417288,
   "phi": 0.553574,
   "theta": 1.256637
  },
  {
   "direction_id": 20,
   "brightness_normalization": 0.13744037151336672,
   "phi": 1.205932,
   "theta": 1.047198
  },
  {
   "direction_id": 21,
   "brightness_normalization": 0.3453471198678018,
   "phi": 1.93566,
   "theta": 1.047198
  },
  {
   "direction_id": 22,
   "brightness_normalization": 1.2606816172599795,
   "phi": 1.570796,
   "theta": 1.570796
  },
  {
   "direction_id": 23,
   "brightness_normalization": 0.054846464283764364,
   "phi": -3.141593,
   "theta": 1.570796
  },
  {
   "direction_id": 24,
   "brightness_normalization": 0.3484217435121537,
   "phi": 2.588018,
   "theta": 1.256637
  }
 ]
 
def get_dir(id, metadata=metadata):
  theta = metadata[id]['theta']
  phi = metadata[id]['phi']

  x = math.sin(theta) * math.cos(phi)
  y = math.sin(theta) * math.sin(phi)
  z = math.cos(theta)

  return torch.tensor([x, z, -y])

def get_dir_polar(id, metadata=metadata):
  theta = metadata[id]['theta']
  phi = metadata[id]['phi']

  return theta, phi

def cartesian_to_polar(vec):
  x = vec[0]
  y = -vec[2]
  z = vec[1] 
  phi = torch.atan2(y, x)
  theta = torch.acos(z / torch.sqrt(x**2 + y**2 + z**2))
  return dict(theta=theta.item(), phi=phi.item())

def angular_distance(angle1, angle2):
  # Normalize angles to be within [-pi, pi]
  angle1 = math.atan2(math.sin(angle1), math.cos(angle1))
  angle2 = math.atan2(math.sin(angle2), math.cos(angle2))
  
  # Compute angular distance
  angular_distance = abs(angle1 - angle2)
  
  # Ensure the distance is within [0, pi]
  if angular_distance > math.pi:
      angular_distance = 2 * math.pi - angular_distance
  
  return angular_distance

for id, entry in enumerate(metadata):
  polar = cartesian_to_polar(get_dir(id))
  assert angular_distance(polar["theta"], entry["theta"]) < 0.0001
  assert angular_distance(polar["phi"], entry["phi"]) < 0.0001

FRONTAL_DIRS = [2, 3, 19, 20, 21, 22, 24]
BACKWARD_DIRS = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23]

# ---------------------------------------------------

import cv2 
import numpy as np 
import torch 
import os, shutil
from torchvision.utils import save_image, make_grid
import math 
from relighting.light_directions import BACKWARD_DIR_IDS

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json 

# generate a mask of a circle in the centre of the 256x256 image
mask = cv2.circle(
    np.zeros((256, 256, 3), dtype=np.uint8),
    (128, 128),
    128-5,
    (255, 255, 255),
    -1
)
# cv2.imwrite("mask.png", mask)

DIRS = BACKWARD_DIR_IDS

# Open a diffuse light probe 
img_probe = torch.stack([torch.from_numpy(cv2.imread(f"multilum_average_probes/average_gray_{i:02d}.png", 1)) for i in DIRS])
chrome_probe = torch.stack([torch.from_numpy(cv2.imread(f"multilum_average_probes/average_chrome_{i:02d}.png", 1)) for i in DIRS])
img_probe_linspace = torch.stack([torch.from_numpy(cv2.imread(f"multilum_average_probes/average_gray_{i:02d}.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)) for i in DIRS])
chrome_probe_linspace = torch.stack([torch.from_numpy(cv2.imread(f"multilum_average_probes/average_chrome_{i:02d}.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)) for i in DIRS])

# draw the nromals of the sphere
# arange a grid of x y coordinates
x = np.linspace(-1, 1, 256)
y = np.linspace(-1, 1, 256)
x, y = np.meshgrid(x, y)
normals = np.stack([x, -y, np.sqrt(np.abs(1 - x**2 - y**2))], axis=-1)
normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
normals: torch.Tensor = torch.from_numpy(normals).permute(2, 0, 1).float().to("cuda")

light_dir = torch.nn.Parameter(torch.stack([get_dir(id) for id in DIRS]).to("cuda"))

mask = torch.from_numpy(mask/255).permute(2, 0, 1).float().to("cuda").mean(dim=0).nan_to_num(0.0)
normals = normals * mask 
albedo = torch.nn.Parameter(torch.ones(len(DIRS))[:, None, None, None].to("cuda")*0.3)
fresnel = torch.nn.Parameter(torch.ones(len(DIRS))[:, None, None, None].to("cuda")*0.1)
ambient = torch.nn.Parameter(torch.ones(len(DIRS))[:, None, None, None].to("cuda")*0.1)
spec = torch.nn.Parameter(torch.zeros(len(DIRS))[:, None, None, None].to("cuda") + 0.5)
roughness = torch.nn.Parameter(torch.zeros(len(DIRS))[:, None, None, None].to("cuda") + 0.3)
spec_power = torch.nn.Parameter(torch.ones(len(DIRS))[:, None, None, None].to("cuda") * 2)

target = (img_probe/255).moveaxis(-1, 1).float().to("cuda").mean(dim=1, keepdim=True).nan_to_num(0.0)
target_linspace = (img_probe_linspace).moveaxis(-1, 1).float().to("cuda").mean(dim=1, keepdim=True).nan_to_num(0.0)
chrome_linspace = (chrome_probe_linspace).moveaxis(-1, 1).float().to("cuda").mean(dim=1, keepdim=True).nan_to_num(0.0)

# ----------------------------------------

def dotpos(a, b, dim=-1):
    return (a * b).sum(dim=dim, keepdim=True).clamp(0.0)

def render():
    view_dir = torch.tensor([0, 0, 1]).to("cuda")[None]
    diffuse = torch.sum(normals[None] * light_dir[:, :, None, None], dim=1, keepdim=True)
    halfway_vec = (light_dir + view_dir).to("cuda")
    halfway_vec = halfway_vec / (torch.norm(halfway_vec, dim=-1, keepdim=True) + 1e-8)
    specular = spec * dotpos(normals[None], halfway_vec[:, :, None, None], dim=1)**torch.exp(spec_power)
    F_schlick = 0.04 + (1.0 - 0.04) * (1.0 - dotpos(normals[None], halfway_vec[:, :, None, None], dim=1))**5
    return (albedo * (diffuse + ambient).clamp(0) + specular + fresnel * F_schlick) * mask 

adam = torch.optim.Adam([light_dir, albedo, ambient, spec, spec_power, fresnel], lr=1e-2)

shutil.rmtree("tmp", ignore_errors=True)
os.makedirs("tmp", exist_ok=True)

NUM_STEPS = 1000
for k in range(NUM_STEPS):
    pred = render()
    loss = (pred - target_linspace).abs().mean(dim=[1,2,3]).sum()

    adam.zero_grad()
    loss.backward()
    adam.step()
    
    with torch.no_grad():
        light_dir.data /= (torch.norm(light_dir.data, dim=-1, keepdim=True) + 1e-16)
    
    print(f"Loss: {loss.item()}")

    if k % 250 == 0 or k == NUM_STEPS - 1:
        def lighting_direction_to_sphere_normal(light_dir):
            view_dir = torch.tensor([0, 0, 1]).to("cuda")
            normal = (light_dir + view_dir) / (torch.linalg.norm(light_dir + view_dir) + 1e-16)
            return normal
        
        _pred = pred.clone().repeat(1, 3, 1, 1)
        _chrome_linspace = chrome_linspace.clone().repeat(1, 3, 1, 1)
        _target_linspace = target_linspace.clone().repeat(1, 3, 1, 1)
        for l, dir in enumerate(light_dir):
            normal = lighting_direction_to_sphere_normal(dir)
            nx = normal[0].item()
            ny = normal[1].item()
            i = int((-ny + 1) * 128)
            j = int((nx + 1) * 128)
            _pred[l, 0, i-3:i+3, j-3:j+3] = 3.0
            _pred[l, 1:3, i-3:i+3, j-3:j+3] = 0.0
            _chrome_linspace[l, 0, i-3:i+3, j-3:j+3] = 3.0
            _chrome_linspace[l, 1:3, i-3:i+3, j-3:j+3] = 0.0
            _target_linspace[l, 0, i-3:i+3, j-3:j+3] = 3.0
            _target_linspace[l, 1:3, i-3:i+3, j-3:j+3] = 0.0

            normal2 = lighting_direction_to_sphere_normal(get_dir(DIRS[l]).to("cuda"))
            nx = normal2[0].item()
            ny = normal2[1].item()
            i = int((-ny + 1) * 128)
            j = int((nx + 1) * 128)
            _pred[l, 0:3, i-3:i+3, j-3:j+3] = 0.0
            _pred[l, 2, i-3:i+3, j-3:j+3] = 3.0
            _chrome_linspace[l, 0:3, i-3:i+3, j-3:j+3] = 0.0
            _chrome_linspace[l, 2, i-3:i+3, j-3:j+3] = 3.0
            _target_linspace[l, 0:3, i-3:i+3, j-3:j+3] = 0.0
            _target_linspace[l, 2, i-3:i+3, j-3:j+3] = 3.0
        
        for m, (a, b, c) in enumerate(zip(_pred.split(8), _target_linspace.split(8), _chrome_linspace.split(8))):
            save_image(torch.cat([a, b, c/4]), f"tmp/result_{k:03d}_{m:02d}.png", nrow=min(len(a), 8))

new_metadata = []
i = 0 
for k in range(25):
    if k in DIRS:
        new_metadata.append({"direction_id": k, **cartesian_to_polar(light_dir[i])})
        i += 1
    else:
        new_metadata.append({}) # this is intentional, provide dummy values so the indicies in the list match the ids

print(json.dumps(new_metadata, indent=2))
  
if False:
  json.dump(new_metadata, open("light_directions.json", "w"))

if False:
  for m, _ in enumerate(_pred.split(8)):
    os.system(f"ffmpeg -y -i tmp/result_%03d_{m:02d}.png -pix_fmt yuv420p tmp/vid_light_direction_optim.mp4")
