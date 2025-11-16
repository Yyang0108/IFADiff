import sys
sys.path.append("./latentdiff")
sys.path.append("./models")
from os.path import join
import matplotlib.pyplot as plt
import utils.config as config
from models.util import create_model, load_state_dict
from models.ddim_hacked_ifadiff import IFAdiffSampler
# from models.ddim_hacked import DDIMSampler
# from models.plms import PLMSSampler
import cv2
import einops
import numpy as np
import os
from tqdm import tqdm
from scipy.io import *
import torch
from PIL import Image
import argparse
import random
import time
from pathlib import Path

# python inference_uncond_ddimode_all.py --num-samples 10 --ddim-steps 5 --save-dir save_uncond/ddim/save_uncond_ddim_5_r=10 --gpu 1 --r 10

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, default=10)
parser.add_argument('--ddim-steps', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save-dir', type=str, default='save_uncond')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--r', type=float, default=3.0)
opts = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
seed_everywhere(opts.seed)

num_samples = opts.num_samples
ddim_steps = opts.ddim_steps
save_dir = opts.save_dir
Path(save_dir).mkdir(parents=True, exist_ok=True)

H, W = 256, 256
shape = (4, H // 4, W // 4)

resolution = 512
local_conditions = [np.zeros((resolution, resolution, 3)) for _ in range(6)]
global_conditions = [np.zeros(768)]
metadata_emb = np.zeros((7))

global_maps = np.concatenate(global_conditions, axis=0)
detected_maps = np.concatenate(local_conditions, axis=2)

local_control = torch.from_numpy(detected_maps.copy()).float().cuda()
local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()

global_control = torch.from_numpy(global_maps.copy()).float().cuda().clone()
global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)
metadata_control = torch.from_numpy(metadata_emb.copy()).float().cuda().clone().squeeze()

model = create_model('configs/inference.yaml').cpu()
state_dict = load_state_dict('checkpoints/last.ckpt', location='cpu')
model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()

ddim_sampler = IFAdiffSampler(model)

total_samples = 1024
batches = total_samples // num_samples

os.makedirs(join(save_dir, 'pngs'), exist_ok=True)
os.makedirs(join(save_dir, 'mats'), exist_ok=True)
os.makedirs("reports", exist_ok=True)

global_idx = 0
cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([''] * num_samples)],
        'global_control': [global_control], "metadata": [metadata_control]}
device = model.betas.device

intermediate_dir = join(save_dir, 'intermediates')
os.makedirs(intermediate_dir, exist_ok=True)

inference_times = []
gpu_usages = []

for batch_idx in tqdm(range(batches)):
    seed_everywhere(opts.seed + global_idx)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    C, H, W = shape
    x_T = torch.randn((num_samples, C, H, W), device=device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        metadata_control,
        opts.r,
        conditioning=cond,
        verbose=False,
        x_T=x_T,
        eta=0.0,
        unconditional_guidance_scale=1,
        unconditional_conditioning=None,
        global_strength=1,
    )

    torch.cuda.synchronize()
    end_time = time.time()

    mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    elapsed = end_time - start_time
    inference_times.append(elapsed)
    gpu_usages.append(mem_used)

    print(f"🕒 Batch {batch_idx+1}/{batches}: {elapsed:.3f}s, GPU memory = {mem_used:.1f} MB")

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    # x_list = intermediates.get("x_inter", [])
    # step_list = intermediates.get("i", list(range(len(x_list))))

    # print(f"Saving {len(x_list)} intermediate steps for batch {batch_idx} ...")

    # for step_idx, (x_t, step_id) in enumerate(zip(x_list, step_list)):
    #     # torch.save(x_t.cpu(), os.path.join(intermediate_dir, f"x_t_{step_id:03d}.pt"))

    #     x_dec = model.decode_first_stage(x_t)
    #     x_dec = einops.rearrange(x_dec, 'b c h w -> b h w c').cpu().numpy()
    #     x_dec = x_dec * 0.5 + 0.5
    #     x_dec = (x_dec * 255).clip(0, 255).astype(np.uint8)

    #     for i in range(x_dec.shape[0]):  
    #         img = x_dec[i][..., [20, 12, 4]]
    #         img = rsshow(img, 0.001)
    #         img = Image.fromarray((img * 255).astype(np.uint8))
    #         img.save(os.path.join(intermediate_dir, f"sample_{i:03d}_step_{step_id:03d}.png"))


    x_samples = model.decode_first_stage(samples)
    x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c').cpu().numpy()
    x_samples = x_samples * 0.5 + 0.5
    results = [(x_samples[i] * 255).clip(0, 255).astype(np.uint8) for i in range(num_samples)]

    for i, image in enumerate(results):
        idx = global_idx + i
        savemat(f'{save_dir}/mats/hsi_{idx}.mat', {'data': image})
        image_rgb = image[..., [20, 12, 4]]
        image_rgb = rsshow(image_rgb, 0.001)
        image_rgb = Image.fromarray((image_rgb * 255).astype(np.uint8))
        image_rgb.save(f'{save_dir}/pngs/hsi_{idx}.png')

    global_idx += num_samples

avg_time = np.mean(inference_times)
avg_mem = np.mean(gpu_usages)
print(f"\n🚀 Average inference time: {avg_time:.3f} s/batch")
print(f"💾 Average GPU memory usage: {avg_mem:.1f} MB")

with open("reports/inference_stats.txt", "a") as f:
    f.write(f"{save_dir}: steps={ddim_steps}, r={opts.r}, samples={num_samples}\n")
    f.write(f"  Avg inference time: {avg_time:.3f} s\n")
    f.write(f"  Avg GPU memory: {avg_mem:.1f} MB\n\n")

