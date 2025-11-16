#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, random, argparse
from os.path import join

sys.path.extend(["./latentdiff", "./models"])

import cv2, einops, numpy as np, torch
from PIL import Image
from scipy.io import savemat
from tqdm import tqdm

import utils.config as config
from models.util import create_model, load_state_dict
# from models.ddim_hacked_nesterov import DDIMSampler
from models.ddim_hacked_nesterovode import DDIMSampler
# from models.ddim_hacked_multimethod import MultiMethodSampler

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I = np.clip(I, low, high)
    return (I - low) / (high - low)

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples',     type=int, default=5)
parser.add_argument('--ddim-steps',      type=int, default=50)
parser.add_argument('--gpu',            type=str, default='7')
parser.add_argument('--save-dir',       type=str, default='save_uncond')
parser.add_argument('--seed',           type=int, default=1234)

parser.add_argument('--global-strength', type=float, default=1.0)
parser.add_argument('--conditions',      type=str, default='hed', help='hed, segmentation, sketch, mlsd, content, text')
parser.add_argument('--prompt',          type=str, default='',
                    choices=['Farmland', 'Architecture', 'City Building', 'Wasteland'])
parser.add_argument('--condition-dir',   type=str, default='data_prepare/conditions')
parser.add_argument('--method', type=str, default="ddim", help='sampling method: ddim | deis3 | unipc3 | dpmspp3')

parser.add_argument('--fns',             type=str, default='')
parser.add_argument('--r', type=float, default=3.0)
opts = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
seed_everywhere(opts.seed)


cond2num = {'hed':1,'segmentation':3,'sketch':5,'mlsd':6,'content':7,'text':0}
num2cond = {v:k for k,v in cond2num.items()}
cond_names = opts.conditions.strip().split()
cond_nums  = [cond2num[c] for c in cond_names]

H, W, RES = 256, 256, 512


print('Loading model ...')
model = create_model('configs/inference.yaml').cpu()
state_dict = load_state_dict('checkpoints/last.ckpt', location='cpu')
model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()


sampler = DDIMSampler(model)
# sampler = MultiMethodSampler(model)

if config.save_memory:
    model.low_vram_shift(is_diffusing=False)


all_fns = [d for d in os.listdir(opts.condition_dir)
           if os.path.isdir(join(opts.condition_dir, d))]
if opts.fns and opts.fns in all_fns:

    all_fns = [opts.fns]

print(f'🗂  Found {len(all_fns)} folders in {opts.condition_dir}: {all_fns}')

device = model.betas.device

for fns in all_fns:
    seed_everywhere(opts.seed)

    print(f'\nProcessing {fns} ...')

    local_conditions, global_conditions = [], []

    for i in range(1, 7):  # hed / seg / sketch / mlsd / ...
        if i in cond_nums:
            fn = join(opts.condition_dir, fns, f'{num2cond[i]}.png')
            if not os.path.exists(fn):
                print(f'{fn} not found, using zeros')
                cond = np.zeros((RES, RES, 3))
            else:
                cond = cv2.imread(fn)
                cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
                cond = cv2.resize(cond.astype(np.float32)/255., (RES, RES))
        else:
            cond = np.zeros((RES, RES, 3))
        local_conditions.append(cond)


    for i in range(7, 8):
        if i in cond_nums:
            from annotator.content import ContentDetector
            content_model = ContentDetector('data_prepare/annotator/ckpts')
            fn = join(opts.condition_dir, fns, f'{num2cond[i]}.png')
            img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
            img = (rsshow(img, 0) * 255).astype(np.uint8)
            cond = content_model(img)
        else:
            cond = np.zeros(768)
        global_conditions.append(cond)


    detected_maps = np.concatenate(local_conditions, axis=2)   # (H,W,C*?)
    local_ctrl = torch.from_numpy(detected_maps).float().cuda()
    local_ctrl = einops.rearrange(local_ctrl, 'h w c -> 1 c h w')
    local_ctrl = local_ctrl.repeat(opts.num_samples, 1, 1, 1)

    global_maps = np.concatenate(global_conditions, axis=0)
    global_ctrl = torch.from_numpy(global_maps).float().cuda()
    global_ctrl = global_ctrl.repeat(opts.num_samples, 1)

    metadata_emb = torch.zeros(7, device=global_ctrl.device)

    cond = {
        'local_control':  [local_ctrl],
        'global_control': [global_ctrl],
        'metadata':       [metadata_emb],
        'c_crossattn':    [model.get_learned_conditioning([''] * opts.num_samples)]
    }

    # ---------- 采样 ----------
    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    shape = (4, H // 4, W // 4)
    C = 4
    x_T = torch.randn((opts.num_samples, C, H, W), device=device)
    samples, _ = sampler.sample(
        opts.ddim_steps, opts.num_samples, shape, metadata_emb, opts.r,
        conditioning=cond, global_strength=opts.global_strength,
        eta=0.0, unconditional_guidance_scale=1, unconditional_conditioning=None,
        verbose=False
    )
    # samples, _ = sampler.sample(
    #     S=opts.ddim_steps,
    #     batch_size=opts.num_samples,
    #     shape=shape,
    #     metadata=metadata_emb,
    #     r=opts.r,
    #     conditioning=cond,
    #     method=opts.method,                  # "unipc3" | "deis3" | "dpmspp3" | "ddim"
    #     verbose=False,
    #     x_T=x_T,
    #     eta=0.0,
    #     unconditional_guidance_scale=1.0,
    #     unconditional_conditioning=None,
    #     global_strength=opts.global_strength,
    #     img_callback=None,
    #     # img_callback=on_step_callback,
    # )

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)


    xs = model.decode_first_stage(samples)
    xs = einops.rearrange(xs, 'b c h w -> b h w c').cpu().numpy()
    xs = (xs * 0.5 + 0.5) * 255
    xs = xs.clip(0, 255).astype(np.uint8)

    save_root = join(opts.save_dir, fns, '_'.join(cond_names))
    os.makedirs(join(save_root, 'pngs'),  exist_ok=True)
    os.makedirs(join(save_root, 'mats'),  exist_ok=True)
    os.makedirs(join(save_root, 'conds'), exist_ok=True)

    for idx, im in enumerate(xs):
        savemat(join(save_root, 'mats', f'f{idx}.mat'), {'data': im})
        rgb = rsshow(im[..., [20, 12, 4]])
        Image.fromarray((rgb * 255).astype(np.uint8))\
             .save(join(save_root, 'pngs', f'f{idx}.png'))


    for ci in range(0, local_ctrl.shape[1], 3):
        cond_im = local_ctrl[0, ci:ci+3].cpu().numpy().transpose(1,2,0)
        Image.fromarray((cond_im*255).astype(np.uint8))\
             .save(join(save_root, 'conds', f'cond{ci//3}.png'))

    print(f'Done: {fns}  to  {save_root}')

print('\n All folders finished.')
