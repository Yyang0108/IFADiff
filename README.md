# Usage Guide

This repository contains the official implementation of **IFADiff: Training-Free Hyperspectral Image Generation via Integer–Fractional Alternating Diffusion Sampling**.

This project is based on the **[HSIGene](https://github.com/LiPang/HSIGene)** framework and provides the IFADiff inference workflow.  

This document introduces the complete procedure, including environment setup, file replacement, and running unconditional and conditional inference scripts.

## Project Structure

```
models/                     # IFADiff inference scripts (source files)
│── inference_cond_ifadiff.py
│── inference_uncond_ifadiff.py
│── requirements.txt
```

## 1. Environment Setup

Install additional dependencies for IFADiff:

```bash
pip install -r models/requirements.txt
```

## 2. Replace IFADiff Inference Scripts

Copy the following files into `HSIGene/models/` and overwrite existing ones:

```
inference_cond_ifadiff.py
inference_uncond_ifadiff.py
```

## 3. Running IFADiff Inference

### Unconditional Inference

```bash
python inference_uncond_ifadiff.py --num-samples 10 --ddim-steps 10 --save-dir save_uncond --gpu 0 --r 1.01
```

### Conditional Inference

```bash
python inference_cond_ifadiff.py --condition-dir data_prepare/conditions --ddim-steps 10 --save-dir save_cond --gpu 0 --r 1.01 --conditions hed
```

## Acknowledgement
We sincerely thank the authors of HSIGene for their excellent work.
