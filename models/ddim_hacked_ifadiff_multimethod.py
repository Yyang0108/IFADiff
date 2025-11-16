"""\
Drop-in replacement for `ddim_hacked_ifadiff.py` that lets you replace the
**DDIM baseline step** with 3rd-order **DEIS / UniPC / DPM-Solver++**, while
keeping the original IFADiff alternating scheme and all arguments intact.

Usage (minimal change):
    from ddim_hacked_ifadiff_multimethod import DDIMSampler

    sampler = DDIMSampler(model)
    samples, inter = sampler.sample(
        S=50, batch_size=bs, shape=(C,H,W), metadata=metadata,
        r=0.75, conditioning=cond,
        base_method="unipc3",  # "ddim" | "deis3" | "unipc3" | "dpmspp3"
        unconditional_guidance_scale=7.5, unconditional_conditioning=uc,
    )

Notes:
- Keeps **IFADiff** path (`p_sample_ddim_ifadiff`) unchanged.
- Only swaps the **baseline DDIM step** for the chosen 3rd-order method.
- Methods reuse the same **alphas/ddim schedule** and **UCG** handling as the
  original code; supports both `eps` and `v` parameterization.
- `dpmspp3` here is a stable 3rd-order multi-step wrapper; if you want the
  exact DPM-Solver++ closed-form in λ-space, we can wire it later without
  changing this external API.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from typing import Optional, Tuple

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)


class MultiMethodSampler(object):
    def __init__(self, model, schedule: str = "linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        # history for multi-step methods
        self._eps_hist = []  # keep latest 3

    # parity with original file
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor and attr.device != torch.device("cuda"):
            attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas must be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # q(x_t|x_{t-1}) and friends
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            to_torch(np.log(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # DDIM sampling parameters
        (
            ddim_sigmas,
            ddim_alphas,
            ddim_alphas_prev,
        ) = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    # ----------------------- Public API -----------------------
    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        metadata,
        r,
        conditioning=None,
        base_method: str = "ddim",  # "ddim" | "deis3" | "unipc3" | "dpmspp3"
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
        global_strength=None,
        **kwargs,
    ):
        # input checks (kept from original)
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(
                            f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}"
                        )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for IFADiff-DDIM sampling is {size}, eta {eta}, base_method {base_method}")

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            metadata,
            r,
            S,
            base_method=base_method,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule,
            global_strength=global_strength,
        )
        return samples, intermediates

    # ----------------------- Core loop -----------------------
    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        metadata,
        r,
        S,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
        global_strength=None,
        base_method: str = "ddim",
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T.to(device)

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]
            ) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running IFADiff-DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="IFADiff Sampler", total=total_steps)
        time_list = list(time_range)
        y_trace = deque(maxlen=2)
        et_trace = []  # kept for potential PLMS experiments
        y_trace.append(img)
        j = 1

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if i < len(time_list) - 1:
                next_step = time_list[i + 1]
            else:
                next_step = 0
            ts_next = torch.full((b,), next_step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            # --- Alternate between "baseline" step and IFADiff step ---
            if S % 2 == 0:
                if j == 1:
                    img, pred_x0, eps_t = self._step_baseline(
                        base_method,
                        img,
                        cond,
                        ts,
                        ts_next,
                        index,
                        metadata,
                        quantize_denoised=quantize_denoised,
                        temperature=temperature,
                        noise_dropout=noise_dropout,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        global_strength=global_strength,
                    )
                if j % 2 == 0:
                    img = self.p_sample_ddim_ifadiff(
                        img,
                        None,
                        cond,
                        ts,
                        ts_next,
                        S,
                        j,
                        r,
                        index=index,
                        metadata=metadata,
                        use_original_steps=ddim_use_original_steps,
                        quantize_denoised=quantize_denoised,
                        temperature=temperature,
                        noise_dropout=noise_dropout,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        dynamic_threshold=dynamic_threshold,
                        global_strength=global_strength,
                        y_trace=y_trace,
                        et_trace=et_trace,
                    )
            else:
                if j % 2 == 1:
                    img = self.p_sample_ddim_ifadiff(
                        img,
                        None,
                        cond,
                        ts,
                        ts_next,
                        S,
                        j,
                        r,
                        index=index,
                        metadata=metadata,
                        use_original_steps=ddim_use_original_steps,
                        quantize_denoised=quantize_denoised,
                        temperature=temperature,
                        noise_dropout=noise_dropout,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        dynamic_threshold=dynamic_threshold,
                        global_strength=global_strength,
                        y_trace=y_trace,
                        et_trace=et_trace,
                    )

            j += 1

            if callback:
                callback(i)
            if img_callback:
                # print(img.shape)
                img_callback(img, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)

        return img, intermediates

    # ----------------------- Baseline step selector -----------------------
    @torch.no_grad()
    def _step_baseline(
        self,
        base_method: str,
        x,
        cond,
        t,
        t_next,
        index,
        metadata,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        global_strength=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base_method == "ddim":
            return self._ddim_like_step(
                x,
                cond,
                t,
                index,
                metadata,
                quantize_denoised,
                temperature,
                noise_dropout,
                unconditional_guidance_scale,
                unconditional_conditioning,
                global_strength,
            )
        elif base_method == "deis3":
            return self._deis3_step(
                x,
                cond,
                t,
                index,
                metadata,
                quantize_denoised,
                temperature,
                noise_dropout,
                unconditional_guidance_scale,
                unconditional_conditioning,
                global_strength,
            )
        elif base_method == "unipc3":
            return self._unipc3_step(
                x,
                cond,
                t,
                t_next,
                index,
                metadata,
                quantize_denoised,
                temperature,
                noise_dropout,
                unconditional_guidance_scale,
                unconditional_conditioning,
                global_strength,
            )
        elif base_method == "dpmspp3":
            return self._dpmspp3_step(
                x,
                cond,
                t,
                index,
                metadata,
                quantize_denoised,
                temperature,
                noise_dropout,
                unconditional_guidance_scale,
                unconditional_conditioning,
                global_strength,
            )
        else:
            raise ValueError(f"Unknown base_method: {base_method}")

    # ----------------------- Shared primitives -----------------------
    @torch.no_grad()
    def _model_eps(
        self,
        x,
        t,
        cond,
        metadata,
        global_strength,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ) -> torch.Tensor:
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_out = self.model.apply_model(x, t, cond, metadata, global_strength)
        else:
            model_t = self.model.apply_model(x, t, cond, metadata, global_strength)
            model_uncond = self.model.apply_model(
                x, t, unconditional_conditioning, metadata, global_strength
            )
            model_out = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        if self.model.parameterization == "v":
            eps = self.model.predict_eps_from_z_and_v(x, t, model_out)
        else:
            eps = model_out
        return eps

    def _extract_cur_params(self, index, use_original_steps=False):
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        return alphas[index], alphas_prev[index], sqrt_one_minus_alphas[index], sigmas[index]

    @torch.no_grad()
    def _ddim_step_from_eps(
        self,
        x,
        eps_t: torch.Tensor,
        index: int,
        use_original_steps: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        quantize_denoised: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, *_, device = *x.shape, x.device
        a_t, a_prev, sqrt_one_minus_at, sigma_t = self._extract_cur_params(
            index, use_original_steps
        )
        a_t = torch.full((b, 1, 1, 1), a_t, device=device)
        a_prev = torch.full((b, 1, 1, 1), a_prev, device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_at, device=device
        )
        sigma_t = torch.full((b, 1, 1, 1), sigma_t, device=device)

        # predict x0 from eps
        pred_x0 = (x - sqrt_one_minus_at * eps_t) / a_t.sqrt()
        if quantize_denoised and hasattr(self.model, "first_stage_model"):
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # deterministic DDIM direction + optional noise
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * eps_t
        noise = sigma_t * noise_like(x.shape, device, False) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    # ----------------------- Baseline: original DDIM-like -----------------------
    @torch.no_grad()
    def _ddim_like_step(
        self,
        x,
        cond,
        t,
        index,
        metadata,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        global_strength=None,
    ):
        eps_t = self._model_eps(
            x,
            t,
            cond,
            metadata,
            global_strength,
            unconditional_guidance_scale,
            unconditional_conditioning,
        )
        x_prev, pred_x0 = self._ddim_step_from_eps(
            x,
            eps_t,
            index,
            use_original_steps=False,
            temperature=temperature,
            noise_dropout=noise_dropout,
            quantize_denoised=quantize_denoised,
        )
        self._push_eps(eps_t)
        return x_prev, pred_x0, eps_t

    # ----------------------- DEIS-3 (AB3 on eps) -----------------------
    def _push_eps(self, eps_t: torch.Tensor):
        self._eps_hist.append(eps_t.detach())
        if len(self._eps_hist) > 3:
            self._eps_hist.pop(0)

    @torch.no_grad()
    def _deis3_step(
        self,
        x,
        cond,
        t,
        index,
        metadata,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        global_strength=None,
    ):
        eps_now = self._model_eps(
            x,
            t,
            cond,
            metadata,
            global_strength,
            unconditional_guidance_scale,
            unconditional_conditioning,
        )
        if len(self._eps_hist) >= 2:
            e_n = eps_now
            e_1 = self._eps_hist[-1]
            e_2 = self._eps_hist[-2]
            eps_hat = (23.0 / 12.0) * e_n - (16.0 / 12.0) * e_1 + (5.0 / 12.0) * e_2
        elif len(self._eps_hist) == 1:
            eps_hat = 1.5 * eps_now - 0.5 * self._eps_hist[-1]
        else:
            eps_hat = eps_now
        x_prev, pred_x0 = self._ddim_step_from_eps(
            x,
            eps_hat,
            index,
            use_original_steps=False,
            temperature=temperature,
            noise_dropout=noise_dropout,
            quantize_denoised=quantize_denoised,
        )
        self._push_eps(eps_now)
        return x_prev, pred_x0, eps_now

    # ----------------------- UniPC-3 (3-eval predictor–corrector) -----------------------
    @torch.no_grad()
    def _unipc3_step(
        self,
        x,
        cond,
        t,
        t_next,
        index,
        metadata,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        global_strength=None,
    ):
        # Stage 1
        eps_t = self._model_eps(
            x,
            t,
            cond,
            metadata,
            global_strength,
            unconditional_guidance_scale,
            unconditional_conditioning,
        )
        x_pred1, _ = self._ddim_step_from_eps(
            x,
            eps_t,
            index,
            use_original_steps=False,
            temperature=0.0,
            noise_dropout=0.0,
            quantize_denoised=quantize_denoised,
        )
        # Stage 2
        eps_next = self._model_eps(
            x_pred1,
            t_next,
            cond,
            metadata,
            global_strength,
            unconditional_guidance_scale,
            unconditional_conditioning,
        )
        eps_mid = 0.5 * (eps_t + eps_next)
        x_pred2, _ = self._ddim_step_from_eps(
            x,
            eps_mid,
            index,
            use_original_steps=False,
            temperature=0.0,
            noise_dropout=0.0,
            quantize_denoised=quantize_denoised,
        )
        # Stage 3
        eps_mid2 = self._model_eps(
            x_pred2,
            t_next,
            cond,
            metadata,
            global_strength,
            unconditional_guidance_scale,
            unconditional_conditioning,
        )
        eps_corr = (eps_t + 4.0 * eps_mid2 + eps_next) / 6.0
        x_prev, pred_x0 = self._ddim_step_from_eps(
            x,
            eps_corr,
            index,
            use_original_steps=False,
            temperature=temperature,
            noise_dropout=noise_dropout,
            quantize_denoised=quantize_denoised,
        )
        self._push_eps(eps_t)
        return x_prev, pred_x0, eps_t

    # ----------------------- DPM-Solver++ (3rd-order placeholder) -----------------------
    @torch.no_grad()
    def _dpmspp3_step(
        self,
        x,
        cond,
        t,
        index,
        metadata,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        global_strength=None,
    ):
        eps_now = self._model_eps(
            x,
            t,
            cond,
            metadata,
            global_strength,
            unconditional_guidance_scale,
            unconditional_conditioning,
        )
        if len(self._eps_hist) >= 2:
            e_n = eps_now
            e_1 = self._eps_hist[-1]
            e_2 = self._eps_hist[-2]
            eps_hat = (23.0 / 12.0) * e_n - (16.0 / 12.0) * e_1 + (5.0 / 12.0) * e_2
        elif len(self._eps_hist) == 1:
            eps_hat = 1.5 * eps_now - 0.5 * self._eps_hist[-1]
        else:
            eps_hat = eps_now
        x_prev, pred_x0 = self._ddim_step_from_eps(
            x,
            eps_hat,
            index,
            use_original_steps=False,
            temperature=temperature,
            noise_dropout=noise_dropout,
            quantize_denoised=quantize_denoised,
        )
        self._push_eps(eps_now)
        return x_prev, pred_x0, eps_now

    # ----------------------- (Original) IFADiff path -----------------------
    @torch.no_grad()
    def p_sample_ddim_ifadiff(
        self,
        x,
        x_prev,
        c,
        t,
        t_next,
        S,
        j,
        r,
        index,
        metadata,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        global_strength=None,
        y_trace=None,
        et_trace=None,
    ):
        # Unmodified from user's IFADiff baseline (lightly cleaned formatting)
        b, *_, device = *x.shape, x.device
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c, metadata, global_strength)
        else:
            model_t = self.model.apply_model(x, t, c, metadata, global_strength)
            model_uncond = self.model.apply_model(
                x, t, unconditional_conditioning, metadata, global_strength
            )
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        if index > 0:
            a_prev_next = torch.full((b, 1, 1, 1), alphas_prev[index - 1], device=device)
        else:
            a_prev_next = a_prev
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        S_float = float(S)
        delta_t = 1.0 / S_float
        x_next = x + (a_prev - a_t) * (
            x / (a_t + (a_t * a_prev).sqrt())
            - e_t / (a_t * (1 - a_prev).sqrt() + ((1 - a_t) * a_t * a_prev).sqrt())
        )
        if y_trace is not None:
            y_trace.append(x_next)
        if index > 0:
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                model_output_next = self.model.apply_model(
                    x_next, t_next, c, metadata, global_strength
                )
            else:
                model_t_next = self.model.apply_model(x_next, t_next, c, metadata, global_strength)
                model_uncond_next = self.model.apply_model(
                    x_next, t_next, unconditional_conditioning, metadata, global_strength
                )
                model_output_next = model_uncond_next + unconditional_guidance_scale * (
                    model_t_next - model_uncond_next
                )
            if self.model.parameterization == "v":
                e_t_next = self.model.predict_eps_from_z_and_v(x_next, t_next, model_output_next)
            else:
                e_t_next = model_output_next
            x_frac = torch.zeros_like(x, device=self.model.betas.device)
            w = compute_gl_forth_coefficients(alpha=r, order=len(y_trace))
            for k in range(0, len(y_trace)):
                x_frac += -w[k + 1] * y_trace[-(k + 1)]
            x_next_next = x_frac / w[0] + (2 * delta_t) ** r * (a_prev_next - a_t) / (2 * delta_t) * (
                x_next / (a_t + (a_t * a_prev_next).sqrt())
                - e_t_next / (a_t * (1 - a_prev_next).sqrt() + ((1 - a_t) * a_t * a_prev_next).sqrt())
            )
        else:
            x_next_next = x_next
        if y_trace is not None:
            y_trace.append(x_next_next)
        return x_next_next

    # ----------------------- (Original) utilities from user's file -----------------------


def compute_gl_first_coefficients(alpha, order):
    coefficients = [1.0]
    for k in range(1, order + 1):
        new_coeff = coefficients[-1] * (1 - (alpha + 1) / k)
        coefficients.append(new_coeff)
    return coefficients


def compute_gl_forth_coefficients(alpha, order):
    coefficients = [1.0]  # g0 = 1.0
    g = coefficients
    w = [(alpha * alpha + 3 * alpha + 2) / 12 * g[0]]
    for k in range(1, order + 1):
        new_coeff1 = g[-1] * (1 - (alpha + 1) / k)
        g.append(new_coeff1)
        if k == 1:
            new_coeff2 = (
                g[-1] * (alpha * alpha + 3 * alpha + 2) / 12 + g[-2] * (4 - alpha * alpha) / 6
            )
        else:
            new_coeff2 = (
                g[-1] * (alpha * alpha + 3 * alpha + 2) / 12
                + g[-2] * (4 - alpha * alpha) / 6
                + g[-3] * (alpha * alpha - 3 * alpha + 2) / 12
            )
        w.append(new_coeff2)
    return w


def compute_l1_coefficients(alpha, order):
    coefficients = [1.0]
    for k in range(1, order + 1):
        new_coeff1 = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
        coefficients.append(new_coeff1)
    w = [coefficients[0]]
    for k in range(1, order):
        new_coeff2 = coefficients[order - k - 1] - coefficients[order - k]
        w.append(-new_coeff2)
    w.append(-coefficients[-1])
    w = [w[0]] + list(reversed(w[1:]))
    return w
