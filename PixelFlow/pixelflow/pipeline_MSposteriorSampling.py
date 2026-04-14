from einops import rearrange
import math
from typing import List, Optional, Union
import time
import os
import torch
import torch.nn.functional as F
import warnings

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_2d_rotary_pos_embed
import copy



class MSPS:
    def __init__(
        self,
        y,
        operator,
        scheduler,
        transformer,
        text_encoder=None,
        tokenizer=None,
        max_token_length=512,
        langevin_num_steps = 30,
        langevin_step_base = 1e-4,
        langevin_step_min_ratio = 1e-2,
        langevin_step_p = 1,
        sigma_n = 0.05,
        langevin_proj = False,
        lagevin_proj = None,
        ds_type = "bilinear",
        du_type = "nearest",
        align_corners = False,
        split_prior_smooth_weight = 0.0,
        split_prior_var_eps = 1e-5,
        langevin_grad_clip = None,
        langevin_x_clamp = None,
    ):
        super().__init__()

        def _resolve_numeric(value, default, name, cast=float):
            if value is Ellipsis or value is None:
                warnings.warn(
                    f"{name} is {value!r}; fallback to default {default}.",
                    UserWarning,
                )
                value = default
            try:
                return cast(value)
            except Exception as exc:
                raise TypeError(f"{name} must be numeric, got {type(value)} ({value!r})") from exc
        def _resolve_optional_float(value, name):
            if value is None:
                return None
            if value is Ellipsis:
                warnings.warn(f"{name} is Ellipsis; disabling it.", UserWarning)
                return None
            try:
                return float(value)
            except Exception as exc:
                raise TypeError(f"{name} must be numeric or None, got {type(value)} ({value!r})") from exc

        self.y = y
        self.operator = operator
        self.langevin_step_base = _resolve_numeric(
            langevin_step_base, 1e-4, "langevin_step_base", float
        )
        self.langevin_step_p = _resolve_numeric(
            langevin_step_p, 1.0, "langevin_step_p", float
        )
        self.langevin_step_min_ratio = _resolve_numeric(
            langevin_step_min_ratio, 1e-2, "langevin_step_min_ratio", float
        )
        self.langevin_num_steps = _resolve_numeric(
            langevin_num_steps, 30, "langevin_num_steps", int
        )
        self.sigma_n = _resolve_numeric(sigma_n, 0.05, "sigma_n", float)
        if self.langevin_step_base <= 0:
            raise ValueError(f"langevin_step_base must be > 0, got {self.langevin_step_base}")
        if self.langevin_step_p <= 0:
            raise ValueError(f"langevin_step_p must be > 0, got {self.langevin_step_p}")
        if self.langevin_step_min_ratio <= 0:
            raise ValueError(
                f"langevin_step_min_ratio must be > 0, got {self.langevin_step_min_ratio}"
            )
        if self.langevin_num_steps <= 0:
            raise ValueError(f"langevin_num_steps must be > 0, got {self.langevin_num_steps}")
        if lagevin_proj is not None and bool(lagevin_proj) != bool(langevin_proj):
            raise ValueError("Conflicting values for 'langevin_proj' and legacy alias 'lagevin_proj'.")
        self.proj = bool(langevin_proj if lagevin_proj is None else lagevin_proj)
        self.ds_type = ds_type
        self.du_type = du_type
        self.align_corners = bool(align_corners)
        self.split_prior_smooth_weight = _resolve_numeric(
            split_prior_smooth_weight, 0.0, "split_prior_smooth_weight", float
        )
        self.split_prior_var_eps = _resolve_numeric(
            split_prior_var_eps, 1e-5, "split_prior_var_eps", float
        )
        if self.split_prior_smooth_weight < 0:
            raise ValueError(
                f"split_prior_smooth_weight must be >= 0, got {self.split_prior_smooth_weight}"
            )
        if self.split_prior_var_eps <= 0:
            raise ValueError(f"split_prior_var_eps must be > 0, got {self.split_prior_var_eps}")
        self.langevin_grad_clip = _resolve_optional_float(langevin_grad_clip, "langevin_grad_clip")
        self.langevin_x_clamp = _resolve_optional_float(langevin_x_clamp, "langevin_x_clamp")
        if self.langevin_grad_clip is not None and self.langevin_grad_clip <= 0:
            raise ValueError(f"langevin_grad_clip must be > 0, got {self.langevin_grad_clip}")
        if self.langevin_x_clamp is not None and self.langevin_x_clamp <= 0:
            raise ValueError(f"langevin_x_clamp must be > 0, got {self.langevin_x_clamp}")
        self.class_cond = text_encoder is None or tokenizer is None
        self.scheduler = scheduler
        self.transformer = transformer
        self.patch_size = transformer.patch_size
        self.head_dim = transformer.attention_head_dim
        self.num_stages = scheduler.num_stages

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    @torch.autocast("cuda", enabled=False)
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = "",
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        use_attention_mask: bool = False,
        max_length: int = 512,
    ):
        # Determine the batch size and normalize prompt input to a list
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Process prompt embeddings if not provided
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=prompt_attention_mask if use_attention_mask else None
            )[0]

        # Determine dtype from available encoder
        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        # Move prompt embeddings to desired dtype and device
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)

        # Handle classifier-free guidance for negative prompts
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # Normalize negative prompt to list and validate length
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                if len(negative_prompt) != batch_size:
                    raise ValueError(f"The negative prompt list must have the same length as the prompt list, but got {len(negative_prompt)} and {batch_size}")
                uncond_tokens = negative_prompt
            else:
                raise ValueError(f"Negative prompt must be a string or a list of strings, but got {type(negative_prompt)}")

            # Tokenize and encode negative prompts
            uncond_inputs = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_input_ids = uncond_inputs.input_ids.to(device)
            negative_prompt_attention_mask = uncond_inputs.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(
                negative_input_ids,
                attention_mask=negative_prompt_attention_mask if use_attention_mask else None
            )[0]

        if do_classifier_free_guidance:
            # Duplicate negative prompt embeddings and attention mask for each generation
            seq_len_neg = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        if do_classifier_free_guidance:
            # Concatenate negative and positive embeddings and their masks
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        return prompt_embeds, prompt_attention_mask

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.scheduler.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4))
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c h w) (p q) -> b c (h p) (w q)',b=bs,c=ch,h=height//2,w=width//2,p=2,q=2)
        return noise
    def __call__(self, z):
        target_h, target_w = self.target_hw
        h, w = z.shape[-2:]

        out = z
        while h < target_h or w < target_w:
            next_h = min(h * 2, target_h)
            next_w = min(w * 2, target_w)

            out = F.interpolate(
                out,
                size=(next_h, next_w),
                mode="bilinear",
                align_corners=self.align_corners
            )
            h, w = next_h, next_w

        if (h, w) != (target_h, target_w):
            out = F.interpolate(
                out,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=self.align_corners
            )
        return out
    def Up_k(self, x, y):
        target_hw = y.shape[-2:]
        target_h, target_w = target_hw
        if x.shape[-2:] == target_hw:
            return x
        else:
            h, w = x.shape[-2:]

            out = x
            while h < target_h or w < target_w:
                next_h = min(h * 2, target_h)
                next_w = min(w * 2, target_w)

                out = F.interpolate(
                    out,
                    size=(next_h, next_w),
                    mode="bilinear",
                    align_corners=self.align_corners
                )
                h, w = next_h, next_w

            if (h, w) != (target_h, target_w):
                out = F.interpolate(
                    out,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=self.align_corners
                )
            return out

    def DownUp_operation(self, z, scale_factor=2):
        B, C, H, W = z.shape
        h_small = H // scale_factor
        w_small = W // scale_factor

        z_down = F.interpolate(
            z, size=(h_small, w_small),
            mode=self.ds_type,
            align_corners=self.align_corners if self.ds_type in ['bilinear', 'bicubic', 'trilinear'] else None
        )
        z_up = F.interpolate(
            z_down, size=(H, W),
            mode=self.du_type,
            align_corners=self.align_corners if self.du_type in ['bilinear', 'bicubic', 'trilinear'] else None
        )
        return z_up
    def objective_split_prior(self, x0hat, stage_t_end, stage_t_start, stage_x_end, stage_x_start, y, sigma_n, o_M):
        """
        L(z, eta*(z))
        """
        # Match data end
        stage_x_end = stage_x_end.requires_grad_(False)
        stage_x_start = stage_x_start.requires_grad_(False)
        var_eps = self.split_prior_var_eps
        denom_end = max(float((1 - stage_t_end) ** 2), var_eps)
        denom_start = max(float((1 - stage_t_start) ** 2), var_eps)
        l1 = (1 / (2 * denom_end) * (stage_x_end - stage_t_end * x0hat) ** 2).mean()
        l2 = (1 / (2 * denom_start) * (stage_x_start - stage_t_start * self.DownUp_operation(x0hat)) ** 2).mean()
        x_up_proj = (1-o_M)*self.operator(self.Up_k(x0hat, y)) + o_M*y
        l3 = (1/(2*sigma_n**2)*(x_up_proj- y)**2).mean()

        # Optional smooth regularization on x0hat.
        if self.split_prior_smooth_weight > 0:
            dx = x0hat[:, :, 1:, :] - x0hat[:, :, :-1, :]
            dy = x0hat[:, :, :, 1:] - x0hat[:, :, :, :-1]
            l_smooth = self.split_prior_smooth_weight * (dx.pow(2).mean() + dy.pow(2).mean())
        else:
            l_smooth = x0hat.new_tensor(0.0)

        return l1 + l2 + l3 + l_smooth
    @torch.no_grad()
    def langevin_update_split_prior(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        stage_t_end, 
        stage_t_start, 
        stage_x_end: torch.Tensor,
        stage_x_start: torch.Tensor,
        sigma_n: float,                   
        step_size: float, 
        o_M,
        device,
    ):
        x = x.to(device)
        y = y.to(device)
        """
        One step overdamped Langevin for p(x) ∝ exp(-L(x)).
        L(x) = 1/(2σ_n^2)||Hx-y||^2 + 1/(2ν_t^2)||x-x1hat||^2
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            loss = self.objective_split_prior(x, stage_t_end, stage_t_start, stage_x_end, stage_x_start, y, sigma_n, o_M)
            grad = torch.autograd.grad(loss, x)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        if self.langevin_grad_clip is not None:
            grad_norm = grad.norm().clamp_min(1e-12)
            clip_val = grad.new_tensor(self.langevin_grad_clip)
            grad = grad * torch.clamp(clip_val / grad_norm, max=1.0)
        x_new = x - step_size*grad + math.sqrt(2.0 * step_size)*torch.randn_like(x)
        x_new = torch.nan_to_num(x_new, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.langevin_x_clamp is not None:
            x_new = torch.clamp(x_new, -self.langevin_x_clamp, self.langevin_x_clamp)
        return x_new, {
                "loss": float(loss.detach().cpu()),
                "z_mean": float(x_new.mean().detach().cpu()),
                "z_std": float(x_new.std().detach().cpu()),
            }
    def langevin_sample_split_prior(
        self,
        x_init, y, stage_t_end,stage_t_start, stage_x_end, stage_x_start,
        sigma_n, 
        step_size=1e-4,
        device = "cpu",
        return_traj = False
    ):
        if self.proj:
            if hasattr(self.operator, "get_mask"):
                o_M = self.operator.get_mask(x=y).to(device)
            else:
                o_M = self.operator.mask.to(device)
        else:
            o_M = torch.zeros_like(y, device=device)
        x = x_init.clone().detach()
        for _ in range(self.langevin_num_steps):
            x,record = self.langevin_update_split_prior(x,y,stage_t_end, stage_t_start, 
                                                    stage_x_end, stage_x_start, sigma_n,  step_size, o_M, device)
            if return_traj:
                if _ % 10 == 0:
                    print(record)
            if torch.isnan(x).any():
                break
        return x.detach()

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height,
        width,
        num_inference_steps=30,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        device=None,
        shift=1.0,
        use_ode_dopri5=False,
        return_t0_traj=False,
        save_t0_images=False,
        save_t0_dir=None,
        save_t0_nrow=4,
    ):
        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * self.num_stages

        if use_ode_dopri5:
            assert self.class_cond, "ODE (dopri5) sampling is only supported for class-conditional models now"
            from pixelflow.solver_ode_wrapper import ODE
            sample_fn = ODE(t0=0, t1=1, sampler_type="dopri5", num_steps=num_inference_steps[0], atol=1e-06, rtol=0.001).sample
        else:
            # default Euler
            sample_fn = None

        self._guidance_scale = guidance_scale
        batch_size = len(prompt)
        if self.class_cond:
            prompt_embeds = torch.tensor(prompt, dtype=torch.int32).to(device)
            negative_prompt_embeds = 1000 * torch.ones_like(prompt_embeds)
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale > 1,
                "",
                prompt_embeds=None,
                negative_prompt_embeds=None,
                use_attention_mask=True,
                max_length=self.max_token_length,
            )

        init_factor = 2 ** (self.num_stages - 1)
        height, width =  height // init_factor, width // init_factor
        shape = (batch_size * num_images_per_prompt, 3, height, width)
        latents = randn_tensor(shape, device=device, dtype=torch.float32)

        for stage_idx in range(self.num_stages):
            stage_start = time.time()
            # Set the number of inference steps for the current stage
            self.scheduler.set_timesteps(num_inference_steps[stage_idx], stage_idx, device=device, shift=shift)
            Timesteps = self.scheduler.Timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                original_start_t = self.scheduler.original_start_t[stage_idx]
                gamma = self.scheduler.gamma
                alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
                beta = alpha * (1 - original_start_t) / math.sqrt(- gamma)

                # bs, ch, height, width = latents.shape
                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=latents.dtype)
                latents = alpha * latents + beta * noise

            size_tensor = torch.tensor([latents.shape[-1] // self.patch_size], dtype=torch.int32, device=device)
            pos_embed = get_2d_rotary_pos_embed(
                embed_dim=self.head_dim,
                crops_coords=((0, 0), (latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size)),
                grid_size=(latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
                device=device,
                output_type="pt",
            )
            rope_pos = torch.stack(pos_embed, -1)

            if sample_fn is not None:
                # dopri5
                model_kwargs = dict(class_labels=prompt_embeds, cfg_scale=self.guidance_scale(None, stage_idx), latent_size=size_tensor, pos_embed=rope_pos)
                if stage_idx == 0:
                    latents = torch.cat([latents] * 2)
                stage_T_start = self.scheduler.Timesteps_per_stage[stage_idx][0].item()
                stage_T_end = self.scheduler.Timesteps_per_stage[stage_idx][-1].item()
                latents = sample_fn(latents, self.transformer.c2i_forward_cfg_torchdiffq, stage_T_start, stage_T_end, **model_kwargs)[-1]
                if stage_idx == self.num_stages - 1:
                    latents = latents[:latents.shape[0] // 2]
            else:
                # euler
                for T in Timesteps:
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    timestep = T.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                    if self.class_cond:
                        noise_pred = self.transformer(latent_model_input, timestep=timestep, class_labels=prompt_embeds, latent_size=size_tensor, pos_embed=rope_pos)
                    else:
                        encoder_hidden_states = prompt_embeds
                        encoder_attention_mask = prompt_attention_mask

                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            timestep=timestep,
                            latent_size=size_tensor,
                            pos_embed=rope_pos,
                        )

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale(T, stage_idx) * (noise_pred_text - noise_pred_uncond)
                    
                    x0_hat = latents + (1. - T/1000.)*noise_pred
                    x1_hat = torch.randn_like(x0_hat) ######## Take random noise shortly

                    ratio = T/1000
                    lr = self.get_lr(ratio)
                    x0_hat = copy.deepcopy(x0_hat)
                    y = self.get_stage_y(x0_hat.shape[-2], x0_hat.shape[-1])
                    x0_hat = self.langevin_sample(x0_hat, y, x0_hat, self.sigma_n, T, lr, device)
                    latents = self.scheduler.step_x0_hat(
                        x0_hat,
                        x1_hat,
                        num_inference_steps[stage_idx],
                        stage_idx,
                    )
                    # latents = self.scheduler.step(model_output=noise_pred, sample=latents)
            stage_end = time.time()

        samples = (latents / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        return samples

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def guidance_scale(self, step=None, stage_idx=None):
        if not self.class_cond:
            return self._guidance_scale
        scale_dict = {0: 0, 1: 1/6, 2: 2/3, 3: 1}
        return (self._guidance_scale - 1) * scale_dict[stage_idx] + 1

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0
    
    
    @torch.no_grad()
    def langevin_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x0hat: torch.Tensor,      
        sigma_n: float,          
        T: float, #inference_step           
        step_size: float, 
        device
    ):
        x = x.to(device)
        y = y.to(device)
        """
        One step overdamped Langevin for p(x) ∝ exp(-L(x)).
        L(x) = 1/(2σ_n^2)||Hx-y||^2 + 1/(2ν_t^2)||x-x1hat||^2
        """
        def nu_t(T):
            nt = T/1000
            # return 1-nt
            return (1 - nt)/math.sqrt(nt**2 + (1 - nt)**2)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            grad_data_sq = self.operator.gradient(x, y)
            grad_loss = (x-x0hat)**2 / (2*nu_t(T)**2 + 1e-4)
            grad_prior = torch.autograd.grad(grad_loss.sum(), x)[0]
        grad_data = grad_data_sq / (2.0 * (sigma_n ** 2))  # => (1/σ_n^2) H^T(Hx-y)
        grad_L = grad_data + grad_prior
        grad_L = torch.nan_to_num(grad_L, nan=0.0, posinf=0.0, neginf=0.0)
        if self.langevin_grad_clip is not None:
            grad_norm = grad_L.norm().clamp_min(1e-12)
            clip_val = grad_L.new_tensor(self.langevin_grad_clip)
            grad_L = grad_L * torch.clamp(clip_val / grad_norm, max=1.0)

        eps = torch.randn_like(x)
        x_new = x - step_size * grad_L + math.sqrt(2.0 * step_size) * eps
        x_new = torch.nan_to_num(x_new, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.langevin_x_clamp is not None:
            x_new = torch.clamp(x_new, -self.langevin_x_clamp, self.langevin_x_clamp)
        return x_new

    def langevin_sample(
        self,
        x_init, y, x0hat,
        sigma_n, T,
        step_size=1e-4,
        device = "cpu"
    ):
        if self.proj:
            if hasattr(self.operator, "get_mask"):
                o_M = self.operator.get_mask(x=y).to(device)
            else:
                o_M = self.operator.mask.to(device)
        else:
            o_M = torch.zeros_like(y, device=device)
        x = x_init.clone().detach()
        for _ in range(self.langevin_num_steps):
            # x = langevin_update(x, operator, y, x0hat, sigma_n, T, step_size, device)
            x = (1-o_M)*self.langevin_update(x, y, x0hat, sigma_n, T, step_size, device) + o_M*y
            if torch.isnan(x).any():
                break
        return x.detach()

    def get_lr(self, ratio):
        p = self.langevin_step_p
        lr_min_ratio = self.langevin_step_min_ratio
        multiplier = (1 ** (1 / p) + ratio * (lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.langevin_step_base

    def _interp(self, x, size):
        if self.ds_type in ("linear", "bilinear", "bicubic", "trilinear"):
            return F.interpolate(x, size=size, mode=self.ds_type, align_corners=False)
        return F.interpolate(x, size=size, mode=self.ds_type)

    def get_stage_y(self, h, w):
        target_h, target_w = int(h), int(w)
        y = self.y
        cur_h, cur_w = int(y.shape[-2]), int(y.shape[-1])

        if (cur_h, cur_w) == (target_h, target_w):
            return y

        # Fallback for upsampling or mixed resize requests.
        if target_h > cur_h or target_w > cur_w:
            return self._interp(y, size=(target_h, target_w))

        out = y
        while out.shape[-2] > target_h or out.shape[-1] > target_w:
            hh, ww = int(out.shape[-2]), int(out.shape[-1])
            next_h = max(target_h, hh // 2) if hh > target_h else hh
            next_w = max(target_w, ww // 2) if ww > target_w else ww
            if (next_h, next_w) == (hh, ww):
                break
            out = self._interp(out, size=(next_h, next_w))

        if out.shape[-2:] != (target_h, target_w):
            out = self._interp(out, size=(target_h, target_w))
        return out

# class MSPS2(MSPS):
#     """
#     MSPS variant:
#     1) Predict target state from current x_t and error_pred with global-T-aware delta.
#     2) Run Langevin on that terminal state with y/operator at matched resolution.
#     3) Re-noise via flow-matching to get next x_t.
#     """
#     def recover_x0_noise_same_x0(self, pixel_values_start, pixel_values_end, start_t, end_t, eps=1e-8):
#         denom = float(end_t - start_t)
#         if abs(denom) < eps:
#             raise ValueError("end_t and start_t are too close; cannot solve.")
#         x0 = ((1.0 - start_t) * pixel_values_end - (1.0 - end_t) * pixel_values_start) / denom
#         noise = (end_t * pixel_values_start - start_t * pixel_values_end) / denom
#         return x0, noise

#     @torch.no_grad()
#     def __call__(
#         self,
#         prompt,
#         height,
#         width,
#         num_inference_steps=30,
#         guidance_scale=4.0,
#         num_images_per_prompt=1,
#         device=None,
#         shift=1.0,
#         stage_start=-1,
#         use_ode_dopri5=False,
#         return_t0_traj=False,
#         save_t0_images=False,
#         save_t0_dir=None,
#         save_t0_nrow=4,
#         return_pixel_end_traj=False,
#         save_pixel_end_images=False,
#         save_pixel_end_dir=None,
#         save_pixel_end_nrow=4,
#         target_global_T=999.0,
#         clamp_target_to_stage_end=False,
#     ):
#         if isinstance(num_inference_steps, int):
#             num_inference_steps = [num_inference_steps] * self.num_stages
#         if len(num_inference_steps) != self.num_stages:
#             raise ValueError(
#                 f"num_inference_steps must have len={self.num_stages}, got {len(num_inference_steps)}"
#             )

#         if use_ode_dopri5:
#             raise NotImplementedError("MSPS2 currently supports Euler path only (use_ode_dopri5=False).")

#         x0_hat_before_langevin = [] if return_t0_traj else None
#         x0_hat_after_langevin = [] if return_t0_traj else None
#         pixel_end_traj = [] if return_pixel_end_traj else None

#         save_image_fn = None
#         before_dir = None
#         after_dir = None
#         pixel_end_dir = None
#         if save_t0_images or save_pixel_end_images:
#             from torchvision.utils import save_image as save_image_fn

#             if save_t0_images:
#                 if save_t0_dir is None:
#                     save_t0_dir = "./msps2_t0_snapshots"
#                 before_dir = os.path.join(save_t0_dir, "before_langevin_t0")
#                 after_dir = os.path.join(save_t0_dir, "after_langevin_t0")
#                 os.makedirs(before_dir, exist_ok=True)
#                 os.makedirs(after_dir, exist_ok=True)
#             if save_pixel_end_images:
#                 if save_pixel_end_dir is None:
#                     save_pixel_end_dir = "./msps2_pixel_end_snapshots"
#                 pixel_end_dir = os.path.join(save_pixel_end_dir, "pixel_end")
#                 os.makedirs(pixel_end_dir, exist_ok=True)

#         self._guidance_scale = guidance_scale
#         batch_size = len(prompt)

#         if self.class_cond:
#             prompt_embeds = torch.tensor(prompt, dtype=torch.int32).to(device)
#             negative_prompt_embeds = 1000 * torch.ones_like(prompt_embeds)
#             if self.do_classifier_free_guidance:
#                 prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#         else:
#             prompt_embeds, prompt_attention_mask = self.encode_prompt(
#                 prompt,
#                 device,
#                 num_images_per_prompt,
#                 guidance_scale > 1,
#                 "",
#                 prompt_embeds=None,
#                 negative_prompt_embeds=None,
#                 use_attention_mask=True,
#                 max_length=self.max_token_length,
#             )

#         init_factor = 2 ** (self.num_stages - 1)
#         height, width = height // init_factor, width // init_factor
#         shape = (batch_size * num_images_per_prompt, 3, height, width)
#         latents = randn_tensor(shape, device=device, dtype=torch.float32)

#         for stage_idx in range(self.num_stages):
#             self.scheduler.set_timesteps(
#                 num_inference_steps[stage_idx], stage_idx, device=device, shift=shift
#             )
#             Timesteps = self.scheduler.Timesteps

#             # if stage_idx + 1 < self.num_stages:
#             #     stage_start = float(self.scheduler.Timesteps_per_stage[stage_idx][0].item())
#             #     next_stage_start = float(self.scheduler.Timesteps_per_stage[stage_idx + 1][0].item())
#             #     past_t_length = stage_start
#             #     stage_t_length = next_stage_start - stage_start
#             #     stage_length = target_global_T - stage_start
#             #     future_t_length = target_global_T - next_stage_start
#             # else:
#             #     stage_start = float(self.scheduler.Timesteps_per_stage[stage_idx][0].item())
#             #     past_t_length = stage_start
#             #     stage_t_length = target_global_T - stage_start
#             #     stage_length = stage_t_length
#             #     future_t_length = 0.0
#             start_t = self.scheduler.start_t[stage_idx]
#             end_t = self.scheduler.end_t[stage_idx]
#             if stage_idx > 0:
#                 height, width = height * 2, width * 2
#                 latents = F.interpolate(latents, size=(height, width), mode="nearest")
#                 original_start_t = self.scheduler.original_start_t[stage_idx]
#                 gamma = self.scheduler.gamma
#                 alpha = 1 / (
#                     math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t
#                 )
#                 beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

#                 noise = self.sample_block_noise(*latents.shape)
#                 noise = noise.to(device=device, dtype=latents.dtype)
#                 latents = alpha * latents + beta * noise

#             size_tensor = torch.tensor([latents.shape[-1] // self.patch_size], dtype=torch.int32, device=device)
#             pos_embed = get_2d_rotary_pos_embed(
#                 embed_dim=self.head_dim,
#                 crops_coords=(
#                     (0, 0),
#                     (latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
#                 ),
#                 grid_size=(latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
#                 device=device,
#                 output_type="pt",
#             )
#             rope_pos = torch.stack(pos_embed, -1)

#             for step_idx, T in enumerate(Timesteps):
#                 latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
#                 timestep = T.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

#                 if self.class_cond:
#                     error_pred = self.transformer(
#                         latent_model_input,
#                         timestep=timestep,
#                         class_labels=prompt_embeds,
#                         latent_size=size_tensor,
#                         pos_embed=rope_pos,
#                     )
#                 else:
#                     error_pred = self.transformer(
#                         latent_model_input,
#                         encoder_hidden_states=prompt_embeds,
#                         encoder_attention_mask=prompt_attention_mask,
#                         timestep=timestep,
#                         latent_size=size_tensor,
#                         pos_embed=rope_pos,
#                     )

#                 if self.do_classifier_free_guidance:
#                     error_pred_uncond, error_pred_text = error_pred.chunk(2)
#                     error_pred = error_pred_uncond + self.guidance_scale(T, stage_idx) * (
#                         error_pred_text - error_pred_uncond
#                     )

#                 # For early stages (<= stage_start), follow the vanilla PixelFlow update.
#                 if stage_idx < stage_start:
#                     latents = self.scheduler.step(model_output=error_pred, sample=latents)
#                     continue

#                 # t_segment_past = past_t_length/stage_t_length
#                 # t_segment_future = future_t_length/stage_length
#                 # t_curr = self.scheduler.t[step_idx]
#                 # t_next = self.scheduler.t[step_idx+1]
#                 # t_past = t_segment_past + t_curr
#                 # t_future = t_segment_future + (1-t_past)
#                 # t_next_past = t_segment_past +t_next
#                 # t_next_future = t_segment_future + (1-t_next)
#                 # x0_hat = latents + t_future * error_pred
#                 t_curr = self.scheduler.t[step_idx].to(device=latents.device, dtype=latents.dtype)
#                 pixel_values_start = latents - t_curr * error_pred
#                 pixel_values_end = latents + (1.0 - t_curr) * error_pred
#                 if return_pixel_end_traj:
#                     pixel_end_traj.append(pixel_values_end.detach().cpu())
#                 if save_pixel_end_images:
#                     T_val = float(T.detach().item()) if torch.is_tensor(T) else float(T)
#                     save_image_fn(
#                         (pixel_values_end / 2 + 0.5).clamp(0, 1).detach().float().cpu(),
#                         os.path.join(
#                             pixel_end_dir,
#                             f"stage{stage_idx:02d}_step{step_idx:03d}_T{T_val:08.3f}.png",
#                         ),
#                         nrow=save_pixel_end_nrow,
#                     )
#                 x0_hat, x1_hat_noise = self.recover_x0_noise_same_x0(pixel_values_start, pixel_values_end, start_t, end_t, eps=1e-8)
#                 if return_t0_traj:
#                     x0_hat_before_langevin.append(x0_hat.detach().cpu())
#                 if save_t0_images:
#                     T_val = float(T.detach().item()) if torch.is_tensor(T) else float(T)
#                     save_image_fn(
#                         (x0_hat / 2 + 0.5).clamp(0, 1).detach().float().cpu(),
#                         os.path.join(
#                             before_dir,
#                             f"stage{stage_idx:02d}_step{step_idx:03d}_T{T_val:08.3f}.png",
#                         ),
#                         nrow=save_t0_nrow,
#                     )

#                 # Data consistency at matched resolution.
#                 y_stage = self.get_stage_y(x0_hat.shape[-2], x0_hat.shape[-1])
#                 ratio = float(T.detach().item()) / 1000.0 if torch.is_tensor(T) else float(T) / 1000.0
#                 lr = float(self.get_lr(ratio))
#                 x0_hat = self.langevin_sample(
#                     x0_hat.clone().detach(),
#                     y_stage,
#                     x0_hat,
#                     self.sigma_n,
#                     T,
#                     step_size=lr,
#                     device=device,
#                 )
#                 if return_t0_traj:
#                     x0_hat_after_langevin.append(x0_hat.detach().cpu())
#                 if save_t0_images:
#                     T_val = float(T.detach().item()) if torch.is_tensor(T) else float(T)
#                     save_image_fn(
#                         (x0_hat / 2 + 0.5).clamp(0, 1).detach().float().cpu(),
#                         os.path.join(
#                             after_dir,
#                             f"stage{stage_idx:02d}_step{step_idx:03d}_T{T_val:08.3f}.png",
#                         ),
#                         nrow=save_t0_nrow,
#                     )

#                 # Flow-matching re-noise to next x_t.
#                 x1_hat = torch.randn_like(x0_hat)
#                 t_global = self.scheduler.t[step_idx + 1] * (end_t - start_t) + start_t
#                 t_global = t_global.to(device=x0_hat.device, dtype=x0_hat.dtype)
#                 x1_hat = t_global * x1_hat + torch.sqrt(torch.clamp(1.0 - t_global, min=0.0)) * x1_hat_noise
#                 # x1_hat = math.sqrt()*(latents - t_cur * error_pred) + t_next_future*torch.randn_like(x_stage_end)
#                 latents = self.scheduler.step_x0_hat(
#                     x0_hat,
#                     x1_hat,
#                     stage_idx,
#                 )
#         samples = (latents / 2 + 0.5).clamp(0, 1)
#         samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
#         if return_t0_traj and return_pixel_end_traj:
#             return samples, x0_hat_before_langevin, x0_hat_after_langevin, pixel_end_traj
#         if return_t0_traj:
#             return samples, x0_hat_before_langevin, x0_hat_after_langevin
#         if return_pixel_end_traj:
#             return samples, pixel_end_traj
#         return samples
class MSPS3(MSPS):
    """
    MSPS variant:
    1) Predict target state from current x_t and error_pred with global-T-aware delta.
    2) Run Langevin on that terminal state with y/operator at matched resolution.
    3) Re-noise via flow-matching to get next x_t.
    """
    def recover_x0_noise_same_x0(self, pixel_values_start, pixel_values_end, start_t, end_t, eps=1e-8):
        denom = float(end_t - start_t)
        if abs(denom) < eps:
            raise ValueError("end_t and start_t are too close; cannot solve.")
        x0 = ((1.0 - start_t) * pixel_values_end - (1.0 - end_t) * pixel_values_start) / denom
        noise = (end_t * pixel_values_start - start_t * pixel_values_end) / denom
        return x0, noise

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height,
        width,
        num_inference_steps=30,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        device=None,
        shift=1.0,
        stage_start=-1,
        use_ode_dopri5=False,
        return_t0_traj=False,
        save_t0_images=False,
        save_t0_dir=None,
        save_t0_nrow=4,
        return_pixel_end_traj=False,
        save_pixel_end_images=False,
        save_pixel_end_dir=None,
        save_pixel_end_nrow=4,
        # target_global_T=999.0,
        # clamp_target_to_stage_end=False,
        return_langevin_traj = False
    ):
        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * self.num_stages
        if len(num_inference_steps) != self.num_stages:
            raise ValueError(
                f"num_inference_steps must have len={self.num_stages}, got {len(num_inference_steps)}"
            )

        if use_ode_dopri5:
            raise NotImplementedError("MSPS3 currently supports Euler path only (use_ode_dopri5=False).")

        x0_hat_before_langevin = [] if return_t0_traj else None
        x0_hat_after_langevin = [] if return_t0_traj else None
        pixel_end_traj = [] if return_pixel_end_traj else None

        save_image_fn = None
        before_dir = None
        after_dir = None
        pixel_end_dir = None
        if save_t0_images or save_pixel_end_images:
            from torchvision.utils import save_image as save_image_fn

            if save_t0_images:
                if save_t0_dir is None:
                    save_t0_dir = "./msps3_t0_snapshots"
                before_dir = os.path.join(save_t0_dir, "before_langevin_t0")
                after_dir = os.path.join(save_t0_dir, "after_langevin_t0")
                os.makedirs(before_dir, exist_ok=True)
                os.makedirs(after_dir, exist_ok=True)
            if save_pixel_end_images:
                if save_pixel_end_dir is None:
                    save_pixel_end_dir = "./msps3_pixel_end_snapshots"
                pixel_end_dir = os.path.join(save_pixel_end_dir, "pixel_end")
                os.makedirs(pixel_end_dir, exist_ok=True)

        self._guidance_scale = guidance_scale
        batch_size = len(prompt)

        if self.class_cond:
            prompt_embeds = torch.tensor(prompt, dtype=torch.int32).to(device)
            negative_prompt_embeds = 1000 * torch.ones_like(prompt_embeds)
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale > 1,
                "",
                prompt_embeds=None,
                negative_prompt_embeds=None,
                use_attention_mask=True,
                max_length=self.max_token_length,
            )

        init_factor = 2 ** (self.num_stages - 1)
        height, width = height // init_factor, width // init_factor
        shape = (batch_size * num_images_per_prompt, 3, height, width)
        latents = randn_tensor(shape, device=device, dtype=torch.float32)

        for stage_idx in range(self.num_stages):
            self.scheduler.set_timesteps(
                num_inference_steps[stage_idx], stage_idx, device=device, shift=shift
            )
            Timesteps = self.scheduler.Timesteps

            # if stage_idx + 1 < self.num_stages:
            #     stage_start = float(self.scheduler.Timesteps_per_stage[stage_idx][0].item())
            #     next_stage_start = float(self.scheduler.Timesteps_per_stage[stage_idx + 1][0].item())
            #     past_t_length = stage_start
            #     stage_t_length = next_stage_start - stage_start
            #     stage_length = target_global_T - stage_start
            #     future_t_length = target_global_T - next_stage_start
            # else:
            #     stage_start = float(self.scheduler.Timesteps_per_stage[stage_idx][0].item())
            #     past_t_length = stage_start
            #     stage_t_length = target_global_T - stage_start
            #     stage_length = stage_t_length
            #     future_t_length = 0.0
            start_t = self.scheduler.start_t[stage_idx]
            end_t = self.scheduler.end_t[stage_idx]
            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                original_start_t = self.scheduler.original_start_t[stage_idx]
                gamma = self.scheduler.gamma
                alpha = 1 / (
                    math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t
                )
                beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=latents.dtype)
                latents = alpha * latents + beta * noise

            size_tensor = torch.tensor([latents.shape[-1] // self.patch_size], dtype=torch.int32, device=device)
            pos_embed = get_2d_rotary_pos_embed(
                embed_dim=self.head_dim,
                crops_coords=(
                    (0, 0),
                    (latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
                ),
                grid_size=(latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
                device=device,
                output_type="pt",
            )
            rope_pos = torch.stack(pos_embed, -1)

            for step_idx, T in enumerate(Timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = T.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                if self.class_cond:
                    error_pred = self.transformer(
                        latent_model_input,
                        timestep=timestep,
                        class_labels=prompt_embeds,
                        latent_size=size_tensor,
                        pos_embed=rope_pos,
                    )
                else:
                    error_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=timestep,
                        latent_size=size_tensor,
                        pos_embed=rope_pos,
                    )

                if self.do_classifier_free_guidance:
                    error_pred_uncond, error_pred_text = error_pred.chunk(2)
                    error_pred = error_pred_uncond + self.guidance_scale(T, stage_idx) * (
                        error_pred_text - error_pred_uncond
                    )

                # For early stages (<= stage_start), follow the vanilla PixelFlow update.
                if stage_idx < stage_start:
                    latents = self.scheduler.step(model_output=error_pred, sample=latents)
                    continue

                # t_segment_past = past_t_length/stage_t_length
                # t_segment_future = future_t_length/stage_length
                # t_curr = self.scheduler.t[step_idx]
                # t_next = self.scheduler.t[step_idx+1]
                # t_past = t_segment_past + t_curr
                # t_future = t_segment_future + (1-t_past)
                # t_next_past = t_segment_past +t_next
                # t_next_future = t_segment_future + (1-t_next)
                # x0_hat = latents + t_future * error_pred
                t_curr = self.scheduler.t[step_idx].to(device=latents.device, dtype=latents.dtype)
                pixel_values_start = latents - t_curr * error_pred
                pixel_values_end = latents + (1.0 - t_curr) * error_pred
                if return_pixel_end_traj:
                    pixel_end_traj.append(pixel_values_end.detach().cpu())
                if save_pixel_end_images:
                    T_val = float(T.detach().item()) if torch.is_tensor(T) else float(T)
                    save_image_fn(
                        (pixel_values_end / 2 + 0.5).clamp(0, 1).detach().float().cpu(),
                        os.path.join(
                            pixel_end_dir,
                            f"stage{stage_idx:02d}_step{step_idx:03d}_T{T_val:08.3f}.png",
                        ),
                        nrow=save_pixel_end_nrow,
                    )
                x0_hat, x1_hat_noise = self.recover_x0_noise_same_x0(pixel_values_start, pixel_values_end, start_t, end_t, eps=1e-8)
                if return_t0_traj:
                    x0_hat_before_langevin.append(x0_hat.detach().cpu())
                if save_t0_images:
                    T_val = float(T.detach().item()) if torch.is_tensor(T) else float(T)
                    save_image_fn(
                        (x0_hat / 2 + 0.5).clamp(0, 1).detach().float().cpu(),
                        os.path.join(
                            before_dir,
                            f"stage{stage_idx:02d}_step{step_idx:03d}_T{T_val:08.3f}.png",
                        ),
                        nrow=save_t0_nrow,
                    )

                # Data consistency at matched resolution.
                # y_stage = self.get_stage_y(x0_hat.shape[-2], x0_hat.shape[-1])
                ratio = float(T.detach().item()) / 1000.0 if torch.is_tensor(T) else float(T) / 1000.0
                lr = float(self.get_lr(ratio))
                x0_hat = self.langevin_sample_split_prior(x0_hat.clone().detach(), self.y, end_t, start_t,
                                                          pixel_values_end, pixel_values_start, self.sigma_n, lr, device, return_langevin_traj )

                if return_t0_traj:
                    x0_hat_after_langevin.append(x0_hat.detach().cpu())
                if save_t0_images:
                    T_val = float(T.detach().item()) if torch.is_tensor(T) else float(T)
                    save_image_fn(
                        (x0_hat / 2 + 0.5).clamp(0, 1).detach().float().cpu(),
                        os.path.join(
                            after_dir,
                            f"stage{stage_idx:02d}_step{step_idx:03d}_T{T_val:08.3f}.png",
                        ),
                        nrow=save_t0_nrow,
                    )

                # Flow-matching re-noise to next x_t.
                x1_hat = torch.randn_like(x0_hat)
                t_global = self.scheduler.t[step_idx + 1] * (end_t - start_t) + start_t
                t_global = t_global.to(device=x0_hat.device, dtype=x0_hat.dtype)
                x1_hat = t_global * x1_hat + torch.sqrt(torch.clamp(1.0 - t_global, min=0.0)) * x1_hat_noise
                # x1_hat = math.sqrt()*(latents - t_cur * error_pred) + t_next_future*torch.randn_like(x_stage_end)
                latents = self.scheduler.step_x0_hat(
                    x0_hat,
                    x1_hat,
                    stage_idx,
                )
        samples = (latents / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        if return_t0_traj and return_pixel_end_traj:
            return samples, x0_hat_before_langevin, x0_hat_after_langevin, pixel_end_traj
        if return_t0_traj:
            return samples, x0_hat_before_langevin, x0_hat_after_langevin
        if return_pixel_end_traj:
            return samples, pixel_end_traj
        return samples
