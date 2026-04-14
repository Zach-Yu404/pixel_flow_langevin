# 

from einops import rearrange
import math
from typing import List, Optional, Union
import time
import torch
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_2d_rotary_pos_embed


class PixelFlowPipeline2:
    def __init__(
        self,
        scheduler,
        transformer,
        text_encoder=None,
        tokenizer=None,
        max_token_length=512,
    ):
        super().__init__()
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
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

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

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                if len(negative_prompt) != batch_size:
                    raise ValueError(f"The negative prompt list must have the same length as the prompt list, but got {len(negative_prompt)} and {batch_size}")
                uncond_tokens = negative_prompt
            else:
                raise ValueError(f"Negative prompt must be a string or a list of strings, but got {type(negative_prompt)}")

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
            seq_len_neg = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        return prompt_embeds, prompt_attention_mask

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.scheduler.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(4),
            torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4)
        )
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)])  # [block number, 4]
        noise = rearrange(
            noise,
            '(b c h w) (p q) -> b c (h p) (w q)',
            b=bs, c=ch, h=height // 2, w=width // 2, p=2, q=2
        )
        return noise

    # ======== 新增：给 mid-start 用的 helper（不改原 step 逻辑）========
    def _jump_one_step_to_next_grid(
        self,
        latents: torch.Tensor,
        stage_idx: int,
        T_start: float,
        prompt_embeds,
        prompt_attention_mask,
        size_tensor,
        rope_pos,
        device,
    ):
        """
        从任意 start_T（训练步尺度）做一次 Euler jump 到该 stage 的“下一格 Timesteps[j]”，
        然后把 scheduler.Timesteps / scheduler.t 切片，从 j 开始继续原本循环。
        """
        scheduler = self.scheduler
        T_start = float(T_start)

        T_grid = scheduler.Timesteps.to(device=device, dtype=torch.float64)  # (K,)
        t_grid = scheduler.t.to(device=device, dtype=torch.float64)          # (K+1,) 但我们只用前 K 个对齐 T_grid
        K = T_grid.numel()
        if K <= 0:
            return latents

        t_pts = t_grid[:K]

        # 判断 Timesteps 是升序还是降序（torch.searchsorted 需要升序，所以降序用 -T trick）
        descending = bool((T_grid[0] - T_grid[-1]).item() > 0)
        if descending:
            key = -T_grid
            val = -torch.tensor(T_start, device=device, dtype=torch.float64)
        else:
            key = T_grid
            val = torch.tensor(T_start, device=device, dtype=torch.float64)

        j = int(torch.searchsorted(key, val, right=False).item())
        j = max(0, min(j, K - 1))

        # 如果刚好落在网格点上：不 jump，只切片
        if abs(float(T_grid[j].item()) - T_start) < 1e-9:
            scheduler.Timesteps = scheduler.Timesteps[j:]
            scheduler.t = scheduler.t[j:]
            scheduler._step_index = None
            return latents

        # 线性插值/外推得到 t_start（用相邻两点）
        if descending:
            key_all = -T_grid
            val_all = -T_start
        else:
            key_all = T_grid
            val_all = T_start

        i = int(torch.searchsorted(key_all, torch.tensor(val_all, device=device, dtype=torch.float64), right=False).item())
        if i <= 0:
            i0, i1 = 0, min(1, K - 1)
        elif i >= K:
            i0, i1 = max(K - 2, 0), K - 1
        else:
            i0, i1 = i - 1, i

        T0 = float(T_grid[i0].item()); T1 = float(T_grid[i1].item())
        t0 = float(t_pts[i0].item());  t1 = float(t_pts[i1].item())
        if abs(T1 - T0) < 1e-12:
            t_start = t0
        else:
            w = (T_start - T0) / (T1 - T0)
            t_start = t0 + w * (t1 - t0)

        t_next = float(t_pts[j].item())
        dt = t_next - t_start

        # === 用原本逻辑算一次 noise_pred(v) 在 T_start ===
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        timestep = torch.full(
            (latent_model_input.shape[0],),
            fill_value=T_start,
            device=device,
            dtype=latent_model_input.dtype,
        )

        if self.class_cond:
            noise_pred = self.transformer(
                latent_model_input,
                timestep=timestep,
                class_labels=prompt_embeds,
                latent_size=size_tensor,
                pos_embed=rope_pos,
            )
        else:
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                latent_size=size_tensor,
                pos_embed=rope_pos,
            )

        if self.do_classifier_free_guidance:
            u, c = noise_pred.chunk(2)
            noise_pred = u + self.guidance_scale(T_start, stage_idx) * (c - u)

        # Euler jump
        latents = latents.to(torch.float32) + float(dt) * noise_pred.to(torch.float32)
        latents = latents.to(noise_pred.dtype)

        # 切片后继续原循环
        scheduler.Timesteps = scheduler.Timesteps[j:]
        scheduler.t = scheduler.t[j:]
        scheduler._step_index = None
        return latents

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
        # ======== 新增：mid-start 参数（默认不影响原行为）========
        xt: Optional[torch.Tensor] = None,
        start_stage: Optional[int] = None,
        start_T: Optional[Union[int, float]] = None,
        normalized_x0_hat: bool = True,
        debug: bool = False,
    ):
        # ======== 新增：device=None 时跟模型走（否则你会遇到 cpu/cuda 混用）========
        if device is None:
            device = self.device

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * self.num_stages

        if use_ode_dopri5:
            # 原逻辑保留；mid-start + dopri5 这里不保证正确（非必要不改动原实现）
            assert xt is None, "mid-start (xt/start_T) 目前只支持 Euler 路径（use_ode_dopri5=False）"
            assert self.class_cond, "ODE (dopri5) sampling is only supported for class-conditional models now"
            from pixelflow.solver_ode_wrapper import ODE
            sample_fn = ODE(t0=0, t1=1, sampler_type="dopri5", num_steps=num_inference_steps[0], atol=1e-06, rtol=0.001).sample
        else:
            sample_fn = None

        self._guidance_scale = guidance_scale
        batch_size = len(prompt)

        # ===== prompt 编码（原逻辑，只有很小的补全）=====
        if self.class_cond:
            prompt_embeds = torch.tensor(prompt, dtype=torch.int32).to(device)
            prompt_attention_mask = None  # 仅为统一接口
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

        # ===== 初始化 latents：原始 or 从 xt 开始（新增，不影响 xt=None 的原行为）=====
        if xt is None:
            init_factor = 2 ** (self.num_stages - 1)
            height, width = height // init_factor, width // init_factor
            shape = (batch_size * num_images_per_prompt, 3, height, width)
            latents = randn_tensor(shape, device=device, dtype=torch.float32)
            stage_begin = 0
        else:
            assert start_stage is not None and start_T is not None, "xt!=None 时必须提供 start_stage 和 start_T"
            assert xt.ndim == 4 and xt.shape[1] == 3, f"xt must be BCHW with C=3, got {tuple(xt.shape)}"
            latents = xt.to(device=device, dtype=torch.float32)
            height, width = latents.shape[-2], latents.shape[-1]
            stage_begin = int(start_stage)

        # ===== stage 循环：唯一改动是 range(stage_begin, ...)（xt=None 等价原版）=====
        for stage_idx in range(stage_begin, self.num_stages):
            self.scheduler.set_timesteps(num_inference_steps[stage_idx], stage_idx, device=device, shift=shift)
            Timesteps = self.scheduler.Timesteps

            # ===== upsample+renoise：原逻辑保持，但 mid-start 的起始 stage 不做这步 =====
            if stage_idx > 0 and (xt is None or stage_idx > stage_begin):
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                original_start_t = self.scheduler.original_start_t[stage_idx]
                gamma = self.scheduler.gamma
                alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
                beta = alpha * (1 - original_start_t) / math.sqrt(- gamma)

                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=latents.dtype)
                latents = alpha * latents + beta * noise

            # ===== RoPE：只加了 .to(device)（否则 rope_pos 默认 CPU 会炸）=====
            size_tensor = torch.tensor([latents.shape[-1] // self.patch_size], dtype=torch.int32, device=device)
            pos_embed = get_2d_rotary_pos_embed(
                embed_dim=self.head_dim,
                crops_coords=((0, 0), (latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size)),
                grid_size=(latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
                output_type="pt"
            )
            rope_pos = torch.stack(pos_embed, -1).to(device=device)

            # ===== mid-start：仅在起始 stage 做一次 jump（新增，不影响原流程）=====
            if xt is not None and stage_idx == stage_begin:
                latents = self._jump_one_step_to_next_grid(
                    latents=latents,
                    stage_idx=stage_idx,
                    T_start=float(start_T),
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    size_tensor=size_tensor,
                    rope_pos=rope_pos,
                    device=device,
                )
                Timesteps = self.scheduler.Timesteps  # jump 后 Timesteps 被切片

            if sample_fn is not None:
                # 原 dopri5 路径不动
                model_kwargs = dict(
                    class_labels=prompt_embeds,
                    cfg_scale=self.guidance_scale(None, stage_idx),
                    latent_size=size_tensor,
                    pos_embed=rope_pos
                )
                if stage_idx == 0:
                    latents = torch.cat([latents] * 2)
                stage_T_start = self.scheduler.Timesteps_per_stage[stage_idx][0].item()
                stage_T_end = self.scheduler.Timesteps_per_stage[stage_idx][-1].item()
                latents = sample_fn(latents, self.transformer.c2i_forward_cfg_torchdiffq, stage_T_start, stage_T_end, **model_kwargs)[-1]
                if stage_idx == self.num_stages - 1:
                    latents = latents[:latents.shape[0] // 2]
            else:
                # ===== euler：原逻辑完全不变（仅去掉强制 pdb）=====
                for T in Timesteps:
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    timestep = T.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                    if self.class_cond:
                        noise_pred = self.transformer(
                            latent_model_input,
                            timestep=timestep,
                            class_labels=prompt_embeds,
                            latent_size=size_tensor,
                            pos_embed=rope_pos
                        )
                    else:
                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=timestep,
                            latent_size=size_tensor,
                            pos_embed=rope_pos,
                        )

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale(T, stage_idx) * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(model_output=noise_pred, sample=latents)

                    if debug:
                        print(f"[debug] stage={stage_idx}, T={float(T.item())}, latents={tuple(latents.shape)}")

        if normalized_x0_hat:
            samples = (latents / 2 + 0.5).clamp(0, 1)
            samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
            return samples
        else:
            return latents

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
        # ✅ 按你原始逻辑，不改（>0 就启用）
        return self._guidance_scale > 0
