"""ReTime engine: frequency-aware temporal RoPE warping for Wan2.2.

This module provides the core ReTime functions used by both the ComfyUI
node and the standalone inference script.  The key function is
``create_patched_rope_apply`` which returns a drop-in replacement for
Wan2.2's ``rope_apply`` with per-band warped temporal positions.
"""

import math
from contextlib import contextmanager

import numpy as np
import torch

from .solver import invert_potential

NUM_ROPE_BANDS = 22
WAN_TEMPORAL_COMPRESS = 4


def build_alpha_schedule(
    alpha_low: float,
    alpha_high: float,
    decay_lambda: float,
    num_bands: int = NUM_ROPE_BANDS,
) -> np.ndarray:
    """Exponential alpha decay across frequency bands (paper Eq.).

    In Wan's RoPE ordering, band 0 is the highest frequency (local detail)
    and band B-1 is the lowest frequency (global structure).  The formula
    yields ``alpha_low`` at band 0 and approaches ``alpha_high`` at band B-1.
    """
    b = np.arange(num_bands, dtype=np.float64)
    return alpha_high + (alpha_low - alpha_high) * np.exp(
        -decay_lambda * b / (num_bands - 1)
    )


def timestep_decay_multiplier(mode: str, t_normalised: float) -> float:
    """Compute gamma(t) in [0, 1] for a given normalised diffusion timestep."""
    if mode == "none":
        return 1.0
    if mode == "exp_up":
        return float((math.exp(3 * t_normalised) - 1) / (math.exp(3) - 1))
    if mode == "exp_down":
        return float((math.exp(3 * (1 - t_normalised)) - 1) / (math.exp(3) - 1))
    if mode == "linear_up":
        return float(t_normalised)
    if mode == "linear_down":
        return float(1.0 - t_normalised)
    if mode == "cosine_up":
        return float(0.5 * (1 + math.cos(math.pi * (1 - t_normalised))))
    if mode == "cosine_down":
        return float(0.5 * (1 + math.cos(math.pi * t_normalised)))
    return 1.0


def frame_to_latent_positions(
    frame_positions: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """Resample frame-space positions to Wan's latent temporal resolution.

    All indices are 0-based.  Wan VAE mapping (0-indexed):
        latent 0  -> frame 0
        latent i (i >= 1) -> centre of 0-indexed frames [4(i-1)+1, 4i],
                             i.e. frame centre = 4i - 1.5
    """
    num_latent = (num_frames - 1) // WAN_TEMPORAL_COMPRESS + 1
    scale = (num_latent - 1) / max(num_frames - 1, 1)

    centres = np.zeros(num_latent)
    for i in range(1, num_latent):
        centres[i] = WAN_TEMPORAL_COMPRESS * i - 1.5

    frame_indices = np.arange(num_frames, dtype=np.float64)

    if frame_positions.ndim == 1:
        return np.interp(centres, frame_indices, frame_positions) * scale

    out = np.zeros((frame_positions.shape[0], num_latent))
    for b in range(frame_positions.shape[0]):
        out[b] = np.interp(centres, frame_indices, frame_positions[b]) * scale
    return out


def extract_temporal_omega(model_freqs: torch.Tensor, num_heads: int, dim: int) -> torch.Tensor:
    """Extract the temporal frequency parameters (omega) from a Wan model's freqs buffer."""
    d = dim // num_heads
    temporal_dim = d - 4 * (d // 6)
    temporal_freqs = model_freqs.detach().cpu()[:, :temporal_dim // 2]
    return torch.angle(temporal_freqs[1])


def compute_warped_rope_freqs(
    latent_positions: np.ndarray,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Compute complex RoPE frequencies for per-band warped temporal positions.

    Args:
        latent_positions: (NUM_ROPE_BANDS, num_latent_steps) warped positions.
        omega: (NUM_ROPE_BANDS,) base frequency for each temporal band.

    Returns:
        Complex frequency tensor of shape (num_latent_steps, NUM_ROPE_BANDS).
    """
    pos = torch.as_tensor(latent_positions, dtype=torch.float64)
    omega_cpu = omega.detach().cpu().to(torch.float64)
    angles = pos.T * omega_cpu.unsqueeze(0)
    return torch.polar(torch.ones_like(angles), angles)


def create_patched_rope_apply(original_rope_apply, warped_temporal_freqs):
    """Return a patched rope_apply that uses custom temporal frequencies."""
    @torch.amp.autocast('cuda', enabled=False)
    def rope_apply_patched(x, grid_sizes, freqs):
        n, c = x.size(2), x.size(3) // 2
        freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # Guard: warped temporal bands must match model's temporal dimension
        assert warped_temporal_freqs.shape[1] == freqs_split[0].shape[1], (
            f"Temporal band mismatch: warped has {warped_temporal_freqs.shape[1]}, "
            f"model has {freqs_split[0].shape[1]}"
        )

        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

            t_freqs = warped_temporal_freqs[:f].to(device=x.device)

            freqs_i = torch.cat([
                t_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)

            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        return torch.stack(output).float()

    return rope_apply_patched


@contextmanager
def retime_context(wan_model_module, warped_temporal_freqs):
    """Context manager that monkey-patches rope_apply for the duration of a block."""
    original = wan_model_module.rope_apply
    wan_model_module.rope_apply = create_patched_rope_apply(original, warped_temporal_freqs)
    try:
        yield
    finally:
        wan_model_module.rope_apply = original


def compute_retime_freqs(
    potential: np.ndarray,
    model_freqs: torch.Tensor,
    num_heads: int,
    dim: int,
    num_frames: int = 0,
    alpha_low: float = 0.2,
    alpha_high: float = 1.0,
    decay_lambda: float = 3.0,
    timestep_decay_mode: str = "exp_up",
    t_normalised: float = 1.0,
) -> torch.Tensor:
    """One-call convenience: potential -> warped temporal RoPE frequencies."""
    nf = len(potential) if num_frames == 0 else num_frames
    alpha = build_alpha_schedule(alpha_low, alpha_high, decay_lambda)
    gamma = timestep_decay_multiplier(timestep_decay_mode, t_normalised)
    alpha_eff = alpha * gamma
    warped = invert_potential(potential, num_frames=nf)
    linear = np.arange(nf, dtype=np.float64)
    a = alpha_eff[:, None]
    per_band_frame = (1 - a) * linear[None, :] + a * warped[None, :]
    latent_positions = frame_to_latent_positions(per_band_frame, nf)
    omega = extract_temporal_omega(model_freqs, num_heads, dim)
    return compute_warped_rope_freqs(latent_positions, omega)


def generate_retimed_f2lf(
    pipeline,
    wan_model_module,
    potential: np.ndarray,
    first_frame,
    last_frame=None,
    prompt: str = "smooth transformation between the two frames",
    n_prompt: str = "",
    frame_num: int = 81,
    max_area: int = 480 * 480,
    seed: int = 42,
    sampling_steps: int = 40,
    shift: float = 5.0,
    guide_scale=(3.5, 3.5),
    alpha_low: float = 0.2,
    alpha_high: float = 1.0,
    decay_lambda: float = 3.0,
    timestep_decay_mode: str = "exp_up",
    offload_model: bool = True,
):
    """Generate a retimed video with first-to-last-frame conditioning.

    This function replaces ``WanI2V.generate()`` with two enhancements:
    1. First+last frame conditioning (F2LF) using the same I2V weights
    2. Per-step frequency-aware RoPE warping (ReTime) with timestep decay

    Args:
        pipeline: A ``WanI2V`` instance (already loaded).
        wan_model_module: The ``wan.modules.model`` module (for monkey-patching).
        potential: SPF potential curve from the input video.
        first_frame: PIL Image for the first frame.
        last_frame: PIL Image for the last frame (optional; if None, first-frame only).
        prompt: Text prompt.
        frame_num: Number of output frames (must be 4n+1).
        max_area: Maximum pixel area (controls resolution).
        seed: Random seed.
        alpha_low / alpha_high / decay_lambda: ReTime alpha schedule params.
        timestep_decay_mode: How warp strength varies during denoising.
        offload_model: Offload models to CPU between steps to save VRAM.

    Returns:
        Video tensor of shape (C, T, H, W) in [-1, 1], or None if not rank 0.
    """
    import gc
    import math as _math
    import random
    import sys as _sys
    from contextlib import contextmanager as _cm
    from tqdm import tqdm
    import torch.nn.functional as F_nn
    import torchvision.transforms.functional as TF

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    device = pipeline.device
    config = pipeline.config

    # -- Preprocess images --
    img_first = TF.to_tensor(first_frame).sub_(0.5).div_(0.5).to(device)
    h_orig, w_orig = img_first.shape[1:]
    aspect_ratio = h_orig / w_orig
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // pipeline.vae_stride[1] //
        pipeline.patch_size[1] * pipeline.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // pipeline.vae_stride[2] //
        pipeline.patch_size[2] * pipeline.patch_size[2])
    h = lat_h * pipeline.vae_stride[1]
    w = lat_w * pipeline.vae_stride[2]

    F = frame_num
    max_seq_len = ((F - 1) // pipeline.vae_stride[0] + 1) * lat_h * lat_w // (
        pipeline.patch_size[1] * pipeline.patch_size[2])
    max_seq_len = int(_math.ceil(max_seq_len / pipeline.sp_size)) * pipeline.sp_size

    # -- Noise --
    seed = seed if seed >= 0 else random.randint(0, _sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16, (F - 1) // pipeline.vae_stride[0] + 1, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device)

    # -- F2LF Mask --
    msk = torch.zeros(1, F, lat_h, lat_w, device=device)
    msk[:, 0] = 1  # first frame = conditioned
    if last_frame is not None:
        msk[:, -1] = 1  # last frame = conditioned
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    # -- F2LF Image (VAE encode first + last frames) --
    first_resized = F_nn.interpolate(
        img_first[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1)
    video_frames = torch.zeros(3, F, h, w)
    video_frames[:, 0] = first_resized[:, 0]
    if last_frame is not None:
        img_last = TF.to_tensor(last_frame).sub_(0.5).div_(0.5)
        last_resized = F_nn.interpolate(
            img_last[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1)
        video_frames[:, -1] = last_resized[:, 0]

    y = pipeline.vae.encode([video_frames.to(device)])[0]
    y = torch.concat([msk, y])

    # -- Text encoding --
    if n_prompt == "":
        n_prompt = pipeline.sample_neg_prompt
    if pipeline.t5_cpu:
        context = pipeline.text_encoder([prompt], torch.device('cpu'))
        context_null = pipeline.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]
    else:
        pipeline.text_encoder.model.to(device)
        context = pipeline.text_encoder([prompt], device)
        context_null = pipeline.text_encoder([n_prompt], device)
        if offload_model:
            pipeline.text_encoder.model.cpu()

    # -- ReTime: extract omega from model (freqs is always on CPU with init_on_cpu) --
    omega = extract_temporal_omega(
        pipeline.high_noise_model.freqs, config.num_heads, config.dim
    )

    alpha = build_alpha_schedule(alpha_low, alpha_high, decay_lambda)
    warped_frame = invert_potential(potential, num_frames=F)
    linear_frame = np.arange(F, dtype=np.float64)

    @_cm
    def noop():
        yield

    no_sync_low = getattr(pipeline.low_noise_model, 'no_sync', noop)
    no_sync_high = getattr(pipeline.high_noise_model, 'no_sync', noop)

    with (
        torch.amp.autocast('cuda', dtype=pipeline.param_dtype),
        torch.no_grad(),
        no_sync_low(),
        no_sync_high(),
    ):
        boundary = pipeline.boundary * pipeline.num_train_timesteps

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps

        latent = noise
        arg_c = {'context': [context[0]], 'seq_len': max_seq_len, 'y': [y]}
        arg_null = {'context': context_null, 'seq_len': max_seq_len, 'y': [y]}

        if offload_model:
            torch.cuda.empty_cache()

        for _, t in enumerate(tqdm(timesteps, desc="ReTime denoising")):
            latent_model_input = [latent.to(device)]
            ts = torch.stack([t]).to(device)

            model = pipeline._prepare_model_for_timestep(t, boundary, offload_model)
            cfg_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

            # Per-step ReTime: compute warped RoPE with timestep decay
            t_norm = t.item() / pipeline.num_train_timesteps
            gamma = timestep_decay_multiplier(timestep_decay_mode, t_norm)
            alpha_eff = alpha * gamma
            a = alpha_eff[:, None]
            per_band = (1 - a) * linear_frame[None, :] + a * warped_frame[None, :]
            latent_pos = frame_to_latent_positions(per_band, F)
            warped_freqs = compute_warped_rope_freqs(latent_pos, omega)

            with retime_context(wan_model_module, warped_freqs):
                pred_cond = model(latent_model_input, t=ts, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                pred_uncond = model(latent_model_input, t=ts, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()

            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            temp_x0 = scheduler.step(
                pred.unsqueeze(0), t, latent.unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            del latent_model_input, ts

        x0 = [latent]

        if offload_model:
            pipeline.low_noise_model.cpu()
            pipeline.high_noise_model.cpu()
            torch.cuda.empty_cache()

        if pipeline.rank == 0:
            videos = pipeline.vae.decode(x0)

    del noise, latent, x0, scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()

    return videos[0] if pipeline.rank == 0 else None
