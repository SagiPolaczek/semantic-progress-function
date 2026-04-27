"""ComfyUI node: ReTime for Wan2.2 via frequency-aware RoPE warping.

This node stores warped positions in ``rope_options["freq_aware_warp"]``,
which is consumed by a patched ComfyUI Wan model (``rope_encode`` method).
Stock ComfyUI does not read this dict — see the project README for setup.
"""

import numpy as np
import torch

from spf.retime import (
    build_alpha_schedule,
    frame_to_latent_positions,
    NUM_ROPE_BANDS,
    WAN_TEMPORAL_COMPRESS,
)
from spf.solver import invert_potential

DECAY_MODES = ["exp_up", "none", "linear_up", "linear_down",
               "cosine_up", "cosine_down", "exp_down"]


class ReTimeWan:
    """Apply frequency-aware temporal RoPE warping to a Wan2.2 model.

    Takes a pre-computed SPF potential and patches the model so that
    generation produces approximately linear perceptual progress.

    This is a single-pass node.  The paper's iterative refinement scheme
    is not included in this minimal release.
    """

    CATEGORY = "ReTime"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "potential": ("POTENTIAL",),
                "num_output_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": "0 = same as potential length",
                }),
                "alpha_low": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Warp strength for highest-frequency RoPE bands",
                }),
                "alpha_high": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Warp strength for lowest-frequency RoPE bands",
                }),
                "decay_lambda": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Controls steepness of alpha decay across bands",
                }),
                "timestep_decay": (DECAY_MODES, {
                    "default": "exp_up",
                    "tooltip": "How warp strength varies during denoising",
                }),
            },
        }

    def run(self, model, potential, num_output_frames,
            alpha_low, alpha_high, decay_lambda, timestep_decay):
        num_frames = len(potential) if num_output_frames == 0 else num_output_frames
        alpha = build_alpha_schedule(alpha_low, alpha_high, decay_lambda)
        warped = invert_potential(potential, num_frames=num_frames)

        linear = np.arange(num_frames, dtype=np.float64)
        a = alpha[:, None]
        per_band_frame = (1 - a) * linear[None, :] + a * warped[None, :]

        linear_latent = frame_to_latent_positions(
            np.broadcast_to(linear, (NUM_ROPE_BANDS, num_frames)).copy(),
            num_frames,
        )
        warped_latent = frame_to_latent_positions(per_band_frame, num_frames)

        patched = model.clone()
        rope_opts = patched.model_options.setdefault(
            "transformer_options", {}
        ).setdefault("rope_options", {})
        rope_opts["freq_aware_warp"] = {
            "linear_positions": torch.tensor(linear_latent, dtype=torch.float32),
            "warped_positions": torch.tensor(warped_latent, dtype=torch.float32),
            "alpha_schedule": torch.tensor(alpha, dtype=torch.float32),
            "num_bands": NUM_ROPE_BANDS,
            "num_latent_steps": (num_frames - 1) // WAN_TEMPORAL_COMPRESS + 1,
            "timestep_decay": timestep_decay,
            "per_band_positions": True,
        }
        return (patched,)
