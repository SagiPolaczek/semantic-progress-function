"""ComfyUI node: compute SPF from video frames."""

import numpy as np
from PIL import Image

from spf import OpenCLIPEmbedder, SigLIPEmbedder, SolveDirect

# Cache embedders to avoid reloading models on every execution
_EMBEDDER_CACHE = {}


def _get_embedder(name: str):
    if name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[name] = SigLIPEmbedder() if name == "SigLIP" else OpenCLIPEmbedder()
    return _EMBEDDER_CACHE[name]


class ComputeSPF:
    """Compute the Semantic Potential Function from video frames."""

    CATEGORY = "ReTime"
    RETURN_TYPES = ("POTENTIAL",)
    RETURN_NAMES = ("potential",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "embedder": (["SigLIP", "OpenCLIP"],),
                "k": ("INT", {"default": 30, "min": 1, "max": 100}),
                "sigma": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 200.0}),
                "lmd": ("FLOAT", {"default": 1e-5, "min": 0.0, "max": 1.0, "step": 1e-6}),
                "p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
        }

    def run(self, video_frames, embedder, k, sigma, lmd, p):
        frames = [
            Image.fromarray((video_frames[i].cpu().numpy() * 255).astype(np.uint8))
            for i in range(video_frames.shape[0])
        ]

        embeddings = _get_embedder(embedder).embed_video(frames)
        potential = SolveDirect(k=k, sigma=sigma, lmd=lmd, p=p)(embeddings)

        return (potential,)
