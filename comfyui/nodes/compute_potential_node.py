"""ComfyUI node: compute SPF from video frames."""

import torch
import numpy as np
from PIL import Image

from spf import OpenCLIPEmbedder, SigLIPEmbedder, SolveDirect


EMBEDDER_CLASSES = {
    "OpenCLIP": OpenCLIPEmbedder,
    "SigLIP": SigLIPEmbedder,
}

SOLVER_CLASSES = {
    "Direct": SolveDirect,
}

# Cache for loaded embedders
_embedder_cache = {}


def get_embedder(embedder_type: str):
    """Get or create cached embedder instance."""
    if embedder_type not in _embedder_cache:
        _embedder_cache[embedder_type] = EMBEDDER_CLASSES[embedder_type]()
    return _embedder_cache[embedder_type]


def tensor_to_pil_list(video_tensor: torch.Tensor):
    """Convert ComfyUI IMAGE tensor (B, H, W, C) in [0,1] to list of PIL Images."""
    images = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i].cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(frame))
    return images


class ComputeVideoPotential:
    """Compute perceptual potential from video frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "embedder_type": (list(EMBEDDER_CLASSES.keys()), {"default": "SigLIP"}),
                "solver_type": (list(SOLVER_CLASSES.keys()), {"default": "Direct"}),
                "k": ("INT", {"default": 30, "min": 1, "max": 50}),
                "sigma": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0}),
                "lmd": ("FLOAT", {"default": 1e-5, "min": 0.0, "max": 1.0, "step": 1e-6}),
                "metric": (["angular"], {"default": "angular"}),
                "normalize": ("BOOLEAN", {"default": True}),
                "p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("POTENTIAL", "EMBEDDINGS", "VARIANCE", "FLOAT")
    RETURN_NAMES = ("potential", "embeddings", "variance", "mean_variance")
    FUNCTION = "compute"
    CATEGORY = "video-potential"

    def compute(self, video_frames, embedder_type, solver_type, k, sigma, lmd, metric, normalize, p):
        pil_images = tensor_to_pil_list(video_frames)

        embedder = get_embedder(embedder_type)
        embeddings = embedder.embed_video(pil_images)

        solver = SOLVER_CLASSES[solver_type](k=k, sigma=sigma, lmd=lmd, normalize=normalize, p=p)
        potential = solver(embeddings)

        return (potential, embeddings, None, 0.0)
