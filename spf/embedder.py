"""Frame embedding extraction using OpenCLIP / SigLIP."""

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
import open_clip

from .video_io import frame_generator


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _BaseEmbedder:
    """Shared embedding logic for all vision-encoder backends."""

    device: str
    model: torch.nn.Module
    preprocess: object  # callable transform

    def _setup(self, model: torch.nn.Module, preprocess):
        """Move model to device and freeze weights."""
        self.device = _get_device()
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.preprocess = preprocess

    def embed_frame(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image. Returns a normalised 1-D vector."""
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def embed_video(self, source: Union[str, Path, List[Image.Image]]) -> np.ndarray:
        """Embed all frames of a video.

        Args:
            source: Path (or string path) to a video file, or a list of PIL images.

        Returns:
            Normalised embeddings of shape ``(N, D)``.
        """
        if isinstance(source, (str, Path)):
            frames = frame_generator(source)
        else:
            frames = source
        return np.stack([self.embed_frame(img) for img in frames])


class SigLIPEmbedder(_BaseEmbedder):
    """SigLIP embedder — default in the paper for ReTime.

    Args:
        model_name: HuggingFace Hub identifier for a SigLIP model loadable
            via ``open_clip.create_model_from_pretrained``.
    """

    def __init__(self, model_name: str = "hf-hub:timm/ViT-B-16-SigLIP2-512"):
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        self._setup(model, preprocess)


class OpenCLIPEmbedder(_BaseEmbedder):
    """OpenCLIP (ViT-B-32) embedder.

    Args:
        model_name: OpenCLIP model architecture.
        pretrained: Pretrained weights identifier.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._setup(model, preprocess)
