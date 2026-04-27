import io
import torch
import numpy as np
from PIL import Image
from typing import Optional, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def tensor_to_frame_strip(video_tensor: torch.Tensor, num_samples: int = 10) -> np.ndarray:
    """Extract evenly spaced frames and create a horizontal strip."""
    num_frames = video_tensor.shape[0]
    num_samples = min(num_samples, num_frames)
    sample_indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
    
    # Extract sampled frames
    frames = []
    for idx in sample_indices:
        frame = video_tensor[idx].cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        # Resize to consistent height
        pil_img = Image.fromarray(frame)
        aspect = pil_img.width / pil_img.height
        new_height = 100
        new_width = int(new_height * aspect)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        frames.append(np.array(pil_img))
    
    # Concatenate horizontally
    strip = np.concatenate(frames, axis=1)
    return strip, sample_indices


def plot_potentials(ax, potentials: List[np.ndarray], labels: List[str], normalize: bool,
                    variances: Optional[List[Optional[np.ndarray]]] = None, variance_scale: float = 10.0):
    """Plot potential curves on axes with optional variance envelopes."""
    colors = plt.cm.tab10.colors
    for i, (potential, label) in enumerate(zip(potentials, labels)):
        color = colors[i % len(colors)]
        x = np.linspace(0, 1, len(potential)) if normalize else np.arange(len(potential))
        
        # Plot variance envelope if provided
        if variances is not None and i < len(variances) and variances[i] is not None:
            var = variances[i]
            std = np.sqrt(var) if var.min() >= 0 else var
            lower = np.clip(potential - variance_scale * std, 0.0, 1.0)
            upper = np.clip(potential + variance_scale * std, 0.0, 1.0)
            ax.fill_between(x, lower, upper, alpha=0.2, color=color)
        
        ax.plot(x, potential, ".-", label=label, color=color, alpha=0.8)

    ax.set_xlabel("Normalized Position" if normalize else "Frame Index")
    ax.set_ylabel("Potential Value")
    ax.set_title("Video Perceptual Potential Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def add_frame_strip(fig, gs, row_idx: int, sprite_img: np.ndarray, 
                    sample_indices: np.ndarray, label: str):
    """Add a frame strip to the figure."""
    ax = fig.add_subplot(gs[row_idx])
    ax.imshow(sprite_img)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)

    h, w = sprite_img.shape[:2]
    centers = (np.arange(len(sample_indices)) + 0.5) * (w / len(sample_indices))
    ax.set_xticks(centers)
    ax.set_xticklabels([str(int(idx)) for idx in sample_indices], fontsize=7)
    ax.tick_params(axis="x", bottom=False, top=False, labelbottom=True)

    for spine in ax.spines.values():
        spine.set_visible(False)


class ComparePotentials:
    """Compare potential curves from up to four videos with frame strip visualization."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames_1": ("IMAGE",),
                "potential_1": ("POTENTIAL",),
                "label_1": ("STRING", {"default": "Video 1"}),
                "normalize_x": ("BOOLEAN", {"default": False}),
                "num_frame_samples": ("INT", {"default": 10, "min": 3, "max": 20}),
            },
            "optional": {
                "video_frames_2": ("IMAGE",),
                "potential_2": ("POTENTIAL",),
                "label_2": ("STRING", {"default": "Video 2"}),
                "video_frames_3": ("IMAGE",),
                "potential_3": ("POTENTIAL",),
                "label_3": ("STRING", {"default": "Video 3"}),
                "video_frames_4": ("IMAGE",),
                "potential_4": ("POTENTIAL",),
                "label_4": ("STRING", {"default": "Video 4"}),
                "variance_1": ("VARIANCE",),
                "variance_2": ("VARIANCE",),
                "variance_3": ("VARIANCE",),
                "variance_4": ("VARIANCE",),
                "variance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison_plot",)
    FUNCTION = "compare"
    CATEGORY = "video-potential"

    def compare(self, video_frames_1: torch.Tensor, potential_1: np.ndarray, 
                label_1: str, normalize_x: bool, num_frame_samples: int,
                video_frames_2: Optional[torch.Tensor] = None,
                potential_2: Optional[np.ndarray] = None,
                label_2: str = "Video 2",
                video_frames_3: Optional[torch.Tensor] = None,
                potential_3: Optional[np.ndarray] = None,
                label_3: str = "Video 3",
                video_frames_4: Optional[torch.Tensor] = None,
                potential_4: Optional[np.ndarray] = None,
                label_4: str = "Video 4",
                variance_1: Optional[np.ndarray] = None,
                variance_2: Optional[np.ndarray] = None,
                variance_3: Optional[np.ndarray] = None,
                variance_4: Optional[np.ndarray] = None,
                variance_scale: float = 10.0):
        
        # Collect data
        videos = [video_frames_1]
        potentials = [potential_1]
        labels = [label_1]
        variances = [variance_1]
        
        if video_frames_2 is not None and potential_2 is not None:
            videos.append(video_frames_2)
            potentials.append(potential_2)
            labels.append(label_2)
            variances.append(variance_2)
        
        if video_frames_3 is not None and potential_3 is not None:
            videos.append(video_frames_3)
            potentials.append(potential_3)
            labels.append(label_3)
            variances.append(variance_3)
        
        if video_frames_4 is not None and potential_4 is not None:
            videos.append(video_frames_4)
            potentials.append(potential_4)
            labels.append(label_4)
            variances.append(variance_4)
        
        # Only use variances if at least one is provided
        if all(v is None for v in variances):
            variances = None
        
        num_videos = len(videos)
        
        # Extract frame strips
        strips = []
        indices_list = []
        for video in videos:
            strip, indices = tensor_to_frame_strip(video, num_frame_samples)
            strips.append(strip)
            indices_list.append(indices)
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(14, 5 + 1.5 * num_videos))
        height_ratios = [4] + [1] * num_videos
        gs = gridspec.GridSpec(1 + num_videos, 1, height_ratios=height_ratios, hspace=0.3)
        
        # Plot potentials - make the plot narrower than the frame strips
        ax_plot = fig.add_subplot(gs[0])
        plot_potentials(ax_plot, potentials, labels, normalize_x, variances, variance_scale)
        # Shrink plot width while keeping it centered
        box = ax_plot.get_position()
        ax_plot.set_position([box.x0 + box.width * 0.15, box.y0, box.width * 0.7, box.height])
        
        # Add frame strips
        for i, (strip, indices, label) in enumerate(zip(strips, indices_list, labels)):
            add_frame_strip(fig, gs, 1 + i, strip, indices, label)
        
        # Convert figure to tensor
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        pil_img = Image.open(buf).convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W, C)
        
        return (img_tensor,)










