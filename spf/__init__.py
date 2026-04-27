"""SPF — Semantic Progress Function for video."""

from .embedder import SigLIPEmbedder, OpenCLIPEmbedder
from .solver import SolveDirect, invert_potential
from .segmentation import LineSegmentor
from .video_io import VideoParams, frame_generator, get_video_params
from .vis import plot_potential, plot_potential_with_segments, plot_spf_report
from .retime import (
    build_alpha_schedule,
    timestep_decay_multiplier,
    frame_to_latent_positions,
    compute_warped_rope_freqs,
    compute_retime_freqs,
    create_patched_rope_apply,
    retime_context,
    extract_temporal_omega,
    generate_retimed_f2lf,
    NUM_ROPE_BANDS,
    WAN_TEMPORAL_COMPRESS,
)

__version__ = "0.1.0"
