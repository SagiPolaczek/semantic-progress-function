"""Run SPF on example videos and save visualisations."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spf import SigLIPEmbedder, SolveDirect, LineSegmentor
from spf import plot_potential, plot_potential_with_segments, plot_spf_report
from spf import get_video_params, frame_generator

# -- Config --
INPUT_DIR = Path(__file__).parent / "inputs"
VIDEOS = {
    "cat_to_lion": INPUT_DIR / "cat_to_lion.mp4",
    "crane_to_origami": INPUT_DIR / "crane_to_origami.mp4",
    "turntable_linear": INPUT_DIR / "turntable_LIN.mp4",
}
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# -- Pipeline --
embedder = SigLIPEmbedder()  # paper default for ReTime
solver = SolveDirect()  # paper defaults: k=30, sigma=20, lmd=1e-5

for name, video_path in VIDEOS.items():
    print(f"\n{'='*60}")
    print(f"  {name}: {video_path}")
    params = get_video_params(video_path)
    print(f"  {params.num_frames} frames, {params.width}x{params.height} @ {params.fps}fps")
    print(f"{'='*60}")

    # Read frames (kept in memory for the frame strip)
    frames = list(frame_generator(video_path))

    # 1. Embed
    embeddings = embedder.embed_video(frames)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 2. Solve SPF
    potential = solver(embeddings)
    print(f"  Potential range: [{potential.min():.3f}, {potential.max():.3f}]")

    # 3. Segment
    seg = LineSegmentor().fit_k(potential, k=4)

    # 4. Individual plots
    plot_potential(
        potential,
        title=f"SPF \u2014 {name}",
        save_path=OUT_DIR / f"{name}_potential.png",
    )
    print(f"  Saved: {name}_potential.png")

    plot_potential_with_segments(
        potential, seg,
        title=f"SPF with segments \u2014 {name}",
        save_path=OUT_DIR / f"{name}_segments.png",
    )
    print(f"  Saved: {name}_segments.png  ({seg.num_segments} segments)")

    # 5. Combined report with frame strip
    plot_spf_report(
        potential, seg,
        frames=frames,
        title=f"SPF Analysis \u2014 {name}",
        save_path=OUT_DIR / f"{name}_report.png",
    )
    print(f"  Saved: {name}_report.png")

print(f"\nAll outputs in: {OUT_DIR}")
