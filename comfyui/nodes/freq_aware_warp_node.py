import math

import torch
import numpy as np

from spf.solver import invert_potential


# ---------------------------------------------------------------------------
# Ported from video_potential — self-contained, no external dependencies
# ---------------------------------------------------------------------------

def invert_potential_to_positions(
    potential: np.ndarray,
    num_output_frames: int = None,
    reference_type: str = "linear",
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute warped temporal positions by inverting the potential curve.

    Extends ``spf.solver.invert_potential`` with non-linear reference curves.
    """
    potential = np.asarray(potential).flatten()
    num_input_frames = len(potential)
    if num_output_frames is None:
        num_output_frames = num_input_frames

    p_min, p_max = potential.min(), potential.max()
    if p_max - p_min < 1e-8:
        return np.linspace(0, num_input_frames - 1, num_output_frames)

    potential_norm = (potential - p_min) / (p_max - p_min)
    frame_indices = np.arange(num_input_frames, dtype=np.float64)

    t = np.linspace(0, 1, num_output_frames)
    if reference_type == "linear":
        target_progress = t
    elif reference_type == "exp_up":
        exp_curve = np.exp(temperature * t) - 1
        target_progress = exp_curve / (np.exp(temperature) - 1)
    elif reference_type == "exp_down":
        exp_curve = np.exp(temperature) - np.exp(temperature * (1 - t))
        target_progress = exp_curve / (np.exp(temperature) - 1)
    elif reference_type == "sin_identity":
        target_progress = np.sin(2 * np.pi * t) / (2 * np.pi) + t
    else:
        raise ValueError(f"Unknown reference_type: {reference_type}")

    return np.interp(target_progress, potential_norm, frame_indices)


def compute_decay_curve(
    timestep_decay: str,
    decay_start: float = 0.0,
    decay_end: float = 1.0,
    num_points: int = 100,
) -> tuple:
    """Compute the full timestep decay curve for visualization."""
    timesteps = np.linspace(1000, 0, num_points)
    t_normalized = timesteps / 1000.0

    curves = {
        "none": lambda t: np.ones_like(t),
        "linear_up": lambda t: t,
        "linear_down": lambda t: 1.0 - t,
        "cosine_up": lambda t: 0.5 * (1 + np.cos(np.pi * (1 - t))),
        "cosine_down": lambda t: 0.5 * (1 + np.cos(np.pi * t)),
        "exp_up": lambda t: (np.exp(3 * t) - 1) / (np.exp(3) - 1),
        "exp_down": lambda t: (np.exp(3 * (1 - t)) - 1) / (np.exp(3) - 1),
        "window": lambda t: np.where(
            (t >= decay_start) & (t <= decay_end), 1.0, 0.0
        ),
    }
    decay_mult = curves.get(timestep_decay, curves["none"])(t_normalized)
    return timesteps, decay_mult


def _create_warp_visualization(
    potential: np.ndarray,
    alpha_schedule: np.ndarray,
    linear_positions: np.ndarray,
    warped_positions: np.ndarray,
    schedule_mode: str,
    alpha_high: float,
    alpha_mid: float,
    alpha_low: float,
    timestep_decay: str = "none",
    decay_start: float = 0.0,
    decay_end: float = 1.0,
    reference_type: str = "linear",
    reference_temperature: float = 3.0,
    width: int = 1400,
    height: int = 700
) -> torch.Tensor:
    """Create visualization plot for frequency-aware warping."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    
    num_frames = len(potential)
    NUM_BANDS = len(alpha_schedule)
    
    # Generate reference curve based on reference_type
    frame_indices = np.arange(num_frames)
    if reference_type == "linear":
        reference_curve = np.linspace(0, 1, num_frames)
        ref_label = "Linear reference"
    elif reference_type == "exp_up":
        # Exponential: slow start, fast end (gradually increasing deviation)
        t = np.linspace(0, 1, num_frames)
        exp_curve = np.exp(reference_temperature * t) - 1
        reference_curve = exp_curve / (np.exp(reference_temperature) - 1)
        ref_label = f"Exp Up reference (T={reference_temperature:.1f}, slow→fast)"
    elif reference_type == "exp_down":
        # Exponential: fast start, slow end (starts with deviation, then slows)
        t = np.linspace(0, 1, num_frames)
        exp_curve = np.exp(reference_temperature) - np.exp(reference_temperature * (1 - t))
        reference_curve = exp_curve / (np.exp(reference_temperature) - 1)
        ref_label = f"Exp Down reference (T={reference_temperature:.1f}, fast→slow)"
    elif reference_type == "sin_identity":
        # Sinusoidal + identity: sin(2πx)/(2π) + x
        t = np.linspace(0, 1, num_frames)
        reference_curve = np.sin(2 * np.pi * t) / (2 * np.pi) + t
        ref_label = "Sin+Identity reference (sin(2πx)/(2π)+x)"
    else:
        reference_curve = np.linspace(0, 1, num_frames)
        ref_label = f"{reference_type} reference"
    
    # Create figure with 5 subplots (2 rows, 3 columns)
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # 1. Potential curve (top-left)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(frame_indices, potential, 'g-', linewidth=2, label='Potential')
    ax1.plot(frame_indices, reference_curve, 'k--', alpha=0.5, linewidth=1.5, label=ref_label)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Perceptual Progress')
    ax1.set_title(f'1. Potential Curve (vs {reference_type})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Alpha schedule (top-middle)
    ax2 = fig.add_subplot(2, 3, 2)
    band_indices = np.arange(NUM_BANDS)
    colors = plt.cm.coolwarm(alpha_schedule)

    # For many bands (LTX=341), use line plot instead of bar for clarity
    if NUM_BANDS > 50:
        ax2.fill_between(band_indices, 0, alpha_schedule, alpha=0.3, color='blue')
        ax2.plot(band_indices, alpha_schedule, 'b-', linewidth=1.5)
    else:
        ax2.bar(band_indices, alpha_schedule, color=colors, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=alpha_high, color='r', linestyle='--', alpha=0.5, label=f'High ({alpha_high:.2f})')
    ax2.axhline(y=alpha_mid, color='orange', linestyle='--', alpha=0.5, label=f'Mid ({alpha_mid:.2f})')
    ax2.axhline(y=alpha_low, color='b', linestyle='--', alpha=0.5, label=f'Low ({alpha_low:.2f})')

    # Determine frequency ordering based on model type
    # LTX (341 bands): band 0 = LOW freq, band 340 = HIGH freq
    # Wan (22 bands): band 0 = HIGH freq, band 21 = LOW freq
    if NUM_BANDS > 50:  # LTX
        freq_label = f'Frequency Band (0=LOW freq → {NUM_BANDS-1}=HIGH freq)'
    else:  # Wan
        freq_label = f'Frequency Band (0=HIGH freq → {NUM_BANDS-1}=LOW freq)'

    ax2.set_xlabel(freq_label)
    ax2.set_ylabel('Alpha (warp strength)')
    ax2.set_title(f'2. Alpha Schedule ({schedule_mode}, {NUM_BANDS} bands)')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Timestep Decay (top-right) - NEW
    ax3 = fig.add_subplot(2, 3, 3)
    timesteps, decay_mult = compute_decay_curve(timestep_decay, decay_start, decay_end)
    
    # Fill area under curve
    ax3.fill_between(timesteps, 0, decay_mult, alpha=0.3, color='green')
    ax3.plot(timesteps, decay_mult, 'green', linewidth=2.5)
    
    # Add reference lines and labels
    ax3.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add text annotations
    ax3.text(950, 1.05, '1 = FULL WARPING', fontsize=9, color='blue', fontweight='bold', ha='left')
    ax3.text(950, -0.08, '0 = NO WARPING (original)', fontsize=9, color='red', fontweight='bold', ha='left')
    
    # Mark denoising direction
    ax3.annotate('', xy=(100, 0.5), xytext=(900, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax3.text(500, 0.55, 'Denoising direction →', fontsize=9, ha='center', color='gray')
    
    ax3.set_xlabel('Diffusion Timestep (1000=noisy → 0=clean)')
    ax3.set_ylabel('Warp Multiplier')
    ax3.set_title(f'3. Timestep Decay: {timestep_decay}')
    ax3.set_xlim(1050, -50)  # Reversed so 1000 is on left
    ax3.set_ylim(-0.15, 1.2)
    ax3.grid(True, alpha=0.3)
    
    # Add decay range indicator for window mode
    if timestep_decay == "window":
        ax3.axvline(x=decay_start * 1000, color='orange', linestyle=':', linewidth=2, label=f'Start: {decay_start:.2f}')
        ax3.axvline(x=decay_end * 1000, color='orange', linestyle=':', linewidth=2, label=f'End: {decay_end:.2f}')
        ax3.legend(fontsize=8)
    
    # 4. Per-band blended positions (bottom-left)
    ax4 = fig.add_subplot(2, 3, 4)
    frame_indices = np.arange(num_frames)
    ax4.plot(frame_indices, linear_positions, 'b--', label='Linear (original)', alpha=0.5, linewidth=1)

    # Dynamic band indices based on NUM_BANDS
    # For LTX (341 bands): band 0 = LOW freq (high alpha), band 340 = HIGH freq (low alpha)
    # For Wan (22 bands): band 0 = HIGH freq (high alpha), band 21 = LOW freq (low alpha)
    if NUM_BANDS > 50:  # LTX
        # Band 0 gets highest alpha (low freq), band 340 gets lowest alpha (high freq)
        band_low_freq = 0  # LOW frequency, HIGH alpha
        band_mid_freq = NUM_BANDS // 2  # MID frequency
        band_high_freq = NUM_BANDS - 1  # HIGH frequency, LOW alpha
        freq_labels = [
            (band_low_freq, 'red', f'LOW freq band {band_low_freq} (α={alpha_schedule[band_low_freq]:.2f})'),
            (band_mid_freq, 'orange', f'MID freq band {band_mid_freq} (α={alpha_schedule[band_mid_freq]:.2f})'),
            (band_high_freq, 'blue', f'HIGH freq band {band_high_freq} (α={alpha_schedule[band_high_freq]:.2f})')
        ]
    else:  # Wan
        # Band 0 = HIGH freq (high alpha), band 21 = LOW freq (low alpha)
        band_high_freq = 2  # HIGH frequency (near band 0)
        band_mid_freq = NUM_BANDS // 2
        band_low_freq = NUM_BANDS - 2  # LOW frequency (near band 21)
        freq_labels = [
            (band_high_freq, 'red', f'HIGH freq band {band_high_freq} (α={alpha_schedule[band_high_freq]:.2f})'),
            (band_mid_freq, 'orange', f'MID freq band {band_mid_freq} (α={alpha_schedule[band_mid_freq]:.2f})'),
            (band_low_freq, 'blue', f'LOW freq band {band_low_freq} (α={alpha_schedule[band_low_freq]:.2f})')
        ]

    for band_idx, color, label in freq_labels:
        alpha_val = alpha_schedule[band_idx]
        blended = (1 - alpha_val) * linear_positions + alpha_val * warped_positions
        ax4.plot(frame_indices, blended, color=color, label=label, linewidth=1.5)

    ax4.set_xlabel('Output Frame Index')
    ax4.set_ylabel('Temporal Position')
    ax4.set_title('4. Per-Band Blended Positions')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # 5. Effective warping per position (bottom-middle)
    ax5 = fig.add_subplot(2, 3, 5)
    warp_diff = warped_positions - linear_positions
    ax5.fill_between(frame_indices, 0, warp_diff, alpha=0.3, color='purple')
    ax5.plot(frame_indices, warp_diff, 'purple', linewidth=2, label='Warp offset (warped - linear)')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Frame Index')
    ax5.set_ylabel('Position Offset')
    ax5.set_title('5. Warp Effect (+stretch, -compress)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary info (bottom-right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Determine model type for summary
    if NUM_BANDS > 50:
        model_type = "LTX"
        freq_order = "Band 0=LOW freq → Band 340=HIGH freq"
    else:
        model_type = "Wan"
        freq_order = "Band 0=HIGH freq → Band 21=LOW freq"

    summary_text = f"""FREQUENCY-AWARE WARP SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: {model_type} ({NUM_BANDS} temporal bands)
Freq ordering: {freq_order}

Alpha Values (per frequency band):
  • α = 0.0: NO WARPING (original linear)
  • α = 1.0: FULL WARPING (warped positions)
  • α > 1.0: EXTRAPOLATE (more aggressive)

  Formula: pos = (1-α)×linear + α×warped

Timestep Decay Multiplier:
  • mult = 1.0: Full alpha applied
  • mult = 0.0: Alpha zeroed (no warping)

Current Settings:
  • Schedule: {schedule_mode}
  • Timestep decay: {timestep_decay}
  • Frames: {num_frames}
  • Alpha: [{alpha_schedule.min():.3f}, {alpha_schedule.max():.3f}]
"""
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Convert to tensor
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
    
    return img_tensor


def resample_positions(positions: np.ndarray, target_len: int) -> np.ndarray:
    """Resample positions array to target length using linear interpolation (legacy uniform)."""
    if len(positions) == target_len:
        return positions
    
    orig_len = len(positions)
    target_to_source = np.linspace(0, orig_len - 1, target_len)
    floor_idx = np.floor(target_to_source).astype(int).clip(0, orig_len - 2)
    ceil_idx = (floor_idx + 1).clip(0, orig_len - 1)
    frac = target_to_source - floor_idx
    return positions[floor_idx] * (1 - frac) + positions[ceil_idx] * frac


def _get_wan_frame_indices(num_latents: int, num_frames: int) -> np.ndarray:
    """
    Get the frame indices that correspond to each latent step for Wan VAE.
    
    Wan VAE temporal compression:
    - Latent 0: encodes frame 0 only (center = 0)
    - Latent i (i >= 1): encodes frames [4*(i-1)+1, 4*i], center = 4*i - 1.5
    
    Args:
        num_latents: Number of latent temporal steps
        num_frames: Number of video frames
    
    Returns:
        Array of frame indices (possibly fractional) for each latent
    """
    frame_indices = np.zeros(num_latents)
    frame_indices[0] = 0.0
    for i in range(1, num_latents):
        frame_indices[i] = 4 * i - 1.5
    
    # Clip to valid range
    frame_indices = np.clip(frame_indices, 0, num_frames - 1)
    return frame_indices


def resample_positions_wan_sample(positions: np.ndarray, num_latents: int) -> np.ndarray:
    """
    Resample positions from video frame space to latent space using SAMPLING at centers.
    
    Wan VAE temporal compression:
    - Latent 0: samples position at frame 0
    - Latent i (i >= 1): samples position at frame center (4*i - 1.5)
    
    Args:
        positions: Warped positions in video frame space [num_frames]
        num_latents: Number of latent temporal steps
    
    Returns:
        Resampled positions for each latent step
    """
    num_frames = len(positions)
    
    if num_latents == num_frames:
        return positions
    
    frame_indices = _get_wan_frame_indices(num_latents, num_frames)
    
    # Linear interpolation at these frame indices
    floor_idx = np.floor(frame_indices).astype(int).clip(0, num_frames - 2)
    ceil_idx = (floor_idx + 1).clip(0, num_frames - 1)
    frac = frame_indices - floor_idx
    
    return positions[floor_idx] * (1 - frac) + positions[ceil_idx] * frac


def resample_positions_wan_average(positions: np.ndarray, num_latents: int) -> np.ndarray:
    """
    Resample positions from video frame space to latent space using AVERAGING.
    
    Wan VAE temporal compression:
    - Latent 0: just frame 0
    - Latent i (i >= 1): average of frames [4*(i-1)+1, 4*i]
    
    Args:
        positions: Warped positions in video frame space [num_frames]
        num_latents: Number of latent temporal steps
    
    Returns:
        Resampled positions for each latent step (averaged over source frames)
    """
    num_frames = len(positions)
    
    if num_latents == num_frames:
        return positions
    
    result = np.zeros(num_latents)
    
    # Latent 0: just frame 0
    result[0] = positions[0]
    
    # Latent i (i >= 1): average of frames [4*(i-1)+1, 4*i]
    for i in range(1, num_latents):
        start_frame = 4 * (i - 1) + 1
        end_frame = min(4 * i + 1, num_frames)  # exclusive end
        result[i] = positions[start_frame:end_frame].mean()
    
    return result


def resample_to_latent(
    positions: np.ndarray, 
    num_latents: int, 
    aggregation: str = "sample"
) -> np.ndarray:
    """
    Resample positions from video frame space to latent space.
    
    NOTE: This function resamples the position VALUES but they remain in VIDEO FRAME space.
    To convert to latent coordinate space, multiply by scale_factor = (num_latents-1)/(num_frames-1).
    
    Args:
        positions: Positions in video frame space [num_frames]
        num_latents: Number of latent temporal steps
        aggregation: "sample" (interpolate at centers) or "average" (mean over contributing frames)
    
    Returns:
        Resampled positions [num_latents] (still in video frame space, just resampled)
    """
    if aggregation == "average":
        return resample_positions_wan_average(positions, num_latents)
    else:  # "sample" or default
        return resample_positions_wan_sample(positions, num_latents)


def validate_base_positions_video_frame_space(
    base_positions: np.ndarray,
    num_output_frames: int,
    tolerance: float = 0.1
) -> bool:
    """
    Validate that base_positions are in VIDEO FRAME space.
    
    Expected range: approximately [0, num_output_frames-1]
    
    Args:
        base_positions: Positions to validate (1D or 2D)
        num_output_frames: Expected number of output frames
        tolerance: Allowed deviation from expected range (as fraction)
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If positions are clearly in wrong coordinate space
    """
    if base_positions.ndim == 1:
        positions_to_check = base_positions
    elif base_positions.ndim == 2:
        positions_to_check = base_positions.flatten()
    else:
        raise ValueError(f"base_positions must be 1D or 2D, got {base_positions.ndim}D")
    
    min_pos = positions_to_check.min()
    max_pos = positions_to_check.max()
    expected_max = num_output_frames - 1
    
    # Check if values are way too large (likely in wrong space)
    if max_pos > expected_max * (1 + tolerance) * 2:
        raise ValueError(
            f"base_positions appear to be in wrong coordinate space! "
            f"Expected max ~{expected_max:.1f} (video frame space), got max={max_pos:.1f}. "
            f"Values suggest latent space or incorrect scaling."
        )
    
    # Check if values are way too small (likely in wrong space)
    if max_pos < expected_max * 0.1 and num_output_frames > 10:
        raise ValueError(
            f"base_positions appear to be in wrong coordinate space! "
            f"Expected max ~{expected_max:.1f} (video frame space), got max={max_pos:.1f}. "
            f"Values suggest latent space or incorrect scaling."
        )
    
    # Warn if values are slightly out of expected range (might be from previous iteration)
    if max_pos > expected_max * (1 + tolerance):
        print(f"[FrequencyAwareWarp] WARNING: base_positions max={max_pos:.1f} exceeds expected range [0, {expected_max:.1f}]")
        print(f"  This may be from a previous iteration with different num_output_frames. Resampling will adjust.")
    
    if min_pos < -tolerance * expected_max:
        print(f"[FrequencyAwareWarp] WARNING: base_positions min={min_pos:.1f} is negative (expected >= 0)")
    
    return True


def create_alpha_schedule(
    alpha_high: float,
    alpha_mid: float, 
    alpha_low: float,
    num_bands: int = 22,
    mode: str = "3_bands"
) -> np.ndarray:
    """
    Create per-frequency-band alpha schedule.
    
    Args:
        alpha_high: Alpha for high frequencies (bands 0-7, local motion)
        alpha_mid: Alpha for mid frequencies (bands 8-14)
        alpha_low: Alpha for low frequencies (bands 15-21, global structure)
        num_bands: Total number of frequency bands (22 for Wan temporal)
        mode: Schedule mode - normal modes warp high freq more, _inv modes warp low freq more
    
    Returns:
        alpha_schedule: Array of shape (num_bands,) with alpha per band
    """
    if mode == "3_bands":
        # Hard boundaries between 3 groups: high freq → high alpha
        schedule = np.zeros(num_bands)
        high_end = num_bands // 3        # 0-7
        mid_end = 2 * num_bands // 3     # 8-14
        schedule[:high_end] = alpha_high
        schedule[high_end:mid_end] = alpha_mid
        schedule[mid_end:] = alpha_low
        
    elif mode == "3_bands_inv":
        # Inverse: low freq → high alpha (warp global structure more)
        schedule = np.zeros(num_bands)
        high_end = num_bands // 3
        mid_end = 2 * num_bands // 3
        schedule[:high_end] = alpha_low      # high freq gets LOW alpha
        schedule[high_end:mid_end] = alpha_mid
        schedule[mid_end:] = alpha_high      # low freq gets HIGH alpha
        
    elif mode == "linear_decay":
        # Linear: high alpha at high freq, low alpha at low freq
        schedule = np.linspace(alpha_high, alpha_low, num_bands)
        
    elif mode == "linear_decay_inv":
        # Inverse linear: low alpha at high freq, high alpha at low freq
        schedule = np.linspace(alpha_low, alpha_high, num_bands)
        
    elif mode == "smooth_decay":
        # Smooth S-curve: high alpha at high freq
        t = np.linspace(0, 1, num_bands)
        smooth_t = 1 / (1 + np.exp(-10 * (t - 0.5)))
        schedule = alpha_high * (1 - smooth_t) + alpha_low * smooth_t
        
    elif mode == "smooth_decay_inv":
        # Inverse S-curve: high alpha at low freq
        t = np.linspace(0, 1, num_bands)
        smooth_t = 1 / (1 + np.exp(-10 * (t - 0.5)))
        schedule = alpha_low * (1 - smooth_t) + alpha_high * smooth_t
        
    elif mode == "exp_decay":
        # Exponential decay: fast drop from high to low
        t = np.linspace(0, 1, num_bands)
        schedule = alpha_low + (alpha_high - alpha_low) * np.exp(-3 * t)
        
    elif mode == "exp_decay_inv":
        # Inverse exponential: fast rise from low to high
        t = np.linspace(0, 1, num_bands)
        schedule = alpha_high - (alpha_high - alpha_low) * np.exp(-3 * t)
    
    elif mode == "flat":
        # Uniform alpha across all bands (equivalent to LinearizeTemporalPositions)
        schedule = np.full(num_bands, alpha_high)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return schedule


class FrequencyAwareWarp:
    """
    Apply frequency-aware temporal warping with different alphas per frequency band.
    
    ALPHA VALUES (range 0.0 to 2.0):
    - α = 0.0: NO WARPING (use original linear positions)
    - α = 1.0: FULL WARPING (use warped positions from potential inversion)
    - α > 1.0: EXTRAPOLATE beyond warped (more aggressive warping)
    - Formula: position = (1-α) × base + α × warped
    
    FREQUENCY BANDS (22 total for Wan temporal axis):
    - High frequencies (bands 0-7): Local/frame-to-frame motion
    - Mid frequencies (bands 8-14): Short sequence patterns  
    - Low frequencies (bands 15-21): Global video structure
    
    ITERATIVE WARPING:
    For iterative refinement, connect the 'warped_positions' output to the 
    'base_positions' input of the next iteration.
    
    TIMESTEP DECAY:
    Varies warping strength during denoising (multiplier applied to alpha):
    - mult = 1.0: Full alpha applied (full warping)
    - mult = 0.0: Alpha zeroed (no warping, original positions)
    
    Decay modes:
    - none: Constant warping throughout (mult=1 always)
    - linear_up: More warping at high timesteps (early denoising, establishing structure)
    - linear_down: More warping at low timesteps (late denoising, detail refinement)
    - window: Full warping only between decay_start and decay_end
    
    METHODS:
    - position_blend: Averages alpha across bands, single RoPE computation (faster)
    - rope_blend: Per-band position blending in angle-space, mathematically correct
                  rotation interpolation (SLERP). Recommended for best results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "potential": ("POTENTIAL",),
            },
            "optional": {
                "base_positions": ("WARP_POSITIONS", {
                    "tooltip": "For iterative warping: connect warped_positions from previous iteration. Leave empty for first iteration."
                }),
                "iteration": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Iteration number (for logging)."
                }),
                "num_output_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Number of output frames. 0 = same as potential length"
                }),
                "method": (["position_blend", "rope_blend"], {
                    "default": "rope_blend",
                    "tooltip": "position_blend: avg alpha, faster. rope_blend: per-band blending, mathematically correct (recommended)"
                }),
                "schedule_mode": (["flat", "3_bands", "3_bands_inv", "linear_decay", "linear_decay_inv", "smooth_decay", "smooth_decay_inv", "exp_decay", "exp_decay_inv"], {
                    "default": "flat",
                    "tooltip": "flat: uniform alpha. Normal: warp high freq more. _inv: warp low freq more (global structure)"
                }),
                "alpha_high": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.00001,
                    "tooltip": "Warping strength for high frequencies. 0=linear, 1=warped, >1=extrapolate beyond warped"
                }),
                "alpha_mid": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.00001,
                    "tooltip": "Warping strength for mid frequencies. 0=linear, 1=warped, >1=extrapolate"
                }),
                "alpha_low": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.00001,
                    "tooltip": "Warping strength for low frequencies. 0=linear, 1=warped, >1=extrapolate"
                }),
                "custom_schedule": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom per-band alphas: '1.0,0.9,0.8,...' (22 values). Overrides other alpha settings."
                }),
                "timestep_decay": (["none", "linear_up", "linear_down", "cosine_up", "cosine_down", "exp_up", "exp_down", "window"], {
                    "default": "none",
                    "tooltip": "Vary warping strength during denoising. none=constant. up=more warp early (high noise). down=more warp late (details phase). window=warp only in specified range."
                }),
                "decay_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.00001,
                    "tooltip": "For 'window' mode: normalized timestep where warping STARTS. 0.0=t1000 (noisy start), 1.0=t0 (clean end). Default 0.0 includes all timesteps."
                }),
                "decay_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.00001,
                    "tooltip": "For 'window' mode: normalized timestep where warping ENDS. Default 1.0 includes all timesteps. Example: start=0.3, end=0.7 = warp only middle 40% of denoising."
                }),
                "reference_type": (["linear", "exp_up", "exp_down", "sin_identity"], {
                    "default": "linear",
                    "tooltip": "Reference curve to compare potential against. linear: uniform progression. exp_up: exponential (slow start, fast end). exp_down: exponential (fast start, slow end). sin_identity: sinusoidal oscillation on top of linear (sin(2πx)/(2π)+x)."
                }),
                "reference_temperature": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Temperature for exponential curves. Higher = more radical deviation from linear. 1.0 = mild, 3.0 = moderate, 5.0+ = very radical. Only affects exp_up/exp_down."
                }),
                "frame_to_latent_aggregation": (["sample", "average"], {
                    "default": "sample",
                    "tooltip": "How to aggregate frame-level positions to latent-level. 'sample': interpolate at latent centers (latent 0=frame 0, latent i=frame 4i-1.5). 'average': mean over contributing frames (latent 0=frame 0, latent i=mean of frames 4(i-1)+1 to 4i)."
                }),
                "correction_space": (["latent", "frame"], {
                    "default": "latent",
                    "tooltip": "'latent': Resample positions first, then compute correction (default, backward compatible). 'frame': Compute correction in frame space (endpoints pinned to 0), then resample. Recommended for F2F conditioning."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "WARP_POSITIONS", "IMAGE", "INT")
    RETURN_NAMES = ("model", "warped_positions", "visualization", "next_iteration")
    FUNCTION = "apply"
    CATEGORY = "video-potential"

    def apply(
        self,
        model,
        potential,
        base_positions=None,
        iteration=0,
        num_output_frames=0,
        method="position_blend",
        schedule_mode="3_bands",
        alpha_high=1.0,
        alpha_mid=0.5,
        alpha_low=0.1,
        custom_schedule="",
        timestep_decay="none",
        decay_start=0.0,
        decay_end=1.0,
        reference_type="linear",
        reference_temperature=3.0,
        frame_to_latent_aggregation="sample",
        correction_space="latent"
    ):
        # Auto-detect iteration from base_positions
        effective_iteration = iteration
        if base_positions is not None and iteration == 0:
            effective_iteration = 1
        
        # Prepare shared frequency-aware data (now returns per-band output positions)
        freq_aware_data, alpha_schedule, linear_positions, warped_positions, potential_arr, output_positions_per_band = _prepare_freq_aware_data(
            potential=potential,
            num_output_frames=num_output_frames,
            method=method,
            schedule_mode=schedule_mode,
            alpha_high=alpha_high,
            alpha_mid=alpha_mid,
            alpha_low=alpha_low,
            custom_schedule=custom_schedule,
            timestep_decay=timestep_decay,
            decay_start=decay_start,
            decay_end=decay_end,
            reference_type=reference_type,
            reference_temperature=reference_temperature,
            base_positions=base_positions,
            iteration=effective_iteration,
            frame_to_latent_aggregation=frame_to_latent_aggregation,
            correction_space=correction_space,
        )
        
        print(f"[FrequencyAwareWarp] Iteration {effective_iteration} - PER-BAND ADDITIVE (aggregation={frame_to_latent_aggregation})")
        print(f"  Alpha range: [{alpha_schedule.min():.3f}, {alpha_schedule.max():.3f}]")
        print(f"  Output positions per band: shape={output_positions_per_band.shape}")
        
        # Apply to model
        model = _apply_freq_aware_warp_to_model(model, freq_aware_data, "model")
        
        # Generate visualization
        visualization = _create_warp_visualization(
            potential=potential_arr,
            alpha_schedule=alpha_schedule,
            linear_positions=linear_positions,
            warped_positions=warped_positions,
            schedule_mode=schedule_mode,
            alpha_high=alpha_high,
            alpha_mid=alpha_mid,
            alpha_low=alpha_low,
            timestep_decay=timestep_decay,
            decay_start=decay_start,
            decay_end=decay_end,
            reference_type=reference_type,
            reference_temperature=reference_temperature,
        )
        
        # Output as 2D list [22, num_frames] for next iteration
        warped_positions_output = output_positions_per_band.tolist()
        next_iteration = effective_iteration + 1
        
        return (model, warped_positions_output, visualization, next_iteration)


def _apply_freq_aware_warp_to_model(
    model,
    freq_aware_data: dict,
    model_name: str = "model"
):
    """
    Apply frequency-aware warp data to a single model.
    
    Args:
        model: The model to modify
        freq_aware_data: The frequency-aware warp configuration dict
        model_name: Name for logging
    
    Returns:
        Modified model clone
    """
    model = model.clone()
    
    if "transformer_options" not in model.model_options:
        model.model_options["transformer_options"] = {}
    if "rope_options" not in model.model_options["transformer_options"]:
        model.model_options["transformer_options"]["rope_options"] = {}
    
    model.model_options["transformer_options"]["rope_options"]["freq_aware_warp"] = freq_aware_data
    
    print(f"[FrequencyAwareWarp] Applied to {model_name}")
    
    return model


def _prepare_freq_aware_data(
    potential,
    num_output_frames: int,
    method: str,
    schedule_mode: str,
    alpha_high: float,
    alpha_mid: float,
    alpha_low: float,
    custom_schedule: str,
    timestep_decay: str,
    decay_start: float,
    decay_end: float,
    reference_type: str = "linear",
    reference_temperature: float = 3.0,
    base_positions: np.ndarray = None,
    iteration: int = 0,
    frame_to_latent_aggregation: str = "sample",
    correction_space: str = "latent",
) -> tuple:
    """
    Prepare frequency-aware warp data structure with PER-BAND position tracking.
    
    ITERATIVE WARPING (Additive Mode with Per-Band Tracking):
    Each band accumulates its own position history:
    
        correction = invert(potential) - linear    # What this iteration suggests
        For each band i:
            new_positions[i] = base[i] + α[i] × correction
    
    This tracks 22 separate position trajectories:
    - Iteration 0: positions[i] = linear + α[i] × correction₀
    - Iteration 1: positions[i] = prev[i] + α[i] × correction₁  
    - Each band accumulates differently based on its alpha!
    
    CORRECTION SPACE:
    - "latent": Resample positions to latent space, then compute correction.
               May have non-zero correction at endpoints due to Wan VAE center mapping.
    - "frame": Compute correction in frame space (where endpoints are exactly pinned),
              then resample correction to latent space. Recommended for F2F conditioning
              as it ensures first/last frames have zero correction.
    
    Args:
        potential: The potential curve for this iteration
        base_positions: Optional base positions from previous iteration.
                       - If None: uses linear positions (iteration 0)
                       - If 1D [num_frames]: legacy mode, same base for all bands
                       - If 2D [22, num_frames]: per-band bases from previous iteration
        iteration: Iteration number (for logging)
        correction_space: "latent" (default) or "frame" - where to compute correction
    
    Returns:
        tuple: (freq_aware_data dict, alpha_schedule, linear_positions, warped_positions, potential_arr, output_positions_per_band)
               Note: linear_positions is always true linear [0, 1, ..., N-1]
                     warped_positions is the TARGET from inverting the potential
                     output_positions_per_band is 2D [22, num_frames] for next iteration
    """
    potential_arr = np.array(potential)
    
    if num_output_frames <= 0:
        num_output_frames = len(potential_arr)
    
    NUM_TEMPORAL_BANDS = 22
    
    # Create alpha schedule
    if custom_schedule.strip():
        alpha_schedule = np.array([
            float(x.strip()) for x in custom_schedule.split(",") if x.strip()
        ])
        if len(alpha_schedule) != NUM_TEMPORAL_BANDS:
            raise ValueError(f"Custom schedule must have {NUM_TEMPORAL_BANDS} values, got {len(alpha_schedule)}")
    else:
        alpha_schedule = create_alpha_schedule(
            alpha_high, alpha_mid, alpha_low,
            num_bands=NUM_TEMPORAL_BANDS,
            mode=schedule_mode
        )
    
    # TRUE linear positions (always the reference)
    linear_positions = np.linspace(0, num_output_frames - 1, num_output_frames)
    
    # Compute warped positions from THIS iteration's potential
    warped_positions = invert_potential_to_positions(
        potential_arr, num_output_frames, 
        reference_type=reference_type,
        temperature=reference_temperature
    )
    
    # Compute the CORRECTION this iteration suggests
    # correction = warped - linear (how much to deviate from linear)
    correction = warped_positions - linear_positions
    
    # Determine base positions for additive warping
    # base_positions_per_band: shape [22, num_frames]
    if base_positions is not None:
        base_positions = np.array(base_positions)
        
        # Validate coordinate space: should be in VIDEO FRAME space
        try:
            validate_base_positions_video_frame_space(base_positions, num_output_frames)
        except ValueError as e:
            raise ValueError(f"[FrequencyAwareWarp] Invalid base_positions: {e}")
        
        if base_positions.ndim == 1:
            # Legacy 1D input: same base for all bands
            if len(base_positions) != num_output_frames:
                base_positions = resample_positions(base_positions, num_output_frames)
            # Broadcast to 2D: [22, num_frames]
            base_positions_per_band = np.tile(base_positions, (NUM_TEMPORAL_BANDS, 1))
            print(f"[FrequencyAwareWarp] ITERATIVE MODE (iteration {iteration}) - 1D base (legacy)")
            print(f"  Base positions: [{base_positions.min():.3f}, {base_positions.max():.3f}]")
        
        elif base_positions.ndim == 2:
            # Per-band 2D input: [22, num_frames]
            if base_positions.shape[0] != NUM_TEMPORAL_BANDS:
                raise ValueError(f"Expected {NUM_TEMPORAL_BANDS} bands, got {base_positions.shape[0]}")
            if base_positions.shape[1] != num_output_frames:
                # Resample each band
                resampled = np.zeros((NUM_TEMPORAL_BANDS, num_output_frames))
                for i in range(NUM_TEMPORAL_BANDS):
                    resampled[i] = resample_positions(base_positions[i], num_output_frames)
                base_positions_per_band = resampled
            else:
                base_positions_per_band = base_positions
            print(f"[FrequencyAwareWarp] ITERATIVE MODE (iteration {iteration}) - PER-BAND bases")
            print(f"  Band 0 (high): [{base_positions_per_band[0].min():.3f}, {base_positions_per_band[0].max():.3f}]")
            print(f"  Band 11 (mid): [{base_positions_per_band[11].min():.3f}, {base_positions_per_band[11].max():.3f}]")
            print(f"  Band 21 (low): [{base_positions_per_band[21].min():.3f}, {base_positions_per_band[21].max():.3f}]")
        else:
            raise ValueError(f"base_positions must be 1D or 2D, got {base_positions.ndim}D")
        
        print(f"  Correction range: [{correction.min():.3f}, {correction.max():.3f}]")
        print(f"  Formula: new[i] = base[i] + α[i] × correction")
    else:
        # First iteration: base is linear for all bands
        base_positions_per_band = np.tile(linear_positions, (NUM_TEMPORAL_BANDS, 1))
        print(f"[FrequencyAwareWarp] INITIAL iteration (starting from linear)")
        print(f"  Correction range: [{correction.min():.3f}, {correction.max():.3f}]")
        print(f"  Formula: new[i] = linear + α[i] × correction")
    
    # Compute per-band OUTPUT positions for next iteration
    # output[i] = base[i] + alpha[i] * correction
    output_positions_per_band = np.zeros((NUM_TEMPORAL_BANDS, num_output_frames))
    for i in range(NUM_TEMPORAL_BANDS):
        output_positions_per_band[i] = base_positions_per_band[i] + alpha_schedule[i] * correction
    
    # Compute latent dimensions
    num_latent_steps = (num_output_frames - 1) // 4 + 1
    scale_factor = (num_latent_steps - 1) / (num_output_frames - 1) if num_output_frames > 1 else 1.0
    
    # ADDITIVE MODE TRANSFORMATION (PER-BAND):
    # The model's RoPE code does: (1-α[i]) * linear_data[i] + α[i] * warped_data[i]
    # We want ADDITIVE: base[i] + α[i] * (warped - linear)
    #
    # To achieve this, we transform the data we send:
    #   linear_data[i] = base[i]
    #   warped_data[i] = base[i] + correction
    #
    # Then model computes for band i:
    #   (1-α[i])*base[i] + α[i]*(base[i] + correction)
    #   = base[i] + α[i]*correction  ← ADDITIVE!
    
    if correction_space == "frame":
        # NEW APPROACH: Compute correction in FRAME space, then resample to latent
        # Benefits:
        # - Endpoints are naturally pinned (correction[0] = correction[-1] = 0)
        # - Linear case correctly gives zero correction everywhere
        # - Better for first/last frame conditioning (F2F)
        
        # Correction in frame space (endpoints are exactly 0)
        correction_frame = warped_positions - linear_positions  # [num_frames]
        
        # Resample CORRECTION (not positions) to latent space
        correction_latent = resample_to_latent(correction_frame, num_latent_steps, frame_to_latent_aggregation) * scale_factor
        
        # Base positions: resample each band's positions
        base_latent_per_band = np.zeros((NUM_TEMPORAL_BANDS, num_latent_steps))
        for i in range(NUM_TEMPORAL_BANDS):
            # For base, we need to compute the correction from linear, then apply
            base_correction_frame = base_positions_per_band[i] - linear_positions
            base_latent_per_band[i] = resample_to_latent(base_correction_frame, num_latent_steps, frame_to_latent_aggregation) * scale_factor
        
        # base_latent_per_band now contains CORRECTIONS from linear, not absolute positions
        # We need to add true_linear_latent to get absolute positions for the model
        true_linear_latent = np.linspace(0, num_latent_steps - 1, num_latent_steps)
        linear_for_model = base_latent_per_band + true_linear_latent[np.newaxis, :]  # [22, num_latent_steps]
        warped_for_model = linear_for_model + correction_latent[np.newaxis, :]  # [22, num_latent_steps]
        
        print(f"  [correction_space=frame] Correction in frame space, then resampled")
        print(f"    Frame-space correction endpoints: first={correction_frame[0]:.6f}, last={correction_frame[-1]:.6f} (should be ~0)")
        print(f"    Latent-space correction endpoints: first={correction_latent[0]:.6f}, last={correction_latent[-1]:.6f}")
        
    else:  # correction_space == "latent" (default, backward compatible)
        # ORIGINAL APPROACH: Resample positions to latent space, then compute correction
        # Note: May have non-zero correction at endpoints due to Wan VAE center mapping
        
        # True linear in latent space
        true_linear_latent = np.linspace(0, num_latent_steps - 1, num_latent_steps)
        
        # Warped positions (from potential inversion) in latent space
        # Use correct Wan mapping: latent 0 = frame 0, latent i = frame center (4i-1.5)
        warped_latent_frame = resample_to_latent(warped_positions, num_latent_steps, frame_to_latent_aggregation)
        true_warped_latent = warped_latent_frame * scale_factor
        
        # Base positions in latent space: PER-BAND [22, num_latent_steps]
        base_latent_per_band = np.zeros((NUM_TEMPORAL_BANDS, num_latent_steps))
        for i in range(NUM_TEMPORAL_BANDS):
            base_latent_frame = resample_to_latent(base_positions_per_band[i], num_latent_steps, frame_to_latent_aggregation)
            base_latent_per_band[i] = base_latent_frame * scale_factor
        
        correction_latent = true_warped_latent - true_linear_latent
        
        linear_for_model = base_latent_per_band  # [22, num_latent_steps]
        warped_for_model = base_latent_per_band + correction_latent[np.newaxis, :]  # [22, num_latent_steps]
        
        print(f"  [correction_space=latent] Resample positions, then compute correction")
        print(f"    Latent-space correction endpoints: first={correction_latent[0]:.6f}, last={correction_latent[-1]:.6f}")
    
    # Build frequency-aware data structure
    freq_aware_data = {
        "method": method,
        "linear_positions": torch.tensor(linear_for_model, dtype=torch.float32),  # [22, T]
        "warped_positions": torch.tensor(warped_for_model, dtype=torch.float32),  # [22, T]
        "alpha_schedule": torch.tensor(alpha_schedule, dtype=torch.float32),
        "num_bands": NUM_TEMPORAL_BANDS,
        "num_latent_steps": num_latent_steps,
        "timestep_decay": timestep_decay,
        "decay_start": decay_start,
        "decay_end": decay_end,
        "per_band_positions": True,  # Flag for model to know positions are 2D
    }
    
    # Print for validation
    print(f"[FrequencyAwareWarp] Method: {method}, Schedule: {schedule_mode}, Reference: {reference_type}, Temperature: {reference_temperature}")
    print(f"  Frames: {num_output_frames} → Latent steps: {num_latent_steps}")
    print(f"  Alpha schedule: high={alpha_high:.2f}, mid={alpha_mid:.2f}, low={alpha_low:.2f}")
    print(f"  Schedule values (first 5, last 5): {alpha_schedule[:5].tolist()} ... {alpha_schedule[-5:].tolist()}")
    if timestep_decay != "none":
        print(f"  Timestep decay: {timestep_decay} (range: {decay_start:.2f} - {decay_end:.2f})")
    print(f"  PER-BAND POSITIONS: linear_for_model shape={linear_for_model.shape}, warped_for_model shape={warped_for_model.shape}")
    print(f"  Correction (latent): [{correction_latent.min():.3f}, {correction_latent.max():.3f}]")
    print(f"  Output positions per band: shape={output_positions_per_band.shape}")
    print(f"    Band 0: [{output_positions_per_band[0].min():.3f}, {output_positions_per_band[0].max():.3f}]")
    print(f"    Band 21: [{output_positions_per_band[21].min():.3f}, {output_positions_per_band[21].max():.3f}]")
    
    return freq_aware_data, alpha_schedule, linear_positions, warped_positions, potential_arr, output_positions_per_band


class FrequencyAwareWarpDual:
    """
    Apply frequency-aware temporal warping to TWO models (Wan 2.2 high/low noise experts).
    
    This is a convenience wrapper for Wan 2.2 workflows that use dual models.
    Both models receive the same warping configuration.
    
    ITERATIVE WARPING:
    For iterative refinement, connect the 'warped_positions' output to the 
    'base_positions' input of the next iteration's FrequencyAwareWarpDual node.
    
    Iteration 0: base_positions = None → starts from linear positions
    Iteration N: base_positions = warped_positions from iteration N-1
    
    The blending formula becomes:
        new_positions = (1 - α) × base_positions + α × inverted(potential)
    
    This compounds the corrections across iterations.
    
    See FrequencyAwareWarp docstring for detailed explanation of:
    - Alpha values (1=full warping, 0=no warping)
    - Frequency bands (high/mid/low)
    - Timestep decay modes (none/linear_up/linear_down/window/etc.)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high_noise": ("MODEL", {"tooltip": "Wan 2.2 high noise expert model"}),
                "model_low_noise": ("MODEL", {"tooltip": "Wan 2.2 low noise expert model"}),
                "potential": ("POTENTIAL",),
            },
            "optional": {
                "base_positions": ("WARP_POSITIONS", {
                    "tooltip": "For iterative warping: connect warped_positions from previous iteration. Leave empty for first iteration (uses linear positions)."
                }),
                "iteration": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Iteration number (for logging). Auto-increments if base_positions is connected."
                }),
                "num_output_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Number of output frames. 0 = same as potential length"
                }),
                "method": (["position_blend", "rope_blend"], {
                    "default": "rope_blend",
                    "tooltip": "position_blend: avg alpha, faster. rope_blend: per-band blending, mathematically correct (recommended)"
                }),
                "schedule_mode": (["flat", "3_bands", "3_bands_inv", "linear_decay", "linear_decay_inv", "smooth_decay", "smooth_decay_inv", "exp_decay", "exp_decay_inv"], {
                    "default": "flat",
                    "tooltip": "flat: uniform alpha. Normal: warp high freq more. _inv: warp low freq more (global structure)"
                }),
                "alpha_high": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.00001,
                    "tooltip": "Warping strength for high frequencies. 0=linear, 1=warped, >1=extrapolate beyond warped"
                }),
                "alpha_mid": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.00001,
                    "tooltip": "Warping strength for mid frequencies. 0=linear, 1=warped, >1=extrapolate"
                }),
                "alpha_low": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.00001,
                    "tooltip": "Warping strength for low frequencies. 0=linear, 1=warped, >1=extrapolate"
                }),
                "custom_schedule": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom per-band alphas: '1.0,0.9,0.8,...' (22 values). Overrides other alpha settings."
                }),
                "timestep_decay": (["none", "linear_up", "linear_down", "cosine_up", "cosine_down", "exp_up", "exp_down", "window"], {
                    "default": "none",
                    "tooltip": "Vary warping strength during denoising. none=constant. up=more warp early (high noise). down=more warp late (details phase). window=warp only in specified range."
                }),
                "decay_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.00001,
                    "tooltip": "For 'window' mode: normalized timestep where warping STARTS. 0.0=t1000 (noisy start), 1.0=t0 (clean end). Default 0.0 includes all timesteps."
                }),
                "decay_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.00001,
                    "tooltip": "For 'window' mode: normalized timestep where warping ENDS. Default 1.0 includes all timesteps. Example: start=0.3, end=0.7 = warp only middle 40% of denoising."
                }),
                "reference_type": (["linear", "exp_up", "exp_down", "sin_identity"], {
                    "default": "linear",
                    "tooltip": "Reference curve to compare potential against. linear: uniform progression. exp_up: exponential (slow start, fast end). exp_down: exponential (fast start, slow end). sin_identity: sinusoidal oscillation on top of linear (sin(2πx)/(2π)+x)."
                }),
                "reference_temperature": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Temperature for exponential curves. Higher = more radical deviation from linear. 1.0 = mild, 3.0 = moderate, 5.0+ = very radical. Only affects exp_up/exp_down."
                }),
                "frame_to_latent_aggregation": (["sample", "average"], {
                    "default": "sample",
                    "tooltip": "How to aggregate frame-level positions to latent-level. 'sample': interpolate at latent centers (latent 0=frame 0, latent i=frame 4i-1.5). 'average': mean over contributing frames (latent 0=frame 0, latent i=mean of frames 4(i-1)+1 to 4i)."
                }),
                "correction_space": (["latent", "frame"], {
                    "default": "latent",
                    "tooltip": "'latent': Resample positions first, then compute correction (default, backward compatible). 'frame': Compute correction in frame space (endpoints pinned to 0), then resample. Recommended for F2F conditioning."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "WARP_POSITIONS", "IMAGE", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("model_high_noise", "model_low_noise", "warped_positions", "visualization", "reference_type", "reference_temperature", "next_iteration")
    FUNCTION = "apply"
    CATEGORY = "video-potential"

    def apply(
        self,
        model_high_noise,
        model_low_noise,
        potential,
        base_positions=None,
        iteration=0,
        num_output_frames=0,
        method="rope_blend",
        schedule_mode="flat",
        alpha_high=1.0,
        alpha_mid=0.5,
        alpha_low=0.1,
        custom_schedule="",
        timestep_decay="none",
        decay_start=0.0,
        decay_end=1.0,
        reference_type="linear",
        reference_temperature=3.0,
        frame_to_latent_aggregation="sample",
        correction_space="latent"
    ):
        # Auto-detect iteration from base_positions
        effective_iteration = iteration
        if base_positions is not None and iteration == 0:
            effective_iteration = 1  # If base_positions provided but iteration=0, assume it's iteration 1
        
        # Prepare shared frequency-aware data (now returns per-band output positions)
        freq_aware_data, alpha_schedule, linear_positions, warped_positions, potential_arr, output_positions_per_band = _prepare_freq_aware_data(
            potential=potential,
            num_output_frames=num_output_frames,
            method=method,
            schedule_mode=schedule_mode,
            alpha_high=alpha_high,
            alpha_mid=alpha_mid,
            alpha_low=alpha_low,
            custom_schedule=custom_schedule,
            timestep_decay=timestep_decay,
            decay_start=decay_start,
            decay_end=decay_end,
            reference_type=reference_type,
            reference_temperature=reference_temperature,
            base_positions=base_positions,
            iteration=effective_iteration,
            frame_to_latent_aggregation=frame_to_latent_aggregation,
            correction_space=correction_space,
        )
        
        print(f"[FrequencyAwareWarpDual] Iteration {effective_iteration} - PER-BAND ADDITIVE (aggregation={frame_to_latent_aggregation})")
        print(f"  Alpha range: [{alpha_schedule.min():.3f}, {alpha_schedule.max():.3f}]")
        print(f"  Output positions per band: shape={output_positions_per_band.shape}")
        print(f"  → Pass 'warped_positions' output to next iteration's 'base_positions' input")
        
        # Apply to both models
        model_high_noise_out = _apply_freq_aware_warp_to_model(
            model_high_noise, freq_aware_data, "model_high_noise"
        )
        model_low_noise_out = _apply_freq_aware_warp_to_model(
            model_low_noise, freq_aware_data, "model_low_noise"
        )
        
        # Generate visualization
        visualization = _create_warp_visualization(
            potential=potential_arr,
            alpha_schedule=alpha_schedule,
            linear_positions=linear_positions,
            warped_positions=warped_positions,
            schedule_mode=schedule_mode,
            alpha_high=alpha_high,
            alpha_mid=alpha_mid,
            alpha_low=alpha_low,
            timestep_decay=timestep_decay,
            decay_start=decay_start,
            decay_end=decay_end,
            reference_type=reference_type,
            reference_temperature=reference_temperature,
        )
        
        # Output as 2D list [22, num_frames] for next iteration
        warped_positions_output = output_positions_per_band.tolist()
        next_iteration = effective_iteration + 1
        
        return (
            model_high_noise_out, 
            model_low_noise_out, 
            warped_positions_output,  # WARP_POSITIONS [22, num_frames] for next iteration
            visualization, 
            reference_type, 
            reference_temperature,
            next_iteration,
        )


class CreateAlphaSchedule:
    """
    Create a custom alpha schedule for frequency-aware warping.
    Useful for experimenting with different frequency response curves.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["3_bands", "linear_decay", "smooth_decay", "custom_curve"],),
            },
            "optional": {
                "alpha_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.00001}),
                "alpha_mid": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.00001}),
                "alpha_low": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.00001}),
                "curve_power": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Power for custom_curve mode: <1 = concave (more high), >1 = convex (more low)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("schedule_string",)
    FUNCTION = "create"
    CATEGORY = "video-potential"

    def create(self, mode, alpha_high=1.0, alpha_mid=0.5, alpha_low=0.1, curve_power=1.0):
        NUM_BANDS = 22
        
        if mode == "custom_curve":
            # Power curve from high to low
            t = np.linspace(0, 1, NUM_BANDS) ** curve_power
            schedule = alpha_high * (1 - t) + alpha_low * t
        else:
            schedule = create_alpha_schedule(alpha_high, alpha_mid, alpha_low, NUM_BANDS, mode)
        
        schedule_str = ",".join([f"{x:.3f}" for x in schedule])
        
        print(f"[CreateAlphaSchedule] Mode: {mode}")
        print(f"  Schedule: {schedule_str}")
        
        return (schedule_str,)



