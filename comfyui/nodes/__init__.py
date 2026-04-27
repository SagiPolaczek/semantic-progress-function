"""ComfyUI custom nodes for SPF / ReTime."""

from .potential_node import ComputeSPF
from .retime_node import ReTimeWan
from .compute_potential_node import ComputeVideoPotential
from .freq_aware_warp_node import (
    FrequencyAwareWarp,
    FrequencyAwareWarpDual,
    CreateAlphaSchedule,
)
from .compare_potentials_node import ComparePotentials

NODE_CLASS_MAPPINGS = {
    "ComputeSPF": ComputeSPF,
    "ReTimeWan": ReTimeWan,
    "ComputeVideoPotential": ComputeVideoPotential,
    "FrequencyAwareWarp": FrequencyAwareWarp,
    "FrequencyAwareWarpDual": FrequencyAwareWarpDual,
    "CreateAlphaSchedule": CreateAlphaSchedule,
    "ComparePotentials": ComparePotentials,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComputeSPF": "Compute SPF",
    "ReTimeWan": "ReTime (Wan2.2)",
    "ComputeVideoPotential": "Compute Video Potential",
    "FrequencyAwareWarp": "Frequency-Aware Warp (Advanced)",
    "FrequencyAwareWarpDual": "Frequency-Aware Warp Dual (Wan 2.2)",
    "CreateAlphaSchedule": "Create Alpha Schedule",
    "ComparePotentials": "Compare Video Potentials",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
