from .audio_video_mask_split import (
    LTXVSetAudioVideoMaskByTimeSplit,
    LTXVStitchAVLatentsWithTransitionMask,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX
from .two_stage_resolution import TwoStageResolution


NODE_CLASS_MAPPINGS = {
    "LTXVSetAudioVideoMaskByTimeSplit": LTXVSetAudioVideoMaskByTimeSplit,
    "LTXVStitchAVLatentsWithTransitionMask": LTXVStitchAVLatentsWithTransitionMask,
    "TwoStageResolution": TwoStageResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVSetAudioVideoMaskByTimeSplit": (
        f"{NODES_DISPLAY_NAME_PREFIX} Set Audio Video Mask By Time Split"
    ),
    "LTXVStitchAVLatentsWithTransitionMask": (
        f"{NODES_DISPLAY_NAME_PREFIX} Stitch AV Latents"
    ),
    "TwoStageResolution": f"{NODES_DISPLAY_NAME_PREFIX} Two Stage Resolution",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
