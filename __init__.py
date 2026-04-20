from .audio_video_mask_split import (
    LTXVPostBlendTransition,
    LTXVSetAudioVideoMaskByTimeSplit,
    LTXVStitchAVLatentsWithTransitionMask,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, camel_case_to_spaces
from .two_stage_resolution import TwoStageResolution


NODE_CLASS_MAPPINGS = {
    "LTXVSetAudioVideoMaskByTimeSplit": LTXVSetAudioVideoMaskByTimeSplit,
    "LTXVStitchAVLatentsWithTransitionMask": LTXVStitchAVLatentsWithTransitionMask,
    "LTXVPostBlendTransition": LTXVPostBlendTransition,
    "TwoStageResolution": TwoStageResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    name: f"{NODES_DISPLAY_NAME_PREFIX} {camel_case_to_spaces(name)}"
    for name in NODE_CLASS_MAPPINGS.keys()
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
