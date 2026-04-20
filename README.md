# ComfyUI-LTXVideo-AVSplit

Minimal custom-node pack with only three AV nodes extracted from `ComfyUI-LTXVideo`.

Included nodes:
- `LTXVSetAudioVideoMaskByTimeSplit`
- `LTXVStitchAVLatentsWithTransitionMask`
- `LTXVPostBlendTransition`

## Installation

1. Place this folder inside `ComfyUI/custom_nodes/`.
2. Install dependencies from `requirements.txt` if needed.
3. Restart ComfyUI.

## Notes

- This pack keeps original node IDs and behavior.
- Do not run this pack together with the original `ComfyUI-LTXVideo` pack if both expose the same IDs.
