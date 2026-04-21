🅛🅣🅧 LTXV Stitch AV Latents With Traimport math
import numpy as np
import torch
from comfy.ldm.lightricks.vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from comfy.nested_tensor import NestedTensor

from .nodes_registry import comfy_node

# LTX-2 video temporal stride in latent space (matches typical VAE); used if vae is not wired.
DEFAULT_LTX_VIDEO_TEMPORAL_STRIDE = 8
# Typical audio latent rate for LTX audio VAE (~25 Hz); used if audio_vae is not wired.
DEFAULT_LTX_AUDIO_LATENTS_PER_SECOND = 25.0


def _audio_latents_per_second(audio_vae) -> float:
    sr = audio_vae.autoencoder.sampling_rate
    hop = audio_vae.autoencoder.mel_hop_length
    return float(sr / hop / LATENT_DOWNSAMPLE_FACTOR)


def _clamp_interval(
    start_idx: int, end_idx_exclusive: int, length: int
) -> tuple[int, int]:
    start_idx = max(0, min(length, int(start_idx)))
    end_idx_exclusive = max(0, min(length, int(end_idx_exclusive)))
    return start_idx, end_idx_exclusive


def _build_temporal_envelope(
    length: int,
    start_idx: int,
    end_idx_exclusive: int,
    slope_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    envelope = torch.zeros(length, device=device, dtype=dtype)
    if length <= 0:
        return envelope

    start_idx, end_idx_exclusive = _clamp_interval(start_idx, end_idx_exclusive, length)
    if end_idx_exclusive <= start_idx:
        return envelope

    slope_len = max(1, int(slope_len))

    ramp_left_start = max(0, start_idx - slope_len)
    if start_idx > ramp_left_start:
        left_len = start_idx - ramp_left_start
        envelope[ramp_left_start:start_idx] = (
            torch.arange(1, left_len + 1, device=device, dtype=dtype) / float(slope_len)
        )

    envelope[start_idx:end_idx_exclusive] = 1.0

    ramp_right_end = min(length, end_idx_exclusive + slope_len)
    if ramp_right_end > end_idx_exclusive:
        right_len = ramp_right_end - end_idx_exclusive
        envelope[end_idx_exclusive:ramp_right_end] = 1.0 - (
            torch.arange(1, right_len + 1, device=device, dtype=dtype)
            / float(slope_len)
        )

    return torch.clamp(envelope, 0.0, 1.0)


def _time_to_video_latent_indices(
    start_time: float,
    end_time: float,
    video_fps: float,
    time_scale_factor: int,
    video_latent_frame_count: int,
) -> tuple[int, int]:
    # Keep conversion compatible with LTXVSetAudioVideoMaskByTime.
    video_pixel_frame_count = (video_latent_frame_count - 1) * time_scale_factor + 1
    xp = np.array(
        [0]
        + list(range(1, video_pixel_frame_count + time_scale_factor, time_scale_factor))
    )

    video_pixel_frame_start_raw = int(round(start_time * video_fps))
    video_latent_frame_index_start = int(
        np.searchsorted(xp, video_pixel_frame_start_raw, side="left")
    )

    video_pixel_frame_end_raw = int(round(end_time * video_fps))
    # Same end index as LTXVSetAudioVideoMaskByTime (used as Python slice end there).
    video_latent_frame_index_end_exclusive = int(
        np.searchsorted(xp, video_pixel_frame_end_raw, side="right") - 1
    )

    return _clamp_interval(
        video_latent_frame_index_start,
        video_latent_frame_index_end_exclusive,
        video_latent_frame_count,
    )


def _time_to_audio_latent_indices(
    start_time: float,
    end_time: float,
    audio_latents_per_second: float,
    audio_latent_frame_count: int,
) -> tuple[int, int]:
    audio_latent_frame_index_start = int(round(start_time * audio_latents_per_second))
    # Match LTXVSetAudioVideoMaskByTime (latents.py): int(round(end * rate)) + 1.
    audio_latent_frame_index_end_exclusive = (
        int(round(end_time * audio_latents_per_second)) + 1
    )
    return _clamp_interval(
        audio_latent_frame_index_start,
        audio_latent_frame_index_end_exclusive,
        audio_latent_frame_count,
    )


def _prepare_spatial_mask(
    spatial_mask: torch.Tensor,
    frame_count: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    spatial_mask = spatial_mask.to(device=device, dtype=dtype)

    if spatial_mask.ndim == 2:
        spatial_mask = spatial_mask.unsqueeze(0)
    elif spatial_mask.ndim == 4 and spatial_mask.shape[0] == 1:
        spatial_mask = spatial_mask.squeeze(0)

    if spatial_mask.ndim != 3:
        raise ValueError("spatial_mask must be 2D (H,W) or 3D (T,H,W).")

    mask_frames = spatial_mask.shape[0]
    if mask_frames < frame_count:
        tail = spatial_mask[-1:, :, :].repeat(frame_count - mask_frames, 1, 1)
        spatial_mask = torch.cat([spatial_mask, tail], dim=0)
    elif mask_frames > frame_count:
        spatial_mask = spatial_mask[:frame_count]

    spatial_mask = torch.nn.functional.interpolate(
        spatial_mask.unsqueeze(1),
        size=(latent_height, latent_width),
        mode="bilinear",
        align_corners=False,
    )
    return spatial_mask.permute(1, 0, 2, 3).unsqueeze(0)


def _validate_av_latent_shapes(
    video_samples_1: torch.Tensor,
    audio_samples_1: torch.Tensor,
    video_samples_2: torch.Tensor,
    audio_samples_2: torch.Tensor,
) -> None:
    if video_samples_1.ndim != 5 or video_samples_2.ndim != 5:
        raise ValueError(
            "Both video latent tensors must be 5D: [batch, channels, frames, height, width]."
        )

    if audio_samples_1.ndim != 4 or audio_samples_2.ndim != 4:
        raise ValueError(
            "Both audio latent tensors must be 4D: [batch, channels, frames, feature_dim]."
        )

    if (
        video_samples_1.shape[0] != video_samples_2.shape[0]
        or video_samples_1.shape[1] != video_samples_2.shape[1]
        or video_samples_1.shape[3] != video_samples_2.shape[3]
        or video_samples_1.shape[4] != video_samples_2.shape[4]
    ):
        raise ValueError(
            "Video latent dimensions must match on batch/channels/height/width. "
            f"Got {video_samples_1.shape} and {video_samples_2.shape}."
        )

    if (
        audio_samples_1.shape[0] != audio_samples_2.shape[0]
        or audio_samples_1.shape[1] != audio_samples_2.shape[1]
        or audio_samples_1.shape[3] != audio_samples_2.shape[3]
    ):
        raise ValueError(
            "Audio latent dimensions must match on batch/channels/feature_dim. "
            f"Got {audio_samples_1.shape} and {audio_samples_2.shape}."
        )


def _build_linear_bridge(
    clip_1: torch.Tensor,
    clip_2: torch.Tensor,
    bridge_latent_frames: int,
) -> torch.Tensor:
    if bridge_latent_frames <= 0:
        shape = list(clip_1.shape)
        shape[2] = 0
        return torch.empty(shape, device=clip_1.device, dtype=clip_1.dtype)

    alpha = torch.linspace(
        0.0,
        1.0,
        steps=bridge_latent_frames + 2,
        device=clip_1.device,
        dtype=clip_1.dtype,
    )[1:-1]
    alpha_shape = [1] * clip_1.ndim
    alpha_shape[2] = bridge_latent_frames
    alpha = alpha.view(*alpha_shape)

    start = clip_1[:, :, -1:, ...]
    end = clip_2[:, :, :1, ...]

    return start * (1.0 - alpha) + end * alpha


def _build_linear_overlap_video_transition(
    clip_1: torch.Tensor,
    clip_2: torch.Tensor,
    overlap_latent_frames: int,
) -> tuple[torch.Tensor, int]:
    overlap_eff = min(int(overlap_latent_frames), clip_1.shape[2], clip_2.shape[2])
    if overlap_eff <= 0:
        return torch.cat([clip_1, clip_2], dim=2), 0

    alpha = torch.linspace(
        0.0,
        1.0,
        steps=overlap_eff + 2,
        device=clip_1.device,
        dtype=clip_1.dtype,
    )[1:-1]
    alpha_shape = [1] * clip_1.ndim
    alpha_shape[2] = overlap_eff
    alpha = alpha.view(*alpha_shape)

    left_tail = clip_1[:, :, -overlap_eff:, ...]
    right_head = clip_2[:, :, :overlap_eff, ...]
    blended_overlap = left_tail * (1.0 - alpha) + right_head * alpha

    stitched = torch.cat(
        [
            clip_1[:, :, :-overlap_eff, ...],
            blended_overlap,
            clip_2[:, :, overlap_eff:, ...],
        ],
        dim=2,
    )
    return stitched, overlap_eff


def _compute_audio_bridge_latent_frames(
    bridge_latent_frames: int,
    time_scale_factor: int,
    video_fps: float,
    audio_latents_per_second: float,
) -> int:
    bridge_latent_frames = int(bridge_latent_frames)
    if bridge_latent_frames <= 0:
        return 0

    if video_fps <= 0:
        raise ValueError("video_fps must be greater than 0")

    bridge_duration_seconds = (
        bridge_latent_frames * float(time_scale_factor) / float(video_fps)
    )
    return max(1, int(math.ceil(bridge_duration_seconds * audio_latents_per_second)))


def _build_temporal_window_with_external_slope(
    length: int,
    start_idx: int,
    end_idx_exclusive: int,
    slope_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Keep the entire window [start_idx, end_idx_exclusive) at 1.0 and apply
    # slope only outside that window.
    if slope_len <= 0:
        envelope = torch.zeros(length, device=device, dtype=dtype)
        if length <= 0:
            return envelope
        start_idx, end_idx_exclusive = _clamp_interval(
            start_idx, end_idx_exclusive, length
        )
        if end_idx_exclusive <= start_idx:
            return envelope
        envelope[start_idx:end_idx_exclusive] = 1.0
        return envelope

    return _build_temporal_envelope(
        length=length,
        start_idx=start_idx,
        end_idx_exclusive=end_idx_exclusive,
        slope_len=slope_len,
        device=device,
        dtype=dtype,
    )


@comfy_node(description="LTXV Mask By Time Split")
class LTXVSetAudioVideoMaskByTimeSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 500.0}),
                "video_start_time": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2000.0},
                ),
                "video_end_time": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 2000.0},
                ),
                "video_slope_len": (
                    "INT",
                    {"default": 3, "min": 1, "max": 100, "step": 1},
                ),
                "mask_video": ("BOOLEAN", {"default": True}),
                "mask_init_value_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0},
                ),
                "audio_start_time": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2000.0},
                ),
                "audio_end_time": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 2000.0},
                ),
                "audio_slope_len": (
                    "INT",
                    {"default": 3, "min": 1, "max": 100, "step": 1},
                ),
                "mask_audio": ("BOOLEAN", {"default": True}),
                "mask_init_value_audio": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0},
                ),
            },
            "optional": {
                "spatial_mask": (
                    "MASK",
                    {
                        "default": None,
                        "tooltip": (
                            "Optional (T,H,W) or (H,W) mask; blended with temporal envelope "
                            "for video (not a hard replace like LTXVSetAudioVideoMaskByTime)."
                        ),
                    },
                ),
                "merge_existing_video_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If the latent already has noise_mask, multiply the new video mask "
                            "by the existing per-frame scalar (same behavior as "
                            "LTXVSetAudioVideoMaskByTime for video only)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "run"
    CATEGORY = "lightricks/LTXV"
    DESCRIPTION = (
        "Audio/video noise masks with separate time windows and temporal ramps (latent space). "
        "Differs from LTXVSetAudioVideoMaskByTime: no conditioning outputs; optional merge of "
        "prior video mask; spatial mask is blended with the envelope, not pasted."
    )

    def run(
        self,
        av_latent,
        model,
        vae,
        audio_vae,
        video_fps,
        video_start_time,
        video_end_time,
        video_slope_len,
        mask_video,
        mask_init_value_video,
        audio_start_time,
        audio_end_time,
        audio_slope_len,
        mask_audio,
        mask_init_value_audio,
        spatial_mask=None,
        merge_existing_video_mask=False,
    ):
        from comfy.ldm.lightricks.av_model import LTXAVModel

        if model.model.diffusion_model.__class__.__name__ != "LTXAVModel":
            raise ValueError("model must use LTXAVModel diffusion model")

        if not isinstance(av_latent["samples"], NestedTensor):
            raise ValueError("av_latent must contain a NestedTensor in 'samples'")

        ltxav: LTXAVModel = model.model.diffusion_model
        video_samples, audio_samples = ltxav.separate_audio_and_video_latents(
            av_latent["samples"].tensors,
            None,
        )

        video_mask = torch.full(
            video_samples.shape,
            fill_value=float(mask_init_value_video),
            dtype=video_samples.dtype,
            device=video_samples.device,
        )
        audio_mask = torch.full(
            audio_samples.shape,
            fill_value=float(mask_init_value_audio),
            dtype=audio_samples.dtype,
            device=audio_samples.device,
        )

        audio_latents_per_second = _audio_latents_per_second(audio_vae)
        time_scale_factor = vae.downscale_index_formula[0]

        video_frame_count = video_samples.shape[2]
        audio_frame_count = audio_samples.shape[2]

        video_start_idx, video_end_idx_exclusive = _time_to_video_latent_indices(
            video_start_time,
            video_end_time,
            video_fps,
            time_scale_factor,
            video_frame_count,
        )
        audio_start_idx, audio_end_idx_exclusive = _time_to_audio_latent_indices(
            audio_start_time,
            audio_end_time,
            audio_latents_per_second,
            audio_frame_count,
        )

        if mask_video:
            video_envelope = _build_temporal_envelope(
                video_frame_count,
                video_start_idx,
                video_end_idx_exclusive,
                video_slope_len,
                video_mask.device,
                video_mask.dtype,
            ).view(1, 1, video_frame_count, 1, 1)

            if spatial_mask is not None:
                spatial_video_mask = _prepare_spatial_mask(
                    spatial_mask,
                    frame_count=video_frame_count,
                    latent_height=video_samples.shape[3],
                    latent_width=video_samples.shape[4],
                    device=video_mask.device,
                    dtype=video_mask.dtype,
                )
                target_video_mask = float(mask_init_value_video) + video_envelope * (
                    spatial_video_mask - float(mask_init_value_video)
                )
            else:
                target_video_mask = float(mask_init_value_video) + video_envelope * (
                    1.0 - float(mask_init_value_video)
                )

            video_mask = target_video_mask.expand_as(video_mask).clone()

        if mask_audio:
            audio_envelope = _build_temporal_envelope(
                audio_frame_count,
                audio_start_idx,
                audio_end_idx_exclusive,
                audio_slope_len,
                audio_mask.device,
                audio_mask.dtype,
            ).view(1, 1, audio_frame_count, 1)

            target_audio_mask = float(mask_init_value_audio) + audio_envelope * (
                1.0 - float(mask_init_value_audio)
            )
            audio_mask = target_audio_mask.expand_as(audio_mask).clone()

        if merge_existing_video_mask and av_latent.get("noise_mask") is not None:
            nm = av_latent["noise_mask"]
            if isinstance(nm, NestedTensor) and len(nm.tensors) > 0:
                base_mask = nm.tensors[0].clone()
                if (
                    base_mask.shape[0]
                    == base_mask.shape[1]
                    == 1
                    == base_mask.shape[3]
                    == base_mask.shape[4]
                ):
                    n_frames = min(base_mask.shape[2], video_mask.shape[2])
                    for frame in range(n_frames):
                        video_mask[:, :, frame, :, :] *= base_mask[0, 0, frame, 0, 0]

        output_latent = av_latent.copy()
        output_latent["noise_mask"] = NestedTensor(
            ltxav.recombine_audio_and_video_latents(
                torch.clamp(video_mask, 0.0, 1.0),
                torch.clamp(audio_mask, 0.0, 1.0),
            )
        )
        return (output_latent,)


@comfy_node(description="LTXV Stitch AV Latents")
class LTXVStitchAVLatentsWithTransitionMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent_1": ("LATENT",),
                "av_latent_2": ("LATENT",),
                "model": ("MODEL",),
                "video_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.0,
                        "max": 500.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": (
                            "Decoded video FPS. Each video latent step spans "
                            "(temporal_stride / fps) seconds; optional vae/audio_vae override stride and audio rate."
                        ),
                    },
                ),
                "bridge_latent_frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Bridge length in video latent frames when stitch_mode='bridge'. "
                            "In stitch_mode='overlap_linear_video' it does not affect video; "
                            "it only controls audio bridge duration."
                        ),
                    },
                ),
                "stitch_mode": (
                    ["bridge", "overlap_linear_video"],
                    {
                        "default": "bridge",
                        "tooltip": (
                            "How to stitch video latents: "
                            "'bridge' inserts transition latents between clips (existing behavior); "
                            "'overlap_linear_video' linearly blends overlapping tail/head and replaces overlap. "
                            "Audio always uses bridge stitching."
                        ),
                    },
                ),
                "overlap_latent_frames": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Overlap size for stitch_mode='overlap_linear_video', in video latent frames. "
                            "Ignored when stitch_mode='bridge'."
                        ),
                    },
                ),
                "video_pre_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Video mask expansion before the transition region, in latent frames "
                            "(used when mask_video=True)."
                        ),
                    },
                ),
                "video_post_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Video mask expansion after the transition region, in latent frames "
                            "(used when mask_video=True)."
                        ),
                    },
                ),
                "video_slope_len": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": (
                            "Video mask slope length around the transition window, in latent frames "
                            "(used when mask_video=True)."
                        ),
                    },
                ),
                "mask_video": ("BOOLEAN", {"default": True}),
                "video_mask_init_value": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01},
                ),
                "audio_start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2000.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Audio mask start time in seconds (used when mask_audio=True).",
                    },
                ),
                "audio_end_time": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 2000.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Audio mask end time in seconds (used when mask_audio=True).",
                    },
                ),
                "audio_slope_len": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Audio mask slope length in latent frames (used when mask_audio=True).",
                    },
                ),
                "mask_audio": ("BOOLEAN", {"default": True}),
                "audio_mask_init_value": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01},
                ),
                "post_blend": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Post-blend mode: no noise_mask during sampling (uniform denoise). "
                            "Use LTXVPostBlendTransition node after sampling to blend originals back. "
                            "Eliminates noise-level discontinuity at mask boundaries. "
                            "Works in both stitch modes."
                        ),
                    },
                ),
                "bridge_init_mode": (
                    ["lerp", "noise", "zeros"],
                    {
                        "default": "lerp",
                        "tooltip": (
                            "How to initialize bridge latents. "
                            "For video this is used only when stitch_mode='bridge'. "
                            "In stitch_mode='overlap_linear_video' it affects only audio bridge initialization. "
                            "'lerp': linear interpolation between endpoints (original). "
                            "'noise': random gaussian noise (model generates from scratch). "
                            "'zeros': zero-filled (neutral latent)."
                        ),
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "default": None,
                        "tooltip": (
                            "If set, video temporal stride = downscale_index_formula[0]. "
                            "If omitted, uses default 8 (LTX-2)."
                        ),
                    },
                ),
                "audio_vae": (
                    "VAE",
                    {
                        "default": None,
                        "tooltip": (
                            "If set, audio latent steps/sec from the VAE. "
                            "If omitted, uses ~25 (LTX-2). Wire both for exact sync if your VAE differs."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "run"
    CATEGORY = "lightricks/LTXV"
    DESCRIPTION = (
        "Stitches two AV latents in two video modes: bridge insertion or linear overlap replacement. "
        "Audio stitching always keeps bridge behavior. "
        "vae/audio_vae optional (defaults: stride 8, ~25 audio latents/s). "
        "Time-based audio mask indices are clamped to actual latent length."
    )

    def run(
        self,
        av_latent_1,
        av_latent_2,
        model,
        video_fps,
        bridge_latent_frames,
        stitch_mode,
        overlap_latent_frames,
        video_pre_frames,
        video_post_frames,
        video_slope_len,
        mask_video,
        video_mask_init_value,
        audio_start_time,
        audio_end_time,
        audio_slope_len,
        mask_audio,
        audio_mask_init_value,
        post_blend=False,
        bridge_init_mode="lerp",
        vae=None,
        audio_vae=None,
    ):
        from comfy.ldm.lightricks.av_model import LTXAVModel

        if model.model.diffusion_model.__class__.__name__ != "LTXAVModel":
            raise ValueError("model must use LTXAVModel diffusion model")

        if not isinstance(av_latent_1["samples"], NestedTensor):
            raise ValueError("av_latent_1 must contain a NestedTensor in 'samples'")
        if not isinstance(av_latent_2["samples"], NestedTensor):
            raise ValueError("av_latent_2 must contain a NestedTensor in 'samples'")

        ltxav: LTXAVModel = model.model.diffusion_model
        video_samples_1, audio_samples_1 = ltxav.separate_audio_and_video_latents(
            av_latent_1["samples"].tensors,
            None,
        )
        video_samples_2, audio_samples_2 = ltxav.separate_audio_and_video_latents(
            av_latent_2["samples"].tensors,
            None,
        )

        video_samples_2 = video_samples_2.to(
            device=video_samples_1.device, dtype=video_samples_1.dtype
        )
        audio_samples_2 = audio_samples_2.to(
            device=audio_samples_1.device, dtype=audio_samples_1.dtype
        )

        _validate_av_latent_shapes(
            video_samples_1, audio_samples_1, video_samples_2, audio_samples_2
        )

        bridge_latent_frames = int(bridge_latent_frames)
        overlap_latent_frames = int(overlap_latent_frames)
        time_scale_factor = (
            int(vae.downscale_index_formula[0])
            if vae is not None
            else DEFAULT_LTX_VIDEO_TEMPORAL_STRIDE
        )
        audio_latents_per_second = (
            _audio_latents_per_second(audio_vae)
            if audio_vae is not None
            else DEFAULT_LTX_AUDIO_LATENTS_PER_SECOND
        )
        audio_bridge_latent_frames = _compute_audio_bridge_latent_frames(
            bridge_latent_frames,
            time_scale_factor,
            video_fps,
            audio_latents_per_second,
        )

        if bridge_init_mode == "lerp":
            audio_bridge = _build_linear_bridge(
                audio_samples_1, audio_samples_2, audio_bridge_latent_frames
            )
        elif bridge_init_mode == "noise":
            ashape = list(audio_samples_1.shape)
            ashape[2] = audio_bridge_latent_frames
            audio_bridge = torch.randn(
                ashape, device=audio_samples_1.device, dtype=audio_samples_1.dtype
            )
        else:  # zeros
            ashape = list(audio_samples_1.shape)
            ashape[2] = audio_bridge_latent_frames
            audio_bridge = torch.zeros(
                ashape, device=audio_samples_1.device, dtype=audio_samples_1.dtype
            )

        transition_start_idx: int
        transition_len: int
        overlap_eff = 0
        if stitch_mode == "bridge":
            print(
                f"[StitchDebug] overlap_latent_frames={overlap_latent_frames} "
                "is ignored in stitch_mode=bridge"
            )
            if bridge_init_mode == "lerp":
                video_bridge = _build_linear_bridge(
                    video_samples_1, video_samples_2, bridge_latent_frames
                )
            elif bridge_init_mode == "noise":
                vshape = list(video_samples_1.shape)
                vshape[2] = bridge_latent_frames
                video_bridge = torch.randn(
                    vshape, device=video_samples_1.device, dtype=video_samples_1.dtype
                )
            else:  # zeros
                vshape = list(video_samples_1.shape)
                vshape[2] = bridge_latent_frames
                video_bridge = torch.zeros(
                    vshape, device=video_samples_1.device, dtype=video_samples_1.dtype
                )
            video_concat = torch.cat([video_samples_1, video_bridge, video_samples_2], dim=2)
            transition_start_idx = video_samples_1.shape[2]
            transition_len = bridge_latent_frames
            print(
                f"[StitchDebug] stitch_mode=bridge, bridge_init_mode={bridge_init_mode}, "
                f"video_bridge shape={video_bridge.shape}, audio_bridge shape={audio_bridge.shape}"
            )
        elif stitch_mode == "overlap_linear_video":
            print(
                f"[StitchDebug] bridge_latent_frames={bridge_latent_frames} "
                "affects audio bridge only in stitch_mode=overlap_linear_video"
            )
            video_concat, overlap_eff = _build_linear_overlap_video_transition(
                video_samples_1, video_samples_2, overlap_latent_frames
            )
            transition_start_idx = video_samples_1.shape[2] - overlap_eff
            transition_len = overlap_eff
            print(
                f"[StitchDebug] stitch_mode=overlap_linear_video, overlap_req={overlap_latent_frames}, "
                f"overlap_eff={overlap_eff}, bridge_init_mode(audio_only)={bridge_init_mode}, "
                f"audio_bridge shape={audio_bridge.shape}"
            )
        else:
            raise ValueError(
                "stitch_mode must be one of: 'bridge', 'overlap_linear_video'"
            )

        audio_concat = torch.cat([audio_samples_1, audio_bridge, audio_samples_2], dim=2)

        video_mask = torch.full(
            video_concat.shape,
            fill_value=float(video_mask_init_value),
            dtype=video_concat.dtype,
            device=video_concat.device,
        )
        audio_mask = torch.full(
            audio_concat.shape,
            fill_value=float(audio_mask_init_value),
            dtype=audio_concat.dtype,
            device=audio_concat.device,
        )

        video_frame_count = video_concat.shape[2]
        audio_frame_count = audio_concat.shape[2]

        if mask_video:
            video_start_idx, video_end_idx_exclusive = _clamp_interval(
                transition_start_idx - int(video_pre_frames),
                transition_start_idx + transition_len + int(video_post_frames),
                video_frame_count,
            )

            video_envelope = _build_temporal_window_with_external_slope(
                video_frame_count,
                video_start_idx,
                video_end_idx_exclusive,
                video_slope_len,
                video_mask.device,
                video_mask.dtype,
            ).view(1, 1, video_frame_count, 1, 1)

            target_video_mask = float(video_mask_init_value) + video_envelope * (
                1.0 - float(video_mask_init_value)
            )
            video_mask = target_video_mask.expand_as(video_mask).clone()

        if mask_audio:
            audio_start_idx, audio_end_idx_exclusive = _time_to_audio_latent_indices(
                audio_start_time,
                audio_end_time,
                audio_latents_per_second,
                audio_frame_count,
            )

            audio_envelope = _build_temporal_envelope(
                audio_frame_count,
                audio_start_idx,
                audio_end_idx_exclusive,
                audio_slope_len,
                audio_mask.device,
                audio_mask.dtype,
            ).view(1, 1, audio_frame_count, 1)

            target_audio_mask = float(audio_mask_init_value) + audio_envelope * (
                1.0 - float(audio_mask_init_value)
            )
            audio_mask = target_audio_mask.expand_as(audio_mask).clone()

        # === DEBUG: print mask values ===
        _vm = video_mask[0, 0, :, 0, 0].tolist()
        print(
            f"[StitchDebug] video_concat shape: {video_concat.shape} "
            f"(mode={stitch_mode}, video1={video_samples_1.shape[2]}L, "
            f"transition_start={transition_start_idx}, transition_len={transition_len}, "
            f"video2={video_samples_2.shape[2]}L)"
        )
        if stitch_mode == "overlap_linear_video":
            print(
                f"[StitchDebug] overlap_linear_video details: "
                f"requested={overlap_latent_frames}L, effective={overlap_eff}L"
            )
        print(f"[StitchDebug] video mask per latent frame (ch0,h0,w0): "
              f"{[f'{v:.2f}' for v in _vm]}")
        if mask_video:
            print(f"[StitchDebug] video mask window: start_idx={video_start_idx}, "
                  f"end_idx={video_end_idx_exclusive}, slope_len={video_slope_len}")
        _am = audio_mask[0, 0, :, 0].tolist()
        print(f"[StitchDebug] audio mask: {len(_am)} frames, "
              f"min={min(_am):.3f}, max={max(_am):.3f}, "
              f"first_nonzero={next((i for i,v in enumerate(_am) if v > 0.01), 'none')}, "
              f"last_nonzero={next((i for i in range(len(_am)-1,-1,-1) if _am[i] > 0.01), 'none')}")
        # === END DEBUG ===

        output_latent = av_latent_1.copy()
        output_latent["samples"] = NestedTensor(
            ltxav.recombine_audio_and_video_latents(video_concat, audio_concat)
        )

        if post_blend:
            # Post-blend mode: NO noise_mask → sampler denoises everything equally.
            # Store originals + blend masks for LTXVPostBlendTransition.
            # Important: reuse the same masks as standard mode so mask_* and
            # *_mask_init_value semantics stay identical.
            if "noise_mask" in output_latent:
                del output_latent["noise_mask"]

            # Blend convention: 0 = keep original, 1 = keep generated.
            video_blend = torch.clamp(video_mask, 0.0, 1.0).clone()
            audio_blend = torch.clamp(audio_mask, 0.0, 1.0).clone()

            # Store for PostBlend node
            output_latent["_postblend_video_original"] = video_concat.clone()
            output_latent["_postblend_audio_original"] = audio_concat.clone()
            output_latent["_postblend_video_blend"] = video_blend
            output_latent["_postblend_audio_blend"] = audio_blend

            print(f"[StitchDebug] POST_BLEND mode: no noise_mask set. "
                  f"Blend envelope stored. Video blend range: "
                  f"{video_blend[0,0,:,0,0].tolist()}")
        else:
            # Standard inpaint mode: per-step masking
            output_latent["noise_mask"] = NestedTensor(
                ltxav.recombine_audio_and_video_latents(
                    torch.clamp(video_mask, 0.0, 1.0),
                    torch.clamp(audio_mask, 0.0, 1.0),
                )
            )

        return (output_latent,)