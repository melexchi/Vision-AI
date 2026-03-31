"""Pre-render avatar state clips using SkyReels-A1.

Generates high-quality idle, thinking, and greeting clips for avatars.
These are used as fallback animations when no real-time audio is driving
the talking head (e.g., while the LLM is processing, or the user is silent).

Usage:
    cd /path/to/SkyReels-A1
    python /path/to/prerender_clips.py \
        --image_path avatar.png \
        --output_dir ./prerendered/ \
        --clips idle thinking greeting \
        --target_fps 24

Requires: SkyReels-A1 environment with pretrained models.
"""

import torch
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import subprocess
import tempfile
import soundfile as sf

# SkyReels-A1 must be on path or CWD
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from skyreels_a1.skyreels_a1_i2v_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from skyreels_a1.src.frame_interpolation import init_frame_interpolation_model
from skyreels_a1.src.multi_fps import multi_fps_tool
from diffposetalk.diffposetalk import DiffPoseTalk


# Clip definitions: type -> config
# Each type has a default number of variants — more variants = more natural idle behaviour.
# Different seeds per variant produce different micro-motions (blink timing, head drift, etc.)
CLIP_CONFIGS = {
    "idle": {
        "duration": 4.0,
        "description": "Subtle breathing and blinking, neutral expression",
        "guidance_scale": 3.0,
        "base_seed": 42,
        "default_variants": 6,
    },
    "thinking": {
        "duration": 3.0,
        "description": "Slight head tilt, eyes drifting to the side",
        "guidance_scale": 3.5,
        "base_seed": 123,
        "default_variants": 4,
    },
    "lookingup": {
        "duration": 3.0,
        "description": "Gentle upward gaze, slight head tilt back",
        "guidance_scale": 3.0,
        "base_seed": 200,
        "default_variants": 4,
    },
    "lippurse": {
        "duration": 2.5,
        "description": "Subtle lip pursing/pressing, contemplative",
        "guidance_scale": 3.0,
        "base_seed": 300,
        "default_variants": 4,
    },
    "greeting": {
        "duration": 2.0,
        "description": "Small nod, friendly neutral expression",
        "guidance_scale": 3.0,
        "base_seed": 77,
        "default_variants": 1,
    },
}


def generate_silence_audio(duration: float, sr: int = 16000) -> str:
    """Generate a silent WAV file for driving idle/ambient animations."""
    samples = np.zeros(int(duration * sr), dtype=np.float32)
    # Add very subtle noise to prevent DiffPoseTalk from producing completely static output
    samples += np.random.randn(len(samples)).astype(np.float32) * 0.001
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, samples, sr)
    return path


def parse_audio_frames(driving_frames, max_frame_num, fps=25):
    """Resample driving frames to 12fps for SkyReels-A1 native rate."""
    video_length = len(driving_frames)
    duration = video_length / fps
    target_times = np.arange(0, duration, 1 / 12)
    frame_indices = (target_times * fps).astype(np.int32)
    frame_indices = frame_indices[frame_indices < video_length]

    new_frames = []
    for idx in frame_indices:
        new_frames.append(driving_frames[idx])
        if len(new_frames) >= max_frame_num - 1:
            break

    pad_count = max_frame_num - len(new_frames) - 1
    new_frames = (
        [new_frames[0]] * 2
        + new_frames[1:-1]
        + [new_frames[-1]] * pad_count
    )
    return new_frames


def load_pipeline(model_name: str, target_fps: int):
    """Load the full SkyReels-A1 pipeline."""
    weight_dtype = torch.bfloat16
    siglip_name = os.path.join(model_name, "siglip-so400m-patch14-384")

    print("Loading SkyReels-A1 pipeline...")

    lmk_extractor = LMKExtractor()
    processor = FaceAnimationProcessor(checkpoint="pretrained_models/smirk/SMIRK_em1.pt")
    vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False)
    face_helper = FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", device="cuda",
    )

    siglip = SiglipVisionModel.from_pretrained(siglip_name)
    siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_name)

    frame_inter_model = None
    if target_fps != 12:
        frame_inter_model = init_frame_interpolation_model(
            "pretrained_models/film_net/film_net_fp16.pt", device="cuda"
        )

    diffposetalk = DiffPoseTalk()

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer"
    ).to(weight_dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name, subfolder="vae"
    ).to(weight_dtype)

    lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
        model_name, subfolder="pose_guider"
    ).to(weight_dtype)

    pipe = SkyReelsA1ImagePoseToVideoPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        vae=vae,
        lmk_encoder=lmk_encoder,
        image_encoder=siglip,
        feature_extractor=siglip_normalize,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    print("Pipeline loaded.")

    return {
        "pipe": pipe,
        "processor": processor,
        "lmk_extractor": lmk_extractor,
        "vis": vis,
        "face_helper": face_helper,
        "diffposetalk": diffposetalk,
        "frame_inter_model": frame_inter_model,
    }


def _windowed_curve(values: np.ndarray) -> np.ndarray:
    """Apply Hanning window so curve starts and ends at zero — no jumps between clips."""
    window = np.hanning(len(values))
    return values * window


def _add_head_motion_variety(driving_outputs, seed, max_amplitude=0.04):
    """Add unique smooth head drifts per variant so idle clips don't all look the same.

    All offsets are windowed (Hanning) so they start and end at zero,
    ensuring seamless transitions when clips play back-to-back.
    """
    rng = np.random.RandomState(seed)
    n_frames = len(driving_outputs)
    t = np.linspace(0, 2 * np.pi, n_frames)

    for axis in range(3):  # pitch, yaw, roll
        freq = rng.uniform(0.4, 1.2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.01, max_amplitude)
        offsets = _windowed_curve(amp * np.sin(freq * t + phase))

        for i, frame_coef in enumerate(driving_outputs):
            frame_coef["pose_params"][0, axis] += offsets[i]


def _apply_motion_profile(driving_outputs, clip_type: str, seed: int):
    """Apply per-clip-type FLAME coefficient tweaks for different expressions/motions.

    All types zero the jaw (no mouth opening) and add unique head drift per seed.
    All motion curves are windowed so every clip starts and ends at neutral pose —
    no jarring jumps when clips transition.

    FLAME expression_params indices (approximate):
      0-9: jaw/mouth area, 10-19: mid-face/cheeks, 20-29: brow/forehead,
      30-39: nose/upper lip, 40-49: lower face detail
    """
    rng = np.random.RandomState(seed)
    n_frames = len(driving_outputs)
    t = np.linspace(0, 2 * np.pi, n_frames)

    for frame_coef in driving_outputs:
        frame_coef["jaw_params"] = torch.zeros_like(frame_coef["jaw_params"])

    if clip_type == "idle":
        for frame_coef in driving_outputs:
            frame_coef["expression_params"] *= 0.08
        _add_head_motion_variety(driving_outputs, seed, max_amplitude=0.04)

    elif clip_type == "thinking":
        for frame_coef in driving_outputs:
            frame_coef["expression_params"] *= 0.12
        yaw_dir = rng.choice([-1, 1])
        yaw_curve = _windowed_curve(yaw_dir * 0.06 * np.sin(0.8 * t + rng.uniform(0, np.pi)))
        for i, frame_coef in enumerate(driving_outputs):
            frame_coef["pose_params"][0, 1] += yaw_curve[i]
        _add_head_motion_variety(driving_outputs, seed, max_amplitude=0.03)

    elif clip_type == "lookingup":
        for frame_coef in driving_outputs:
            frame_coef["expression_params"] *= 0.10
        pitch_curve = _windowed_curve(-0.07 * np.sin(0.6 * t + rng.uniform(0, np.pi)))
        brow_curve = _windowed_curve(0.15 * np.sin(0.5 * t + rng.uniform(0, np.pi)))
        for i, frame_coef in enumerate(driving_outputs):
            frame_coef["pose_params"][0, 0] += pitch_curve[i]
            frame_coef["expression_params"][0, 20:26] += brow_curve[i]
        _add_head_motion_variety(driving_outputs, seed, max_amplitude=0.025)

    elif clip_type == "lippurse":
        for frame_coef in driving_outputs:
            frame_coef["expression_params"] *= 0.10
        purse_curve = _windowed_curve(0.2 * np.sin(1.0 * t + rng.uniform(0, np.pi)))
        for i, frame_coef in enumerate(driving_outputs):
            frame_coef["expression_params"][0, 30:38] += purse_curve[i]
        _add_head_motion_variety(driving_outputs, seed, max_amplitude=0.03)

    elif clip_type == "greeting":
        for frame_coef in driving_outputs:
            frame_coef["expression_params"] *= 0.08
        nod = _windowed_curve(0.06 * np.sin(np.pi * t / t[-1]))
        for i, frame_coef in enumerate(driving_outputs):
            frame_coef["pose_params"][0, 0] += nod[i]
        _add_head_motion_variety(driving_outputs, seed, max_amplitude=0.02)

    else:
        for frame_coef in driving_outputs:
            frame_coef["expression_params"] *= 0.08
        _add_head_motion_variety(driving_outputs, seed)


def generate_clip(
    pipeline_components: dict,
    image_path: str,
    clip_type: str,
    output_path: str,
    target_fps: int = 24,
    sample_size=(480, 720),
    max_frame_num: int = 49,
    seed_override: int | None = None,
):
    """Generate a single pre-rendered clip."""
    config = CLIP_CONFIGS[clip_type]
    seed = seed_override if seed_override is not None else config["base_seed"]
    print(f"\nGenerating '{clip_type}' clip (seed={seed}, {config['duration']}s, {config['description']})...")

    pipe = pipeline_components["pipe"]
    processor = pipeline_components["processor"]
    lmk_extractor = pipeline_components["lmk_extractor"]
    vis = pipeline_components["vis"]
    face_helper = pipeline_components["face_helper"]
    diffposetalk = pipeline_components["diffposetalk"]
    frame_inter_model = pipeline_components["frame_inter_model"]

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate silent audio to drive subtle motion
    audio_path = generate_silence_audio(config["duration"])

    # Load and prep image
    image = load_image(image=image_path)
    image = processor.crop_and_resize(image, sample_size[0], sample_size[1])

    ref_image, x1, y1 = processor.face_crop(np.array(image))
    face_h, face_w = ref_image.shape[:2]
    source_image = ref_image

    # Generate motion from audio (silence → subtle idle motion)
    source_outputs, source_tform, image_original = processor.process_source_image(source_image)
    driving_outputs = diffposetalk.infer_from_file(
        audio_path,
        source_outputs["shape_params"].view(-1)[:100].detach().cpu().numpy(),
    )

    # Apply per-type motion post-processing.
    # FLAME expression_params has 50 blendshapes — we scale/modify them per clip type.
    # Jaw is always zeroed (no mouth opening). Head motion variety is injected per seed.
    _apply_motion_profile(driving_outputs, clip_type, seed)

    out_frames = processor.preprocess_lmk3d_from_coef(
        source_outputs, source_tform, image_original.shape, driving_outputs
    )
    out_frames = parse_audio_frames(out_frames, max_frame_num)

    # Build motion control video
    rescale_motions = np.zeros_like(image)[np.newaxis, :].repeat(48, axis=0)
    for ii in range(rescale_motions.shape[0]):
        rescale_motions[ii][y1 : y1 + face_h, x1 : x1 + face_w] = out_frames[ii]

    ref_resized = cv2.resize(ref_image, (512, 512))
    ref_lmk = lmk_extractor(ref_resized[:, :, ::-1])
    ref_img = vis.draw_landmarks_v3(
        (512, 512), (face_w, face_h),
        ref_lmk["lmks"].astype(np.float32), normed=True,
    )

    first_motion = np.zeros_like(np.array(image))
    first_motion[y1 : y1 + face_h, x1 : x1 + face_w] = ref_img
    first_motion = first_motion[np.newaxis, :]

    motions = np.concatenate([first_motion, rescale_motions])
    input_video = motions[:max_frame_num]

    # Get aligned face for pipeline
    face_helper.clean_all()
    face_helper.read_image(np.array(image)[:, :, ::-1])
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    align_face = face_helper.cropped_faces[0]
    image_face = align_face[:, :, ::-1]

    input_video_tensor = torch.from_numpy(np.array(input_video)).permute([3, 0, 1, 2]).unsqueeze(0)
    input_video_tensor = input_video_tensor / 255

    # Run inference
    with torch.no_grad():
        sample = pipe(
            image=image,
            image_face=image_face,
            control_video=input_video_tensor,
            prompt="",
            negative_prompt="",
            height=sample_size[0],
            width=sample_size[1],
            num_frames=49,
            generator=generator,
            guidance_scale=config["guidance_scale"],
            num_inference_steps=10,
        )
    out_samples = sample.frames[0][2:]  # skip first 2 duplicate frames

    # Frame interpolation for higher FPS
    if target_fps != 12 and frame_inter_model is not None:
        out_samples = multi_fps_tool(out_samples, frame_inter_model, target_fps)

    export_to_video(out_samples, output_path, fps=target_fps)

    # Cleanup temp audio
    os.unlink(audio_path)

    print(f"  Saved: {output_path} ({len(out_samples)} frames @ {target_fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Pre-render avatar state clips with SkyReels-A1")
    parser.add_argument("--image_path", type=str, required=True, help="Path to avatar reference image")
    parser.add_argument("--output_dir", type=str, default="./prerendered", help="Output directory")
    parser.add_argument("--clips", nargs="+", default=["idle", "thinking", "greeting"],
                        choices=list(CLIP_CONFIGS.keys()), help="Which clips to generate")
    parser.add_argument("--target_fps", type=int, default=24, help="Output FPS (12, 24, 48, 60)")
    parser.add_argument("--model_name", type=str, default="pretrained_models/SkyReels-A1-5B/",
                        help="Path to SkyReels-A1 model")
    parser.add_argument("--avatar_id", type=str, default=None,
                        help="Avatar ID for naming (defaults to image filename)")
    parser.add_argument("--variants", type=int, default=None,
                        help="Override number of variants per clip type (default: per-type config)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    avatar_id = args.avatar_id or Path(args.image_path).stem

    # Load pipeline once
    components = load_pipeline(args.model_name, args.target_fps)

    # Generate variants for each clip type
    total_clips = 0
    for clip_type in args.clips:
        config = CLIP_CONFIGS[clip_type]
        num_variants = args.variants if args.variants is not None else config["default_variants"]

        for v in range(num_variants):
            output_path = os.path.join(args.output_dir, f"{avatar_id}_{clip_type}_{v}.mp4")
            seed = config["base_seed"] + v * 17  # spread seeds for variety
            generate_clip(
                components,
                args.image_path,
                clip_type,
                output_path,
                target_fps=args.target_fps,
                seed_override=seed,
            )
            total_clips += 1

    print(f"\n{total_clips} clips generated in {args.output_dir}/")
    print("These clips will be used as idle/thinking animations in the Ditto pipeline.")


if __name__ == "__main__":
    main()
