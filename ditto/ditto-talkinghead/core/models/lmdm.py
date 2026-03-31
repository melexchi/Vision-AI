"""LMDM (Latent Motion Diffusion Model): predicts facial motion from audio.

Uses DDIM (Denoising Diffusion Implicit Models) to generate a sequence of
motion parameters conditioned on:
  - kp_cond: Previous frame's keypoint state (1, 265) — temporal continuity
  - aud_cond: Audio conditioning features (1, 80, 1103) — speech content
  - time_cond: Diffusion timestep (1,) — noise level

The DDIM sampler iteratively denoises a random noise tensor into clean motion.
With 5 steps it runs ~2x faster than 10 steps with minimal quality loss.

Supports PyTorch, ONNX, and TensorRT backends for inference.
"""

import numpy as np
import torch
from ..utils.load_model import load_model


def make_cosine_beta_schedule(num_timesteps: int, cosine_offset: float = 8e-3) -> np.ndarray:
    """Create a cosine noise schedule for the diffusion process.

    The cosine schedule provides smoother noise transitions compared to linear,
    reducing artifacts at low step counts (important for our 5-step DDIM).

    Returns:
        Beta values (noise amounts) for each timestep, shape (num_timesteps,).
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps + cosine_offset
    alpha_bars = torch.cos(steps / (1 + cosine_offset) * np.pi / 2).pow(2)
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    return np.clip(betas.numpy(), a_min=0, a_max=0.999)


class LMDM:
    """Latent Motion Diffusion Model wrapper supporting multiple backends.

    The model predicts both:
      - pred_noise: The noise component of the input
      - x_start: The denoised (clean) motion prediction

    These are used in the DDIM update rule to iteratively refine the motion.
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        kwargs["module_name"] = "LMDM"
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.motion_feat_dim = kwargs.get("motion_feat_dim", 265)
        self.audio_feat_dim = kwargs.get("audio_feat_dim", 1024 + 35)
        self.seq_frames = kwargs.get("seq_frames", 80)

        if self.model_type != "pytorch":
            self._init_diffusion_params()

    def setup(self, sampling_timesteps: int):
        """Configure the DDIM sampler for the given number of denoising steps."""
        if self.model_type == "pytorch":
            self.model.setup(sampling_timesteps)
        else:
            self._setup_ddim(sampling_timesteps)

    def _init_diffusion_params(self):
        """Initialize diffusion schedule parameters (ONNX/TRT only)."""
        self.sampling_timesteps = None
        self.total_timesteps = 1000

        betas = torch.Tensor(make_cosine_beta_schedule(self.total_timesteps))
        alphas = 1.0 - betas
        self.cumulative_alpha = torch.cumprod(alphas, axis=0).cpu().numpy()

    def _setup_ddim(self, sampling_timesteps: int = 25):
        """Pre-compute DDIM sampling schedule and constants.

        DDIM (eta=0) is deterministic — the same noise input always produces
        the same motion output. This is important for reproducibility and
        debugging the pipeline.
        """
        if self.sampling_timesteps == sampling_timesteps:
            return  # Already configured

        self.sampling_timesteps = sampling_timesteps
        eta = 0  # DDIM deterministic mode (no added noise between steps)
        motion_shape = (1, self.seq_frames, self.motion_feat_dim)

        # Create evenly-spaced timestep schedule: [T-1, ..., 1, 0, -1]
        timestep_values = torch.linspace(-1, self.total_timesteps - 1, steps=sampling_timesteps + 1)
        timestep_values = list(reversed(timestep_values.int().tolist()))
        self.timestep_pairs = list(zip(timestep_values[:-1], timestep_values[1:]))

        # Pre-compute per-step constants for the DDIM update rule:
        #   x_{t-1} = sqrt(alpha_{t-1}) * x_start + c * pred_noise + sigma * noise
        self.timestep_conditions = []
        self.alpha_next_sqrt_values = []
        self.noise_coefficient_values = []
        self.sigma_values = []
        self.precomputed_noise = []

        for current_time, next_time in self.timestep_pairs:
            self.timestep_conditions.append(np.full((1,), current_time, dtype=np.int64))

            if next_time < 0:
                continue  # Last step — directly use x_start

            alpha_current = self.cumulative_alpha[current_time]
            alpha_next = self.cumulative_alpha[next_time]

            # DDIM update coefficients (sigma=0 for deterministic)
            sigma = eta * np.sqrt((1 - alpha_current / alpha_next) * (1 - alpha_next) / (1 - alpha_current))
            noise_coeff = np.sqrt(1 - alpha_next - sigma ** 2)
            noise = np.random.randn(*motion_shape).astype(np.float32)

            self.alpha_next_sqrt_values.append(np.sqrt(alpha_next))
            self.noise_coefficient_values.append(noise_coeff)
            self.sigma_values.append(sigma)
            self.precomputed_noise.append(noise)

    def _run_single_step(
        self,
        noisy_motion: np.ndarray,
        keypoint_cond: np.ndarray,
        audio_cond: np.ndarray,
        timestep_cond: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one denoising step through the LMDM model.

        Returns:
            (predicted_noise, predicted_clean_motion)
        """
        if self.model_type == "onnx":
            outputs = self.model.run(None, {
                "x": noisy_motion,
                "cond_frame": keypoint_cond,
                "cond": audio_cond,
                "time_cond": timestep_cond,
            })
            return outputs[0], outputs[1]

        elif self.model_type == "tensorrt":
            self.model.setup({
                "x": noisy_motion,
                "cond_frame": keypoint_cond,
                "cond": audio_cond,
                "time_cond": timestep_cond,
            })
            self.model.infer()
            return (
                self.model.buffer["pred_noise"][0],
                self.model.buffer["x_start"][0],
            )

        elif self.model_type == "pytorch":
            with torch.no_grad():
                pred_noise, x_start = self.model(
                    noisy_motion, keypoint_cond, audio_cond, timestep_cond,
                )
            return pred_noise, x_start

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _ddim_sample(
        self,
        keypoint_cond: np.ndarray,
        audio_cond: np.ndarray,
        sampling_timesteps: int,
    ) -> np.ndarray:
        """Run full DDIM sampling loop (ONNX/TRT backend).

        Starting from random noise, iteratively denoise using the LMDM model
        for the configured number of steps.
        """
        self._setup_ddim(sampling_timesteps)

        # Start from pure noise
        noisy_motion = np.random.randn(
            1, self.seq_frames, self.motion_feat_dim
        ).astype(np.float32)

        step_idx = 0
        for _, next_time in self.timestep_pairs:
            timestep = self.timestep_conditions[step_idx]
            pred_noise, clean_prediction = self._run_single_step(
                noisy_motion, keypoint_cond, audio_cond, timestep,
            )

            if next_time < 0:
                # Final step — return clean prediction directly
                noisy_motion = clean_prediction
                continue

            # DDIM update: x_{t-1} = sqrt(a_{t-1}) * x0 + c * noise_pred + sigma * noise
            alpha_sqrt = self.alpha_next_sqrt_values[step_idx]
            noise_coeff = self.noise_coefficient_values[step_idx]
            sigma = self.sigma_values[step_idx]
            noise = self.precomputed_noise[step_idx]

            noisy_motion = (
                clean_prediction * alpha_sqrt
                + noise_coeff * pred_noise
                + sigma * noise
            )
            step_idx += 1

        return noisy_motion

    def __call__(
        self,
        keypoint_cond: np.ndarray,
        audio_cond: np.ndarray,
        sampling_timesteps: int,
    ) -> np.ndarray:
        """Generate motion keypoints from audio conditioning.

        Args:
            keypoint_cond: Previous keypoint state (1, 265) for temporal continuity.
            audio_cond: Audio + emotion + eye conditioning (1, 80, cond_dim).
            sampling_timesteps: Number of DDIM denoising steps (e.g., 5).

        Returns:
            Predicted motion sequence (1, 80, 265).
        """
        if self.model_type == "pytorch":
            predicted_motion = self.model.ddim_sample(
                torch.from_numpy(keypoint_cond).to(self.device),
                torch.from_numpy(audio_cond).to(self.device),
                sampling_timesteps,
            ).cpu().numpy()
        else:
            predicted_motion = self._ddim_sample(keypoint_cond, audio_cond, sampling_timesteps)

        return predicted_motion
