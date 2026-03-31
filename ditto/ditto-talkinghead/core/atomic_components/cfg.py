"""Configuration: loads and parses the Ditto pipeline configuration.

The configuration is stored as a pickle file containing model paths,
hyperparameters, and default generation settings. This module parses
the pickle and returns structured configs for each pipeline component.

Configuration hierarchy:
  cfg.pkl -> {
    base_cfg: {
      insightface_det_cfg, landmark106_cfg, landmark203_cfg, landmark478_cfg,
      appearance_extractor_cfg, motion_extractor_cfg,
      stitch_network_cfg, warp_network_cfg, decoder_cfg,
      hubert_cfg / wavlm_cfg
    },
    audio2motion_cfg: {
      model_path, device, motion_feat_dim, audio_feat_dim, seq_frames,
      use_emo, use_sc, use_eye_open, use_eye_ball, w2f_type
    },
    default_kwargs: {
      sampling_timesteps, fps, overlap_v2, smo_k_d, ...
    }
  }
"""

import os
import pickle
import numpy as np


def load_pickle(path: str):
    """Load a Python object from a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)


# Backward-compatible alias
load_pkl = load_pickle


def parse_cfg(
    cfg_pkl_path: str,
    data_root: str,
    override_cfg: dict | None = None,
) -> list:
    """Parse a Ditto configuration pickle into per-component configs.

    Args:
        cfg_pkl_path: Path to the configuration pickle file.
        data_root: Root directory for resolving relative model paths.
        override_cfg: Optional dict to override specific config values.
            Format: {"section_name": {"key": value}}.

    Returns:
        List of 8 config dicts/values in order:
            [avatar_registrar_cfg, condition_handler_cfg, lmdm_cfg,
             stitch_network_cfg, warp_network_cfg, decoder_cfg,
             wav2feat_cfg, default_kwargs]
    """
    def resolve_path(path: str) -> str:
        """Resolve a model path — use absolute if exists, otherwise join with data_root."""
        if os.path.isfile(path):
            return path
        return os.path.join(data_root, path)

    cfg = load_pickle(cfg_pkl_path)

    # Apply optional config overrides (for debugging/experimentation)
    if isinstance(override_cfg, dict):
        for section_name, section_overrides in override_cfg.items():
            if not isinstance(section_overrides, dict):
                continue
            for key, value in section_overrides.items():
                cfg[section_name][key] = value

    base_cfg = cfg["base_cfg"]
    audio2motion_cfg = cfg["audio2motion_cfg"]
    default_kwargs = cfg["default_kwargs"]

    # Resolve all model paths to absolute paths
    for section_name in base_cfg:
        if section_name == "landmark478_cfg":
            # MediaPipe has multiple model files
            for path_key in ["task_path", "blaze_face_model_path", "face_mesh_model_path"]:
                if path_key in base_cfg[section_name] and base_cfg[section_name][path_key]:
                    base_cfg[section_name][path_key] = resolve_path(
                        base_cfg[section_name][path_key]
                    )
        else:
            base_cfg[section_name]["model_path"] = resolve_path(
                base_cfg[section_name]["model_path"]
            )

    audio2motion_cfg["model_path"] = resolve_path(audio2motion_cfg["model_path"])

    # Build per-component configs
    avatar_registrar_cfg = {
        key: base_cfg[key]
        for key in [
            "insightface_det_cfg",
            "landmark106_cfg",
            "landmark203_cfg",
            "landmark478_cfg",
            "appearance_extractor_cfg",
            "motion_extractor_cfg",
        ]
    }

    stitch_network_cfg = base_cfg["stitch_network_cfg"]
    warp_network_cfg = base_cfg["warp_network_cfg"]
    decoder_cfg = base_cfg["decoder_cfg"]

    condition_handler_cfg = {
        key: audio2motion_cfg[key]
        for key in ["use_emo", "use_sc", "use_eye_open", "use_eye_ball", "seq_frames"]
    }

    lmdm_cfg = {
        key: audio2motion_cfg[key]
        for key in ["model_path", "device", "motion_feat_dim", "audio_feat_dim", "seq_frames"]
    }

    # Audio feature extractor config
    feature_type = audio2motion_cfg["w2f_type"]
    wav2feat_cfg = {
        "w2f_cfg": (
            base_cfg["hubert_cfg"] if feature_type == "hubert"
            else base_cfg["wavlm_cfg"]
        ),
        "w2f_type": feature_type,
    }

    return [
        avatar_registrar_cfg,
        condition_handler_cfg,
        lmdm_cfg,
        stitch_network_cfg,
        warp_network_cfg,
        decoder_cfg,
        wav2feat_cfg,
        default_kwargs,
    ]


def print_cfg(**kwargs):
    """Print configuration values for debugging."""
    for key, value in kwargs.items():
        if key == "ch_info":
            print(key, type(value))
        elif key == "ctrl_info":
            print(key, type(value), len(value))
        elif isinstance(value, np.ndarray):
            print(key, type(value), value.shape)
        else:
            print(key, type(value), value)
