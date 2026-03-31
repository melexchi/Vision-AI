"""Source to Info: extracts all required features from a source face image.

Runs the full face analysis pipeline on a single image/frame:
  1. Face detection (InsightFace RetinaFace)
  2. 106-point landmark detection (for face alignment)
  3. 203-point landmark detection (for precise crop)
  4. Final crop at 512x512 with configurable scale/offset
  5. 478-point MediaPipe landmarks (for eye tracking)
  6. Appearance feature extraction (3D feature volume)
  7. Motion parameter extraction (pose, expression, keypoints)

The output is a dict containing everything needed to animate this face.
"""

import numpy as np
import cv2

from ..aux_models.insightface_det import InsightFaceDet
from ..aux_models.insightface_landmark106 import Landmark106
from ..aux_models.landmark203 import Landmark203
from ..aux_models.mediapipe_landmark478 import Landmark478
from ..models.appearance_extractor import AppearanceExtractor
from ..models.motion_extractor import MotionExtractor

from ..utils.crop import crop_image
from ..utils.eye_info import EyeAttrUtilsByMP


class Source2Info:
    """Extracts all face features from a single image/frame.

    Orchestrates the detection -> landmark -> crop -> feature extraction pipeline.
    """

    def __init__(
        self,
        insightface_det_cfg: dict,
        landmark106_cfg: dict,
        landmark203_cfg: dict,
        landmark478_cfg: dict,
        appearance_extractor_cfg: dict,
        motion_extractor_cfg: dict,
    ):
        # Face detection and landmark models
        self.face_detector = InsightFaceDet(**insightface_det_cfg)
        self.landmark_106 = Landmark106(**landmark106_cfg)
        self.landmark_203 = Landmark203(**landmark203_cfg)
        self.landmark_478 = Landmark478(**landmark478_cfg)

        # Neural feature extractors
        self.appearance_extractor = AppearanceExtractor(**appearance_extractor_cfg)
        self.motion_extractor = MotionExtractor(**motion_extractor_cfg)

    def _detect_and_crop(
        self, image_rgb: np.ndarray, previous_landmarks=None, **crop_kwargs
    ):
        """Detect face, extract landmarks, and crop to 512x512.

        For the first frame: runs face detection + 106-pt landmarks.
        For subsequent frames: uses previous 203-pt landmarks for tracking.

        Returns:
            (cropped_image, crop_to_original_matrix, landmarks_203)
        """
        if previous_landmarks is None:
            # First frame: detect face and get initial landmarks
            detections, _ = self.face_detector(image_rgb)
            # Sort by face area (largest first)
            face_areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            sorted_detections = detections[np.argsort(-face_areas)]
            if len(sorted_detections) == 0:
                return None
            tracking_landmarks = self.landmark_106(image_rgb, sorted_detections[0])
        else:
            tracking_landmarks = previous_landmarks

        # First crop: rough alignment for 203-pt landmark detection
        rough_crop = crop_image(
            image_rgb,
            tracking_landmarks,
            dsize=self.landmark_203.dsize,
            scale=1.5,
            vy_ratio=-0.1,
            pt_crop_flag=False,
        )
        landmarks_203 = self.landmark_203(rough_crop["img_crop"], rough_crop["M_c2o"])

        # Final crop: precise alignment at 512x512 with user-configurable parameters
        final_crop = crop_image(
            image_rgb,
            landmarks_203,
            dsize=512,
            scale=crop_kwargs.get("crop_scale", 2.3),
            vx_ratio=crop_kwargs.get("crop_vx_ratio", 0),
            vy_ratio=crop_kwargs.get("crop_vy_ratio", -0.125),
            flag_do_rot=crop_kwargs.get("crop_flag_do_rot", True),
            pt_crop_flag=False,
        )

        cropped_image = final_crop["img_crop"]
        crop_to_original = final_crop["M_c2o"]

        return cropped_image, crop_to_original, landmarks_203

    @staticmethod
    def _prepare_model_input(cropped_image: np.ndarray) -> np.ndarray:
        """Convert a cropped face image to model input format.

        Resizes to 256x256, normalizes to [0, 1], and rearranges to (1, 3, H, W).
        """
        resized = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        batch_chw = normalized[None].transpose(0, 3, 1, 2)  # (1, 3, 256, 256)
        return batch_chw

    def _extract_motion_params(self, model_input: np.ndarray) -> dict:
        """Extract motion parameters (pose, expression, keypoints)."""
        return self.motion_extractor(model_input)

    def _extract_appearance_features(self, model_input: np.ndarray) -> np.ndarray:
        """Extract 3D appearance features for the warp network."""
        return self.appearance_extractor(model_input)

    def _extract_eye_info(self, cropped_image: np.ndarray) -> tuple:
        """Extract eye openness and gaze direction using MediaPipe landmarks.

        Returns:
            (eye_openness, eye_ball_position):
                eye_openness: (1, 2) — left/right eye openness ratio
                eye_ball_position: (1, 6) — left/right gaze direction (3 axes each)
        """
        landmarks_478 = self.landmark_478(cropped_image)  # (1, 478, 3)
        eye_attributes = EyeAttrUtilsByMP(landmarks_478)
        eye_openness = eye_attributes.LR_open().reshape(-1, 2)
        eye_ball_position = eye_attributes.LR_ball_move().reshape(-1, 6)
        return eye_openness, eye_ball_position

    def __call__(self, image_rgb: np.ndarray, previous_landmarks=None, **crop_kwargs) -> dict:
        """Extract all features from a single face image.

        Args:
            image_rgb: Input image (H, W, 3), RGB, uint8.
            previous_landmarks: 203-pt landmarks from previous frame (for video tracking).
            **crop_kwargs: Crop configuration overrides.

        Returns:
            Dict containing:
                x_s_info: Motion parameters (pose, expression, keypoints, scale)
                f_s: 3D appearance features (1, 32, 16, 64, 64)
                M_c2o: Affine matrix mapping crop -> original coordinates
                eye_open: Eye openness (1, 2)
                eye_ball: Eye gaze direction (1, 6)
                lmk203: 203-point landmarks (for video frame tracking)
        """
        cropped_image, crop_to_original, landmarks_203 = self._detect_and_crop(
            image_rgb, previous_landmarks, **crop_kwargs,
        )

        eye_openness, eye_ball_position = self._extract_eye_info(cropped_image)

        model_input = self._prepare_model_input(cropped_image)
        motion_params = self._extract_motion_params(model_input)
        appearance_features = self._extract_appearance_features(model_input)

        return {
            "x_s_info": motion_params,
            "f_s": appearance_features,
            "M_c2o": crop_to_original,
            "eye_open": eye_openness,
            "eye_ball": eye_ball_position,
            "lmk203": landmarks_203,
        }
