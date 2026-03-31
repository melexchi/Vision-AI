"""Video Writer: writes rendered frames to an MP4 video file.

Uses imageio's FFmpeg backend to encode frames into H.264 video. The writer
accepts both RGB and BGR frames, converting as needed.

Note: The hardcoded fps=25 matches the pipeline's DEFAULT_FPS. If fps needs
to change, pass it as a parameter to the constructor.
"""

import os
import imageio


class VideoWriterByImageIO:
    """FFmpeg-backed video writer using imageio.

    Encodes frames incrementally — each __call__ appends one frame.
    Call close() when done to finalize the video file.
    """

    def __init__(self, video_path: str, fps: int = 25, **kwargs):
        """Initialize video writer.

        Args:
            video_path: Output file path (e.g., "/tmp/output.mp4").
            fps: Frames per second (default 25, matching Ditto pipeline).
            **kwargs: Optional overrides:
                format: Container format (default "mp4").
                vcodec: Video codec (default "libx264").
                quality: Video quality level.
                pixelformat: Pixel format (default "yuv420p").
                macro_block_size: Macroblock alignment (default 2).
                crf: Constant rate factor for quality (default 18, lower = better).
        """
        video_format = kwargs.get("format", "mp4")
        codec = kwargs.get("vcodec", "libx264")
        quality = kwargs.get("quality")
        pixel_format = kwargs.get("pixelformat", "yuv420p")
        macro_block_size = kwargs.get("macro_block_size", 2)
        crf_value = kwargs.get("crf", 18)
        ffmpeg_params = ["-crf", str(crf_value)]

        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        self.writer = imageio.get_writer(
            video_path,
            fps=fps,
            format=video_format,
            codec=codec,
            quality=quality,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixel_format,
            macro_block_size=macro_block_size,
        )

    def __call__(self, image: "np.ndarray", fmt: str = "bgr"):
        """Append a single frame to the video.

        Args:
            image: Frame data (H, W, 3), uint8.
            fmt: Color format — "bgr" (OpenCV default) or "rgb".
        """
        if fmt == "bgr":
            frame = image[..., ::-1]  # BGR -> RGB for imageio
        else:
            frame = image
        self.writer.append_data(frame)

    def close(self):
        """Finalize and close the video file."""
        self.writer.close()
