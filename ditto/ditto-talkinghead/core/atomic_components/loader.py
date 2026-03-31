"""Media Loader: loads images and videos as RGB frame lists.

Handles loading source media (single image or video) and normalizing
dimensions to meet the pipeline's requirements. Supports:
  - Automatic image/video detection via file magic bytes
  - Resolution capping via max_dim parameter
  - Dimension alignment to multiples of 2 (required by the SPADE decoder)
"""

import filetype
import imageio
import cv2


def is_image(file_path: str) -> bool:
    """Check if a file is an image based on its magic bytes."""
    return filetype.is_image(file_path)


def is_video(file_path: str) -> bool:
    """Check if a file is a video based on its magic bytes."""
    return filetype.is_video(file_path)


def compute_resize_dimensions(
    height: int,
    width: int,
    max_dim: int = 1920,
    alignment: int = 2,
) -> tuple[int, int, bool]:
    """Compute target dimensions that satisfy max_dim and alignment constraints.

    Args:
        height: Original image height.
        width: Original image width.
        max_dim: Maximum allowed dimension (0 or negative = no limit).
        alignment: Dimensions must be multiples of this value.

    Returns:
        (new_height, new_width, needs_resize): Tuple indicating target
        dimensions and whether resizing is actually needed.
    """
    needs_resize = False

    # Cap to max dimension while preserving aspect ratio
    if max_dim > 0 and max(height, width) > max_dim:
        needs_resize = True
        if height > width:
            new_height = max_dim
            new_width = int(round(width * max_dim / height))
        else:
            new_width = max_dim
            new_height = int(round(height * max_dim / width))
    else:
        new_height = height
        new_width = width

    # Align dimensions to multiples of alignment (required by decoder)
    if new_height % alignment != 0:
        new_height -= new_height % alignment
        needs_resize = True
    if new_width % alignment != 0:
        new_width -= new_width % alignment
        needs_resize = True

    return new_height, new_width, needs_resize


def load_image(image_path: str, max_dim: int = -1) -> "np.ndarray":
    """Load a single image as an RGB numpy array.

    Args:
        image_path: Path to image file.
        max_dim: Maximum dimension (height or width). -1 = no limit.

    Returns:
        RGB image as numpy array (H, W, 3), uint8.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    new_height, new_width, needs_resize = compute_resize_dimensions(height, width, max_dim)
    if needs_resize:
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
    return image_rgb


def load_video(video_path: str, max_frames: int = -1, max_dim: int = -1) -> list:
    """Load video frames as a list of RGB numpy arrays.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to load (-1 = all).
        max_dim: Maximum dimension per frame (-1 = no limit).

    Returns:
        List of RGB frames, each (H, W, 3) uint8.
    """
    reader = imageio.get_reader(video_path, "ffmpeg")

    target_height, target_width, needs_resize = None, None, None
    frames = []

    for frame_idx, frame_rgb in enumerate(reader):
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # Compute resize dimensions from first frame
        if needs_resize is None:
            height, width = frame_rgb.shape[:2]
            target_height, target_width, needs_resize = compute_resize_dimensions(
                height, width, max_dim,
            )

        if needs_resize:
            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))

        frames.append(frame_rgb)

    reader.close()
    return frames


def load_source_frames(
    source_path: str, max_dim: int = -1, n_frames: int = -1
) -> tuple[list, bool]:
    """Load source media (image or video) as RGB frame list.

    Args:
        source_path: Path to image or video file.
        max_dim: Maximum dimension per frame.
        n_frames: Maximum number of video frames to load.

    Returns:
        (frame_list, is_image): List of RGB frames and whether source is a single image.
    """
    if is_image(source_path):
        rgb = load_image(source_path, max_dim)
        return [rgb], True
    elif is_video(source_path):
        rgb_list = load_video(source_path, n_frames, max_dim)
        return rgb_list, False
    else:
        raise ValueError(f"Unsupported source type: {source_path}")


def mirror_index(index: int, sequence_length: int) -> int:
    """Map an unbounded index into a sequence using ping-pong (mirror) looping.

    For a sequence of length 5: 0,1,2,3,4,3,2,1,0,1,2,3,4,...
    """
    period = index // sequence_length
    position = index % sequence_length
    if period % 2 == 0:
        return position
    else:
        return sequence_length - position - 1


# Keep old name for backward compatibility
_mirror_index = mirror_index


class LoopLoader:
    """Iterator that loops through a list with optional mirror-bouncing.

    Used to cycle through video template frames when the driving motion
    sequence is longer than the source video.
    """

    def __init__(
        self,
        items: list,
        max_iterations: int = -1,
        use_mirror_loop: bool = True,
    ):
        self.items = items
        self.current_index = 0
        self.num_items = len(self.items)
        self.max_iterations = max_iterations if max_iterations > 0 else self.num_items
        self.use_mirror_loop = use_mirror_loop

    def __len__(self):
        return self.max_iterations

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.max_iterations:
            raise StopIteration

        if self.use_mirror_loop:
            item_idx = mirror_index(self.current_index, self.num_items)
        else:
            item_idx = self.current_index % self.num_items

        item = self.items[item_idx]
        self.current_index += 1
        return item

    def __call__(self):
        return self.__iter__()

    def reset(self, max_iterations: int = -1):
        """Reset iterator to the beginning."""
        self.current_index = 0
        self.max_iterations = max_iterations if max_iterations > 0 else self.num_items
