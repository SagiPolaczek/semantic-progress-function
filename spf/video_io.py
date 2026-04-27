"""Video reading utilities."""

from pathlib import Path
from typing import Iterator, NamedTuple, Union

import cv2
from PIL import Image
from tqdm import tqdm


class VideoParams(NamedTuple):
    width: int
    height: int
    fps: int
    num_frames: int


def get_video_params(video_path: Union[str, Path]) -> VideoParams:
    """Read video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        return VideoParams(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS)),
            num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def frame_generator(video_path: Union[str, Path]) -> Iterator[Image.Image]:
    """Yield PIL images from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total, desc="Reading frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pbar.update(1)
    finally:
        cap.release()
