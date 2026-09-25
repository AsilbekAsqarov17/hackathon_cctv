from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if self.fps <= 0 or not np_is_finite(self.fps):
            self.fps = 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(
            path=self.path,
            fps=self.fps,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
        )

    def frames(self, stride: int = 1) -> Iterator[tuple[int, float, object]]:
        stride = max(1, int(stride))
        index = 0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                if index % stride == 0:
                    yield index, index / self.fps, frame
                index += 1
        finally:
            self.cap.release()

    def close(self) -> None:
        self.cap.release()


def np_is_finite(value: float) -> bool:
    # Kept local to avoid making numpy a hard import for metadata-only users.
    return value == value and value not in (float("inf"), float("-inf"))
