# From https://github.com/albumentations-team/albumentations/blob/main/albumentations/augmentations/crops/functional.py

from typing import Optional, Sequence, Tuple, cast
import numpy as np


def get_random_crop_coords(
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
) -> Tuple[int, int, int, int]:
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(
    img: np.ndarray,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
) -> np.ndarray:
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height,
                crop_width=crop_width,
                height=height,
                width=width,
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(
        height, width, crop_height, crop_width, h_start, w_start
    )
    return img[y1:y2, x1:x2]
