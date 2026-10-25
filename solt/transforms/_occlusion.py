"""Transforms, which drop parts of images: CutOut and GridMask."""

import random

import cv2
import numpy as np

from ..core import DataContainer, ImageTransform
from ..constants import ALLOWED_GRIDMASK_MODES
from ..utils import (
    img_shape_checker,
    validate_numeric_range_parameter,
    validate_parameter,
)


class CutOut(ImageTransform):
    """
    Does cutout augmentation.

    https://arxiv.org/abs/1708.04552

    Parameters
    ----------
    cutout_size : tuple or int or float
        The size of the cutout ``(width, height)``. If ``int`` or ``float``, then the cutout is a square.
        ``int`` values are in pixels, and ``float`` values are fractions of the image size. Each of the width and
        the height can also be a range ``(min, max)``, from which the size of every cutout is sampled, e.g.
        ``((4, 16), (4, 16))``. By default, it is 2.
    n_cuts: int or tuple of int
        Number of cutouts to make. If tuple ``(min, max)``, then it is sampled from this range. By default, it is 1.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.
    p : float
        Probability of applying this transform.

    Notes
    -----
    To avoid copying, the transform modifies the image it receives in place. If the input image must stay unchanged,
    e.g. because it is cached or augmented several times, use ``Stream(..., in_place=False)`` or pass a copy.
    """

    serializable_name = "cutout"
    """How the class should be stored in the registry"""

    def __init__(self, cutout_size=2, n_cuts=1, data_indices=None, p=0.5):
        super(CutOut, self).__init__(p=p, data_indices=data_indices)
        if isinstance(cutout_size, list):
            cutout_size = tuple(cutout_size)
        if isinstance(cutout_size, (int, float)):
            cutout_size = (cutout_size, cutout_size)
        if not isinstance(cutout_size, tuple):
            raise TypeError("Cutout size is of an incorrect type!")
        if len(cutout_size) != 2:
            raise ValueError("Cutout size must be (width, height)!")
        cutout_size = tuple(CutOut._validate_size(size) for size in cutout_size)
        if CutOut._size_type(cutout_size[0]) is not CutOut._size_type(cutout_size[1]):
            raise TypeError("CutOut sizes must be of the same type")

        if isinstance(n_cuts, list):
            n_cuts = tuple(n_cuts)
        if isinstance(n_cuts, tuple):
            if len(n_cuts) != 2 or not all(isinstance(n, int) for n in n_cuts):
                raise TypeError("Range of n_cuts must be a tuple of two int!")
            if n_cuts[0] > n_cuts[1]:
                raise ValueError("Range of n_cuts must be (min, max)!")
        elif not isinstance(n_cuts, int):
            raise TypeError("Incorrect n_cuts type")
        if min(n_cuts) < 0 if isinstance(n_cuts, tuple) else n_cuts < 0:
            raise ValueError("Cannot make less than 0 cutouts")

        self.cutout_size = cutout_size
        self.n_cuts = n_cuts

    @staticmethod
    def _validate_size(size):
        """Validates the width or the height of a cutout: a number or a range (min, max) of numbers."""
        if isinstance(size, list):
            size = tuple(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError("Range of a cutout size must be (min, max)!")
            if not all(isinstance(s, (int, float)) for s in size) or not isinstance(size[0], type(size[1])):
                raise TypeError("Range of a cutout size must be two int or two float!")
            if size[0] < 0 or size[0] > size[1]:
                raise ValueError("Range of a cutout size must be (min, max) of non-negative numbers!")
        elif not isinstance(size, (int, float)):
            raise TypeError("Cutout size must be int or float, or a range of them")
        return size

    @staticmethod
    def _size_type(size):
        return type(size[0]) if isinstance(size, tuple) else type(size)

    @staticmethod
    def _sample_size(size, image_size):
        """Cutout size in pixels. Random numbers are drawn only for a range."""
        if isinstance(size, tuple):
            if isinstance(size[0], int):
                return random.randint(size[0], size[1])
            return int(random.uniform(size[0], size[1]) * image_size)
        if isinstance(size, float):
            return int(size * image_size)
        return size

    def sample_transform(self, data: DataContainer):
        h, w = super(CutOut, self).sample_transform(data)
        max_sizes = [s[1] if isinstance(s, tuple) else s for s in self.cutout_size]
        max_size_x, max_size_y = [int(s * dim) if isinstance(s, float) else s for s, dim in zip(max_sizes, (w, h))]
        if max_size_x > w or max_size_y > h:
            raise ValueError("Cutout size is too large!")

        n_cuts = random.randint(*self.n_cuts) if isinstance(self.n_cuts, tuple) else self.n_cuts
        for key in ("x", "y", "cut_size_x", "cut_size_y"):
            self.state_dict[key] = list()
        for _ in range(n_cuts):
            cut_size_x = CutOut._sample_size(self.cutout_size[0], w)
            cut_size_y = CutOut._sample_size(self.cutout_size[1], h)
            self.state_dict["cut_size_x"].append(cut_size_x)
            self.state_dict["cut_size_y"].append(cut_size_y)
            self.state_dict["x"].append(int(random.random() * (w - cut_size_x)))
            self.state_dict["y"].append(int(random.random() * (h - cut_size_y)))

    def __cutout_img(self, img):
        for x, y, cut_size_x, cut_size_y in zip(
            self.state_dict["x"], self.state_dict["y"], self.state_dict["cut_size_x"], self.state_dict["cut_size_y"]
        ):
            img[y : y + cut_size_y, x : x + cut_size_x] = 0
        return img

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = self.__cutout_img(img)
        return img


class GridMask(ImageTransform):
    """
    Performs a GridMask augmentation.

    https://arxiv.org/abs/2001.04086

    The image is split into units of ``d x d`` pixels. Every unit has a kept band of width ``l = ceil(d * ratio)``
    along its top and left sides, which forms a grid of kept lines, and the remaining square of side ``d - l`` is
    dropped (set to 0). Without rotation, approximately ``(1 - ratio) ** 2`` of the image is dropped (exactly
    ``((d - l) / d) ** 2``, up to the incomplete units at the borders), i.e. ``ratio`` is the ratio ``r`` of the paper,
    and the keep ratio is ``1 - (1 - ratio) ** 2``. The grid has a random offset, and the same mask is applied to all
    the channels of the image.

    Parameters
    ----------
    d_range: tuple or None
        The range of the unit size d. The d is chosen between ``(d_range[0], d_range[1])``, both inclusive.
        If None, then the d is chosen between ``(1, 10)``.
    ratio: float
        The ratio of the width of the kept lines to the unit size d, between 0 and 1. The dropped squares have the side
        ``d - ceil(d * ratio)``, i.e. ``0`` drops the whole image and ``1`` keeps it.
    rotate : int or tuple or None
        If int, then the angle (in degrees) of grid mask rotation is between ``(-rotate, rotate)``.
        If tuple, then the angle of grid mask rotation is between ``(rotate[0], rotate[1])``.
        The integer angle is chosen with both ends inclusive. If 0 or None, then the mask is not rotated.
    mode : str or None
        GridMask mode. If ``'crop'`` or ``'none'`` (default, used when ``mode`` is None), then the default GridMask
        is applied: the squares are dropped, and the grid lines are kept.
        If ``'reverse'``, the inversed GridMask is applied: only the squares are kept, and the grid lines are dropped,
        i.e. exactly the complement of the default mode is dropped. ``'reserve'`` is an alias of ``'reverse'``,
        kept for backward compatibility.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.
    p : float
        Probability of applying this transform.

    Notes
    -----
    To avoid copying, the transform modifies the image it receives in place. If the input image must stay unchanged,
    e.g. because it is cached or augmented several times, use ``Stream(..., in_place=False)`` or pass a copy.
    """

    serializable_name = "gridmask"
    """How the class should be stored in the registry"""

    def __init__(self, d_range=None, ratio=0.6, rotate=0, mode=None, data_indices=None, p=0.5):
        super(GridMask, self).__init__(p=p, data_indices=data_indices)

        if d_range is None:
            d_range = (1, 10)

        self.d_range = validate_numeric_range_parameter(d_range, (1, np.inf), min_val=1)

        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be between 0 and 1")

        self.ratio = ratio

        if rotate is None:
            rotate = 0

        if not isinstance(rotate, int) and not isinstance(rotate, tuple):
            raise TypeError("Rotate must be an int or a tuple of int")

        if isinstance(rotate, tuple):
            if len(rotate) != 2:
                raise ValueError("Rotate must be a tuple of two ints")
            if not isinstance(rotate[0], int) or not isinstance(rotate[1], int):
                raise TypeError("Rotate must be an int or a tuple of int")
            if rotate[0] > rotate[1]:
                raise ValueError("Rotate must be a range: rotate[0] <= rotate[1]")
        elif rotate < 0:
            raise ValueError("Rotate must be non-negative, when it is an int")

        if isinstance(rotate, int) and rotate:
            rotate = (-rotate, rotate)

        self.rotate = rotate

        self.mode = validate_parameter(mode, ALLOWED_GRIDMASK_MODES, "none", heritable=False)

    def sample_transform(self, data: DataContainer):
        h, w = super(GridMask, self).sample_transform(data)

        # The mask is made for the diagonal, so that it covers the image after any rotation around the center
        hh = int(np.ceil(np.sqrt(h**2 + w**2)))
        d = random.randint(self.d_range[0], self.d_range[1])

        # 1 on the kept grid lines, and 0 on the dropped squares between them
        mask = np.zeros((hh, hh), np.float32)
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        # Rounding makes e.g. 10 * 0.3 = 3.0000000000000004 a line of width 3
        line = int(np.ceil(np.round(d * self.ratio, 6)))

        for i in range(-1, hh // d + 1):
            s_row = max(min(d * i + st_h, hh), 0)
            t_row = max(min(d * i + st_h + line, hh), 0)
            mask[s_row:t_row, :] = 1

            s_col = max(min(d * i + st_w, hh), 0)
            t_col = max(min(d * i + st_w + line, hh), 0)
            mask[:, s_col:t_col] = 1

        if self.rotate:
            angle = random.randint(self.rotate[0], self.rotate[1])
            rotation_matrix = cv2.getRotationMatrix2D((hh // 2, hh // 2), angle, 1)
            mask = cv2.warpAffine(mask, rotation_matrix, (hh, hh), flags=cv2.INTER_NEAREST)

        mask = mask[(hh - h) // 2 : (hh - h) // 2 + h, (hh - w) // 2 : (hh - w) // 2 + w]
        if self.mode in ("reverse", "reserve"):
            mask = 1 - mask

        self.state_dict["mask"] = mask.astype(np.uint8)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if img.ndim == 2:
            img *= self.state_dict["mask"]
            return img

        for k in range(img.shape[2]):
            img[:, :, k] *= self.state_dict["mask"]

        return img
