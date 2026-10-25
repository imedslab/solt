"""Transforms, which change the size or the layout of the data: flipping, padding, cropping and resizing."""

import random
import math

import cv2
import numpy as np

from ..core import (
    BaseTransform,
    DataContainer,
    InterpolationPropertyHolder,
    Keypoints,
    PaddingPropertyHolder,
)
from ..constants import (
    ALLOWED_CROPS,
    ALLOWED_INTERPOLATIONS,
    ALLOWED_PADDINGS,
)
from ..utils import (
    distance_map_border,
    distance_map_handler,
    img_shape_checker,
    validate_numeric_range_parameter,
)


class Flip(BaseTransform):
    """
    Random Flipping transform.

    Parameters
    ----------
    p : float
        Probability of flip
    axis : int
        Axis of flip. Here, 1 stands for horizontal flipping, 0 stands for the vertical one. -1 stands for
        both axes.
    """

    serializable_name = "flip"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, axis=1, data_indices=None):
        super(Flip, self).__init__(p=p, data_indices=data_indices)
        if axis not in [-1, 0, 1]:
            raise ValueError("Incorrect Value of axis!")

        self.axis = axis

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.axis == 0:
            return np.ascontiguousarray(img[::-1, ...])
        elif self.axis == 1:
            if img.shape[2] > 1 and img.dtype == np.uint8:
                return cv2.flip(img, 1)
            return np.ascontiguousarray(img[:, ::-1, ...])
        else:
            return np.ascontiguousarray(img[::-1, ::-1, ...])

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        mask_new = cv2.flip(mask, self.axis)
        return mask_new

    @distance_map_handler
    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        return cv2.flip(dmap, self.axis)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        # We should guarantee that we do not change the original data.
        pts_data = pts.data.copy()
        if self.axis == 0:
            pts_data[:, 1] = pts.height - 1 - pts_data[:, 1]
        elif self.axis == 1:
            pts_data[:, 0] = pts.width - 1 - pts_data[:, 0]
        elif self.axis == -1:
            pts_data[:, 1] = pts.height - 1 - pts_data[:, 1]
            pts_data[:, 0] = pts.width - 1 - pts_data[:, 0]

        return Keypoints(pts=pts_data, height=pts.height, width=pts.width)


class Pad(BaseTransform, PaddingPropertyHolder):
    """
    Transformation, which pads the input to a given size

    Parameters
    ----------
    pad_to : tuple or int or None
        Target size ``(width_new, height_new)``.
        The padding is computed using the following equations:

        ``left_pad = (pad_to[0] - w) // 2``
        ``right_pad = (pad_to[0] - w) // 2 + (pad_to[0] - w) % 2``
        ``top_pad = (pad_to[1] - h) // 2``
        ``bottom_pad = (pad_to[1] - h) // 2 + (pad_to[1] - h) % 2``

    padding : str
        Padding type.

    See also
    --------
    solt.constants.allowed_paddings

    """

    serializable_name = "pad"
    """How the class should be stored in the registry"""

    def __init__(self, pad_to=None, padding=None):
        BaseTransform.__init__(self, p=1)
        PaddingPropertyHolder.__init__(self, padding)

        if not isinstance(pad_to, (tuple, list, int)) and (pad_to is not None):
            raise TypeError("The argument pad_to has to be tuple, list or None!")
        if isinstance(pad_to, int):
            pad_to = (pad_to, pad_to)

        self.pad_to = pad_to

    def sample_transform(self, data: DataContainer):
        if self.pad_to is not None:
            h, w = super(Pad, self).sample_transform(data)

            pad_w = (self.pad_to[0] - w) // 2
            pad_h = (self.pad_to[1] - h) // 2

            pad_h_top = pad_h
            pad_h_bottom = pad_h + (self.pad_to[1] - h) % 2

            pad_w_left = pad_w
            pad_w_right = pad_w + (self.pad_to[0] - w) % 2

            if pad_h < 0:
                pad_h_top = 0
                pad_h_bottom = 0

            if pad_w < 0:
                pad_w_left = 0
                pad_w_right = 0

            self.state_dict = {
                "pad_h": (pad_h_top, pad_h_bottom),
                "pad_w": (pad_w_left, pad_w_right),
            }

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict, is_distance_map=False):
        if self.pad_to is None:
            return img
        pad_h_top, pad_h_bottom = self.state_dict["pad_h"]
        pad_w_left, pad_w_right = self.state_dict["pad_w"]
        padding = ALLOWED_PADDINGS[self.padding[0]]

        if settings["padding"][1] == "strict":
            padding = ALLOWED_PADDINGS[settings["padding"][0]]

        value = 0
        if is_distance_map:
            padding, value = distance_map_border(padding, settings)

        return cv2.copyMakeBorder(img, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, padding, value=value)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    @distance_map_handler
    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        return self._apply_img_or_mask(dmap, settings, is_distance_map=True)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        if self.pad_to is None:
            return pts
        if self.padding[0] != "z":
            raise ValueError("Padding must be 'z'")
        pts_data = pts.data.copy()

        pad_h_top, pad_h_bottom = self.state_dict["pad_h"]
        pad_w_left, pad_w_right = self.state_dict["pad_w"]

        pts_data[:, 0] += pad_w_left
        pts_data[:, 1] += pad_h_top

        return Keypoints(
            pts_data,
            pad_h_top + pts.height + pad_h_bottom,
            pad_w_left + pts.width + pad_w_right,
        )


class Resize(BaseTransform, InterpolationPropertyHolder):
    """
    Transformation, which resizes the input to a given size

    Parameters
    ----------
    resize_to : list or tuple or int or None
        Target size ``(width_new, height_new)``. If ``int``, the output is square, unless ``keep_ar`` is ``True``.
    interpolation : str
        Interpolation type.
    keep_ar : bool
        Whether or not to keep the aspect ratio for non-squared images. Only used when ``resize_to`` is an ``int``.
    height_first : bool
        Used only if ``keep_ar`` is ``True`` and ``resize_to`` is an ``int``. If ``True``, the height is resized to
        ``resize_to`` and the width is computed from the original aspect ratio. If ``False``, the width is resized
        to ``resize_to`` and the height is computed from the original aspect ratio.

    See also
    --------
    solt.constants.allowed_interpolations

    """

    serializable_name = "resize"
    """How the class should be stored in the registry"""

    def __init__(self, resize_to=None, interpolation="bilinear", keep_ar=False, height_first=True):
        BaseTransform.__init__(self, p=1)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        if resize_to is not None:
            if not isinstance(resize_to, (tuple, list, int)) and (resize_to is not None):
                raise TypeError("The argument resize_to has an incorrect type!")

        self.resize_to = resize_to
        self.keep_ar = keep_ar
        self.height_first = height_first

    def _handle_resize(self, h, w):
        resize_to = self.resize_to
        # Handling the non-square case
        if isinstance(resize_to, int):
            if h != w and self.keep_ar:
                # resize_to is (width, height)
                if self.height_first:
                    resize_to = (int(self.resize_to * w / h), self.resize_to)
                else:
                    resize_to = (self.resize_to, int(self.resize_to * h / w))
            else:
                resize_to = (resize_to, resize_to)

        return resize_to

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        if self.resize_to is None:
            return img
        interp = ALLOWED_INTERPOLATIONS[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS[settings["interpolation"][0]]

        resize_to = self._handle_resize(img.shape[0], img.shape[1])

        return cv2.resize(img, resize_to, interpolation=interp)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    @distance_map_handler
    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        if self.resize_to is None:
            return dmap
        h, w = dmap.shape[:2]
        resize_to = self._handle_resize(h, w)
        # Distances change by the square root of the area change
        scale = math.sqrt(resize_to[0] / w * resize_to[1] / h)
        return self._apply_img_or_mask(dmap * scale, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        if self.resize_to is None:
            return pts
        resize_x, resize_y = self._handle_resize(pts.height, pts.width)
        # resize_points returns new coordinates, the input ones stay unchanged
        pts_data = Resize.resize_points(pts.data, resize_x / pts.width, resize_y / pts.height)
        return Keypoints(pts_data, resize_y, resize_x)

    @staticmethod
    def resize_points(pts_data, scale_x, scale_y):
        """
        Resizes the coordinates of points (``Nx2``, x and y). The points are the centres of pixels, so the same
        mapping as in ``cv2.resize`` keeps them aligned with the image.
        """
        # astype makes a copy, so the input coordinates stay unchanged
        pts_data = pts_data.astype(float)
        pts_data[:, 0] = (pts_data[:, 0] + 0.5) * scale_x - 0.5
        pts_data[:, 1] = (pts_data[:, 1] + 0.5) * scale_y - 0.5
        return pts_data


class Crop(BaseTransform):
    """
    Center / Random / Corner crop transform.

    Object performs center, random or corner cropping depending on the parameters. The center and the four corner
    crops make the five crops, which are often used for test-time augmentation.

    Parameters
    ----------
    crop_to : tuple or int or None
        Size of the crop ``(width_new, height_new)``. If ``int``, then a square crop will be made.
    crop_mode : str
        Crop mode. Can be ``'c'`` - center, ``'r'`` - random, or a corner: ``'lt'`` - left top, ``'rt'`` - right top,
        ``'lb'`` - left bottom and ``'rb'`` - right bottom.

    See also
    --------
    solt.constants.ALLOWED_CROPS

    Notes
    -----
    To avoid copying, the cropped images, masks and distance maps are views of the input arrays.

    """

    serializable_name = "crop"
    """How the class should be stored in the registry"""

    def __init__(self, crop_to=None, crop_mode="c"):
        super(Crop, self).__init__(p=1, data_indices=None)

        if crop_to is not None:
            if not isinstance(crop_to, (int, tuple, list)):
                raise TypeError("Argument crop_to has an incorrect type!")
            if crop_mode not in ALLOWED_CROPS:
                raise ValueError("Argument crop_mode has an incorrect type!")

            if isinstance(crop_to, list):
                crop_to = tuple(crop_to)

            if isinstance(crop_to, tuple):
                if not isinstance(crop_to[0], int) or not isinstance(crop_to[1], int):
                    raise TypeError("Incorrect type of the crop_to!")

            if isinstance(crop_to, int):
                crop_to = (crop_to, crop_to)

        self.crop_to = crop_to
        self.crop_mode = crop_mode

    def sample_transform(self, data: DataContainer):
        h, w = super(Crop, self).sample_transform(data)
        if self.crop_to is not None:
            if self.crop_to[0] > w or self.crop_to[1] > h:
                raise ValueError("Crop size must be less than the image size")

            if self.crop_mode == "r":
                self.state_dict["x"] = int(random.random() * (w - self.crop_to[0]))
                self.state_dict["y"] = int(random.random() * (h - self.crop_to[1]))
            elif self.crop_mode == "c":
                self.state_dict["x"] = w // 2 - self.crop_to[0] // 2
                self.state_dict["y"] = h // 2 - self.crop_to[1] // 2
            else:
                # Corner crops: the first letter is the horizontal position, the second one is the vertical one
                self.state_dict["x"] = 0 if self.crop_mode[0] == "l" else w - self.crop_to[0]
                self.state_dict["y"] = 0 if self.crop_mode[1] == "t" else h - self.crop_to[1]

    def __crop_img_or_mask(self, img_mask):
        if self.crop_to is not None:
            return img_mask[
                self.state_dict["y"] : self.state_dict["y"] + self.crop_to[1],
                self.state_dict["x"] : self.state_dict["x"] + self.crop_to[0],
            ]
        return img_mask

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(img)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(mask)

    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(dmap)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        if self.crop_to is None:
            return pts
        pts_data = pts.data.copy()
        x, y = self.state_dict["x"], self.state_dict["y"]

        pts_data[:, 0] -= x
        pts_data[:, 1] -= y

        return Keypoints(pts_data, self.crop_to[1], self.crop_to[0])


class RandomResizedCrop(BaseTransform, InterpolationPropertyHolder):
    """Random resized crop transform.

    Random cropping of a random size with a random aspect ratio, and then resizes the crop to a
    target size.

    Parameters
    ----------
    resize_to : tuple or int or None
        Size of the crop ``(width_new, height_new)``. If ``int``, then a square crop will be made.
    scale : tuple of float or None
        Range of size of the origin size cropped. If None, then default Imagenet values are used -- ``(0.08, 1.0)``.
    ratio : tuple of float or None
        Range of the aspect ratio. If None, then default Imagenet values are used -- ``(3 / 4., 4 / 3.)``.

    """

    serializable_name = "random_resized_crop"
    """How the class should be stored in the registry"""

    def __init__(self, resize_to, scale=None, ratio=None, interpolation="bilinear"):
        BaseTransform.__init__(self, p=1)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        self.scale = validate_numeric_range_parameter(scale, (0.08, 1.0), min_val=0, max_val=1)
        self.ratio = validate_numeric_range_parameter(ratio, (0.75, 1.3333333333333333), min_val=0)
        if self.ratio[0] == 0:
            raise ValueError("Aspect ratio must be positive!")
        if resize_to is not None:
            if not isinstance(resize_to, (int, tuple, list)):
                raise TypeError("Argument crop_to has an incorrect type!")

            if isinstance(resize_to, list):
                resize_to = tuple(resize_to)

            if isinstance(resize_to, tuple):
                if not isinstance(resize_to[0], int) or not isinstance(resize_to[1], int):
                    raise TypeError("Incorrect type of the crop_to!")

            if isinstance(resize_to, int):
                resize_to = (resize_to, resize_to)

        self.resize_to = resize_to

    def sample_transform(self, data: DataContainer):
        h, w = super(RandomResizedCrop, self).sample_transform(data)
        area = h * w

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            new_w = int(round(math.sqrt(target_area * aspect_ratio)))
            new_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < new_w <= w and 0 < new_h <= h:
                i = random.randint(0, h - new_h)
                j = random.randint(0, w - new_w)
                self.state_dict.update(
                    {
                        "crop_to": (new_w, new_h),
                        "y": i,
                        "x": j,
                    }
                )
                return

        # Fallback to central crop
        new_h, new_w = h, w
        in_ratio = w / h
        if in_ratio < min(self.ratio):
            new_w = w
            new_h = max(1, int(round(new_w / min(self.ratio))))
        elif in_ratio > max(self.ratio):
            new_h = h
            new_w = max(1, int(round(new_h * max(self.ratio))))

        i = (h - new_h) // 2
        j = (w - new_w) // 2

        self.state_dict.update(
            {
                "crop_to": (new_w, new_h),
                "y": i,
                "x": j,
            }
        )

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        img = img[
            self.state_dict["y"] : self.state_dict["y"] + self.state_dict["crop_to"][1],
            self.state_dict["x"] : self.state_dict["x"] + self.state_dict["crop_to"][0],
        ]
        if self.resize_to is None:
            return img
        interp = ALLOWED_INTERPOLATIONS[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS[settings["interpolation"][0]]

        return cv2.resize(img, self.resize_to, interpolation=interp)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    @distance_map_handler
    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        if self.resize_to is not None:
            w_crop, h_crop = self.state_dict["crop_to"]
            # Distances change by the square root of the area change
            dmap = dmap * math.sqrt(self.resize_to[0] / w_crop * self.resize_to[1] / h_crop)
        return self._apply_img_or_mask(dmap, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        crop_w, crop_h = self.state_dict["crop_to"]
        pts_data = pts.data.astype(float)
        pts_data[:, 0] -= self.state_dict["x"]
        pts_data[:, 1] -= self.state_dict["y"]
        if self.resize_to is None:
            return Keypoints(pts_data, crop_h, crop_w)

        new_w, new_h = self.resize_to
        return Keypoints(Resize.resize_points(pts_data, new_w / crop_w, new_h / crop_h), new_h, new_w)
