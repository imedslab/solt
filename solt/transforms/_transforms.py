import random
from functools import wraps

import cv2
import numpy as np
import scipy
import scipy.signal
import math

from ..core import (
    BaseTransform,
    ImageTransform,
    InterpolationPropertyHolder,
    MatrixTransform,
    PaddingPropertyHolder,
)
from ..constants import (
    ALLOWED_BLURS,
    ALLOWED_COLOR_CONVERSIONS,
    ALLOWED_CROPS,
    ALLOWED_INTERPOLATIONS,
    ALLOWED_PADDINGS,
    DTYPES_MAX,
    ALLOWED_GRIDMASK_MODES,
)
from ..core import Stream
from ..core import DataContainer, Keypoints
from ..utils import (
    distance_map_border,
    distance_map_handler,
    img_shape_checker,
    validate_numeric_range_parameter,
    validate_parameter,
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


class Rotate(MatrixTransform):
    """
    Random rotation around the center clockwise

    Parameters
    ----------
    angle_range : tuple or float or None
        Range of rotation.
        If float, then (-angle_range, angle_range) will be used for transformation sampling.
        if None, then angle_range=(0,0).
    interpolation : str or tuple or None
        Interpolation type. Check the allowed interpolation types.
    padding : str or tuple or None
        Padding mode. Check the allowed padding modes.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    _default_range = (0, 0)

    serializable_name = "rotate"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        angle_range=None,
        interpolation="bilinear",
        padding="z",
        p=0.5,
        ignore_state=True,
        ignore_fast_mode=False,
    ):
        super(Rotate, self).__init__(
            interpolation=interpolation,
            padding=padding,
            p=p,
            ignore_state=ignore_state,
            affine=True,
            ignore_fast_mode=ignore_fast_mode,
        )
        if isinstance(angle_range, (int, float)):
            angle_range = (-angle_range, angle_range)

        self.angle_range = validate_numeric_range_parameter(angle_range, self._default_range)

    def sample_angle(self):
        self.state_dict["rot"] = np.deg2rad(random.uniform(self.angle_range[0], self.angle_range[1]))
        return self.state_dict["rot"]

    def sample_transform_matrix(self, data):
        """
        Samples random rotation within specified range and saves it as an object state.

        """
        self.sample_angle()

        self.state_dict["transform_matrix"][0, 0] = np.cos(self.state_dict["rot"])
        self.state_dict["transform_matrix"][0, 1] = -np.sin(self.state_dict["rot"])
        self.state_dict["transform_matrix"][0, 2] = 0

        self.state_dict["transform_matrix"][1, 0] = np.sin(self.state_dict["rot"])
        self.state_dict["transform_matrix"][1, 1] = np.cos(self.state_dict["rot"])
        self.state_dict["transform_matrix"][1, 2] = 0

        self.state_dict["transform_matrix"][2, 0] = 0
        self.state_dict["transform_matrix"][2, 1] = 0
        self.state_dict["transform_matrix"][2, 2] = 1


class Rotate90(Rotate):
    """
    Random rotation around the center by 90 degrees.

    Parameters
    ----------
    k : int
        How many times to rotate the data. If positive, indicates the clockwise direction.
        Zero by default.
    p : float
        Probability of using this transform

    """

    serializable_name = "rotate_90"
    """How the class should be stored in the registry"""

    def __init__(self, k=0, p=0.5, ignore_fast_mode=False):
        if not isinstance(k, int):
            raise TypeError("Argument `k` must be an integer!")
        super(Rotate90, self).__init__(p=p, angle_range=(k * 90, k * 90), ignore_fast_mode=ignore_fast_mode)
        self.k = k

    def reset_state(self):
        super(Rotate90, self).reset_state()
        self.state_dict["fused"] = False

    def fuse_with(self, trf):
        super(Rotate90, self).fuse_with(trf)
        self.state_dict["fused"] = True

    @staticmethod
    def _exact_unless_fused(method):
        """
        Decorator for the methods, which rotate the data exactly by 90 degrees. When other transforms are fused into
        this one (Stream with optimize_stack), the rotation is only a part of the transform matrix, so the general
        matrix transform is applied instead.
        """
        matrix_method = getattr(MatrixTransform, method.__name__)

        @wraps(method)
        def wrapper(self, item, settings):
            if self.state_dict.get("fused", False):
                return matrix_method(self, item, settings)
            return method(self, item, settings)

        return wrapper

    @img_shape_checker
    @_exact_unless_fused
    def _apply_img(self, img: np.ndarray, settings: dict):
        return np.ascontiguousarray(np.rot90(img, -self.k))

    @_exact_unless_fused
    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return np.ascontiguousarray(np.rot90(mask, -self.k))

    @_exact_unless_fused
    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        return np.ascontiguousarray(np.rot90(dmap, -self.k))

    @_exact_unless_fused
    def _apply_pts(self, pts: Keypoints, settings: dict):
        # The exact mapping of np.rot90, which rotates the images. A positive k rotates clockwise.
        x, y = pts.data[:, 0].astype(float), pts.data[:, 1].astype(float)
        h, w = pts.height, pts.width
        k = self.k % 4
        if k == 0:
            pts_data, h_new, w_new = np.stack([x, y], axis=1), h, w
        elif k == 1:
            pts_data, h_new, w_new = np.stack([h - 1 - y, x], axis=1), w, h
        elif k == 2:
            pts_data, h_new, w_new = np.stack([w - 1 - x, h - 1 - y], axis=1), h, w
        else:
            pts_data, h_new, w_new = np.stack([y, w - 1 - x], axis=1), w, h
        return Keypoints(pts_data, h_new, w_new)


class Shear(MatrixTransform):
    """
    Random shear around the center.

    Parameters
    ----------
    range_x : tuple or float or None
        Shearing range along X-axis.
        If float, then (-range_x, range_x) will be used.
        If None, then range_x=(0, 0)
    range_y : tuple or float or None
        Shearing range along Y-axis. If float, then (-range_y, range_y) will be used.
        If None, then range_y=(0, 0)
    interpolation : str or tuple or None or tuple or None
        Interpolation type. Check the allowed interpolation types.
    padding : str
        Padding mode. Check the allowed padding modes.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    _default_range = (0, 0)

    serializable_name = "shear"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        range_x=None,
        range_y=None,
        interpolation="bilinear",
        padding="z",
        p=0.5,
        ignore_state=True,
        ignore_fast_mode=False,
    ):
        super(Shear, self).__init__(
            p=p,
            padding=padding,
            interpolation=interpolation,
            ignore_state=ignore_state,
            affine=True,
            ignore_fast_mode=ignore_fast_mode,
        )
        if isinstance(range_x, (int, float)):
            range_x = (-range_x, range_x)

        if isinstance(range_y, (int, float)):
            range_y = (-range_y, range_y)

        self.range_x = validate_numeric_range_parameter(range_x, self._default_range)
        self.range_y = validate_numeric_range_parameter(range_y, self._default_range)

    def sample_shear(self):
        self.state_dict["shear_x"] = random.uniform(self.range_x[0], self.range_x[1])
        self.state_dict["shear_y"] = random.uniform(self.range_y[0], self.range_y[1])
        return self.state_dict["shear_x"], self.state_dict["shear_y"]

    def sample_transform_matrix(self, data):
        shear_x, shear_y = self.sample_shear()

        self.state_dict["transform_matrix"][0, 0] = 1
        self.state_dict["transform_matrix"][0, 1] = shear_x
        self.state_dict["transform_matrix"][0, 2] = 0

        self.state_dict["transform_matrix"][1, 0] = shear_y
        self.state_dict["transform_matrix"][1, 1] = 1
        self.state_dict["transform_matrix"][1, 2] = 0

        self.state_dict["transform_matrix"][2, 0] = 0
        self.state_dict["transform_matrix"][2, 1] = 0
        self.state_dict["transform_matrix"][2, 2] = 1


class Scale(MatrixTransform):
    """
    Random scale transform.

    Parameters
    ----------
    range_x : tuple or float or None
        Scaling range along X-axis.
        If float, then (min(1, range_x), max(1, range_x)) will be used.
        If None, then range_x =(1,1) by default.
    range_y : tuple or float
        Scaling range along Y-axis.
        If float, then (min(1, range_y), max(1, range_y)) will be used.
        If None, then range_y=(1,1) by default.
    same: bool
        Indicates whether to use the same scaling factor for width and height.
    interpolation : str or tuple or None
        Interpolation type. Check the allowed interpolation types.
        one indicates default behavior - the ``bilinear`` mode.
    padding : str
        ``z`` (zero pad) or ``r`` - reflective pad.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    _default_range = (1, 1)

    serializable_name = "scale"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        range_x=None,
        range_y=None,
        same=True,
        interpolation="bilinear",
        p=0.5,
        ignore_state=True,
        padding=None,
        ignore_fast_mode=False,
    ):

        super(Scale, self).__init__(
            interpolation=interpolation,
            padding=padding,
            p=p,
            ignore_state=ignore_state,
            affine=True,
            ignore_fast_mode=ignore_fast_mode,
        )

        if isinstance(range_x, (int, float)):
            range_x = (min(1, range_x), max(1, range_x))

        if isinstance(range_y, (int, float)):
            range_y = (min(1, range_y), max(1, range_y))

        self.same = same
        self.range_x = (
            None if range_x is None else validate_numeric_range_parameter(range_x, self._default_range, min_val=0)
        )
        self.range_y = (
            None if range_y is None else validate_numeric_range_parameter(range_y, self._default_range, min_val=0)
        )

    def sample_scale(self):
        if self.range_x is None:
            scale_x = 1
        else:
            scale_x = random.uniform(self.range_x[0], self.range_x[1])

        if self.range_y is None:
            scale_y = 1
        else:
            scale_y = random.uniform(self.range_y[0], self.range_y[1])

        if self.same:
            if self.range_x is None:
                scale_x = scale_y
            else:
                scale_y = scale_x

        self.state_dict["scale_x"] = scale_x
        self.state_dict["scale_y"] = scale_y

        return scale_x, scale_y

    def sample_transform_matrix(self, data):
        scale_x, scale_y = self.sample_scale()

        self.state_dict["transform_matrix"][0, 0] = scale_x
        self.state_dict["transform_matrix"][0, 1] = 0
        self.state_dict["transform_matrix"][0, 2] = 0

        self.state_dict["transform_matrix"][1, 0] = 0
        self.state_dict["transform_matrix"][1, 1] = scale_y
        self.state_dict["transform_matrix"][1, 2] = 0

        self.state_dict["transform_matrix"][2, 0] = 0
        self.state_dict["transform_matrix"][2, 1] = 0
        self.state_dict["transform_matrix"][2, 2] = 1


class Translate(MatrixTransform):
    """
    Random Translate transform..

    Parameters
    ----------
    range_x: tuple or int or None
        Translation range along the horizontal axis. If int, then range_x=(-range_x, range_x).
        If None, then range_x=(0,0).
    range_y: tuple or int or None
        Translation range along the vertical axis. If int, then range_y=(-range_y, range_y).
        If None, then range_y=(0,0).
    interpolation: str
        Interpolation type. See allowed_interpolations in constants.
    padding: str
        Padding mode. See allowed_paddings  in constants
    p: float
        probability of applying this transform.
    """

    _default_range = (0, 0)

    serializable_name = "translate"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        range_x=None,
        range_y=None,
        interpolation="bilinear",
        padding="z",
        p=0.5,
        ignore_state=True,
        ignore_fast_mode=False,
    ):
        super(Translate, self).__init__(
            interpolation=interpolation,
            padding=padding,
            p=p,
            ignore_state=ignore_state,
            affine=True,
            ignore_fast_mode=ignore_fast_mode,
        )
        if isinstance(range_x, (int, float)):
            range_x = (min(range_x, -range_x), max(range_x, -range_x))

        if isinstance(range_y, (int, float)):
            range_y = (min(range_y, -range_y), max(range_y, -range_y))

        self.range_x = validate_numeric_range_parameter(range_x, self._default_range)
        self.range_y = validate_numeric_range_parameter(range_y, self._default_range)

    def sample_translate(self):
        self.state_dict["translate_x"] = random.uniform(self.range_x[0], self.range_x[1])
        self.state_dict["translate_y"] = random.uniform(self.range_y[0], self.range_y[1])
        return self.state_dict["translate_x"], self.state_dict["translate_y"]

    def sample_transform_matrix(self, data):
        tx, ty = self.sample_translate()

        self.state_dict["transform_matrix"][0, 0] = 1
        self.state_dict["transform_matrix"][0, 1] = 0
        self.state_dict["transform_matrix"][0, 2] = tx

        self.state_dict["transform_matrix"][1, 0] = 0
        self.state_dict["transform_matrix"][1, 1] = 1
        self.state_dict["transform_matrix"][1, 2] = ty

        self.state_dict["transform_matrix"][2, 0] = 0
        self.state_dict["transform_matrix"][2, 1] = 0
        self.state_dict["transform_matrix"][2, 2] = 1


class Projection(MatrixTransform):
    """
    Random Projective transform.

    Takes a set of affine transforms.

    Parameters
    ----------
    affine_transforms : Stream or None
        Stream object, which has a parameterized Affine Transform.
        If None, then a zero degrees rotation matrix is instantiated.
    v_range : tuple or None
        Projective parameters range. If None, then ``v_range = (0, 0)``
    p : float
        Probability of using this transform.
    """

    _default_range = (0, 0)

    serializable_name = "projection"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        affine_transforms=None,
        v_range=None,
        interpolation="bilinear",
        padding="z",
        p=0.5,
        ignore_state=True,
        ignore_fast_mode=False,
    ):

        super(Projection, self).__init__(
            interpolation=interpolation,
            padding=padding,
            p=p,
            ignore_state=ignore_state,
            affine=False,
            ignore_fast_mode=ignore_fast_mode,
        )

        if affine_transforms is None:
            affine_transforms = Stream()

        if not isinstance(affine_transforms, Stream):
            raise TypeError("Affine transforms must be a Stream")
        for trf in affine_transforms.transforms:
            if not isinstance(trf, MatrixTransform):
                raise TypeError("Affine transforms must be a MatrixTransform")

        self.affine_transforms = affine_transforms
        self.vrange = validate_numeric_range_parameter(v_range, self._default_range)  # projection components.

    def sample_transform_matrix(self, data):
        if len(self.affine_transforms.transforms) > 1:
            trf = Stream.optimize_transforms_stack(self.affine_transforms.transforms, data)
            if len(trf) == 0:
                trf = None
            else:
                trf = trf[0]
        elif len(self.affine_transforms.transforms) == 0:
            trf = None
        else:
            trf = self.affine_transforms.transforms[0]
            trf.sample_transform(data)

        if trf is None:
            transform_matrix = np.eye(3)
        else:
            transform_matrix = trf.state_dict["transform_matrix"].copy()

        transform_matrix[-1, 0] = random.uniform(self.vrange[0], self.vrange[1])
        transform_matrix[-1, 1] = random.uniform(self.vrange[0], self.vrange[1])
        self.state_dict["transform_matrix"] = transform_matrix


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


class Noise(BaseTransform):
    """
    Adds noise to an image. Other types of data than the image are ignored.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gain_range : tuple or float or None
        Gain of the noise. Final image is created as ``(1-gain)*img + gain*noise``.
        If float, then ``gain_range = (0, gain_range)``. If None, then ``gain_range=(0, 0)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (0, 0)

    serializable_name = "noise"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, gain_range=0.1, data_indices=None):
        super(Noise, self).__init__(p=p, data_indices=data_indices)
        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        self.gain_range = validate_numeric_range_parameter(gain_range, self._default_range, min_val=0, max_val=1)

    def sample_transform(self, data: DataContainer):
        super(Noise, self).sample_transform(data)
        gain = random.uniform(self.gain_range[0], self.gain_range[1])
        h = None
        w = None
        c = None
        obj = None
        for obj, t, _ in data:
            if t == "I":
                h = obj.shape[0]
                w = obj.shape[1]
                c = obj.shape[2]
                break

        if w is None or h is None or c is None:
            raise ValueError("Image must have 3 channels")

        random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
        noise_img = random_state.randn(h, w, c)

        noise_img -= noise_img.min()
        noise_img /= noise_img.max()
        noise_img *= 255
        noise_img = noise_img.astype(obj.dtype)

        self.state_dict = {"noise": noise_img, "gain": gain}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.addWeighted(
            img,
            (1 - self.state_dict["gain"]),
            self.state_dict["noise"],
            self.state_dict["gain"],
            0,
        )

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return mask

    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        return dmap

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        return pts


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


class SaltAndPepper(ImageTransform):
    """
    Adds salt and pepper noise to an image. Other types of data than the image are ignored.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gain_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then ``gain_range = (0, gain_range)``.
    salt_p : float or tuple or None
        Percentage of salt. Percentage of pepper is ``1-salt_p``. If tuple, then ``salt_p`` is chosen
        uniformly from the given range.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (0, 0)

    serializable_name = "salt_and_pepper"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, gain_range=0.1, salt_p=0.5, data_indices=None):
        super(SaltAndPepper, self).__init__(p=p, data_indices=data_indices)
        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        if isinstance(salt_p, float):
            salt_p = (salt_p, salt_p)

        self.gain_range = validate_numeric_range_parameter(gain_range, self._default_range, 0, 1)
        self.salt_p = validate_numeric_range_parameter(salt_p, self._default_range, 0, 1)

    def sample_transform(self, data: DataContainer):
        h, w = super(SaltAndPepper, self).sample_transform(data)
        gain = random.uniform(self.gain_range[0], self.gain_range[1])
        salt_p = random.uniform(self.salt_p[0], self.salt_p[1])

        random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
        sp = random_state.rand(h, w) <= gain
        salt = sp.copy() * 1.0
        pepper = sp.copy() * 1.0
        salt_mask = random_state.rand(sp.sum()) <= salt_p
        pepper_mask = 1 - salt_mask
        salt[np.where(salt)] *= salt_mask
        pepper[np.where(pepper)] *= pepper_mask

        self.state_dict = {"salt": salt, "pepper": pepper}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = img.copy()
        img[np.where(self.state_dict["salt"])] = DTYPES_MAX[img.dtype]
        img[np.where(self.state_dict["pepper"])] = 0
        return img


class GammaCorrection(ImageTransform):
    """
    Transform applies random gamma correction

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gamma_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then ``gain_range = (1-gamma_range, 1+gamma_range)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (1, 1)

    serializable_name = "gamma_correction"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, gamma_range=0.1, data_indices=None):
        super(GammaCorrection, self).__init__(p=p, data_indices=data_indices)

        if isinstance(gamma_range, float):
            gamma_range = (1 - gamma_range, 1 + gamma_range)

        self.gamma_range = validate_numeric_range_parameter(gamma_range, self._default_range, 0)

    def sample_transform(self, data):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        self.state_dict = {"gamma": gamma, "LUT": lut}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict["LUT"])


class Contrast(ImageTransform):
    """
    Transform randomly changes the contrast

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    contrast_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then ``gain_range = (1-contrast_range, 1+contrast_range)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (1, 1)
    serializable_name = "contrast"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, contrast_range=0.1, data_indices=None):
        super(Contrast, self).__init__(p=p, data_indices=data_indices)
        self.contrast_range = Contrast.parse_contrast_range(contrast_range)

    @staticmethod
    def parse_contrast_range(contrast_range):
        """Validates the contrast range. A float ``c`` becomes ``(1 - c, 1 + c)``."""
        if isinstance(contrast_range, float):
            contrast_range = (1 - contrast_range, 1 + contrast_range)
        return validate_numeric_range_parameter(contrast_range, Contrast._default_range, 0)

    def sample_transform(self, data):
        contrast_mul = random.uniform(self.contrast_range[0], self.contrast_range[1])
        lut = np.arange(0, 256) * contrast_mul
        lut = np.clip(lut, 0, 255).astype("uint8")
        self.state_dict = {"contrast_mul": contrast_mul, "LUT": lut}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict["LUT"])


class Blur(ImageTransform):
    """
    Transform blurs an image

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    blur_type : str
        Blur type. See allowed blurs in `solt.constants`
    k_size: int or tuple
        Kernel sizes of the blur. if int, then sampled from ``(k_size, k_size)``. If tuple,
        then sampled from the whole tuple. All the values here must be odd.
    gaussian_sigma: int or float or tuple
        Gaussian sigma value. Used for both X and Y axes. If None, then gaussian_sigma=1.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    See also
    --------
    solt.constants.ALLOWED_BLURS

    """

    _default_range = (1, 1)

    serializable_name = "blur"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, blur_type="g", k_size=3, gaussian_sigma=None, data_indices=None):
        super(Blur, self).__init__(p=p, data_indices=data_indices)
        if not isinstance(k_size, (int, tuple, list)):
            raise TypeError("Incorrect kernel size")

        if isinstance(k_size, list):
            k_size = tuple(k_size)

        if isinstance(k_size, int):
            k_size = (k_size, k_size)

        for k in k_size:
            if k % 2 == 0 or k < 1 or not isinstance(k, int):
                raise ValueError("Kernel size must be an odd integer")

        if isinstance(gaussian_sigma, (int, float)):
            gaussian_sigma = (gaussian_sigma, gaussian_sigma)

        self.blur = validate_parameter(blur_type, ALLOWED_BLURS, "g", basic_type=str, heritable=False)
        self.k_size = k_size
        self.gaussian_sigma = validate_numeric_range_parameter(gaussian_sigma, self._default_range, 0)

    def sample_transform(self, data):
        k = random.choice(self.k_size)
        s = random.uniform(self.gaussian_sigma[0], self.gaussian_sigma[1])
        self.state_dict = {"k_size": k, "sigma": s}

        if self.blur == "mo":
            if self.k_size[0] <= 2:
                raise ValueError("Lower bound for blur kernel size cannot be less than 2 for motion blur")

            kernel = np.zeros((k, k), dtype=np.uint8)
            xs, xe = random.randint(0, k - 1), random.randint(0, k - 1)

            if xs == xe:
                ys, ye = random.sample(range(k), 2)
            else:
                ys, ye = random.randint(0, k - 1), random.randint(0, k - 1)
            cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
            kernel = kernel / np.sum(kernel)
            self.state_dict.update({"motion_kernel": kernel})

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.blur == "g":
            return cv2.GaussianBlur(
                img,
                ksize=(self.state_dict["k_size"], self.state_dict["k_size"]),
                sigmaX=self.state_dict["sigma"],
            )
        if self.blur == "m":
            return cv2.medianBlur(img, ksize=self.state_dict["k_size"])

        if self.blur == "mo":
            return cv2.filter2D(img, -1, self.state_dict["motion_kernel"])


class HSV(ImageTransform):
    """
    Performs a random HSV color shift.

    Parameters
    ----------
    h_range: tuple or None
        Hue shift range. If None, than ``h_range=(0, 0)``.
    s_range: tuple or None
        Saturation shift range. If None, then ``s_range=(0, 0)``.
    v_range: tuple or None
        Value shift range. If None, then ``v_range=(0, 0)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.
    p : float
        Probability of applying this transform,

    """

    _default_range = (0, 0)

    serializable_name = "hsv"
    """How the class should be stored in the registry"""

    def __init__(self, h_range=None, s_range=None, v_range=None, data_indices=None, p=0.5):
        super(HSV, self).__init__(p=p, data_indices=data_indices)
        self.h_range = validate_numeric_range_parameter(h_range, self._default_range)
        self.s_range = validate_numeric_range_parameter(s_range, self._default_range)
        self.v_range = validate_numeric_range_parameter(v_range, self._default_range)

    def sample_transform(self, data):
        h = random.uniform(self.h_range[0], self.h_range[1])
        s = random.uniform(self.s_range[0], self.s_range[1])
        v = random.uniform(self.v_range[0], self.v_range[1])
        self.state_dict = {"h_mod": h, "s_mod": s, "v_mod": v}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = img.copy()
        dtype = img.dtype
        if img.shape[-1] != 3:
            raise ValueError("Image has to have 3 channels!")

        if dtype != np.uint8:
            raise TypeError("Image type has to be uint8 in this version of SOLT!")

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv.astype(np.int32))

        h = np.clip(abs((h + self.state_dict["h_mod"]) % 180), 0, 180).astype(dtype)
        s = np.clip(s + self.state_dict["s_mod"], 0, 255).astype(dtype)
        v = np.clip(v + self.state_dict["v_mod"], 0, 255).astype(dtype)

        img_hsv_shifted = cv2.merge((h, s, v))
        img = cv2.cvtColor(img_hsv_shifted, cv2.COLOR_HSV2RGB)

        return img


class Brightness(ImageTransform):
    """
    Performs a random brightness augmentation

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    brightness_range: tuple or None
        brightness_range shift range. If None, then ``brightness_range=(0, 0)``.
    relative : bool
        Whether the shift is relative to the mean intensity of the image. If ``True``, the values of
        ``brightness_range`` are fractions of the mean, e.g. ``(-0.2, 0.2)`` changes the brightness by up to 20% of
        the mean intensity of every image. ``False`` by default.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (0, 0)

    serializable_name = "brightness"
    """How the class should be stored in the registry"""

    def __init__(self, brightness_range=None, data_indices=None, p=0.5, relative=False):
        super(Brightness, self).__init__(p=p, data_indices=data_indices)
        self.brightness_range, self.relative = Brightness.parse_brightness(brightness_range, relative)

    @staticmethod
    def parse_brightness(brightness_range, relative):
        """Validates the brightness range and whether it is relative to the mean intensity."""
        if not isinstance(relative, bool):
            raise TypeError("Argument relative must be bool!")
        return validate_numeric_range_parameter(brightness_range, Brightness._default_range), relative

    @staticmethod
    def _lut(shift, contrast=1.0):
        """Look-up table of the linear intensity change ``x * contrast + shift``."""
        lut = np.arange(0, 256) * contrast + shift
        return np.clip(lut, 0, 255).astype("uint8")

    def sample_transform(self, data):
        brightness_fact = random.uniform(self.brightness_range[0], self.brightness_range[1])
        self.state_dict = {"brightness_fact": brightness_fact}
        if not self.relative:
            # The shift does not depend on the image
            self.state_dict["LUT"] = Brightness._lut(brightness_fact)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.relative:
            return cv2.LUT(img, Brightness._lut(self.state_dict["brightness_fact"] * img.mean()))
        return cv2.LUT(img, self.state_dict["LUT"])


class BrightnessContrast(ImageTransform):
    """
    Performs a random brightness and contrast augmentation in one step: ``x * contrast + brightness``.

    It is the same as ``Contrast`` followed by ``Brightness``, but the intensities are clipped only once, so no
    information is lost between the two steps.

    Parameters
    ----------
    brightness_range: tuple or None
        Brightness shift range. If None, then ``brightness_range=(0, 0)``.
    contrast_range : tuple or float or None
        Contrast multiplier range. If float, then ``contrast_range = (1-contrast_range, 1+contrast_range)``.
        If None, then ``contrast_range=(1, 1)``.
    relative : bool
        Whether the brightness shift is relative to the mean intensity of the image. See ``Brightness``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.
    p : float
        Probability of applying this transform.

    """

    serializable_name = "brightness_contrast"
    """How the class should be stored in the registry"""

    def __init__(self, brightness_range=None, contrast_range=0.1, relative=False, data_indices=None, p=0.5):
        super(BrightnessContrast, self).__init__(p=p, data_indices=data_indices)
        self.brightness_range, self.relative = Brightness.parse_brightness(brightness_range, relative)
        self.contrast_range = Contrast.parse_contrast_range(contrast_range)

    def sample_transform(self, data):
        contrast_mul = random.uniform(self.contrast_range[0], self.contrast_range[1])
        brightness_fact = random.uniform(self.brightness_range[0], self.brightness_range[1])
        self.state_dict = {"contrast_mul": contrast_mul, "brightness_fact": brightness_fact}
        if not self.relative:
            # The look-up table does not depend on the image
            self.state_dict["LUT"] = Brightness._lut(brightness_fact, contrast_mul)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.relative:
            shift = self.state_dict["brightness_fact"] * img.mean()
            return cv2.LUT(img, Brightness._lut(shift, self.state_dict["contrast_mul"]))
        return cv2.LUT(img, self.state_dict["LUT"])


class IntensityRemap(ImageTransform):
    """
    Performs random intensity remapping [1]_.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    kernel_size: int
        Size of medial filter kernel used during the generation of intensity mapping.
        Higher value yield more monotonic mapping.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    References
    ----------
    .. [1] Hesse, L. S., Kuling, G., Veta, M., & Martel, A. L. (2019).
       Intensity augmentation for domain transfer of whole breast
       segmentation in MRI. https://arxiv.org/abs/1909.02642
    """

    serializable_name = "intensity_remap"
    """How the class should be stored in the registry"""

    def __init__(self, kernel_size=9, data_indices=None, p=0.5):
        super(IntensityRemap, self).__init__(p=p, data_indices=data_indices)
        self.kernel_size = kernel_size

    def sample_transform(self, data):
        m = random.sample(range(256), k=256)
        m = scipy.signal.medfilt(m, kernel_size=self.kernel_size)
        m = m + np.linspace(0, 255, 256)

        m = m - min(m)
        m = m / max(m) * 255
        m = np.floor(m).astype(np.uint8)

        self.state_dict = {"LUT": m}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if img.dtype != np.uint8:
            raise ValueError("IntensityRemap supports uint8 ndarrays only")
        if img.ndim == 3 and img.shape[-1] != 1:
            raise ValueError("Only grayscale 2D images are supported")
        return cv2.LUT(img, self.state_dict["LUT"])


class CvtColor(ImageTransform):
    """
    RGB to grayscale or grayscale to RGB image conversion.

    If converting from grayscale to RGB, then the gs channel is simply clonned.
    If converting from RGB to grayscale, then opencv is used.

    Parameters
    ----------
    mode : str or None
        Color conversion mode. If None, then no conversion happens and mode=none.
        If ``mode == 'rgb2gs'`` and the image is already grayscale,
        then nothing happens. If ``mode == 'gs2rgb'`` and the image is already RGB,
        then also nothing happens.
    keep_dim : bool
        Whether to enforce having three channels when performing rgb to grayscale conversion
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.
    p : float
        Probability of the transform's use. 1 by default
    See also
    --------
    solt.constants.ALLOWED_COLOR_CONVERSIONS

    """

    serializable_name = "cvt_color"
    """How the class should be stored in the registry"""

    def __init__(self, mode=None, keep_dim=True, data_indices=None, p=1):
        super(CvtColor, self).__init__(p=p, data_indices=data_indices)
        self.mode = validate_parameter(mode, ALLOWED_COLOR_CONVERSIONS, "none", heritable=False)
        if not isinstance(keep_dim, bool):
            raise TypeError("Incorrect type of keepdim")
        self.keepdim = keep_dim

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.mode == "none":
            return img
        elif self.mode == "gs2rgb":
            if len(img.shape) != 3:
                raise ValueError("Grayscale images must also have 3 channels")
            if img.shape[-1] == 1:
                return np.dstack((img, img, img))
            elif img.shape[-1] == 3:
                return img
        elif self.mode == "rgb2gs":
            if len(img.shape) != 3:
                raise ValueError("RGB images must have 3 channels")
            if img.shape[-1] == 1:
                return img
            elif img.shape[-1] == 3:
                res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if not self.keepdim:
                    return res
                return np.dstack((res, res, res))
        raise ValueError(f"CvtColor supports images with 1 or 3 channels, got {img.shape[-1]} channels")


class KeypointsJitter(BaseTransform):
    """
    Applies the jittering to the keypoints in X- and Y-durections. Only keypoints are changed, e.g. bounding boxes
    stay as they are.

    Parameters
    ----------
    p : float
        Probability of applying the transform
    dx_range: None or float or tuple of float
        Jittering across X-axis. Valid range is ``(-1, 1)``
    dy_range: None or float or tuple of float
        Jittering across Y-axis. Valid range is ``(-1, 1)``

    """

    serializable_name = "keypoints_jitter"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, dx_range=None, dy_range=None):
        super(KeypointsJitter, self).__init__(data_indices=None, p=p)

        self.dx_range = validate_numeric_range_parameter(dx_range, (0, 0), -1, 1)
        self.dy_range = validate_numeric_range_parameter(dy_range, (0, 0), -1, 1)

    def sample_transform(self, data: DataContainer):
        pass

    @img_shape_checker
    def _apply_img(self, img, settings: dict):
        return img

    def _apply_mask(self, mask, settings: dict):
        return mask

    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        return dmap

    def _apply_pts(self, pts: Keypoints, settings: dict):
        pts_data = pts.data.copy()
        h = pts.height
        w = pts.width

        for j in range(pts.data.shape[0]):
            dx = int(random.uniform(self.dx_range[0], self.dx_range[1]) * w)
            dy = int(random.uniform(self.dy_range[0], self.dy_range[1]) * h)
            pts_data[j, 0] = min(pts_data[j, 0] + dx, w - 1)
            pts_data[j, 1] = min(pts_data[j, 1] + dy, h - 1)

        return Keypoints(pts_data, h, w)

    def _apply_bboxes(self, bboxes, settings: dict):
        # Jittering the corners independently would distort the boxes
        return bboxes

    def _apply_labels(self, labels, settings: dict):
        return labels


class JPEGCompression(ImageTransform):
    """
    Performs random JPEG-based worsening of images.

    Parameters
    ----------
    quality_range : float or tuple of int or int or None
        If float, then the lower bound to sample the quality is
        between ``(quality_range*100%, 100%)``.

        If tuple of int, then it directly sets the quality range.

        If tuple of float, then the quality is sampled from
        ``[quality_range[0]*100%, quality_range[1]*100%]``.

        If int, then the quality is sampled from ``[quality_range, 100]``.
        .
        If None, that the quality range is ``[100, 100]``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    serializable_name = "jpeg_compression"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, quality_range=None, data_indices=None):
        super(JPEGCompression, self).__init__(p=p, data_indices=data_indices)
        if quality_range is None:
            quality_range = (100, 100)

        if isinstance(quality_range, int):
            quality_range = (quality_range, 100)

        if isinstance(quality_range, float):
            quality_range = (int(quality_range * 100), 100)

        if isinstance(quality_range, tuple):
            if isinstance(quality_range[0], float) and isinstance(quality_range[1], float):
                quality_range = (
                    int(quality_range[0] * 100),
                    int(quality_range[1] * 100),
                )
            elif isinstance(quality_range[0], int) and isinstance(quality_range[1], int):
                pass
            else:
                raise TypeError("Wrong type of quality range!")
        else:
            raise TypeError("Wrong type of quality range!")

        self.quality_range = validate_numeric_range_parameter(quality_range, (100, 100), 0, 100)

    def sample_transform(self, data):
        self.state_dict["quality"] = random.randint(self.quality_range[0], self.quality_range[1])

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.state_dict["quality"] == 100:
            return img
        _, encoded_img = cv2.imencode(".jpg", img, (cv2.IMWRITE_JPEG_QUALITY, self.state_dict["quality"]))
        return cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)


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
        if self.mode not in ALLOWED_GRIDMASK_MODES:
            raise ValueError(f"Mode must be one of {sorted(ALLOWED_GRIDMASK_MODES)}")

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
            new_h = int(round(new_w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            new_h = h
            new_w = int(round(new_h * max(self.ratio)))

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
