"""Geometric transforms defined by a transform matrix: rotations, shear, scaling, translation and projection."""

import random
from functools import wraps

import numpy as np

from ..core import (
    Keypoints,
    MatrixTransform,
    Stream,
)
from ..utils import img_shape_checker, validate_numeric_range_parameter


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
