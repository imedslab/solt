"""Transforms, which add noise or degrade the data: noise, blur, compression and jittering of keypoints."""

import random

import cv2
import numpy as np

from ..core import (
    BaseTransform,
    DataContainer,
    ImageTransform,
    Keypoints,
)
from ..constants import ALLOWED_BLURS, DTYPES_MAX
from ..utils import (
    img_shape_checker,
    validate_numeric_range_parameter,
    validate_parameter,
)


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
