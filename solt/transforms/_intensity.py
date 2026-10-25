"""Transforms of the intensities of images: brightness, contrast, gamma, colour."""

import random

import cv2
import numpy as np
import scipy
import scipy.signal

from ..core import ImageTransform
from ..constants import ALLOWED_COLOR_CONVERSIONS
from ..utils import (
    img_shape_checker,
    validate_numeric_range_parameter,
    validate_parameter,
)


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
