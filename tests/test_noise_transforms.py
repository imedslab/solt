import random

import numpy as np
import pytest

import solt.core as slc
import solt.transforms as slt
from solt.constants import DTYPES_MAX

from .fixtures import *


def test_gaussian_noise_no_image_throws_value_error():
    trf = slt.Noise(p=1)
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 6, 6)
    dc = slc.DataContainer((kpts,), "P")

    with pytest.raises(ValueError):
        trf(dc)


def test_gaussian_noise_float_gain():
    trf = slt.Noise(gain_range=0.2, p=1)
    assert isinstance(trf.gain_range, tuple)
    assert len(trf.gain_range) == 2
    assert trf.gain_range[0] == 0 and trf.gain_range[1] == 0.2


def test_salt_and_pepper_no_gain(img_6x6):
    trf = slt.SaltAndPepper(gain_range=0.0, p=1)
    dc_res = trf(slc.DataContainer((img_6x6.astype(np.uint8),), "I"))
    assert np.array_equal(img_6x6, dc_res[0][0])


def test_dtypes_max():
    assert DTYPES_MAX == {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_salt_and_pepper_uses_dtype_max(dtype):
    img = np.full((20, 20, 1), 100, dtype=dtype)
    trf = slt.SaltAndPepper(gain_range=(0.5, 0.5), salt_p=(0.5, 0.5), p=1)
    res = trf(slc.DataContainer((img,), "I")).data[0]

    salt = trf.state_dict["salt"].astype(bool)
    pepper = trf.state_dict["pepper"].astype(bool)
    assert salt.any() and pepper.any()
    assert res.dtype == dtype
    assert np.all(res[salt] == np.iinfo(dtype).max)
    assert np.all(res[pepper] == 0)
    assert np.all(res[~(salt | pepper)] == 100)


def test_image_doesnt_change_when_gain_0_in_gaussian_noise_addition(img_3x3):
    dc = slc.DataContainer((img_3x3,), "I")
    trf = slt.Noise(gain_range=(0, 0), p=1)
    dc_res = trf(dc)
    np.testing.assert_array_equal(img_3x3, dc_res.data[0])


@pytest.mark.parametrize(
    "salt_p",
    [
        (1, 2),
        (2, 2),
        (0.7, 0.3),
        (-0.1),
        (-0.8, 0.1),
        (0.3, 0.7, 0.8),
    ],
)
def test_wrong_salt_p_salt_and_pepper(salt_p):
    with pytest.raises(ValueError):
        slt.SaltAndPepper(salt_p=salt_p)


@pytest.mark.parametrize(
    "trf, gain_range",
    [
        (slt.SaltAndPepper, (1, 2)),
        (slt.SaltAndPepper, (2, 2)),
        (slt.SaltAndPepper, (0.7, 0.3)),
        (slt.SaltAndPepper, -0.1),
        (slt.SaltAndPepper, (-0.8, 0.1)),
        (slt.SaltAndPepper, (0.3, 0.7, 0.8)),
        (slt.Noise, (1, 2)),
        (slt.Noise, (2, 2)),
        (slt.Noise, (0.7, 0.3)),
        (slt.Noise, -0.1),
        (slt.Noise, (-0.8, 0.1)),
        (slt.Noise, (0.3, 0.7, 0.8)),
    ],
)
def test_wrong_gain_range_in_noises(trf, gain_range):
    with pytest.raises(ValueError):
        trf(gain_range=gain_range)


def test_wrong_types_in_gain_and_salt_p_salt_and_peper():
    with pytest.raises(TypeError):
        slt.SaltAndPepper(salt_p="2")

    with pytest.raises(TypeError):
        slt.SaltAndPepper(gain_range="2")


@pytest.mark.parametrize(
    "blur_t, k_size, sigma, to_catch",
    [
        (None, "123", None, TypeError),
        (None, (1, 3, 0), None, ValueError),
        (None, (1, 3, 5, -7), None, ValueError),
        (
            None,
            (
                1,
                3,
                5,
            ),
            -1,
            ValueError,
        ),
        (
            None,
            (
                1,
                3,
                5,
            ),
            "123",
            TypeError,
        ),
        (
            None,
            (
                1,
                3,
                5,
            ),
            (0, -4),
            ValueError,
        ),
        (
            None,
            (
                1,
                3,
                5,
            ),
            (1, "34"),
            TypeError,
        ),
        ("mo", (1, 1), 1, ValueError),
    ],
)
def test_blur_arguments(blur_t, k_size, sigma, to_catch, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    with pytest.raises(to_catch):
        b = slt.Blur(blur_type=blur_t, k_size=k_size, gaussian_sigma=sigma)
        b.sample_transform(dc)


@pytest.mark.parametrize(
    "blur_t, k_size, sigma",
    [
        (None, (1, 3), None),
        (None, 3, None),
        (None, (1, 3, 5), None),
        (
            None,
            (
                1,
                3,
                5,
            ),
            1,
        ),
        ("g", (1, 3), None),
        ("g", 3, None),
        ("g", (1, 3, 5), None),
        (
            "g",
            (
                1,
                3,
                5,
            ),
            1,
        ),
        ("m", (1, 3), None),
        ("m", 3, None),
        ("m", (1, 3, 5), None),
        (
            "m",
            (
                1,
                3,
                5,
            ),
            1,
        ),
    ],
)
def test_blur_samples_correctly(blur_t, k_size, sigma, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Blur(blur_type=blur_t, k_size=k_size, gaussian_sigma=sigma)
    trf.sample_transform(dc)

    if isinstance(k_size, int):
        k_size = (k_size,)
    if sigma is None:
        sigma = 1

    assert trf.state_dict["k_size"] in k_size
    assert trf.state_dict["sigma"] == sigma


@pytest.mark.parametrize(
    "jitter_x,jitter_y,exp_x,exp_y",
    [
        (-0.5, -0.5, 0, 0),
        (-0.5, 0.5, 0, 1),
        (0.5, 0.5, 1, 1),
        (0, 0, 1, 1),
    ],
)
def test_keypoint_jitter_works_correctly(jitter_x, jitter_y, exp_x, exp_y):
    kpts_data = np.array(
        [
            [1, 1],
        ]
    ).reshape((1, 2))
    kpts = slc.Keypoints(kpts_data.copy(), 2, 2)

    dc = slc.DataContainer((kpts,), "P")
    trf = slc.Stream([slt.KeypointsJitter(p=1, dx_range=(jitter_x, jitter_x), dy_range=(jitter_y, jitter_y))])
    dc_res = trf(dc, return_torch=False)

    assert np.array_equal(dc_res.data[0].data, np.array([exp_x, exp_y]).reshape((1, 2)))


def test_keypoint_jitter_does_not_change_img_mask_or_target(img_3x3, mask_3x3):
    trf = slc.Stream([slt.KeypointsJitter(p=1, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2))])
    dc_res = trf({"image": img_3x3.copy(), "mask": mask_3x3.copy(), "label": 1}, return_torch=False)

    assert np.array_equal(dc_res.data[0], img_3x3)
    assert np.array_equal(dc_res.data[1], mask_3x3)
    assert np.array_equal(dc_res.data[2], 1)


@pytest.mark.parametrize("ks", [3, (3, 3), 5, (5, 5)])
def test_motion_blur_samples_transform(ks, img_6x6_rgb):
    blur = slt.Blur(p=1, blur_type="mo", k_size=ks)
    random.seed(42)
    dc = slc.DataContainer(img_6x6_rgb, "I")
    for i in range(100):
        blur.sample_transform(dc)
        if isinstance(ks, int):
            assert blur.state_dict["motion_kernel"].shape[0] == ks
        else:
            assert blur.state_dict["motion_kernel"].shape == ks


@pytest.mark.parametrize(
    "quality, different",
    [
        (None, False),
        (90, True),
        (0.9, True),
        ((50, 90), True),
        ((0.5, 0.9), True),
        (50, True),
    ],
)
def test_jpeg_transform(img_6x6_rgb, quality, different):
    trf = slt.JPEGCompression(quality_range=quality, p=1)
    dc_res = trf({"image": img_6x6_rgb.copy()})
    assert (not np.array_equal(img_6x6_rgb, dc_res.data[0])) == different


@pytest.mark.parametrize("quality", ["1", (0.4, 1), (10, 20.0)])
def test_jpeg_quality_range_raises_error_when_wrong(quality):
    with pytest.raises(TypeError):
        slt.JPEGCompression(quality_range=quality, p=1)
