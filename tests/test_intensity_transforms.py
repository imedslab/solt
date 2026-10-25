import random
from contextlib import ExitStack as does_not_raise

import cv2
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

from .fixtures import *


@pytest.mark.parametrize(
    "trf_cls,trf_params",
    [
        (slt.Noise, {"gain_range": 0.5, "p": 1}),
        (slt.SaltAndPepper, {"p": 1}),
        (slt.GammaCorrection, {"p": 1}),
        (slt.Contrast, {"p": 1}),
        (slt.Brightness, {"p": 1}),
        (slt.Blur, {"p": 1, "blur_type": "g"}),
        (slt.Blur, {"p": 1, "blur_type": "m"}),
        (slt.Blur, {"p": 1, "blur_type": "mo"}),
    ],
)
def test_image_trfs_dont_change_mask_labels_kpts(trf_cls, trf_params, img_3x4, mask_3x4):
    trf = trf_cls(**trf_params)
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)
    dc = slc.DataContainer((img_3x4, mask_3x4, kpts, 1), "IMPL")
    dc_res = trf(dc)

    assert np.all(dc.data[1] == dc_res.data[1])
    assert np.all(dc.data[2].data == dc_res.data[2].data)
    assert dc.data[3] == dc_res.data[3]


def test_brightness_returns_correct_number_of_channels(img_3x4, img_3x4_rgb):
    trf = slt.Brightness(p=1, brightness_range=(10, 10))
    dc = slc.DataContainer((img_3x4, img_3x4_rgb), "II")
    dc_res = trf(dc)

    img1, img2 = dc_res.data

    assert len(img1.shape) == 3
    assert img1.shape[-1] == 1

    assert len(img2.shape) == 3
    assert img2.shape[-1] == 3


@pytest.mark.parametrize(
    "value_range,to_catch",
    [
        ((-1, 1), ValueError),
        ((-1.0, 1.0), ValueError),
        ((1.0, -1.0), ValueError),
        (1000.0, ValueError),
        ((0.1, 0.8, 0.1), ValueError),
        ((0.8, 0.1), ValueError),
        (("123", 0.1), TypeError),
        ((0.1, "123"), TypeError),
    ],
)
def test_lut_transforms_raise_errors(value_range, to_catch):
    with pytest.raises(to_catch):
        slt.GammaCorrection(gamma_range=value_range)

    with pytest.raises(to_catch):
        slt.Contrast(contrast_range=value_range)


@pytest.mark.parametrize(
    "h_range, s_range, v_range",
    [
        ((0, 0), None, None),
        (None, (0, 0), None),
        (None, None, (0, 0)),
    ],
)
def test_hsv_doesnt_change_an_image(h_range, s_range, v_range, img_6x6):
    trf = slt.HSV(p=1, h_range=h_range, s_range=s_range, v_range=v_range)
    img_rgb = np.dstack((img_6x6, img_6x6, img_6x6)).astype(np.uint8) * 255
    dc = slc.DataContainer(img_rgb, "I")

    dc_res = trf(dc)

    assert 0 == trf.state_dict["h_mod"]
    assert 0 == trf.state_dict["s_mod"]
    assert 0 == trf.state_dict["v_mod"]

    np.testing.assert_array_equal(img_rgb, dc_res.data[0])


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.int32,
        np.float64,
        np.int64,
    ],
)
def test_hsv_trying_use_not_uint8(dtype, img_6x6):
    trf = slt.HSV(p=1)
    img_rgb = np.dstack((img_6x6, img_6x6, img_6x6)).astype(dtype)
    dc = slc.DataContainer(img_rgb, "I")

    with pytest.raises(TypeError):
        trf(dc)


def test_hsv_doesnt_work_for_1_channel(img_6x6):
    trf = slt.HSV(p=1)
    dc = slc.DataContainer(img_6x6.astype(np.uint8), "I")

    with pytest.raises(ValueError):
        trf(dc)


def test_intensity_remap_values():
    trf = slt.IntensityRemap(p=1)
    img = np.arange(0, 256, 1, dtype=np.uint8).reshape((16, 16, 1))
    dc = slc.DataContainer(img, "I")
    out = trf(dc).data[0]

    # Mapping is applied correctly
    img_expected = trf.state_dict["LUT"].reshape((16, 16, 1))
    np.testing.assert_array_equal(out, img_expected)

    # Mapping has a positive trendline
    assert np.sum(np.diff(out.astype(float).ravel())) > 0

    # Higher kernel size yields more monotonic mapping
    trf_noisy = slt.IntensityRemap(p=1, kernel_size=1)
    trf_low_pass = slt.IntensityRemap(p=1, kernel_size=5)
    out_noisy = trf_noisy(dc).data[0].astype(float)
    out_low_pass = trf_low_pass(dc).data[0].astype(float)
    std_noisy = np.std(np.diff(out_noisy.ravel()))
    std_low_pass = np.std(np.diff(out_low_pass.ravel()))
    assert std_low_pass < std_noisy


@pytest.mark.parametrize(
    "img, expected",
    [
        (lf("img_3x3"), does_not_raise()),
        (lf("img_3x3_rgb"), pytest.raises(ValueError)),
    ],
)
def test_intensity_remap_channels(img, expected):
    trf = slt.IntensityRemap(p=1)
    dc = slc.DataContainer(img, "I")

    with expected:
        trf(dc)


@pytest.mark.parametrize(
    "img, dtype, expected",
    [
        (lf("img_3x3"), np.int8, pytest.raises(ValueError)),
        (lf("img_3x3"), np.uint16, pytest.raises(ValueError)),
        (lf("img_3x3"), float, pytest.raises(ValueError)),
        (lf("img_3x3"), np.bool_, pytest.raises(ValueError)),
    ],
)
def test_intensity_remap_dtypes(img, dtype, expected):
    trf = slt.IntensityRemap(p=1)
    dc = slc.DataContainer(img.astype(dtype), "I")

    with expected:
        trf(dc)


@pytest.mark.parametrize(
    "mode, img, expected",
    [
        ("none", lf("img_6x6"), lf("img_6x6")),
        ("none", lf("img_6x6_rgb"), lf("img_6x6_rgb")),
        ("gs2rgb", lf("img_6x6"), lf("img_6x6_rgb")),
        # ("gs2rgb", lf("img_6x6_rgb"), lf("img_6x6_rgb")),
        # ("rgb2gs", lf("img_6x6_rgb"), lf("img_6x6")),
        # ("rgb2gs", lf("img_6x6"), lf("img_6x6")),
    ],
)
def test_cvt_color_returns_expected_results(mode, img, expected):
    trf = slt.CvtColor(mode=mode, keep_dim=False)
    dc = slc.DataContainer(img, "I")
    dc_res = trf(dc)
    np.testing.assert_array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize(
    "img, expected",
    [
        (lf("img_6x6_rgb"), lf("img_6x6_rgb")),
        (lf("img_6x6"), lf("img_6x6")),
    ],
)
def test_cvtcolor_keeps_dimensions(img, expected):
    trf = slt.CvtColor(mode="rgb2gs")
    dc_res = trf({"image": img})
    np.testing.assert_array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("mode", ["gs2rgb", "rgb2gs"])
def test_image_color_conversion_raises_error(mode, mask_3x4):
    trf = slt.CvtColor(mode=mode)
    dc = slc.DataContainer(mask_3x4.squeeze(), "I")
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize("keep_dim", [1, 1.1, "s"])
def test_image_color_conversion_keepdim_type(keep_dim):
    with pytest.raises(TypeError):
        slt.CvtColor(keep_dim=keep_dim)


def test_random_contrast_multiplies_the_data(img_5x5):
    img = img_5x5
    dc = slc.DataContainer((img,), "I")

    ppl = slt.Contrast(p=1, contrast_range=(2, 2))
    dc_res = ppl(dc)

    assert np.array_equal(dc.data[0] * 2, dc_res.data[0])


def _expected_brightness(img, shift):
    lut = np.clip(np.arange(0, 256) + shift, 0, 255).astype(np.uint8)
    return lut[img]


@pytest.mark.parametrize("value, factor", [(100, 0.2), (50, 0.2), (200, -0.5), (240, 0.3)])
def test_brightness_relative_to_mean_constant_image(value, factor):
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    trf = slt.Brightness(brightness_range=(factor, factor), relative=True, p=1)
    res = trf(slc.DataContainer((img,), "I")).data[0]
    np.testing.assert_array_equal(res, _expected_brightness(img, factor * value))


def test_brightness_relative_to_mean():
    img = np.random.default_rng(0).integers(0, 200, (16, 16, 3), dtype=np.uint8)
    trf = slt.Brightness(brightness_range=(0.1, 0.1), relative=True, p=1)
    res = trf(slc.DataContainer((img,), "I")).data[0]
    np.testing.assert_array_equal(res, _expected_brightness(img, 0.1 * img.mean()))


def test_brightness_relative_uses_the_mean_of_every_image():
    dark = np.full((8, 8, 1), 40, dtype=np.uint8)
    bright = np.full((8, 8, 1), 160, dtype=np.uint8)
    trf = slt.Brightness(brightness_range=(0.25, 0.25), relative=True, p=1)
    dark_res, bright_res = trf(slc.DataContainer((dark, bright), "II")).data
    assert np.all(dark_res == 50)
    assert np.all(bright_res == 200)


def test_brightness_absolute_by_default():
    img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    trf = slt.Brightness(brightness_range=(20, 20), p=1)
    assert trf.relative is False
    res = trf(slc.DataContainer((img,), "I")).data[0]
    np.testing.assert_array_equal(res, _expected_brightness(img, 20))


def test_brightness_relative_serialization():
    trf = slu.from_yaml(slc.Stream([slt.Brightness(brightness_range=(-0.1, 0.2), relative=True)]).to_yaml())
    assert trf.transforms[0].relative is True


@pytest.mark.parametrize("relative", [1, "yes", None])
def test_brightness_relative_must_be_bool(relative):
    with pytest.raises(TypeError):
        slt.Brightness(brightness_range=(0.1, 0.2), relative=relative)


@pytest.mark.parametrize("contrast, brightness", [(2.0, 10.0), (1.0, -20.0), (0.5, 0.0), (1.0, 0.0)])
def test_brightness_contrast_equals_contrast_then_brightness(contrast, brightness):
    # Values, for which neither rounding nor clipping happens between the two transforms
    img = np.arange(0, 120, 2, dtype=np.uint8).reshape(6, 10, 1)
    combined = slt.BrightnessContrast(
        brightness_range=(brightness, brightness), contrast_range=(contrast, contrast), p=1
    )
    sequential = slc.Stream(
        [
            slt.Contrast(contrast_range=(contrast, contrast), p=1),
            slt.Brightness(brightness_range=(brightness, brightness), p=1),
        ]
    )
    res = combined(slc.DataContainer((img,), "I")).data[0]
    np.testing.assert_array_equal(res, sequential({"image": img}, return_torch=False).data[0])


def test_brightness_contrast_clips_once():
    img = np.full((4, 4, 1), 200, dtype=np.uint8)
    trf = slt.BrightnessContrast(brightness_range=(-100, -100), contrast_range=(2.0, 2.0), p=1)
    # 200 * 2 - 100 = 300 -> 255. Contrast followed by Brightness would clip 400 to 255 first, and give 155.
    assert np.all(trf(slc.DataContainer((img,), "I")).data[0] == 255)


def test_brightness_contrast_relative():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    trf = slt.BrightnessContrast(brightness_range=(0.2, 0.2), contrast_range=(1.5, 1.5), relative=True, p=1)
    # 100 * 1.5 + 0.2 * 100
    assert np.all(trf(slc.DataContainer((img,), "I")).data[0] == 170)


def test_brightness_contrast_ranges():
    trf = slt.BrightnessContrast(contrast_range=0.2)
    assert trf.contrast_range == (0.8, 1.2)
    assert trf.brightness_range == (0, 0)
    random.seed(0)
    for _ in range(20):
        trf = slt.BrightnessContrast(brightness_range=(-10, 10), contrast_range=(0.5, 1.5), p=1)
        trf(slc.DataContainer((np.zeros((4, 4, 1), dtype=np.uint8),), "I"))
        assert -10 <= trf.state_dict["brightness_fact"] <= 10
        assert 0.5 <= trf.state_dict["contrast_mul"] <= 1.5


def test_brightness_contrast_serialization():
    trf = slt.BrightnessContrast(brightness_range=(-0.1, 0.1), contrast_range=(0.8, 1.2), relative=True)
    restored = slu.from_yaml(slc.Stream([trf]).to_yaml()).transforms[0]
    assert isinstance(restored, slt.BrightnessContrast)
    assert tuple(restored.brightness_range) == (-0.1, 0.1)
    assert tuple(restored.contrast_range) == (0.8, 1.2)
    assert restored.relative is True


def test_brightness_contrast_keeps_masks():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    trf = slt.BrightnessContrast(brightness_range=(10, 10), contrast_range=(1.2, 1.2), p=1)
    assert trf(slc.DataContainer((img, mask), "IM")).data[1] is mask


@pytest.mark.parametrize("relative", [1, "yes", None])
def test_brightness_contrast_relative_must_be_bool(relative):
    with pytest.raises(TypeError):
        slt.BrightnessContrast(brightness_range=(0.1, 0.2), relative=relative)


_RANGE_INPUTS = {
    "none": None,
    "float": 0.2,
    "zero_float": 0.0,
    "tuple": (0.5, 1.5),
    "list": [0.5, 1.5],
    "int_tuple": (1, 3),
    "int": 2,
    "negative": (-0.5, 0.5),
    "reversed": (1.5, 0.5),
    "too_long": (0.5, 1.0, 1.5),
    "string": "0.5",
}


@pytest.fixture(params=list(_RANGE_INPUTS), ids=list(_RANGE_INPUTS))
def range_input(request):
    return _RANGE_INPUTS[request.param]


def _parse(make_transform, attribute):
    try:
        return getattr(make_transform(), attribute)
    except Exception as e:
        return type(e)


def test_brightness_contrast_parses_contrast_as_contrast(range_input):
    expected = _parse(lambda: slt.Contrast(contrast_range=range_input), "contrast_range")
    assert _parse(lambda: slt.BrightnessContrast(contrast_range=range_input), "contrast_range") == expected


def test_brightness_contrast_parses_brightness_as_brightness(range_input):
    expected = _parse(lambda: slt.Brightness(brightness_range=range_input), "brightness_range")
    assert _parse(lambda: slt.BrightnessContrast(brightness_range=range_input), "brightness_range") == expected


@pytest.fixture(params=[np.uint8, np.uint16], ids=["uint8", "uint16"])
def cvt_dtype(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4], ids=lambda c: f"{c}ch")
def cvt_n_channels(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["keep_dim", "no_keep_dim"])
def cvt_keep_dim(request):
    return request.param


@pytest.fixture
def cvt_img(cvt_n_channels, cvt_dtype):
    rng = np.random.default_rng(42)
    return rng.integers(0, np.iinfo(cvt_dtype).max, size=(6, 7, cvt_n_channels), endpoint=True).astype(cvt_dtype)


@pytest.fixture
def cvt_mask():
    return (np.arange(42).reshape(6, 7) % 3).astype(np.uint8)


def _cvt_expected(mode, img, keep_dim):
    n_channels = img.shape[-1]
    if mode == "none":
        return img
    if n_channels not in (1, 3):
        return None
    if mode == "gs2rgb":
        return np.dstack((img, img, img)) if n_channels == 1 else img
    if n_channels == 1:
        return img
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.dstack((gs, gs, gs)) if keep_dim else gs[:, :, None]


@pytest.mark.parametrize("mode", ["none", "gs2rgb", "rgb2gs"])
def test_cvt_color_values_and_mask_for_channel_counts(mode, cvt_img, cvt_mask, cvt_keep_dim):
    img_orig = cvt_img.copy()
    mask_orig = cvt_mask.copy()
    expected = _cvt_expected(mode, img_orig, cvt_keep_dim)
    trf = slt.CvtColor(mode=mode, keep_dim=cvt_keep_dim)
    dc = slc.DataContainer((cvt_img, cvt_mask), "IM")
    if expected is None:
        with pytest.raises(ValueError, match="1 or 3 channels"):
            trf(dc, return_torch=False)
        return
    dc_res = trf(dc, return_torch=False)
    res_img, res_mask = dc_res.data
    assert res_img.dtype == cvt_img.dtype
    np.testing.assert_array_equal(res_img, expected)
    np.testing.assert_array_equal(res_mask, mask_orig)
    np.testing.assert_array_equal(cvt_img, img_orig)
