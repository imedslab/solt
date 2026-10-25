import random

import cv2
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu
from solt.constants import ALLOWED_INTERPOLATIONS

from .fixtures import *


@pytest.mark.parametrize(
    "img, mask, resize_to",
    [
        (lf("img_6x6"), lf("mask_6x6"), (20, 20)),
        (lf("img_6x6"), lf("mask_6x6"), 20),
        (lf("img_6x6"), lf("mask_6x6"), (3, 3)),
        (lf("img_6x6"), lf("mask_6x6"), 3),
        (lf("img_6x6"), lf("mask_6x6"), (5, 5)),
        (lf("img_6x6"), lf("mask_6x6"), 5),
        (lf("img_6x6"), lf("mask_6x6"), (4, 4)),
        (lf("img_6x6"), lf("mask_6x6"), 4),
        (lf("img_6x6"), lf("mask_6x6"), (7, 6)),
        (lf("img_6x6"), lf("mask_6x6"), (5, 7)),
        (lf("img_6x6"), lf("mask_6x6"), 6),
        (lf("img_6x6"), lf("mask_6x6"), (2, 3)),
        (lf("img_5x5"), lf("mask_5x5"), (20, 20)),
        (lf("img_5x5"), lf("mask_5x5"), 20),
        (lf("img_5x5"), lf("mask_5x5"), (3, 3)),
        (lf("img_5x5"), lf("mask_5x5"), 3),
        (lf("img_5x5"), lf("mask_5x5"), (5, 5)),
        (lf("img_5x5"), lf("mask_5x5"), 5),
        (lf("img_5x5"), lf("mask_5x5"), (4, 4)),
        (lf("img_5x5"), lf("mask_5x5"), 4),
        (lf("img_5x5"), lf("mask_5x5"), (7, 6)),
        (lf("img_5x5"), lf("mask_5x5"), (5, 7)),
        (lf("img_5x5"), lf("mask_5x5"), 6),
        (lf("img_5x5"), lf("mask_5x5"), (2, 3)),
    ],
)
def test_resize_img_to_arbitrary_size(img, mask, resize_to):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))  # Top left corner
    kpts = slc.Keypoints(kpts_data.copy(), img.shape[0], img.shape[1])

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )
    transf = slt.Resize(resize_to=resize_to)
    res = transf(dc).data

    if isinstance(resize_to, int):
        scale_x = resize_to / img.shape[1]
        scale_y = resize_to / img.shape[0]
        assert (res[0].shape[0] == resize_to) and (res[0].shape[1] == resize_to)
        assert (res[1].shape[0] == resize_to) and (res[1].shape[1] == resize_to)
        assert (res[2].height == resize_to) and (res[2].width == resize_to)
    else:
        scale_x = resize_to[0] / img.shape[1]
        scale_y = resize_to[1] / img.shape[0]
        assert (res[0].shape[0] == resize_to[1]) and (res[0].shape[1] == resize_to[0])
        assert (res[1].shape[0] == resize_to[1]) and (res[1].shape[1] == resize_to[0])
        assert (res[2].height == resize_to[1]) and (res[2].width == resize_to[0])

    assert transf.resize_to == resize_to

    # Keypoints are the centres of pixels, as in the resized image
    kpts_data = kpts_data.astype(float)
    kpts_data[:, 0] = (kpts_data[:, 0] + 0.5) * scale_x - 0.5
    kpts_data[:, 1] = (kpts_data[:, 1] + 0.5) * scale_y - 0.5
    np.testing.assert_allclose(res[2].data, kpts_data)


@pytest.mark.parametrize("resize_to", ["1123", 123.0])
def test_wrong_resize_types(resize_to):
    with pytest.raises(TypeError):
        slt.Resize(resize_to=resize_to)


def test_resize_does_not_change_labels():
    trf = slt.Resize(resize_to=(5, 5))
    dc = slc.DataContainer((1,), "L")
    dc = trf(dc)
    assert dc.data[0] == 1


@pytest.mark.parametrize(
    "transform_settings",
    [
        None,
        {0: {"interpolation": "nearest"}},
        {0: {"interpolation": "bicubic"}},
    ],
)
def test_different_interpolations_per_item_per_transform(img_6x6, transform_settings):
    dc = slc.DataContainer((img_6x6,), "I", transform_settings=transform_settings)
    dc_res = slt.Resize(resize_to=(10, 15), interpolation="bilinear")(dc)

    interp = ALLOWED_INTERPOLATIONS["bilinear"]
    if transform_settings is not None:
        interp = ALLOWED_INTERPOLATIONS[transform_settings[0]["interpolation"]]
    assert np.array_equal(
        cv2.resize(img_6x6, (10, 15), interpolation=interp).reshape(15, 10, 1),
        dc_res.data[0],
    )


@pytest.mark.parametrize("resize_to", [(16, 12), 20, None])
@pytest.mark.parametrize("mask_shape", [(40, 30), (40, 30, 5)])
def test_random_resized_crop_img_mask(resize_to, mask_shape):
    img = np.random.randint(0, 255, (40, 30, 3), dtype=np.uint8)
    mask = np.random.randint(0, 3, mask_shape, dtype=np.uint8)
    trf = slt.RandomResizedCrop(resize_to=resize_to)

    res = trf(slc.DataContainer((img, mask), "IM"))
    img_res, mask_res = res.data

    w_crop, h_crop = trf.state_dict["crop_to"]
    y, x = trf.state_dict["y"], trf.state_dict["x"]
    expected_img = img[y : y + h_crop, x : x + w_crop]
    expected_mask = mask[y : y + h_crop, x : x + w_crop]
    if resize_to is not None:
        expected_img = cv2.resize(expected_img, trf.resize_to, interpolation=cv2.INTER_LINEAR)
        expected_mask = cv2.resize(expected_mask, trf.resize_to, interpolation=cv2.INTER_NEAREST)

    assert np.array_equal(img_res, expected_img)
    assert np.array_equal(mask_res, expected_mask)
    assert img_res.shape[:2] == mask_res.shape[:2]
    assert mask_res.dtype == mask.dtype


def test_random_resized_crop_uses_item_interpolation():
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 1
    trf = slt.RandomResizedCrop(resize_to=(17, 17), scale=(1, 1), ratio=(1, 1))

    res_nearest = trf(slc.DataContainer((mask,), "M")).data[0]
    res_bilinear = trf(slc.DataContainer((mask,), "M", {0: {"interpolation": "bilinear"}})).data[0]

    assert set(np.unique(res_nearest)) == {0, 1}
    assert np.array_equal(res_bilinear, cv2.resize(mask, (17, 17), interpolation=cv2.INTER_LINEAR))


def test_random_resized_crop_custom_scale_and_ratio():
    img = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
    trf = slt.RandomResizedCrop(resize_to=None, scale=(1, 1), ratio=(1, 1))
    res = trf(slc.DataContainer((img,), "I"))

    assert trf.scale == (1, 1)
    assert trf.ratio == (1, 1)
    assert np.array_equal(res.data[0], img)


@pytest.mark.parametrize(
    "scale, ratio, expected_error",
    [
        ((0.5, 1.5), None, ValueError),
        ((-0.1, 1.0), None, ValueError),
        ((0.9, 0.5), None, ValueError),
        (None, (0, 1), ValueError),
        (None, (2, 1), ValueError),
        (0.5, None, TypeError),
        (None, (1, 2, 3), ValueError),
    ],
)
def test_random_resized_crop_wrong_scale_or_ratio(scale, ratio, expected_error):
    with pytest.raises(expected_error):
        slt.RandomResizedCrop(resize_to=32, scale=scale, ratio=ratio)


def test_random_resized_crop_serialization():
    trf = slt.RandomResizedCrop(resize_to=(32, 24), scale=(0.5, 1.0), ratio=(0.5, 2.0), interpolation="bicubic")
    trf_restored = slu.from_yaml(slc.Stream([trf]).to_yaml()).transforms[0]

    assert isinstance(trf_restored, slt.RandomResizedCrop)
    assert tuple(trf_restored.resize_to) == (32, 24)
    assert tuple(trf_restored.scale) == (0.5, 1.0)
    assert tuple(trf_restored.ratio) == (0.5, 2.0)
    assert trf_restored.interpolation[0] == "bicubic"


def _blob(h, w, cx, cy, sigma=3.0):
    """Gaussian blob centred at a sub-pixel position (x, y) given in pixel centres."""
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def _centroid(img):
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    return (xx * img).sum() / img.sum(), (yy * img).sum() / img.sum()


@pytest.mark.parametrize("resize_to", [(30, 30), (120, 120), (45, 100), (80, 37)])
@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_resize_keypoints_follow_the_image(resize_to, interpolation):
    h, w, cx, cy = 60, 64, 25.3, 33.7
    img = _blob(h, w, cx, cy).astype(np.float32)[..., np.newaxis]
    kpts = slc.Keypoints(np.array([[cx, cy]]), h, w)

    img_res, kpts_res = slt.Resize(resize_to=resize_to, interpolation=interpolation)(
        slc.DataContainer((img, kpts), "IP")
    ).data

    assert kpts_res.data.dtype == np.float64
    np.testing.assert_allclose(kpts_res.data[0], _centroid(img_res[..., 0]), atol=0.05)


@pytest.mark.parametrize("resize_to", [(32, 32), (50, 20), None])
def test_random_resized_crop_keypoints(resize_to):
    h, w = 60, 64
    kpts = slc.Keypoints(np.array([[10.0, 20.0], [40.5, 33.25]]), h, w)
    img = np.zeros((h, w, 1), dtype=np.uint8)
    random.seed(0)
    for _ in range(10):
        trf = slt.RandomResizedCrop(resize_to=resize_to)
        img_res, kpts_res = trf(slc.DataContainer((img, kpts), "IP")).data

        (crop_w, crop_h), x, y = trf.state_dict["crop_to"], trf.state_dict["x"], trf.state_dict["y"]
        new_w, new_h = resize_to if resize_to is not None else (crop_w, crop_h)
        expected = kpts.data - (x, y)
        expected[:, 0] = (expected[:, 0] + 0.5) * new_w / crop_w - 0.5
        expected[:, 1] = (expected[:, 1] + 0.5) * new_h / crop_h - 0.5

        np.testing.assert_allclose(kpts_res.data, expected)
        assert (kpts_res.height, kpts_res.width) == img_res.shape[:2] == (new_h, new_w)


def test_random_resized_crop_keypoints_follow_the_image():
    h, w, cx, cy = 80, 80, 40.3, 39.6
    img = _blob(h, w, cx, cy).astype(np.float32)[..., np.newaxis]
    kpts = slc.Keypoints(np.array([[cx, cy]]), h, w)
    random.seed(0)
    checked = 0
    for _ in range(30):
        trf = slt.RandomResizedCrop(resize_to=(48, 48), scale=(0.5, 1.0))
        img_res, kpts_res = trf(slc.DataContainer((img, kpts), "IP")).data
        (crop_w, crop_h), x, y = trf.state_dict["crop_to"], trf.state_dict["x"], trf.state_dict["y"]
        # Only crops, which contain the whole blob, have a meaningful centroid
        if x + 10 < cx < x + crop_w - 10 and y + 10 < cy < y + crop_h - 10:
            np.testing.assert_allclose(kpts_res.data[0], _centroid(img_res[..., 0]), atol=0.1)
            checked += 1
    assert checked > 5


@pytest.fixture(
    params=[(128, 96), (32, 24), (100, 40), (64, 48), (17, 91)],
    ids=["upscale", "downscale", "non_uniform", "identity", "odd"],
)
def resize_to(request):
    return request.param


def test_random_resized_crop_of_the_whole_image_maps_keypoints_as_resize(resize_to):
    h, w = 48, 64
    kpts = slc.Keypoints(np.array([[0.0, 0.0], [63.0, 47.0], [10.3, 20.7], [31.5, 23.5]]), h, w)
    img = np.zeros((h, w, 1), dtype=np.uint8)

    resized = slt.Resize(resize_to=resize_to)(slc.DataContainer((img, kpts), "IP")).data[1]
    crop = slt.RandomResizedCrop(resize_to=resize_to, scale=(1, 1), ratio=(w / h, w / h))
    cropped = crop(slc.DataContainer((img, kpts), "IP")).data[1]

    assert crop.state_dict["crop_to"] == (w, h)
    np.testing.assert_allclose(cropped.data, resized.data)
    assert (cropped.height, cropped.width) == (resized.height, resized.width)


@pytest.fixture(params=[(100, 200), (200, 100), (60, 60), (7, 13), (13, 7)], ids=lambda s: f"hw{s[0]}x{s[1]}")
def resize_keep_ar_hw(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["height_first", "width_first"])
def resize_height_first(request):
    return request.param


@pytest.fixture(params=[50, 5, (30, 40), [9, 4]], ids=lambda r: f"to{r}")
def resize_keep_ar_to(request):
    return request.param


@pytest.fixture
def resize_keep_ar_expected_hw(resize_keep_ar_hw, resize_height_first, resize_keep_ar_to):
    h, w = resize_keep_ar_hw
    if not isinstance(resize_keep_ar_to, int):
        # keep_ar is ignored when the target size (width, height) is given explicitly
        return resize_keep_ar_to[1], resize_keep_ar_to[0]
    if h == w:
        return resize_keep_ar_to, resize_keep_ar_to
    if resize_height_first:
        return resize_keep_ar_to, int(resize_keep_ar_to * w / h)
    return int(resize_keep_ar_to * h / w), resize_keep_ar_to


def test_resize_keep_ar_targets_follow_the_image(
    resize_keep_ar_hw, resize_height_first, resize_keep_ar_to, resize_keep_ar_expected_hw
):
    h, w = resize_keep_ar_hw
    exp_h, exp_w = resize_keep_ar_expected_hw
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    dmap = np.full((h, w), 10.0, np.float32)
    kpts_data = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [w // 2, h // 3]])
    kpts = slc.Keypoints(kpts_data, h, w)
    dc = slc.DataContainer((img, mask, dmap, kpts), "IMDP")

    res = slt.Resize(resize_to=resize_keep_ar_to, keep_ar=True, height_first=resize_height_first)(dc).data

    assert res[0].shape == (exp_h, exp_w, 3)
    assert res[1].shape == (exp_h, exp_w)
    assert res[2].shape == (exp_h, exp_w)
    np.testing.assert_allclose(res[2], 10 * np.sqrt(exp_w / w * exp_h / h), rtol=1e-5)
    assert (res[3].height, res[3].width) == (exp_h, exp_w)
    np.testing.assert_allclose(res[3].data, slt.Resize.resize_points(kpts_data, exp_w / w, exp_h / h))


@pytest.mark.parametrize("height_first, expected_hw", [(True, (50, 100)), (False, (25, 50))])
def test_resize_keep_ar_landscape_keeps_orientation(height_first, expected_hw):
    dc = slc.DataContainer((np.zeros((100, 200, 3), np.uint8),), "I")
    res = slt.Resize(resize_to=50, keep_ar=True, height_first=height_first)(dc).data[0]
    assert res.shape[:2] == expected_hw


@pytest.fixture(
    params=[
        # (h, w), scale, ratio, the expected central crop (w, h)
        ((10, 100), (0.9, 1.0), (1, 1), (10, 10)),
        ((100, 10), (0.9, 1.0), (1, 1), (10, 10)),
        ((11, 101), (0.9, 1.0), (1, 1), (11, 11)),
        ((10, 100), (0.9, 1.0), (0.5, 2), (20, 10)),
        ((100, 10), (0.9, 1.0), (0.5, 2), (10, 20)),
        ((10, 100), (0.9, 1.0), (3, 3), (30, 10)),
        # The aspect ratio of the image is within the range, so the whole image is the crop
        ((40, 40), (1, 1), (0.2, 5), (40, 40)),
        ((30, 60), (1, 1), (0.25, 4), (60, 30)),
        ((61, 31), (1, 1), (0.25, 4), (31, 61)),
    ],
    ids=[
        "landscape",
        "portrait",
        "landscape_odd",
        "landscape_ratio_range",
        "portrait_ratio_range",
        "landscape_wide_ratio",
        "square_ratio_in_range",
        "landscape_ratio_in_range",
        "portrait_ratio_in_range",
    ],
)
def rrc_fallback_case(request):
    return request.param


@pytest.fixture(
    params=[
        # Too thin images for the requested aspect ratio: the crop must not become empty
        ((60, 1), (0.9, 1.0), (3, 3), (1, 1)),
        ((1, 60), (0.9, 1.0), (0.3, 0.3), (1, 1)),
    ],
    ids=["thin_portrait", "thin_landscape"],
)
def rrc_thin_fallback_case(request):
    return request.param


@pytest.fixture(params=[None, 16, (24, 12)], ids=["no_resize", "resize_int", "resize_tuple"])
def rrc_resize_to(request):
    return request.param


def _check_rrc_central_crop(trf, h, w, expected_crop):
    crop_w, crop_h = trf.state_dict["crop_to"]
    x, y = trf.state_dict["x"], trf.state_dict["y"]
    assert (crop_w, crop_h) == expected_crop
    # Inside of the image and centred
    assert 0 <= x and x + crop_w <= w and 0 <= y and y + crop_h <= h
    assert abs(x - (w - x - crop_w)) <= 1 and abs(y - (h - y - crop_h)) <= 1


def test_random_resized_crop_falls_back_to_a_central_crop(rrc_fallback_case):
    (h, w), scale, ratio, expected_crop = rrc_fallback_case
    random.seed(42)
    trf = slt.RandomResizedCrop(resize_to=None, scale=scale, ratio=ratio)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img_res = trf(slc.DataContainer((img,), "I")).data[0]

    _check_rrc_central_crop(trf, h, w, expected_crop)
    crop_w, crop_h = expected_crop
    # The aspect ratio closest to the image one within the range (up to the rounding of the crop size)
    expected_ratio = min(max(w / h, ratio[0]), ratio[1])
    assert crop_w / crop_h == pytest.approx(expected_ratio, rel=0.5 / min(crop_w, crop_h))
    x, y = trf.state_dict["x"], trf.state_dict["y"]
    assert np.array_equal(img_res, img[y : y + crop_h, x : x + crop_w])


def test_random_resized_crop_central_crop_of_a_thin_image_is_not_empty(rrc_thin_fallback_case, rrc_resize_to):
    (h, w), scale, ratio, expected_crop = rrc_thin_fallback_case
    random.seed(42)
    trf = slt.RandomResizedCrop(resize_to=rrc_resize_to, scale=scale, ratio=ratio)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img_res = trf(slc.DataContainer((img,), "I")).data[0]

    _check_rrc_central_crop(trf, h, w, expected_crop)
    new_w, new_h = trf.resize_to if rrc_resize_to is not None else expected_crop
    assert img_res.shape == (new_h, new_w, 3)


class _CropAt(slt.Crop):
    """Crop at a given location, the reference for RandomResizedCrop."""

    serializable_name = None

    def __init__(self, crop_to, x, y):
        super(_CropAt, self).__init__(crop_to=crop_to)
        self.x, self.y = x, y

    def sample_transform(self, data):
        super(_CropAt, self).sample_transform(data)
        self.state_dict.update({"x": self.x, "y": self.y})


def _rrc_items(h, w):
    rng = np.random.default_rng(0)
    pts = np.array([[0.0, 0.0], [w - 1.0, h - 1.0], [w / 2 - 0.3, h / 2 + 0.2], [w / 3, h / 4]])
    return (
        rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        rng.integers(0, 3, (h, w), dtype=np.uint8),
        rng.random((h, w)).astype(np.float32) * 10,
        slc.Keypoints(pts, h, w),
        slc.BoundingBoxes([[0.0, 0.0, w, h], [w / 2 - 3, h / 2 - 3, w / 2 + 3, h / 2 + 3]], h, w),
    )


def test_random_resized_crop_central_crop_keeps_all_data_aligned(rrc_fallback_case, rrc_resize_to):
    (h, w), scale, ratio, _ = rrc_fallback_case
    # max_bbox_* are disabled, so that the boxes are compared as they are
    kwargs = {"max_bbox_area_loss": None, "max_bbox_outside": None}
    random.seed(42)
    trf = slt.RandomResizedCrop(resize_to=rrc_resize_to, scale=scale, ratio=ratio)
    res = slc.Stream([trf], **kwargs)(slc.DataContainer(_rrc_items(h, w), "IMDPB"), return_torch=False)

    reference = slc.Stream(
        [
            _CropAt(trf.state_dict["crop_to"], trf.state_dict["x"], trf.state_dict["y"]),
            slt.Resize(resize_to=trf.resize_to),
        ],
        **kwargs,
    )
    expected = reference(slc.DataContainer(_rrc_items(h, w), "IMDPB"), return_torch=False)

    new_w, new_h = trf.resize_to if rrc_resize_to is not None else trf.state_dict["crop_to"]
    for item, expected_item in zip(res.data, expected.data):
        if isinstance(item, (slc.Keypoints, slc.BoundingBoxes)):
            assert (item.height, item.width) == (expected_item.height, expected_item.width) == (new_h, new_w)
            np.testing.assert_allclose(item.data, expected_item.data, atol=1e-10)
        else:
            assert item.shape[:2] == (new_h, new_w)
            assert item.dtype == expected_item.dtype
            np.testing.assert_allclose(item, expected_item, rtol=1e-6)


@pytest.mark.parametrize("resize_to, expected", [(32, (32, 32)), ([40, 20], (40, 20)), ((20, 40), (20, 40))])
def test_random_resized_crop_resize_to_is_a_tuple(resize_to, expected):
    trf = slt.RandomResizedCrop(resize_to=resize_to)
    assert isinstance(trf.resize_to, tuple)
    assert trf.resize_to == expected


@pytest.mark.parametrize("resize_to", [(32.0, 32), (32, "32"), [32, 32.5], 32.0, "32", {32: 32}])
def test_random_resized_crop_wrong_resize_to(resize_to):
    with pytest.raises(TypeError):
        slt.RandomResizedCrop(resize_to=resize_to)


@pytest.fixture(
    params=[((40, 60), (0.3, 1.0), (0.5, 2)), ((10, 100), (0.9, 1.0), (1, 1)), ((64, 64), (0.08, 1.0), (0.75, 4 / 3))],
    ids=["random_crop", "central_crop", "imagenet"],
)
def rrc_dmap_case(request):
    return request.param


def test_random_resized_crop_rescales_distance_maps_only_when_resizing(rrc_dmap_case, rrc_resize_to):
    (h, w), scale, ratio = rrc_dmap_case
    dmap = np.random.default_rng(0).random((h, w, 2)).astype(np.float32) * 10
    random.seed(42)
    for _ in range(5):
        trf = slt.RandomResizedCrop(resize_to=rrc_resize_to, scale=scale, ratio=ratio)
        dmap_res = trf(slc.DataContainer((dmap,), "D")).data[0]

        (crop_w, crop_h), x, y = trf.state_dict["crop_to"], trf.state_dict["x"], trf.state_dict["y"]
        expected = dmap[y : y + crop_h, x : x + crop_w]
        if rrc_resize_to is not None:
            new_w, new_h = trf.resize_to
            expected = cv2.resize(expected, trf.resize_to, interpolation=cv2.INTER_LINEAR)
            expected = expected * np.sqrt(new_w * new_h / (crop_w * crop_h))
        assert dmap_res.shape == expected.shape
        assert dmap_res.dtype == dmap.dtype
        np.testing.assert_allclose(dmap_res, expected, rtol=1e-5)


def test_random_resized_crop_keeps_labels(rrc_fallback_case, rrc_resize_to):
    (h, w), scale, ratio, _ = rrc_fallback_case
    img = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.array([1, 2, 3])
    random.seed(42)
    trf = slt.RandomResizedCrop(resize_to=rrc_resize_to, scale=scale, ratio=ratio)
    res = trf(slc.DataContainer((img, labels, 7), "ILL"))

    assert res.data[1] is labels
    assert res.data[2] == 7
