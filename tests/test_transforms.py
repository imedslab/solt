import copy
import random
from contextlib import ExitStack as does_not_raise

import cv2
import numpy as np
import pytest

import solt.core as slc

import solt.transforms as slt
import solt.utils as slu
from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS, DTYPES_MAX

from pytest_lazy_fixtures import lf

from .fixtures import *
from .utils import gen_gs_img_black_edge


def test_img_mask_vertical_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream([slt.Flip(p=1, axis=0)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_flip_invalid_axis():
    with pytest.raises(ValueError):
        slt.Flip(p=1, axis=100)


def test_img_mask_mask_vertical_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask, mask), "IMM")

    stream = slc.Stream([slt.Flip(p=1, axis=0)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_horizontal_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream([slt.Flip(p=1, axis=1)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 1), mask_res)


def test_img_mask_vertical_horizontal_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream([slt.Flip(p=1, axis=0), slt.Flip(p=1, axis=1)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)


def test_img_mask_vertical_horizontal_flip_negative_axes(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")

    stream = slt.Flip(p=1, axis=-1)

    dc = stream(dc)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)


def test_img_mask__kptsvertical_horizontal_flip_negative_axes(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4

    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data.copy(), 3, 4)
    dc = slc.DataContainer((img, mask, kpts), "IMP")

    stream = slt.Flip(p=1, axis=-1)

    dc = stream(dc)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]
    kpts_res, _, _ = dc[2]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)

    kpts_data[:, 0] = 4 - 1 - kpts_data[:, 0]
    kpts_data[:, 1] = 3 - 1 - kpts_data[:, 1]

    assert np.array_equal(kpts_data, kpts_res.data)


def test_keypoints_vertical_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 2, 2)
    stream = slt.Flip(p=1, axis=0)
    dc = slc.DataContainer((kpts,), "P")

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_keypoints_horizontal_flip_within_stream():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 2, 2)
    stream = slc.Stream([slt.Flip(p=1, axis=1)])
    dc = slc.DataContainer((kpts,), "P")

    dc_res = stream(dc, return_torch=False)

    assert np.array_equal(dc_res[0][0].data, np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2)))


def test_keypoints_vertical_flip_within_stream():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 2, 2)
    stream = slc.Stream([slt.Flip(p=1, axis=0)])
    dc = slc.DataContainer((kpts,), "P")

    dc_res = stream(dc, return_torch=False)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_rotate_range_none():
    trf = slt.Rotate(None)
    assert trf.angle_range == (0, 0)


@pytest.mark.parametrize("angle", [1, 2.5])
def test_rotate_range_conversion_from_number(angle):
    trf = slt.Rotate(angle_range=angle)
    assert trf.angle_range == (-angle, angle)


def test_shear_range_none():
    trf = slt.Shear(None, None)
    assert trf.range_x == (0, 0)
    assert trf.range_y == (0, 0)


@pytest.mark.parametrize("ignore_state", [True, False])
@pytest.mark.parametrize(
    "transform_settings",
    [
        None,
        {0: {"interpolation": "nearest", "padding": "z"}},
        {0: {"interpolation": "nearest", "padding": "r"}},
        {0: {"interpolation": "bilinear", "padding": "z"}},
        {0: {"interpolation": "bilinear", "padding": "r"}},
        {0: {"interpolation": "bicubic", "padding": "z"}},
        {0: {"interpolation": "bicubic", "padding": "r"}},
        {0: {"interpolation": "area", "padding": "z"}},
        {0: {"interpolation": "area", "padding": "r"}},
        {0: {"interpolation": "lanczos", "padding": "z"}},
        {0: {"interpolation": "lanczos", "padding": "r"}},
    ],
)
def test_rotate_90_img_mask_keypoints_destructive(img_3x3, mask_3x3, transform_settings, ignore_state):
    # Setting up the data

    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3
    H, W = mask.shape

    dc = slc.DataContainer(
        (img, mask, kpts, 1),
        "IMPL",
        transform_settings=copy.deepcopy(transform_settings),
    )
    # Defining the 90 degrees transform (clockwise)
    stream = slt.Rotate(angle_range=(90, 90), p=1, ignore_state=ignore_state)
    dc_res = stream(dc)

    img_res, _, _ = dc_res[0]
    mask_res, _, _ = dc_res[1]
    kpts_res, _, _ = dc_res[2]
    label_res, _, _ = dc_res[3]
    M = cv2.getRotationMatrix2D((W // 2, H // 2), -90, 1)

    img_inter = ALLOWED_INTERPOLATIONS["bicubic"]
    img_pad = ALLOWED_PADDINGS["z"]
    if transform_settings is not None:
        img_inter = ALLOWED_INTERPOLATIONS[transform_settings[0]["interpolation"]]
        img_pad = ALLOWED_PADDINGS[transform_settings[0]["padding"]]

    expected_img_res = cv2.warpAffine(img, M, (W, H), flags=img_inter, borderMode=img_pad).reshape((H, W, 1))
    expected_mask_res = cv2.warpAffine(mask, M, (W, H))
    expected_kpts_res = np.array([[2, 0], [0, 0], [0, 2], [2, 2]]).reshape((4, 2))

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)
    np.testing.assert_array_almost_equal(expected_kpts_res, kpts_res.data)
    assert label_res == 1


@pytest.mark.parametrize("k", list(range(-4, 5)))
def test_rotate_90_img_mask_nondestructive(k, img_3x3, mask_3x3):
    # Setting up the data
    img, mask = img_3x3, mask_3x3
    H, W = mask.shape

    dc = slc.DataContainer((img, mask), "IM")
    # Defining the 90 degrees transform (counterclockwise)
    stream = slt.Rotate90(k=k, p=1)
    dc_res = stream(dc)

    img_res, _, _ = dc_res[0]
    mask_res, _, _ = dc_res[1]

    expected_img_res = np.rot90(img, -k).reshape((H, W, 1))
    expected_mask_res = np.rot90(mask, -k)

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)


@pytest.mark.parametrize("k", [None, "123", 123.0])
def test_rotate_nondestructive_does_not_accept_non_int_k(k):
    with pytest.raises(TypeError):
        slt.Rotate90(k=k)


@pytest.mark.parametrize("k", list(range(-4, 5)))
def test_rotate_90_transforms_have_same_behaviour(k, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf_1 = slt.Rotate(angle_range=(k * 90, k * 90), p=1)
    trf_1.sample_transform(dc)

    trf_2 = slt.Rotate90(k=k, p=1)
    trf_2.sample_transform(dc)

    assert np.array_equal(trf_1.state_dict["transform_matrix"], trf_2.state_dict["transform_matrix"])


def test_zoom_x_axis_odd(img_5x5):
    stream = slt.Scale(range_x=(0.5, 0.5), range_y=(1, 1), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_5x5,), "I")
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_x_axis_even(img_6x6):
    stream = slt.Scale((0.5, 0.5), (1, 1), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_6x6,), "I")
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_odd(img_5x5):
    stream = slt.Scale((0.5, 0.5), (3, 3), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_5x5,), "I")
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 3 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_even(img_6x6):
    stream = slt.Scale((0.5, 0.5), (2, 2), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_6x6,), "I")
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 2 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_img_mask(img_3x4, mask_3x4):
    img_mask_3x4 = img_3x4, mask_3x4
    stream = slt.Scale((0.5, 0.5), (2, 2), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer(img_mask_3x4, "IM")
    H, W = img_mask_3x4[0].shape[0], img_mask_3x4[0].shape[1]
    img_res = stream(dc)[0][0]
    mask_res = stream(dc)[1][0]
    assert H * 2 == img_res.shape[0], W // 2 == img_res.shape[1]
    assert H * 2 == mask_res.shape[0], W // 2 == mask_res.shape[1]


def test_keypoints_assert_reflective(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )
    # Defining the 90 degrees transform (clockwise)
    stream = slt.Rotate(angle_range=(20, 20), p=1, padding="r")
    with pytest.raises(ValueError):
        stream(dc)


def test_padding_img_2x2_4x4(img_2x2):
    img = img_2x2
    dc = slc.DataContainer((img,), "I")
    transf = slt.Pad((4, 4))
    res = transf(dc)
    assert (res[0][0].shape[0] == 4) and (res[0][0].shape[1] == 4)


def test_padding_img_2x2_2x2(img_2x2):
    img = img_2x2
    dc = slc.DataContainer((img,), "I")
    transf = slt.Pad((2, 2))
    res = transf(dc)
    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)


def test_padding_img_mask_2x2_4x4(img_2x2, mask_2x2):
    img, mask = img_2x2, mask_2x2
    dc = slc.DataContainer((img, mask), "IM")
    transf = slt.Pad((4, 4))
    res = transf(dc)
    assert (res[0][0].shape[0] == 4) and (res[0][0].shape[1] == 4)
    assert (res[1][0].shape[0] == 4) and (res[1][0].shape[1] == 4)


def test_padding_img_2x2_3x3(img_2x2):
    img = img_2x2
    dc = slc.DataContainer((img,), "I")
    transf = slt.Pad((3, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)


def test_padding_img_mask_2x2_3x3(img_2x2, mask_2x2):
    img, mask = img_2x2, mask_2x2
    dc = slc.DataContainer((img, mask), "IM")
    transf = slt.Pad((3, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)


def test_padding_img_mask_3x4_3x4(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")
    transf = slt.Pad((4, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 4)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 4)


def test_padding_img_mask_3x4_5x5(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")
    transf = slt.Pad((5, 5))
    res = transf(dc)
    assert (res[0][0].shape[0] == 5) and (res[0][0].shape[1] == 5)
    assert (res[1][0].shape[0] == 5) and (res[1][0].shape[1] == 5)


def test_pad_to_20x20_img_mask_keypoints_3x3(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )
    transf = slt.Pad((20, 20))
    res = transf(dc)

    assert (res[0][0].shape[0] == 20) and (res[0][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[2][0].height == 20) and (res[2][0].width == 20)

    assert np.array_equal(res[2][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]).reshape((4, 2)))


@pytest.mark.parametrize("trf", [slt.Pad, slt.Crop, slt.Resize])
def test_pad_crop_resize_dont_change_data_when_parameters_are_not_set(img_3x3, mask_3x3, trf):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )
    res = trf()(dc, return_torch=False)
    assert dc == res


def _verify_resize_img_to_arbitrary_size(img, mask, resize_to):
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
        resize_to = (resize_to, resize_to)

    scale_x = resize_to[0] / img.shape[1]
    scale_y = resize_to[1] / img.shape[0]

    assert transf.resize_to == resize_to
    assert (res[0].shape[0] == resize_to[1]) and (res[0].shape[1] == resize_to[0])
    assert (res[1].shape[0] == resize_to[1]) and (res[1].shape[1] == resize_to[0])
    assert (res[2].height == resize_to[1]) and (res[2].width == resize_to[0])

    # Keypoints are the centres of pixels, as in the resized image
    kpts_data = kpts_data.astype(float)
    kpts_data[:, 0] = (kpts_data[:, 0] + 0.5) * scale_x - 0.5
    kpts_data[:, 1] = (kpts_data[:, 1] + 0.5) * scale_y - 0.5
    np.testing.assert_allclose(res[2].data, kpts_data)


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


def test_pad_to_20x20_img_mask_keypoints_3x3_kpts_first(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = slc.DataContainer((kpts, img, mask), "PIM")
    transf = slt.Pad((20, 20))
    res = transf(dc)

    assert (res[2][0].shape[0] == 20) and (res[2][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[0][0].height == 20) and (res[0][0].width == 20)

    assert np.array_equal(res[0][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]).reshape((4, 2)))


def test_3x3_pad_to_20x20_center_crop_3x3_shape_stayes_unchanged(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )

    stream = slc.Stream([slt.Pad((20, 20)), slt.Crop((3, 3))])
    res = stream(dc, return_torch=False)

    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)
    assert (res[2][0].height == 3) and (res[2][0].width == 3)


@pytest.mark.parametrize(
    "pad_size,crop_size",
    [
        (20, 2),
        (20, (2, 2)),
        ((20, 20), (2, 2)),
        ((20, 20), 2),
    ],
)
def test_2x2_pad_to_20x20_center_crop_2x2(pad_size, crop_size, img_2x2, mask_2x2):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 2, 2)
    img, mask = img_2x2, mask_2x2

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )

    stream = slc.Stream([slt.Pad(pad_to=pad_size), slt.Crop(crop_to=crop_size)])
    res = stream(dc, return_torch=False)

    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)
    assert (res[1][0].shape[0] == 2) and (res[1][0].shape[1] == 2)
    assert (res[2][0].height == 2) and (res[2][0].width == 2)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0], mask)
    assert np.array_equal(res[2][0].data, kpts_data)


@pytest.mark.parametrize("crop_mode", ["c", "r", "d"])
def test_different_crop_modes(crop_mode, img_2x2, mask_2x2):
    if crop_mode == "d":
        with pytest.raises(ValueError):
            slt.Crop(crop_to=2, crop_mode=crop_mode)
    else:
        stream = slc.Stream(
            [
                slt.Pad(pad_to=20),
                slt.Crop(crop_to=2, crop_mode=crop_mode),
            ]
        )
        img, mask = img_2x2, mask_2x2
        dc = slc.DataContainer(
            (
                img,
                mask,
            ),
            "IM",
        )
        dc_res = stream(dc, return_torch=False)

        for el in dc_res.data:
            assert el.shape[0] == 2
            assert el.shape[1] == 2


def test_6x6_pad_to_20x20_center_crop_6x6_img_kpts(img_6x6):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 6, 6)
    img = img_6x6

    dc = slc.DataContainer((img, kpts, 1), "IPL")

    stream = slc.Stream([slt.Pad((20, 20)), slt.Crop((6, 6))])
    res = stream(dc, return_torch=False)

    assert (res[0][0].shape[0] == 6) and (res[0][0].shape[1] == 6)
    assert (res[1][0].height == 6) and (res[1][0].width == 6)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0].data, kpts_data)


def test_6x6_pad_to_20x20_center_crop_6x6_kpts_img(img_6x6):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 6, 6)
    img = img_6x6

    dc = slc.DataContainer((kpts, img), "PI")

    stream = slc.Stream([slt.Pad((20, 20)), slt.Crop((6, 6))])
    res = stream(dc, return_torch=False)

    assert (res[1][0].shape[0] == 6) and (res[1][0].shape[1] == 6)
    assert (res[0][0].height == 6) and (res[0][0].width == 6)

    assert np.array_equal(res[1][0], img)
    assert np.array_equal(res[0][0].data, kpts_data)


def test_translate_forward_backward_sampling(img_6x6_rgb):
    stream = slc.Stream(
        [
            slt.Translate(range_x=(1, 1), range_y=(1, 1), p=1),
            slt.Translate(range_x=(-1, -1), range_y=(-1, -1), p=1),
        ]
    )
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = stream.optimize_transforms_stack(stream.transforms, dc)[0]
    assert 1 == trf.state_dict["translate_x"]  # The settings will be overrided by the first transform
    assert 1 == trf.state_dict["translate_y"]  # The settings will be overrided by the first transform
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_empty_sampling_no_trf(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(affine_transforms=slc.Stream(), p=1)
    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_empty_sampling_low_prob_trf(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(affine_transforms=slc.Stream([slt.Rotate(p=0)]), p=1)

    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_empty_sampling_from_many_low_prob_trf(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(
        affine_transforms=slc.Stream(
            [
                slt.Rotate(angle_range=(70, 70), p=0),
                slt.Rotate(angle_range=(10, 10), p=0),
                slt.Rotate(angle_range=(-2, -2), p=0),
            ]
        ),
        p=1,
    )
    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_translate_plus_minus_1(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(
        affine_transforms=slc.Stream(
            [
                slt.Translate(range_x=(1, 1), range_y=(1, 1), p=1),
                slt.Translate(range_x=(-1, -1), range_y=(-1, -1), p=1),
            ]
        ),
        p=1,
    )

    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


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


@pytest.mark.parametrize(
    "pad_size,pad_type",
    [
        (2, "z"),
        ((2, 2), "z"),
        (2, "r"),
        ((2, 2), "r"),
    ],
)
def test_pad_does_not_change_the_data_when_the_image_and_the_mask_are_big(pad_size, pad_type, img_3x3, mask_3x3):
    dc = slc.DataContainer((img_3x3, mask_3x3), "IM")
    trf = slt.Pad(pad_to=pad_size, padding=pad_type)
    dc_res = trf(dc)

    np.testing.assert_array_equal(dc_res.data[0], img_3x3)
    np.testing.assert_array_equal(dc_res.data[1], mask_3x3)


def test_image_doesnt_change_when_gain_0_in_gaussian_noise_addition(img_3x3):
    dc = slc.DataContainer((img_3x3,), "I")
    trf = slt.Noise(gain_range=(0, 0), p=1)
    dc_res = trf(dc)
    np.testing.assert_array_equal(img_3x3, dc_res.data[0])


@pytest.mark.parametrize("scale,expected", [(2, (1, 2)), (2.5, (1, 2.5)), (0.5, (0.5, 1)), (-1, None)])
def test_scale_range_from_number(scale, expected):
    if expected is not None:
        trf = slt.Scale(range_x=scale, range_y=scale)
        assert trf.range_x == expected
        assert trf.range_x == expected
    else:
        with pytest.raises(ValueError):
            slt.Scale(range_x=scale)
        with pytest.raises(ValueError):
            slt.Scale(range_x=None, range_y=scale)


@pytest.mark.parametrize(
    "same, scale_x, scale_y, expected",
    [
        (True, (2, 2), (2, 2), (2, 2)),
        (True, (2, 2), (1, 1), (2, 2)),
        (True, (2, 2), None, (2, 2)),
        (True, None, (2, 2), (2, 2)),
        (False, (2, 2), (2, 2), (2, 2)),
        (False, (2, 2), (3, 3), (2, 3)),
        (False, (2, 2), None, (2, 1)),
        (False, None, (2, 2), (1, 2)),
    ],
)
def test_scale_sampling_scale(same, scale_x, scale_y, expected, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Scale(range_x=scale_x, range_y=scale_y, same=same)
    trf.sample_transform(dc)
    assert expected == (trf.state_dict["scale_x"], trf.state_dict["scale_y"])


@pytest.mark.parametrize(
    "translate,expected",
    [
        (2, (-2, 2)),
        (2.5, (-2.5, 2.5)),
        (0.5, (-0.5, 0.5)),
        (-0.5, (-0.5, 0.5)),
    ],
)
def test_translate_range_from_number(translate, expected):
    trf = slt.Translate(range_x=translate, range_y=translate)
    assert trf.range_x == expected
    assert trf.range_y == expected


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


def test_padding_cant_be_float():
    with pytest.raises(TypeError):
        slt.Pad(pad_to=2.5)


def test_reflective_padding_cant_be_applied_to_kpts():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)
    dc = slc.DataContainer((1, kpts), "LP")
    trf = slt.Pad(pad_to=(10, 10), padding="r")
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize(
    "cutout_crop_size",
    [
        (2, 3),
        (3, 2),
    ],
)
def test_crop_or_cutout_size_are_too_big(img_2x2, cutout_crop_size):
    dc = slc.DataContainer((img_2x2,), "I")
    trf = slt.Crop(crop_to=cutout_crop_size)
    with pytest.raises(ValueError):
        trf(dc)

    trf = slt.CutOut(p=1, cutout_size=cutout_crop_size)
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize("cutout_crop_size", ["123", ("23", 2), (2.5, 2), (2, 2.2)])
def test_wrong_crop_size_types(cutout_crop_size):
    with pytest.raises(TypeError):
        slt.Crop(crop_to=cutout_crop_size)

    with pytest.raises(TypeError):
        slt.CutOut(cutout_size=cutout_crop_size)


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
    "inp, transform_settings, pad_to, expected",
    [
        (np.ones((5, 5, 1), dtype=np.uint8), None, (7, 7), gen_gs_img_black_edge(7, 7) // 255),
        (np.ones((5, 5, 1), dtype=np.uint8), {0: {"padding": "z"}}, (7, 7), gen_gs_img_black_edge(7, 7) // 255),
        (np.ones((5, 5, 1), dtype=np.uint8), {0: {"padding": "r"}}, (7, 7), np.ones((7, 7, 1))),
        # (np.ones((3, 5, 1), dtype=np.uint8), None, (7, 7), gen_gs_img_black_edge(7, 7)  // 255),
        # (np.ones((4, 5, 1), dtype=np.uint8), {0: {"padding": "z"}}, (7, 7), gen_gs_img_black_edge(7, 7) // 255),
        # (np.ones((4, 3, 1), dtype=np.uint8), {0: {"padding": "r"}}, (7, 7), np.ones((7, 7, 1))),
        # (np.ones((3, 4, 1), dtype=np.uint8), {0: {"padding": "r"}}, (7, 7), np.ones((7, 7, 1))),
        # (np.ones((5, 5, 1), dtype=np.uint8), None, (7, 6), gen_gs_img_black_edge(7, 6)  // 255),
        # (np.ones((5, 5, 1), dtype=np.uint8), {0: {"padding": "z"}}, (7, 9), gen_gs_img_black_edge(7, 9) // 255),
        # (np.ones((5, 5, 1), dtype=np.uint8), {0: {"padding": "r"}}, (4, 3), np.ones((5, 5, 1))),
    ],
)
def test_padding_for_item(inp, transform_settings, pad_to, expected):
    dc = slc.DataContainer((inp,), "I", transform_settings)
    dc_res = slt.Pad(pad_to=(pad_to[0], pad_to[1]))(dc)
    assert np.array_equal(expected, dc_res.data[0])


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
    "scale_x, scale_y, same, expected",
    [
        (None, (2, 2), True, (2, 2)),
        ((2, 2), None, True, (2, 2)),
        (None, None, True, (1, 1)),
        (None, (2, 2), False, (1, 2)),
        ((2, 2), None, False, (2, 1)),
        (None, None, False, (1, 1)),
    ],
)
def test_scale_when_range_x_is_none(scale_x, scale_y, same, expected, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Scale(range_x=scale_x, range_y=scale_y, same=same, p=1)
    trf.sample_transform(dc)
    assert (trf.state_dict["scale_x"], trf.state_dict["scale_y"]) == expected


@pytest.mark.parametrize(
    "translate_x, translate_y, expected",
    [
        (None, (2, 2), (0, 2)),
        ((2, 2), None, (2, 0)),
        (None, None, (0, 0)),
    ],
)
def test_translate_when_range_x_is_none(translate_x, translate_y, expected, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Translate(range_x=translate_x, range_y=translate_y, p=1)
    trf.sample_transform(dc)
    assert (trf.state_dict["translate_x"], trf.state_dict["translate_y"]) == expected


@pytest.mark.parametrize(
    "param_set",
    [
        {"affine_transforms": "123"},
        {"affine_transforms": 123},
        {"affine_transforms": []},
        {
            "affine_transforms": slc.Stream(
                [
                    slt.Flip(),
                ]
            )
        },
        {
            "affine_transforms": slc.Stream(
                [
                    slt.Flip(),
                ]
            )
        },
        {"v_range": "123"},
        {"v_range": 123},
        {"v_range": ("123", "456")},
        {"v_range": ((2,), (4,))},
    ],
)
def test_random_projection_raises_type_errors(param_set):
    with pytest.raises(TypeError):
        slt.Projection(**param_set)


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


def test_random_proj_and_selective_stream(img_5x5):
    img = img_5x5
    dc = slc.DataContainer((img,), "I")

    ppl = slt.Projection(
        slc.SelectiveStream(
            [
                slt.Rotate(angle_range=(90, 90), p=0),
                slt.Scale(range_y=(0, 0.1), same=True, p=0),
                slt.Shear(range_y=(-0.1, 0.1), p=0),
            ],
            n=3,
        ),
        v_range=(0, 0),
    )

    dc_res = ppl(dc)

    assert np.array_equal(dc.data, dc_res.data)


def test_random_contrast_multiplies_the_data(img_5x5):
    img = img_5x5
    dc = slc.DataContainer((img,), "I")

    ppl = slt.Contrast(p=1, contrast_range=(2, 2))
    dc_res = ppl(dc)

    assert np.array_equal(dc.data[0] * 2, dc_res.data[0])


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


@pytest.mark.parametrize(
    "img, expected, cut_size",
    [
        (lf("img_7x7"), np.zeros((7, 7, 1), dtype=np.uint8), 7),
        (lf("img_6x6"), np.zeros((6, 6, 1), dtype=np.uint8), 6),
        (lf("img_6x6_rgb"), np.zeros((6, 6, 3), dtype=np.uint8), 6),
        (lf("img_7x7"), np.zeros((7, 7, 1), dtype=np.uint8), 1.0),
        (lf("img_6x6"), np.zeros((6, 6, 1), dtype=np.uint8), 1.0),
        (lf("img_6x6_rgb"), np.zeros((6, 6, 3), dtype=np.uint8), 1.0),
    ],
)
def test_cutout_blacks_out_image(img, expected, cut_size):
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.CutOut(p=1, cutout_size=cut_size)])

    dc_res = trf(dc, return_torch=False)

    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("cut_size", [1, (1, 1), 0.5, (0.5, 0.5)])
def test_cutout_1x1_blacks_corner_pixels_2x2_img(img_2x2, cut_size):
    dc = slc.DataContainer((img_2x2.copy(),), "I")
    trf = slc.Stream([slt.CutOut(p=1, cutout_size=cut_size)])
    dc_res = trf(dc, return_torch=False)

    equal = 0
    for i in range(2):
        for j in range(2):
            tmp_opt = img_2x2.copy()
            tmp_opt[i, j] = 0
            if np.array_equal(dc_res.data[0], tmp_opt):
                equal += 1

    assert equal == 1


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


@pytest.mark.parametrize(
    "img, d_range, rotate, ratio, expected",
    # With this seed, the 3x3 image is inside a single dropped square, apart from one kept vertical line in the middle
    [(np.ones((3, 3, 3)), (5, 10), 1, 0.2, np.broadcast_to(np.array([0.0, 1.0, 0.0])[None, :, None], (3, 3, 3)))],
)
def test_gridmask_transform(img, d_range, rotate, ratio, expected):
    np.random.seed(42)
    random.seed(42)
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.GridMask(d_range=d_range, rotate=rotate, ratio=ratio, p=1)])

    dc_res = trf(dc, return_torch=False)
    print(dc_res.data[0])
    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("d_range", [(-1, 0)])
def test_gridmask_d_range_raises_error_when_wrong(d_range):
    with pytest.raises(ValueError):
        slt.GridMask(d_range=d_range, p=1)


@pytest.mark.parametrize("rotate", [0.1])
def test_gridmask_rotate_raises_error_when_wrong(rotate):
    with pytest.raises(TypeError):
        slt.GridMask(rotate=rotate, p=1)


@pytest.mark.parametrize("rotate", [(0.1, 0.1)])
def test_gridmask_rotate_raises_error_when_wrong_tuple(rotate):
    with pytest.raises(TypeError):
        slt.GridMask(rotate=rotate, p=1)


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


@pytest.mark.parametrize(
    "crop_mode, x, y",
    [("c", 3, 2), ("lt", 0, 0), ("rt", 6, 0), ("lb", 0, 5), ("rb", 6, 5)],
)
def test_crop_modes(crop_mode, x, y):
    h, w, crop_w, crop_h = 9, 10, 4, 4
    img = np.arange(h * w, dtype=np.uint8).reshape(h, w, 1)
    mask = img[..., 0].copy()
    dmap = img[..., 0].astype(np.float32)
    kpts = slc.Keypoints(np.array([[x + 1.0, y + 2.0]]), h, w)

    trf = slt.Crop(crop_to=(crop_w, crop_h), crop_mode=crop_mode)
    img_res, mask_res, dmap_res, kpts_res = trf(slc.DataContainer((img, mask, dmap, kpts), "IMDP")).data

    np.testing.assert_array_equal(img_res, img[y : y + crop_h, x : x + crop_w])
    np.testing.assert_array_equal(mask_res, mask[y : y + crop_h, x : x + crop_w])
    np.testing.assert_array_equal(dmap_res, dmap[y : y + crop_h, x : x + crop_w])
    np.testing.assert_array_equal(kpts_res.data, [[1.0, 2.0]])


def test_crop_five_crops_for_test_time_augmentation():
    # 3x3 blocks of 4x4 pixels, each block filled with its own value
    blocks = np.arange(9, dtype=np.uint8).reshape(3, 3)
    img = np.kron(blocks, np.ones((4, 4), dtype=np.uint8))[..., np.newaxis]

    values = {}
    for crop_mode in ["lt", "rt", "c", "lb", "rb"]:
        res = slt.Crop(crop_to=4, crop_mode=crop_mode)(slc.DataContainer((img,), "I")).data[0]
        assert len(np.unique(res)) == 1
        values[crop_mode] = int(res[0, 0, 0])

    assert values == {"lt": 0, "rt": 2, "c": 4, "lb": 6, "rb": 8}


def test_crop_random_mode_stays_inside_the_image():
    img = np.zeros((9, 10, 1), dtype=np.uint8)
    for _ in range(50):
        trf = slt.Crop(crop_to=(4, 4), crop_mode="r")
        trf(slc.DataContainer((img,), "I"))
        assert 0 <= trf.state_dict["x"] <= 6 and 0 <= trf.state_dict["y"] <= 5


@pytest.mark.parametrize("crop_mode", ["lc", "center", "t", None])
def test_crop_wrong_mode(crop_mode):
    with pytest.raises(ValueError):
        slt.Crop(crop_to=4, crop_mode=crop_mode)


@pytest.mark.parametrize("crop_mode", ["lt", "rt", "lb", "rb"])
def test_crop_mode_serialization(crop_mode):
    trf = slu.from_yaml(slc.Stream([slt.Crop(crop_to=4, crop_mode=crop_mode)]).to_yaml()).transforms[0]
    assert trf.crop_mode == crop_mode


def _cutout_rectangle(img):
    """Width and height of the single blacked out rectangle in an image of ones."""
    ys, xs = np.nonzero(img[..., 0] == 0)
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


@pytest.mark.parametrize(
    "cutout_size, w_range, h_range",
    [
        (((2, 6), (3, 5)), (2, 6), (3, 5)),
        (((0.1, 0.3), (0.2, 0.4)), (4, 12), (8, 16)),
        ((4, (3, 5)), (4, 4), (3, 5)),
    ],
)
def test_cutout_random_size(cutout_size, w_range, h_range):
    random.seed(0)
    sizes = set()
    for _ in range(100):
        img = np.ones((40, 40, 1), dtype=np.uint8)
        res = slt.CutOut(cutout_size=cutout_size, p=1)(slc.DataContainer((img,), "I")).data[0]
        w, h = _cutout_rectangle(res)
        assert w_range[0] <= w <= w_range[1]
        assert h_range[0] <= h <= h_range[1]
        sizes.add((w, h))
    assert len(sizes) > 1


def test_cutout_sizes_are_sampled_per_cut():
    random.seed(0)
    trf = slt.CutOut(cutout_size=((2, 10), (2, 10)), n_cuts=5, p=1)
    trf(slc.DataContainer((np.ones((40, 40, 1), dtype=np.uint8),), "I"))
    sizes = list(zip(trf.state_dict["cut_size_x"], trf.state_dict["cut_size_y"]))
    assert len(sizes) == 5
    assert len(set(sizes)) > 1


def test_cutout_random_number_of_cuts():
    random.seed(0)
    n_cuts = set()
    for _ in range(50):
        trf = slt.CutOut(cutout_size=2, n_cuts=(1, 4), p=1)
        trf(slc.DataContainer((np.ones((40, 40, 1), dtype=np.uint8),), "I"))
        assert 1 <= len(trf.state_dict["x"]) <= 4
        n_cuts.add(len(trf.state_dict["x"]))
    assert n_cuts == {1, 2, 3, 4}


def test_cutout_fixed_size_keeps_the_random_sequence():
    # A fixed size must not draw random numbers, so that seeded pipelines give the same results as before
    random.seed(0)
    trf = slt.CutOut(cutout_size=(3, 4), n_cuts=2, p=1)
    trf(slc.DataContainer((np.ones((20, 20, 1), dtype=np.uint8),), "I"))
    random.seed(0)
    random.random()  # Whether to apply the transform
    expected = [
        int(random.random() * 17),
        int(random.random() * 16),
        int(random.random() * 17),
        int(random.random() * 16),
    ]
    xy = [v for pair in zip(trf.state_dict["x"], trf.state_dict["y"]) for v in pair]
    assert xy == expected


@pytest.mark.parametrize(
    "cutout_size, n_cuts, error",
    [
        (((6, 2), 2), 1, ValueError),
        (((-1, 2), 2), 1, ValueError),
        (((2, 4), 0.5), 1, TypeError),
        (((2, 4.5), 2), 1, TypeError),
        (((2, 4, 6), 2), 1, ValueError),
        ((2, 2, 2), 1, ValueError),
        (2, (3, 1), ValueError),
        (2, (-1, 2), ValueError),
        (2, (1.0, 2), TypeError),
        (2, 1.5, TypeError),
    ],
)
def test_cutout_wrong_random_size_or_n_cuts(cutout_size, n_cuts, error):
    with pytest.raises(error):
        slt.CutOut(cutout_size=cutout_size, n_cuts=n_cuts)


def test_cutout_random_size_too_large():
    trf = slt.CutOut(cutout_size=((2, 50), 2), p=1)
    with pytest.raises(ValueError):
        trf(slc.DataContainer((np.ones((20, 20, 1), dtype=np.uint8),), "I"))


def test_cutout_random_size_serialization():
    trf = slt.CutOut(cutout_size=((2, 6), (0, 4)), n_cuts=(1, 3))
    restored = slu.from_yaml(slc.Stream([trf]).to_yaml()).transforms[0]
    assert tuple(map(tuple, restored.cutout_size)) == ((2, 6), (0, 4))
    assert tuple(restored.n_cuts) == (1, 3)


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


@pytest.mark.parametrize("h, w", [(60, 80), (61, 81), (60, 60), (61, 61), (7, 4)])
@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_rotate_90_keypoints_follow_the_image(h, w, k):
    rng = np.random.default_rng(0)
    xy = np.stack([rng.integers(0, w, 5), rng.integers(0, h, 5)], axis=1)
    for x, y in xy:
        img = np.zeros((h, w), dtype=np.uint8)
        img[y, x] = 1
        kpts = slc.Keypoints(np.array([[float(x), float(y)]]), h, w)
        img_res, kpts_res = slt.Rotate90(k=k, p=1)(slc.DataContainer((img, kpts), "MP")).data
        ys, xs = np.nonzero(img_res)
        np.testing.assert_array_equal(kpts_res.data, [[xs[0], ys[0]]])
        assert (kpts_res.height, kpts_res.width) == img_res.shape


@pytest.mark.parametrize("rotate_first", [True, False])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate_90_in_optimized_stream(rotate_first, k):
    h, w = 41, 51
    img = np.zeros((h, w), dtype=np.uint8)
    img[10, 25] = 1
    kpts = slc.Keypoints(np.array([[25.0, 10.0]]), h, w)

    def make_transforms():
        rotate = slt.Rotate90(k=k, p=1)
        translate = slt.Translate(range_x=(5, 5), range_y=(-3, -3), p=1)
        return [rotate, translate] if rotate_first else [translate, rotate]

    expected = slc.Stream(make_transforms())(slc.DataContainer((img, kpts), "MP"), return_torch=False)
    res = slc.Stream(make_transforms(), optimize_stack=True)(slc.DataContainer((img, kpts), "MP"), return_torch=False)

    np.testing.assert_array_equal(np.argwhere(res.data[0] == 1), np.argwhere(expected.data[0] == 1))
    np.testing.assert_allclose(res.data[1].data, expected.data[1].data, atol=1e-6)


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


def _rotate90_data(fmt):
    h, w = 41, 51
    rng = np.random.default_rng(0)
    if fmt == "I":
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    if fmt == "M":
        return (rng.random((h, w)) > 0.5).astype(np.uint8)
    if fmt == "D":
        return rng.random((h, w)).astype(np.float32)
    if fmt == "P":
        return slc.Keypoints(np.array([[25.0, 10.0], [3.0, 37.0]]), h, w)
    return slc.BoundingBoxes([[10.0, 5.0, 30.0, 20.0]], h, w)


@pytest.fixture(params=["I", "M", "D", "P", "B"], ids=["image", "mask", "distance_map", "keypoints", "bboxes"])
def rotate90_item(request):
    return request.param, _rotate90_data(request.param)


@pytest.fixture(params=[1, 2, 3], ids=["k1", "k2", "k3"])
def rotate90_k(request):
    return request.param


def _as_array(item):
    return item.data if isinstance(item, (slc.Keypoints, slc.BoundingBoxes)) else item


def test_fused_rotate_90_transforms_all_data_types(rotate90_item, rotate90_k):
    fmt, item = rotate90_item

    def make_transforms():
        return [slt.Rotate90(k=rotate90_k, p=1), slt.Translate(range_x=(5, 5), range_y=(-3, -3), p=1)]

    # max_bbox_* are disabled, so that the boxes are compared as they are
    kwargs = {"max_bbox_area_loss": None, "max_bbox_outside": None}
    expected = slc.Stream(make_transforms(), **kwargs)(slc.DataContainer((item,), fmt), return_torch=False).data[0]
    res = slc.Stream(make_transforms(), optimize_stack=True, **kwargs)(
        slc.DataContainer((item,), fmt), return_torch=False
    ).data[0]
    np.testing.assert_allclose(_as_array(res), _as_array(expected), atol=1e-4)


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


# GridMask. Without rotation, the grid is made of kept lines of width ``l = ceil(d * ratio)`` repeated with the
# period ``d``, and the dropped region is the squares of side ``s = d - l`` between them.
def _gridmask_line_and_square(d, ratio):
    line = int(np.ceil(np.round(d * ratio, 6)))
    return line, d - line


def _gridmask_dropped(shape, seed, **kwargs):
    random.seed(seed)
    img = np.ones((*shape, 1), dtype=np.float32)
    res = slt.GridMask(p=1, **kwargs)(slc.DataContainer((img,), "I"), return_torch=False).data[0]
    assert set(np.unique(res)).issubset({0, 1})
    return res[:, :, 0] == 0


def _runs(vec):
    """Lengths of the runs of ``True`` and ``False`` in ``vec``, without the runs touching the borders."""
    edges = np.flatnonzero(np.diff(vec.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [vec.size]))
    runs = [(bool(vec[s]), e - s) for s, e in zip(bounds[:-1], bounds[1:])]
    return runs[1:-1]


@pytest.fixture(params=[4, 10, 16], ids=lambda d: f"d{d}")
def gridmask_d(request):
    return request.param


@pytest.fixture(params=[0.3, 0.5, 0.7], ids=lambda r: f"ratio{r}")
def gridmask_ratio(request):
    return request.param


@pytest.fixture(params=[(64, 64), (48, 80), (37, 53)], ids=["square", "non_square", "odd"])
def gridmask_shape(request):
    return request.param


@pytest.fixture(params=["crop", "none", None])
def gridmask_default_mode(request):
    return request.param


@pytest.fixture(params=["reverse", "reserve"])
def gridmask_inverse_mode(request):
    return request.param


@pytest.fixture(params=[0, 15, (30, 45), (-20, -10)], ids=["no_rotation", "int", "tuple", "negative_tuple"])
def gridmask_rotate(request):
    return request.param


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gridmask_drops_squares_between_kept_lines(
    gridmask_d, gridmask_ratio, gridmask_shape, gridmask_default_mode, seed
):
    d, (h, w) = gridmask_d, gridmask_shape
    line, square = _gridmask_line_and_square(d, gridmask_ratio)
    dropped = _gridmask_dropped(gridmask_shape, seed, d_range=(d, d), ratio=gridmask_ratio, mode=gridmask_default_mode)

    rows, cols = dropped.any(axis=1), dropped.any(axis=0)
    # The dropped region is a union of squares: a pixel is dropped iff both its row and its column are between lines
    np.testing.assert_array_equal(dropped, np.outer(rows, cols))
    for vec in (rows, cols):
        # Any d consecutive rows (columns) contain exactly one side of a square, also at the image borders
        windows = np.lib.stride_tricks.sliding_window_view(vec, d).sum(axis=1)
        np.testing.assert_array_equal(windows, square)
        for is_dropped, length in _runs(vec):
            assert length == (square if is_dropped else line)

    assert abs(dropped.mean() - (square / d) ** 2) <= 2 * square / min(h, w)


def test_gridmask_dropped_fraction_is_one_minus_ratio_squared():
    dropped = _gridmask_dropped((256, 256), 0, d_range=(16, 16), ratio=0.5, mode="crop")
    assert dropped.mean() == pytest.approx(0.25, abs=0.02)


def test_gridmask_inverse_mode_drops_the_complement(
    gridmask_d, gridmask_ratio, gridmask_shape, gridmask_inverse_mode, gridmask_rotate
):
    kwargs = {"d_range": (gridmask_d, gridmask_d), "ratio": gridmask_ratio, "rotate": gridmask_rotate}
    dropped = _gridmask_dropped(gridmask_shape, 0, mode="crop", **kwargs)
    dropped_inverse = _gridmask_dropped(gridmask_shape, 0, mode=gridmask_inverse_mode, **kwargs)
    np.testing.assert_array_equal(dropped_inverse, ~dropped)


@pytest.mark.parametrize("modes", [("crop", "none"), ("crop", None), ("reverse", "reserve")])
def test_gridmask_mode_aliases_are_identical(modes, gridmask_rotate):
    kwargs = {"d_range": (5, 20), "ratio": 0.4, "rotate": gridmask_rotate}
    res = [_gridmask_dropped((50, 70), 7, mode=mode, **kwargs) for mode in modes]
    np.testing.assert_array_equal(res[0], res[1])


def test_gridmask_rotation_angle_is_sampled_in_range(gridmask_rotate, gridmask_default_mode, monkeypatch):
    angles = []
    get_rotation_matrix = cv2.getRotationMatrix2D

    def recording_get_rotation_matrix(center, angle, scale):
        angles.append(angle)
        return get_rotation_matrix(center, angle, scale)

    monkeypatch.setattr(cv2, "getRotationMatrix2D", recording_get_rotation_matrix)
    low, high = (-gridmask_rotate, gridmask_rotate) if isinstance(gridmask_rotate, int) else gridmask_rotate
    for seed in range(20):
        dropped = _gridmask_dropped(
            (40, 60), seed, d_range=(8, 12), ratio=0.5, rotate=gridmask_rotate, mode=gridmask_default_mode
        )
        # Every unit of the grid drops about (1 - ratio) ** 2 of its area, whatever the angle
        assert dropped.mean() == pytest.approx(0.25, abs=0.12)

    if gridmask_rotate == 0:
        assert angles == []
    else:
        assert len(angles) == 20
        assert all(low <= angle <= high for angle in angles)


@pytest.mark.parametrize("rotate, expected", [(0, 0), (15, (-15, 15)), ((30, 45), (30, 45)), ((-5, -5), (-5, -5))])
def test_gridmask_rotate_range(rotate, expected):
    assert slt.GridMask(rotate=rotate).rotate == expected


@pytest.mark.parametrize("rotate", [-10, (1, 2, 3), (10,), (45, 30)])
def test_gridmask_rotate_raises_value_error_when_wrong(rotate):
    with pytest.raises(ValueError):
        slt.GridMask(rotate=rotate)


@pytest.mark.parametrize("ratio", [-0.1, 1.5])
def test_gridmask_ratio_raises_error_when_wrong(ratio):
    with pytest.raises(ValueError):
        slt.GridMask(ratio=ratio)


@pytest.mark.parametrize("mode", ["inverse", "reversed"])
def test_gridmask_mode_raises_error_when_wrong(mode):
    with pytest.raises(ValueError):
        slt.GridMask(mode=mode)


@pytest.mark.parametrize("n_channels", [1, 3, 5])
def test_gridmask_applies_the_same_mask_to_every_channel(n_channels, gridmask_rotate, gridmask_inverse_mode):
    random.seed(3)
    img = np.random.randint(1, 255, (45, 61, n_channels), dtype=np.uint8)
    img_orig = img.copy()
    trf = slt.GridMask(d_range=(6, 15), ratio=0.5, rotate=gridmask_rotate, mode=gridmask_inverse_mode, p=1)
    res = trf(slc.DataContainer((img,), "I"), return_torch=False).data[0]

    dropped = res == 0
    assert dropped.any() and not dropped.all()
    for k in range(n_channels):
        np.testing.assert_array_equal(dropped[:, :, k], dropped[:, :, 0])
    np.testing.assert_array_equal(res[~dropped], img_orig[~dropped])


def test_gridmask_supports_2d_images():
    random.seed(0)
    img = np.ones((40, 50), dtype=np.uint8)
    res = slt.GridMask(d_range=(10, 10), ratio=0.5, p=1)(slc.DataContainer((img,), "I"), return_torch=False).data[0]
    assert res.shape == (40, 50, 1)
    assert (res == 0).mean() == pytest.approx(0.25, abs=0.1)


def test_gridmask_modifies_the_image_in_place():
    img = np.ones((32, 32, 3), dtype=np.uint8)
    res = slt.GridMask(d_range=(8, 8), p=1)(slc.DataContainer((img,), "I"), return_torch=False).data[0]
    assert res is img
    assert (img == 0).any()


@pytest.mark.parametrize("mode", ["crop", "reverse"])
def test_gridmask_does_not_change_masks_keypoints_distance_maps_boxes_and_labels(mode, gridmask_rotate):
    h, w = 41, 51
    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, (h, w, 3), dtype=np.uint8)
    mask = (rng.random((h, w)) > 0.5).astype(np.uint8)
    dmap = rng.random((h, w)).astype(np.float32)
    kpts = slc.Keypoints(np.array([[25.0, 10.0], [3.0, 37.0]]), h, w)
    boxes = slc.BoundingBoxes([[10.0, 5.0, 30.0, 20.0]], h, w)
    data = (img, mask.copy(), dmap.copy(), slc.Keypoints(kpts.data.copy(), h, w), boxes, 1)

    trf = slt.GridMask(d_range=(5, 10), rotate=gridmask_rotate, mode=mode, p=1)
    res = trf(slc.DataContainer(data, "IMDPBL"), return_torch=False).data
    assert (res[0] == 0).any()
    np.testing.assert_array_equal(res[1], mask)
    np.testing.assert_array_equal(res[2], dmap)
    np.testing.assert_array_equal(res[3].data, kpts.data)
    np.testing.assert_array_equal(res[4].data, [[10.0, 5.0, 30.0, 20.0]])
    assert res[5] == 1
