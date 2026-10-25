import cv2
import numpy as np
import pytest

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

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
