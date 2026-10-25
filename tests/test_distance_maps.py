import inspect
import random

import cv2
import numpy as np
import pytest
import torch

import solt.core as slc
import solt.transforms as slt

FLOAT_DTYPES = [np.float16, np.float32, np.float64]
# Odd size, so that the transforms' origin is exactly in the center of the image
SIZE = 65
CENTER = 32
RADIUS = 10


def disc_sdf(h=SIZE, w=SIZE, cx=CENTER, cy=CENTER, r=RADIUS, dtype=np.float32):
    """Signed distance field of a disc: negative inside, positive outside, in pixels."""
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    return (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r).astype(dtype)


def make_dmap(shape, dtype=np.float32):
    dmap = disc_sdf(dtype=np.float64)
    if len(shape) == 3:
        dmap = np.stack([dmap + i for i in range(shape[2])], axis=-1)
    return dmap.astype(dtype)


def make_img():
    return np.random.randint(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)


def geometric_transforms():
    return [
        slt.Flip(p=1, axis=1),
        slt.Flip(p=1, axis=-1),
        slt.Rotate90(k=1, p=1),
        slt.Rotate(angle_range=(30, 30), p=1),
        slt.Shear(range_x=(0.1, 0.1), range_y=(0.1, 0.1), p=1),
        slt.Scale(range_x=(1.3, 1.3), p=1),
        slt.Translate(range_x=(5, 5), range_y=(5, 5), p=1),
        slt.Projection(v_range=(1e-4, 1e-4), p=1),
        slt.Pad(pad_to=80),
        slt.Crop(crop_to=40),
        slt.Resize(resize_to=(33, 33)),
        slt.RandomResizedCrop(resize_to=33),
    ]


def transform_ids(trfs):
    return [type(t).__name__ for t in trfs]


# ----------------------------------------------------------------------------------------------------------------------
# DataContainer
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(SIZE, SIZE), (SIZE, SIZE, 1), (SIZE, SIZE, 5)])
def test_data_container_accepts_float_distance_maps(dtype, shape):
    dmap = make_dmap(shape, dtype)
    dc = slc.DataContainer((make_img(), dmap), "ID")
    assert dc.data_format == "ID"
    assert dc.data[1] is dmap


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.uint32, bool])
def test_data_container_rejects_non_float_distance_maps(dtype):
    with pytest.raises(TypeError):
        slc.DataContainer((np.zeros((5, 5), dtype=dtype),), "D")


@pytest.mark.parametrize("shape", [(5,), (5, 5, 2, 1)])
def test_data_container_rejects_wrong_distance_map_dims(shape):
    with pytest.raises(ValueError):
        slc.DataContainer((np.zeros(shape, dtype=np.float32),), "D")


def test_data_container_rejects_non_array_distance_map():
    with pytest.raises(TypeError):
        slc.DataContainer(([[0.0, 1.0], [1.0, 0.0]],), "D")


def test_distance_map_coordinate_frame_is_validated():
    with pytest.raises(ValueError):
        slc.DataContainer((make_img(), np.zeros((SIZE, SIZE + 1), dtype=np.float32)), "ID")


def test_distance_map_default_settings():
    dc = slc.DataContainer((make_dmap((SIZE, SIZE)),), "D")
    assert dc.transform_settings[0] == {
        "interpolation": ("bilinear", "strict"),
        "padding": ("z", "inherit"),
        "fill": None,
    }


def test_distance_map_custom_settings():
    settings = {0: {"interpolation": "bicubic", "padding": "r", "fill": 100}}
    dc = slc.DataContainer((make_dmap((SIZE, SIZE)),), "D", settings)
    assert dc.transform_settings[0] == {
        "interpolation": ("bicubic", "strict"),
        "padding": ("r", "strict"),
        "fill": 100,
    }


def test_distance_map_fill_none_means_edge_padding():
    dc = slc.DataContainer((make_dmap((SIZE, SIZE)),), "D", {0: {"fill": None}})
    assert dc.transform_settings[0]["fill"] is None


@pytest.mark.parametrize("fill", ["100", True, [1.0], float("nan"), float("inf")])
def test_distance_map_wrong_fill(fill):
    with pytest.raises((TypeError, ValueError)):
        slc.DataContainer((make_dmap((SIZE, SIZE)),), "D", {0: {"fill": fill}})


@pytest.mark.parametrize(
    "data, fmt",
    [
        ((np.zeros((SIZE, SIZE, 3), dtype=np.uint8),), "I"),
        ((np.zeros((SIZE, SIZE), dtype=np.uint8),), "M"),
        ((1,), "L"),
    ],
)
def test_fill_is_only_allowed_for_distance_maps(data, fmt):
    with pytest.raises(TypeError):
        slc.DataContainer(data, fmt, {0: {"fill": 1.0}})


def test_distance_map_from_dict_order():
    img, mask, dmap = make_img(), np.zeros((SIZE, SIZE), np.uint8), make_dmap((SIZE, SIZE))
    dc = slc.DataContainer.from_dict({"label": 1, "distance_map": dmap, "mask": mask, "image": img})
    assert dc.data_format == "IMDL"

    dc = slc.DataContainer.from_dict({"distance_maps": [dmap, dmap + 1], "image": img})
    assert dc.data_format == "IDD"
    assert np.array_equal(dc.data[2], dmap + 1)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape, expected_shape",
    [((SIZE, SIZE), (1, SIZE, SIZE)), ((SIZE, SIZE, 1), (1, SIZE, SIZE)), ((SIZE, SIZE, 5), (5, SIZE, SIZE))],
)
def test_distance_map_to_torch(dtype, shape, expected_shape):
    dmap = make_dmap(shape, dtype)
    res = slc.DataContainer((dmap,), "D").to_torch(as_dict=True)["distance_map"]

    assert tuple(res.shape) == expected_shape
    assert isinstance(res, torch.FloatTensor)
    expected = dmap[np.newaxis] if dmap.ndim == 2 else dmap.transpose((2, 0, 1))
    assert np.array_equal(res.numpy(), expected.astype(np.float32))


def test_distance_maps_to_torch_plural():
    dmap = make_dmap((SIZE, SIZE))
    res = slc.DataContainer((make_img(), dmap, dmap), "IDD").to_torch(as_dict=True)
    assert set(res.keys()) == {"image", "distance_maps"}
    assert len(res["distance_maps"]) == 2


# ----------------------------------------------------------------------------------------------------------------------
# Generic behaviour of all transforms
# ----------------------------------------------------------------------------------------------------------------------


def test_all_transforms_define_distance_map_behaviour():
    for name in slt.__all__:
        trf = getattr(slt, name)
        if not inspect.isclass(trf) or not issubclass(trf, slc.BaseTransform):
            continue
        assert (
            trf._apply_distance_map is not slc.BaseTransform._apply_distance_map
        ), f"{name} does not define how to transform distance maps"


@pytest.mark.parametrize("trf", geometric_transforms(), ids=transform_ids(geometric_transforms()))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(SIZE, SIZE), (SIZE, SIZE, 1), (SIZE, SIZE, 5)])
def test_geometric_transforms_keep_distance_map_dtype_and_shape(trf, dtype, shape):
    dmap = make_dmap(shape, dtype)
    img_res, dmap_res = trf(slc.DataContainer((make_img(), dmap), "ID")).data

    assert dmap_res.dtype == dtype
    assert dmap_res.ndim == dmap.ndim
    assert dmap_res.shape[:2] == img_res.shape[:2]
    if dmap.ndim == 3:
        assert dmap_res.shape[2] == dmap.shape[2]
    assert np.all(np.isfinite(dmap_res))


@pytest.mark.parametrize("trf", geometric_transforms(), ids=transform_ids(geometric_transforms()))
def test_float16_distance_maps_match_float32(trf):
    dmap = make_dmap((SIZE, SIZE), np.float32)
    # Same random parameters for both calls, so the difference comes from float16 precision only
    random.seed(0)
    res_f32 = trf(slc.DataContainer((dmap,), "D")).data[0]
    random.seed(0)
    res_f16 = trf(slc.DataContainer((dmap.astype(np.float16),), "D")).data[0]
    np.testing.assert_allclose(res_f16.astype(np.float32), res_f32, rtol=1e-3, atol=2e-2)


@pytest.mark.parametrize(
    "trf",
    [
        slt.Noise(p=1),
        slt.Blur(p=1),
        slt.CutOut(cutout_size=10, p=1),
        slt.SaltAndPepper(p=1),
        slt.GammaCorrection(p=1),
        slt.Contrast(p=1),
        slt.Brightness(p=1),
        slt.HSV(h_range=(10, 10), p=1),
        slt.JPEGCompression(p=1),
        slt.GridMask(d_range=(10, 20), p=1),
        slt.KeypointsJitter(p=1),
    ],
    ids=lambda t: type(t).__name__,
)
def test_intensity_transforms_do_not_change_distance_maps(trf):
    dmap = make_dmap((SIZE, SIZE))
    dmap_res = trf(slc.DataContainer((make_img(), dmap), "ID")).data[1]
    assert np.array_equal(dmap_res, dmap)


def test_custom_transform_without_distance_map_support_raises():
    class MyTransform(slc.BaseTransform):
        def _apply_img(self, img, settings):
            return img

        def _apply_mask(self, mask, settings):
            return mask

        def _apply_labels(self, labels, settings):
            return labels

        def _apply_pts(self, pts, settings):
            return pts

    trf = MyTransform(p=1)
    # Data without distance maps keeps working
    trf(slc.DataContainer((make_img(),), "I"))
    with pytest.raises(NotImplementedError):
        trf(slc.DataContainer((make_img(), make_dmap((SIZE, SIZE))), "ID"))


def test_distance_map_outside_of_data_indices_is_not_changed():
    dmap = make_dmap((SIZE, SIZE))
    res = slt.Flip(p=1, data_indices=(0,))(slc.DataContainer((make_img(), dmap), "ID")).data[1]
    assert np.array_equal(res, dmap)
    assert res is not dmap


def test_distance_map_settings_are_kept_in_stream():
    settings = {1: {"interpolation": "bicubic", "fill": 7.5}}
    dc = slc.DataContainer((make_img(), make_dmap((SIZE, SIZE))), "ID", settings)
    ppl = slc.Stream([slt.Rotate(angle_range=(10, 10), p=1), slt.Flip(p=1), slt.Scale(range_x=(1.2, 1.2), p=1)])
    assert ppl(dc, return_torch=False).transform_settings == dc.transform_settings


# ----------------------------------------------------------------------------------------------------------------------
# Rigid transforms: values do not change
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_flip_distance_map(axis):
    dmap = make_dmap((SIZE, SIZE, 5))
    res = slt.Flip(p=1, axis=axis)(slc.DataContainer((dmap,), "D")).data[0]
    expected = {0: dmap[::-1], 1: dmap[:, ::-1], -1: dmap[::-1, ::-1]}[axis]
    assert np.array_equal(res, expected)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate90_distance_map(k):
    dmap = disc_sdf(cx=20, cy=40)
    res = slt.Rotate90(k=k, p=1)(slc.DataContainer((dmap,), "D")).data[0]
    assert np.array_equal(res, np.rot90(dmap, -k))


def test_crop_distance_map():
    dmap = make_dmap((SIZE, SIZE, 2))
    trf = slt.Crop(crop_to=(20, 30), crop_mode="r")
    res = trf(slc.DataContainer((dmap,), "D")).data[0]
    x, y = trf.state_dict["x"], trf.state_dict["y"]
    assert np.array_equal(res, dmap[y : y + 30, x : x + 20])


def test_rotate_distance_map_keeps_values():
    # A centered disc is rotation invariant, so the SDF must stay the same, besides interpolation errors
    dmap = disc_sdf()
    res = slt.Rotate(angle_range=(37, 37), p=1)(slc.DataContainer((dmap,), "D")).data[0]
    inside_frame = disc_sdf(r=0) < CENTER - 2
    np.testing.assert_allclose(res[inside_frame], dmap[inside_frame], atol=0.5)


# ----------------------------------------------------------------------------------------------------------------------
# Scaling transforms: values are rescaled
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.6, 1.5])
def test_scale_rescales_distance_map_values(scale):
    dmap = disc_sdf()
    res = slt.Scale(range_x=(scale, scale), p=1)(slc.DataContainer((dmap,), "D")).data[0]
    expected = disc_sdf(r=RADIUS * scale)
    # Close to the borders, the values come from the padding
    valid = disc_sdf(r=0) < min(CENTER, CENTER * scale) - 2
    np.testing.assert_allclose(res[valid], expected[valid], atol=0.5)


def test_scale_with_keypoints_rescales_distance_map_values():
    # Keypoints switch off the fast mode, so the coordinate frame changes. The values must be scaled the same way.
    dmap = disc_sdf()
    kpts = slc.Keypoints(np.array([[CENTER, CENTER]]), SIZE, SIZE)
    dc_res = slt.Scale(range_x=(1.5, 1.5), p=1)(slc.DataContainer((dmap, kpts), "DP"))
    res, kpts_res = dc_res.data
    cx, cy = kpts_res.data[0]
    expected = disc_sdf(h=res.shape[0], w=res.shape[1], cx=cx, cy=cy, r=RADIUS * 1.5)
    # The outermost pixels of the new frame may come from the padding
    np.testing.assert_allclose(res[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1.0)


def test_shear_rescales_distance_map_values_by_area_change():
    dmap = np.full((SIZE, SIZE), 10.0, dtype=np.float32)
    trf = slt.Shear(range_x=(0.5, 0.5), range_y=(0.5, 0.5), p=1)
    res = trf(slc.DataContainer((dmap,), "D")).data[0]
    # The linear part is [[1, 0.5], [0.5, 1]], so the area changes by 0.75
    assert np.allclose(res[CENTER, CENTER], 10 * np.sqrt(0.75))


def test_projection_without_perspective_matches_affine_scaling():
    dmap = np.full((SIZE, SIZE), 10.0, dtype=np.float32)
    scale = slt.Scale(range_x=(2, 2), p=1)
    res = slt.Projection(affine_transforms=slc.Stream([scale]), v_range=(0, 0), p=1)(
        slc.DataContainer((dmap,), "D")
    ).data[0]
    assert np.allclose(res[CENTER, CENTER], 20)


@pytest.mark.parametrize("resize_to", [(33, 33), (130, 130), (40, 90)])
def test_resize_rescales_distance_map_values(resize_to):
    dmap = disc_sdf()
    res = slt.Resize(resize_to=resize_to)(slc.DataContainer((dmap,), "D")).data[0]
    sx, sy = resize_to[0] / SIZE, resize_to[1] / SIZE
    assert res.shape == (resize_to[1], resize_to[0])

    if sx == sy:
        # Pixel centers of the output in the input coordinates (OpenCV convention)
        yy, xx = np.mgrid[: resize_to[1], : resize_to[0]].astype(np.float64)
        xs, ys = (xx + 0.5) / sx - 0.5, (yy + 0.5) / sy - 0.5
        expected = (np.sqrt((xs - CENTER) ** 2 + (ys - CENTER) ** 2) - RADIUS) * sx
        np.testing.assert_allclose(res, expected, atol=0.5 * max(sx, 1))
    else:
        assert np.allclose(res.mean(), dmap.mean() * np.sqrt(sx * sy), rtol=0.05)


@pytest.mark.parametrize("trf", [slt.Resize(resize_to=None), slt.Pad(pad_to=None), slt.Crop(crop_to=None)])
def test_no_op_size_transforms_do_not_change_distance_map(trf):
    dmap = make_dmap((SIZE, SIZE, 2))
    assert np.array_equal(trf(slc.DataContainer((dmap,), "D")).data[0], dmap)


def test_random_resized_crop_rescales_distance_map_values():
    dmap = disc_sdf()
    trf = slt.RandomResizedCrop(resize_to=33, scale=(1, 1), ratio=(1, 1))
    res = trf(slc.DataContainer((dmap,), "D")).data[0]
    expected = slt.Resize(resize_to=(33, 33))(slc.DataContainer((dmap,), "D")).data[0]
    assert np.array_equal(res, expected)


# ----------------------------------------------------------------------------------------------------------------------
# Padding
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(SIZE, SIZE), (SIZE, SIZE, 5)])
def test_pad_distance_map_replicates_edges_by_default(shape):
    dmap = make_dmap(shape)
    res = slt.Pad(pad_to=80)(slc.DataContainer((dmap,), "D")).data[0]
    pad_width = [(7, 8), (7, 8)] + ([(0, 0)] if len(shape) == 3 else [])
    assert np.array_equal(res, np.pad(dmap, pad_width, mode="edge"))


@pytest.mark.parametrize("shape", [(SIZE, SIZE), (SIZE, SIZE, 5)])
def test_pad_distance_map_with_fill(shape):
    dmap = make_dmap(shape)
    res = slt.Pad(pad_to=80)(slc.DataContainer((dmap,), "D", {0: {"fill": 100}})).data[0]
    pad_width = [(7, 8), (7, 8)] + ([(0, 0)] if len(shape) == 3 else [])
    assert np.array_equal(res, np.pad(dmap, pad_width, mode="constant", constant_values=100))


def test_pad_distance_map_follows_reflect_padding_of_transform():
    dmap = make_dmap((SIZE, SIZE))
    res = slt.Pad(pad_to=80, padding="r")(slc.DataContainer((dmap,), "D", {0: {"fill": 100}})).data[0]
    assert np.array_equal(res, np.pad(dmap, [(7, 8), (7, 8)], mode="reflect"))


@pytest.mark.parametrize("shape", [(SIZE, SIZE), (SIZE, SIZE, 5)])
def test_rotate_distance_map_with_fill(shape):
    dmap = make_dmap(shape)
    res = slt.Rotate(angle_range=(45, 45), p=1)(slc.DataContainer((dmap,), "D", {0: {"fill": 100}})).data[0]
    # The corners come from outside of the original image
    assert np.all(res[0, 0] == 100)
    assert np.all(res[-1, -1] == 100)


def test_fill_value_is_not_rescaled():
    dmap = disc_sdf()
    res = slt.Scale(range_x=(0.5, 0.5), p=1)(slc.DataContainer((dmap,), "D", {0: {"fill": 100}})).data[0]
    assert res[0, 0] == 100
    np.testing.assert_allclose(res[CENTER, CENTER], -RADIUS * 0.5, atol=0.1)


def test_rotate_distance_map_replicates_edges_by_default():
    dmap = disc_sdf()
    res = slt.Rotate(angle_range=(45, 45), p=1)(slc.DataContainer((dmap,), "D")).data[0]
    # Zero padding would create a false object boundary in the corners
    assert res[0, 0] > 0
    assert res.min() >= dmap.min() - 0.5
    assert np.all(res[disc_sdf(r=0) > CENTER] >= dmap[0, :].min() - 0.5)


# ----------------------------------------------------------------------------------------------------------------------
# Multi-channel distance maps
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic", "lanczos", "area", "nearest"])
@pytest.mark.parametrize(
    "trf",
    [slt.Rotate(angle_range=(30, 30), p=1), slt.Resize(resize_to=(40, 50))],
    ids=["Rotate", "Resize"],
)
def test_multichannel_distance_map_matches_per_channel(trf, interpolation):
    dmap = make_dmap((SIZE, SIZE, 6))
    settings = {0: {"interpolation": interpolation, "fill": 50}}
    res = trf(slc.DataContainer((dmap,), "D", settings)).data[0]

    for c in range(dmap.shape[2]):
        res_c = trf(slc.DataContainer((np.ascontiguousarray(dmap[:, :, c]),), "D", settings)).data[0]
        # OpenCV warps 1-channel arrays with exact coordinates, but multi-channel ones with coordinates rounded to
        # 1/32 px, so the results are not bit-exact. The channels of the map differ by 1, so a mix-up would show here.
        diff = np.abs(res[:, :, c] - res_c)
        assert diff.mean() < 0.05
        assert np.percentile(diff, 99) < 0.5
