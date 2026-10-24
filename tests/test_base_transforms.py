import solt.transforms as slt
import solt.core as slc
import solt.utils as slu
import numpy as np
import pytest
import sys, inspect
import torch

from pytest_lazy_fixtures import lf

from .fixtures import *


def get_transforms_solt():
    trfs = []
    for name, obj in inspect.getmembers(sys.modules["solt.transforms"]):
        if inspect.isclass(obj):
            trfs.append(obj)
    return trfs


def class_accepts(obj, parameter):
    return parameter in inspect.signature(obj.__init__).parameters.keys()


def filter_trfs(trfs, parameter):
    return [t for t in trfs if class_accepts(t, parameter)]


def filter_trfs_subclass(trfs, superclass):
    return [t for t in trfs if issubclass(t, superclass)]


all_trfs_solt = get_transforms_solt()


@pytest.mark.parametrize("trf", filter_trfs(all_trfs_solt, "data_indices"))
def test_data_indices_cant_be_list(trf):
    with pytest.raises(TypeError):
        trf(data_indices=[])


@pytest.mark.parametrize("trf", filter_trfs(all_trfs_solt, "p"))
def test_base_transform_can_take_none_prop_and_it_becomes_0_5(trf):
    assert 0.5 == trf(p=None).p


@pytest.mark.parametrize("trf", filter_trfs(all_trfs_solt, "data_indices"))
def test_data_indices_can_be_only_int(trf):
    with pytest.raises(TypeError):
        trf(data_indices=("2", 34))


@pytest.mark.parametrize("trf", filter_trfs(all_trfs_solt, "data_indices"))
def test_data_indices_can_be_only_nonnegative(trf):
    with pytest.raises(ValueError):
        trf(data_indices=(0, 1, -2))


def test_transform_without_data_indices_is_applied_to_all_items_of_every_call():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
    mask = np.arange(12, dtype=np.uint8).reshape(3, 4)
    flip = slt.Flip(p=1, axis=1)

    flip(slc.DataContainer((img,), "I"))
    img_res, mask_res = flip(slc.DataContainer((img, mask), "IM")).data

    assert np.array_equal(img_res, img[:, ::-1])
    assert np.array_equal(mask_res, mask[:, ::-1])
    assert flip.data_indices is None


@pytest.mark.parametrize("trf", filter_trfs(all_trfs_solt, "p"))
def test_transform_returns_original_data_if_use_transform_is_false(img_2x2, trf):
    dc = slc.DataContainer((img_2x2,), "I")
    trf = trf(p=0)
    res = trf(dc)
    np.testing.assert_array_equal(res.data[0], img_2x2)


@pytest.mark.parametrize(
    "trf",
    [
        slt.Flip,
        slt.HSV,
        slt.Brightness,
    ],
)
def test_transform_returns_original_data_if_not_in_specified_indices(trf, img_3x3_rgb):
    img_3x3 = img_3x3_rgb * 128
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data.copy(), 3, 3)
    dc = slc.DataContainer((img_3x3.copy(), img_3x3.copy(), img_3x3.copy(), img_3x3.copy(), 1, kpts, 2), "IIIILPL")

    kwargs = {"p": 1, "data_indices": (0, 1, 4)}
    if class_accepts(trf, "gain_range"):
        kwargs["gain_range"] = (0.7, 0.9)
    if class_accepts(trf, "brightness_range"):
        kwargs["brightness_range"] = (10, 20)
    if class_accepts(trf, "h_range"):
        kwargs["h_range"] = (50, 50)
        kwargs["s_range"] = (50, 50)
    if class_accepts(trf, "h_range"):
        kwargs["h_range"] = (50, 50)
        kwargs["s_range"] = (50, 50)

    trf = trf(**kwargs)
    res = trf(dc)

    assert np.linalg.norm(res.data[0] - img_3x3) > 0
    assert np.linalg.norm(res.data[1] - img_3x3) > 0
    np.testing.assert_array_equal(res.data[2], img_3x3)
    np.testing.assert_array_equal(res.data[3], img_3x3)
    assert res.data[-1] == 2
    np.testing.assert_array_equal(res.data[5].data, kpts_data)


@pytest.mark.parametrize(
    "img_1,img_2",
    [
        (lf("img_2x2"), lf("img_6x6")),
        (lf("img_3x3"), lf("img_3x4")),
    ],
)
def test_data_dep_trf_raises_value_error_when_imgs_are_of_different_size(img_1, img_2):
    trf = slt.SaltAndPepper(gain_range=0.0, p=1)
    with pytest.raises(ValueError):
        trf(
            slc.DataContainer(
                (
                    1,
                    img_1,
                    img_2,
                ),
                "LII",
            )
        )


def test_transform_returns_original_data_when_not_used_and_applied(img_2x2):
    trf = slt.Flip(p=0)
    dc = slc.DataContainer(img_2x2, "I")
    dc_res = trf(dc)
    assert dc_res == dc


@pytest.mark.parametrize(
    "data",
    [
        [
            123,
        ],
        "123",
        2.3,
    ],
)
def test_wrap_data_throws_a_type_error_when_unknown_type(data):
    with pytest.raises(TypeError):
        slc.BaseTransform.wrap_data(data)


@pytest.mark.parametrize("img", [lf("img_2x2"), lf("img_5x5"), lf("img_7x7")])
def test_data_container_wraps_correctly(img):
    dc1 = slc.DataContainer(img, "I")
    dc2 = slc.DataContainer(img, "I")
    assert slc.BaseTransform.wrap_data(dc1) == dc2
    assert slc.BaseTransform.wrap_data(img) == dc1


@pytest.mark.parametrize("return_torch", [False, True])
@pytest.mark.parametrize("trf", [t for t in all_trfs_solt if not issubclass(t, (slt.HSV, slt.RandomResizedCrop))])
def test_transforms_return_torch(img_3x3, trf, return_torch):
    if "p" in inspect.getfullargspec(trf.__init__):
        trf: slc.BaseTransform = trf(p=1)
    else:
        trf: slc.BaseTransform = trf()
    res = trf({"image": img_3x3}, return_torch=return_torch, as_dict=False, mean=(0.5,), std=(0.5,))

    assert isinstance(res, torch.FloatTensor) == return_torch


# Arguments, without which some transforms do nothing or cannot be created
_TRANSFORM_KWARGS = {
    slt.Crop: {"crop_to": 16},
    slt.CutOut: {"cutout_size": 8},
    slt.HSV: {"h_range": (10, 10)},
    slt.Pad: {"pad_to": 40},
    slt.Projection: {"v_range": (1e-4, 1e-4)},
    slt.RandomResizedCrop: {"resize_to": 16},
    slt.Resize: {"resize_to": 16},
}


def _make_transform(trf_cls):
    kwargs = dict(_TRANSFORM_KWARGS.get(trf_cls, {}))
    if class_accepts(trf_cls, "p"):
        kwargs["p"] = 1
    return trf_cls(**kwargs)


def _make_data(trf_cls):
    rng = np.random.default_rng(0)
    channels = 1 if trf_cls is slt.IntensityRemap else 3
    img = rng.integers(0, 255, (32, 32, channels), dtype=np.uint8)
    mask = (rng.random((32, 32)) > 0.5).astype(np.uint8)
    dmap = rng.random((32, 32)).astype(np.float32)
    kpts = slc.Keypoints(np.array([[5.0, 5.0], [20.0, 10.0]]), 32, 32)
    if trf_cls is slt.RandomResizedCrop:
        return (img, mask, dmap), "IMD"
    return (img, mask, dmap, kpts), "IMDP"


def _arrays(data):
    return [d.data if isinstance(d, slc.Keypoints) else d for d in data]


@pytest.mark.parametrize("trf_cls", all_trfs_solt, ids=lambda t: t.__name__)
def test_stream_not_in_place_does_not_modify_inputs(trf_cls):
    data, fmt = _make_data(trf_cls)
    originals = [a.copy() for a in _arrays(data)]

    res = slc.Stream([_make_transform(trf_cls)], in_place=False)(slc.DataContainer(data, fmt), return_torch=False)

    for item, original in zip(_arrays(data), originals):
        np.testing.assert_array_equal(item, original)
    for item, item_res in zip(_arrays(data), _arrays(res.data)):
        assert not np.shares_memory(item, item_res)


@pytest.mark.parametrize(
    "stream_cls, transforms",
    [
        (slc.Stream, [slt.Crop(16), slt.CutOut(8, p=1)]),
        (slc.Stream, [slt.Rotate(angle_range=(10, 10), p=0), slt.CutOut(8, p=1)]),
        (slc.Stream, [slc.Stream([slt.GridMask(d_range=(4, 8), p=1)])]),
        (slc.SelectiveStream, [slt.CutOut(8, p=1)]),
    ],
)
def test_stream_not_in_place_keeps_inputs_in_pipelines(stream_cls, transforms):
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    stream_cls(transforms, in_place=False)({"image": img}, return_torch=False)
    assert np.all(img == 100)


@pytest.mark.parametrize("stream_cls", [slc.Stream, slc.SelectiveStream])
def test_stream_is_in_place_by_default(stream_cls):
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    stream = stream_cls([slt.CutOut(8, p=1)])
    assert stream.in_place
    stream({"image": img}, return_torch=False)
    # CutOut modifies the image in place to avoid copying
    assert np.any(img == 0)


@pytest.mark.parametrize("stream_cls", [slc.Stream, slc.SelectiveStream])
def test_stream_in_place_serialization(stream_cls):
    stream = stream_cls([slt.Flip(p=1)], in_place=False)
    assert stream.to_dict()["in_place"] is False
    assert slu.from_yaml(slc.Stream([stream]).to_yaml()).transforms[0].in_place is False


def test_data_container_copy():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    kpts = slc.Keypoints(np.array([[1.0, 2.0]]), 4, 5)
    dc = slc.DataContainer((img, img[..., 0], kpts, 1), "IMPL", {0: {"interpolation": "bicubic"}})
    dc_copy = dc.copy()

    assert dc_copy == dc
    assert dc_copy.transform_settings == dc.transform_settings
    for item, item_copy in zip(_arrays(dc.data[:3]), _arrays(dc_copy.data[:3])):
        assert not np.shares_memory(item, item_copy)
