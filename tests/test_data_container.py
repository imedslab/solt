import itertools

import numpy as np
import pytest
import torch

import solt.core as slc
import solt.utils as slu

from .fixtures import *
from .utils import assert_data_containers_equal


def generate_data_container_based_on_presence(img, mask, kpts_data, order, presence):
    kpts = slc.Keypoints(kpts_data.copy(), 3, 4)

    n_obj1, n_obj2, n_obj3, n_obj4, n_obj5, n_obj6, n_obj7, n_obj8 = presence
    dc_content = []
    dc_format = ""
    order = list(order)
    d = {}

    if n_obj1:
        d["image"] = img
        dc_content.append(img)
        dc_format += "I"
    else:
        del order[order.index("image")]
    if n_obj2:
        d["images"] = [img.copy() for _ in range(n_obj2)]
        dc_content.extend(d["images"])
        dc_format += "I" * n_obj2
    else:
        del order[order.index("images")]
    if n_obj3:
        d["mask"] = mask
        dc_content.append(mask)
        dc_format += "M"
    else:
        del order[order.index("mask")]
    if n_obj4:
        d["masks"] = [mask.copy() for i in range(n_obj4)]
        dc_content.extend(d["masks"])
        dc_format += "M" * n_obj4
    else:
        del order[order.index("masks")]
    if n_obj5:
        d["keypoints"] = slc.Keypoints(kpts_data.copy(), 3, 4)
        dc_content.append(kpts)
        dc_format += "P"
    else:
        del order[order.index("keypoints")]
    if n_obj6:
        d["keypoints_array"] = [slc.Keypoints(kpts_data.copy(), 3, 4) for _ in range(n_obj6)]
        dc_content.extend(d["keypoints_array"])
        dc_format += "P" * n_obj6
    else:
        del order[order.index("keypoints_array")]
    if n_obj7:
        d["label"] = 1
        dc_content.append(1)
        dc_format += "L"
    else:
        del order[order.index("label")]
    if n_obj8:
        d["labels"] = [1 for _ in range(n_obj8)]
        dc_content.extend([1 for _ in range(n_obj8)])
        dc_format += "L" * n_obj8
    else:
        del order[order.index("labels")]
    dc = slc.DataContainer(tuple(dc_content), dc_format)

    reordered_d = {k: d[k] for k in order}

    # This tests whether the creation from dict works as expected
    dc_reordered = slc.DataContainer.from_dict(reordered_d)

    return dc, dc_reordered


@pytest.mark.parametrize(
    "order",
    list(
        itertools.permutations(["image", "images", "mask", "masks", "keypoints", "keypoints_array", "label", "labels"])
    )[30:50],
)
@pytest.mark.parametrize(
    "presence",
    [
        [1, 2, 1, 2, 1, 0, 1, 2],
        [1, 0, 1, 2, 0, 2, 0, 3],
        [0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2],
        [0, 0, 1, 0, 1, 0, 1, 0],
    ],
)
def test_assert_data_containers_equal(img_3x4, mask_3x4, order, presence):
    img, mask = img_3x4, mask_3x4
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2)).astype(float)
    dc, dc_reordered = generate_data_container_based_on_presence(img, mask, kpts_data, order, presence)
    assert_data_containers_equal(dc, dc_reordered)
    assert dc == dc_reordered


def test_img_shape_checker_decorator_shape_check():
    img = np.random.rand(3, 4, 5, 6)
    func = slu.img_shape_checker(lambda x: x)
    with pytest.raises(ValueError):
        func(img)


def test_data_container_different_length_of_data_and_format(img_2x2):
    with pytest.raises(ValueError):
        slc.DataContainer((img_2x2,), "II")


def test_data_container_create_from_any_data(img_2x2):
    d = slc.DataContainer(img_2x2, "I")
    assert np.array_equal(img_2x2, d.data[0])
    assert d.data_format == "I"


def test_data_container_can_be_only_tuple_if_iterable_single(img_2x2):
    with pytest.raises(TypeError):
        slc.DataContainer(
            [
                img_2x2,
            ],
            "I",
        )


def test_data_container_can_be_only_tuple_if_iterable_multple(img_2x2):
    with pytest.raises(TypeError):
        slc.DataContainer([img_2x2, img_2x2], "II")


def test_data_item_create_img(img_2x2):
    img = img_2x2
    dc = slc.DataContainer((img,), "I")
    assert len(dc) == 1
    assert np.array_equal(img, dc[0][0])
    assert dc[0][1] == "I"


def test_create_empty_keypoints():
    kpts = slc.Keypoints()
    assert kpts.height is None
    assert kpts.width is None
    assert kpts.data is None


def test_create_4_keypoints():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)
    assert kpts.height == 3
    assert kpts.width == 4
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)
    kpts.height = 2
    kpts.width = 2

    assert kpts.height == 2
    assert kpts.width == 2
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_grid_and_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)

    kpts_data_new = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]).reshape((5, 2))
    kpts.height = 2
    kpts.width = 2
    kpts.pts = kpts_data_new

    assert kpts.height == 2
    assert kpts.width == 2
    assert np.array_equal(kpts_data_new, kpts.pts)


def test_putting_wrong_format_in_data_container(img_2x2):
    with pytest.raises(TypeError):
        slc.DataContainer(img_2x2, "Q")


def test_manually_specified_padding_and_interpolation(img_5x5, mask_5x5):
    dc = slc.DataContainer(
        (img_5x5, img_5x5, mask_5x5, mask_5x5, 1),
        "IIMML",
        {0: {"interpolation": "bicubic", "padding": "z"}, 2: {"interpolation": "bilinear"}, 3: {"padding": "r"}},
    )

    assert dc.transform_settings[0]["interpolation"] == ("bicubic", "strict")
    assert dc.transform_settings[1]["interpolation"] == ("bilinear", "inherit")
    assert dc.transform_settings[2]["interpolation"] == ("bilinear", "strict")
    assert dc.transform_settings[3]["interpolation"] == ("nearest", "strict")

    assert dc.transform_settings[0]["padding"] == ("z", "strict")
    assert dc.transform_settings[1]["padding"] == ("z", "inherit")
    assert dc.transform_settings[2]["padding"] == ("z", "inherit")
    assert dc.transform_settings[3]["padding"] == ("r", "strict")


def test_transform_settings_wrong_type(img_5x5):
    with pytest.raises(TypeError):
        slc.DataContainer((img_5x5, img_5x5, 1), "IIL", ())


def test_transform_settings_wrong_length(img_5x5):
    with pytest.raises(ValueError):
        slc.DataContainer((img_5x5, img_5x5, 1), "IIL", {1: {}, 2: {}, 3: {}, 4: {}})


def test_transform_settings_wrong_type_for_item(img_5x5):
    with pytest.raises(TypeError):
        slc.DataContainer((img_5x5, img_5x5, 1), "IIL", {1: 123, 0: None})


@pytest.mark.parametrize(
    "setting", [{"interpolation": "bilinear"}, {"interpolation": "bilinear", "padding": "z"}, {"padding": "z"}]
)
def test_interpolation_or_padding_settings_for_labels_or_keypoints(setting):
    kpts = slc.Keypoints(pts=np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2)), height=3, width=3)
    with pytest.raises(TypeError):
        slc.DataContainer(data=(kpts,), fmt="P", transform_settings={0: setting})


def test_transform_settings_are_not_modified_and_can_be_reused(img_5x5, mask_5x5):
    settings = {1: {"interpolation": "bilinear"}}
    for _ in range(2):
        dc = slc.DataContainer((img_5x5, mask_5x5), "IM", settings)
        assert dc.transform_settings[1]["interpolation"] == ("bilinear", "strict")
    assert settings == {1: {"interpolation": "bilinear"}}


def test_transform_settings_accept_already_parsed_settings(img_5x5, mask_5x5):
    dc = slc.DataContainer((img_5x5, mask_5x5), "IM", {0: {"padding": "r"}, 1: {"interpolation": "bilinear"}})
    dc_copy = slc.DataContainer(dc.data, dc.data_format, dc.transform_settings)
    assert dc_copy.transform_settings == dc.transform_settings


@pytest.mark.parametrize("dtype, scale", [(np.uint8, 255.0), (np.uint16, 65535.0)])
@pytest.mark.parametrize("normalize", [False, True])
def test_grayscale_image_without_channel_axis_to_torch(dtype, scale, normalize):
    img = np.random.randint(0, 255, (3, 4)).astype(dtype)
    res = slc.DataContainer((img,), "I").to_torch(normalize=normalize, mean=(0.5,), std=(0.5,))
    res_with_axis = slc.DataContainer((img[:, :, np.newaxis],), "I").to_torch(
        normalize=normalize, mean=(0.5,), std=(0.5,)
    )

    assert tuple(res.shape) == (1, 3, 4)
    assert isinstance(res, torch.FloatTensor)
    assert torch.equal(res, res_with_axis)
    if not normalize:
        assert np.allclose(res[0].numpy(), img / scale)


@pytest.mark.parametrize(
    "mask_shape, expected_shape",
    [
        ((3, 4), (1, 3, 4)),
        ((3, 4, 1), (1, 3, 4)),
        ((3, 4, 5), (5, 3, 4)),
        ((1, 4), (1, 1, 4)),
        ((3, 1), (1, 3, 1)),
    ],
)
def test_mask_to_torch_shape(mask_shape, expected_shape):
    mask = np.random.rand(*mask_shape).astype(np.float32)
    res = slc.DataContainer((mask,), "M").to_torch(as_dict=True)["mask"]
    assert tuple(res.shape) == expected_shape
    assert isinstance(res, torch.FloatTensor)
    expected = mask[np.newaxis] if mask.ndim == 2 else mask.transpose((2, 0, 1))
    assert np.array_equal(res.numpy(), expected)


def test_keypoints_get_set():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)

    assert np.array_equal(kpts[0], np.array([0, 0]))
    kpts[0] = np.array([2, 2])
    assert np.array_equal(kpts[0], np.array([2, 2]))

    with pytest.raises(TypeError):
        kpts[0] = [2, 2]


@pytest.mark.parametrize(
    "order",
    list(
        itertools.permutations(["image", "images", "mask", "masks", "keypoints", "keypoints_array", "label", "labels"])
    )[:20],
)
@pytest.mark.parametrize(
    "presence",
    [
        [1, 2, 1, 2, 1, 0, 1, 2],
        [1, 0, 1, 2, 0, 2, 0, 3],
        [0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2],
        [0, 0, 1, 0, 1, 0, 1, 0],
    ],
)
def test_data_container_from_and_to_dict(img_3x4, mask_3x4, order, presence):
    img, mask = img_3x4, mask_3x4
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2)).astype(float)
    dc, dc_reordered = generate_data_container_based_on_presence(img, mask, kpts_data, order, presence)
    assert dc == dc_reordered

    # Now we will also test whether conversion to dict and back works well.
    tensor_dict = dc_reordered.to_torch(as_dict=True, normalize=False, scale_keypoints=False)
    for k in tensor_dict:
        if isinstance(tensor_dict[k], (list, tuple)):
            tmp = []
            for el in tensor_dict[k]:
                tmp.append(el.numpy() if isinstance(el, torch.Tensor) else el)
                if "imag" in k:
                    tmp[-1] = (tmp[-1].transpose((1, 2, 0)) * 255).astype(np.uint8)
                if "mask" in k:
                    tmp[-1] = tmp[-1].astype(np.uint8).squeeze()
                if "keypoints" in k:
                    tmp[-1] = slc.Keypoints(tmp[-1], 3, 4)

            tensor_dict[k] = tmp
        else:
            el = tensor_dict[k]
            tensor_dict[k] = (el.numpy()).astype(np.uint8) if isinstance(el, torch.Tensor) else el
            if "imag" in k:
                tensor_dict[k] = (tensor_dict[k].transpose((1, 2, 0)) * 255).astype(np.uint8)
            if "mask" in k:
                tensor_dict[k] = tensor_dict[k].astype(np.uint8).squeeze()
            if "keypoints" in k:
                tensor_dict[k] = slc.Keypoints(tensor_dict[k], 3, 4)

    assert dc == slc.DataContainer.from_dict(tensor_dict)
