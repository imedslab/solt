import itertools
import random

import cv2
import numpy as np
import pytest
import torch

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

from .fixtures import *


def assert_data_containers_equal(dc, dc_new):
    assert dc_new.data_format == dc.data_format
    for d1, d2 in zip(dc_new.data, dc.data):
        if isinstance(d1, np.ndarray):
            np.testing.assert_array_equal(d1, d2)
        elif isinstance(d1, slc.Keypoints):
            assert d1.height == d2.height and d1.width == d2.width
            np.testing.assert_array_equal(d1.data, d2.data)
        else:
            assert d1 == d2


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


def test_stream_empty(img_2x2):
    img = img_2x2
    dc = slc.DataContainer((img,), "I")
    stream = slc.Stream()
    res, _, _ = stream(dc, return_torch=False)[0]
    assert np.all(res == img)


def test_empty_stream_selective():
    with pytest.raises(ValueError):
        slc.SelectiveStream()


def test_nested_stream(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream(
        [
            slt.Flip(p=1, axis=0),
            slt.Flip(p=1, axis=1),
            slc.Stream(
                [
                    slt.Flip(p=1, axis=1),
                    slt.Flip(p=1, axis=0),
                ]
            ),
        ]
    )

    dc = stream(dc, return_torch=False)
    img_res, t0, _ = dc[0]
    mask_res, t1, _ = dc[1]

    assert np.array_equal(img, img_res)
    assert np.array_equal(mask, mask_res)


def test_image_shape_equal_3_after_nested_flip(img_3x4):
    img = img_3x4
    dc = slc.DataContainer((img,), "I")

    stream = slc.Stream(
        [
            slt.Flip(p=1, axis=0),
            slt.Flip(p=1, axis=1),
            slc.Stream(
                [
                    slt.Flip(p=1, axis=1),
                    slt.Flip(p=1, axis=0),
                ]
            ),
        ]
    )

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]

    assert np.array_equal(len(img.shape), 3)


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


def test_fusion_happens(img_5x5):
    ppl = slc.Stream(
        [
            slt.Scale((0.5, 1.5), (0.5, 1.5), p=1),
            slt.Rotate((-50, 50), padding="z", p=1),
            slt.Shear((-0.5, 0.5), (-0.5, 0.5), padding="z", p=1),
        ]
    )
    dc = slc.DataContainer(img_5x5, "I")
    st = ppl.optimize_transforms_stack(ppl.transforms, dc)
    assert len(st) == 1


def test_fusion_rotate_360(img_5x5):
    img = img_5x5
    dc = slc.DataContainer(img, "I")

    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
            slt.Rotate((45, 45), padding="z", p=1),
        ],
        optimize_stack=True,
    )

    img_res = ppl(dc, return_torch=False)[0][0]

    np.testing.assert_array_almost_equal(img, img_res)


def test_fusion_rotate_360_flip_rotate_360(img_5x5):
    img = img_5x5
    dc = slc.DataContainer((img,), "I")

    ppl = slc.Stream(
        [
            slc.Stream(
                [
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                ],
                optimize_stack=True,
            ),
            slt.Flip(p=1, axis=1),
            slc.Stream(
                [
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                    slt.Rotate((45, 45), padding="z", p=1),
                ],
                optimize_stack=True,
            ),
        ]
    )

    img_res = ppl(dc, return_torch=False)[0][0]

    np.testing.assert_array_almost_equal(cv2.flip(img, 1).reshape(5, 5, 1), img_res)


def test_stream_settings():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Shear(0.1, 0.1, interpolation="bilinear", padding="z"),
        ],
        interpolation="nearest",
        padding="z",
    )

    for trf in ppl.transforms:
        assert trf.interpolation[0] == "nearest"
        assert trf.padding[0] == "z"


def test_stream_settings_replacement():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Shear(0.1, 0.1, interpolation="bilinear", padding="z"),
        ],
        interpolation="nearest",
        padding="z",
    )

    ppl.reset_interpolation("bilinear")
    ppl.reset_padding("r")

    for trf in ppl.transforms:
        assert trf.interpolation[0] == "bilinear"
        assert trf.padding[0] == "r"


def test_stream_settings_strict():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slt.Rotate((45, 45), interpolation=("bicubic", "strict"), padding=("r", "strict"), p=1),
            slt.Shear(0.1, 0.1, interpolation="bilinear", padding="z"),
        ],
        interpolation="nearest",
        padding="z",
    )

    for idx, trf in enumerate(ppl.transforms):
        if idx == 2:
            assert trf.interpolation[0] == "bicubic"
            assert trf.padding[0] == "r"
        else:
            assert trf.interpolation[0] == "nearest"
            assert trf.padding[0] == "z"


def test_stream_nested_settings():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slc.Stream(
                [
                    slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
                    slt.Rotate((45, 45), padding="r", p=1),
                ],
                interpolation="bicubic",
                padding="r",
            ),
        ],
        interpolation="nearest",
        padding="z",
    )

    trfs = ppl.transforms[:2] + ppl.transforms[-1].transforms

    for idx, trf in enumerate(trfs):
        assert trf.interpolation[0] == "nearest"
        assert trf.padding[0] == "z"


def test_stream_raises_assertion_error_when_not_basetransform_or_stream_in_the_transforms():
    with pytest.raises(TypeError):
        slc.Stream([1, 2, 3])


def test_selective_pipeline_selects_transforms_and_does_the_fusion():
    ppl = slc.SelectiveStream(
        [
            slt.Rotate(angle_range=(90, 90), p=1),
            slt.Rotate(angle_range=(-90, -90), p=1),
        ],
        n=2,
        probs=[0.5, 0.5],
        optimize_stack=True,
    )

    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 4)
    dc = slc.DataContainer(kpts, "P")
    dc_res = ppl(dc, return_torch=False)

    assert np.array_equal(np.eye(3), ppl.transforms[0].state_dict["transform_matrix"])


def test_value_error_when_optimizeing_wrong_elements_transforms_list():
    trfs = [slt.Rotate(angle_range=(90, 90), p=1), slt.Rotate(angle_range=(-90, -90), p=1), lambda x: x**2]

    with pytest.raises(TypeError):
        slc.Stream.optimize_transforms_stack(trfs)


def test_nested_streams_are_not_fused_with_matrix_trf(img_3x3_rgb):
    dc = slc.DataContainer(img_3x3_rgb, "I")

    trfs = [
        slt.Rotate(angle_range=(90, 90), p=1),
        slt.Rotate(angle_range=(-90, -90), p=1),
        slc.Stream(
            [
                slt.Rotate(angle_range=(90, 90), p=1),
            ]
        ),
        slt.Rotate(angle_range=(-90, -90), p=1),
    ]

    with pytest.raises(TypeError):
        slc.Stream.optimize_transforms_stack(trfs, dc)


def test_putting_wrong_format_in_data_container(img_2x2):
    with pytest.raises(TypeError):
        slc.DataContainer(img_2x2, "Q")


def test_wrong_transform_type_in_a_stream(img_2x2):
    dc = slc.DataContainer(img_2x2, "I")
    with pytest.raises(TypeError):
        slc.Stream.exec_stream([slt.Pad(4), lambda x: x**2], dc, False)


def test_selective_stream_too_many_probs():
    with pytest.raises(ValueError):
        slc.SelectiveStream(
            [
                slt.Rotate(angle_range=(90, 90), p=1),
                slt.Rotate(angle_range=(-90, -90), p=1),
            ],
            n=2,
            probs=[0.4, 0.3, 0.3],
        )


def test_selective_stream_low_prob_transform_should_not_change_the_data(img_5x5):
    img = img_5x5
    dc = slc.DataContainer((img,), "I")

    ppl = slc.SelectiveStream([slt.Rotate(angle_range=(90, 90), p=0), slt.Rotate(angle_range=(-90, -90), p=0)])

    dc_res = ppl(dc, return_torch=False)

    assert np.array_equal(dc.data, dc_res.data)


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


@pytest.mark.parametrize(
    "ppl",
    [
        slt.Flip(p=1),
        slt.Rotate(angle_range=(30, 30), p=1),
        slc.Stream([slt.Rotate(angle_range=(30, 30), p=1), slt.Flip(p=1), slt.Rotate(angle_range=(30, 30), p=1)]),
    ],
)
def test_transform_settings_are_kept_after_transforms(img_5x5, mask_5x5, ppl):
    dc = slc.DataContainer((img_5x5, mask_5x5), "IM", {0: {"padding": "r"}, 1: {"interpolation": "bilinear"}})
    dc_res = ppl(dc, return_torch=False) if isinstance(ppl, slc.Stream) else ppl(dc)
    assert dc_res.transform_settings == dc.transform_settings


def test_transform_settings_are_used_by_subsequent_transforms_in_stream():
    mask = np.zeros((9, 9), dtype=np.float32)
    mask[3:6, 3:6] = 1
    rotate = slt.Rotate(angle_range=(30, 30), p=1)
    settings = {0: {"interpolation": "bilinear"}}

    dc_res = slc.Stream([rotate, rotate])(slc.DataContainer((mask,), "M", settings), return_torch=False)
    expected = rotate(rotate(slc.DataContainer((mask,), "M", settings)))

    assert np.array_equal(dc_res.data[0], expected.data[0])
    # Bilinear interpolation produces values between 0 and 1, nearest would not
    assert np.any((dc_res.data[0] > 0) & (dc_res.data[0] < 1))


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


@pytest.mark.parametrize("ignore_state", [True, False])
@pytest.mark.parametrize("pipeline", [True, False])
def test_matrix_transforms_state_reset(img_5x5, ignore_state, pipeline):
    n_iter = 50
    if pipeline:
        ppl = slc.Stream(
            [
                slt.Rotate(angle_range=(-180, 180), p=1, ignore_state=ignore_state),
                slt.Pad(pad_to=(10, 10)),
            ]
        )
    else:
        ppl = slt.Rotate(angle_range=(-180, 180), p=1, ignore_state=ignore_state)

    img_test = img_5x5.copy()
    img_test[0, 0] = 1
    random.seed(42)

    trf_not_eq = 0
    imgs_not_eq = 0
    for i in range(n_iter):
        dc1 = slc.DataContainer((img_test.copy(),), "I")
        dc2 = slc.DataContainer((img_test.copy(),), "I")
        if pipeline:
            dc1_res = ppl(dc1, return_torch=False).data[0].squeeze()
        else:
            dc1_res = ppl(dc1).data[0].squeeze()
        if pipeline:
            trf_state1 = ppl.transforms[0].state_dict["transform_matrix_corrected"]
        else:
            trf_state1 = ppl.state_dict["transform_matrix_corrected"]
        if pipeline:
            dc2_res = ppl(dc2, return_torch=False).data[0].squeeze()
        else:
            dc2_res = ppl(dc2).data[0].squeeze()
        if pipeline:
            trf_state2 = ppl.transforms[0].state_dict["transform_matrix_corrected"]
        else:
            trf_state2 = ppl.state_dict["transform_matrix_corrected"]

        if not np.array_equal(trf_state1, trf_state2):
            trf_not_eq += 1

        if not np.array_equal(dc1_res, dc2_res):
            imgs_not_eq += 1

    random.seed(None)
    assert trf_not_eq > n_iter // 2
    assert imgs_not_eq > n_iter // 2


@pytest.mark.parametrize("pipeline", [True, False])
def test_matrix_transforms_use_cache_for_different_dc_items_raises_error(img_5x5, mask_3x4, pipeline):
    if pipeline:
        ppl = slc.Stream(
            [
                slt.Rotate(angle_range=(-180, 180), p=1, ignore_state=False),
                slt.Pad(pad_to=(10, 10)),
            ]
        )
    else:
        ppl = slt.Rotate(angle_range=(-180, 180), p=1, ignore_state=False)

    with pytest.raises(ValueError):
        if pipeline:
            ppl({"images": img_5x5, "mask": mask_3x4}, return_torch=False)
        else:
            ppl({"images": img_5x5, "mask": mask_3x4})


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


def test_image_mask_pipeline_to_torch(img_3x4, mask_3x4):
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(90, 90), p=1),
            slt.Rotate(angle_range=(90, 90), p=1),
        ],
    )
    img, mask = ppl({"image": img_3x4, "mask": mask_3x4}, normalize=False, as_dict=False)
    assert img.max().item() == 1
    assert mask.max().item() == 1
    assert isinstance(img, torch.FloatTensor)
    assert isinstance(mask, torch.FloatTensor)


def test_image_mask_pipeline_to_torch_uint16(img_3x4, mask_3x4):
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(90, 90), p=1),
            slt.Rotate(angle_range=(90, 90), p=1),
        ],
    )
    img, mask = ppl(
        {"image": (img_3x4 // 255).astype(np.uint16) * 65535, "mask": mask_3x4}, as_dict=False, normalize=False
    )
    assert img.max() == 1
    assert mask.max() == 1
    assert isinstance(img, torch.FloatTensor)
    assert isinstance(mask, torch.FloatTensor)


@pytest.mark.parametrize("stream_ignore_fast", [True, False])
@pytest.mark.parametrize("r1_ignore_fast", [True, False])
@pytest.mark.parametrize("r2_ignore_fast", [True, False])
def test_ignore_fast_mode_for_a_stream(stream_ignore_fast, r1_ignore_fast, r2_ignore_fast):
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(90, 90), p=1, ignore_fast_mode=r1_ignore_fast),
            slt.Rotate(angle_range=(70, 75), p=1, ignore_fast_mode=r2_ignore_fast),
        ],
        ignore_fast_mode=stream_ignore_fast,
    )

    # The setting of the stream always overrides the one of the transforms
    for trf in ppl.transforms:
        assert stream_ignore_fast == trf.ignore_fast_mode


@pytest.mark.parametrize(
    "mean,std",
    [
        [None, None],
        [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        [np.array((0.5, 0.5, 0.5)), (0.5, 0.5, 0.5)],
        [(0.5, 0.5, 0.5), np.array((0.5, 0.5, 0.5))],
        [np.array((0.5, 0.5, 0.5)), np.array((0.5, 0.5, 0.5))],
    ],
)
def test_image_mask_pipeline_to_torch_normalization(img_3x3_rgb, mask_3x3, mean, std):
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(90, 90), p=1),
            slt.Rotate(angle_range=(90, 90), p=1),
        ],
    )
    img, mask = ppl({"image": img_3x3_rgb, "mask": mask_3x3}, as_dict=False, mean=mean, std=std)

    if mean is None:
        np.testing.assert_almost_equal(img[0, :, :].max().item(), 0.515 / 0.229)
    else:
        assert img.max() == 1
    assert mask.max() == 1
    assert isinstance(img, torch.FloatTensor)
    assert isinstance(mask, torch.FloatTensor)


@pytest.mark.parametrize(
    "mean,std, expected",
    [
        [(0.5, 0.5), (0.5, 0.5, 0.5), ValueError],
        [(0.5, 0.5, 0.5), (0.5, 0.5), ValueError],
        [(0.5, 0.5, 0.5), "123", TypeError],
        ["123", (0.5, 0.5, 0.5), TypeError],
        [torch.tensor((0.5, 0.5, 0.5)).byte(), (0.5, 0.5, 0.5), TypeError],
        [torch.tensor((0.5, 0.5, 0.5)).byte(), torch.tensor((0.5, 0.5, 0.5)).double(), TypeError],
        [torch.tensor((0.5, 0.5, 0.5)), torch.tensor((0.5, 0.5, 0.5)).double(), TypeError],
        [torch.tensor((0.5, 0.5)), torch.tensor((0.5, 0.5, 0.5)), ValueError],
        [torch.tensor((0.5, 0.5, 0.5)), torch.tensor((0.5, 0.5)), ValueError],
    ],
)
def test_image_mask_pipeline_to_torch_checks_mean_type_and_shape_rgb(img_3x3_rgb, mask_3x3, mean, std, expected):
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(90, 90), p=1),
            slt.Rotate(angle_range=(90, 90), p=1),
        ],
    )
    dc_res = ppl({"image": img_3x3_rgb, "mask": mask_3x3}, return_torch=False)
    with pytest.raises(expected):
        dc_res.to_torch(normalize=True, mean=mean, std=std)


def test_data_container_keypoints_rescale_to_torch():
    kpts_data = np.array([[100, 20], [1023, 80], [20, 20], [100, 700]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 768, 1024)
    ppl = slc.Stream()
    k, label = ppl({"keypoints": kpts, "label": 1}, as_dict=False)
    assert isinstance(k, torch.FloatTensor)
    np.testing.assert_almost_equal(k.max().item() * 1023, 1023)
    np.testing.assert_almost_equal(k.min().item() * 1023, 20)
    assert label == 1


def test_reset_ignore_fast_mode_raises_error_for_streams_for_not_bool():
    ppl = slc.Stream()
    with pytest.raises(TypeError):
        ppl.reset_ignore_fast_mode("123")


def test_selective_stream_returns_torch_when_asked(img_5x5):
    img = img_5x5 * 255
    dc = slc.DataContainer((img,), "I")

    ppl = slc.SelectiveStream([slt.Rotate(angle_range=(90, 90), p=0), slt.Rotate(angle_range=(-90, -90), p=0)])

    res = ppl(dc, normalize=False)
    res_img = (res["image"] * 255).numpy().astype(np.uint8)
    assert isinstance(res, dict)
    assert tuple(res) == ("image",)
    # We can do squeeze here because it is a grayscale image!
    assert np.array_equal(img.squeeze(), res_img.squeeze())


class _OrderRecorder(slt.Brightness):
    """Brightness transform, which records the order in which the transforms are applied."""

    def __init__(self, name, log):
        super().__init__(p=1)
        self.name, self.log = name, log

    def sample_transform(self, data):
        self.log.append(self.name)
        return super().sample_transform(data)


def test_stream_shuffle_changes_the_order_of_transforms():
    log = []
    transforms = [_OrderRecorder(i, log) for i in range(4)]
    stream = slc.Stream(transforms, shuffle=True)
    img = np.zeros((4, 4, 1), dtype=np.uint8)

    orders = set()
    for seed in range(20):
        random.seed(seed)
        log.clear()
        stream({"image": img}, return_torch=False)
        assert sorted(log) == [0, 1, 2, 3]
        orders.add(tuple(log))

    assert len(orders) > 1
    # The order of the transforms in the stream itself is kept
    assert [trf.name for trf in stream.transforms] == [0, 1, 2, 3]


def test_stream_without_shuffle_keeps_the_order_of_transforms():
    log = []
    stream = slc.Stream([_OrderRecorder(i, log) for i in range(4)])
    for seed in range(5):
        random.seed(seed)
        log.clear()
        stream({"image": np.zeros((4, 4, 1), dtype=np.uint8)}, return_torch=False)
        assert log == [0, 1, 2, 3]


# Pairs of matrix transforms, whose order does not matter, so that SelectiveStream can apply them in any order.
# Factories, since the transforms keep their state.
_COMMUTING_MATRIX_TRANSFORMS = {
    "two_translations": lambda: [
        slt.Translate(range_x=(10, 10), range_y=(0, 0), p=1),
        slt.Translate(range_x=(0, 0), range_y=(-7, -7), p=1),
    ],
    "rotation_and_uniform_scaling": lambda: [
        slt.Rotate(angle_range=(30, 30), p=1),
        slt.Scale(range_x=(1.3, 1.3), p=1),
    ],
    "two_rotations": lambda: [slt.Rotate(angle_range=(30, 30), p=1), slt.Rotate(angle_range=(60, 60), p=1)],
}


@pytest.fixture(params=list(_COMMUTING_MATRIX_TRANSFORMS), ids=list(_COMMUTING_MATRIX_TRANSFORMS))
def commuting_matrix_transforms(request):
    return _COMMUTING_MATRIX_TRANSFORMS[request.param]


@pytest.fixture(params=[(41, 51), (40, 50)], ids=["odd_size", "even_size"])
def keypoints_frame(request):
    h, w = request.param
    return slc.Keypoints(np.array([[25.0, 10.0], [5.0, 30.0], [20.0, 20.0]]), h, w)


def test_selective_stream_optimize_stack_applies_all_selected_transforms(commuting_matrix_transforms, keypoints_frame):
    expected = slc.Stream(commuting_matrix_transforms(), optimize_stack=True)(
        slc.DataContainer((keypoints_frame,), "P"), return_torch=False
    ).data[0]
    for seed in range(5):
        random.seed(seed)
        stream = slc.SelectiveStream(commuting_matrix_transforms(), n=2, optimize_stack=True)
        res = stream(slc.DataContainer((keypoints_frame,), "P"), return_torch=False).data[0]
        np.testing.assert_allclose(res.data, expected.data, atol=1e-6)


class _StreamWithCustomCall(slc.Stream):
    """A user stream, which overrides __call__ (the way SelectiveStream used to be implemented)."""

    def __call__(self, data, return_torch=True, **kwargs):
        data = slc.BaseTransform.wrap_data(data)
        img = data.data[0]
        return slc.DataContainer((img + 1,), "I")


class _StreamWithCustomApply(slc.Stream):
    def _apply(self, data):
        return slc.DataContainer((data.data[0] + 1,), "I")


@pytest.fixture(
    params=[
        (lambda: _StreamWithCustomCall(), 1),
        (lambda: _StreamWithCustomApply(), 1),
        (lambda: slc.Stream([slt.Brightness(brightness_range=(1, 1), p=1)]), 1),
        (lambda: slc.SelectiveStream([slt.Brightness(brightness_range=(1, 1), p=1)]), 1),
    ],
    ids=["custom_call", "custom_apply", "stream", "selective_stream"],
)
def nested_stream(request):
    return request.param


def test_nested_streams_run_their_own_logic(nested_stream):
    make_stream, increment = nested_stream
    img = np.full((4, 4, 1), 10, dtype=np.uint8)
    res = slc.Stream([make_stream()])({"image": img}, return_torch=False).data[0]
    assert np.all(res == 10 + increment)


def _selective_stream_counts(n, probs=None, n_seeds=200):
    """Applies SelectiveStream of 3 transforms over many seeds, returns the set of numbers of the applied transforms."""
    log = []
    stream = slc.SelectiveStream([_OrderRecorder(i, log) for i in range(3)], n=n, probs=probs)
    counts = set()
    for seed in range(n_seeds):
        random.seed(seed)
        log.clear()
        stream({"image": np.zeros((4, 4, 1), dtype=np.uint8)}, return_torch=False)
        assert len(set(log)) == len(log)
        counts.add(len(log))
        if probs is not None:
            assert all(probs[i] > 0 for i in log)
    return counts


# n and the numbers of transforms, which SelectiveStream of 3 transforms must apply.
@pytest.fixture(
    params=[
        (2, {2}),
        ((1, 3), {1, 3}),
        ([1, 3], {1, 3}),
        ((2,), {2}),
        ((0, 2), {0, 2}),
        ((3,), {3}),
        ((0, 1, 2, 3), {0, 1, 2, 3}),
    ],
    ids=["int", "tuple", "list", "single_value_tuple", "tuple_with_0", "tuple_with_k", "tuple_full_range"],
)
def selective_stream_valid_n(request):
    return request.param


# Invalid n for SelectiveStream of 3 transforms and the expected error.
@pytest.fixture(
    params=[
        (4, ValueError),
        (-1, ValueError),
        ((1, 4), ValueError),
        ((-1, 1), ValueError),
        ((), ValueError),
        ([], ValueError),
        ((1.0, 2), TypeError),
        ((True, 2), TypeError),
        ([False], TypeError),
    ],
    ids=[
        "int_too_large",
        "int_negative",
        "tuple_too_large",
        "tuple_negative",
        "empty_tuple",
        "empty_list",
        "tuple_with_float",
        "tuple_with_true",
        "list_with_false",
    ],
)
def selective_stream_invalid_n(request):
    return request.param


def test_selective_stream_applies_only_and_all_listed_numbers_of_transforms(selective_stream_valid_n):
    n, expected_counts = selective_stream_valid_n
    assert _selective_stream_counts(n) == expected_counts


def test_selective_stream_converts_list_n_to_tuple():
    stream = slc.SelectiveStream([slt.Flip(), slt.Flip(), slt.Flip()], n=[1, 3])
    assert stream.n == (1, 3)


def test_selective_stream_raises_on_invalid_n(selective_stream_invalid_n):
    n, expected = selective_stream_invalid_n
    with pytest.raises(expected):
        slc.SelectiveStream([slt.Flip(), slt.Flip(), slt.Flip()], n=n)


def test_selective_stream_tuple_n_respects_probs():
    assert _selective_stream_counts((1, 2), probs=[0.5, 0.5, 0.0]) == {1, 2}


@pytest.mark.parametrize("n", [0, (0,)])
def test_selective_stream_n_0_leaves_data_unchanged(img_5x5, n):
    log = []
    stream = slc.SelectiveStream([_OrderRecorder(0, log), slt.Rotate(angle_range=(90, 90), p=1)], n=n)
    expected = img_5x5.copy()
    res = stream({"image": img_5x5}, return_torch=False)
    assert log == []
    np.testing.assert_array_equal(res.data[0], expected)


@pytest.mark.parametrize("n", [(1, 2), [1, 2]])
def test_selective_stream_tuple_n_serialization_round_trip(n):
    stream = slc.Stream([slc.SelectiveStream([slt.Flip(axis=0), slt.Flip(axis=1)], n=n)])
    restored = slu.from_yaml(stream.to_yaml())
    assert isinstance(restored.transforms[0], slc.SelectiveStream)
    assert restored.transforms[0].n == (1, 2)
    assert restored.to_dict() == stream.to_dict()


CV2_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "lanczos": cv2.INTER_LANCZOS4,
}
CV2_PADDINGS = {"z": cv2.BORDER_CONSTANT, "r": cv2.BORDER_REFLECT_101}
# Interpolation of a transform which differs from all the stream-wide modes below
OWN_INTERPOLATION = "lanczos"


def other_padding(padding):
    return "r" if padding == "z" else "z"


def expected_warp(img, trf, interpolation, padding):
    """Warps the image with the geometry sampled by ``trf``, using the explicitly given cv2 flags."""
    m = trf.state_dict["transform_matrix_corrected"]
    size = (trf.state_dict["w_new"], trf.state_dict["h_new"])
    res = cv2.warpAffine(img, m[:2, :], size, flags=CV2_INTERPOLATIONS[interpolation], borderMode=CV2_PADDINGS[padding])
    return res.reshape(res.shape[0], res.shape[1], -1) if img.ndim == 3 else res


def innermost(trf):
    while isinstance(trf, slc.Stream):
        trf = trf.transforms[0]
    return trf


def rotate_30(**kwargs):
    return slt.Rotate(angle_range=(30, 30), p=1, **kwargs)


@pytest.fixture
def img_16x16_rgb():
    return np.random.RandomState(42).randint(0, 256, size=(16, 16, 3)).astype(np.uint8)


@pytest.fixture
def mask_16x16():
    return np.random.RandomState(43).randint(0, 5, size=(16, 16)).astype(np.uint8)


@pytest.fixture(params=["nearest", "bilinear", "bicubic"])
def stream_interpolation(request):
    return request.param


@pytest.fixture(params=["z", "r"])
def stream_padding(request):
    return request.param


@pytest.fixture(params=["inherit", "strict"])
def inheritance(request):
    return request.param


@pytest.fixture(params=[0, 1, 2], ids=lambda x: f"depth{x}")
def nesting_depth(request):
    return request.param


@pytest.fixture(params=[slc.Stream, slc.SelectiveStream], ids=["Stream", "SelectiveStream"])
def nested_stream_cls(request):
    return request.param


@pytest.fixture
def wrap_in_streams(nesting_depth, nested_stream_cls):
    def wrap(trf):
        for _ in range(nesting_depth):
            trf = nested_stream_cls([trf])
        return trf

    return wrap


def test_stream_interpolation_propagates_unless_strict(
    img_16x16_rgb, stream_interpolation, inheritance, wrap_in_streams
):
    random.seed(42)
    trf = rotate_30(interpolation=(OWN_INTERPOLATION, inheritance))
    ppl = slc.Stream([wrap_in_streams(trf)], interpolation=stream_interpolation)

    expected_mode = OWN_INTERPOLATION if inheritance == "strict" else stream_interpolation
    other_mode = stream_interpolation if inheritance == "strict" else OWN_INTERPOLATION
    assert trf.interpolation == (expected_mode, inheritance)

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, expected_warp(img_16x16_rgb, trf, expected_mode, "z"))
    assert not np.array_equal(res, expected_warp(img_16x16_rgb, trf, other_mode, "z"))


def test_stream_padding_propagates_unless_strict(img_16x16_rgb, stream_padding, inheritance, wrap_in_streams):
    random.seed(42)
    own_padding = other_padding(stream_padding)
    trf = rotate_30(padding=(own_padding, inheritance))
    ppl = slc.Stream([wrap_in_streams(trf)], padding=stream_padding)

    expected_padding = own_padding if inheritance == "strict" else stream_padding
    assert trf.padding == (expected_padding, inheritance)

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, expected_warp(img_16x16_rgb, trf, "bilinear", expected_padding))
    assert not np.array_equal(res, expected_warp(img_16x16_rgb, trf, "bilinear", other_padding(expected_padding)))


def test_stream_settings_mix_strict_and_inherited_transforms_in_one_nested_stream(
    img_16x16_rgb, stream_interpolation, stream_padding, nested_stream_cls
):
    random.seed(42)
    strict_trf = rotate_30(
        interpolation=(OWN_INTERPOLATION, "strict"), padding=(other_padding(stream_padding), "strict")
    )
    inherit_trf = rotate_30(interpolation=OWN_INTERPOLATION, padding=other_padding(stream_padding))
    ppl = slc.Stream(
        [nested_stream_cls([strict_trf]), slc.Stream([inherit_trf])],
        interpolation=stream_interpolation,
        padding=stream_padding,
    )
    assert strict_trf.interpolation == (OWN_INTERPOLATION, "strict")
    assert strict_trf.padding == (other_padding(stream_padding), "strict")
    assert inherit_trf.interpolation == (stream_interpolation, "inherit")
    assert inherit_trf.padding == (stream_padding, "inherit")

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    expected = expected_warp(img_16x16_rgb, strict_trf, OWN_INTERPOLATION, other_padding(stream_padding))
    expected = expected_warp(expected, inherit_trf, stream_interpolation, stream_padding)
    np.testing.assert_array_equal(res, expected)


def test_stream_settings_do_not_affect_transforms_without_interpolation_and_padding(
    img_16x16_rgb, stream_interpolation, stream_padding
):
    random.seed(42)
    flip = slt.Flip(axis=1, p=1)
    brightness = slt.Brightness(brightness_range=(20, 20), p=1)
    trf = rotate_30()
    ppl = slc.Stream([flip, brightness, trf], interpolation=stream_interpolation, padding=stream_padding)

    for t in (flip, brightness):
        assert not hasattr(t, "interpolation")
        assert not hasattr(t, "padding")

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    flipped = slc.Stream([slt.Flip(axis=1, p=1), slt.Brightness(brightness_range=(20, 20), p=1)])(
        {"image": img_16x16_rgb}, return_torch=False
    ).data[0]
    np.testing.assert_array_equal(res, expected_warp(flipped, trf, stream_interpolation, stream_padding))


def test_stream_reset_interpolation_and_padding_after_construction(img_16x16_rgb, inheritance, wrap_in_streams):
    random.seed(42)
    trf = rotate_30(interpolation=(OWN_INTERPOLATION, inheritance), padding=("z", inheritance))
    ppl = slc.Stream([wrap_in_streams(trf)])
    assert ppl.interpolation is None and ppl.padding is None
    assert trf.interpolation == (OWN_INTERPOLATION, inheritance)

    ppl.reset_interpolation("nearest")
    ppl.reset_padding("r")
    # None means "do not change"
    ppl.reset_interpolation(None)
    ppl.reset_padding(None)
    assert ppl.interpolation == "nearest" and ppl.padding == "r"

    strict = inheritance == "strict"
    expected_mode, expected_padding = (OWN_INTERPOLATION, "z") if strict else ("nearest", "r")
    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, expected_warp(img_16x16_rgb, trf, expected_mode, expected_padding))


def test_data_container_item_settings_win_over_stream_settings(
    img_16x16_rgb, mask_16x16, stream_interpolation, stream_padding, inheritance, nested_stream_cls
):
    random.seed(42)
    item_interpolation = "bicubic" if stream_interpolation == "nearest" else "nearest"
    item_padding = other_padding(stream_padding)
    trf = rotate_30(interpolation=(OWN_INTERPOLATION, inheritance))
    ppl = slc.Stream([nested_stream_cls([trf])], interpolation=stream_interpolation, padding=stream_padding)

    dc = slc.DataContainer(
        (img_16x16_rgb, img_16x16_rgb, mask_16x16),
        "IIM",
        transform_settings={0: {"interpolation": item_interpolation, "padding": item_padding}},
    )
    res = ppl(dc, return_torch=False)

    trf_mode = OWN_INTERPOLATION if inheritance == "strict" else stream_interpolation
    # Item 0: user-given settings are strict
    np.testing.assert_array_equal(res.data[0], expected_warp(img_16x16_rgb, trf, item_interpolation, item_padding))
    # Item 1: no item settings, the transform settings are used
    np.testing.assert_array_equal(res.data[1], expected_warp(img_16x16_rgb, trf, trf_mode, stream_padding))
    # Item 2: masks are always interpolated with nearest neighbours
    np.testing.assert_array_equal(res.data[2], expected_warp(mask_16x16, trf, "nearest", stream_padding))


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_stream_settings_survive_serialization(img_16x16_rgb, mask_16x16, stream_interpolation, stream_padding, fmt):
    def make_leaves():
        return [
            rotate_30(interpolation=(OWN_INTERPOLATION, "strict"), padding=(other_padding(stream_padding), "strict")),
            rotate_30(interpolation=OWN_INTERPOLATION, padding=other_padding(stream_padding)),
            rotate_30(),
        ]

    a, b, c = make_leaves()
    ppl = slc.Stream(
        [a, slt.Flip(axis=1, p=1), slc.SelectiveStream([slc.Stream([b])]), slc.Stream([slc.SelectiveStream([c])])],
        interpolation=stream_interpolation,
        padding=stream_padding,
    )
    if fmt == "yaml":
        ppl_new = slu.from_yaml(ppl.to_yaml())
    else:
        ppl_new = slu.from_json(ppl.to_json())

    assert ppl_new.interpolation == stream_interpolation
    assert ppl_new.padding == stream_padding
    a_new, b_new, c_new = ppl_new.transforms[0], innermost(ppl_new.transforms[2]), innermost(ppl_new.transforms[3])
    for old, new in ((a, a_new), (b, b_new), (c, c_new)):
        assert new is not old
        assert tuple(new.interpolation) == old.interpolation
        assert tuple(new.padding) == old.padding
    assert a_new.interpolation == (OWN_INTERPOLATION, "strict")
    assert b_new.interpolation == (stream_interpolation, "inherit")
    assert c_new.padding == (stream_padding, "inherit")

    data = {"image": img_16x16_rgb, "mask": mask_16x16}
    random.seed(42)
    res = ppl(data, return_torch=False)
    random.seed(42)
    res_new = ppl_new(data, return_torch=False)
    assert_data_containers_equal(res, res_new)


@pytest.mark.parametrize("ignore_fast_mode", [False, True])
def test_stream_ignore_fast_mode_propagates_to_matrix_transforms(img_16x16_rgb, ignore_fast_mode):
    random.seed(42)
    trf = rotate_30(ignore_fast_mode=not ignore_fast_mode)
    flip = slt.Flip(axis=1, p=1)
    ppl = slc.Stream([flip, trf], ignore_fast_mode=ignore_fast_mode)
    assert trf.ignore_fast_mode is ignore_fast_mode
    assert not hasattr(flip, "ignore_fast_mode")

    def out_shape():
        return ppl({"image": img_16x16_rgb}, return_torch=False).data[0].shape

    # The full geometric transform enlarges the frame to fit the rotated image, the fast mode keeps it
    assert (out_shape() != img_16x16_rgb.shape) is ignore_fast_mode

    ppl.reset_ignore_fast_mode(not ignore_fast_mode)
    assert trf.ignore_fast_mode is not ignore_fast_mode
    assert (out_shape() != img_16x16_rgb.shape) is not ignore_fast_mode


@pytest.mark.parametrize("second_padding, expected_padding", [(None, "r"), ("z", "z")])
def test_fused_matrix_transforms_keep_padding_when_fused_one_has_none(img_16x16_rgb, second_padding, expected_padding):
    random.seed(42)
    first = rotate_30(padding=("r", "strict"))
    second = rotate_30(padding=("z", "strict"))
    second.padding = None if second_padding is None else (second_padding, "strict")
    ppl = slc.Stream([first, second], optimize_stack=True)

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]

    def rotate_60(padding):
        # Two fused rotations by 30 degrees around the centre are one rotation by 60 degrees
        trf = slt.Rotate(angle_range=(60, 60), p=1, padding=(padding, "strict"), ignore_fast_mode=True)
        return trf({"image": img_16x16_rgb}, return_torch=False).data[0]

    np.testing.assert_array_equal(res, rotate_60(expected_padding))
    assert not np.array_equal(res, rotate_60(other_padding(expected_padding)))
    # The transforms of the stream keep their own settings
    assert first.padding == ("r", "strict")


@pytest.fixture(
    params=[
        (("bicubic", "strict"), "z", "nearest"),
        ("bicubic", "z", "nearest"),
        (("bicubic", "strict"), ("r", "strict"), None),
        ("bilinear", "r", "nearest"),
    ],
    ids=["strict_interpolation", "inherited_interpolation", "strict_padding", "no_stream_setting"],
)
def optimized_stack_settings(request):
    return request.param


@pytest.fixture(params=["rotate", "rotate90"], ids=["Rotate_first", "Rotate90_first"])
def first_matrix_transform(request):
    return request.param


def test_optimized_stream_does_not_change_its_transforms(optimized_stack_settings, first_matrix_transform):
    interpolation, padding, stream_interpolation = optimized_stack_settings
    if first_matrix_transform == "rotate":
        first = slt.Rotate(angle_range=(20, 20), p=1, interpolation=interpolation, padding=padding)
    else:
        first = slt.Rotate90(k=1, p=1)
    second = slt.Translate(range_x=(3, 3), range_y=(0, 0), p=1, interpolation="nearest")
    stream = slc.Stream([first, second], interpolation=stream_interpolation, optimize_stack=True)
    before = [(trf.interpolation, trf.padding, trf.ignore_fast_mode) for trf in stream.transforms]

    img = np.random.default_rng(0).integers(0, 255, (21, 25, 3), dtype=np.uint8)
    results = [stream({"image": img}, return_torch=False).data[0] for _ in range(3)]

    assert [(trf.interpolation, trf.padding, trf.ignore_fast_mode) for trf in stream.transforms] == before
    assert stream.transforms == [first, second]
    for res in results[1:]:
        np.testing.assert_array_equal(res, results[0])


def test_optimized_transform_can_be_used_alone_afterwards():
    rotate = slt.Rotate(angle_range=(20, 20), p=1, interpolation=("bicubic", "strict"))
    img = np.random.default_rng(0).integers(0, 255, (21, 25, 3), dtype=np.uint8)
    expected = rotate(slc.DataContainer((img,), "I")).data[0]

    slc.Stream([rotate, slt.Translate(range_x=(3, 3), p=1)], interpolation="nearest", optimize_stack=True)(
        {"image": img}, return_torch=False
    )

    np.testing.assert_array_equal(rotate(slc.DataContainer((img,), "I")).data[0], expected)


def _rotate_fast(ignore_fast_mode=False):
    return slt.Rotate(angle_range=(30, 30), p=1, ignore_fast_mode=ignore_fast_mode)


# Pipelines with the rotation nested at different depths: factories taking the outer stream kwargs
_NESTED_FAST_MODE_PIPELINES = {
    "depth_0": lambda rotate, **kw: slc.Stream([rotate], **kw),
    "depth_1": lambda rotate, **kw: slc.Stream([slc.Stream([rotate])], **kw),
    "depth_2": lambda rotate, **kw: slc.Stream([slc.Stream([slc.Stream([rotate])])], **kw),
    "selective": lambda rotate, **kw: slc.Stream([slc.SelectiveStream([rotate])], **kw),
}


@pytest.fixture(params=list(_NESTED_FAST_MODE_PIPELINES), ids=list(_NESTED_FAST_MODE_PIPELINES))
def nested_fast_mode_pipeline(request):
    return _NESTED_FAST_MODE_PIPELINES[request.param]


def _frame_enlarged(stream, img):
    return stream({"image": img}, return_torch=False).data[0].shape != img.shape


def test_stream_ignore_fast_mode_reaches_nested_streams(nested_fast_mode_pipeline):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    rotate = _rotate_fast()
    stream = nested_fast_mode_pipeline(rotate, ignore_fast_mode=True)
    assert rotate.ignore_fast_mode is True
    assert _frame_enlarged(stream, img)


def test_stream_default_overrides_the_fast_mode_of_transforms(nested_fast_mode_pipeline):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    rotate = _rotate_fast(ignore_fast_mode=True)
    stream = nested_fast_mode_pipeline(rotate)
    assert rotate.ignore_fast_mode is False
    assert stream.ignore_fast_mode is False
    assert not _frame_enlarged(stream, img)


@pytest.mark.parametrize("outer, inner", [(False, True), (True, False)])
def test_outer_stream_overrides_the_fast_mode_of_nested_streams(outer, inner):
    rotate = _rotate_fast(ignore_fast_mode=inner)
    nested = slc.Stream([rotate], ignore_fast_mode=inner)
    slc.Stream([nested], ignore_fast_mode=outer)
    assert rotate.ignore_fast_mode is outer
    assert nested.ignore_fast_mode is outer


@pytest.mark.parametrize("value", [True, False])
def test_reset_ignore_fast_mode_is_applied_recursively(nested_fast_mode_pipeline, value):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    rotate = _rotate_fast(ignore_fast_mode=not value)
    stream = nested_fast_mode_pipeline(rotate, ignore_fast_mode=not value)
    stream.reset_ignore_fast_mode(value)
    assert rotate.ignore_fast_mode is value
    assert stream.ignore_fast_mode is value
    assert _frame_enlarged(stream, img) is value


_FUSION_H, _FUSION_W = 61, 81
_FUSION_POINTS = np.array([[20.0, 15.0], [60.0, 20.0], [40.0, 45.0], [10.0, 50.0], [45.0, 30.0]])

_AFFINE_TRANSFORMS = {
    "Rotate": lambda: slt.Rotate(angle_range=(25, 25), p=1),
    "Rotate90": lambda: slt.Rotate90(k=1, p=1),
    "Scale": lambda: slt.Scale(range_x=(1.2, 1.2), range_y=(0.9, 0.9), same=False, p=1),
    "Translate": lambda: slt.Translate(range_x=(5, 5), range_y=(-4, -4), p=1),
    "Shear": lambda: slt.Shear(range_x=(0.15, 0.15), range_y=(0, 0), p=1),
}


@pytest.fixture(params=list(_AFFINE_TRANSFORMS), ids=list(_AFFINE_TRANSFORMS))
def affine_transform(request):
    return _AFFINE_TRANSFORMS[request.param]


@pytest.fixture(params=[2e-4, 1e-3, -6e-4], ids=["weak", "strong", "negative"])
def perspective(request):
    return request.param


@pytest.fixture(params=["projection_last", "projection_middle", "projection_first"])
def projection_position(request):
    return request.param


def _blobs():
    img = np.zeros((_FUSION_H, _FUSION_W), dtype=np.float32)
    for x, y in _FUSION_POINTS:
        cv2.circle(img, (int(x), int(y)), 3, 1.0, -1)
    return img


def _blob_centroid(img, point, radius=6):
    x0, y0 = int(round(point[0])), int(round(point[1]))
    ys, xs = np.mgrid[: img.shape[0], : img.shape[1]]
    near = (np.abs(xs - x0) <= radius) & (np.abs(ys - y0) <= radius)
    weights = img * near
    assert weights.sum() > 0, f"no image content around the keypoint {point}"
    return np.array([(xs * weights).sum(), (ys * weights).sum()]) / weights.sum()


def test_fused_stream_with_projection_keeps_keypoints_on_the_image(affine_transform, perspective, projection_position):
    projection = slt.Projection(v_range=(perspective, perspective), p=1)
    transforms = {
        "projection_last": [affine_transform(), affine_transform(), projection],
        "projection_middle": [affine_transform(), projection, affine_transform()],
        "projection_first": [projection, affine_transform(), affine_transform()],
    }[projection_position]
    stream = slc.Stream(transforms, optimize_stack=True)
    dc = slc.DataContainer(
        (_blobs(), slc.Keypoints(_FUSION_POINTS.copy(), _FUSION_H, _FUSION_W)), "MP", {0: {"interpolation": "bilinear"}}
    )
    random.seed(0)
    img_res, kpts_res = stream(dc, return_torch=False).data
    for point in kpts_res.data:
        np.testing.assert_allclose(_blob_centroid(img_res, point), point, atol=0.15)
