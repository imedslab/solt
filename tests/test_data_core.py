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
