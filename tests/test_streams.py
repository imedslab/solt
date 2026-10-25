import random

import numpy as np
import pytest
import torch

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

from .fixtures import *


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


def test_stream_raises_assertion_error_when_not_basetransform_or_stream_in_the_transforms():
    with pytest.raises(TypeError):
        slc.Stream([1, 2, 3])


def test_value_error_when_optimizeing_wrong_elements_transforms_list():
    trfs = [slt.Rotate(angle_range=(90, 90), p=1), slt.Rotate(angle_range=(-90, -90), p=1), lambda x: x**2]

    with pytest.raises(TypeError):
        slc.Stream.optimize_transforms_stack(trfs)


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
