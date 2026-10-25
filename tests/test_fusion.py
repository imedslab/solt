import random

import cv2
import numpy as np
import pytest

import solt.core as slc
import solt.transforms as slt

from .fixtures import *
from .utils import other_padding, rotate_30


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


@pytest.fixture(
    params=[
        lambda: slt.Flip(p=1),
        lambda: slt.Crop(crop_to=10),
        lambda: slt.Brightness(brightness_range=(5, 5), p=1),
        lambda: slc.Stream([slt.Rotate(angle_range=(10, 10), p=1)]),
    ],
    ids=["Flip", "Crop", "Brightness", "nested_Stream"],
)
def non_matrix_transform(request):
    return request.param


@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize("stream_cls", [slc.Stream, slc.SelectiveStream])
def test_fusion_rejects_non_matrix_transforms(non_matrix_transform, position, stream_cls):
    transforms = [slt.Rotate(angle_range=(10, 10), p=1)]
    transforms.insert(0 if position == "first" else 1, non_matrix_transform())
    kwargs = {"n": 2} if stream_cls is slc.SelectiveStream else {}
    stream = stream_cls(transforms, optimize_stack=True, **kwargs)
    with pytest.raises(TypeError):
        stream({"image": np.zeros((20, 20, 3), dtype=np.uint8)}, return_torch=False)
