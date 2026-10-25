import random

import numpy as np
import pytest
import torch

import solt
import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

H, W = 60, 80
# x_min, y_min, x_max, y_max: continuous pixel coordinates, the image spans [0, W] x [0, H]
BOXES = np.array([[20.0, 15.0, 45.0, 40.0], [50.0, 30.0, 62.0, 44.0]])


def boxes_mask(boxes, h=H, w=W):
    """Masks of the boxes: a box [x_min, x_max) covers the pixels x_min, ..., x_max - 1."""
    masks = np.zeros((h, w, len(boxes)), dtype=np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes.astype(int)):
        masks[y0:y1, x0:x1, i] = 1
    return masks


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


# ----------------------------------------------------------------------------------------------------------------------
# BoundingBoxes
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, boxes",
    [
        ("voc", [[20, 15, 45, 40]]),
        ("coco", [[20, 15, 25, 25]]),
        ("cxcywh", [[32.5, 27.5, 25, 25]]),
        ("yolo", [[32.5 / W, 27.5 / H, 25 / W, 25 / H]]),
    ],
)
def test_bounding_boxes_formats(fmt, boxes):
    bboxes = slc.BoundingBoxes(boxes, H, W, fmt=fmt)
    np.testing.assert_allclose(bboxes.data, [[20, 15, 45, 40]])
    np.testing.assert_allclose(bboxes.to(fmt), boxes)


@pytest.mark.parametrize("fmt", ["voc", "coco", "cxcywh", "yolo"])
def test_bounding_boxes_format_round_trip(fmt):
    bboxes = slc.BoundingBoxes(BOXES, H, W)
    np.testing.assert_allclose(slc.BoundingBoxes(bboxes.to(fmt), H, W, fmt=fmt).data, BOXES)


def test_bounding_boxes_properties():
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=[3, 7])
    assert len(bboxes) == 2
    assert (bboxes.height, bboxes.width) == (H, W)
    np.testing.assert_array_equal(bboxes.labels, [3, 7])
    np.testing.assert_allclose(bboxes.areas(), [625, 168])
    assert bboxes == slc.BoundingBoxes(BOXES.copy(), H, W, labels=[3, 7])
    assert bboxes != slc.BoundingBoxes(BOXES, H, W, labels=[3, 8])
    assert solt.BoundingBoxes is slc.BoundingBoxes


def test_bounding_boxes_empty():
    bboxes = slc.BoundingBoxes(np.zeros((0, 4)), H, W, labels=[])
    assert len(bboxes) == 0
    assert bboxes.data.shape == (0, 4)
    res = slt.Rotate(angle_range=(30, 30), p=1)(slc.DataContainer((np.zeros((H, W, 3), np.uint8), bboxes), "IB"))
    assert len(res.data[1]) == 0


@pytest.mark.parametrize(
    "boxes, kwargs, error",
    [
        ([[10, 10, 5, 20]], {}, ValueError),
        ([[10, 10, 20]], {}, ValueError),
        ([[10, 10, 20, 20]], {"fmt": "xyxy"}, ValueError),
        ([[10, 10, 20, 20]], {"labels": [1, 2]}, ValueError),
        ([[10, 10, 20, 20]], {"height": 0}, ValueError),
        ([[10, 10, -1, 20]], {"fmt": "coco"}, ValueError),
        ([[10, 10, 20, 20]], {"unclipped": [[0, 0, 30, 30], [0, 0, 1, 1]]}, ValueError),
    ],
)
def test_bounding_boxes_wrong_input(boxes, kwargs, error):
    args = {"height": H, "width": W, **kwargs}
    with pytest.raises(error):
        slc.BoundingBoxes(boxes, **args)


# ----------------------------------------------------------------------------------------------------------------------
# DataContainer
# ----------------------------------------------------------------------------------------------------------------------


def test_data_container_with_bounding_boxes():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=[1, 2])
    dc = slc.DataContainer((img, bboxes), "IB")
    assert dc.data[1] is bboxes

    with pytest.raises(ValueError):
        slc.DataContainer((img, slc.BoundingBoxes(BOXES, H + 1, W)), "IB")
    with pytest.raises(TypeError):
        slc.DataContainer((img, BOXES), "IB")
    with pytest.raises(TypeError):
        slc.DataContainer((img, bboxes), "IB", {1: {"interpolation": "nearest"}})


def test_data_container_from_dict_with_bounding_boxes():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    bboxes = slc.BoundingBoxes(BOXES, H, W)
    kpts = slc.Keypoints(np.array([[1.0, 2.0]]), H, W)
    dc = slc.DataContainer.from_dict({"label": 1, "bboxes": bboxes, "keypoints": kpts, "image": img})
    assert dc.data_format == "IPBL"
    dc = slc.DataContainer.from_dict({"image": img, "bboxes_array": [bboxes, bboxes]})
    assert dc.data_format == "IBB"


def test_bounding_boxes_to_torch():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    res = slc.DataContainer((img, slc.BoundingBoxes(BOXES, H, W, labels=[1, 2])), "IB").to_torch(as_dict=True)
    target = res["bboxes"]
    assert target["boxes"].dtype == torch.float32
    np.testing.assert_allclose(target["boxes"].numpy(), BOXES)
    assert target["labels"].dtype == torch.int64
    assert target["labels"].tolist() == [1, 2]

    no_labels = slc.DataContainer((img, slc.BoundingBoxes(BOXES, H, W)), "IB").to_torch(as_dict=True)["bboxes"]
    assert set(no_labels) == {"boxes"}

    names = slc.DataContainer((img, slc.BoundingBoxes(BOXES, H, W, labels=["cat", "dog"])), "IB").to_torch(
        as_dict=True
    )["bboxes"]
    assert names["labels"] == ["cat", "dog"]

    several = slc.DataContainer((img, slc.BoundingBoxes(BOXES, H, W), slc.BoundingBoxes(BOXES, H, W)), "IBB")
    assert len(several.to_torch(as_dict=True)["bboxes_array"]) == 2


def test_data_container_copy_with_bounding_boxes():
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=[1, 2])
    dc = slc.DataContainer((bboxes,), "B")
    dc_copy = dc.copy()
    assert dc_copy == dc
    assert not np.shares_memory(dc_copy.data[0].data, bboxes.data)


# ----------------------------------------------------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------------------------------------------------


# Transforms and the tolerances: the masks have pixel steps, so the boxes are compared up to a pixel at most
EXACT, PIXEL_STEPS = 1e-6, 0.75
GEOMETRIC_TRANSFORMS = [
    (slt.Flip(p=1, axis=0), EXACT),
    (slt.Flip(p=1, axis=1), EXACT),
    (slt.Flip(p=1, axis=-1), EXACT),
    (slt.Rotate90(k=1, p=1), EXACT),
    (slt.Rotate90(k=2, p=1), EXACT),
    (slt.Rotate90(k=3, p=1), EXACT),
    (slt.Translate(range_x=(7, 7), range_y=(-4, -4), p=1), EXACT),
    (slt.Crop(crop_to=(64, 50), crop_mode="rb"), EXACT),
    (slt.Pad(pad_to=(100, 90)), EXACT),
    (slt.Resize(resize_to=(160, 120)), EXACT),
    (slt.Resize(resize_to=(40, 30)), PIXEL_STEPS),
    (slt.Scale(range_x=(1.2, 1.2), p=1), PIXEL_STEPS),
    (slt.Scale(range_x=(0.8, 0.8), p=1), PIXEL_STEPS),
    (slt.Rotate(angle_range=(20, 20), p=1), 1.0),
    (slt.Shear(range_x=(0.2, 0.2), range_y=(0, 0), p=1), 1.0),
    (slt.Projection(v_range=(2e-4, 2e-4), p=1), 1.0),
    (slt.RandomResizedCrop(resize_to=(80, 60), scale=(1, 1), ratio=(W / H, W / H)), EXACT),
]


@pytest.mark.parametrize("trf, atol", GEOMETRIC_TRANSFORMS, ids=lambda t: type(t).__name__)
def test_bounding_boxes_follow_the_image(trf, atol):
    masks = boxes_mask(BOXES)
    masks_res, bboxes_res = trf(slc.DataContainer((masks, slc.BoundingBoxes(BOXES, H, W)), "MB")).data
    assert (bboxes_res.height, bboxes_res.width) == masks_res.shape[:2]
    for i in range(len(BOXES)):
        np.testing.assert_allclose(bboxes_res.data[i], mask_box(masks_res[..., i]), atol=atol)


@pytest.mark.parametrize("trf", [slt.Brightness(brightness_range=(10, 10), p=1), slt.Noise(p=1), slt.CutOut(4, p=1)])
def test_bounding_boxes_unchanged_by_intensity_transforms(trf):
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=[1, 2])
    res = trf(slc.DataContainer((np.zeros((H, W, 3), np.uint8), bboxes), "IB")).data[1]
    assert res == bboxes


def test_rotation_gives_the_enclosing_box():
    box = np.array([[30.0, 20.0, 50.0, 40.0]])  # A square around the center (40, 30)
    trf = slt.Rotate(angle_range=(45, 45), p=1)
    res = trf(slc.DataContainer((slc.BoundingBoxes(box, H, W),), "B")).data[0]
    half = 10 * np.sqrt(2)
    np.testing.assert_allclose(res.data[0], [40 - half, 30 - half, 40 + half, 30 + half], atol=1)


def test_transforms_keep_labels():
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=["cat", "dog"])
    res = slt.Flip(p=1)(slc.DataContainer((bboxes,), "B")).data[0]
    assert list(res.labels) == ["cat", "dog"]


# ----------------------------------------------------------------------------------------------------------------------
# Clipping and dropping in Stream
# ----------------------------------------------------------------------------------------------------------------------


def _stream_boxes(transforms, boxes, labels=None, h=100, w=100, **kwargs):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    res = slc.Stream(transforms, **kwargs)(
        {"image": img, "bboxes": slc.BoundingBoxes(boxes, h, w, labels=labels)}, return_torch=False
    )
    return res.data[1]


def test_stream_clips_and_drops_boxes_outside_of_the_image():
    boxes = [[10, 10, 30, 30], [15, 15, 35, 35], [60, 60, 80, 80]]
    # The crop starts at (20, 20): 75% of the first box and 44% of the second one are outside
    res = _stream_boxes([slt.Crop(crop_to=80, crop_mode="rb")], boxes, labels=["a", "b", "c"])
    assert list(res.labels) == ["b", "c"]
    np.testing.assert_allclose(res.data, [[0, 0, 15, 15], [40, 40, 60, 60]])


# Corner cases of the area loss, which is measured relative to the image: (transforms, image size, box, kept)
_AREA_LOSS_CASES = {
    "downscale_4x": (lambda: [slt.Resize(resize_to=(25, 25))], 100, [20, 20, 60, 60], True),
    "downscale_8x": (lambda: [slt.Resize(resize_to=(100, 100))], 800, [100, 100, 400, 400], True),
    "downscale_800_to_200": (lambda: [slt.Resize(resize_to=(200, 200))], 800, [100, 100, 400, 400], True),
    "whole_image_random_resized_crop": (
        lambda: [slt.RandomResizedCrop(resize_to=224, scale=(1, 1), ratio=(1, 1))],
        800,
        [100, 100, 400, 400],
        True,
    ),
    "non_uniform_resize": (lambda: [slt.Resize(resize_to=(50, 200))], 100, [20, 20, 60, 60], True),
    "scale_down_4x_in_the_same_frame": (lambda: [slt.Scale(range_x=(0.25, 0.25), p=1)], 100, [30, 30, 70, 70], False),
    "scale_down_2x_in_the_same_frame": (lambda: [slt.Scale(range_x=(0.5, 0.5), p=1)], 100, [30, 30, 70, 70], True),
    "zoom_crop": (lambda: [slt.Crop(crop_to=50, crop_mode="c")], 100, [40, 40, 60, 60], True),
    "pad_to_4x_area": (lambda: [slt.Pad(pad_to=200)], 100, [20, 20, 60, 60], True),
    "pad_to_11x_area": (lambda: [slt.Pad(pad_to=332)], 100, [20, 20, 60, 60], False),
}


@pytest.fixture(params=list(_AREA_LOSS_CASES), ids=list(_AREA_LOSS_CASES))
def area_loss_case(request):
    return _AREA_LOSS_CASES[request.param]


def test_stream_box_area_loss_is_relative_to_the_image(area_loss_case):
    make_transforms, size, box, kept = area_loss_case
    res = _stream_boxes(make_transforms(), [box], h=size, w=size)
    assert len(res) == int(kept)


def test_stream_box_thresholds():
    boxes = [[10, 10, 30, 30]]
    res = _stream_boxes([slt.Crop(crop_to=80, crop_mode="rb")], boxes, max_bbox_outside=0.8)
    assert len(res) == 1
    # Scaling by 0.25 in the same frame loses 93.75% of the share of the image
    res = _stream_boxes([slt.Scale(range_x=(0.25, 0.25), p=1)], [[30, 30, 70, 70]], max_bbox_area_loss=0.95)
    assert len(res) == 1
    res = _stream_boxes([slt.Crop(crop_to=80, crop_mode="rb")], boxes, max_bbox_outside=None, max_bbox_area_loss=None)
    # Without thresholds, the boxes are only clipped
    np.testing.assert_allclose(res.data, [[0, 0, 10, 10]])


def test_stream_drops_boxes_fully_outside():
    res = _stream_boxes([slt.Translate(range_x=(90, 90), range_y=(0, 0), p=1)], [[20, 20, 40, 40]])
    assert len(res) == 0


def test_stream_filters_boxes_only_at_the_end():
    # The box becomes tiny in the nested stream and grows back, so it must be kept
    transforms = [
        slc.Stream([slt.Scale(range_x=(0.25, 0.25), p=1)]),
        slt.Scale(range_x=(4, 4), p=1),
    ]
    res = _stream_boxes(transforms, [[40, 40, 60, 60]])
    np.testing.assert_allclose(res.data, [[40, 40, 60, 60]], atol=1)


# Corner cases of clipping after every transform: (transforms, box, expected boxes after the stream)
_CLIPPING_CASES = {
    "crop_then_pad": (
        lambda: [slt.Crop(crop_to=100, crop_mode="c"), slt.Pad(pad_to=200)],
        [0, 0, 200, 200],
        [[50, 50, 150, 150]],
    ),
    "crop_then_translate_back": (
        lambda: [
            slt.Crop(crop_to=150, crop_mode="rb"),
            slt.Translate(range_x=(-50, -50), range_y=(0, 0), p=1),
        ],
        [20, 60, 120, 140],
        # The crop starts at (50, 50): [-30, 10, 70, 90] -> [0, 10, 70, 90], then moved by -50 in x
        [[0, 10, 20, 90]],
    ),
    "box_leaves_the_image_and_comes_back": (
        lambda: [
            slt.Translate(range_x=(190, 190), range_y=(0, 0), p=1),
            slt.Translate(range_x=(-190, -190), range_y=(0, 0), p=1),
        ],
        [20, 20, 40, 40],
        # The content of the box is lost, so the box is dropped
        [],
    ),
}


@pytest.fixture(params=list(_CLIPPING_CASES), ids=list(_CLIPPING_CASES))
def clipping_case(request):
    return _CLIPPING_CASES[request.param]


def test_boxes_are_clipped_after_every_transform(clipping_case):
    make_transforms, box, expected = clipping_case
    res = _stream_boxes(make_transforms(), [box], h=200, w=200, max_bbox_outside=None, max_bbox_area_loss=None)
    np.testing.assert_allclose(res.data.reshape(-1, 4), np.array(expected, dtype=float).reshape(-1, 4), atol=1e-6)


def test_crop_then_rotate_rotates_the_visible_part():
    # The crop keeps the left half of the box: [40, 80, 100, 120] -> [40, 80, 60, 120] in the frame [0, 60)
    box_in_crop = np.array([[40.0, 80.0, 60.0, 120.0]])
    rotate = lambda: slt.Rotate(angle_range=(90, 90), p=1)  # noqa: E731
    res = _stream_boxes(
        [slt.Crop(crop_to=(60, 200), crop_mode="lt"), rotate()],
        [[40, 80, 100, 120]],
        h=200,
        w=200,
        max_bbox_outside=None,
        max_bbox_area_loss=None,
    )
    expected = rotate()(slc.DataContainer((slc.BoundingBoxes(box_in_crop, 200, 60),), "B")).data[0]
    np.testing.assert_allclose(res.data, expected.data, atol=1e-6)


@pytest.mark.parametrize("trf", [t for t, _ in GEOMETRIC_TRANSFORMS], ids=lambda t: type(t).__name__)
def test_every_transform_returns_boxes_inside_the_image(trf):
    boxes = np.array([[-10.0, -5.0, 30.0, 20.0], [60.0, 40.0, 95.0, 70.0], [20.0, 15.0, 45.0, 40.0]])
    res = trf(slc.DataContainer((slc.BoundingBoxes(boxes, H, W),), "B")).data[0]
    assert np.all(res.data >= 0)
    assert np.all(res.data[:, [0, 2]] <= res.width) and np.all(res.data[:, [1, 3]] <= res.height)


@pytest.mark.parametrize(
    "box, kept",
    [([-30, 0, 30, 20], True), ([-40, 0, 20, 20], False)],
    ids=["half_outside", "two_thirds_outside"],
)
def test_boxes_outside_on_entry_count_as_outside(box, kept):
    bboxes = slc.BoundingBoxes([box], 100, 100)
    dc = slc.DataContainer((bboxes,), "B")
    assert dc.data[0].data[0, 0] == 0
    res = slc.Stream([slt.Flip(p=0)])(dc, return_torch=False).data[0]
    assert len(res) == int(kept)


def test_outside_fraction_accumulates_over_transforms():
    # The first crop starts at x=30 and cuts 30% of the box [0, 100], the second one starts at x=21 and cuts 30% of
    # the remaining [0, 70], so 49% of the box stays visible
    transforms = [slt.Crop(crop_to=(170, 200), crop_mode="rt"), slt.Crop(crop_to=(149, 200), crop_mode="rt")]
    res = _stream_boxes(transforms, [[0, 0, 100, 100]], h=200, w=200)
    assert len(res) == 0
    res = _stream_boxes(transforms, [[0, 0, 100, 100]], h=200, w=200, max_bbox_outside=0.55)
    assert len(res) == 1


def test_data_container_does_not_copy_boxes_inside_the_image():
    bboxes = slc.BoundingBoxes(BOXES, H, W)
    assert slc.DataContainer((bboxes,), "B").data[0] is bboxes


def test_selective_stream_filters_boxes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    stream = slc.SelectiveStream([slt.Crop(crop_to=80, crop_mode="rb")])
    res = stream({"image": img, "bboxes": slc.BoundingBoxes([[10, 10, 30, 30]], 100, 100)}, return_torch=False)
    assert len(res.data[1]) == 0


def test_stream_box_results_are_inside_the_image():
    random.seed(0)
    stream = slc.Stream(
        [slt.Rotate(angle_range=(-45, 45), p=1), slt.Scale(range_x=(0.7, 1.4), p=1), slt.Crop(64, crop_mode="r")]
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        xy = rng.uniform(0, 70, (10, 2))
        boxes = np.hstack([xy, xy + rng.uniform(5, 30, (10, 2))])
        res = stream(
            {"image": np.zeros((100, 100, 3), np.uint8), "bboxes": slc.BoundingBoxes(boxes, 100, 100)},
            return_torch=False,
        ).data[1]
        assert (res.height, res.width) == (64, 64)
        assert np.all(res.data >= 0) and np.all(res.data[:, [0, 2]] <= 64) and np.all(res.data[:, [1, 3]] <= 64)


@pytest.mark.parametrize("stream_cls", [slc.Stream, slc.SelectiveStream])
def test_stream_box_thresholds_serialization(stream_cls):
    stream = stream_cls([slt.Flip()], max_bbox_area_loss=0.8, max_bbox_outside=None)
    restored = slu.from_yaml(slc.Stream([stream]).to_yaml()).transforms[0]
    assert restored.max_bbox_area_loss == 0.8
    assert restored.max_bbox_outside is None


@pytest.mark.parametrize("kwargs", [{"max_bbox_area_loss": 1.5}, {"max_bbox_outside": -0.1}, {"max_bbox_outside": "a"}])
def test_stream_wrong_box_thresholds(kwargs):
    with pytest.raises((ValueError, TypeError)):
        slc.Stream([], **kwargs)


def test_stream_boxes_to_torch():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = slc.BoundingBoxes([[10, 10, 30, 30], [60, 60, 80, 80]], 100, 100, labels=[1, 2])
    res = slc.Stream([slt.Crop(crop_to=80, crop_mode="rb")])({"image": img, "bboxes": bboxes}, normalize=False)
    assert res["bboxes"]["labels"].tolist() == [2]
    np.testing.assert_allclose(res["bboxes"]["boxes"].numpy(), [[40, 40, 60, 60]])


def test_bounding_boxes_float_labels_to_torch():
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=[0.5, 1.5])
    target = slc.DataContainer((bboxes,), "B").to_torch(as_dict=True)["bboxes"]
    assert target["labels"].dtype == torch.float64
    assert target["labels"].tolist() == [0.5, 1.5]


def test_bounding_boxes_to_unknown_format():
    with pytest.raises(ValueError):
        slc.BoundingBoxes(BOXES, H, W).to("xywh")


def test_bounding_boxes_comparison():
    bboxes = slc.BoundingBoxes(BOXES, H, W, labels=[1, 2])
    assert bboxes != BOXES
    assert bboxes != slc.BoundingBoxes(BOXES, H, W + 1, labels=[1, 2])
    assert bboxes != slc.BoundingBoxes(BOXES, H, W)
    assert slc.BoundingBoxes(BOXES, H, W) == slc.BoundingBoxes(BOXES, H, W)


@pytest.fixture(
    params=[
        ([], torch.int64),
        ((), torch.int64),
        (np.array([], dtype=np.int32), torch.int64),
        (np.array([], dtype=np.uint8), torch.int64),
        (np.array([], dtype=str), list),
    ],
    ids=["list", "tuple", "int32_array", "uint8_array", "str_array"],
)
def empty_labels(request):
    return request.param


def test_bounding_boxes_without_objects_to_torch(empty_labels):
    labels, expected_type = empty_labels
    bboxes = slc.BoundingBoxes(np.zeros((0, 4)), H, W, labels=labels)
    target = slc.DataContainer((bboxes,), "B").to_torch(as_dict=True)["bboxes"]
    assert target["boxes"].shape == (0, 4)
    if expected_type is list:
        assert target["labels"] == []
    else:
        assert target["labels"].dtype == expected_type
        assert target["labels"].shape == (0,)


def test_all_boxes_dropped_keep_integer_labels():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = slc.BoundingBoxes([[10, 10, 30, 30]], 100, 100, labels=[3])
    res = slc.Stream([slt.Crop(crop_to=50, crop_mode="rb")])({"image": img, "bboxes": bboxes}, normalize=False)
    assert res["bboxes"]["boxes"].shape == (0, 4)
    assert res["bboxes"]["labels"].dtype == torch.int64


def test_copy_keeps_the_unclipped_boxes():
    # 2/3 of the box is outside, also after the stream copies the data container
    dc = slc.DataContainer((slc.BoundingBoxes([[-40, 0, 20, 20]], 100, 100),), "B")
    assert len(slc.Stream([slt.Flip(p=0)], in_place=False)(dc, return_torch=False).data[0]) == 0
    np.testing.assert_array_equal(dc.copy().data[0].unclipped, [[-40, 0, 20, 20]])


@pytest.fixture(
    params=[((0, 0), (0, 0)), ((0.05, 0.1), (0.05, 0.1)), ((-0.1, -0.05), (-0.1, -0.05)), ((-0.1, 0.1), (-0.1, 0.1))],
    ids=["no_jitter", "positive", "negative", "both"],
)
def jitter_ranges(request):
    return request.param


@pytest.fixture(
    params=[
        [[0, 0, W, H]],
        [[50, 30, W, H], [0, 0, 20, 10]],
        [[20, 15, 45, 40]],
        np.zeros((0, 4)),
    ],
    ids=["whole_image", "touching_right_and_bottom", "inside", "no_boxes"],
)
def jitter_boxes(request):
    return request.param


def test_keypoints_jitter_does_not_change_boxes(jitter_ranges, jitter_boxes):
    dx_range, dy_range = jitter_ranges
    bboxes = slc.BoundingBoxes(jitter_boxes, H, W, labels=list(range(len(jitter_boxes))))
    kpts = slc.Keypoints(np.array([[30.0, 20.0], [10.0, 40.0]]), H, W)
    random.seed(0)
    bboxes_res, kpts_res = slt.KeypointsJitter(p=1, dx_range=dx_range, dy_range=dy_range)(
        slc.DataContainer((bboxes, kpts), "BP")
    ).data
    assert bboxes_res == bboxes
    if dx_range != (0, 0):
        assert not np.array_equal(kpts_res.data, kpts.data)


# Corner cases of boxes, which end up fully outside of the image: (transforms, box, kept)
_FULLY_OUTSIDE_CASES = {
    "moved_out": (lambda: [slt.Translate(range_x=(250, 250), range_y=(0, 0), p=1)], [20, 20, 40, 40], False),
    "left_and_came_back": (
        lambda: [
            slt.Translate(range_x=(190, 190), range_y=(0, 0), p=1),
            slt.Translate(range_x=(-190, -190), range_y=(0, 0), p=1),
        ],
        [20, 20, 40, 40],
        False,
    ),
    "cropped_out": (lambda: [slt.Crop(crop_to=100, crop_mode="rb")], [10, 10, 50, 50], False),
    "point_inside": (lambda: [slt.Flip(p=1)], [50, 50, 50, 50], True),
    "point_outside": (lambda: [slt.Crop(crop_to=100, crop_mode="rb")], [50, 50, 50, 50], False),
    "touching_the_border": (lambda: [slt.Flip(p=1)], [0, 0, 200, 30], True),
}


@pytest.fixture(params=list(_FULLY_OUTSIDE_CASES), ids=list(_FULLY_OUTSIDE_CASES))
def fully_outside_case(request):
    return _FULLY_OUTSIDE_CASES[request.param]


@pytest.fixture(
    params=[{}, {"max_bbox_area_loss": None, "max_bbox_outside": None}], ids=["default_thresholds", "no_thresholds"]
)
def box_thresholds(request):
    return request.param


def test_boxes_fully_outside_are_always_dropped(fully_outside_case, box_thresholds):
    make_transforms, box, kept = fully_outside_case
    res = _stream_boxes(make_transforms(), [box], labels=[7], h=200, w=200, **box_thresholds)
    assert len(res) == int(kept)
    assert list(res.labels) == ([7] if kept else [])
