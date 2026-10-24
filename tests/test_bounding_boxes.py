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
    # The box leaves the image in the nested stream and comes back, so it must be kept
    transforms = [
        slc.Stream([slt.Translate(range_x=(90, 90), range_y=(0, 0), p=1)]),
        slt.Translate(range_x=(-90, -90), range_y=(0, 0), p=1),
    ]
    res = _stream_boxes(transforms, [[20, 20, 40, 40]])
    np.testing.assert_allclose(res.data, [[20, 20, 40, 40]], atol=1e-6)


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
