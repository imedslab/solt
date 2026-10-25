"""Annotations in the coordinates of an image: keypoints and bounding boxes."""

import numbers

import numpy as np


class Keypoints(object):
    """
    Keypoints class

    Parameters
    ----------
    pts : numpy.ndarray
        Key points as an numpy.ndarray in (x, y) format.
    height : int
        Height of the coordinate frame.
    width : int
        Width of the coordinate frame.
    """

    def __init__(self, pts=None, height=None, width=None):
        self.__data = pts
        self.__height = height
        self.__width = width

    @property
    def data(self):
        return self.__data

    def __getitem__(self, idx):
        return self.__data[idx, :]

    def __setitem__(self, idx, value):
        if not isinstance(value, np.ndarray):
            raise TypeError
        self.__data[idx, :] = value

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    @height.setter
    def height(self, value):
        self.__height = value

    @width.setter
    def width(self, value):
        self.__width = value

    def __eq__(self, other):
        dim_equal = (self.height == other.height) and (self.width == other.width)
        data_equal = np.array_equal(self.data, other.data)

        return dim_equal and data_equal


class BoundingBoxes(object):
    """
    Bounding boxes, optionally with a label for every box.

    The boxes are stored in the VOC format: ``x_min, y_min, x_max, y_max``, in continuous pixel coordinates,
    i.e. the image spans ``[0, width] x [0, height]``, and the pixel ``i`` covers ``[i, i + 1)``.
    The transforms move the corners of the boxes, and the result is the box enclosing the transformed corners.
    ``DataContainer`` clips the boxes to the image after every transform, since the content outside of the image is
    lost. At the end, ``Stream`` drops the boxes, which became too small or are mostly outside of the image (see
    ``Stream``), together with their labels.

    Parameters
    ----------
    boxes : numpy.ndarray or list
        Boxes ``Nx4`` in the format ``fmt``.
    height : int
        Height of the coordinate frame.
    width : int
        Width of the coordinate frame.
    fmt : str
        Format of ``boxes``: ``'voc'`` (``x_min, y_min, x_max, y_max``), ``'coco'`` (``x_min, y_min, width,
        height``), ``'cxcywh'`` (``x_center, y_center, width, height``) or ``'yolo'`` (``x_center, y_center, width,
        height``, as fractions of the image size).
    labels : numpy.ndarray or list or None
        Labels of the boxes, one per box.
    unclipped : numpy.ndarray or None
        The boxes ``Nx4`` in the VOC format before clipping to the image, if they differ from ``boxes``. SOLT tracks
        them to measure, which fraction of the boxes is outside of the image.

    """

    FORMATS = ("voc", "coco", "cxcywh", "yolo")

    def __init__(self, boxes=None, height=None, width=None, fmt="voc", labels=None, unclipped=None):
        if fmt not in self.FORMATS:
            raise ValueError(f"Unknown format of bounding boxes: {fmt}. Allowed: {self.FORMATS}")
        for size in (height, width):
            if not isinstance(size, numbers.Real) or size <= 0:
                raise ValueError("Height and width of bounding boxes must be positive numbers!")

        boxes = np.zeros((0, 4)) if boxes is None else np.array(boxes, dtype=float)
        if boxes.size == 0:
            boxes = boxes.reshape((0, 4))
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Bounding boxes must be an Nx4 array!")

        boxes = BoundingBoxes._to_voc(boxes, fmt, height, width)
        if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
            raise ValueError("Bounding boxes must have x_max >= x_min and y_max >= y_min!")

        if labels is not None:
            labels = np.asarray(labels)
            if labels.size == 0 and labels.dtype == np.float64:
                # An empty list has no type, the labels of objects are integers
                labels = labels.astype(np.int64)
            if labels.shape != (boxes.shape[0],):
                raise ValueError("There must be one label per bounding box!")

        if unclipped is not None:
            unclipped = np.asarray(unclipped, dtype=float)
            if unclipped.shape != boxes.shape:
                raise ValueError("Unclipped boxes must have the same shape as the boxes!")

        self.__data = boxes
        self.__height = height
        self.__width = width
        self.__labels = labels
        self.__unclipped = unclipped

    @staticmethod
    def _to_voc(boxes, fmt, height, width):
        boxes = boxes.copy()
        if fmt == "yolo":
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
        if fmt == "coco":
            boxes[:, 2:] += boxes[:, :2]
        elif fmt in ("cxcywh", "yolo"):
            centers, sizes = boxes[:, :2].copy(), boxes[:, 2:].copy()
            boxes[:, :2] = centers - sizes / 2
            boxes[:, 2:] = centers + sizes / 2
        return boxes

    def to(self, fmt):
        """
        Returns the boxes in a given format.

        Parameters
        ----------
        fmt : str
            ``'voc'``, ``'coco'``, ``'cxcywh'`` or ``'yolo'``.

        Returns
        -------
        out : numpy.ndarray
            Boxes ``Nx4``

        """
        if fmt not in self.FORMATS:
            raise ValueError(f"Unknown format of bounding boxes: {fmt}. Allowed: {self.FORMATS}")
        boxes = self.__data.copy()
        if fmt == "voc":
            return boxes
        sizes = boxes[:, 2:] - boxes[:, :2]
        if fmt == "coco":
            return np.hstack([boxes[:, :2], sizes])
        res = np.hstack([boxes[:, :2] + sizes / 2, sizes])
        if fmt == "yolo":
            res[:, [0, 2]] /= self.__width
            res[:, [1, 3]] /= self.__height
        return res

    @staticmethod
    def _areas(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def areas(self):
        """Areas of the boxes in pixels."""
        return BoundingBoxes._areas(self.__data)

    def unclipped_areas(self):
        """Areas of the boxes before clipping to the image, in pixels."""
        return BoundingBoxes._areas(self.unclipped)

    def clipped(self):
        """
        Returns the boxes clipped to the image. The unclipped boxes are kept to measure the fraction outside of
        the image. If all the boxes are inside of the image, the object itself is returned.

        Returns
        -------
        out : BoundingBoxes
            Clipped boxes

        """
        boxes = self.__data
        inside = (boxes >= 0).all() and (boxes[:, [0, 2]] <= self.__width).all()
        if inside and (boxes[:, [1, 3]] <= self.__height).all():
            return self
        clipped = boxes.copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, self.__width)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, self.__height)
        return BoundingBoxes(clipped, self.__height, self.__width, labels=self.__labels, unclipped=self.unclipped)

    @property
    def data(self):
        """Boxes ``Nx4`` in the VOC format."""
        return self.__data

    @property
    def labels(self):
        return self.__labels

    @property
    def unclipped(self):
        """Boxes ``Nx4`` in the VOC format before clipping to the image."""
        return self.__data if self.__unclipped is None else self.__unclipped

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    def __len__(self):
        return self.__data.shape[0]

    def __eq__(self, other):
        if not isinstance(other, BoundingBoxes):
            return False
        if (self.height, self.width) != (other.height, other.width) or not np.array_equal(self.data, other.data):
            return False
        if self.labels is None or other.labels is None:
            return self.labels is None and other.labels is None
        return np.array_equal(self.labels, other.labels)
