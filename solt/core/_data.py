import copy
import numbers

import numpy as np
import torch

from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS, ALLOWED_TYPES
from solt.utils import validate_parameter


class DataContainer(object):
    """
    Data container to encapsulate different types of data, such as images, bounding boxes, etc.

    The container itself is iterable according to the format.
    Data container assumes the same coordinate system for all the items it is storing.
    For instance, the behavior shown below is not supported:

    .. highlight:: python
    .. code-block:: python

        from solt.core import DataContainer
        import numpy as np

        img1 = np.ones((100, 100, 3), dtype=np.int32)
        img2 = np.ones((110, 110, 3), dtype=np.int32)
        trf_img = DataContainer.from_dict({'images': (img1, img2)})

    Parameters
    ----------
    data : tuple
        Data items stored in a tuple
    fmt : str
        Data format. Example: 'IMMM' - image and three masks. See ``solt.constants.ALLOWED_TYPES``.
    transform_settings : dict or None
        Settings for each data item. At this stage, the settings include padding and interpolation, and for
        distance maps also the fill value.
        The key in this dict corresponds to the index of the element in the given data tuple.
        The value is another dict, which has all the settings. Segmentation masks have nearest neighbor interpolation
        by default, this can be changed manually if needed.
        Example: ``transform_settings={0:{'interpolation':'bilinear'}, 1: {'interpolation':'bicubic'}}``

        Bounding boxes (``'B'``) are clipped to the image, when a container is created, i.e. after every transform.

        Distance maps (``'D'``), e.g. signed distance fields, are floating point arrays ``HxW`` or ``HxWxC``.
        They have bilinear interpolation by default, and their values are rescaled when a transform changes the
        scale of the image (the rescaling is exact for similarity transforms and approximate otherwise).
        The zero padding would create a false object boundary, so a distance map is padded with its ``fill``
        value instead, if it is set, and with its edge values otherwise.
        Example: ``transform_settings={1: {'fill': 100.0}}``

    """

    def __init__(self, data, fmt, transform_settings=None):
        if len(fmt) == 1 and not isinstance(data, tuple):
            if not isinstance(data, list):
                data = (data,)
            else:
                raise TypeError

        if not isinstance(data, tuple):
            raise TypeError

        if len(data) != len(fmt):
            raise ValueError

        if transform_settings is not None:
            if not isinstance(transform_settings, dict):
                raise TypeError
            if not set(transform_settings).issubset(range(len(data))):
                raise ValueError
        else:
            transform_settings = {}

        # Element-wise settings. They are parsed into a new dict, so that the caller's dict stays untouched
        # and can be re-used. If no settings provided for certain items, they will be created
        parsed_settings = {}
        for idx in range(len(data)):
            item_settings = transform_settings.get(idx, {})
            if not isinstance(item_settings, dict):
                raise TypeError
            item_settings = dict(item_settings)

            if fmt[idx] in ("I", "M", "D"):
                val = {"I": None, "M": ("nearest", "strict"), "D": ("bilinear", "strict")}[fmt[idx]]
                item_settings["interpolation"] = self._parse_item_setting(
                    item_settings, "interpolation", ALLOWED_INTERPOLATIONS, "bilinear", val
                )
                item_settings["padding"] = self._parse_item_setting(item_settings, "padding", ALLOWED_PADDINGS, "z")
            else:
                if "interpolation" in item_settings or "padding" in item_settings:
                    raise TypeError

            if fmt[idx] == "D":
                item_settings["fill"] = self._parse_fill(item_settings)
            elif "fill" in item_settings:
                raise TypeError("Fill value can only be set for distance maps!")

            parsed_settings[idx] = item_settings
        transform_settings = parsed_settings

        for t in fmt:
            if t not in ALLOWED_TYPES:
                raise TypeError

        # The content outside of the image is lost, so the bounding boxes are clipped to it after every transform
        data = tuple(
            item.clipped() if t == "B" and isinstance(item, BoundingBoxes) else item for item, t in zip(data, fmt)
        )

        self.__data = data
        self.__fmt = fmt
        self.__transform_settings = transform_settings

        self.__imagenet_mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.__imagenet_std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

        self.validate()

    @staticmethod
    def _parse_item_setting(item_settings, name, allowed_modes, default_value, value_if_missing=None):
        """
        Parses a single item-wise setting, e.g. interpolation.

        A plain value (e.g. ``'bilinear'``) is set by the user for this very item, so it becomes strict.
        An already parsed value (e.g. ``('bilinear', 'inherit')``), which comes from another ``DataContainer``,
        is kept as is.
        """
        if name not in item_settings:
            return validate_parameter(value_if_missing, allowed_modes, default_value, str, True)

        value = item_settings[name]
        if not isinstance(value, (tuple, list)):
            value = (value, "strict")
        return validate_parameter(value, allowed_modes, default_value, str, True)

    @staticmethod
    def _parse_fill(item_settings):
        """
        Parses the fill value of a distance map. ``None`` (default) means that the edge values are used for padding.
        """
        fill = item_settings.get("fill")
        if fill is None:
            return None
        if not isinstance(fill, numbers.Real) or isinstance(fill, (bool, np.bool_)):
            raise TypeError("Fill value must be a number!")
        if not np.isfinite(fill):
            raise ValueError("Fill value must be finite!")
        return fill

    def validate(self):
        prev_h = None
        prev_w = None
        # Let's make sure that all the objects have the same coordinate frame
        data = self
        for obj, t, settings in data:
            if t == "D":
                if not isinstance(obj, np.ndarray) or not np.issubdtype(obj.dtype, np.floating):
                    raise TypeError("Distance maps must be floating point numpy arrays!")
                if obj.ndim not in (2, 3):
                    raise ValueError("Distance maps must be HxW or HxWxC arrays!")

            if t == "B" and not isinstance(obj, BoundingBoxes):
                raise TypeError("Bounding boxes must be stored in a BoundingBoxes object!")

            if t in ("M", "I", "D"):
                h = obj.shape[0]
                w = obj.shape[1]
            elif t == "P" or t == "B":
                h = obj.height
                w = obj.width
            elif t == "L":
                continue

            if prev_h is None:
                prev_h = h
            else:
                if prev_h != h:
                    raise ValueError("Coordinate frames within DataContainer do not match (height)!")

            if prev_w is None:
                prev_w = w
            else:
                if prev_w != w:
                    raise ValueError("Coordinate frames within DataContainer do not match (width)!")

        return prev_h, prev_w

    def copy(self):
        """
        Returns a copy of the container, in which all the arrays are copied as well. Transforms may modify the
        arrays in place, so a copy keeps the original data unchanged.

        Returns
        -------
        out : DataContainer
            Copy of the container

        """
        data = []
        for item, t, _ in self:
            if t in ("I", "M", "D"):
                data.append(item.copy())
            elif t == "P":
                data.append(Keypoints(item.data.copy(), item.height, item.width))
            elif t == "B":
                labels = None if item.labels is None else item.labels.copy()
                unclipped = None if item.unclipped is item.data else item.unclipped.copy()
                data.append(
                    BoundingBoxes(item.data.copy(), item.height, item.width, labels=labels, unclipped=unclipped)
                )
            else:
                data.append(copy.copy(item))
        return DataContainer(tuple(data), self.data_format, self.transform_settings)

    @property
    def data_format(self):
        return self.__fmt

    @property
    def data(self):
        return self.__data

    @property
    def transform_settings(self):
        return self.__transform_settings

    @staticmethod
    def from_dict(data):
        """
        Creates a data container from a dictionary.

        If data is a dict, then the ``solt.data.DataContainer`` will be created so that the
        ``image`` data will be stored first. Subsequently, multiple images under the key ``images``
        will be stored. The same applies to masks (first ``mask`` and then ``masks``),
        distance maps (``distance_map`` and ``distance_maps``), labels (``label`` and ``labels``),
        the keypoints (``keypoints`` and ``keypoints_array``) and the bounding boxes (``bboxes`` and
        ``bboxes_array``). The bounding boxes must be stored in ``solt.core.BoundingBoxes`` objects.
        You must use ``solt.data.KeyPoints`` object here. Labels will always be stored last.

        For example, if the input ``dict`` looks like this: ``d = {'label': l1, 'image': i1, 'mask': m1}`` or
        ``d = {'mask': m1, 'image': i1, 'label': l1}``, the ``DataContainer`` will convert this
        into ``sld.DataContainer((i1, m1, l1), 'IML')``. The order of the types is images, masks, distance maps,
        keypoints, bounding boxes and labels.

        More complex case:

        .. highlight:: python
        .. code-block:: python

            d = {'image': i1, masks: (m1, m2, m3, m4), 'labels': (l1, l2, l3, l4, l5),
            'keypoints': solt.core.KeyPoints(k, h, w)}
            dc_from_dict = solt.core.DataContainer.from_dict(d)

        will be equivalent to

        .. highlight:: python
        .. code-block:: python

            dc = solt.core.DataContainer((i1, m1, m2, m3, m4, solt.core.KeyPoints(k, h, w), l1, l2, l3, l4, l5),
            'IMMMMPLLLLLL').


        Please note, that when you create DataContainer using such a simplified interface,
        you cannot setup the transform parameters per item. Use a proper constructor instead.

        Parameters
        ----------
        data : dict
            Data stored in a dictionary

        Returns
        -------
        out : sld.DataContainer
            Newly instantiated data container object
        """
        dc_content = []
        dc_format = []
        if "image" in data:
            dc_content.append(data["image"])
            dc_format.append("I")
        if "images" in data:
            dc_content.extend(data["images"])
            dc_format.extend("I" * len(data["images"]))
        if "mask" in data:
            dc_content.append(data["mask"])
            dc_format.append("M")
        if "masks" in data:
            dc_content.extend(data["masks"])
            dc_format.extend("M" * len(data["masks"]))
        if "distance_map" in data:
            dc_content.append(data["distance_map"])
            dc_format.append("D")
        if "distance_maps" in data:
            dc_content.extend(data["distance_maps"])
            dc_format.extend("D" * len(data["distance_maps"]))
        if "keypoints" in data:
            dc_content.append(data["keypoints"])
            dc_format.append("P")
        if "keypoints_array" in data:
            dc_content.extend(data["keypoints_array"])
            dc_format.extend("P" * len(data["keypoints_array"]))
        if "bboxes" in data:
            dc_content.append(data["bboxes"])
            dc_format.append("B")
        if "bboxes_array" in data:
            dc_content.extend(data["bboxes_array"])
            dc_format.extend("B" * len(data["bboxes_array"]))
        if "label" in data:
            dc_content.append(data["label"])
            dc_format.append("L")
        if "labels" in data:
            dc_content.extend(data["labels"])
            dc_format.extend("L" * len(data["labels"]))

        return DataContainer(tuple(dc_content), "".join(dc_format))

    def wrap_mean_std(self, img, mean, std):
        if not isinstance(mean, (tuple, list, np.ndarray, torch.FloatTensor)):
            raise TypeError(
                f"Unknown type ({type(mean)}) of mean vector! " f"Expected tuple, list, np.ndarray or torch.FloatTensor"
            )

        if not isinstance(std, (tuple, list, np.ndarray, torch.FloatTensor)):
            raise TypeError(
                f"Unknown type ({type(mean)}) of mean vector! " f"Expected tuple, list, np.ndarray or torch.FloatTensor"
            )
        if len(mean) != img.size(0):
            raise ValueError("Size of the mean vector does not match the number of channels")
        if len(std) != img.size(0):
            raise ValueError("Size of the std vector does not match the number of channels")

        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean).view(img.size(0), 1, 1)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std).view(img.size(0), 1, 1)

        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).view(img.size(0), 1, 1).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).view(img.size(0), 1, 1).float()

        return mean, std

    def to_torch(self, as_dict=False, scale_keypoints=True, normalize=False, mean=None, std=None):
        """
        This method converts the DataContainer Content into a dict or a list PyTorch objects

        Parameters
        ----------
        as_dict : bool
            Whether to return the result as a dictionary. If a single item is present, then the singular naming
            will be used. If plural, then the plural will be used. The items will be stored and
            sorted a similar manner to the method ``from_dict``: images, masks, distance maps, keypoints_array,
            bounding boxes and labels. Bounding boxes become dicts with ``boxes`` (``Nx4`` ``float32`` tensor in the
            VOC format, in pixels) and ``labels`` (``int64`` tensor for integer labels), the target format of the
            detection models of torchvision.
            The same applies to a singular case/
        scale_keypoints : bool
            Whether to scale keypoints to 0-1 range. ``True`` by default.
        normalize : bool
            Whether to subtract mean
        mean : torch.Tensor
            Mean to subtract. If None, then the ImageNet mean will be subtracted.
        std : torch.Tensor
            Std to subtract. If None, then the ImageNet std will be subtracted.
        """
        res_dict = {
            "images": list(),
            "masks": list(),
            "distance_maps": list(),
            "keypoints_array": list(),
            "bboxes_array": list(),
            "labels": list(),
        }
        not_as_dict = []
        for el, f in zip(self.__data, self.__fmt):
            if f == "I":
                scale = 255.0
                if el.dtype == np.uint16:
                    scale = 65535.0
                    el = el.astype(np.float32)
                if el.ndim == 2:
                    # HxW (grayscale) -> HxWx1
                    el = el[:, :, np.newaxis]
                img = torch.from_numpy(el.transpose((2, 0, 1))).div(scale)
                if normalize:
                    if mean is None or std is None:
                        mean, std = self.__imagenet_mean, self.__imagenet_std
                    else:
                        mean, std = self.wrap_mean_std(img, mean, std)

                    img.sub_(mean)
                    img.div_(std)
                res_dict["images"].append(img)
                not_as_dict.append(img)

            elif f == "M" or f == "D":
                # HxW -> 1xHxW, HxWxC -> CxHxW
                el = el[np.newaxis] if el.ndim == 2 else el.transpose((2, 0, 1))
                mask = torch.from_numpy(np.ascontiguousarray(el)).float()
                res_dict["masks" if f == "M" else "distance_maps"].append(mask)
                not_as_dict.append(mask)
            elif f == "P":
                landmarks = torch.from_numpy(el.data).float()
                if scale_keypoints:
                    landmarks[:, 0] /= el.width - 1
                    landmarks[:, 1] /= el.height - 1
                res_dict["keypoints_array"].append(landmarks)
                not_as_dict.append(landmarks)
            elif f == "B":
                # The target format of the detection models of torchvision
                target = {"boxes": torch.from_numpy(el.data.astype(np.float32))}
                if el.labels is not None:
                    if np.issubdtype(el.labels.dtype, np.integer) or np.issubdtype(el.labels.dtype, np.bool_):
                        target["labels"] = torch.from_numpy(el.labels.astype(np.int64))
                    elif np.issubdtype(el.labels.dtype, np.floating):
                        target["labels"] = torch.from_numpy(el.labels)
                    else:
                        target["labels"] = el.labels.tolist()
                res_dict["bboxes_array"].append(target)
                not_as_dict.append(target)
            elif f == "L":
                res_dict["labels"].append(el)
                not_as_dict.append(el)
        if not as_dict:
            if len(not_as_dict) == 1:
                return not_as_dict[0]
            return not_as_dict

        return self.remap_results_dict(res_dict)

    @staticmethod
    def remap_results_dict(res_dict):
        res = {}
        if len(res_dict["images"]) == 1:
            res["image"] = res_dict["images"][0]
        elif len(res_dict["images"]) > 1:
            res["images"] = res_dict["images"]

        if len(res_dict["masks"]) == 1:
            res["mask"] = res_dict["masks"][0]
        elif len(res_dict["masks"]) > 1:
            res["masks"] = res_dict["masks"]

        if len(res_dict["distance_maps"]) == 1:
            res["distance_map"] = res_dict["distance_maps"][0]
        elif len(res_dict["distance_maps"]) > 1:
            res["distance_maps"] = res_dict["distance_maps"]

        if len(res_dict["keypoints_array"]) == 1:
            res["keypoints"] = res_dict["keypoints_array"][0]
        elif len(res_dict["keypoints_array"]) > 1:
            res["keypoints_array"] = res_dict["keypoints_array"]

        if len(res_dict["bboxes_array"]) == 1:
            res["bboxes"] = res_dict["bboxes_array"][0]
        elif len(res_dict["bboxes_array"]) > 1:
            res["bboxes_array"] = res_dict["bboxes_array"]

        if len(res_dict["labels"]) == 1:
            res["label"] = res_dict["labels"][0]
        elif len(res_dict["labels"]) > 1:
            res["labels"] = res_dict["labels"]
        return res

    def __getitem__(self, idx):
        """
        Returns a data item and its type using index.

        Parameters
        ----------
        idx : int
            Index of an element to return according to the specified format

        Returns
        -------
        out : tuple
            Data item (e.g. numpy.ndarray), its type, e.g. 'I' - image and the transform settings,
            e.g. interpolation

        """
        return self.__data[idx], self.__fmt[idx], self.__transform_settings[idx]

    def __len__(self):
        return len(self.__data)

    def __eq__(self, other):
        fmt_equal = self.data_format == other.data_format
        data_equal = True
        for d1, d2 in zip(self.data, other.data):
            if isinstance(d1, np.ndarray):
                data_equal = data_equal and np.array_equal(d1, d2)
            elif isinstance(d1, Keypoints):
                data_equal = data_equal and (d1 == d2)
            else:
                data_equal = data_equal and d1 == d2

        return fmt_equal and data_equal


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
