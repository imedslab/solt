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
        Data format. Example: 'IMMM' - image and three masks.
    transform_settings : dict or None
        Settings for each data item. At this stage, the settings include only padding and interpolation.
        The key in this dict corresponds to the index of the element in the given data tuple.
        The value is another dict, which has all the settings. Segmentation masks have nearest neighbor interpolation
        by default, this can be changed manually if needed.
        Example: ``transform_settings={0:{'interpolation':'bilinear'}, 1: {'interpolation':'bicubic'}}``

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
        else:
            transform_settings = {}

        # Element-wise settings
        # If no settings provided for certain items, they will be created
        for idx in range(len(data)):
            if idx not in transform_settings:
                transform_settings[idx] = {}

            if fmt[idx] == "I" or fmt[idx] == "M":
                val = ("nearest", "strict") if fmt[idx] == "M" else None
                if "interpolation" not in transform_settings[idx]:
                    transform_settings[idx]["interpolation"] = validate_parameter(
                        val, ALLOWED_INTERPOLATIONS, "bilinear", str, True
                    )
                else:
                    transform_settings[idx]["interpolation"] = validate_parameter(
                        (transform_settings[idx]["interpolation"], "strict"),
                        ALLOWED_INTERPOLATIONS,
                        "bilinear",
                        str,
                        True,
                    )

                if "padding" not in transform_settings[idx]:
                    transform_settings[idx]["padding"] = validate_parameter(None, ALLOWED_PADDINGS, "z", str, True)
                else:
                    transform_settings[idx]["padding"] = validate_parameter(
                        (transform_settings[idx]["padding"], "strict"),
                        ALLOWED_PADDINGS,
                        "z",
                        str,
                        True,
                    )
            else:
                if "interpolation" in transform_settings[idx] or "padding" in transform_settings[idx]:
                    raise TypeError

        if len(data) != len(transform_settings):
            raise ValueError

        for t in fmt:
            if t not in ALLOWED_TYPES:
                raise TypeError

        self.__data = data
        self.__fmt = fmt
        self.__transform_settings = transform_settings

        self.__imagenet_mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.__imagenet_std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

        self.validate()

    def validate(self):
        prev_h = None
        prev_w = None
        # Let's make sure that all the objects have the same coordinate frame
        data = self
        for obj, t, settings in data:
            if t == "M" or t == "I":
                h = obj.shape[0]
                w = obj.shape[1]
            elif t == "P":
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
        labels (``label`` and ``labels``), and the keypoints (``keypoints`` and ``keypoints_array``).
        You must use ``solt.data.KeyPoints`` object here. Labels will always be stored last.

        For example, if the input ``dict`` looks like this: ``d = {'label': l1, 'image': i1, 'mask': m1}`` or
        ``d = {'mask': m1, 'image': i1, 'label': l1}``, the ``DataContainer`` will convert this
        into ``sld.DataContainer((i1, m1, l1), 'IML')``.

        More complex case:

        .. highlight:: python
        .. code-block:: python

            d = {'image': i1, masks: (m1, m2, m3, m4), 'labels': (l1, l2, l3, l4, l5),
            'keypoints': solt.core.KeyPoints(k, h, w)
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
        if "keypoints" in data:
            dc_content.append(data["keypoints"])
            dc_format.append("P")
        if "keypoints_array" in data:
            dc_content.extend(data["keypoints_array"])
            dc_format.extend("P" * len(data["keypoints_array"]))
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
            sorted a similar manner to the method ``from_dict``: images, masks, keypoints_array, and labels.
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
            "keypoints_array": list(),
            "labels": list(),
        }
        not_as_dict = []
        for el, f in zip(self.__data, self.__fmt):
            if f == "I":
                scale = 255.0
                if el.dtype == np.uint16:
                    scale = 65535.0
                    el = el.astype(np.float32)
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

            elif f == "M":
                mask = torch.from_numpy(el).squeeze().unsqueeze(0).float()
                res_dict["masks"].append(mask)
                not_as_dict.append(mask)
            elif f == "P":
                landmarks = torch.from_numpy(el.data).float()
                if scale_keypoints:
                    landmarks[:, 0] /= el.width - 1
                    landmarks[:, 1] /= el.height - 1
                res_dict["keypoints_array"].append(landmarks)
                not_as_dict.append(landmarks)
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

        if len(res_dict["keypoints_array"]) == 1:
            res["keypoints"] = res_dict["keypoints_array"][0]
        elif len(res_dict["keypoints_array"]) > 1:
            res["keypoints_array"] = res_dict["keypoints_array"]

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
