"""Base classes of the transforms."""

import copy
import random
from abc import ABCMeta, abstractmethod

import numpy as np

from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS
from solt.utils import Serializable, validate_parameter

from ._annotations import BoundingBoxes, Keypoints
from ._data import DataContainer


class BaseTransform(Serializable, metaclass=ABCMeta):
    """
    Transformation abstract class.

    Parameters
    ----------
    p : float or None
        Probability of executing this transform
    data_indices : tuple or None
        Indices where the transforms need to be applied
    """

    def __init__(self, p=None, data_indices=None):
        super(BaseTransform, self).__init__()

        if p is None:
            p = 0.5

        self.p = p
        if data_indices is not None and not isinstance(data_indices, tuple):
            raise TypeError("Data indices must be a tuple!")
        if isinstance(data_indices, tuple):
            for el in data_indices:
                if not isinstance(el, int):
                    raise TypeError("Data indices must be integers!")
                if el < 0:
                    raise ValueError("Data indices must be >= 0!")

        self.data_indices = data_indices

        self.state_dict = None
        self.reset_state()

    def reset_state(self):
        self.state_dict = {"use": False}

    def use_transform(self):
        """
        Method to randomly determine whether to use this transform.

        Returns
        -------
        out : bool
            Boolean flag. True if the transform is used.
        """
        if random.random() <= self.p:
            self.state_dict["use"] = True
            return True

        self.state_dict["use"] = False
        return False

    def sample_transform(self, data: DataContainer):
        """
        Samples transform parameters based on data.

        Parameters
        ----------
        data : DataContainer
            Data container to be used for sampling.

        Returns
        -------
        out : tuple
            Coordinate frame (h, w).
        """

        self.state_dict["h"], self.state_dict["w"] = data.validate()
        return self.state_dict["h"], self.state_dict["w"]

    def apply(self, data: DataContainer):
        """
        Applies transformation to a DataContainer items depending on the type.

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result

        """
        result = []
        types = []
        settings = {}
        # Not stored in the transform: it can be applied to data containers of different lengths
        data_indices = self.data_indices if self.data_indices is not None else range(len(data))
        tmp_item = None
        for i, (item, t, item_settings) in enumerate(data):
            if i in data_indices:
                if t == "I":  # Image
                    tmp_item = self._apply_img(item, item_settings)
                elif t == "M":  # Mask
                    tmp_item = self._apply_mask(item, item_settings)
                elif t == "D":  # Distance map
                    tmp_item = self._apply_distance_map(item, item_settings)
                elif t == "P":  # Points
                    tmp_item = self._apply_pts(item, item_settings)
                elif t == "B":  # Bounding boxes
                    tmp_item = self._apply_bboxes(item, item_settings)
                elif t == "L":  # Labels
                    tmp_item = self._apply_labels(item, item_settings)
            else:
                if t == "I" or t == "M" or t == "D":
                    tmp_item = item.copy()
                elif t == "L":
                    tmp_item = copy.copy(item)
                elif t == "P" or t == "B":
                    tmp_item = copy.copy(item)

            types.append(t)
            result.append(tmp_item)
            settings[i] = item_settings

        return DataContainer(data=tuple(result), fmt="".join(types), transform_settings=settings)

    @staticmethod
    def wrap_data(data):
        if isinstance(data, np.ndarray):
            data = DataContainer((data,), "I")
        elif isinstance(data, dict):
            data = DataContainer.from_dict(data)
        elif not isinstance(data, DataContainer):
            raise TypeError("Unknown data type!")
        return data

    def __call__(
        self,
        data,
        return_torch=False,
        as_dict=True,
        scale_keypoints=True,
        normalize=True,
        mean=None,
        std=None,
    ):
        """
        Applies the transform to a DataContainer

        Parameters
        ----------
        data : DataContainer or dict or np.ndarray.
            Data to be augmented. See ``solt.core.DataContainer.from_dict`` for details.
            If np.ndarray, then the data will be wrapped as a data container with format
            ``I``.
        return_torch : bool
            Whether to convert the result into a torch tensors.
            By default, it is `False` for transforms and ``True`` for the streams.
        as_dict : bool
            Whether to pool the results into a dict.
            See ``solt.core.DataContainer.to_dict`` for details
        scale_keypoints : bool
            Whether to scale the keypoints into 0-1 range
        normalize : bool
            Whether to normalize the resulting tensor. If mean or std args are None,
            ImageNet statistics will be used
        mean : None or tuple of float or np.ndarray or torch.FloatTensor
            Mean to subtract for the converted tensor
        std : None or tuple of float or np.ndarray or torch.FloatTensor
            Mean to subtract for the converted tensor

        Returns
        -------
        out : DataContainer or dict or list
            Result

        """

        data = BaseTransform.wrap_data(data)

        self.reset_state()
        if self.use_transform():
            self.sample_transform(data)
            res = self.apply(data)
        else:
            res = data

        if return_torch:
            return res.to_torch(
                as_dict=as_dict,
                scale_keypoints=scale_keypoints,
                normalize=normalize,
                mean=mean,
                std=std,
            )
        return res

    @abstractmethod
    def _apply_img(self, img: np.ndarray, settings: dict):
        """
        Abstract method, which determines the transform's behaviour when it is applied to images HxWxC.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be augmented

        Returns
        -------
        out : numpy.ndarray

        """

    @abstractmethod
    def _apply_mask(self, mask: np.ndarray, settings: dict):
        """
        Abstract method, which determines the transform's behaviour when it is applied to masks HxW.

        Parameters
        ----------
        mask : numpy.ndarray
            Mask to be augmented

        Returns
        -------
        out : numpy.ndarray
            Result

        """

    def _apply_distance_map(self, dmap: np.ndarray, settings: dict):
        """
        Method, which determines the transform's behaviour when it is applied to distance maps HxW or HxWxC,
        e.g. signed distance fields.

        Unlike masks, the values of distance maps have a geometric meaning, so every transform has to define
        explicitly how it changes them. By default, the transforms do not support distance maps.

        Parameters
        ----------
        dmap : numpy.ndarray
            Distance map to be augmented
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Result

        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support distance maps!")

    def _apply_bboxes(self, bboxes: BoundingBoxes, settings: dict):
        """
        Method, which determines the transform's behaviour when it is applied to bounding boxes.

        The corners of the boxes are transformed as keypoints, and the result is the box enclosing the transformed
        corners. ``DataContainer`` clips them to the image, and ``Stream`` drops the ones, which became too small,
        at the end.

        Parameters
        ----------
        bboxes : BoundingBoxes
            Bounding boxes to be augmented
        settings : dict
            Item-wise settings

        Returns
        -------
        out : BoundingBoxes
            Result

        """
        n = len(bboxes)
        tracks_unclipped = bboxes.unclipped is not bboxes.data
        # The unclipped boxes are transformed together with the boxes, to measure the fraction outside of the image
        boxes = np.concatenate([bboxes.data, bboxes.unclipped]) if tracks_unclipped else bboxes.data
        corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
        # Keypoints are the centres of pixels, while the coordinates of the boxes are continuous
        pts = self._apply_pts(Keypoints(corners.reshape(-1, 2) - 0.5, bboxes.height, bboxes.width), settings)
        corners = pts.data.reshape(-1, 4, 2) + 0.5
        boxes = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
        unclipped = boxes[n:] if tracks_unclipped else None
        # DataContainer clips the boxes to the new image
        return BoundingBoxes(boxes[:n], pts.height, pts.width, labels=bboxes.labels, unclipped=unclipped)

    @abstractmethod
    def _apply_labels(self, labels, settings: np.ndarray):
        """
        Abstract method, which determines the transform's behaviour when it is applied to labels (e.g. label smoothing)

        Parameters
        ----------
        labels : numpy.ndarray
            Array of labels.

        Returns
        -------
        out : numpy.ndarray
            Result

        """

    @abstractmethod
    def _apply_pts(self, pts: Keypoints, settings: dict):
        """
        Abstract method, which determines the transform's behaviour when it is applied to keypoints.

        Parameters
        ----------
        pts : Keypoints
            Keypoints object

        Returns
        -------
        out : Keypoints
            Result

        """


class ImageTransform(BaseTransform):
    """
    Abstract class, allowing the application of a transform only to an image

    """

    def __init__(self, p=None, data_indices=None):
        super(ImageTransform, self).__init__(p=p, data_indices=data_indices)

    def _apply_mask(self, mask, settings: dict):
        return mask

    def _apply_distance_map(self, dmap, settings: dict):
        return dmap

    def _apply_pts(self, pts: Keypoints, settings: dict):
        return pts

    def _apply_bboxes(self, bboxes: BoundingBoxes, settings: dict):
        return bboxes

    def _apply_labels(self, labels, settings: dict):
        return labels

    @abstractmethod
    def _apply_img(self, img: np.ndarray, settings: dict):
        """
        Abstract method, which determines the transform's behaviour when it is applied to images HxWxC.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be augmented

        Returns
        -------
        out : numpy.ndarray

        """


class PaddingPropertyHolder(object):
    """
    PaddingPropertyHolder

    Adds padding property to a class and validates it using the allowed paddings from constants.

    Parameters
    ----------
    padding : None or str or tuple
        Padding mode. Inheritance can be specified as the second argument of the `padding` tuple.

    """

    def __init__(self, padding=None):
        super(PaddingPropertyHolder, self).__init__()
        self.padding = validate_parameter(padding, ALLOWED_PADDINGS, "z")


class InterpolationPropertyHolder(object):
    """
    InterpolationPropertyHolder

    Adds interpolation property to a class and validates it using the allowed interpolations from constants.

    Parameters
    ----------
    interpolation : None or str or tuple
        Interpolation mode. Inheritance can be specified as the second argument of the `interpolation` tuple.

    """

    def __init__(self, interpolation=None):
        super(InterpolationPropertyHolder, self).__init__()
        self.interpolation = validate_parameter(interpolation, ALLOWED_INTERPOLATIONS, "bilinear")
