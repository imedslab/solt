import copy
import random
from abc import ABCMeta, abstractmethod
import numpy as np

from solt.utils import Serializable
from ._data import DataContainer, Keypoints


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
        if self.data_indices is None:
            self.data_indices = tuple(range(len(data)))
        tmp_item = None
        for i, (item, t, item_settings) in enumerate(data):
            if i in self.data_indices:
                if t == "I":  # Image
                    tmp_item = self._apply_img(item, item_settings)
                elif t == "M":  # Mask
                    tmp_item = self._apply_mask(item, item_settings)
                elif t == "P":  # Points
                    tmp_item = self._apply_pts(item, item_settings)
                elif t == "L":  # Labels
                    tmp_item = self._apply_labels(item, item_settings)
            else:
                if t == "I" or t == "M":
                    tmp_item = item.copy()
                elif t == "L":
                    tmp_item = copy.copy(item)
                elif t == "P":
                    tmp_item = copy.copy(item)

            types.append(t)
            result.append(tmp_item)
            settings[i] = item_settings

        return DataContainer(data=tuple(result), fmt="".join(types))

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

    def _apply_vol(self, vol: np.ndarray, settings: dict):
        """
        Abstract method, which determines the transform's behaviour when it is applied to volumes HxWxDxC.

        Parameters
        ----------
        vol : numpy.ndarray
            Volume to be augmented

        Returns
        -------
        out : numpy.ndarray
            Result
        """

    def _apply_vol_mask(self, vol_mask: np.ndarray, settings: dict):
        """
        Abstract method, which determines the transform's behaviour when it is applied to volumetric masks HxWxD.

        Parameters
        ----------
        vol_mask : numpy.ndarray
            Volumetric mask to be augmented

        """


class ImageTransform(BaseTransform):
    """
    Abstract class, allowing the application of a transform only to an image

    """

    def __init__(self, p=None, data_indices=None):
        super(ImageTransform, self).__init__(p=p, data_indices=data_indices)

    def _apply_mask(self, mask, settings: dict):
        return mask

    def _apply_pts(self, pts: Keypoints, settings: dict):
        return pts

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
