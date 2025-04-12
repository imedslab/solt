import copy
import random
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np

from solt.utils import Serializable
from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS
from ._data import DataContainer, Keypoints
from solt.utils import (
    img_shape_checker,
    validate_parameter,
)


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
            raise TypeError
        if isinstance(data_indices, tuple):
            for el in data_indices:
                if not isinstance(el, int):
                    raise TypeError
                if el < 0:
                    raise ValueError

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


class MatrixTransform(BaseTransform, InterpolationPropertyHolder, PaddingPropertyHolder):
    """
    Matrix Transform abstract class. (Affine and Homography).
    Does all the transforms around the image /  center.

    Parameters
    ----------
    interpolation : str
        Interpolation mode.
    padding : str or None
        Padding Mode.
    p : float
        Probability of transform's execution.
    ignore_state : bool
        Whether to ignore the pre-calculated transformation or not. If False,
        then it will lead to an incorrect behavior when the objects are of different sizes.
        Should be used only when it is assumed that the image, mask and keypoints are of
        the same size.

    """

    def __init__(
        self,
        interpolation="bilinear",
        padding="z",
        p=0.5,
        ignore_state=True,
        affine=True,
        ignore_fast_mode=False,
    ):
        BaseTransform.__init__(self, p=p, data_indices=None)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        PaddingPropertyHolder.__init__(self, padding=padding)

        self.ignore_fast_mode = ignore_fast_mode
        self.fast_mode = False
        self.affine = affine
        self.ignore_state = ignore_state
        self.reset_state()

    def reset_state(self):
        BaseTransform.reset_state(self)
        self.state_dict["transform_matrix"] = np.eye(3)

    def fuse_with(self, trf):
        """
        Takes a transform an performs a matrix fusion. This is useful to optimize the computations

        Parameters
        ----------
        trf : MatrixTransform

        """

        if trf.padding is not None:
            self.padding = trf.padding
        self.interpolation = trf.interpolation

        self.state_dict["transform_matrix"] = trf.state_dict["transform_matrix"] @ self.state_dict["transform_matrix"]

    def sample_transform(self, data):
        """
        Samples the transform and corrects for frame change.

        Returns
        -------
        None

        """
        super(MatrixTransform, self).sample_transform(data)
        self.sample_transform_matrix(data)  # Only this method needs to be implemented!

        # If we are in fast mode, we do not have to recompute the the new coordinate frame!
        if "P" not in data.data_format and not self.ignore_fast_mode:
            width = self.state_dict["w"]
            height = self.state_dict["h"]
            origin = [(width - 1) // 2, (height - 1) // 2]
            # First, let's make sure that our transformation matrix is applied at the origin
            transform_matrix_corr = MatrixTransform.move_transform_to_origin(
                self.state_dict["transform_matrix"], origin
            )
            self.state_dict["h_new"], self.state_dict["w_new"] = (
                self.state_dict["h"],
                self.state_dict["w"],
            )
            self.state_dict["transform_matrix_corrected"] = transform_matrix_corr
        else:
            # If we have the keypoints or the transform is a homographic one, we can't use the fast mode at all.
            self.correct_transform()

    @staticmethod
    def move_transform_to_origin(transform_matrix, origin):
        # First we correct the transformation so that it is performed around the origin
        transform_matrix = transform_matrix.copy()
        t_origin = np.array([1, 0, -origin[0], 0, 1, -origin[1], 0, 0, 1]).reshape((3, 3))

        t_origin_back = np.array([1, 0, origin[0], 0, 1, origin[1], 0, 0, 1]).reshape((3, 3))
        transform_matrix = np.dot(t_origin_back, np.dot(transform_matrix, t_origin))

        return transform_matrix

    @staticmethod
    def recompute_coordinate_frame(transform_matrix, width, height):
        coord_frame = np.array([[0, 0, 1], [0, height, 1], [width, height, 1], [width, 0, 1]])
        new_frame = np.dot(transform_matrix, coord_frame.T).T
        new_frame[:, 0] /= new_frame[:, -1]
        new_frame[:, 1] /= new_frame[:, -1]
        new_frame = new_frame[:, :-1]
        # Computing the new coordinates

        # If during the transform, we obtained negative coordinates, we have to move to the origin
        if np.any(new_frame[:, 0] < 0):
            new_frame[:, 0] += abs(new_frame[:, 0].min())
        if np.any(new_frame[:, 1] < 0):
            new_frame[:, 1] += abs(new_frame[:, 1].min())

        new_frame[:, 0] -= new_frame[:, 0].min()
        new_frame[:, 1] -= new_frame[:, 1].min()
        w_new = int(np.round(new_frame[:, 0].max()))
        h_new = int(np.round(new_frame[:, 1].max()))

        return h_new, w_new

    @staticmethod
    def correct_for_frame_change(transform_matrix: np.ndarray, width: int, height: int):
        """
        Method takes a matrix transform, and modifies its origin.

        Parameters
        ----------
        transform_matrix : numpy.ndarray
            Transform (3x3) matrix
        width : int
            Width of the coordinate frame
        height : int
            Height of the coordinate frame
        Returns
        -------
        out : numpy.ndarray
            Modified Transform matrix

        """
        origin = [(width - 1) // 2, (height - 1) // 2]
        # First, let's make sure that our transformation matrix is applied at the origin
        transform_matrix = MatrixTransform.move_transform_to_origin(transform_matrix, origin)
        # Now, if we think of scaling, rotation and translation, the image size gets increased
        # when we apply any geometric transform. Default behaviour in OpenCV is designed to crop the
        # image edges, however it is not desired when we want to deal with Keypoints (don't want them
        # to exceed teh image size).

        # If we imagine that the image edges are a rectangle, we can rotate it around the origin
        # to obtain the new coordinate frame
        h_new, w_new = MatrixTransform.recompute_coordinate_frame(transform_matrix, width, height)
        transform_matrix[0, -1] += w_new // 2 - origin[0]
        transform_matrix[1, -1] += h_new // 2 - origin[1]

        return transform_matrix, w_new, h_new

    @abstractmethod
    def sample_transform_matrix(self, data):
        """
        Method that is called to sample the transform matrix

        """

    def correct_transform(self):
        h, w = self.state_dict["h"], self.state_dict["w"]
        tm = self.state_dict["transform_matrix"]
        tm_corr, w_new, h_new = MatrixTransform.correct_for_frame_change(tm, w, h)
        self.state_dict["h_new"], self.state_dict["w_new"] = h_new, w_new
        self.state_dict["transform_matrix_corrected"] = tm_corr

    def parse_settings(self, settings):
        interp = ALLOWED_INTERPOLATIONS[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS[settings["interpolation"][0]]

        padding = ALLOWED_PADDINGS[self.padding[0]]
        if settings["padding"][1] == "strict":
            padding = ALLOWED_PADDINGS[settings["padding"][0]]

        return interp, padding

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        """
        Applies a transform to an image or mask without controlling the shapes.

        Parameters
        ----------
        img : numpy.ndarray
            Image or mask
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Warped image

        """

        if self.affine:
            return self._apply_img_or_mask_affine(img, settings)
        else:
            return self._apply_img_or_mask_perspective(img, settings)

    def _apply_img_or_mask_perspective(self, img: np.ndarray, settings: dict):
        h_new, w_new = self.state_dict["h_new"], self.state_dict["w_new"]
        interp, padding = self.parse_settings(settings)
        transf_m = self.state_dict["transform_matrix_corrected"]
        return cv2.warpPerspective(img, transf_m, (w_new, h_new), flags=interp, borderMode=padding)

    def _apply_img_or_mask_affine(self, img: np.ndarray, settings: dict):
        h_new, w_new = self.state_dict["h_new"], self.state_dict["w_new"]
        interp, padding = self.parse_settings(settings)
        transf_m = self.state_dict["transform_matrix_corrected"]
        return cv2.warpAffine(img, transf_m[:2, :], (w_new, h_new), flags=interp, borderMode=padding)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        """
        Applies a matrix transform to an image.
        If padding is None, the default behavior (zero padding) is expected.

        Parameters
        ----------
        img : numpy.ndarray
            Input Image
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Output Image

        """

        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        """
        Abstract method, which defines the transform's behaviour when it is applied to masks HxW.

        If padding is None, the default behavior (zero padding) is expected.

        Parameters
        ----------
        mask : numpy.ndarray
            Mask to be augmented
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Result

        """
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        """
        Transform's application to labels. Simply returns them back without modifications.

        Parameters
        ----------
        labels : numpy.ndarray
            Array of labels.
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Result

        """
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        """
        Abstract method, which defines the transform's behaviour when it is applied to keypoints.

        Parameters
        ----------
        pts : Keypoints
            Keypoints object
        settings : dict
            Item-wise settings

        Returns
        -------
        out : Keypoints
            Result

        """
        if self.padding[0] == "r":
            raise ValueError("Cannot apply transform to keypoints with reflective padding!")

        pts_data = pts.data.copy()

        w_new = self.state_dict["w_new"]
        h_new = self.state_dict["h_new"]
        tm_corr = self.state_dict["transform_matrix_corrected"]

        pts_data = np.hstack((pts_data, np.ones((pts_data.shape[0], 1))))
        pts_data = np.dot(tm_corr, pts_data.T).T

        pts_data[:, 0] /= pts_data[:, 2]
        pts_data[:, 1] /= pts_data[:, 2]

        return Keypoints(pts_data[:, :-1], h_new, w_new)
