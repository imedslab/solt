import numpy as np
import cv2

from ._base_transforms import BaseTransform
from ._property_holders import InterpolationPropertyHolder, PaddingPropertyHolder
from ._data import Keypoints
from ..constants import ALLOWED_INTERPOLATIONS_2D, ALLOWED_PADDINGS_2D
from ..utils import img_shape_checker
from abc import abstractmethod


class MatrixTransform2D(BaseTransform, InterpolationPropertyHolder, PaddingPropertyHolder):
    """
    Matrix Transform abstract class. (Affine and Homography). Works for 2D data.
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
        super(MatrixTransform2D, self).sample_transform(data)
        self.sample_transform_matrix(data)  # Only this method needs to be implemented!

        # If we are in fast mode, we do not have to recompute the the new coordinate frame!
        if "P" not in data.data_format and not self.ignore_fast_mode:
            width = self.state_dict["w"]
            height = self.state_dict["h"]
            origin = [(width - 1) // 2, (height - 1) // 2]
            # First, let's make sure that our transformation matrix is applied at the origin
            transform_matrix_corr = MatrixTransform2D.move_transform_to_origin(
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
        transform_matrix = MatrixTransform2D.move_transform_to_origin(transform_matrix, origin)
        # Now, if we think of scaling, rotation and translation, the image size gets increased
        # when we apply any geometric transform. Default behaviour in OpenCV is designed to crop the
        # image edges, however it is not desired when we want to deal with Keypoints (don't want them
        # to exceed teh image size).

        # If we imagine that the image edges are a rectangle, we can rotate it around the origin
        # to obtain the new coordinate frame
        h_new, w_new = MatrixTransform2D.recompute_coordinate_frame(transform_matrix, width, height)
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
        tm_corr, w_new, h_new = MatrixTransform2D.correct_for_frame_change(tm, w, h)
        self.state_dict["h_new"], self.state_dict["w_new"] = h_new, w_new
        self.state_dict["transform_matrix_corrected"] = tm_corr

    def parse_settings(self, settings):
        interp = ALLOWED_INTERPOLATIONS_2D[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS_2D[settings["interpolation"][0]]

        padding = ALLOWED_PADDINGS_2D[self.padding[0]]
        if settings["padding"][1] == "strict":
            padding = ALLOWED_PADDINGS_2D[settings["padding"][0]]

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
