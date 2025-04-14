import numpy as np

from ._base_transforms import (
    BaseTransform,
    MatrixTransform,
)
import copy
import random

from solt.utils import Serializable
from ._data import DataContainer


class Stream(Serializable):
    """
    Stream class. Executes the list of transformations.

    The stream can be called directly on a dict of items,
    which get converted into a DataContainer. Note that the DataContainer assumes
    **the same** coordinate system for all the items it is storing.
    For instance, the behavior shown below is not supported for any stream:

    .. highlight:: python
    .. code-block:: python

        import solt.transforms as slt
        import numpy as np

        trf = solt.Stream([slt.Resize(resize_to=(50, 50)),
                               slt.Flip()])
        img1 = np.ones((100, 100, 3), dtype=np.int32)
        img2 = np.ones((110, 110, 3), dtype=np.int32)
        trf({'images': (img1, img2)})

    Parameters
    ----------
    transforms : list or None
        List of transforms to execute
    interpolation : str or None
        Stream-wide settings for interpolation. If for some particular transform your would like
        to still use its own mode, simply pass ``(<interpolation_value>, 'strict')``
        in the constructor of that transform.
    padding : str or None
        Stream-wide settings for padding. If for some particular transform your would like
        to still use its own mode, simply pass ``(<padding_value>, 'strict')``
        in the constructor of that transform.
    optimize_stack : bool
        Whether to run transforms stack optimization. It can only be useful if many matrix transformations are
        in a row.
    ignore_fast_mode : bool
        Whether to ignore the fast mode. This option enables full geometric transforms.
    shuffle : bool
        Whether to shuffle the transforms before applying them.

    """

    serializable_name = "stream"
    """
    How the class should be stored in the registry"""

    def __init__(
        self,
        transforms=None,
        interpolation=None,
        padding=None,
        optimize_stack=False,
        ignore_fast_mode=False,
        shuffle=False,
    ):
        super(Stream, self).__init__()

        if transforms is None:
            transforms = []

        for trf in transforms:
            if not isinstance(trf, BaseTransform) and not isinstance(trf, Stream):
                raise TypeError("Unknown transform type found in the Stream!")
        self.shuffle = shuffle
        self.optimize_stack = optimize_stack
        self.interpolation = interpolation
        self.padding = padding
        self.ignore_fast_mode = ignore_fast_mode
        self.transforms = transforms

        self.reset_ignore_fast_mode(ignore_fast_mode)
        self.reset_padding(padding)
        self.reset_interpolation(interpolation)

    def reset_ignore_fast_mode(self, value):
        if not isinstance(value, bool):
            raise TypeError("Ignore fast mode must be bool!")
        for trf in self.transforms:
            if isinstance(trf, MatrixTransform):
                trf.ignore_fast_mode = value

    def reset_interpolation(self, value):
        """
        Resets the interpolation for the whole pipeline of transforms.

        Parameters
        ----------
        value : str or None
            A value from ``solt.constants.ALLOWED_INTERPOLATIONS``

        See also
        --------
        solt.constants.ALLOWED_INTERPOLATIONS

        """
        if value is None:
            return
        self.interpolation = value
        for trf in self.transforms:
            if self.interpolation is not None and hasattr(trf, "interpolation"):
                if isinstance(trf, BaseTransform):
                    if trf.interpolation[1] != "strict":
                        trf.interpolation = (self.interpolation, trf.interpolation[1])
                elif isinstance(trf, Stream):
                    trf.reset_interpolation(self.interpolation)

    def reset_padding(self, value):
        """
        Allows to reset the padding for the whole Stream

        Parameters
        ----------
        value : str
            Should be a string from ``solt.constants.ALLOWED_PADDINGS``

        See also
        --------
        solt.constants.ALLOWED_PADDINGS

        """
        if value is None:
            return
        self.padding = value
        for trf in self.transforms:
            if self.padding is not None and hasattr(trf, "padding"):
                if isinstance(trf, BaseTransform):
                    if trf.padding[1] != "strict":
                        trf.padding = (self.padding, trf.padding[1])
                elif isinstance(trf, Stream):
                    trf.reset_padding(self.padding)

    def __call__(
        self,
        data,
        return_torch=True,
        as_dict=True,
        scale_keypoints=True,
        normalize=True,
        mean=None,
        std=None,
    ):
        """
        Executes the list of the pre-defined transformations for a given data container.

        Parameters
        ----------
        data : DataContainer or dict
            Data to be augmented. See ``solt.core.DataContainer.from_dict`` for details.
        return_torch : bool
            Whether to convert the result into a torch tensors. By default, it is false for transforms and
            true for the streams.
        as_dict : bool
            Whether to pool the results into a dict. See ``solt.core.DataContainer.to_dict`` for details
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

        res: DataContainer = Stream.exec_stream(self.transforms, data, self.optimize_stack)

        if return_torch:
            return res.to_torch(
                as_dict=as_dict,
                scale_keypoints=scale_keypoints,
                normalize=normalize,
                mean=mean,
                std=std,
            )
        return res

    @staticmethod
    def optimize_transforms_stack(transforms, data):
        """
        Static method which fuses the transformations

        Parameters
        ----------
        transforms : list
            A list of transforms
        data : DataContainer
            Data container to be used to sample the transforms

        Returns
        -------
        out : list
            An optimized list of transforms

        """
        # First we should create a stack
        transforms_stack = []
        for trf in transforms:
            if isinstance(trf, MatrixTransform):
                trf.ignore_fast_mode = True
                trf.reset_state()
                if trf.use_transform():
                    trf.sample_transform(data)
                    if len(transforms_stack) == 0:
                        transforms_stack.append(trf)
                    else:
                        transforms_stack[-1].fuse_with(trf)
            else:
                raise TypeError("Nested streams or other transforms but the `Matrix` ones are not supported!")

        if len(transforms_stack) > 0:
            transforms_stack[-1].correct_transform()
        return transforms_stack

    @staticmethod
    def exec_stream(transforms, data, optimize_stack, shuffle=False):
        """
        Static method, executes the list of transformations for a given data point.

        Parameters
        ----------
        transforms : list
            List of transformations to execute
        data : DataContainer or dict
            Data to be augmented. See ``solt.data.DataContainer.from_dict``
            to check how a conversion from dict is done.
        optimize_stack : bool
            Whether to execute augmentations stack optimization.

        Returns
        -------
        out : DataContainer
            Result
        """

        data = BaseTransform.wrap_data(data)
        if shuffle:
            random.shuffle(transforms)
        # Performing the transforms using the optimized stack
        if optimize_stack:
            transforms = Stream.optimize_transforms_stack(transforms, data)
        for trf in transforms:
            if isinstance(trf, BaseTransform):
                if not optimize_stack:
                    data = trf(data, return_torch=False)
                else:
                    data = trf.apply(data)
            elif isinstance(trf, Stream):
                data = trf(data, return_torch=False)
            else:
                raise TypeError("Unknown transform type found in the Stream!")
        return data


class SelectiveStream(Stream):
    """
    Stream that uniformly selects n out of k given transforms.

    """

    serializable_name = "selective_stream"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        transforms=None,
        n=1,
        probs=None,
        optimize_stack=False,
        ignore_fast_mode=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        transforms : list
            List of k transforms to sample from
        n : int or tuple of int
            How many transform to sample. If tuple, the number of transforms
            to sample will be chosen randomly from the tuple.
        probs : list of float
            Probabilities of the transforms to sample. If None, the transforms will be sampled uniformly.
        optimize_stack : bool
            Whether to execute stack optimization for augmentations.
        """
        super(SelectiveStream, self).__init__(
            transforms=transforms,
            optimize_stack=optimize_stack,
            ignore_fast_mode=ignore_fast_mode,
        )
        if transforms is None:
            transforms = []
        if n < 0 or n > len(transforms):
            raise ValueError("Number of transforms to sample must be between 0 and the number of transforms")
        if probs is not None:
            if len(probs) != len(transforms):
                raise ValueError("Number of probabilities must be equal to the number of transforms")
        self.n = n
        self.probs = probs

    def __call__(
        self,
        data,
        return_torch=True,
        as_dict=True,
        scale_keypoints=True,
        normalize=True,
        mean=None,
        std=None,
    ):
        """
        Applies randomly selected n transforms to the given data item

        Parameters
        ----------
        data : DataContainer
            Data to be augmented
        return_torch : bool
            Whether to convert the result into a torch tensors
        as_dict : bool
            Whether to pool the results into a dict. See ``solt.core.DataContainer.to_dict``
            for details.
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
        out : DataContainer
            Result
        """
        data = BaseTransform.wrap_data(data)

        if len(self.transforms) > 0:
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            if isinstance(self.n, tuple):
                n = random_state.choice(self.n, replace=False)
            else:
                n = self.n

            trfs = random_state.choice(self.transforms, n, replace=False, p=self.probs)
            if self.optimize_stack:
                trfs = [copy.deepcopy(x) for x in trfs]
                trfs = Stream.optimize_transforms_stack(trfs, data)
            data = Stream.exec_stream(trfs, data, self.optimize_stack)

        if return_torch:
            return data.to_torch(
                as_dict=as_dict,
                scale_keypoints=scale_keypoints,
                normalize=normalize,
                mean=mean,
                std=std,
            )

        return data
