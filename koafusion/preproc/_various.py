import random
import logging
import numpy as np
import torch


logging.basicConfig()
logger = logging.getLogger('preproc_custom')
logger.setLevel(logging.DEBUG)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        if mask is not None:
            mask = mask.astype(np.float32)
        return img, mask


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, *args):
        return [(a * self.std + self.mean) for a in args]


class PercentileClip:
    """Change the histogram of image by doing global contrast normalization."""
    def __init__(self, cut_min=0.5, cut_max=99.5):
        """
        cut_min - lowest percentile which is used to cut the image histogram
        cut_max - highest percentile
        """
        self.cut_min = cut_min
        self.cut_max = cut_max

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)
        lim_low, lim_high = np.percentile(img, [self.cut_min, self.cut_max])
        img = np.clip(img, lim_low, lim_high)

        img -= lim_low
        img /= img.max()

        img = img.astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)

        return img, mask


class NumpyToTensor(object):
    def __call__(self, *args):
        if len(args) > 1:
            return [torch.from_numpy(e.copy()) for e in args]
        else:
            return torch.from_numpy(args[0].copy())


class TensorToNumpy(object):
    def __call__(self, *args):
        if len(args) > 1:
            return [e.numpy() for e in args]
        else:
            return args[0].numpy()


class TensorToDevice(object):
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, *args):
        return [e.to(self.device) for e in args]
