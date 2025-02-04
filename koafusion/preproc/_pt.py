import random
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


class PTPercentileClip(object):
    def __init__(self, cut_min, cut_max, to_unit_range=False):
        self.cut_min = cut_min
        self.cut_max = cut_max
        self.to_unit_range = to_unit_range

    def _image_percentiles(self, input_, percentiles):
        """Find the percentiles of a tensor along the first dimension.

        Parameters
        ----------
        input_: (B, D0, ...) tensor
        percentiles: list-like

        References
        ----------
        Based on https://github.com/aliutkus/torchpercentile.
        """
        input_dtype = input_.dtype
        input_shape = input_.shape

        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)

        input_ = input_.double()
        input_ = input_.unsqueeze(0)

        percentiles = percentiles.to(input_.device).double()
        input_ = input_.view(input_.shape[0], -1)

        in_sorted, in_argsort = torch.sort(input_, dim=1)
        positions = percentiles * (input_.shape[1] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input_.shape[1] - 1] = input_.shape[1] - 1

        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled

        d0 = in_sorted[:, floored.long()] * weight_floored[None, :]
        d1 = in_sorted[:, ceiled.long()] * weight_ceiled[None, :]

        result = (d0 + d1).reshape(-1)
        return result.type(input_dtype)

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (D0, ...) 1D-nD Tensor
        mask: (D0, ...) 1D-nD Tensor
        """
        image = image.float()

        limits = self._image_percentiles(image, percentiles=[self.cut_min, self.cut_max])
        image = image.clamp(min=limits[0].item(), max=limits[1].item())

        if self.to_unit_range:
            image = image.sub(limits[0])
            image = image.div(limits[1] - limits[0])

        if mask is not None:
            return image, mask
        else:
            return image


class PTToUnitRange(object):
    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (D0, ...) 1D-nD Tensor
        mask: (D0, ...) 1D-nD Tensor
        """
        image = image.float()

        val_min = torch.min(image)
        val_max = torch.max(image)

        image = image.sub(val_min)
        image = image.div(val_max - val_min)

        if mask is not None:
            return image, mask
        else:
            return image


class PTNormalize(object):
    def __init__(self, mean, std):
        if isinstance(mean, (int, float)):
            self.mean = [mean, ]
        else:
            self.mean = mean

        if isinstance(std, (int, float)):
            self.std = [std, ]
        else:
            self.std = std

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (CH, D0, ...) 2D-nD Tensor
        mask: (CH, D0, ...) 2D-nD Tensor
        """
        shape_in = image.shape

        image = image.reshape(shape_in[0], -1)

        mean = torch.tensor(self.mean)[:, None]
        std = torch.tensor(self.std)[:, None]

        image = (image - mean) / std
        image = image.reshape(*shape_in).float()

        if mask is not None:
            mask = mask.float()
            return image, mask
        else:
            return image


class PTDenormalize(object):
    def __init__(self, mean, std):
        if isinstance(mean, (int, float)):
            self.mean = [mean, ]
        else:
            self.mean = mean

        if isinstance(std, (int, float)):
            self.std = [std, ]
        else:
            self.std = std

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (CH, D0, ...) 2D-nD Tensor
        mask: (CH, D0, ...) 2D-nD Tensor
        """
        shape_in = image.shape

        image = image.view(shape_in[0], -1)

        mean = torch.tensor(self.mean)[:, None]
        std = torch.tensor(self.std)[:, None]

        image = (image * std) + mean
        image = image.view(*shape_in).float()

        if mask is not None:
            mask = mask.float()
            return image, mask
        else:
            return image


class PTInterpolate(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (B, CH, D0, ...) 3D-5D Tensor
        mask: (B, CH, D0, ...) 3D-5D Tensor
        """
        t_mode = {3: "linear", 4: "bilinear", 5: "trilinear"}[image.ndim]

        image = torch.nn.functional.interpolate(image, scale_factor=self.scale_factor,
                                                recompute_scale_factor=True,
                                                align_corners=False,
                                                mode=t_mode)
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask, scale_factor=self.scale_factor,
                                                   recompute_scale_factor=True,
                                                   align_corners=False,
                                                   mode="nearest")
            return image, mask
        else:
            return image


class PTGammaCorrection(object):
    def __init__(self, gamma_range=(0.5, 2.0), prob=0.5, clip_to_unit=False):
        self.gamma_range = gamma_range
        self.prob = prob
        self.clip_to_unit = clip_to_unit

        self.state = dict()
        self.randomize()

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (D0, ...) 1D-nD Tensor
        mask: (D0, ...) 1D-nD Tensor
        """
        if self.state["p"] < self.prob:
            image = torch.pow(image, (1. / self.state["gamma"]))
            if self.clip_to_unit:
                image = torch.clamp(image, min=0., max=1.)

        if mask is not None:
            return image, mask
        else:
            return image

    def randomize(self):
        self.state["p"] = random.random()
        self.state["gamma"] = random.uniform(*self.gamma_range)


class PTReproject(object):
    def __init__(self, dims_in, dims_out):
        self.dims_in = dims_in
        self.dims_out = dims_out

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (D0, ...) 1D-nD Tensor
        mask: (D0, ...) 1D-nD Tensor
        """
        image = torch.movedim(image, source=self.dims_in, destination=self.dims_out)

        if mask is not None:
            mask = torch.movedim(mask, source=self.dims_in, destination=self.dims_out)
            return image, mask
        else:
            return image


class PTRotate3DInSlice(object):
    def __init__(self, degree_range=(-30., 30.), prob=0.5):
        self.theta_range = torch.deg2rad(torch.Tensor(degree_range))
        self.prob = prob

        self.state = dict()
        self.randomize()

    @staticmethod
    def rotation_matrix(theta):
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (CH, R, C, S) 4D Tensor
        mask: (CH, R, C, S) 4D Tensor
        """
        if image.dim() == 4:  # multichannel 3D
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")

        if self.state["p"] < self.prob:
            image = rearrange(image, "ch r c s -> s ch r c")
            dtype = image.dtype
            shape_in = image.shape
            # the code below needs (B, CH, D0, D1)
            rot_mat = self.rotation_matrix(self.state["theta"])
            rot_mat = rot_mat[None, ...].type(dtype).repeat(shape_in[0], 1, 1)
            # affine_grid: N*2*3 / N*3*4 + N*C*D0*D1[*D2] -> N*D0*D1*2
            grid = F.affine_grid(rot_mat, image.size(), align_corners=False).type(dtype)
            image = F.grid_sample(image, grid, align_corners=False)
            image = rearrange(image, "s ch r c -> ch r c s")

            if mask is not None:
                mask = rearrange(mask, "ch r c s -> s ch r c")
                mask = F.grid_sample(mask, grid=grid, mode="nearest")
                mask = rearrange(mask, "s ch r c -> ch r c s")

        if mask is not None:
            return image, mask
        else:
            return image

    def randomize(self):
        self.state["p"] = random.random()
        self.state["theta"] = random.uniform(*self.theta_range)


class PTRotate2D(object):
    def __init__(self, degree_range=(-30., 30.), prob=0.5):
        self.theta_range = torch.deg2rad(torch.Tensor(degree_range))
        self.prob = prob

        self.state = dict()
        self.randomize()

    @staticmethod
    def rotation_matrix(theta):
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        image: (CH, R, C) 3D Tensor
        mask: (CH, R, C) 3D Tensor
        """
        if image.dim() == 3:  # multichannel 2D
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")

        if self.state["p"] < self.prob:
            image = rearrange(image, "ch r c -> 1 ch r c")
            dtype = image.dtype
            shape_in = image.shape
            # the code below needs (B, CH, D0, D1)
            rot_mat = self.rotation_matrix(self.state["theta"])
            rot_mat = rot_mat[None, ...].type(dtype).repeat(shape_in[0], 1, 1)
            # affine_grid: N*2*3 / N*3*4 + N*C*D0*D1[*D2] -> N*D0*D1*2
            grid = F.affine_grid(rot_mat, image.size(), align_corners=False).type(dtype)
            image = F.grid_sample(image, grid, align_corners=False)
            image = rearrange(image, "1 ch r c -> ch r c")

            if mask is not None:
                mask = rearrange(mask, "ch r c -> 1 ch r c")
                mask = F.grid_sample(mask, grid=grid, mode="nearest")
                mask = rearrange(mask, "1 ch r c -> ch r c")

        if mask is not None:
            return image, mask
        else:
            return image

    def randomize(self):
        self.state["p"] = random.random()
        self.state["theta"] = random.uniform(*self.theta_range)
