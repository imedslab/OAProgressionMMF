from glob import glob
import logging

import numpy as np
import cv2
import nibabel as nib


logging.basicConfig()
logger = logging.getLogger('formats')
logger.setLevel(logging.DEBUG)


def png_series_to_numpy(pattern_fname_in, reverse=False):
    """

    Args:
        pattern_fname_in: str
            String or regexp compatible with `glob`.
        reverse: bool
            Whether to use reverse slice order.

    Returns:
        stack: [R, C, P] ndarray
    """
    fnames_in = sorted(glob(str(pattern_fname_in)))

    stack = [cv2.imread(fn, cv2.IMREAD_GRAYSCALE) for fn in fnames_in]
    stack = np.stack(stack, axis=2)
    if reverse:
        stack = stack[..., ::-1]
    return stack


def png_series_to_nifti(pattern_fname_in, fname_out, spacings=None, reverse=False,
                        ipr_to_ras=False):
    """

    Args:
        pattern_fname_in: str
            String or regexp compatible with `glob`.
        fname_out: str
            Full path to the output file.
        spacings: 3-tuple of float
            (pixel spacing in r, pixel spacing in c, slice thickness).
        reverse: bool
            Whether to use reverse slice order.
        ipr_to_ras: bool
            Whether to convert IPR+ (row-column-plane for sag) to RAS+ coordinates.

    """
    fnames_in = sorted(glob(str(pattern_fname_in)))

    stack = [cv2.imread(fn, cv2.IMREAD_GRAYSCALE) for fn in fnames_in]
    stack = np.stack(stack, axis=2)
    if reverse:
        stack = stack[..., ::-1]

    numpy_to_nifti(stack=stack, fname_out=fname_out, spacings=spacings,
                   ipr_to_ras=ipr_to_ras)


def nifti_to_png_series(fname_in, pattern_fname_out, reverse=False, ras_to_ipr=False):
    """

    Args:
        fname_in: str
            Full path to the input file.
        pattern_fname_out: str
            Must include `{i}`, which is to be substituted with the running index.
        reverse: bool
            Whether to use reverse slice order.
        ras_to_ipr: bool
            Whether to convert from RAS+ to IPR+ (row-column-plane for sag) coordinates.
    """
    stack, spacings = nifti_to_numpy(fname_in=fname_in, ras_to_ipr=ras_to_ipr)

    if reverse:
        stack = stack[..., ::-1]

    for i in range(stack.shape[-1]):
        fn = pattern_fname_out.format(i=i)
        cv2.imwrite(fn, stack[..., i])


def nifti_to_numpy(fname_in, ras_to_ipr=False, ras_to_irp=False):
    """

    Args:
        fname_in: str
            Full path to the input file.
        ras_to_ipr: bool
            Whether to convert from RAS+ to IPR+ (row-column-plane for sag) coordinates.
        ras_to_irp: bool
            Whether to convert from RAS+ to IRP+ (row-column-plane for cor) coordinates.

    Returns:
        stack: [d0, d1, d2] ndarray
        spacings: 3-tuple of float
            (pixel spacing in r, pixel spacing in c, slice thickness).

    """
    scan = nib.load(fname_in)
    stack = scan.get_fdata()
    spacings = [scan.affine[i, i] for i in range(3)]

    if ras_to_ipr:
        stack = np.moveaxis(stack, [2, 1, 0], [0, 1, 2])
        spacings = [-spacings[2], -spacings[1], spacings[0]]
    elif ras_to_irp:
        stack = np.moveaxis(stack, [2, 1, 0], [0, 2, 1])
        spacings = [-spacings[2], spacings[0], -spacings[1]]

    return stack, spacings


def numpy_to_nifti(stack, fname_out, spacings=None, ipr_to_ras=False, irp_to_ras=False):
    """

    Args:
        stack: (d0, d1, d2) ndarray
            Data array.
        fname_out:
            Full path to the output file.
        spacings: 3-tuple of float
            (pixel spacing in r, pixel spacing in c, slice thickness).
        ipr_to_ras: bool
            Whether to convert from IPR+ (row-column-plane for sag) to RAS+ coordinates.
        irp_to_ras: bool
            Whether to convert from IRP+ (row-column-plane for cor) to RAS+ coordinates.
    """

    if ipr_to_ras:
        stack = np.moveaxis(stack, [0, 1, 2], [2, 1, 0])
        affine = np.diag([1., -1., -1., 1.]).astype(np.float)
        if spacings is not None:
            affine[0, 0] = spacings[2]
            affine[1, 1] = -spacings[1]
            affine[2, 2] = -spacings[0]
    elif irp_to_ras:
        stack = np.moveaxis(stack, [0, 1, 2], [2, 0, 1])
        affine = np.diag([1., -1., -1., 1.]).astype(np.float)
        if spacings is not None:
            affine[0, 0] = spacings[1]
            affine[1, 1] = -spacings[2]
            affine[2, 2] = -spacings[0]
    else:
        affine = np.eye(4, dtype=np.float)
        if spacings is not None:
            affine[0, 0] = spacings[0]
            affine[1, 1] = spacings[1]
            affine[2, 2] = spacings[2]

    scan = nib.Nifti1Image(stack, affine=affine)
    nib.save(scan, fname_out)


def png_to_numpy(fname_in):
    """

    Args:
        fname_in: str
            Full path to the input file.

    Returns:
        image: [R, C] ndarray
    """
    image = cv2.imread(fname_in, cv2.IMREAD_GRAYSCALE)
    return image


def numpy_to_png(image, fname_out):
    """

    Args:
        image: [R, C] ndarray
        fname_out: str
            Full path to the output file.
    """
    cv2.imwrite(fname_out, image)
