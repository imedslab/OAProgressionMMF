"""
Dataset for prediction of knee osteoarthritis progression from multi-sequence MRI data.
Based on the OAI data. Created using independently developed selection routine.
"""

import tempfile
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import dicom2nifti
import pydicom
import nibabel as nib

from koafusion.datasets.oai import (release_to_prefix_var, release_to_visit_month)
from koafusion.datasets import fit_t2_map
from koafusion.various import numpy_to_nifti, nifti_to_numpy


logging.basicConfig()
logger = logging.getLogger("prepare")
logger.setLevel(logging.DEBUG)


def dicom_series_to_numpy_meta(dir_dicom):
    # INFO: configuration has to be changed here for it to work with parallel execution
    #       (i.e. joblib). This is due to the way `settings` are implemented in
    #       `dicom2nifti`.
    # dicom2nifti.settings.disable_validate_slice_increment()
    # dicom2nifti.settings.enable_resampling()
    # dicom2nifti.settings.set_resample_spline_interpolation_order(1)
    # dicom2nifti.settings.set_resample_padding(-1000)

    # Read, sort, and orient the 3D scan via `dicom2nifti`
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Convert to NIfTI and orient in LAS+
            dicom2nifti.convert_directory(dir_dicom, tmp, compression=True, reorient=True)
            # Load the image
            fname_nii_in = str(list(Path(tmp).glob("*.nii*"))[0])
            image = nib.load(fname_nii_in).get_fdata()
    except Exception as _:
        logger.warning(f"Skipped {dir_dicom}")
        return None

    # Read DICOM tags from the first slice
    path_dicom = str(list(Path(dir_dicom).glob("*"))[0])
    data = pydicom.dcmread(path_dicom)
    meta = dict()

    if hasattr(data, "ImagerPixelSpacing"):
        meta["pixel_spacing_0"] = float(data.ImagerPixelSpacing[0])
        meta["pixel_spacing_1"] = float(data.ImagerPixelSpacing[1])
    elif hasattr(data, "PixelSpacing"):
        meta["pixel_spacing_0"] = float(data.PixelSpacing[0])
        meta["pixel_spacing_1"] = float(data.PixelSpacing[1])
    else:
        msg = f"DICOM {path_dicom} does not contain spacing info"
        raise AttributeError(msg)

    meta["slice_thickness"] = float(data.SliceThickness)
    if hasattr(data, "BodyPartExamined"):
        meta["body_part"] = str.upper(data.BodyPartExamined)
    else:
        meta["body_part"] = "KNEE"

    if "RIGHT" in data.SeriesDescription:
        meta["side"] = "RIGHT"
    elif "LEFT" in data.SeriesDescription:
        meta["side"] = "LEFT"
    else:
        msg = f"DICOM {path_dicom} does not contain side info"
        raise AttributeError(msg)

    meta["series"] = str.upper(data.SeriesDescription)

    supported_seqs = ("SAG_3D_DESS", "COR_IW_TSE")
    meta["sequence"] = None
    for seq in supported_seqs:
        if seq in meta["series"]:
            meta["sequence"] = seq

    # Reorient the axes
    if meta["sequence"] == "SAG_3D_DESS":
        # Convert from LAS+ to IPR+
        image = np.moveaxis(image, [0, 1, 2], [2, 1, 0])
        image = np.flip(image)
    elif meta["sequence"] == "COR_IW_TSE":
        # Convert from LAS+ to IRP+
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        image = np.flip(image)
    else:
        logger.error(f"Unsupported series: {dir_dicom}, {meta['series']}")
        return None

    # Apply the corrections based on the DICOM tags
    if data.PhotometricInterpretation == "MONOCHROME1":
        image = image.max(initial=0) - image

    return image, meta


def dicom_series_to_t2_map_meta(dir_dicom):
    def assemble_4d_mese(img_dir):
        """Assembles 4D MESE stack from OAI SAG_T2_MAP sample.

        Args:
            img_dir (str): path to a directory that contains all the echoes for all
                           the slices for one MRI volume

        Returns:
            vol (4D ndarray (slices, echoes, height, width)): MESE image
            times (dict): echo times for each slice. keys are slice indices (int),
                          values are echo times (list of floats)
        """
        file_list = list(sorted(Path(img_dir).glob("*")))
        slice_location_list = []
        echo_num_list = []

        num_files = len(file_list)
        if num_files == 0:
            return None

        for i, f in enumerate(file_list):
            try:
                if i != num_files - 1:
                    dcm = pydicom.dcmread(f, stop_before_pixels=True)
                else:
                    dcm = pydicom.read_file(f, stop_before_pixels=False)
                slice_location_list.append(dcm.SliceLocation)
                echo_num_list.append(dcm.EchoNumbers)
            except Exception as e:
                logger.error(f"Error while assembling {img_dir}, {f}")
                logger.error(repr(e))
                return None

        slice_location_list = np.array(slice_location_list)
        echo_num_list = np.array(echo_num_list)

        slice_locations = np.sort(np.unique(slice_location_list))
        num_slices = len(slice_locations)
        echo_nums = np.sort(np.unique(echo_num_list))
        num_echoes = len(echo_nums)

        nrows = dcm.pixel_array.shape[0]
        ncols = dcm.pixel_array.shape[1]

        vol = np.empty((num_slices, nrows, ncols, num_echoes))
        tes = np.full((num_slices, num_echoes), np.nan)

        for i, f in enumerate(file_list):
            try:
                dcm = pydicom.dcmread(f, stop_before_pixels=False)
            except Exception as e:
                logger.error(f"Error while reading {f}")
                logger.error(repr(e))
                return None

            slice_idx = np.where(float(dcm.SliceLocation) == slice_locations)[0][0]
            echo_idx = np.where(int(dcm.EchoNumbers) == echo_nums)[0][0]

            vol[slice_idx, :, :, echo_idx] = dcm.pixel_array
            if dcm.EchoTime is not None:
                # Convert from ms to s
                tes[slice_idx, echo_idx] = float(dcm.EchoTime) / 1000.
            else:
                logger.warning(f"Missing EchoTime in {img_dir}, {f}")

        return vol, tes

    # Load data from disk
    t_ret = assemble_4d_mese(dir_dicom)  # (slice, rows, cols, TEs)
    if t_ret is None:
        return None
    else:
        t_vol, t_tes = t_ret

    # Estimate T2 map
    t_vol = t_vol.astype(np.float64)
    # t_tes = np.stack([te for te in t_tes.values()], axis=0).astype(np.float64)
    t_tes = t_tes.astype(np.float64)
    t2_map = fit_t2_map(t_vol, t_tes)
    t2_map = np.round(t2_map, decimals=6)

    # Read DICOM tags from the first slice
    path_dicom = str(list(Path(dir_dicom).glob("*"))[0])
    data = pydicom.dcmread(path_dicom)
    meta = dict()

    if hasattr(data, "ImagerPixelSpacing"):
        meta["pixel_spacing_0"] = float(data.ImagerPixelSpacing[0])
        meta["pixel_spacing_1"] = float(data.ImagerPixelSpacing[1])
    elif hasattr(data, "PixelSpacing"):
        meta["pixel_spacing_0"] = float(data.PixelSpacing[0])
        meta["pixel_spacing_1"] = float(data.PixelSpacing[1])
    else:
        msg = f"DICOM {path_dicom} does not contain spacing info"
        raise AttributeError(msg)

    meta["slice_thickness"] = float(data.SliceThickness)
    if hasattr(data, "BodyPartExamined"):
        meta["body_part"] = str.upper(data.BodyPartExamined)
    else:
        meta["body_part"] = "KNEE"

    if "RIGHT" in data.SeriesDescription:
        meta["side"] = "RIGHT"
    elif "LEFT" in data.SeriesDescription:
        meta["side"] = "LEFT"
    else:
        msg = f"DICOM {path_dicom} does not contain side info"
        raise AttributeError(msg)

    meta["series"] = str.upper(data.SeriesDescription)
    meta["sequence"] = "SAG_T2_MAP"

    # Reorient the axes
    if meta["sequence"] == "SAG_T2_MAP":
        # Convert from LAS+ to IPR+
        t2_map = np.moveaxis(t2_map, [0, 1, 2], [2, 0, 1])
    else:
        logger.error(f"Unsupported series: {dir_dicom}, {meta['series']}")
        return None

    return t2_map, meta


def preproc_compress_series(image_in, meta, path_stack):
    # Version 2
    if meta["sequence"] == "SAG_3D_DESS":
        image_tmp = image_in
        # Truncate least significant bits
        image_tmp = image_tmp.astype(np.uint16)
        image_tmp = image_tmp >> 3
        # Clip outlier intensities
        percents = np.percentile(image_tmp, q=(0., 99.9))
        if percents[1] > 255:
            raise ValueError(f"Out-of-range intensity after clipping: {path_stack}")
        image_tmp = np.clip(image_tmp, percents[0], percents[1])
        # Discretize
        image_tmp = image_tmp.astype(np.uint8)
        # Crop to exclude registration artefacts on margins
        margin = 16
        image_out = np.ascontiguousarray(image_tmp[margin:-margin, margin:-margin, :])

        return image_out, meta

    elif meta["sequence"] == "COR_IW_TSE":
        image_tmp = image_in
        # Truncate least significant bits
        image_tmp = image_tmp.astype(np.uint16)
        image_tmp = image_tmp >> 3
        # Clip outlier intensities
        percents = np.percentile(image_tmp, q=(0., 99.9))
        image_tmp = np.clip(image_tmp, percents[0], percents[1])
        # Discretize
        image_tmp = image_tmp.astype(np.uint16)
        # Crop to exclude registration artefacts on margins
        margin = 16
        image_out = np.ascontiguousarray(image_tmp[margin:-margin, margin:-margin, :])

        return image_out, meta

    elif meta["sequence"] == "SAG_T2_MAP":
        image_tmp = image_in
        # Crop to exclude registration artefacts on margins
        margin = 16
        image_out = np.ascontiguousarray(image_tmp[margin:-margin, margin:-margin, :])

        return image_out, meta

    else:
        raise NotImplementedError(f"Preprocessing is not available: {meta['sequence']}")


def handle_series(config, path_stack):
    if config.debug:
        print(path_stack)

    if "SAG_3D_DESS" in path_stack or "COR_IW_TSE" in path_stack:
        ret = dicom_series_to_numpy_meta(path_stack)
    elif "SAG_T2_MAP" in path_stack:
        ret = dicom_series_to_t2_map_meta(path_stack)
    else:
        raise ValueError("Error guessing sequence")
    if ret is None:
        logger.warning(f"Error reading: {path_stack}")
        return None
    else:
        image, meta = ret

    image, meta = preproc_compress_series(image, meta, path_stack)

    meta["release"], meta["patient"] = path_stack.split("/")[-4:-2]
    meta["visit_month"] = release_to_visit_month[meta["release"]]
    meta["prefix_var"] = release_to_prefix_var[meta["release"]]

    # Save image and mask
    protocol = f"{meta['body_part']}__{meta['side']}__{meta['sequence']}"
    dir_out = Path(config.dir_root_output, meta["patient"],
                   meta["visit_month"], protocol)
    dir_out.mkdir(exist_ok=True, parents=True)

    spacings = (meta["pixel_spacing_0"],
                meta["pixel_spacing_1"],
                meta["slice_thickness"])

    path_image = str(Path(dir_out, "image.nii.gz"))

    if meta["sequence"] == "SAG_3D_DESS":
        numpy_to_nifti(image, path_image, spacings=spacings, ipr_to_ras=True)
    elif meta["sequence"] == "COR_IW_TSE":
        numpy_to_nifti(image, path_image, spacings=spacings, irp_to_ras=True)
    elif meta["sequence"] == "SAG_T2_MAP":
        numpy_to_nifti(image, path_image, spacings=spacings, ipr_to_ras=True)
    else:
        numpy_to_nifti(image, path_image, spacings=spacings)

    sel = (
        "patient", "release", "visit_month", "prefix_var",
        "sequence", "body_part", "side",
        "pixel_spacing_0", "pixel_spacing_1", "slice_thickness",
    )
    return {k: meta[k] for k in sel}


@dataclass
class Config:
    dir_root_oai_mri: str
    path_csv_extract: str
    dir_root_output: str
    num_threads: int
    debug: bool = False
    ignore_cache: bool = False


cs = ConfigStore.instance()
cs.store(name="base", node=Config)


@hydra.main(config_path=None, config_name="base")
def main(config: Config) -> None:
    logger.warning(f"Only SAG_3D_DESS, COR_IW_TSE, SAG_T2_MAP are currently supported!")
    logger.warning(f"Only baseline (00m) images are processed!")

    path_df_images = Path(config.dir_root_output, "meta_images.csv")
    if path_df_images.exists() and not config.ignore_cache:
        logger.info("Cached version of the index exists")
        # _ = pd.read_csv(path_df_images)
    else:
        # OAI data path structure:
        #   root / examination / release / patient / date / barcode (/ slices)
        df_extract = pd.read_csv(config.path_csv_extract)

        # paths_stacks = [str(p) for p in Path(config.dir_root_oai_mri).glob("*/*/*/*/*")]
        # paths_stacks.sort(key=lambda x: int(x.split("/")[-3]))
        paths_stacks = [str(Path(config.dir_root_oai_mri, "00m", subdir))
                        for subdir in df_extract["Folder"].tolist()]
        paths_stacks.sort(key=lambda x: int(x.split("/")[-3]))

        if config.num_threads == 1:  # Debug mode
            if config.debug:
                # Single series
                metas = [handle_series(config, paths_stacks[0]), ]
            else:
                # All series in 1 thread
                metas = [handle_series(config, path_stack)
                         for path_stack in tqdm(paths_stacks)]
        else:
            # metas = Parallel(config.num_threads, backend="multiprocessing")(
            metas=Parallel(n_jobs=config.num_threads,
                           # backend="threading",
                           # batch_size=32,
                           verbose=10,
                           )(
                delayed(handle_series)(*[config, path_stack])
                for path_stack in tqdm(paths_stacks))

        # Merge meta information from different stacks
        tmp = defaultdict(list)
        for d in metas:
            if d is None:
                continue
            for k, v in d.items():
                tmp[k].append(v)
        df_images = pd.DataFrame.from_dict(tmp)
        dtypes = {"patient": str, "visit_month": str, "side": str, "sequence": str}
        df_images = df_images.astype(dtypes)

        # Save the meta
        df_images.to_csv(path_df_images, index=False)


if __name__ == "__main__":
    main()
