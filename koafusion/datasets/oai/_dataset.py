import os
import logging
from pathlib import Path
from functools import reduce
from collections import defaultdict

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from koafusion.various import nifti_to_numpy, png_to_numpy


logging.basicConfig()
logger = logging.getLogger("dataset")
logger.setLevel(logging.DEBUG)


VARS_DTYPES = {
    # exam
    "patient": str, "release": str, "prefix_var": str, "visit_month": str,
    "visit": int, "side": str,
    # clinical
    "P02SEX": str, "P02RACE": str, "V00SITE": str, "AGE": int, "P01BMI": float,
    "XRKL": int,
    "XROSFL": int, "XROSFM": int, "XROSTL": int, "XROSTM": int, "XRJSL": float, "XRJSM": float,
    "XRSCFL": int, "XRSCFM": int, "XRSCTL": int, "XRSCTM": int, "XRATTL": int, "XRATTM": int,
    "WOMADL-": float, "WOMKP-": float, "WOMSTF-": float, "WOMTS-": float,
    'KP-30CV': int, "KRS-12": int, "P01INJ-": int, "P01KSURG-": int, "P01KRS-": int,
    "P01ART-": int, "P01ART-INJ": int, "P01MEN-": int, "P01MEN-INJ": int,
    "P01LR-": int, "P01OTSURG-": int, "P01OTS-INJ": int,
    # custom
    "tiulpin2019_kl_diff": int,
    "tiulpin2019_prog": int,
    "tiulpin2019_sel": int,
    "prog_kl_12": int, "prog_kl_24": int, "prog_kl_36": int,
    "prog_kl_48": int, "prog_kl_72": int, "prog_kl_96": int,
    "panfilov_sel_kl_12": int, "panfilov_sel_kl_24": int, "panfilov_sel_kl_36": int,
    "panfilov_sel_kl_48": int, "panfilov_sel_kl_72": int, "panfilov_sel_kl_96": int,
    "reason_kl_12": str, "reason_kl_24": str, "reason_kl_36": str,
    "reason_kl_48": str, "reason_kl_72": str, "reason_kl_96": str,
    # MRI
    "body_part": str,  "sequence": str,
    "pixel_spacing_0": float, "pixel_spacing_1": float, "slice_thickness": float,
    "path_image": str,
    # generated
    "exam_knee_id": str,
}


MODALS = {
    "clin": {"sub-dir": "OAI_Clin_prep", "kind": "clin"},
    "sag_3d_dess": {"sub-dir": "OAI_SAG_3D_DESS_prep", "kind": "mri"},
    "cor_iw_tse": {"sub-dir": "OAI_COR_IW_TSE_prep", "kind": "mri"},
    "sag_t2_map": {"sub-dir": "OAI_SAG_T2_MAP_prep", "kind": "mri"},
    "xr_pa": {"sub-dir": "OAI_XR_PA_prep", "kind": "xr"},
}


def _prepare_meta_clin(df_i):
    vars_exam = [
        "patient", "visit_month", "side",
    ]
    vars_clin = [
        # basic and demographics
        "P02SEX", "P02RACE", "V00SITE", "AGE", "P01BMI",
        # KL-grading
        "XRKL",
        # OARSI-grading
        "XROSFL", "XROSFM", "XROSTL", "XROSTM", "XRJSL", "XRJSM",
        "XRSCFL", "XRSCFM", "XRSCTL", "XRSCTM", "XRATTL", "XRATTM",
        # WOMAC
        "WOMADL-", "WOMKP-", "WOMSTF-", "WOMTS-",
        # injury, surgery
        'KP-30CV', "KRS-12", "P01INJ-", "P01KSURG-", "P01KRS-",
        "P01ART-", "P01ART-INJ", "P01MEN-", "P01MEN-INJ",
        "P01LR-", "P01OTSURG-", "P01OTS-INJ",
        # ... others are skipped
    ]
    vars_targets = [
        "tiulpin2019_kl_diff",
        "tiulpin2019_prog",
        "tiulpin2019_sel",
        "panfilov_sel_kl_12", "panfilov_sel_kl_24", "panfilov_sel_kl_36",
        "panfilov_sel_kl_48", "panfilov_sel_kl_72", "panfilov_sel_kl_96",
        "prog_kl_12", "prog_kl_24", "prog_kl_36",
        "prog_kl_48", "prog_kl_72", "prog_kl_96",
        "reason_kl_12", "reason_kl_24", "reason_kl_36",
        "reason_kl_48", "reason_kl_72", "reason_kl_96",
    ]
    df_o = df_i.copy()
    df_o = df_o.loc[:, vars_exam + vars_clin + vars_targets]
    return df_o


def _prepare_meta_mri(df_i):
    vars_exam = [
        "patient", "visit_month", "side",
    ]
    vars_protocol = [
        "body_part", "sequence",
        "pixel_spacing_0", "pixel_spacing_1", "slice_thickness",
    ]
    df_o = df_i.copy()
    df_o = df_o.loc[:, vars_exam + vars_protocol]
    return df_o


def _prepare_meta_xr(df_i):
    vars_exam = [
        "patient", "visit_month", "side",
    ]
    vars_protocol = [
        "body_part", "sequence",
        "pixel_spacing_0", "pixel_spacing_1",
    ]
    df_o = df_i.copy()
    df_o = df_o.loc[:, vars_exam + vars_protocol]
    return df_o


def index_from_path_oai(path_root, modals_all, ignore_cache=False):
    """Build an aggregated index of all present modalities.

    Args:
        path_root (str, Path): path to data root
        modals_all (list): list of modalities
        ignore_cache (bool): whether to recreate index

    Returns:
        df_agg (DataFrame): aggregated index
    """
    fn_meta_agg = Path(path_root, "meta_agg_oai.csv")
    modals = {k: v for k, v in MODALS.items() if k in modals_all}

    if fn_meta_agg.exists() and not ignore_cache:
        df_agg = pd.read_csv(fn_meta_agg, header=[0, 1], index_col=None)
        # Convert dtypes of multi-index columns
        for c in df_agg.columns:
            df_agg[c] = df_agg[c].astype(VARS_DTYPES[c[1]])
    else:
        dfs_modal = dict()

        for m_name, m_prop in modals.items():
            logger.info(f"Reading modality {m_name}")
            t_p = Path(path_root, m_prop["sub-dir"])
            if not t_p.exists():
                logger.warning(f"Not found in {t_p}")
                continue

            if m_prop["kind"] == "clin":
                fn_meta_base = Path(t_p, "meta_base.csv")
                t_df = pd.read_csv(fn_meta_base, dtype=VARS_DTYPES, index_col=None)
                t_df = _prepare_meta_clin(t_df)
            elif m_prop["kind"] == "mri":
                fn_meta_base = Path(t_p, "meta_images.csv")
                t_df = pd.read_csv(fn_meta_base, dtype=VARS_DTYPES, index_col=None)
                t_df = _prepare_meta_mri(t_df)
            elif m_prop["kind"] == "xr":
                fn_meta_base = Path(t_p, "meta_images.csv")
                t_df = pd.read_csv(fn_meta_base, dtype=VARS_DTYPES, index_col=None)
                t_df = _prepare_meta_xr(t_df)
            else:
                raise ValueError(f"Unknown kind {m_prop['kind']}")

            logger.info(f"{len(t_df)} metadata records found")

            # Assign unique markers to samples
            t_col = [f"{e['patient']}__{e['visit_month']}__{e['side']}"
                     for _, e in t_df.iterrows()]
            t_df["exam_knee_id"] = t_col

            # Find all imaging
            if m_prop["kind"] in ("mri", "xr"):
                if m_prop["kind"] == "mri":
                    t_name = "image.nii.gz"
                else:
                    t_name = "image.png"

                t_fns_image = list(Path(t_p).glob(f"**/{t_name}"))
                logger.info(f"{len(t_fns_image)} images found")

                if len(t_fns_image) != len(t_df):
                    logger.warning("Number of images does not match with the metadata")

                # Compose full path to image files
                t_col = []
                for idx in range(len(t_df)):
                    t_entry = Path(t_p,
                                   t_df["patient"].iloc[idx],
                                   t_df["visit_month"].iloc[idx],
                                   (f"{t_df['body_part'].iloc[idx]}"
                                    f"__{t_df['side'].iloc[idx]}"
                                    f"__{t_df['sequence'].iloc[idx]}"),
                                   t_name)
                    if os.path.exists(t_entry):
                        t_col.append(str(t_entry))
                    else:
                        t_col.append("")
                t_df["path_image"] = t_col
                t_df = t_df[t_df["path_image"] != ""]
                logger.info(f"{len(t_df)} metadata records with images available")

            dfs_modal[m_name] = t_df

        # Aggregate meta from all modalities
        for m_name, t_df in dfs_modal.items():
            t_df = t_df.set_index(keys=["patient", "visit_month", "side", "exam_knee_id"])

            # Create multi-index in columns to avoid name collisions
            if modals[m_name]["kind"] == "clin":
                # Append for clinical, to match with the removed index vars later
                t_df.columns = pd.MultiIndex.from_product([["-"], t_df.columns])
            elif modals[m_name]["kind"] in ("mri", "xr"):
                # Prepend for imaging
                t_df.columns = pd.MultiIndex.from_product([[m_name], t_df.columns])
            dfs_modal[m_name] = t_df

        df_agg = reduce(lambda df_l, df_r: pd.merge(
            df_l, df_r, left_index=True, right_index=True, how="inner"), dfs_modal.values())

        # Sort the records
        df_agg = (df_agg
                  .reset_index(col_level=1, col_fill="-")
                  .sort_values(by=("-", "exam_knee_id")))
                  # .drop("index", axis=1))
        df_agg.to_csv(fn_meta_agg, index=False)

    return df_agg


class DatasetOAI3d(Dataset):
    def __init__(self, df_meta, modals, transforms=None, **kwargs):
        logger.warning(f"Redundant dataset init arguments:\n{repr(kwargs)}")
        logger.warning("Images are flipped:")
        logger.warning("SAG_3D_DESS, COR_IW_TSE, SAG_T2_MAP, XR_PA -> LEFT side")

        self.df_meta = df_meta
        self.modals = modals
        self.transforms = transforms

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        item = dict(self.df_meta.iloc[idx])

        # INFO: experimental
        clin_vec = []
        t = item[("-", "AGE")] ; t_proc = (t - 60.945) / 9.209 ; clin_vec.append(t_proc)
        t = item[("-", "P02SEX")]
        if t == "MALE":
            clin_vec.extend([1, 0])
        else:
            clin_vec.extend([0, 1])
        t = item[("-", "P01BMI")] ; t_proc = (t - 28.734) / 4.917 ; clin_vec.append(t_proc)
        # t = item[("-", "XRKL")] ; t_proc = [0., ] * 5 ; t_proc[t] = 1. ; clin_vec.extend(t_proc)
        t = item[("-", "P01INJ-")] ; t_proc = [0., ] * 2 ; t_proc[t] = 1. ; clin_vec.extend(t_proc)
        t = item[("-", "P01KSURG-")] ; t_proc = [0., ] * 2 ; t_proc[t] = 1. ; clin_vec.extend(t_proc)
        t = item[("-", "WOMTS-")] ; t_proc = (t - 10.940) / 14.573 ; clin_vec.append(t_proc)
        item["clin_vec"] = torch.Tensor(np.asarray(clin_vec))
        # INFO: ^^^^

        for m in self.modals:
            # INFO: experimental
            if m == "clin":
                item[f"image__{m}"] = item["clin_vec"].unsqueeze(dim=0)  # to (CH, D)
                continue
            # INFO: ^^^^

            t_side = item[("-", "side")]
            t_seq = item[(m, "sequence")]
            t_path = item[(m, "path_image")]

            t_image = self.read_image(t_path, t_seq)
            if t_seq == "SAG_3D_DESS":
                t_shape_curr = np.array(t_image.shape[-3:])
                t_shape_min = np.array([320, 320, 128])
            elif t_seq == "COR_IW_TSE":
                t_shape_curr = np.array(t_image.shape[-3:])
                t_shape_min = np.array([320, 320, 32])
            elif t_seq == "SAG_T2_MAP":
                t_shape_curr = np.array(t_image.shape[-3:])
                t_shape_min = np.array([320, 320, 25])
            elif t_seq == "XR_PA":
                t_shape_curr = np.array(t_image.shape[-2:])
                t_shape_min = np.array([700, 700])
            else:
                raise ValueError(f"Unsupported sequence: {t_seq}")
            if np.any(t_shape_curr < t_shape_min):
                logger.error(f"{t_path} is {t_shape_curr}, expected >{t_shape_min}")

            # Flip the image to LEFT knee orientation.
            # Dims are (ch, row, col, plane) for 3D, (ch, row, col) for 2D
            if t_seq == "SAG_3D_DESS":
                if t_side == "RIGHT":
                    t_image = np.flip(t_image, axis=-1)
            elif t_seq == "COR_IW_TSE":
                if t_side == "RIGHT":
                    t_image = np.flip(t_image, axis=2)
            elif t_seq == "SAG_T2_MAP":
                if t_side == "RIGHT":
                    t_image = np.flip(t_image, axis=-1)
            elif t_seq == "XR_PA":
                if t_side == "RIGHT":
                    t_image = np.flip(t_image, axis=2)
            else:
                raise ValueError(f"Unsupported sequence: {t_seq}")

            # Apply transformations
            t_transfs = self.transforms[m]
            if t_transfs is not None:
                for t_fn in t_transfs:
                    if hasattr(t_fn, "randomize"):
                        t_fn.randomize()
                    t_image = t_fn(t_image)

            # Compose sample
            item[f"image__{m}"] = t_image

        # Target is defined in `sources`
        item["target"] = np.asarray([item[("-", "target")], ])

        return item

    @staticmethod
    def read_image(path_file, sequence):
        if sequence == "SAG_3D_DESS":
            image, spacings = nifti_to_numpy(path_file, ras_to_ipr=True)
        elif sequence == "COR_IW_TSE":
            image, spacings = nifti_to_numpy(path_file, ras_to_irp=True)
        elif sequence == "SAG_T2_MAP":
            image, spacings = nifti_to_numpy(path_file, ras_to_ipr=True)
        elif sequence == "XR_PA":
            image = png_to_numpy(path_file)
        else:
            raise ValueError(f"Unsupported sequence: {sequence}")
        return image.reshape((1, *image.shape))

    def describe(self, num_samples=None):
        info = defaultdict(list)

        if num_samples is None:
            num_samples = len(self)

        for i in tqdm(range(num_samples), total=num_samples):
            item = self.__getitem__(i)

            for m in self.modals:
                if np.sum(np.sum(item[f"image__{m}"], axis=(0, 1, 2)) == 0) >= 1:
                    logger.error(f"Multiple zero slices in {item[(m, 'path_image')]}")

                if np.sum(np.any(np.isnan(item[f"image__{m}"]), axis=(0, 1, 2))) >= 1:
                    logger.error(f"Multiple NaN slices in {item[(m, 'path_image')]}")
            info["targets"].append(item["target"])

        # for m in self.modals:
        #     info[f"image__{m}__mean_est"] = np.mean(info[f"image__{m}__means"])
        #     info[f"image__{m}__std_est"] = np.mean(info[f"image__{m}__stds"])
        #     del info[f"image__{m}__means"]
        #     del info[f"image__{m}__stds"]

        u, c = np.unique(info["targets"], return_counts=True)
        info["target_counts"] = dict(zip(u, c))
        del info["targets"]

        logger.info("Dataset statistics:")
        logger.info(sorted(info.items()))

    def test_all_readable(self, n_jobs=24, verbose=5):
        def dummy(idx):
            try:
                _ = self[idx]
            except Exception as e:
                msg = f"{type(e)} while reading {dict(self.df_meta.iloc[idx])}"
                logger.error(msg)
                raise e

        tasks = []

        for i in tqdm(range(len(self))):
            tasks.append(delayed(dummy)(i))

        _ = Parallel(n_jobs=n_jobs, verbose=verbose)(tasks)
        logger.info("Reading completed")
