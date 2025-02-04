"""
Dataset for prediction of knee osteoarthritis progression from multi-modal data.
Based on the OAI data. Created using independently developed selection routine.
"""

import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

from koafusion.various import numpy_to_png, png_to_numpy


logging.basicConfig()
logger = logging.getLogger("prepare")
logger.setLevel(logging.DEBUG)


def png_to_numpy_meta(path_png):
    try:
        image = png_to_numpy(path_png)
    except Exception as _:
        logger.warning(f"Skipped {path_png}")
        return None

    meta = dict()
    meta["sequence"] = "XR_PA"
    meta["pixel_spacing_0"] = 0.195
    meta["pixel_spacing_1"] = 0.195
    meta["body_part"] = "KNEE"

    t = Path(path_png).stem
    meta["patient"] = t.split("_")[0]
    meta["visit_month"] = f"0{t.split('_')[1]}m"
    meta["side"] = {"L": "LEFT", "R": "RIGHT"}[t.split("_")[2]]

    return image, meta


def handle_series(config, path_image):
    if config.debug:
        print(path_image)

    ret = png_to_numpy_meta(path_image)
    if ret is None:
        logger.warning(f"Error reading: {path_image}")
        return None
    else:
        image, meta = ret

    # Save image
    protocol = f"{meta['body_part']}__{meta['side']}__{meta['sequence']}"
    dir_out = Path(config.dir_root_output, meta["patient"],
                   meta["visit_month"], protocol)
    dir_out.mkdir(exist_ok=True, parents=True)

    path_image = str(Path(dir_out, "image.png"))
    numpy_to_png(image, path_image)

    sel = (
        "patient", "visit_month",
        "sequence", "body_part", "side",
        "pixel_spacing_0", "pixel_spacing_1",
    )
    return {k: meta[k] for k in sel}


@dataclass
class Config:
    dir_root_mipt_xr: str
    dir_root_output: str
    num_threads: int
    debug: bool = False
    ignore_cache: bool = False


cs = ConfigStore.instance()
cs.store(name="base", node=Config)


@hydra.main(config_path=None, config_name="base")
def main(config: Config) -> None:
    path_df_images = Path(config.dir_root_output, "meta_images.csv")
    if path_df_images.exists() and not config.ignore_cache:
        logger.info("Loading from the cache")
        df_images = pd.read_csv(path_df_images)
    else:
        # Oulu data path structure:
        #   root / (patient_visitmonth_side.png)
        paths_images = [str(p) for p in Path(config.dir_root_mipt_xr).glob("*")]
        paths_images.sort()

        logger.warning("Selected only the baseline images!")
        logger.warning(f"Number of scans:")
        logger.warning(f"- before selection: {len(paths_images)}")
        paths_images = [p for p in paths_images if "_00_" in p.split("/")[-1]]
        logger.warning(f"- after selection: {len(paths_images)}")

        if config.num_threads == 1:  # Debug mode
            if config.debug:
                # Single series
                metas = [handle_series(config, paths_images[0]), ]
            else:
                # All series in 1 thread
                metas = [handle_series(config, path_stack)
                         for path_stack in tqdm(paths_images)]
        else:
            metas = Parallel(config.num_threads, backend="multiprocessing")(
                delayed(handle_series)(*[config, path_stack])
                for path_stack in tqdm(paths_images))

        # Merge meta information from different stacks
        tmp = defaultdict(list)
        for d in metas:
            if d is None:
                continue
            for k, v in d.items():
                tmp[k].append(v)
        df_images = pd.DataFrame.from_dict(tmp)
        df_images.to_csv(path_df_images, index=False)

    # Save the meta
    df_out = df_images
    path_output_meta = Path(config.dir_root_output, "meta_base.csv")
    df_out = df_out.sort_values(by=["patient", "visit_month", "side", "sequence"])
    df_out.to_csv(path_output_meta, index=False)


if __name__ == "__main__":
    main()
