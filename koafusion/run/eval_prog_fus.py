import gc
import os
import logging
import time
import pickle
import functools
from pathlib import Path
from collections import defaultdict

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
# import torchsummary
import thop  # see also: https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

from koafusion.datasets import prepare_datasets_loaders
from koafusion.models import dict_models
from koafusion import preproc
from koafusion.various import CheckpointHandler, set_ultimate_seed
from koafusion.various._metrics_stat_anlys import calc_metrics_v2 as calc_metrics

from captum.attr import FeatureAblation


# Fix to PyTorch multiprocessing issue: "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('eval')
logger.setLevel(logging.DEBUG)

set_ultimate_seed()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ProgressionPrediction(object):
    def __init__(self, *, config):
        self.config = config

        # Build a list of folds to run on
        if config.testing.folds.idx == -1:
            self.fold_idcs = list(range(config.training.folds.num))
        else:
            self.fold_idcs = [config.testing.folds.idx, ]
        if config.testing.folds.ignore is not None:
            for g in config.testing.folds.ignore:
                self.fold_idcs = [i for i in self.fold_idcs if i != g]

        if self.config.model.downscale:
            logger.warning("Downscaling is enabled!")

        # Initialize datasets, loaders, and transforms
        t = prepare_datasets_loaders(config=config, fold_idx=0)
        self.datasets = t[0]
        self.data_loaders = t[1]

        if self.config.testing.describe_data:
            self.describe_data()
            quit()

        # Init experiment paths
        self.path_weights = Path(self.config.path_experiment_root, "weights")
        if "sag_t2_map" in self.config.data.modals_all:
            sel_knee = "incid"
        else:
            sel_knee = "all"
        self.path_logs = Path(self.config.path_experiment_root, "logs_eval", sel_knee)

        self.tb = SummaryWriter(self.path_logs)

    def describe_data(self):
        # Describe distribution of the dataset variables
        def _describe_df(df):
            print(df.head())
            fs_subj_cont = (("-", "AGE"),
                            ("-", "P01BMI"))
            fs_subj_cat = (("-", "P02SEX"),)
            fs_knee_cont = (("-", "WOMTS-"),)
            fs_knee_cat = (("-", "XRKL"),
                           ("-", "P01INJ-"),
                           ("-", "P01KSURG-"))
            fs_knee_target = (("-", "target"),)

            df_subj = df.drop_duplicates(subset=[("-", "patient"),
                                                 ("-", "visit_month")])
            df_knee = df.copy()

            for f in fs_subj_cont:
                print("\n", f)
                print(df_subj[f].describe())
            for f in fs_subj_cat:
                print("\n", f)
                print(df_subj[f].value_counts(ascending=True))
            for f in fs_knee_cont:
                print("\n", f)
                print(df_knee[f].describe())
            for f in (("-", "WOMTS-"),):
                print("\n", f)
                print(pd.value_counts(pd.cut(df_knee[f],
                                             bins=[-10, -1, 10, 100],
                                             right=True)))
            for f in fs_knee_cat:
                print("\n", f)
                print(df_knee[f].value_counts())
            for f in fs_knee_target:
                print("\n", f)
                print(df_knee[f].value_counts())

        ks = list(self.datasets.keys())
        subsets = ("sel",)  # ACTION: select
        # subsets = ("test",)
        logger.info(f"Describing subsets: {repr(subsets)}")
        for k in ks:
            for s in subsets:
                print(f"---- Dataset: {k}, subset: {s}")
                _describe_df(self.datasets[k][s].df_meta)

    def eval(self):
        paths_cache = {
            "raw_fold-w": Path(self.path_logs, f"eval_fus_raw_foldw.pkl"),
            "raw_ens": Path(self.path_logs, f"eval_fus_raw_ens.pkl"),
            "metrics_fold-w": Path(self.path_logs, f"eval_fus_metrics_foldw.pkl"),
            "metrics_ens": Path(self.path_logs, f"eval_fus_metrics_ens.pkl"),
        }

        # Only raw predicts are restored from cache
        raw_foldw = dict()
        raw_ens = dict()

        # Fold-w predicts
        if self.config.testing.use_cached and paths_cache["raw_fold-w"].exists():
            logger.info(f"Reading from {paths_cache['raw_fold-w']}")
            with open(paths_cache["raw_fold-w"], "rb") as f:
                raw_foldw = pickle.load(f)
        else:
            for fold_idx in self.fold_idcs:
                # Init fold-wise paths
                paths_weights_fold = dict()
                paths_weights_fold["prog"] = \
                    Path(self.path_weights, "prog", f"fold_{fold_idx}")

                handlers_ckpt = dict()
                handlers_ckpt["prog"] = CheckpointHandler(paths_weights_fold["prog"])

                paths_ckpt_sel = dict()
                paths_ckpt_sel["prog"] = handlers_ckpt["prog"].get_last_ckpt()

                # Initialize and configure model
                models = dict()
                models["prog"] = dict_models[self.config.model.name](
                    config=self.config.model,
                    path_weights=paths_ckpt_sel["prog"])

                models["prog"] = models["prog"].to(device)
                if self.config.testing.profile == "none":
                    models["prog"] = nn.DataParallel(models["prog"])
                # Switch to eval regime
                models["prog"] = models["prog"].eval()

                # Eval model on subset
                t = self.eval_epoch(models=models)
                raw_foldw[fold_idx] = t

            # Save fold-w to cache
            with open(paths_cache["raw_fold-w"], "wb") as f:
                pickle.dump(raw_foldw, f, pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved fold-w raw predicts to: {paths_cache['raw_fold-w']}")

        def _pprint_metrics(d):
            for k, v in d.items():
                if k == "roc_curve":
                    continue
                logger.info(f"{k}: {np.round(v, decimals=3)}")

        # Metrics fold-w
        if self.config.testing.metrics_foldw:
            metrics_foldw = dict()
            for fold_idx in self.fold_idcs:
                if fold_idx in raw_foldw:
                    metrics_foldw[fold_idx] = calc_metrics(
                        prog_target=np.asarray(raw_foldw[fold_idx]["target"]),
                        prog_pred_proba=np.asarray(raw_foldw[fold_idx]["predict_proba"]),
                        target=self.config.data.target)

            with open(paths_cache["metrics_fold-w"], "wb") as f:
                pickle.dump(metrics_foldw, f, pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved fold-w metrics to: {paths_cache['metrics_fold-w']}")
            logger.info(f"Metrics fold-w:")
            for fold_idx in self.fold_idcs:
                logger.info(f"Fold {fold_idx}:")
                _pprint_metrics(metrics_foldw[fold_idx])

        # Ens predicts
        if self.config.testing.ensemble_foldw and len(raw_foldw) > 0:
            if self.config.testing.use_cached and paths_cache["raw_ens"].exists():
                logger.info(f"Reading from {paths_cache['raw_ens']}")
                with open(paths_cache["raw_ens"], "rb") as f:
                    raw_ens = pickle.load(f)
            else:
                raw_ens = self.ensemble_eval_foldw(raw_foldw=raw_foldw)

                # Save ens to cache
                with open(paths_cache["raw_ens"], "wb") as f:
                    pickle.dump(raw_ens, f, pickle.HIGHEST_PROTOCOL)
                    logger.info(f"Saved ens raw predicts to: {paths_cache['raw_ens']}")

            # Metrics ens
            if self.config.testing.metrics_ensemble:
                metrics_ens = calc_metrics(
                    prog_target=np.asarray(raw_ens["target"]),
                    prog_pred_proba=np.asarray(raw_ens["predict_proba"]),
                    target=self.config.data.target)

                with open(paths_cache["metrics_ens"], "wb") as f:
                    pickle.dump(metrics_ens, f, pickle.HIGHEST_PROTOCOL)
                    logger.info(f"Saved ens metrics to: {paths_cache['metrics_ens']}")
                logger.info("Metrics ens:")
                _pprint_metrics(metrics_ens)

    @staticmethod
    def _extract_modal(batch, modal):
        assert modal in ("sag_3d_dess", "cor_iw_tse", "sag_t2_map", "xr_pa", "clin")
        return batch[f"image__{modal}"]

    @staticmethod
    def _downscale_x(modal, x, factor):
        if factor:
            x = preproc.PTInterpolate(scale_factor=factor)(x)
            x = x.contiguous()
        return x

    def eval_epoch(self, models):
        """Evaluation regime"""
        acc = defaultdict(list)

        ds = next(iter(self.config.data.sets.values()))
        dl = self.data_loaders[ds.name]["test"]
        steps_dl = len(dl)

        prog_bar_params = {"total": steps_dl, "desc": "Testing"}

        if self.config.testing.profile == "time":
            sum_time = 0
            sum_samples = 0

        with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
            for step_idx, data_batch_ds in enumerate(dl):
                # Select vars from batch
                xs_vec_ds = tuple(self._extract_modal(data_batch_ds, m)
                                  for m in ds.modals)
                xs_vec_ds = tuple(x.to(device) for x in xs_vec_ds)
                ys_true_ds = data_batch_ds["target"]
                ys_true_ds = ys_true_ds.to(device)

                # Last-chance preprocessing
                if self.config.model.downscale:
                    xs_vec_ds = tuple(self._downscale_x(m, x, tuple(f))
                                      for m, x, f in zip(ds.modals, xs_vec_ds,
                                                         self.config.model.downscale))

                # Model inference
                if self.config.testing.profile == "compute":
                    xs_vec_dummy = tuple(x[0:1] for x in xs_vec_ds)
                    macs, params = thop.profile(models["prog"], inputs=xs_vec_dummy)
                    macs, params = thop.clever_format([macs, params], "%.3f")
                    logger.info(f"MACs: {macs}, params: {params}")
                    quit()
                if self.config.testing.profile == "time":
                    time_pre = time.time()

                ys_pred_ds = models["prog"](*xs_vec_ds)["main"]

                if self.config.testing.profile == "time":
                    time_post = time.time()
                    sum_time += (time_post - time_pre)
                    sum_samples += int(xs_vec_ds[0].shape[0])

                if self.config.testing.debug:
                    print(f"Pred: {torch.argmax(ys_pred_ds, dim=1)}")
                    print(f"True: {ys_true_ds}")

                # Accumulate the predictions
                ys_true_ds_np = ys_true_ds.detach().to("cpu").numpy()
                t = ys_pred_ds.detach().to("cpu")
                ys_pred_ds_np = torch.argmax(t, dim=1).numpy()
                ys_pred_proba_ds_np = nn.functional.softmax(t, dim=1)

                acc["exam_knee_id"].extend(data_batch_ds[("-", "exam_knee_id")])
                acc["target"].extend(ys_true_ds_np.tolist())
                acc["predict"].extend(ys_pred_ds_np.tolist())
                acc["predict_proba"].extend(ys_pred_proba_ds_np.tolist())

                prog_bar.update(1)

        if self.config.testing.profile == "time":
            logger.info(f"Inference time per sample: {sum_time / sum_samples}")
            quit()

        return acc

    def ensemble_eval_foldw(self, raw_foldw):
        """Merge the predictions over all folds"""
        dfs = []
        for fold_idx, d in raw_foldw.items():
            t = pd.DataFrame.from_dict(d)
            t = t.rename(columns={"predict": f"predict__{fold_idx}",
                                  "predict_proba": f"predict_proba__{fold_idx}"})
            dfs.append(t)

        selectors = ["exam_knee_id", ]
        # Drop repeating columns with dtype not supported by merge
        dfs[1:] = [e.drop(columns="target") for e in dfs[1:]]
        df_ens = functools.reduce(
            lambda l, r: pd.merge(l, r, on=selectors, validate="1:1"), dfs)

        # Average fold predictions
        cols = [c for c in df_ens.columns if c.startswith("predict_proba__")]
        t = np.asarray(df_ens[cols].values.tolist())
        # samples * folds * classes
        t = softmax(np.mean(t, axis=1), axis=-1)
        df_ens["predict_proba"] = t.tolist()
        df_ens["predict"] = np.argmax(t, axis=-1).tolist()

        raw_ens = df_ens.to_dict(orient="list")
        return raw_ens

    def explain(self):
        paths_cache = {
            "raw_fold-w": Path(self.path_logs, f"explain_fus_raw_foldw.pkl"),
            "raw_ens": Path(self.path_logs, f"explain_fus_raw_ens.pkl"),
        }
        raw_foldw = dict()
        # raw_ens = dict()

        # Fold-w explanations
        if self.config.testing.use_cached and paths_cache["raw_fold-w"].exists():
            logger.info(f"Reading from {paths_cache['raw_fold-w']}")
            with open(paths_cache["raw_fold-w"], "rb") as f:
                raw_foldw = pickle.load(f)
        else:
            for fold_idx in self.fold_idcs:
                # Init fold-wise paths
                paths_weights_fold = dict()
                paths_weights_fold["prog"] = \
                    Path(self.path_weights, "prog", f"fold_{fold_idx}")

                handlers_ckpt = dict()
                handlers_ckpt["prog"] = CheckpointHandler(paths_weights_fold["prog"])

                paths_ckpt_sel = dict()
                paths_ckpt_sel["prog"] = handlers_ckpt["prog"].get_last_ckpt()

                # Initialize and configure model
                models = dict()
                models["prog"] = dict_models[self.config.model.name](
                    config=self.config.model,
                    path_weights=paths_ckpt_sel["prog"])

                models["prog"] = models["prog"].to(device)
                logger.warning("DataParallel disabled!")
                # Switch to eval regime
                models["prog"] = models["prog"].eval()

                # Make explanations of model predictions on subset
                t = self.explain_epoch(models=models)
                raw_foldw[fold_idx] = t

                del models
                gc.collect()
                torch.cuda.empty_cache()

            # Save fold-w to cache
            with open(paths_cache["raw_fold-w"], "wb") as f:
                pickle.dump(raw_foldw, f, pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved fold-w explanations to: {paths_cache['raw_fold-w']}")

        # Ens fold-w
        if self.config.testing.ensemble_foldw and len(raw_foldw) > 0:
            if self.config.testing.use_cached and paths_cache["raw_ens"].exists():
                logger.info(f"Cache exists in {paths_cache['raw_ens']}. Nop")
                # with open(paths_cache["raw_ens"], "rb") as f:
                #     raw_ens = pickle.load(f)
                pass
            else:
                raw_ens = self.ensemble_explain_foldw(raw_foldw=raw_foldw)

                # Save ens to cache
                with open(paths_cache["raw_ens"], "wb") as f:
                    pickle.dump(raw_ens, f, pickle.HIGHEST_PROTOCOL)
                    logger.info(f"Saved ens raw explanations to: {paths_cache['raw_ens']}")

    def explain_epoch(self, models):
        """Explanation regime"""
        acc = defaultdict(list)

        ds = next(iter(self.config.data.sets.values()))
        dl = self.data_loaders[ds.name]["test"]
        steps_dl = len(dl)

        prog_bar_params = {"total": steps_dl, "desc": "Explanation"}

        with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
            for step_idx, data_batch_ds in enumerate(dl):
                # Select vars from batch
                xs_vec_ds = tuple(self._extract_modal(data_batch_ds, m)
                                  for m in ds.modals)
                xs_vec_ds = tuple(x.to(device) for x in xs_vec_ds)
                ys_true_ds = data_batch_ds["target"]
                ys_true_ds = ys_true_ds.to(device)

                # Last-chance preprocessing
                if self.config.model.downscale:
                    xs_vec_ds = tuple(self._downscale_x(m, x, tuple(f))
                                      for m, x, f in zip(ds.modals, xs_vec_ds,
                                                         self.config.model.downscale))

                # Model inference
                ys_pred_raw_ds = models["prog"](*xs_vec_ds)
                ys_pred_proba_ds = torch.softmax(ys_pred_raw_ds, dim=1)
                ys_pred_ds_np = ys_pred_proba_ds.detach().to("cpu").numpy()
                ys_pred_ds_np = np.argmax(ys_pred_ds_np, axis=1)

                # Model explanation
                if self.config.testing.explain_fn == "modal_abl":
                    explainer = FeatureAblation(models["prog"])

                    masks_vec = tuple(torch.zeros_like(x, dtype=torch.uint8) for x in xs_vec_ds)
                    masks_vec = tuple(x+m for m, x in enumerate(masks_vec))
                    attrs = explainer.attribute(inputs=xs_vec_ds,
                                                baselines=None,  # zeroing
                                                target=ys_true_ds.squeeze(),
                                                feature_mask=masks_vec,
                                                perturbations_per_eval=1,
                                                # show_progress=True,
                                                )
                    # Derive importance
                    t = tuple(a.detach().to("cpu") for a in attrs)
                    attrs_pool = tuple(torch.mean(a.reshape(a.shape[0], -1), dim=1)
                                       for a in t)

                    t = torch.stack(attrs_pool, axis=1)  # batch * modal
                    t = t / torch.sum(torch.abs(t), dim=1, keepdim=True)  # norm sum to 1
                    t = np.round(np.abs(t.numpy()) * 100., decimals=3)  # to per cent
                    percent = t

                else:
                    msg = f"Unknown explain_fn: {self.config.testing.explain_fn}"
                    raise ValueError(msg)

                # ------
                # Accumulate the predictions
                ys_true_ds_np = ys_true_ds.detach().to("cpu").numpy()
                attrs_np = torch.stack(attrs_pool, axis=1).numpy()

                acc["exam_knee_id"].extend(data_batch_ds[("-", "exam_knee_id")])
                acc["target"].extend(ys_true_ds_np.tolist())
                acc["modal_names"].extend([ds.modals, ]*len(ys_true_ds_np))
                acc["modal_abl_attrs"].extend(attrs_np.tolist())
                acc["modal_abl_percent"].extend(percent.tolist())

                prog_bar.update(1)

        gc.collect()
        torch.cuda.empty_cache()

        return acc

    def ensemble_explain_foldw(self, raw_foldw):
        """Merge the explanations over all folds"""
        dfs = []
        for fold_idx, d in raw_foldw.items():
            t = pd.DataFrame.from_dict(d)
            t = t.rename(columns={"modal_abl_attrs": f"modal_abl_attrs__{fold_idx}",
                                  "modal_abl_percent": f"modal_abl_percent__{fold_idx}"})
            dfs.append(t)

        selectors = ["exam_knee_id", ]
        # Drop repeating columns with dtype not supported by merge
        for field in ("target", "modal_names"):
            dfs[1:] = [e.drop(columns=field) for e in dfs[1:]]

        df_ens = functools.reduce(
            lambda l, r: pd.merge(l, r, on=selectors, validate="1:1"), dfs)

        # Average fold predictions
        cols = [c for c in df_ens.columns if c.startswith("modal_abl_percent__")]
        t = np.asarray(df_ens[cols].values.tolist())
        # samples * folds * modals
        t = np.mean(t, axis=1)
        t = t / np.sum(t, axis=1, keepdims=True)  # norm sum to 1
        df_ens["modal_abl_percent"] = t.tolist()

        raw_ens = df_ens.to_dict(orient="list")
        return raw_ens


@hydra.main(config_path="conf", config_name="prog_fus")
def main(config: DictConfig) -> None:
    Path(config.path_logs).mkdir(exist_ok=True, parents=True)
    logging_fh = logging.FileHandler(
        Path(config.path_logs, "eval_prog_fus_{}.log".format(config.testing.folds.idx)))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    logger.info(OmegaConf.to_yaml(config, resolve=True))

    prog_pred = ProgressionPrediction(config=config)
    if config.testing.regime == "eval":
        prog_pred.eval()
    elif config.testing.regime == "explain":
        prog_pred.explain()
    else:
        raise ValueError(f"Unknown regime: {config.testing.regime}")


if __name__ == '__main__':
    main()
