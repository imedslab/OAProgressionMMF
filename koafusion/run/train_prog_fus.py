import os
import gc
import logging
from pathlib import Path
from collections import defaultdict

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from koafusion.datasets import prepare_datasets_loaders
from koafusion.models import dict_models
from koafusion import preproc
from koafusion.various import (dict_losses, dict_optimizers, dict_schedulers,
                               CheckpointHandler, set_ultimate_seed)
from koafusion.various._metrics_stat_anlys import calc_metrics_v2


# Fix to PyTorch multiprocessing issue: "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)

set_ultimate_seed()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class ProgressionPrediction(object):
    def __init__(self, *, config, fold_idx=None):
        self.config = config
        self.fold_idx = fold_idx

        if self.config.model.downscale:
            logger.warning("Downscaling is enabled!")

        # Initialize datasets, loaders, and transforms
        t = prepare_datasets_loaders(config=config, fold_idx=fold_idx)
        self.datasets = t[0]
        self.data_loaders = t[1]

        Path(self.config.path_experiment_root).mkdir(exist_ok=True)

        # Init experiment
        self.path_weights = Path(self.config.path_experiment_root, "weights")
        self.paths_weights_fold = dict()
        self.paths_weights_fold["prog"] = \
            Path(self.path_weights, "prog", f"fold_{self.fold_idx}")
        self.paths_weights_fold["prog"].mkdir(exist_ok=True, parents=True)

        self.path_logs = Path(self.config.path_experiment_root, "logs_train")
        self.path_logs_fold = Path(self.config.path_logs, f"fold_{self.fold_idx}")
        self.path_logs_fold.mkdir(exist_ok=True, parents=True)

        self.handlers_ckpt = dict()
        self.handlers_ckpt["prog"] = CheckpointHandler(self.paths_weights_fold["prog"])

        paths_ckpt_sel = dict()
        paths_ckpt_sel["prog"] = self.handlers_ckpt["prog"].get_last_ckpt()

        # Initialize and configure the models
        self.models = dict()
        self.models["prog"] = dict_models[self.config.model.name](
            config=self.config.model,
            path_weights=paths_ckpt_sel["prog"])

        self.models["prog"] = self.models["prog"].to(device)
        self.models["prog"] = nn.DataParallel(self.models["prog"])

        # Configure the training
        self.optimizers = dict()
        self.optimizers["prog"] = dict_optimizers[self.config.training.optim.name](
            self.models["prog"].parameters(),
            lr=self.config.training.optim.lr_init,
            weight_decay=self.config.training.optim.weight_decay)

        self.schedulers = dict()
        params = dict(self.config.training.sched.params)
        self.schedulers["prog"] = dict_schedulers[self.config.training.sched.name](
            optimizer=self.optimizers["prog"], **params)

        self.loss_fns = dict()
        self.loss_fns["prog"] = dict_losses[self.config.training.loss.name](
            **dict(self.config.training.loss.params),
            num_classes=self.config.model.output_channels)
        self.loss_fns["prog"] = self.loss_fns["prog"].to(device)

        self.tb = SummaryWriter(self.path_logs_fold)

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

    def train_epoch(self, epoch_idx):
        """Training regime"""
        metrics = {"batch-w": defaultdict(list),
                   "epoch-w": defaultdict(list)}

        ds = next(iter(self.config.data.sets.values()))
        dl = self.data_loaders[ds.name]["train"]
        steps_dl = len(dl)

        prog_bar_params = {"postfix": {"epoch": epoch_idx},
                           "total": steps_dl,
                           "desc": f"Training, epoch {epoch_idx}"}

        with tqdm(**prog_bar_params) as prog_bar:
            for step_idx, data_batch_ds in enumerate(dl):
                self.optimizers["prog"].zero_grad()

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
                ys_pred_ds = self.models["prog"](*xs_vec_ds)["main"]
                # Loss calculation
                loss_prog = self.loss_fns["prog"](input=ys_pred_ds.squeeze(1),
                                                  target=ys_true_ds.long().squeeze(1))

                if self.config.training.debug:
                    print(f"Pred: {torch.argmax(ys_pred_ds, dim=1)}")
                    print(f"True: {ys_true_ds[:, 0]}")

                # Logging
                metrics["batch-w"]["loss_prog"].append(loss_prog.item())
                tag = f"fold_{self.fold_idx}/loss_prog_batch"
                self.tb.add_scalars(
                    tag, {"train": float(loss_prog.item())},
                    global_step=epoch_idx * steps_dl + step_idx)
                # Optimization step
                loss_prog.backward()
                self.optimizers["prog"].step()

                prog_bar.update(1)

        return metrics

    def val_epoch(self, epoch_idx):
        """Validation regime"""
        metrics = {"batch-w": defaultdict(list),
                   "epoch-w": dict()}

        ds = next(iter(self.config.data.sets.values()))
        dl = self.data_loaders[ds.name]["val"]
        steps_dl = len(dl)

        prog_bar_params = {"postfix": {"epoch": epoch_idx},
                           "total": steps_dl,
                           "desc": f"Validation, epoch {epoch_idx}"}

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
                ys_pred_ds = self.models["prog"](*xs_vec_ds)["main"]
                # Loss calculation
                loss_prog = self.loss_fns["prog"](input=ys_pred_ds.squeeze(1),
                                                  target=ys_true_ds.long().squeeze(1))

                if self.config.validation.debug:
                    print(f"Pred: {torch.argmax(ys_pred_ds, dim=1)}")
                    print(f"True: {ys_true_ds}")

                # Logging
                metrics["batch-w"]["loss_prog"].append(np.round(loss_prog.item(), 3))
                tag = f"fold_{self.fold_idx}/loss_prog_batch"
                self.tb.add_scalars(
                    tag, {"val": float(loss_prog.item())},
                    global_step=epoch_idx * steps_dl + step_idx)

                # Accumulate the predictions
                ys_true_ds_np = ys_true_ds.detach().to("cpu").numpy()
                ys_pred_proba_ds = torch.softmax(ys_pred_ds.detach(), dim=1)
                ys_pred_proba_ds_np = ys_pred_proba_ds.detach().to("cpu").numpy()

                metrics["batch-w"]["target"].append(ys_true_ds_np)
                metrics["batch-w"]["predict_proba"].append(ys_pred_proba_ds_np)

                prog_bar.update(1)

        # Calculate metrics
        t_target = np.concatenate(metrics["batch-w"]["target"], axis=0)
        t_pred_proba = np.concatenate(metrics["batch-w"]["predict_proba"], axis=0)
        for f in ("target", "predict_proba"):
            del metrics["batch-w"][f]

        metrics["epoch-w"] = calc_metrics_v2(prog_target=t_target,
                                             prog_pred_proba=t_pred_proba,
                                             target=self.config.data.target)
        return metrics

    def fit(self):
        epoch_idx_best = -1
        crit_name = self.config.validation.criterion
        if crit_name in ("loss", ):
            crit_best = float("inf")
            crit_rule = lambda new, ref: new <= ref
        elif crit_name in ("b_accuracy", "avg_precision", "ap_sampleweighted"):
            crit_best = 0.0
            crit_rule = lambda new, ref: new >= ref
        else:
            raise ValueError(f"Unknown criterion: {crit_name}")
        metrics_train_best = dict()
        metrics_val_best = dict()

        for epoch_idx in range(self.config.training.epochs.num):
            # Training
            self.models = {n: m.train() for n, m in self.models.items()}
            metrics_train = self.train_epoch(epoch_idx)

            for k, v in metrics_train["batch-w"].items():
                if k.startswith("loss"):
                    metrics_train["epoch-w"][k] = np.mean(np.asarray(v))
                else:
                    logger.warning(f"Non-processed batch-wise entry: {k}")

            # Validation
            self.models = {n: m.eval() for n, m in self.models.items()}
            metrics_val = self.val_epoch(epoch_idx)

            for k, v in metrics_val["batch-w"].items():
                if k.startswith("loss"):
                    metrics_val["epoch-w"][k] = np.mean(np.asarray(v))
                else:
                    logger.warning(f"Non-processed batch-wise entry: {k}")

            # Learning rate update
            for n, s in self.schedulers.items():
                s.step()

            # Logging
            for subset, metrics in (("train", metrics_train),
                                    ("val", metrics_val)):
                for k, v in metrics["epoch-w"].items():
                    if isinstance(v, np.ndarray) and v.squeeze().ndim <= 1:
                        tag = f"fold_{self.fold_idx}/{k}_{subset}"
                        self.tb.add_scalars(
                            tag,
                            {f"class{i}": e for i, e in enumerate(v.ravel().tolist())},
                            global_step=epoch_idx)
                    elif isinstance(v, (str, int, float)):
                        tag = f"fold_{self.fold_idx}/{k}"
                        self.tb.add_scalars(
                            tag, {subset: float(v)},
                            global_step=epoch_idx)
                    else:
                        tag = f"fold_{self.fold_idx}/{k}_{subset}"
                        self.tb.add_text(tag, str(v), global_step=epoch_idx)

            for name, optim in self.optimizers.items():
                for param_group in optim.param_groups:
                    self.tb.add_scalar(
                        f"fold_{self.fold_idx}/learning_rate/{name}",
                        param_group["lr"],
                        global_step=epoch_idx)

            # Select and save the model
            crit_curr = {
                "loss": metrics_val["epoch-w"]["loss_prog"],
                "b_accuracy": metrics_val["epoch-w"]["b_accuracy"],
                "avg_precision": metrics_val["epoch-w"]["avg_precision"],
                # "ap_sampleweighted": metrics_val["epoch-w"]["ap_sampleweighted"],
            }[crit_name]

            if crit_rule(crit_curr, crit_best):
                crit_best = crit_curr
                epoch_idx_best = epoch_idx
                metrics_train_best = metrics_train
                metrics_val_best = metrics_val

                msg = (f"Metrics at so-far-best epoch {epoch_idx_best}:"
                       f"\n  train: {repr(metrics_train_best['epoch-w'])}"
                       f"\n  val: {repr(metrics_val_best['epoch-w'])}")
                logger.info(msg)

                self.handlers_ckpt["prog"].save_new_ckpt(
                    model=self.models["prog"],
                    model_name=self.config["model"]["name"],
                    fold_idx=self.fold_idx,
                    epoch_idx=epoch_idx)

        msg = (f"Finished fold {self.fold_idx} "
               f"with the {crit_name} value {np.round(crit_best, decimals=6)} "
               f"on epoch {epoch_idx_best}, "
               f"weights: ({self.paths_weights_fold})")
        logger.info(msg)


@hydra.main(config_path="conf", config_name="prog_fus")
def main(config: DictConfig) -> None:
    Path(config.path_logs).mkdir(exist_ok=True, parents=True)
    logging_fh = logging.FileHandler(
        Path(config.path_logs, "train_prog_fus_{}.log".format(config.training.folds.idx)))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    # Build a list of folds to run on
    if config.training.folds.idx == -1:
        fold_idcs = list(range(config.training.folds.num))
    else:
        fold_idcs = [config.training.folds.idx, ]
    if config.training.folds.ignore is not None:
        for g in config.training.folds.ignore:
            fold_idcs = [i for i in fold_idcs if i != g]

    logger.info(OmegaConf.to_yaml(config, resolve=True))

    for fold_idx in fold_idcs:
        logger.info(f"Training fold {fold_idx}")
        prog_pred = ProgressionPrediction(config=config, fold_idx=fold_idx)
        prog_pred.fit()

        del prog_pred.models, prog_pred.data_loaders, prog_pred.loss_fns,\
            prog_pred.schedulers, prog_pred.optimizers, prog_pred
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
