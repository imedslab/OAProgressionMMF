import logging
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import sklearn
from sklearn import model_selection, preprocessing, linear_model, tree
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from koafusion.datasets import sources_from_path, DatasetOAI3d
from koafusion.various import set_ultimate_seed


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


def prepare_datasets(config):
    # Collect the available and specified sources
    sources = sources_from_path(
        path_data_root=config.path_data_root,
        modals_all=config.data.modals_all,
        target=config.data.target,
        fold_num=config.training.folds.num,
        # scheme_trainval_test=config.scheme_trainval_test,
        scheme_train_val=config.scheme_train_val,
        seed_trainval_test=config.seed_trainval_test,
        seed_train_val=config.seed_train_val,
        site_test=config.site_test,
        ignore_cache=config.data.ignore_cache,
    )

    # Compose the main parts
    d = config.data.sets.n0
    tmp = {
        "trainval_df": sources[d.name]["trainval_df"]["-"],
        "test_df": sources[d.name]["test_df"]["-"],
        "trainval_folds": list(sources[d.name]["trainval_folds"]),
    }
    datasets = {d.name: tmp}
    return datasets


class ProgressionPrediction(object):
    def __init__(self, *, config):
        self.config = config

        # Initialize datasets, loaders, and transforms
        self.datasets = prepare_datasets(config=config)

        Path(self.config.path_experiment_root).mkdir(exist_ok=True)

        # Init experiment paths
        self.path_weights = Path(self.config.path_experiment_root, "weights")
        self.path_weights.mkdir(exist_ok=True, parents=True)
        if "sag_t2_map" in self.config.data.modals_all:
            sel_knee = "incid"
        else:
            sel_knee = "all"
        self.path_logs = Path(self.config.path_experiment_root, "logs_eval", sel_knee)
        self.path_logs.mkdir(exist_ok=True, parents=True)

    def fit(self):
        # Extract the subsets
        d = self.config.data.sets.n0

        df_trainval = self.datasets[d.name]["trainval_df"]
        df_test = self.datasets[d.name]["test_df"]
        folds_trainval = self.datasets[d.name]["trainval_folds"]

        # Select and preprocess features
        var_to_col = {
            "age": "AGE", "sex": "P02SEX", "bmi": "P01BMI", "kl": "XRKL",
            "inj": "P01INJ-", "surg": "P01KSURG-", "womac": "WOMTS-",
        }
        fns_prep = {
            "age": preprocessing.StandardScaler(),
            "sex": preprocessing.OneHotEncoder(),
            "bmi": preprocessing.StandardScaler(),
            "kl": preprocessing.OneHotEncoder(),
            "inj": preprocessing.OneHotEncoder(),
            "surg": preprocessing.OneHotEncoder(),
            "womac": preprocessing.StandardScaler(),
        }

        for v in ("age", "sex", "bmi", "kl", "inj", "surg", "womac"):
            fns_prep[v].fit(df_trainval[var_to_col[v]].to_numpy().reshape(-1, 1))

        cols_x = ["AGE", "P02SEX", "P01BMI"]
        for v in ("kl", "inj", "surg", "womac"):
            if v in self.config.model.vars:
                cols_x.append(var_to_col[v])

        col_y = self.config.data.target

        X_trainval = df_trainval.loc[:, cols_x]
        y_trainval = df_trainval.loc[:, col_y]
        X_test = df_test.loc[:, cols_x]
        y_test = df_test.loc[:, col_y]
        vars_test = df_test.loc[:, cols_x + ["exam_knee_id", ]]

        t_trainval = [
            fns_prep["age"].transform(X_trainval["AGE"].to_numpy().reshape(-1, 1)),
            fns_prep["sex"].transform(X_trainval["P02SEX"].to_numpy().reshape(-1, 1)).toarray(),
            fns_prep["bmi"].transform(X_trainval["P01BMI"].to_numpy().reshape(-1, 1)),
        ]
        for v in ("kl", "inj", "surg"):
            if v in self.config.model.vars:
                t_trainval.append(
                    fns_prep[v].transform(X_trainval[var_to_col[v]].to_numpy().reshape(-1, 1)).toarray())
        for v in ("womac",):
            if v in self.config.model.vars:
                t_trainval.append(
                    fns_prep[v].transform(X_trainval[var_to_col[v]].to_numpy().reshape(-1, 1)))
        X_trainval = np.concatenate(t_trainval, axis=1)

        t_test = [
            fns_prep["age"].transform(X_test["AGE"].to_numpy().reshape(-1, 1)),
            fns_prep["sex"].transform(X_test["P02SEX"].to_numpy().reshape(-1, 1)).toarray(),
            fns_prep["bmi"].transform(X_test["P01BMI"].to_numpy().reshape(-1, 1)),
        ]
        for v in ("kl", "inj", "surg"):
            if v in self.config.model.vars:
                t_test.append(
                    fns_prep[v].transform(X_test[var_to_col[v]].to_numpy().reshape(-1, 1)).toarray())
        for v in ("womac",):
            if v in self.config.model.vars:
                t_test.append(
                    fns_prep[v].transform(X_test[var_to_col[v]].to_numpy().reshape(-1, 1)))
        X_test = np.concatenate(t_test, axis=1)

        # Run grid search over params
        clfs = {
            "LR": linear_model.LogisticRegression,
            "DT": tree.DecisionTreeClassifier,
        }
        param_grids = {
            "LR": {
                "class_weight": [None, "balanced"],
            },
            "DT": {
                "max_depth": [3, 10, 30],
                "min_samples_split": [10, 30, 100, 300],
                "min_samples_leaf": [10, 30, 100],
                "max_features": [None, "sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        }
        params = {"LR": None, "DT": None}
        models = {"LR": None, "DT": None}

        if self.config.model.params_init == "grid_search":
            for name in clfs:
                gs = model_selection.GridSearchCV(
                    estimator=clfs[name](),
                    param_grid=param_grids[name],
                    scoring=self.config.validation.criterion,
                    n_jobs=12,
                    cv=iter(folds_trainval),
                    refit=False,
                    verbose=0,
                    # verbose=1,
                    return_train_score=True,
                )
                gs.fit(X_trainval, y_trainval)
                params[name] = gs.best_params_
                logger.info(f"{name} grid search best params")
                logger.info(repr(gs.best_params_))
        elif self.config.model.params_init == "prev_best":
            params = {
                "LR": {'class_weight': 'balanced', },
                "DT": {'class_weight': 'balanced', 'max_depth': 10, 'max_features': 'log2',
                       'min_samples_leaf': 100, 'min_samples_split': 100},
            }
        else:
            raise ValueError(f"Unknown `params_init`: {self.config.model.params_init}")

        # Train an ensemble
        for name in clfs:
            cv_results = model_selection.cross_validate(
                estimator=clfs[name](random_state=0, **params[name]),
                X=X_trainval,
                y=y_trainval,
                scoring=self.config.validation.criterion,
                cv=iter(folds_trainval),
                n_jobs=12,
                return_estimator=True,
            )
            models[name] = cv_results["estimator"]
            logger.info(f"{name} OOF evaluation")
            logger.info(f"{self.config.validation.criterion}: {cv_results['test_score']}")

        # Specify cache paths
        paths_cache = {
            # "raw_fold-w": Path(self.path_logs, "eval_clin_raw_foldw.pkl"),
            "raw_ens": Path(self.path_logs, "eval_clin_raw_ens.pkl"),
            # "metrics_fold-w": Path(self.path_logs, "eval_clin_metrics_foldw.pkl"),
            "metrics_ens": Path(self.path_logs, "eval_clin_metrics_foldw.pkl")
        }

        # Make predictions on test subset
        raw_ens = defaultdict(dict)

        for name in clfs:
            raw_ens[name] = vars_test.to_dict(orient="list")

            pred_proba_test_foldw = np.asarray([m.predict_proba(X_test)
                                                for m in models[name]])
            pred_proba_test = np.mean(pred_proba_test_foldw, axis=0)
            pred_test = np.argmax(pred_proba_test, axis=1)
            for fold_idx in range(self.config.training.folds.num):
                raw_ens[name].update({
                    f"predict_proba__{fold_idx}": pred_proba_test_foldw[fold_idx],
                    f"predict__{fold_idx}": np.argmax(pred_proba_test_foldw[fold_idx], axis=1)
                })
            raw_ens[name].update({
                "predict_proba": pred_proba_test,
                "predict": pred_test,
                "target": y_test.to_numpy(),
            })

        # Save predictions
        logger.info(f"Saved test subset prediction to {paths_cache['raw_ens']}")
        with open(paths_cache["raw_ens"], "wb") as f:
            pickle.dump(raw_ens, f, pickle.HIGHEST_PROTOCOL)

        # Save the model snapshots
        for name in clfs:
            t_path = Path(self.path_weights, f"{name}_all-folds.pkl")

            with open(t_path, "wb") as f:
                pickle.dump(models[name], f)
            logger.info(f"Saved model {name} to {t_path}")


@hydra.main(config_path="conf", config_name="prog_clin")
def main(config: DictConfig) -> None:
    Path(config.path_logs).mkdir(exist_ok=True, parents=True)
    logging_fh = logging.FileHandler(Path(config.path_logs, "train_prog_clin.log"))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    logger.info(OmegaConf.to_yaml(config, resolve=True))

    logger.info(f"Training all {config.training.folds.num} folds")

    prog_pred = ProgressionPrediction(config=config)
    prog_pred.fit()


if __name__ == "__main__":
    main()
