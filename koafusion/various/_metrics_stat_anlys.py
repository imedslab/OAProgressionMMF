import copy
import numpy as np
from scipy import integrate
from sklearn.metrics import (balanced_accuracy_score, average_precision_score,
                             roc_auc_score, precision_recall_curve, roc_curve,
                             recall_score)
from tqdm import tqdm
from ._metrics_wissam import average_precision_score_calib, precision_recall_curve_calib


def avg_precision_at_recall_range(y_true, probas_pred, recall_range=(0.0, 1.0),
                                  sample_weight=None):
    precs, recs, thrs = precision_recall_curve(y_true=y_true,
                                               probas_pred=probas_pred,
                                               sample_weight=sample_weight)
    precs = precs[::-1]
    recs = recs[::-1]

    idx_low = np.argwhere(recs <= recall_range[0])[-1][0]
    idx_high = np.argwhere(recs >= recall_range[1])[0][0]

    rec_interval = recs[idx_high] - recs[idx_low]
    ap = integrate.trapezoid(x=recs[idx_low:idx_high + 1],
                             y=precs[idx_low:idx_high + 1]) / rec_interval
    return ap


def calc_bootstrap(metric, y_true, y_pred, n_bootstrap=100, seed=0, stratified=True,
                   alpha=95., ddof=0, verbose=True):
    """

    Inspired by https://github.com/MIPT-Oulu/OAProgression/blob/master/oaprogression/evaluation/stats.py

    Parameters
    ----------
    metric : fucntion
        Metric to compute
    y_true : ndarray
        Ground truth
    y_pred : ndarray
        Predictions
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width
    """
    if len(np.unique(y_true)) > 2:
        raise ValueError(f"Expected binary target, got: {np.unique(y_true)}")

    np.random.seed(seed)
    metric_vals = []
    ind_pos = np.where(y_true == 1)[0]
    ind_neg = np.where(y_true == 0)[0]

    if verbose:
        loop = tqdm(range(n_bootstrap), total=n_bootstrap, desc="Bootstrap:")
    else:
        loop = range(n_bootstrap)
    for _ in loop:
        if stratified:
            ind_pos_bs = np.random.choice(ind_pos, ind_pos.shape[0])
            ind_neg_bs = np.random.choice(ind_neg, ind_neg.shape[0])
            ind = np.hstack((ind_pos_bs, ind_neg_bs))
        else:
            ind = np.random.choice(y_true.shape[0], y_true.shape[0])

        if y_true[ind].sum() == 0:
            continue
        metric_vals.append(metric(y_true[ind], y_pred[ind]))

    metric_val = metric(y_true, y_pred)
    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)
    std_err = np.std(metric_vals, ddof=ddof)

    return metric_val, std_err, ci_l, ci_h


def calc_metrics_v2(prog_target, prog_pred_proba, target, with_curves=False,
                    bootstrap=False, kws_ppv=None, kws_bs=None):
    """
    Args:
        prog_target: (sample, )
        prog_pred_proba: (sample, class)
        target: str
        with_curves: bool
        bootstrap: bool
        kws_ppv: dict
        kws_bs: dict

    Returns:
        out: dict
    """
    out = dict()

    kws_bs_def = {"n_bootstrap": 1000, "seed": 0, "stratified": True, "alpha": 95}
    if kws_bs is None:
        kws_bs_all = kws_bs_def
    else:
        kws_bs_all = copy.deepcopy(kws_bs_def)
        kws_bs_all.update(kws_bs)

    kws_ppv_def = {"pi0": 0.12,}
    if kws_ppv is None:
        kws_ppv_all = kws_ppv_def
    else:
        kws_ppv_all = copy.deepcopy(kws_ppv_def)
        kws_ppv_all.update(kws_ppv)

    if len(np.unique(prog_target)) < 2:
        out["sample_size"] = prog_target.shape[0]
        out["num_pos"] = np.sum(prog_target == 1)
        out["num_neg"] = np.sum(prog_target == 0)
        out["prevalence"] = np.nan
        out["roc_auc"] = np.nan
        out["avg_precision"] = np.nan
        out["avg_ppv_calib"] = np.nan
        out["avg_npv"] = np.nan
        out["cutoff"] = np.nan
        out["youdens_index"] = np.nan
        out["b_accuracy"] = np.nan
        out["roc_curve"] = np.nan
        out["pr_curve"] = np.nan
        return out

    if target in ("prog_kl_12", "prog_kl_24", "prog_kl_36", "prog_kl_48",
                  "prog_kl_72", "prog_kl_96",
                  "tiulpin2019_prog_bin"):
        prog_target_bin = prog_target
        prog_pred_proba_bin_pos = prog_pred_proba[:, 1]
        prog_pred_proba_bin_neg = prog_pred_proba[:, 0]

        out["sample_size"] = prog_target_bin.shape[0]
        out["num_pos"] = np.sum(prog_target_bin == 1)
        out["num_neg"] = np.sum(prog_target_bin == 0)
        out["prevalence"] = np.sum(prog_target_bin) / prog_target_bin.shape[0]

        if bootstrap:
            out["roc_auc"] = calc_bootstrap(metric=roc_auc_score,
                                            y_true=prog_target_bin,
                                            y_pred=prog_pred_proba_bin_pos,
                                            **kws_bs_all)
        else:
            out["roc_auc"] = roc_auc_score(prog_target_bin, prog_pred_proba_bin_pos)

        if bootstrap:
            out["avg_precision"] = calc_bootstrap(metric=average_precision_score,
                                                  y_true=prog_target_bin,
                                                  y_pred=prog_pred_proba_bin_pos,
                                                  **kws_bs_all)
        else:
            out["avg_precision"] = average_precision_score(prog_target_bin,
                                                           prog_pred_proba_bin_pos)

        if bootstrap:
            fn_ppv = lambda t, p: average_precision_score_calib(t, p, pi0=kws_ppv_all["pi0"])
            out["avg_ppv_calib"] = calc_bootstrap(metric=fn_ppv,
                                                  y_true=prog_target_bin,
                                                  y_pred=prog_pred_proba_bin_pos,
                                                  **kws_bs_all)
        else:
            out["avg_ppv_calib"] = average_precision_score_calib(prog_target_bin,
                                                                 prog_pred_proba_bin_pos,
                                                                 pi0=kws_ppv_all["pi0"])

        if bootstrap:
            fn_npv = lambda y1, y2: average_precision_score(y1, y2, pos_label=0)
            out["avg_npv"] = calc_bootstrap(metric=fn_npv,
                                            y_true=prog_target_bin,
                                            y_pred=prog_pred_proba_bin_neg,
                                            **kws_bs_all)
        else:
            out["avg_npv"] = average_precision_score(prog_target_bin,
                                                     prog_pred_proba_bin_neg,
                                                     pos_label=0)

        if not bootstrap:
            out["cutoff"] = sensitivity_specificity_cutoff(prog_target_bin,
                                                           prog_pred_proba_bin_pos)
            out["youdens_index"] = youdens_index(prog_target_bin,
                                                 prog_pred_proba_bin_pos,
                                                 threshold=out["cutoff"])
            out["b_accuracy"] = balanced_accuracy_score(prog_target_bin,
                                                        prog_pred_proba_bin_pos > 0.5)

        if with_curves and not bootstrap:
            fpr, tpr, _ = roc_curve(prog_target_bin, prog_pred_proba_bin_pos)
            out["roc_curve"] = (fpr, tpr)

            prec, rec, thr = precision_recall_curve(y_true=prog_target_bin,
                                                    probas_pred=prog_pred_proba_bin_pos,
                                                    sample_weight=None)
            out["pr_curve"] = (prec, rec)
            prec, rec, thr = precision_recall_curve_calib(y_true=prog_target_bin,
                                                          y_pred=prog_pred_proba_bin_pos,
                                                          sample_weight=None,
                                                          pi0=kws_ppv_all["pi0"])
            out["pr_calib_curve"] = (prec, rec)

    else:
        raise ValueError(f"Unknown target: {target}")

    for k, v in out.items():
        if k in ("prevalence", "roc_auc",
                 "avg_precision",
                 "avg_ppv_calib",
                 "avg_npv",
                 "cutoff", "youdens_index",
                 "b_accuracy"):
            out[k] = np.round(v, 3)

    return out


def mc_bacc(y_true, y_pred):
    ret = recall_score(y_true, y_pred, average="macro")
    return ret


def sensitivity_specificity_cutoff(y_true, y_pred_proba):
    """Find data-driven cut-off for classification

    Cut-off is determined using Youden's index defined as sensitivity + specificity - 1.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.
    y_pred_proba : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    References
    ----------
    Taken from https://gist.github.com/twolodzko/4fae2980a1f15f8682d243808e5859bb

    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.

    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.

    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def youdens_index(y_true, y_pred_proba, threshold):
    y_pred = y_pred_proba >= threshold

    sensit = recall_score(y_true, y_pred, pos_label=1)
    specif = recall_score(y_true, y_pred, pos_label=0)

    index = sensit + specif - 1.
    return index
