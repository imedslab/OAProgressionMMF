"""
Code source: https://github.com/wissam-sib/calibrated_metrics
"""

import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.utils import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import assert_all_finite
from sklearn.utils import check_consistent_length


"""
Taken from https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/metrics/_ranking.py#L498
See alternative: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/metrics_impl.py#L1861
"""
def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
            classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def precision_recall_curve_calib(y_true, y_pred, pos_label=None,
                                 sample_weight=None, pi0=None):
    """Compute precision-recall pairs for different probability thresholds
    This is a modification of sklearn's "precision_recall_curve" that adds calibration.

    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    y_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    pi0 : float
        Prevalence to be used for calibrated precision estimates.

    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    
    if pi0 is not None:
        pi = np.sum(y_true) / float(np.array(y_true).shape[0])
        ratio = pi * (1 - pi0) / (pi0 * (1 - pi))
        precision = tps / (tps + ratio * fps)
    else:
        precision = tps / (tps + fps)
    
    precision[np.isnan(precision)] = 0
        
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score_calib(y_true, y_pred, pos_label=1, sample_weight=None, pi0=None):
    precision, recall, _ = precision_recall_curve_calib(y_true, y_pred,
                                                        pos_label=pos_label,
                                                        sample_weight=sample_weight, pi0=pi0)
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def f1score_calib(y_true, y_pred, pi0=None):
    """

    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier. (must be binary)
    pi0 : float, None by default
        The reference ratio for calibration
    """
    cm = confusion_matrix(y_true, y_pred)
    
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
        
    pos = fn + tp
    
    recall = tp / float(pos)
    
    if pi0 is not None:
        pi = pos / float(tn + fn + tp + fp)
        ratio = pi * (1 - pi0) / (pi0 * (1-pi))
        precision = tp / float(tp + ratio * fp)
    else:
        precision = tp / float(tp + fp)
    
    if np.isnan(precision):
        precision = 0

    if (precision + recall) == 0.0:
        f = 0.0
    else:
        f = (2*precision*recall)/(precision+recall)

    return f


def bestf1score_calib(y_true, y_pred, pi0=None):
    """

    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    pi0 : float, None by default
        The reference ratio for calibration
    """
    precision, recall, _ = precision_recall_curve_calib(y_true, y_pred, pi0=pi0)

    fscores = (2 * precision * recall) / (precision + recall)
    fscores = np.nan_to_num(fscores, nan=0, posinf=0, neginf=0)

    return np.max(fscores)
