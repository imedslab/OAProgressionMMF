import logging

import torch
import torch.nn.functional as F
from torch import nn


logging.basicConfig()
logger = logging.getLogger('losses')
logger.setLevel(logging.DEBUG)


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, batch_avg=True, batch_weight=None,
                 class_avg=True, class_weight=None, **kwargs):
        """

        Parameters
        ----------
        batch_avg:
            Whether to average over the batch dimension.
        batch_weight:
            Batch samples importance coefficients.
        class_avg:
            Whether to average over the class dimension.
        class_weight:
            Classes importance coefficients.
        """
        super().__init__()
        self.num_classes = num_classes
        self.batch_avg = batch_avg
        self.class_avg = class_avg
        self.batch_weight = batch_weight
        self.class_weight = class_weight
        logger.warning(f"Redundant loss function arguments:\n{repr(kwargs)}")
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, input, target, **kwargs):
        """

        Parameters
        ----------
        input: tensor
        target: tensor

        Returns
        -------
        out: float tensor
        """
        return self.ce(input, target)


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, batch_avg=True, batch_weight=None,
                 class_avg=True, class_weight=None, gamma=2, reduction="mean", **kwargs):
        """

        Parameters
        ----------
        num_classes: scalar
            Total number of classes.
        batch_avg:
            Whether to average over the batch dimension.
        batch_weight:
            Batch samples importance coefficients.
        class_avg:
            Whether to average over the class dimension.
        class_weight:
            Classes importance coefficients.
        gamma: scalar
            Gamma coefficient from the paper.
        reduction : {"mean", "sum"}

        """
        super().__init__()
        self.num_classes = num_classes
        self.batch_avg = batch_avg
        self.class_avg = class_avg
        self.batch_weight = batch_weight
        self.class_weight = class_weight
        if reduction not in ("mean", "sum"):
            raise ValueError("Unknown `reduction` value")
        else:
            self.reduction = reduction

        self.gamma = gamma
        logger.warning(f"Redundant loss function arguments:\n{repr(kwargs)}")

    def forward(self, input, target, **kwargs):
        """

        Parameters
        ----------
        input: (b, ch, d0, d1) tensor
        target: (b, d0, d1) tensor

        Returns
        -------
        out: float tensor
        """
        logpt = -F.cross_entropy(input, target,
                                 weight=self.class_weight, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()


dict_losses = {
    "bce_loss": nn.BCELoss,
    "bce_wlogits_loss": nn.BCEWithLogitsLoss,
    # "ce_loss": nn.CrossEntropyLoss,
    "CrossEntropyLoss": CrossEntropyLoss,
    "FocalLoss": FocalLoss,
}
