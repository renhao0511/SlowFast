#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

class BalanceBCELoss(nn.Module):
    def __init__(self, negative_ratio=1.0):
        super(BalanceBCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = 1e-5

    def forward(self, pred, gt):
        positive = gt
        negative = 1 - gt
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            # int(max(50, positive_count) * self.negative_ratio))
            int(positive_count * self.negative_ratio))

        
        assert gt.max() <= 1 and gt.min() >= 0
        assert pred.max() <= 1 and pred.min() >= 0
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps)

        return balance_loss

class BalanceCELoss(nn.Module):
    def __init__(self, reduction, negative_ratio=1.0):
        super(BalanceCELoss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred, gt):
        assert pred.shape[-1] == gt.shape[-1]

        balance_loss = 0
        # for i in range(1, pred.shape[-1]):
        for i in range(pred.shape[-1]):
            balance_loss += BalanceBCELoss(self.negative_ratio)(pred[:, :, :, :, i], gt[:, :, :, :, i])

        return balance_loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "BalanceCELoss": BalanceCELoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
