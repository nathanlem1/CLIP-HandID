"""
This implements label smoothing which prevents over-fitting and over-confidence for a classification task i.e. it is a
regularization technique. You can refer for more information on label smoothing here:
https://arxiv.org/pdf/1906.02629.pdf

"""
import torch
import torch.nn.functional as F
import torch.nn as nn


def reduce_loss(loss, reduction='mean'):
    """ Reduce loss
    Args:
        loss: The output (loss here).
        reduction: The reduction to apply to the output (loss here) such as 'mean', 'sum' or 'none'.
    return:
        reduced output (loss here).
    """
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """ Label smoothing cross entropy loss
    Args:
        epsilon: A small constant (smoothing value) to encourage the model to be less confident on the training set.
        reduction: The reduction to apply to the output (loss here) such as 'mean', 'sum' or 'none'.
        preds: Predictions
        target: Labels
    """
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        num_classes = preds.size()[-1]  # Number of classes.
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)  # The negative log likelihood loss

        ls_ce = (loss / num_classes) * self.epsilon + (1 - self.epsilon) * nll

        return ls_ce


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, preds, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        num_classes = preds.size()[-1]  # Number of classes.
        log_probs = self.logsoftmax(preds)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(preds.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        Args:
            smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, preds, target):
        logprobs = F.log_softmax(preds, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Usage
if __name__ == "__main__":
    loss_fun = LabelSmoothingCrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)   # Predictions
    target = torch.empty(3, dtype=torch.long).random_(5)  # Label
    loss = loss_fun(input.cuda(), target.cuda())
    print(loss)
    # loss.backward()
    # optimizer.step()

