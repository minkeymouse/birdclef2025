import torch
import torchvision.ops as ops

class CrossEntropyLoss(torch.nn.Module):
    """
    Standard multi-class cross-entropy loss.
    Wraps torch.nn.CrossEntropyLoss for hard-label classification.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor of shape [B, C]
        targets: Tensor of shape [B] with class indices in [0, C-1]
        """
        return self.loss(logits, targets)


class FocalLossBCE(torch.nn.Module):
    """
    Hybrid BCEWithLogits + sigmoid focal loss for multi-label targets.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
        bce_weight: float = 0.6,
        focal_weight: float = 1.4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor of shape [B, C]
        targets: Tensor of shape [B, C] with binary labels {0,1}
        """
        focal = ops.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce = self.bce(logits, targets)
        return self.bce_weight * bce + self.focal_weight * focal


def get_criterion(
    loss_type: str = "cross_entropy",
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
    bce_weight: float = 0.6,
    focal_weight: float = 1.4,
) -> torch.nn.Module:
    """
    Factory to select the loss function.

    Args:
      loss_type: 'cross_entropy' or 'focal_loss_bce'
      alpha, gamma: focal loss parameters
      reduction: 'mean', 'sum', or 'none'
      bce_weight, focal_weight: weights for FocalLossBCE components

    Returns:
      nn.Module ready to use in training.
    """
    if loss_type == "cross_entropy":
        return CrossEntropyLoss(reduction=reduction)
    elif loss_type == "focal_loss_bce":
        return FocalLossBCE(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            bce_weight=bce_weight,
            focal_weight=focal_weight,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}")
