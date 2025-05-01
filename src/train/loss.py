import torch
import torchvision

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, targets):
        return self.loss(logits, targets)


class FocalLossBCE(torch.nn.Module):
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

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss

def get_criterion(
    loss_type: str = "cross_entropy",
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
    bce_weight: float = 0.6,
    focal_weight: float = 1.4,
):
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
