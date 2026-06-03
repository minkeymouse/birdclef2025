import torch
import torch.nn as nn


class SmallSedCnn(nn.Module):
    def __init__(self, in_chans: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.head(x)


def build_model(cfg: dict, pretrained: bool | None = None) -> nn.Module:
    mcfg = cfg["model"]
    backend = str(mcfg.get("backend", "timm")).lower()
    in_chans = int(mcfg.get("in_chans", 2))
    num_classes = int(mcfg.get("num_classes", 234))
    if pretrained is None:
        pretrained = bool(mcfg.get("pretrained", False))

    if backend == "small_cnn":
        return SmallSedCnn(in_chans=in_chans, num_classes=num_classes)

    if backend == "timm":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for model.backend=timm. Install it on RunPod, "
                "or set model.backend=small_cnn for a wiring smoke test."
            ) from exc
        return timm.create_model(
            str(mcfg.get("name", "convnext_tiny.fb_in22k")),
            pretrained=bool(pretrained),
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=float(mcfg.get("drop_rate", 0.0)),
        )

    raise ValueError(f"Unknown model backend: {backend}")

