"""Model factory and losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_model(name="efficientnet_b0", num_classes=9, pretrained=True,
                drop_rate=0.2, drop_path=0.1, **kwargs):
    """timm backbone adapted to 3-channel wafer images.

    Extra kwargs are forwarded to timm.create_model (e.g. img_size=128 for the
    Swin transformer, which needs the input resolution at construction time)."""
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes,
        in_chans=3, drop_rate=drop_rate, drop_path_rate=drop_path, **kwargs,
    )
    return model


class FocalLoss(nn.Module):
    """Class-weighted focal loss with label smoothing.

    Focal term down-weights easy 'none' examples; class weights and the
    balanced sampler together counter the strong class imbalance."""
    def __init__(self, weight=None, gamma=1.5, smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("weight",
                             weight if weight is not None else None)

    def forward(self, logits, target):
        n = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        # label smoothing target distribution
        with torch.no_grad():
            true = torch.zeros_like(logp).fill_(self.smoothing / (n - 1))
            true.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        p = logp.exp()
        focal = (1 - p).clamp(min=1e-6) ** self.gamma
        loss = -(focal * true * logp)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(1).mean()
