
import torch.nn as nn

from lightly.models.modules.heads import VICRegProjectionHead


class VICRegModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.projector = VICRegProjectionHead(
            input_dim=backbone.feature_dim,
            hidden_dim=2048,
            output_dim=2048,
            num_layers=2,
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)

        return projections
