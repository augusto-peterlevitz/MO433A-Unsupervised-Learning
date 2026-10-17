
import torch

from models.student_cnn import StudentCNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

for name, width in [
    ("small", 0.5),
    ("large", 1.0),
]:
    model = StudentCNN(width_mult=width).to(device)

    x = torch.randn(4, 3, 96, 96, device=device)

    with torch.no_grad():
        features = model.forward_features(x)

    print(f"\n{name.upper()} STUDENT")
    print("Parameters:", f"{count_parameters(model):,}")
    print("Input shape:", tuple(x.shape))
    print("Feature shape:", tuple(features.shape))
    print("Feature dimension:", model.feature_dim)
    print("Finite:", torch.isfinite(features).all().item())

    assert features.shape == (4, model.feature_dim)
    assert torch.isfinite(features).all()