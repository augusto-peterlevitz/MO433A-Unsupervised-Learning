
import torch
import torch.nn as nn
from tqdm import tqdm

class ConvBlock(nn.Module):
    """Two convolution layers with normalization and ReLU."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StudentCNN(nn.Module):
    """
    CNN encoder for self-supervised learning and distillation.

    Returns a global-average-pooled feature vector.
    No classification head is included.
    """

    def __init__(self, base_channels):
        super().__init__()

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ConvBlock(
            channels[0], channels[0], stride=1
        )
        self.stage2 = ConvBlock(
            channels[0], channels[1], stride=2
        )
        self.stage3 = ConvBlock(
            channels[1], channels[2], stride=2
        )
        self.stage4 = ConvBlock(
            channels[2], channels[3], stride=2
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = channels[3]

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # [batch, channels, height, width]
        x = self.pool(x)

        # [batch, channels, 1, 1] -> [batch, channels]
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.forward_features(x)
    
    def load_weights(self, checkpoint_path):
        """Load weights from a checkpoint file."""
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.load_state_dict(state_dict, strict=True)

    def extract_features(self, dataloader, device):
        
        feature_batches = []
        label_batches = []
        
        self.eval()
        self.requires_grad_(False)

        print("Extracting features...")
        with torch.inference_mode():
            for batch_idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
                images = images.to(device, non_blocking=True)
                features = self(images)
                if features.ndim != 2:
                    raise RuntimeError(
                        f"Expected a 2D feature matrix, got {tuple(features.shape)}."
                    )
                if not torch.isfinite(features).all():
                    raise RuntimeError("Non-finite VICReg features detected.")

                feature_batches.append(features.cpu())
                label_batches.append(labels.cpu())
        
        if not feature_batches:
            raise RuntimeError("No features were extracted; the dataset is empty.")

        return (
            torch.cat(feature_batches).numpy(),
            torch.cat(label_batches).numpy(),
        )


class SmallStudent(StudentCNN):
    """Narrow student variant with a 128-dimensional feature vector."""

    def __init__(self):
        super().__init__(base_channels=16)


class LargerStudent(StudentCNN):
    """Wider student variant with a 256-dimensional feature vector."""

    def __init__(self):
        super().__init__(base_channels=32)


def build_student(student_cnn):
    """Construct a named student architecture."""
    architectures = {
        "small": SmallStudent,
        "large": LargerStudent,
    }
    try:
        architecture = architectures[student_cnn]
    except KeyError as error:
        raise ValueError(
            f"Unsupported student CNN {student_cnn!r}; choose 'small' or 'large'."
        ) from error
    return architecture()