from torchvision import transforms
from lightly.transforms.vicreg_transform import VICRegTransform

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


vicreg_transform = VICRegTransform(input_size=64)


class DistillationTransform:
    """Return one VICReg-style augmented view for row C distillation."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        views = self.transform(image)
        return views[0]


distillation_transform = DistillationTransform(vicreg_transform)
