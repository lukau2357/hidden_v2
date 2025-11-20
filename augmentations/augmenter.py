import torch
import torch.nn as nn

from augmentations.valuemetric import Brightness, Contrast, DiffJPEG, GaussianBlur, Hue, Saturation
from augmentations.geometric import HorizontalFlip, Perspective, Resize, Crop, Rotate, Combine
from augmentations.splicing import PixelSplicing, BoxSplicing

class Augmenter(nn.Module):
    def __init__(self, conf):
        super().__init__()

        # TODO: Better idea than manually maping?
        self.str2class = {
            "DiffJPEG": DiffJPEG,
            "Brightness": Brightness,
            "Contrast": Contrast,
            "GaussianBlur": GaussianBlur,
            "Hue": Hue,
            "Saturation": Saturation,
            "HorizontalFlip": HorizontalFlip,
            "Perspective": Perspective,
            "Resize": Resize,
            "Crop": Crop,
            "Rotate": Rotate,
            "Combine": Combine,
            "PixelSplicing": PixelSplicing,
            "BoxSplicing": BoxSplicing,
            "Identity": nn.Identity
        }

        def _parse_conf(conf):
            modules = []
            for key, params in conf.items():
                new_module = self.str2class[key](**params) if params is not None else self.str2class[key]()
                modules.append(new_module)
            
            return modules
        
        self.augmentations = nn.ModuleList(_parse_conf(conf))
        self.selection_probabilities = torch.ones((len(self.augmentations,)), dtype = torch.float32)
        self.selection_probabilities /= self.selection_probabilities.sum()

    def forward(self, X, X_wm):
        # X [B, C, H, W], assuming [-1, 1] range.
        # Applies a randomly sampled augmentation to the input batch of images
        # Output range is also [-1, 1]
        aug_index = torch.multinomial(self.selection_probabilities, 1).item()
        # print(f"Applying augmentation: {self.augmentations[aug_index].__class__}")
        # For splicing operations, we combine the original and watermarked images.
        if isinstance(self.augmentations[aug_index], PixelSplicing) or isinstance(self.augmentations[aug_index], BoxSplicing):
            return self.augmentations[aug_index](X, X_wm)
        
        # Otherwise, we just augment the watermarked image.
        return self.augmentations[aug_index](X_wm)