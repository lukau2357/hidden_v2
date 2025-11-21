import torch
import torch.nn as nn
import json

from augmentations.valuemetric import Brightness, Contrast, DiffJPEG, GaussianBlur, Hue, Saturation
from augmentations.geometric import HorizontalFlip, Perspective, Resize, Crop, Rotate, Combine
from augmentations.splicing import PixelSplicing, BoxSplicing

class Augmenter(nn.Module):
    def __init__(self, max_steps_ratio: float = 0, id_start_prob : float = 0.8, aug_strength: dict = {}):
        super().__init__()
        self.max_steps_ratio = max_steps_ratio
        self.id_start_prob = id_start_prob
        self.t = 0
        self.aug_strength = aug_strength

        # TODO: Better idea than manually mapping?
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

        def _parse_aug_params(aug_strength):
            modules = []
            for key in self.str2class.keys():
                new_module = self.str2class[key](**aug_strength[key]) if aug_strength[key] is not None else self.str2class[key]()
                modules.append(new_module)
            
            return modules
        
        # TODO: Correctness of the sampling algorithm relies on the fact that Identity probability corresponds to last element of probability vector
        # Is that ensured by iterating over keys of self.str2class, even though Identity is specified last?
        self.augmentations = nn.ModuleList(_parse_aug_params(aug_strength))
        self.id_end_prob = 1 / len(self.augmentations)

        self.selection_probabilities_uniform = torch.ones((len(self.augmentations,)), dtype = torch.float32)
        self.selection_probabilities_uniform /= self.selection_probabilities_uniform.sum()

    def forward(self, X, X_wm, train_steps: int = 0):
        # X [B, C, H, W], assuming [-1, 1] range.
        # Applies a randomly sampled augmentation to the input batch of images
        # Output range is also [-1, 1]

        effective_steps = int(train_steps * self.max_steps_ratio)

        # Uniform augmentation sapmling
        if train_steps == 0 or self.t > effective_steps or self.max_steps_ratio == 0:
            probs = self.selection_probabilities_uniform
        
        # Temperature softmax augmentation sapmling, as time steps progress the augmentation distribution is uniformized
        else:
            current_id_prob = self.id_start_prob + (self.t / effective_steps) * (self.id_end_prob - self.id_start_prob)
            self.t += 1
            probs = torch.zeros((len(self.augmentations,)), dtype = torch.float32)
            probs[-1] = current_id_prob
            probs[:-1] = (1 - current_id_prob) / (len(self.augmentations) - 1)

        aug_index = torch.multinomial(probs, 1).item()
        # print(f"Applying augmentation: {self.augmentations[aug_index].__class__}")
        # For splicing operations, we combine the original and watermarked images.
        if isinstance(self.augmentations[aug_index], PixelSplicing) or isinstance(self.augmentations[aug_index], BoxSplicing):
            return self.augmentations[aug_index](X, X_wm)
        
        # Otherwise, we just augment the watermarked image.
        return self.augmentations[aug_index](X_wm)
    
    def to_dict(self):
        return {
            "t": self.t,
            "max_steps_ratio": self.max_steps_ratio,
            "id_start_prob": self.id_start_prob,
            "aug_strength": self.aug_strength
        }
    
    @classmethod
    def load(cls, d):
        d_tmp = {k: d[k] for k in d.keys() if k != "t"}
        res = cls(**d_tmp)
        res.t = d["t"]
        return res