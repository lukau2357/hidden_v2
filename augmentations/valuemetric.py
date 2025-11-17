import torch
import torch.nn as nn
from utils import jpeg_compress, normalize, unnormalize

class JPEG(nn.Module):
    # Adapted from: https://github.com/facebookresearch/watermark-anything/blob/main/watermark_anything/augmentation/valuemetric.py#L23
    def __init__(self, min_quality = None, max_quality = None, passthrough = True):
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.passthrough = passthrough

    def jpeg_single(self, image, quality):
        return jpeg_compress(image, quality).to(image.device)

    def forward(self, image: torch.tensor):
        quality = torch.randint(self.min_quality, self.max_quality + 1, size = (1, )).item()
        # image = unnormalize(image).clamp(0, 255)

        if len(image.shape) == 4:  # batched compression
            for ii in range(image.shape[0]):
                image[ii] = self.jpeg_single(image[ii], quality)
        else:
            image = self.jpeg_single(image, quality)

        return normalize(image)