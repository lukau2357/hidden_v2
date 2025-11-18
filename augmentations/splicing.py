import torch
import torch.nn as nn

from torchvision import transforms

class PixelSplicing(nn.Module):
    def __init__(self, min_p = 0.05, max_p = 0.3):
        super().__init__()
        self.min_p = min_p
        self.max_p = max_p

    def forward(self, clean, watermarked) -> torch.Tensor:
        r = torch.rand((1,)).item()
        p = (1 - r) * self.min_p + r * self.max_p
        mask = (torch.rand(clean.shape, device = clean.device) < p).to(torch.float32)
        return mask * clean + (1 - mask) * watermarked
    
class BoxSplicing(nn.Module):
    def __init__(self, min_size = None, max_size = None):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
    
    def forward(self, clean, watermarked) -> torch.Tensor:
        h, w = clean.shape[-2:]
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size = (1, )).item(), 
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size = (1, )).item()
        )

        i, j, h, w = transforms.RandomCrop.get_params(clean, output_size = output_size)
        mask = torch.zeros(clean.shape, device = clean.device)
        mask[:, :, i : i + h, j : j + w] = 1
        return mask * clean + (1 - mask) * watermarked