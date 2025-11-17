import torch
import io

from torchvision import transforms
from PIL import Image

def normalize(X: torch.Tensor):
    return torch.clamp(X / 127.5 - 1, -1, 1)

def unnormalize(X: torch.Tensor):
    return torch.clamp((X + 1) * 127.5, 0, 255)

def jpeg_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    # Adapted from: https://github.com/facebookresearch/watermark-anything/blob/main/watermark_anything/augmentation/valuemetric.py#L23
    # assert image.min() >= 0 and image.max() <= 1, f'Image pixel values must be in the range [0, 1], got [{image.min()}, {image.max()}]'
    pil_image = transforms.ToPILImage()(image)  # convert to PIL image
    # Create a BytesIO object and save the PIL image as JPEG to this object
    buffer = io.BytesIO()
    pil_image.save(buffer, format = "JPEG", quality = quality)
    # Load the JPEG image from the BytesIO object and convert back to a PyTorch tensor
    buffer.seek(0)  
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image