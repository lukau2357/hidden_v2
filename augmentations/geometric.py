import torch
import torch.nn as nn

from torchvision import transforms
from augmentations.valuemetric import DiffJPEG, Brightness

# Most augmentations taken from: https://github.com/facebookresearch/watermark-anything/blob/main/watermark_anything/augmentation/geometric.py#L1

class Rotate(nn.Module):
    def __init__(self, min_angle = None, max_angle = None):
        super(Rotate, self).__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle

    def forward(self, image):
        # Input is assumed to be [-1, 1]
        angle = torch.randint(self.min_angle, self.max_angle + 1, size = (1,)).item()
        image = transforms.functional.rotate(image, angle, fill = -1)
        return image

class Resize(nn.Module):
    def __init__(self, min_size = None, max_size = None):
        super(Resize, self).__init__()
        # Values in [0, 1] imply downsampling
        # Values greater than 1 imply upsampling
        self.min_size = min_size  
        self.max_size = max_size

    def forward(self, image):
        h, w = image.shape[-2:]
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size = (1, )).item(), 
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size = (1, )).item()
        )
        image = transforms.functional.resize(image, output_size, antialias = True)
        return image

class Crop(nn.Module):
    def __init__(self, min_size = None, max_size = None):
        super(Crop, self).__init__()
        # Size of a random crop, relative to the size of the input image.
        self.min_size = min_size
        self.max_size = max_size
    
    def forward(self, image):
        h, w = image.shape[-2:]
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size = (1, )).item(), 
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size = (1, )).item()
        )

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size = output_size)
        image = transforms.functional.crop(image, i, j, h, w)
        return image
    
class Perspective(nn.Module):
    def __init__(self, min_distortion_scale = None, max_distortion_scale = None):
        super(Perspective, self).__init__()
        # Distortion scales should be given in [0, 1] range
        self.min_distortion_scale = min_distortion_scale
        self.max_distortion_scale = max_distortion_scale

    def get_random_distortion_scale(self):
        return self.min_distortion_scale + torch.rand(1).item() * \
            (self.max_distortion_scale - self.min_distortion_scale)

    def forward(self, image):
        distortion_scale = self.get_random_distortion_scale()
        width, height = image.shape[-1], image.shape[-2]
        startpoints, endpoints = self.get_perspective_params(width, height, distortion_scale)
        image = transforms.functional.perspective(image, startpoints, endpoints, fill = -1) # Input image is assumed to be in [-1, 1], so -1 corresponds to black.
        return image

    @staticmethod
    def get_perspective_params(width, height, distortion_scale):
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size = (1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size = (1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size = (1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size = (1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size = (1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size = (1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size = (1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size = (1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

class HorizontalFlip(nn.Module):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def forward(self, image):
        image = transforms.functional.hflip(image)
        return image
    
class Combine(nn.Module):
    def __init__(self, min_quality = None, max_quality = None, min_brightness_factor = None, max_brightness_factor = None, min_distortion_scale = None, max_distortion_scale = None):
        # Perspective wrap + Brightness + JPEG
        super(Combine, self).__init__()
        self.jpeg = DiffJPEG(min_quality = min_quality, max_quality = max_quality)
        self.brightness = Brightness(min_factor = min_brightness_factor, max_factor = max_brightness_factor)
        self.perspective = Perspective(min_distortion_scale = min_distortion_scale, max_distortion_scale = max_distortion_scale)

    def forward(self, image):
        # jpeg, brightness, perspective = params
        # Simulates taking a picture with a camera
        image = self.perspective(image)
        image = self.brightness(image)
        image = self.jpeg(image)
        return image