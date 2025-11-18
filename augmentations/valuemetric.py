import torch
import torch.nn as nn
import numpy as np
import math

from utils import quantize, dequantize, img_to_blocks, blocks_to_img, normalize, unnormalize
from torchvision import transforms

# Most augmentations (except for DiffJPEG) are taken from WAM implementation:
# https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/watermark_anything/augmentation/valuemetric.py

# We omit Median filtering because it's not CUDA friendly
# WAM implementation of median filtering: https://github.com/facebookresearch/watermark-anything/blob/main/watermark_anything/utils/image.py#L59

class DiffJPEG(nn.Module):
    def __init__(self, min_quality = 40, max_quality = 90):
        super().__init__()

        self.min_quality = min_quality
        self.max_quality = max_quality

        self.rgb_2_ycbcr_map = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],dtype = np.float32).T
        self.ycbcr_2_rgb_map = np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype = np.float32).T
        self.rgb2ycbcb_shift = [0., 128.0, 128.0]

        self.rgb_2_ycbcr_map = nn.Parameter(torch.tensor(self.rgb_2_ycbcr_map), requires_grad = False)
        self.ycbcr_2_rgb_map = nn.Parameter(torch.tensor(self.ycbcr_2_rgb_map), requires_grad = False)
        self.rgb_2_ycbcr_shift = nn.Parameter(torch.tensor(self.rgb2ycbcb_shift), requires_grad = False)

        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = (2, 2))

        def create_dct_matrix(N = 8):
            # Creates an NxN orthonormal DCT-II matrix.
            D = np.zeros((N, N), dtype = np.float32)
            for k in range(N):
                for n in range(N):
                    D[k, n] = math.cos((math.pi / N) * (n + 0.5) * k)

            # Ensures that inverse of D is D^T, convenient for inverse DCT
            D[0, :] *= 1 / math.sqrt(N)
            D[1:, :] *= math.sqrt(2 / N)
            return D
        
        self.DCT = nn.Parameter(torch.tensor(create_dct_matrix()), requires_grad = False)
        
        # JPEG quantization tables, higher frequencies get quantized more aggresively.
        # Luminance and chroma channels use separate quantization tables.
        self.QY = nn.Parameter(torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype = torch.float32), requires_grad = False)

        self.QC = nn.Parameter(torch.tensor([
            [17, 18, 24, 47, 99, 99, 99,99],
            [18, 21, 26, 66, 99, 99, 99,99],
            [24, 26, 56, 99, 99, 99, 99,99],
            [47, 66, 99, 99, 99, 99, 99,99],
            [99, 99, 99, 99, 99, 99, 99,99],
            [99, 99, 99, 99, 99, 99, 99,99],
            [99, 99, 99, 99, 99, 99, 99,99],
            [99, 99, 99, 99, 99, 99, 99,99]
        ], dtype = torch.float32), requires_grad = False)

    def rgb_2_ycbcr(self, X: torch.Tensor):
        # X => [B, C, H, W], RGB images with [0, 255] quantization?
        # Returns [B, C, H, W]
        X = X.permute(0, 2, 3, 1) # [B, H, W, C]
        # https://docs.pytorch.org/docs/stable/generated/torch.tensordot.html
        X = torch.tensordot(X, self.rgb_2_ycbcr_map, dims = 1) + self.rgb_2_ycbcr_shift
        # print(X.shape)
        X = X.permute(0, 3, 1, 2)
        return X
    
    def ycbcr_2_rgb(self, X : torch.Tensor):
        X = X.permute(0, 2, 3, 1)
        X = X - self.rgb_2_ycbcr_shift
        X = torch.tensordot(X, self.ycbcr_2_rgb_map, dims = 1)
        X = X.permute(0, 3, 1, 2)
        return X
    
    def chroma_subsampling(self, X : torch.Tensor):
        # Downsample height and width of chroma channels by 2 with an averege pooling operation
        # Luminance is untouched.
        cb_subsampled = self.avg_pool(X[:, 1:2]).squeeze(1)
        cr_subsampled = self.avg_pool(X[:, 2:3]).squeeze(1)

        return X[:, 0], cb_subsampled, cr_subsampled

    def dct(self, X):
        # For 8x8 blocks this might work better than fast DCT
        return self.DCT @ X @ self.DCT.T

    def idct(self, X):
        # Same argument for iDCT
        return self.DCT.T @ X @ self.DCT
    
    def compress(self, X, quality):
        B, C, H, W = X.shape

        X = self.rgb_2_ycbcr(X)
        Y, Cb, Cr = self.chroma_subsampling(X)

        Y = img_to_blocks(Y)
        Cb = img_to_blocks(Cb)
        Cr = img_to_blocks(Cr)

        Y = self.dct(Y)
        Cb = self.dct(Cb)
        Cr = self.dct(Cr)

        Y = quantize(Y, self.QY, quality)
        Cb = quantize(Cb, self.QC, quality)
        Cr = quantize(Cr, self.QC, quality)

        Y = blocks_to_img(Y, H, W)
        Cb = blocks_to_img(Cb, H, W)
        Cr = blocks_to_img(Cr, H, W)

        return Y, Cb, Cr
    
    def decompress(self, Y, Cb, Cr, quality, H_orig, W_orig):
        Y = img_to_blocks(Y)
        Cb = img_to_blocks(Cb)
        Cr = img_to_blocks(Cr)

        Y = dequantize(Y, self.QY, quality)
        Cb = dequantize(Cb, self.QC, quality)
        Cr = dequantize(Cr, self.QC, quality)

        Y = self.idct(Y)
        Cb = self.idct(Cb)
        Cr = self.idct(Cr)

        Y = blocks_to_img(Y, H_orig, W_orig)
        Cb = blocks_to_img(Cb, H_orig, W_orig)
        Cr = blocks_to_img(Cr, H_orig, W_orig)

        Cb = nn.functional.interpolate(Cb.unsqueeze(1), scale_factor = 2, mode = "bilinear").squeeze(1)
        Cr = nn.functional.interpolate(Cr.unsqueeze(1), scale_factor = 2, mode = "bilinear").squeeze(1)

        X = torch.stack([Y, Cb, Cr], dim = 1)
        return self.ycbcr_2_rgb(X)

    def forward(self, X):
        # X [B, C, H, W], batch of images
        quality = torch.randint(self.min_quality, self.max_quality + 1, size = (1,)).item()
        B, C, H, W = X.shape
        # TODO: JPEG requires [0, 255] quantization. This assumes that the input is always [-1, 1]?
        # Check if the input is in satisfiable range before applying reverse transformation?
        X = unnormalize(X) 
        Y, Cb, Cr = self.compress(X, quality)
        out = self.decompress(Y, Cb, Cr, quality, H, W)
        return normalize(out)
    
class GaussianBlur(nn.Module):
    def __init__(self, min_kernel_size = None, max_kernel_size = None):
        super(GaussianBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    def forward(self, image, kernel_size = None):
        # Can operate on [-1, 1] inputs as well.
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1, )).item()
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size # Always use odd sized kernels
        # Convolution with 2D Gaussian filter + reflection padding to preserve input shape.
        image = transforms.functional.gaussian_blur(image, kernel_size)
        return image

class Brightness(nn.Module):
    def __init__(self, min_factor = None, max_factor = None):
        super(Brightness, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image, factor = None):
        image = unnormalize(image) / 255.0
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = transforms.functional.adjust_brightness(image, factor) # Returns [0, 1] range, and also expects input in the same range.
        # Not quite in line with our [-1, 1] normalization!
        return normalize(image * 255)

class Contrast(nn.Module):
    def __init__(self, min_factor = None, max_factor = None):
        super(Contrast, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image, factor = None):
        # Also expects [0, 1] inputs and returns [0, 1 outputs]
        image = unnormalize(image) / 255.0
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = transforms.functional.adjust_contrast(image, factor)
        return normalize(image * 255)

class Saturation(nn.Module):
    def __init__(self, min_factor = None, max_factor = None):
        super(Saturation, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image):
        image = unnormalize(image) / 255.0
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = transforms.functional.adjust_saturation(image, factor)
        return normalize(image * 255)

class Hue(nn.Module):
    def __init__(self, min_factor = None, max_factor = None):
        super(Hue, self).__init__()
        assert min_factor <= max_factor and min_factor >= -0.5 and max_factor <= 0.5, f"Hue factor has to be in range [-0.5, 0.5], but [{min_factor}, {max_factor}] was given."
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image):
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = unnormalize(image) / 255
        image = transforms.functional.adjust_hue(image, factor) # factor must be in range [-0.5, 0.5]
        return normalize(image * 255)
