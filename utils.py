import torch
import io
import torch.nn as nn

from torchvision import transforms
from PIL import Image

def normalize(X: torch.Tensor):
    return torch.clamp(X / 127.5 - 1, -1, 1)

def unnormalize(X: torch.Tensor):
    return torch.clamp((X + 1) * 127.5, 0, 255)

def diff_round(x: torch.Tensor):
    """
    Approximation to round(x) that yields sound derivatives. It can easily verified that this function is differentiable on segments 
    (k - 0.5, k + 0.5) for any integer k (it's a simple polynomial on this interval). Furthermore, it can be shown that it is NOT 
    differentiable in the points of the form k + 0.5, for integer k. The set of points where this function is not differentiable is countable,
    so it's not an issue for practice. Furthermore, let x in (k - 0.5, k + 0.5) for fixed integer k. Then:

    f(x) = k + (x - k)^3
    f'(x) = 3 (x - k)^2 = 0 => x = k. 

    On every interval of the form (k - 0.5, k + 0.5) the derivative is zero only in k. The number of points where the derivative is 0 is infinite, 
    but this set is also countable (same cardinality as Z), therefore we can say that the derivative of this function is non-zero almost everywhere!

    This is a very convenient approximation for quantization step in JPEG. A common augmentation/attack strategy for image watermarking models is JPEG,
    but the original JPEG uses ordinary quantization with floor(x), which would lead to having 0 derivatives on all points where floor(x) is differentiable,
    stopping gradient flow during backpropagation.

    The paper that introduced this approximation: 
    https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf
    """
    r = torch.round(x)
    return r + (x - r) ** 3

def quantize(X, q, quality):
    # X [B, Hb, Wb, 8, 8] given channel divided into 8x8 blocks
    # q [8, 8] quantization matrix. For luminance we use QY, for chrome channels we use QC
    # Returns [B, Hb, Wb, 8, 8] tensor.
    # quality to scale formula
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    # modified quantization matrix
    q = ((scale * q + 50) / 100).view((1, 1, 1, 8, 8))
    
    X = diff_round(X / q)
    return X

def dequantize(X, q, quality):
    # Inverse of the previous operation, modulo rounding.
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = ((scale * q + 50) / 100).view((1, 1, 1, 8, 8))
    return X * q

def pad_to_block_size(X, block = 8):
    # X [B, H, W], one of Y, Cb, Cr channels (last 2 chanels are subsampled)
    # If needed, width and height are padded with replication to multiple of 8.
    B, H, W = X.shape
    # JPEG implementations in practice performs replication padding.
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    X = nn.functional.pad(X, (0, pad_w, 0, pad_h), mode = "replicate")
    return X

def img_to_blocks(X, block = 8):
    # X [B, H, W]
    # Result is [B, H / 8, W / 8, 8, 8]. It's guaranteed that H and W will be multiples of 8
    # because the preceding call will be pad_to_block_size
    X = pad_to_block_size(X, block = block)
    B, H, W = X.shape

    # X.unfold(dim, size, step) - takes slices of the input tensor in the given dimension of length size with given step size
    # New appropriate dimension is appended, and the input dim is shredded accordingly.
    X = X.unfold(1, block, block).unfold(2, block, block)
    # shape: (B, H//8, W//8, 8, 8)
    return X

def blocks_to_img(X, H_orig, W_orig, block = 8):
    # X [B, Hb, Wb, block, block]
    B, Hb, Wb, _, _ = X.shape
    X = X.reshape((B, Hb * Wb, block ** 2)).permute((0, 2, 1)) # [B, block ** 2, Hb * Wb]
    X = nn.functional.fold(X, output_size = (Hb * 8, Wb * 8), kernel_size = (8, 8), stride = (8, 8)) # [B, C, H, W]
    X = X[:, 0, :H_orig, :W_orig] # C = 1 since this function is only called for each YCbCr channel during JPEG
    return X

def psnr(X: torch.Tensor, Y: torch.Tensor):
    # Assumes [-1, 1] normalization of inputs
    return 10 * torch.log10(1 / ((X - Y) ** 2).mean())