import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, 1, 1, 1) for images
        mask = torch.rand(shape, device = x.device) < keep_prob # For some samples residual connection is kept alive, for some it's turned off.
        return x * mask / keep_prob # Preserves expectation of the input tensor 

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion = 4, layer_scale_init = 1e-6, drop_prob = 0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size = 7, padding = 3, groups = dim)  # depthwise
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)  # 1x1 conv implemented as Linear (ConvNeXt style)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_prob = drop_prob)

    def forward(self, x):
        # x: (B, C, H, W)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale

        x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)

        x = input + self.drop_path(x)
        return x

class JND(nn.Module):
    def __init__(self, 
                 gamma: float = 0.3, 
                 eps: float = 1e-6,
                 luminance_scale: float = 1.0,
                 contrast_scale: float = 0.117):
        # Luminance and contrast scaling values taken from MaskMark
        # https://github.com/hurunyi/MaskWM/blob/master/models/Mask_Model.py#L518

        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.luminance_scale = luminance_scale
        self.contrast_scale = contrast_scale

        self.luminance_kernel = nn.Parameter(torch.tensor([[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 0, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]], dtype = torch.float32).unsqueeze(0).unsqueeze(0), requires_grad = False)
        self.sobelx = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = torch.float32).unsqueeze(0).unsqueeze(0), requires_grad = False)
        self.sobely = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = torch.float32).unsqueeze(0).unsqueeze(0), requires_grad = False)
    
    def forward(self, X):
        # X [B, 3, H, W], input is assumed to be RGB with standard [0, 255] quantization
        L = (0.299 * X[:, 0, :, :] + 0.587 * X[:, 1, :, :] + 0.114 * X[:, 2, :, :]).unsqueeze(1) # convert to luminance channel [B, 1, H, W]
        LA = (1 / 32) * nn.functional.conv2d(L, self.luminance_kernel, padding = 2)
        LA_mask = LA <= 127

        LA[LA_mask] = 17 * (1 - torch.sqrt(LA[LA_mask] / 127 + self.eps)) + 3
        LA[~LA_mask] = (3 / 128) * (LA[~LA_mask] - 127) + 3
        LA *= self.luminance_scale

        edgex = nn.functional.conv2d(L, self.sobelx, padding = 1)
        edgey = nn.functional.conv2d(L, self.sobely, padding = 1)
        C = torch.sqrt(edgex ** 2 + edgey ** 2)
        C = (16 * C ** (2.4)) / (C ** 2 + 26 ** 2)
        C *= self.contrast_scale

        # gamma controls luminance masking vs contrast masking tradeoff.
        # Bigger gamma => smaller quantity between (LA, C) is suppressed more, meaning that the larger component will have bigger influence.  
        H = torch.clamp_min(LA + C - self.gamma * torch.minimum(LA, C), 0)
        H = H.repeat(1, 3, 1, 1)
        # Per-channel JND heatmaps. Eye is less sensitive to changes in B => more distortion in B channel
        # We achieve this by halving distortion coefficients in R and G channels
        H[:, 0] *= 0.5 
        H[:, 1] *= 0.5 

        # H is roughly in [0, 255] range, and we want a heatmap for modulation, so we keep values in [0, 1] roughly.
        return H / 255 # [B, 3, H, W]
    
class ConvNeXtLayerNorm(nn.Module):
    # Used in Extractor component, explicit layer for ConvNeXt style layer normalization.
    # Not used in Embedder, for now.
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
    
    def forward(self, X):
        # X [B, C, H, W]
        X = X.permute(0, 2, 3, 1) # [B, H, W, C]
        X = self.ln(X)
        X = X.permute(0, 3, 1, 2) # [B, C, H, W]
        return X
    
class MobileNetV2Block(nn.Module):
    """
    Try MobileNetV2 blocks instead of ConvNeXt blocks in embedder?
    """
    def __init__(self, in_channels, out_channels, stride = 1, expansion = 6):
        super().__init__()
        assert stride in [1, 2], f"Stride for depthwise convolution must be either 1 or 2, but {stride} given."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion = expansion
        hidden_dim = int(in_channels * expansion)
        # Residual connection without layer scale and stochastic drop path.
        # Residual connection is used iff in_channels == out_channels and stride == 1 (no downsampling in depthwise block)
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []

        # Expansion (1x1 conv)
        if expansion != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size = 1 , bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ]

        # Depthwise 3x3 conv
        layers += [
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size = 3,
                stride = stride,
                padding = 1,
                groups = hidden_dim,
                bias = False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()            
        ]

        # Projection (1x1 conv)
        layers += [
            nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        
        else:
            return self.conv(x)