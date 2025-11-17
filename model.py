import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from modules import ConvNeXtBlock, JND
from augmentations.valuemetric import JPEG
from utils import normalize, unnormalize

class Embedder(nn.Module):
    def __init__(self, capacity, 
                 true_resolution = 256, 
                 in_channels = 3, 
                 base_channels = 96, 
                 num_layers = 4, 
                 expansion = 4, 
                 cnext_ls = 1e-6, 
                 cnext_drop = 0.1,
                 jnd_alpha = 1.0,
                 jnd_gamma = 0.3,
                 jnd_eps = 1e-6):
        super().__init__()

        self.capacity = capacity
        self.true_resolution = true_resolution
        self.bottleneck_res = self.true_resolution // 2 ** num_layers
        self.jnd_alpha = jnd_alpha

        initial_channels = in_channels
        self.initial = nn.Conv2d(in_channels, base_channels, kernel_size = 1)
        encoder_conv = []
        encoder_down = []
        in_channels = 1
        out_channels = base_channels

        for i in range(num_layers):
            in_channels = out_channels
            out_channels = 2 * in_channels

            print(f"{in_channels} {out_channels}")

            encoder_conv.append(nn.Sequential(
                ConvNeXtBlock(in_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop),
                ConvNeXtBlock(in_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop),
            ))
            encoder_down.append(nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2))

        self.encoder_conv = nn.ModuleList(encoder_conv)
        self.encoder_down = nn.ModuleList(encoder_down)

        self.bottleneck = nn.Sequential(
            ConvNeXtBlock(out_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop),
            ConvNeXtBlock(out_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop)
        )

        # TODO: Parametrize length of bit embeddings as well?
        # For now, set dimension of a single message embedding to 2 * capacity
        self.message_embeddings = nn.Embedding(2 * capacity, 2 * capacity)

        decoder_up = []
        decoder_conv = []

        prev_decoder_out = out_channels + 2 * capacity
        prev_encoder_out = in_channels

        for i in range(num_layers):
            new_input = 2 * prev_encoder_out

            decoder_up.append(nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = "bilinear"),
                nn.Conv2d(prev_decoder_out, prev_encoder_out, kernel_size = 1)
            ))

            decoder_conv.append(nn.Sequential(
                nn.Conv2d(new_input, prev_encoder_out, kernel_size = 1),
                ConvNeXtBlock(prev_encoder_out, expansion = expansion),
                ConvNeXtBlock(prev_encoder_out, expansion = expansion)
            ))

            prev_decoder_out = prev_encoder_out
            prev_encoder_out //= 2

        self.decoder_up = nn.ModuleList(decoder_up)
        self.decoder_conv = nn.ModuleList(decoder_conv)

        self.last = nn.Conv2d(prev_decoder_out, initial_channels, kernel_size = 1)

        # Interpolation for images of variable size
        self.interp = transforms.Resize((true_resolution, true_resolution), interpolation = transforms.InterpolationMode.BILINEAR)

        # JND - Just Noticeable Difference
        self.jnd = JND(gamma = jnd_gamma, eps = jnd_eps)

    def forward(self, X, messages):
        # X => [B, in_channels, H, W], RGB image with [0, 255] quantization
        # messages => [B, capacity], binary vectors
        encoder_conv_out = []
        input = X

        X = self.initial(X)

        H, W = X.shape[-2], X.shape[-1]

        to_reverse_interp = False
        if H != self.true_resolution or W != self.true_resolution:
            inverse_inter = transforms.Resize((X.shape[-2:]), interpolation = transforms.InterpolationMode.BILINEAR, antialias = True)
            X = self.interp(X)
            to_reverse_interp = True

        X = normalize(X) # Normalize to [-1, 1] before passing through the model.
        for i in range(len(self.encoder_conv)):
            X = self.encoder_conv[i](X)
            encoder_conv_out.append(X)
            X = self.encoder_down[i](X)
        
        X = self.bottleneck(X)
        message_index = torch.arange(0, self.capacity, device = X.device, dtype = torch.int32).unsqueeze(0) * 2
        message_index = message_index + messages # [B, capacity]
        embeddings = self.message_embeddings(message_index).mean(dim = 1).unsqueeze(-1).unsqueeze(-1) # Average over capacity dimension, [B, capacity, 1, 1]
        embeddings = embeddings.repeat((1, 1, self.bottleneck_res, self.bottleneck_res)) # Repeat over all spatial positions => [B, capacity, H', W']

        X = torch.cat([X, embeddings], dim = 1)
        index = len(encoder_conv_out) - 1

        for i in range(len(self.decoder_conv)):
            X = self.decoder_up[i](X)
            # Stride 2 convolution downsamples by 2, if input signal is of odd length, information is lost when upsampling
            # We expect that the corresponding encoder output is at least as long as upsampled decoder input.
            pad_amount = (encoder_conv_out[index].shape[-1] - X.shape[-1])
            pad_left = pad_amount // 2
            pad_right = pad_left + (pad_amount % 2 == 1)

            X = nn.functional.pad(X, (pad_left, pad_right, pad_left, pad_right))

            X = torch.cat([X, encoder_conv_out[index]], dim = 1)
            X = self.decoder_conv[i](X)
            index -= 1
        
        X = self.last(X)
        X = nn.functional.tanh(X)

        if to_reverse_interp:
            X = inverse_inter(X)
        
        jnd_attn = self.jnd(input)
        input_norm = normalize(input) # Watermark is in range [-1, 1], so convert the input to same range before applying it.
        # JND on the other hand has to use an input image with quantization [0, 255]!
        return input_norm + self.jnd_alpha * jnd_attn * X # MaskMark does jnd_attn (X - input), is that better?