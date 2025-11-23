import torch
import torch.nn as nn

from torchvision import transforms
from modules import ConvNeXtBlock, JND, ConvNeXtLayerNorm, MobileNetV2Block
from utils import normalize, unnormalize

class Embedder(nn.Module):
    def __init__(self, 
                 capacity,
                 true_resolution = 128, 
                 in_channels = 3, 
                 base_channels = 96, 
                 num_layers = 4, 
                 expansion = 4, 
                 cnext_ls = 1e-6, 
                 cnext_drop = 0.1,
                 jnd_alpha = 1.0,
                 jnd_gamma = 0.3,
                 jnd_eps = 1e-6,
                 jnd_luminance_scale = 1.0,
                 jnd_contrast_scale = 0.117,
                 conv_next_blocks = True):
        super().__init__()

        self.capacity = capacity
        self.true_resolution = true_resolution
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.expansion = expansion
        self.cnext_ls = cnext_ls
        self.cnext_drop = cnext_drop
        self.jnd_alpha = jnd_alpha
        self.jnd_gamma = jnd_gamma
        self.jnd_eps = jnd_eps
        self.jnd_luminance_scale = jnd_luminance_scale
        self.jnd_contrast_scale = jnd_contrast_scale

        # Either use ConvNeXt blocks or MobileNetV2 blocks
        self.conv_next_blocks = conv_next_blocks

        self.bottleneck_res = self.true_resolution // 2 ** num_layers
        initial_channels = in_channels
        self.initial = nn.Conv2d(in_channels, base_channels, kernel_size = 1)
        encoder_conv = []
        encoder_down = []
        in_channels = 1
        out_channels = base_channels

        for i in range(num_layers):
            in_channels = out_channels
            out_channels = 2 * in_channels
            
            current_ops = ([ConvNeXtBlock(in_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop),
                           ConvNeXtBlock(in_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop)] 
                           if conv_next_blocks
                           else 
                           [MobileNetV2Block(in_channels, in_channels, stride = 1), 
                            MobileNetV2Block(in_channels, in_channels, stride = 1)])
            
            encoder_conv.append(nn.Sequential(*current_ops))

            current_op = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2) if conv_next_blocks else nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2)
            encoder_down.append(current_op)

        self.encoder_conv = nn.ModuleList(encoder_conv)
        self.encoder_down = nn.ModuleList(encoder_down)

        current_ops = ([ConvNeXtBlock(out_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop),
                        ConvNeXtBlock(out_channels, expansion = expansion, layer_scale_init = cnext_ls, drop_prob = cnext_drop)] 
                        if conv_next_blocks
                        else 
                        [MobileNetV2Block(out_channels, out_channels, stride = 1),
                         MobileNetV2Block(out_channels, out_channels, stride = 1)])

        self.bottleneck = nn.Sequential(*current_ops)

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

            current_ops = ([ConvNeXtBlock(prev_encoder_out, expansion = expansion),
                            ConvNeXtBlock(prev_encoder_out, expansion = expansion)] 
                            if conv_next_blocks
                            else 
                            [MobileNetV2Block(prev_encoder_out, prev_encoder_out, stride = 1),
                             MobileNetV2Block(prev_encoder_out, prev_encoder_out, stride = 1)])

            decoder_conv.append(nn.Sequential(
                nn.Conv2d(new_input, prev_encoder_out, kernel_size = 1),
                *current_ops
            ))

            prev_decoder_out = prev_encoder_out
            prev_encoder_out //= 2

        self.decoder_up = nn.ModuleList(decoder_up)
        self.decoder_conv = nn.ModuleList(decoder_conv)

        self.last = nn.Conv2d(prev_decoder_out, initial_channels, kernel_size = 1)

        # Interpolation for images of variable size
        self.interp = transforms.Resize((true_resolution, true_resolution), interpolation = transforms.InterpolationMode.BILINEAR)
        # JND - Just Noticeable Difference
        self.jnd = JND(gamma = jnd_gamma, eps = jnd_eps, luminance_scale = self.jnd_luminance_scale, contrast_scale = self.jnd_contrast_scale)

    def forward(self, X, messages):
        # X => [B, in_channels, H, W], RGB image with [0, 255] quantization
        # messages => [B, capacity], binary vectors
        # Output is [B, in_channels, H, W] with [-1, 1] quantization, keep this in mind for other transformations!
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
    
    def to_dict(self):
        return {
            "args": {
                "capacity": self.capacity,
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "num_layers": self.num_layers,
                "expansion": self.expansion,
                "cnext_ls": self.cnext_ls,
                "cnext_drop": self.cnext_drop,
                "jnd_alpha": self.jnd_alpha,
                "jnd_gamma": self.jnd_gamma,
                "jnd_eps": self.jnd_eps,
                "true_resolution": self.true_resolution,
                "conv_next_blocks": self.conv_next_blocks,
                "jnd_luminance_scale": self.jnd_luminance_scale,
                "jnd_contrast_scale": self.jnd_contrast_scale
            },
            "state_dict": self.state_dict()
        }

    @classmethod
    def load(cls, d):
        res = cls(**d["args"])
        res.load_state_dict(d["state_dict"])
        return res

class Extractor(nn.Module):
    """
    Follows original ConvNeXt, in fact any given model in that paper can be constructed by giving appropriate channel_muls, base_channels and blocks 
    values. For example, ConvNeXt-T corresponds to channel_muls = (1, 2, 4, 8), blocks = (3, 3, 9, 3), base_channels = 96.

    ConvNeXt paper: https://arxiv.org/pdf/2201.03545
    """
    def __init__(self, capacity, channel_muls, blocks,
                 true_resolution = 128,
                 in_channels = 3,
                 base_channels = 96,
                 expansion = 4,
                 cnext_ls = 1e-6, 
                 cnext_drop = 0.1):
    
        super().__init__()

        self.capacity = capacity
        self.channel_muls = channel_muls
        self.blocks = blocks
        self.true_resolution = true_resolution
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.expansion = expansion
        self.cnext_ls = cnext_ls
        self.cnext_drop = cnext_drop

        self.stem = nn.Sequential(nn.Conv2d(in_channels, base_channels, kernel_size = 4, stride = 4), ConvNeXtLayerNorm(base_channels))
        modules = []
        self.true_resolution = true_resolution
        N = len(blocks)

        for i in range(N):
            for j in range(blocks[i]):
                current_layer = ConvNeXtBlock(base_channels * channel_muls[i],
                                              expansion = expansion,
                                              layer_scale_init = cnext_ls,
                                              drop_prob = cnext_drop)
                modules.append(current_layer)

            # No norm + downsampling for after last block
            if i < N - 1:
                modules.append(ConvNeXtLayerNorm(base_channels * channel_muls[i]))
                modules.append(nn.Conv2d(base_channels * channel_muls[i], base_channels * channel_muls[i + 1], kernel_size = 2, stride = 2))
        
        self.main = nn.Sequential(*modules)

        self.interp = transforms.Resize(size = (true_resolution, true_resolution), interpolation = transforms.InterpolationMode.BILINEAR)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.last_ln = nn.LayerNorm(base_channels * channel_muls[-1])
        self.pred_linear = nn.Linear(base_channels * channel_muls[-1], capacity, bias = False)
        
    def forward(self, X):
        # X [B, C, H, W], the input should be quantized to [-1, 1], will be ensured during training.
        # Make sure this holds during inference as well!
        _, _, H, W = X.shape

        if H != self.true_resolution or W != self.true_resolution:
            X = self.interp(X)

        X = self.stem(X) # [B, base_channels, H // 4, W // 4], roughly
        X = self.main(X) # [B, base_channels * channel_muls[-1], H', W'], severe downsampling
        X  = self.global_average_pooling(X).squeeze(-1).squeeze(-1) # [B, base_channels * channel_muls[-1]]
        X = self.last_ln(X)
        X = self.pred_linear(X)
        return X
    
    def to_dict(self):
        return {
            "args": {
                "capacity": self.capacity,
                "channel_muls": self.channel_muls,
                "blocks": self.blocks,
                "true_resolution": self.true_resolution,
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "expansion": self.expansion,
                "cnext_ls": self.cnext_ls,
                "cnext_drop": self.cnext_drop,
            },
            "state_dict": self.state_dict()
        }
    
    @classmethod
    def load(cls, d):
        res = cls(**d["args"])
        res.load_state_dict(d["state_dict"])
        return res