"""Pure PyTorch implementation of RRDBNet for Real-ESRGAN super-resolution.
No external compilation or C++ extensions required.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features=64, num_grow_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_grow_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + num_grow_channels, num_grow_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + num_grow_channels * 2, num_grow_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + num_grow_channels * 3, num_grow_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + num_grow_channels * 4, num_features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, num_features, num_grow_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_features, num_grow_channels)
        self.rdb2 = ResidualDenseBlock(num_features, num_grow_channels)
        self.rdb3 = ResidualDenseBlock(num_features, num_grow_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block.
    Default parameters correspond to RealESRGAN_x4plus (num_block=23, num_feat=64).
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Upsample 2x
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # Upsample 4x
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def tile_sr_inference(model, tensor, tile_size=384, overlap=32, target_scale=4, progress_cb=None):
    """Run super-resolution inference on GPU using overlapping tiles with progress callbacks.
    Args:
        model: RRDBNet model
        tensor: (1, 3, H, W) normalized to [0, 1] on GPU
        tile_size: input tile size
        overlap: pixel overlap between input tiles
        target_scale: 4 for RRDBNet
        progress_cb: optional callable(tile_index, total_tiles)
    Returns:
        (1, 3, H*scale, W*scale) tensor on GPU
    """
    b, c, h, w = tensor.shape
    scale = target_scale
    out_h, out_w = h * scale, w * scale

    if h <= tile_size and w <= tile_size:
        if progress_cb is not None:
            progress_cb(0, 1)
        res = model(tensor).clamp(0, 1)
        if progress_cb is not None:
            progress_cb(1, 1)
        return res

    output = torch.zeros((b, c, out_h, out_w), device=tensor.device, dtype=tensor.dtype)
    weights = torch.zeros((b, 1, out_h, out_w), device=tensor.device, dtype=tensor.dtype)

    step = tile_size - overlap

    def make_weight_window(th, tw, dev, dt):
        wy = torch.sin(torch.linspace(0.05, math.pi - 0.05, th * scale, device=dev, dtype=dt)) ** 2
        wx = torch.sin(torch.linspace(0.05, math.pi - 0.05, tw * scale, device=dev, dtype=dt)) ** 2
        return (wy[:, None] * wx[None, :])[None, None]

    y_starts = list(range(0, h - tile_size + 1, step))
    if not y_starts or y_starts[-1] != h - tile_size:
        y_starts.append(max(0, h - tile_size))
    y_starts = sorted(set(y_starts))

    x_starts = list(range(0, w - tile_size + 1, step))
    if not x_starts or x_starts[-1] != w - tile_size:
        x_starts.append(max(0, w - tile_size))
    x_starts = sorted(set(x_starts))

    total_tiles = len(y_starts) * len(x_starts)
    curr_tile = 0

    for y in y_starts:
        for x in x_starts:
            if progress_cb is not None:
                progress_cb(curr_tile, total_tiles)
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)
            patch = tensor[:, :, y:y + th, x:x + tw]
            sr_patch = model(patch).clamp(0, 1)

            out_y, out_x = y * scale, x * scale
            out_th, out_tw = th * scale, tw * scale

            weight = make_weight_window(th, tw, tensor.device, tensor.dtype)
            output[:, :, out_y:out_y + out_th, out_x:out_x + out_tw] += sr_patch * weight
            weights[:, :, out_y:out_y + out_th, out_x:out_x + out_tw] += weight
            curr_tile += 1

    if progress_cb is not None:
        progress_cb(total_tiles, total_tiles)

    output /= weights.clamp_min(1e-7)
    return output.clamp(0, 1)
