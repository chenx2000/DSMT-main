import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum


def uniform(a, b, shape, device='cuda'):
    return (b - a) * torch.rand(shape, device=device) + a


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class DWM_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size1=(8, 8),
            window_size2=(16, 16),
            dim_head=28,
            heads=2,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size1 = window_size1
        self.window_size2 = window_size2

        # position embedding
        seq_l1 = window_size1[0] * window_size1[1]
        self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l1, seq_l1))
        h, w = 128 // self.heads, 128 // self.heads
        seq_l2 = h * w * 4 // seq_l1
        self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l2, seq_l2))
        seq_l3 = window_size2[0] * window_size2[1]
        self.pos_emb3 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l3, seq_l3))
        h, w = 128 // self.heads, 128 // self.heads
        seq_l4 = h * w * 4 // seq_l3
        self.pos_emb4 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l4, seq_l4))

        trunc_normal_(self.pos_emb1)
        trunc_normal_(self.pos_emb2)
        trunc_normal_(self.pos_emb3)
        trunc_normal_(self.pos_emb4)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, _ = x.shape
        w_size1 = self.window_size1
        w_size2 = self.window_size2
        assert h % w_size1[0] == 0 and w % w_size1[1] == 0, 'fmap dimensions must be divisible by the window size 1'
        assert h % w_size2[0] == 0 and w % w_size2[1] == 0, 'fmap dimensions must be divisible by the window size 2'

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        _, _, _, c = q.shape
        q1, q2, q3, q4 = q[:, :, :, :c // 4], q[:, :, :, c // 4:c // 2], \
                         q[:, :, :, c // 2:c // 4 * 3], q[:, :, :, c // 4 * 3:]
        k1, k2, k3, k4 = k[:, :, :, :c // 4], k[:, :, :, c // 4:c // 2], \
                         k[:, :, :, c // 2:c // 4 * 3], k[:, :, :, c // 4 * 3:]
        v1, v2, v3, v4 = v[:, :, :, :c // 4], v[:, :, :, c // 4:c // 2], \
                         v[:, :, :, c // 2:c // 4 * 3], v[:, :, :, c // 4 * 3:]
        # local branch of window size 1
        q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
                         (q1, k1, v1))
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q1, k1, v1))
        q1 *= self.scale
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        sim1 = sim1 + self.pos_emb1
        attn1 = sim1.softmax(dim=-1)
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

        # non-local branch of window size 1
        q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
                         (q2, k2, v2))
        q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q2, k2, v2))
        q2 *= self.scale
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = out2.permute(0, 2, 1, 3)

        out_1 = torch.cat([out1, out2], dim=-1).contiguous()
        out_1 = rearrange(out_1, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size1[0], w=w // w_size1[1],
                          b0=w_size1[0])

        # local branch of window size 2
        q3, k3, v3 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),
                         (q3, k3, v3))
        q3, k3, v3 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q3, k3, v3))
        q3 *= self.scale
        sim3 = einsum('b n h i d, b n h j d -> b n h i j', q3, k3)
        sim3 = sim3 + self.pos_emb3
        attn3 = sim3.softmax(dim=-1)
        out3 = einsum('b n h i j, b n h j d -> b n h i d', attn3, v3)
        out3 = rearrange(out3, 'b n h mm d -> b n mm (h d)')

        # non-local of window size 2
        q4, k4, v4 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),
                         (q4, k4, v4))
        q4, k4, v4 = map(lambda t: t.permute(0, 2, 1, 3), (q4.clone(), k4.clone(), v4.clone()))
        q4, k4, v4 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q4, k4, v4))
        q4 *= self.scale
        sim4 = einsum('b n h i d, b n h j d -> b n h i j', q4, k4)
        sim4 = sim4 + self.pos_emb4
        attn4 = sim4.softmax(dim=-1)
        out4 = einsum('b n h i j, b n h j d -> b n h i d', attn4, v4)
        out4 = rearrange(out4, 'b n h mm d -> b n mm (h d)')
        out4 = out4.permute(0, 2, 1, 3)

        out_2 = torch.cat([out3, out4], dim=-1).contiguous()
        out_2 = rearrange(out_2, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size2[0], w=w // w_size2[1],
                          b0=w_size2[0])

        out = torch.cat([out_1, out_2], dim=-1).contiguous()
        out = self.to_out(out)

        return out


class DWMAB(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            shift_size=0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.attn = PreNorm(dim, DWM_MSA(dim=dim, dim_head=dim, heads=heads))
        self.dim = dim
        self.heads = heads
        self.ffn = PreNorm(dim, FeedForward(dim=dim))
        self.shift_size = shift_size
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Conditional position embedding
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x) + x
        x = x.permute(0, 2, 3, 1)

        # Attention calculation
        x = self.attn(x) + x

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = self.ffn(x) + x

        return x


class DWMABs(nn.Module):
    def __init__(
            self,
            dim,
            shift_size=8,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(DWMAB(heads=heads, dim=dim, shift_size=0 if (_ % 2 == 0) else shift_size))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return x: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Coarse_Feature_Extraction(nn.Module):
    def __init__(self, dim=28, expand=2):
        super(Coarse_Feature_Extraction, self).__init__()
        self.dim = dim
        self.stage = 2

        # Input projection
        self.in_proj = nn.Conv2d(28, dim, 1, 1, 0, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(self.stage):
            self.encoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                nn.Conv2d(dim_stage * expand, dim_stage * expand, 1, 1, 0, bias=False),
            ]))
            dim_stage *= 2

        # The modules to achieve full-scale skip connections
        self.downsample = nn.AdaptiveAvgPool2d(128)
        self.conv_skip = nn.Conv2d(28, 56, 3, 1, 1)

        # Bottleneck
        self.bottleneck = ASPP(dim_stage, [3, 6], dim_stage)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.bottle_conv_skip = nn.Conv2d(dim_stage, dim_stage // 4, 3, 1, 1)

        # Decoder:
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(x))

        # Encoder
        fea_encoder = []
        fea_encoder_skip_layer = []
        for i, (Conv1, Conv2, Conv3) in enumerate(self.encoder_layers):
            fea_encoder.append(fea)
            if i == 0:
                fea_encoder_skip_layer.append(self.conv_skip(self.downsample(fea)))
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))

        # Bottleneck
        fea_bottleneck = []
        fea_bottleneck.append(self.bottle_conv_skip(self.upsample(fea)))
        fea = self.bottleneck(fea) + fea

        # Decoder
        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage - 1 - i]
            if i == 0:
                fea = fea + fea_encoder_skip_layer[0]
            if i == 1:
                fea = fea + fea_bottleneck[0]

        # Output projection
        out = self.out_conv(fea)
        return out


# Cross attention fusion module
class CAFM(nn.Module):
    def __init__(self, channel=112):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.channel = channel

        self.fusion = nn.Conv2d(channel * 2, channel, 1, 1, 0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = rearrange(f1, 'b n (h d) -> b n h d', h=256 * 28 // self.channel, d=256 * 28 // self.channel)
        f2 = rearrange(f2, 'b n (h d) -> b n h d', h=256 * 28 // self.channel, d=256 * 28 // self.channel)

        out = self.fusion(torch.cat((f1, f2), dim=1))
        return out


class DSMT(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=[2, 4, 6]):
        super(DSMT, self).__init__()
        self.dim = dim
        self.stage = stage

        # Fution physical mask and shifted measurement
        self.fution = nn.Conv2d(56, 28, 1, 1, 0, bias=False)

        # Coarse Feature Extraction
        self.fe_1 = Coarse_Feature_Extraction(dim=28, expand=2)
        self.fe_2 = Coarse_Feature_Extraction(dim=28, expand=2)
        self.original_fusion = nn.Conv2d(56, 28, 1, 1, 0, bias=False)

        # First branch of encoder
        self.encoder_layers_1 = nn.ModuleList([])
        dim_stage = dim
        self.encoder_conv = nn.ModuleList([])
        for i in range(stage):
            self.encoder_layers_1.append(nn.ModuleList([
                DWMABs(dim=dim_stage, num_blocks=num_blocks[i], heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            self.encoder_conv.append(nn.Conv2d(dim_stage * 2, dim_stage, 1, 1, 0))
            dim_stage *= 2

        # Second branch of encoder
        self.encoder_layers_2 = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers_2.append(nn.ModuleList([
                DWMABs(dim=dim_stage, num_blocks=num_blocks[i], heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Cross attention fusion module
        self.CAFM = CAFM()

        # The modules to achieve full-scale skip connections
        self.downsample = nn.AdaptiveAvgPool2d(128)
        self.conv_skip = nn.Conv2d(28, 56, 3, 1, 1)

        # Bottleneck
        self.bottleneck = DWMABs(dim=dim_stage, heads=dim_stage // dim, num_blocks=num_blocks[-1])
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.bottle_conv = nn.Conv2d(dim_stage, dim_stage // 4, 3, 1, 1)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                DWMABs(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim, dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        mask: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Fution physical mask and shifted measurement
        x = self.fution(torch.cat([x, mask], dim=1))

        # Coarse feature extraction
        fea = self.fe_1(x)
        fea_second_branch = torch.cat((fea, x), dim=1)
        fea_second_branch = self.original_fusion(fea_second_branch)
        fea_second_branch = self.fe_2(fea_second_branch)

        # This will be added to the final residual
        fea_final = fea

        # First branch of encoder
        fea_encoder_1 = []
        i = 0
        for (Blcok, FeaDownSample) in self.encoder_layers_1:
            fea = Blcok(fea)
            fea_encoder_1.append(fea)
            fea = FeaDownSample(fea)
            i = i + 1

        # Second branch of encoder
        fea_encoder_2 = []
        for (Blcok, FeaDownSample) in self.encoder_layers_2:
            fea_second_branch = Blcok(fea_second_branch)
            fea_encoder_2.append(fea_second_branch)
            fea_second_branch = FeaDownSample(fea_second_branch)

        fea_encoder = []
        for j in range(i):
            fea_res = torch.cat((fea_encoder_1[j], fea_encoder_2[j]), dim=1)
            fea_res = self.encoder_conv[j](fea_res)
            fea_encoder.append(fea_res)
            if j == 0:
                fea_encoder_skip = self.conv_skip(self.downsample(fea_res))

        # Fusion the outputs of the two branches
        fea = self.CAFM(fea, fea_second_branch)

        # Bottleneck
        fea = self.bottleneck(fea)
        fea_bottleneck = self.upsample(fea)
        fea_bottleneck = self.bottle_conv(fea_bottleneck)

        # Decoder
        for i, (FeaUpSample, Blcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = fea + fea_encoder[self.stage - 1 - i]
            if i == 0:
                fea = fea + fea_encoder_skip
            if i == 1:
                fea = fea + fea_bottleneck
            fea = Blcok(fea)

        fea = fea_final + fea
        # Output projection
        out = self.out_proj(fea) + x
        return out