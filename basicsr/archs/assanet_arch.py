import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange
# for idynamic
from basicsr.archs.idynamic_dwconv import IDynamicDWConv

# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, use_pixelunshuffle=False, bias=False):    # for better performance and less params we set bias=False
        super(OverlapPatchEmbed, self).__init__()
        if use_pixelunshuffle:
            self.proj = nn.Sequential(nn.PixelUnshuffle(2),
                nn.Conv2d(in_c * 4, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias))
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
# ---------------------------------------------------------------------------------------------------------------------

# FFN
class FeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        mask = x > 0

        x = self.softmax(x)

        x = torch.where(mask > 0, x, torch.zeros_like(x))

        return x

class TopK(nn.Module):
    def __init__(self):
        super(TopK, self).__init__()

    def forward(self, x):
        b, h, C, _ = x.shape

        mask = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)

        index = torch.topk(x, k=int(C/4), dim=-1, largest=True)[1]

        mask.scatter_(-1, index, 1.)

        result = torch.where(mask > 0, x, torch.zeros_like(x))

        return result

# Sparse Self-Attention
class SparseSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_flag=True, tlc_kernel=48, activation='relu'):
        super(SparseSelfAttention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_in = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.dynamic_conv = IDynamicDWConv(dim * 2, kernel_size=3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif activation == 'maskedsoftmax':
            self.act = MaskedSoftmax()
        elif activation == 'topk':
            self.act = TopK()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qv):
        q, v = qv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(v, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = self.act(attn)

        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qv = self.dynamic_conv(self.project_in(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qv = self.grids(qv)  # convert to local windows
        out = self._forward(qv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qv.shape[-2], w=qv.shape[-1])
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 2, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
# ---------------------------------------------------------------------------------------------------------------------

class AttBlock(nn.Module):
    def __init__(self, dim, num_heads=6, ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu'):
        super(AttBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.attn = SparseSelfAttention(dim, num_heads=num_heads, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, bias=False)
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

# ---------------------------------------------------------------------------------------------------------------------

# BuildBlocks
class BuildBlock(nn.Module):
    def __init__(self, dim, blocks, num_heads=6, ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu'):
        super(BuildBlock, self).__init__()

        body = [AttBlock(dim, num_heads, ffn_expansion_factor, tlc_flag, tlc_kernel, activation) for _ in range(blocks)]
        body.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x
# ---------------------------------------------------------------------------------------------------------------------

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, use_pixelunshuffle=False, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        if use_pixelunshuffle:
            m.append(nn.Conv2d(num_feat, num_out_ch * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        else:
            m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
            m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)
# ---------------------------------------------------------------------------------------------------------------------

@ARCH_REGISTRY.register()
class ASSANet(nn.Module):
    def __init__(self, dim, num_blocks, num_layers, upscale, num_heads, in_chans=3, ffn_expansion_factor=2., img_range=1., use_pixelunshuffle=False, tlc_flag=True, tlc_kernel=48, activation='relu'):
        super().__init__()

        # MeanShift for Image Input
        self.img_range = img_range
        self.use_pixelunshuffle = use_pixelunshuffle
        self.tlc_kernel = tlc_kernel
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale

        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, use_pixelunshuffle, bias=False))

        body = [BuildBlock(dim, num_blocks, num_heads, ffn_expansion_factor, tlc_flag, tlc_kernel, activation) for _ in range(num_layers)]

        body.append(nn.Conv2d(dim, dim, 3, 1, 1))

        self.deep_feature_extraction = nn.Sequential(*body)

        self.upsample = UpsampleOneStep(upscale, dim, in_chans, use_pixelunshuffle)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def spatial_padding(self, x, pad_scale):
        _, _, h, w = x.size()
        pad_h = (pad_scale - h % pad_scale) % pad_scale
        pad_w = (pad_scale - w % pad_scale) % pad_scale
        return F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        _, _, h, w = x.size()

        if self.use_pixelunshuffle:
            if h % 2 != 0 or w % 2 != 0:
                x = self.spatial_padding(x, 2)

        x = self.overlap_embed(x)
        x = self.deep_feature_extraction(x) + x
        x = self.upsample(x)

        if self.use_pixelunshuffle:
            x = x[:, :, :h, :w]

        x = x / self.img_range + self.mean
        return x

if __name__== '__main__':
    from thop import profile
    
    device = torch.device("cuda")
    x = (torch.rand((1, 3, 240, 240),device=device),)

    model = ASSANet(dim=90, num_blocks=4, num_layers=8, upscale=4, num_heads=1, ffn_expansion_factor=2., img_range=1., use_pixelunshuffle=False, tlc_flag=False, tlc_kernel=96, activation='relu').cuda()
    flops, params = profile(model, inputs=(x))
    print(flops/1e9)
    print(params)