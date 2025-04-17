import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm.auto import tqdm
from einops import rearrange, repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import einsum
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from datasets import load_dataset

from PIL import Image
import requests

from IPython import display
import pylab as pl


# ヘルパー関数
def exists(x):
    return x is not None


class Residual(nn.Module):
    """ラップした層にskip connectionを追加するクラス"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    """ラップした層の前にgroup normalization [7]を追加するクラス"""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out if exists(dim_out) else dim, kernel_size=3, padding=1),
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(
        dim, dim_out if exists(dim_out) else dim, kernel_size=4, stride=2, padding=1
    )

# Positional Encoding
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """整数のタイムステップを受け取り，そのembeddingベクトルを返す．
        Args:
            time ( b, ): 拡散過程のタイムステップ．Positional encoding式のposに対応
        Returns:
            pos_enc ( b, time_emb_dim ): Positional encoding
        """
        half_dim = self.dim // 2

        pos_enc = math.log(10000) / (half_dim - 1)  # scalar
        pos_enc = torch.exp(torch.arange(half_dim, device=time.device) * -pos_enc)  # ( time_emb_dim // 2, )
        pos_enc = time[:, None] * pos_enc[None, :]  # ( batch, 1 ) * ( 1, time_emb_dim // 2 ) -> ( batch, time_emb_dim // 2 )
        pos_enc = torch.cat((pos_enc.sin(), pos_enc.cos()), dim=-1)  # ( batch, time_emb_dim )

        return pos_enc
    
    class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """
        Args:
            x ( b, dim, h', w' ): UNetの中間表現
        Returns:
            x ( b, dim_out, h', w' ): 時間情報が追加された中間表現
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

# Wide ResNet Block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Args:
            x ( b, dim, h', w' ): UNetの中間表現
            time_emb ( b, time_emb_dim ): 拡散過程のタイムステップ情報
        Returns:
            x ( b, dim_out, h', w' )
        """
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)  # ( b, dim_out * 2 )
            time_emb = rearrange(time_emb, "b c -> b c 1 1")  # ( b, dim_out * 2, 1, 1 )
            scale_shift = time_emb.chunk(2, dim=1)  # tuple(( b, dim_out, 1, 1 ), ( b, dim_out, 1, 1 ))

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
    
# Attention module
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x ( b, dim, h', w' )
        Returns:
            out ( b, dim, h', w' )
        """
        b, c, h, w = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.heads)
        k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.heads)
        v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.heads)
        # ( b, heads, dim_head, h(=x) * w(=y) )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out)

# モデルの定義
class Unet(nn.Module):
    def __init__(
        self,
        dim, # image_size
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        # 最初の畳み込み
        init_dim = init_dim if exists(init_dim) else dim // 3 * 2
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=7, padding=3)

        # Wide ResNet ブロック（functools.partial()でgroupsだけ固定して使い回す）
        ResnetBlock_ = partial(ResnetBlock, groups=resnet_block_groups)

        # position (time) encoding
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEncoding(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # 各層の入力・出力次元数のリスト
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_resolutions = len(in_out)

        # エンコーダ
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList(
                    [
                        ResnetBlock_(dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock_(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # ボトルネック部分
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock_(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock_(mid_dim, mid_dim, time_emb_dim=time_dim)

        # デコーダ
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (num_resolutions - 1)

            self.ups.append(nn.ModuleList(
                    [
                        ResnetBlock_(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        ResnetBlock_(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # 最後の畳み込み
        out_dim = out_dim if exists(out_dim) else channels
        self.final_res_block = ResnetBlock_(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, out_dim, 1)

    def forward(self, x, time):  # time: ( b, )
        b = x.shape[0]

        x = self.init_conv(x)
        r = x.clone()

        # Embedされた時間情報
        t = self.time_mlp(time) if exists(self.time_mlp) else None # ( B, time_emb_dim )

        # エンコーダ-デコーダ間のskip connection
        h = []

        # エンコーダ
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t) # 全レイヤーに対して時間情報を与えている
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # ボトルネック
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # デコーダ
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        return self.final_conv(x)

# dataset　dataloader


# 学習の実装
timesteps = 800 # T

# 様々なbeta scheduling

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def cosine_beta_schedule(timesteps, s=0.008, beta_end=0.02):
    """[8] で提案されたcosine schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) * beta_end

fig = plt.figure(figsize=(5, 3))
plt.plot(linear_beta_schedule(timesteps), label='linear')
plt.plot(quadratic_beta_schedule(timesteps), label='quadratic')
plt.plot(sigmoid_beta_schedule(timesteps), label='sigmoid')
plt.plot(cosine_beta_schedule(timesteps), label='cosine')
plt.legend()
plt.title('Beta scheduling')


# betaのスケジューリング
betas = linear_beta_schedule(timesteps)

# alpha
alphas_cumprod = torch.cumprod(1. - betas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

def extract(a, t, x_shape: torch.Size):
    """各サンプルのステップtに対応するインデックスの要素を抽出する
    Args:
        a ( T, ): alphasやbetas
        t ( b, ): バッチ内各サンプルのタイムステップt
        x_shape: 画像サイズ
    Returns:
        out ( b, 1, 1, 1 ): xとの計算に使うためxに次元数を合わせて返す
    """
    batch_size = t.shape[0]

    out = a.gather(-1, t.to(a.device))
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    # out = torch.stack([a[_t] for _t in t.to(a.device)]) と同じ

    # xと同じ次元数にする
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise):
    """Reparameterizationを用いて，実画像x_0からノイズ画像x_tをサンプリングする（拡散過程）
    Args:
        x_start ( b, c, h, w ): 実画像x_0
        t ( b, ): バッチ内各サンプルのタイムステップt
        noise ( b, c, h, w ): 標準ガウス分布からのノイズ (epsilon)
    Returns:
        x_noisy: ( b, c, h, w ): ノイズ画像x_t
    """
    alphas_cumprod_t = extract(alphas_cumprod, t, x_start.shape) # ( b, 1, 1, 1 )

    return # WRITE ME

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    """
    Args:
        denoise_model ( nn.Module ): U-Net
        x_start ( b, c, h, w ): 実画像x_0
        t ( b, ): バッチ内各サンプルのタイムステップt
    """
    # [Algorithm 1 4行目] epsilonを標準正規分布からサンプリング
    if noise is None:
        noise = # WRITE ME

    # [Algorithm 1 5行目] x_tをサンプリング -> U-Netでそのノイズを予測
    x_noisy = # WRITE ME

    predicted_noise = denoise_model(x_noisy, t) # U-Net

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
    
