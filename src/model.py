import torch
from torch import nn
import torch.nn.functional as F

from .cross_transformer import Trans_C
from .patch_layers import DepatchSampling


class Filter(nn.Module):
    def __init__(self, channel=1, kernel_size=12):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channel,
            channel,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
            padding_mode="replicate",
            bias=True,
            groups=channel,
        )
        self.conv.weight = nn.Parameter(1 / kernel_size * torch.ones([channel, 1, kernel_size]))

    def forward(self, inp):
        return self.conv(inp.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        context_window: int,
        target_window: int,
        ablation: int,
        mlp_drop: float,
        mlp_hidden: int,
        cf_dim: int,
        cf_depth: int,
        cf_heads: int,
        cf_mlp: int,
        cf_head_dim: int,
        cf_drop: float,
        c_in: int,
        patch_len: int,
        stride: int,
        d_model: int,
        head_dropout=0,
        padding_patch=None,
        individual=False,
        target_index=-1,
    ):
        super().__init__()
        if not -c_in <= target_index < c_in:
            raise ValueError("target_index is outside input channels")
        self.target_index = target_index % c_in
        self.deformable = True
        self.ablation = ablation
        self.beta = 0.3
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow = target_window
        patch_num = int((context_window - patch_len) / stride + 1)
        self.Filter = Filter(c_in, kernel_size=25)
        seq_len = (patch_num - 1) * self.stride + self.patch_len
        self.deformable_sampling = DepatchSampling(c_in, seq_len, self.patch_len, self.stride)
        self.fre_transformer = Trans_C(
            dim=cf_dim,
            depth=cf_depth,
            heads=cf_heads,
            mlp_dim=cf_mlp,
            dim_head=cf_head_dim,
            dropout=cf_drop,
            patch_dim=patch_len * 2,
            horizon=self.targetwindow * 2,
            d_model=d_model * 2,
        )
        self.head_nf_f = d_model * 2 * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.Linear1 = nn.Linear(c_in, d_model * 2)
        self.adaptive = nn.Parameter(torch.ones(2))
        self.fc = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.inn = INN_all(num_layers=1)
        self.CIAA = CIAA(d_model, n_heads=8, use_cls=True, aligned_values=False)
        self.linear = nn.Linear(3072, 24)

    def fftpatch(self, tensor, patch_len, stride):
        fft_result = torch.fft.fft(tensor)
        real_part = fft_result.real
        imag_part = fft_result.imag
        if not self.deformable:
            real_unfolded = real_part.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            imag_unfolded = imag_part.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            real_unfolded = self.deformable_sampling(real_part)
            imag_unfolded = self.deformable_sampling(imag_part)
        return (real_unfolded, imag_unfolded)

    def encode_ccia(self, z):
        target_history = z[:, :, self.target_index]
        x = z - self.beta * self.Filter(z)
        x1, x2 = self.fftpatch(x.permute(0, 2, 1), self.patch_len, self.stride)
        z_perm = z.permute(0, 2, 1)
        z1, z2 = self.fftpatch(z_perm, self.patch_len, self.stride)
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        x = torch.cat((x1, x2), 1)
        x = self.inn(x)
        x = self.Linear1(x)
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z = torch.reshape(z, (batch_size, patch_num, c_in, z.shape[-1]))
        z = z.permute(0, 2, 1, 3)
        z = x * self.adaptive[0] + z * self.adaptive[1]
        return x, z, target_history

    def decode_ccia(self, spectrum):
        x = torch.fft.ifft(spectrum)
        x = self.fc(torch.cat((x.real, x.imag), -1))
        return x.unsqueeze(1)

    def forward(self, z):
        enhanced, fused, _ = self.encode_ccia(z)
        spectrum = self.CIAA(enhanced[:, 1:, :, :], fused)
        return self.decode_ccia(spectrum)


class INN(nn.Module):
    def __init__(self, input, output, ratio):
        super().__init__()
        hidden_dim = int(input * ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(input, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, output, 1, bias=False),
            nn.BatchNorm2d(output),
        )

    def forward(self, x):
        out = self.bottleneckBlock(x)
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out


class Feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = INN(input=32, output=32, ratio=2)
        self.seta = INN(input=32, output=32, ratio=2)

    def forward(self, f1, f2):
        f2 = f2 + self.phi(f1)
        f1 = f1 + self.seta(f2)
        return (f1, f2)


class INN_all(nn.Module):
    def __init__(self, num_layers=None):
        super().__init__()
        self.net = nn.Sequential(*[Feature() for _ in range(num_layers)])
        self.shffle = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv2d(64, 12, kernel_size=3, stride=1, padding=1)
        self.Merge = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)

    def separate(self, x):
        return (x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]])

    def forward(self, x):
        f1, f2 = self.separate(self.shffle(x))
        for layer in self.net:
            f1, f2 = layer(f1, f2)
        f_out = self.fusion(torch.cat((f1, f2), dim=1))
        return self.Merge(f_out)


class CIAA(nn.Module):
    def __init__(self, d_model, n_heads, use_cls=True, aligned_values=False):
        super().__init__()
        self.aligned_values = aligned_values
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_cls = use_cls
        self.q_proj = nn.Linear(512, d_model)
        self.k_proj = nn.Linear(512, d_model)
        self.v_proj = nn.Linear(512, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)
        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 512))
            self.linear = nn.Linear(d_model, 24)

    def forward(self, x, z, target_history=None, observed_history=None):
        B = x.shape[0]
        flatten = nn.Flatten(start_dim=-2)
        runoff = flatten(x)
        exogenous = flatten(z)
        if self.use_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1, -1).squeeze(2)
            runoff = torch.cat([cls_tokens, runoff], dim=1)
        q = self.q_proj(runoff)
        k = self.k_proj(exogenous)
        v = self.v_proj(exogenous if self.aligned_values else runoff)
        q = q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1) * (1.0 / self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        heads = attn @ v
        x = heads.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.head_dim)
        x = self.out_proj(x)
        x = self.norm(x)
        if self.use_cls:
            x = self.linear(x[:, 0])
        else:
            x = x.mean(dim=1)
        return x
