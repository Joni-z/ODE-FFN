
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    A simple SwiGLU FFN block that maps dim -> dim.
    """
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


def patchify(x: torch.Tensor, patch_size: int):
    """
    x: [B, 3, H, W] -> patches: [B, N, P*P*3]
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    p = patch_size
    h = H // p
    w = W // p
    # [B, C, h, p, w, p] -> [B, h, w, p, p, C] -> [B, N, p*p*C]
    patches = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).contiguous()
    patches = patches.view(B, h * w, p * p * C)
    return patches, (h, w)


def unpatchify(patches: torch.Tensor, hw: tuple[int, int], patch_size: int, channels: int = 3):
    """
    patches: [B, N, P*P*3] -> x: [B, 3, H, W]
    """
    B, N, D = patches.shape
    h, w = hw
    p = patch_size
    assert N == h * w
    assert D == p * p * channels

    patches = patches.view(B, h, w, p, p, channels).permute(0, 5, 1, 3, 2, 4).contiguous()
    x = patches.view(B, channels, h * p, w * p)
    return x


class PatchAE(nn.Module):
    """
    Patch autoencoder:
      - patchify image into [B, N, patch_dim]
      - encode to [B, N, latent_dim]
      - decode to [B, N, patch_dim]
      - unpatchify back to image

    Two designs:
      - linear: Linear(patch_dim -> latent_dim) and Linear(latent_dim -> patch_dim)
      - ffn: small MLP w/ SwiGLU blocks (residual), still per-patch (no attention)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        latent_dim: int = 128,
        arch: str = "linear",         # ["linear", "ffn"]
        ffn_ratio: int = 4,
        drop: float = 0.0,
        bias: bool = True,
        normalize_patches: bool = False,  # optional: per-patch mean/std normalization
        clamp_output: bool = True,        # clamp recon to [0,1]
    ):
        super().__init__()
        # assert arch in ["linear", "ffn"]
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.latent_dim = latent_dim
        self.arch = arch
        self.normalize_patches = normalize_patches
        self.clamp_output = clamp_output

        self.patch_dim = patch_size * patch_size * in_chans

        if arch == "linear":
            self.enc = nn.Linear(self.patch_dim, latent_dim, bias=bias)
            self.dec = nn.Linear(latent_dim, self.patch_dim, bias=bias)
        elif arch == "linear_ft":
            self.enc = nn.Linear(self.patch_dim, latent_dim, bias=False)
            self.dec = nn.Linear(latent_dim, self.patch_dim, bias=True)
        elif arch == "linear_ff":
            self.enc = nn.Linear(self.patch_dim, latent_dim, bias=False)
            self.dec = nn.Linear(latent_dim, self.patch_dim, bias=False)
        else:
            # encode: patch_dim -> latent_dim using a small FFN stack
            self.enc_in = nn.Linear(self.patch_dim, latent_dim, bias=bias)
            self.enc_ffn = SwiGLUFFN(latent_dim, latent_dim*ffn_ratio, drop=drop, bias=bias)

            self.dec_in = nn.Linear(latent_dim, latent_dim, bias=bias)
            self.dec_ffn = SwiGLUFFN(latent_dim, latent_dim*ffn_ratio, drop=drop, bias=bias)
            self.dec_out = nn.Linear(latent_dim, self.patch_dim, bias=bias)

        # init: keep it simple and stable
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, patches: torch.Tensor):
        if "linear" in self.arch:
            return self.enc(patches)
        z = self.enc_in(patches)
        z = z + self.enc_ffn(z)
        return z

    def decode(self, z: torch.Tensor):
        if "linear" in self.arch:
            return self.dec(z)
        h = self.dec_in(z)
        h = h + self.dec_ffn(h)
        patches = self.dec_out(h)
        return patches

    def forward(self, x: torch.Tensor):
        """
        x in [0,1] float tensor: [B, 3, H, W]
        returns:
          recon_x: [B, 3, H, W]
          z: [B, N, latent_dim]
        """
        x = 2 * x - 1.0  # to [-1,1]
        patches, hw = patchify(x, self.patch_size)

        if self.normalize_patches:
            mean = patches.mean(dim=-1, keepdim=True)
            std = patches.std(dim=-1, keepdim=True).clamp_min(1e-6)
            patches_n = (patches - mean) / std
        else:
            patches_n = patches
            mean, std = None, None

        z = self.encode(patches_n)
        recon_patches = self.decode(z)

        if self.normalize_patches:
            recon_patches = recon_patches * std + mean

        recon_x = unpatchify(recon_patches, hw, self.patch_size, channels=self.in_chans)
        recon_x = (recon_x + 1.0) / 2.0  # to [0,1]

        if self.clamp_output:
            recon_x = recon_x.clamp(0.0, 1.0)

        return recon_x, z
