# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from util.compile_control import maybe_compile
from ffn_factory import build_ffn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _build_distance_matrix(grid_size: tuple[int, int], prefix_tokens: int = 0) -> torch.Tensor:
    grid_h, grid_w = grid_size
    y = torch.arange(grid_h).unsqueeze(1).expand(grid_h, grid_w)
    x = torch.arange(grid_w).unsqueeze(0).expand(grid_h, grid_w)
    coords = torch.stack([y, x], dim=-1).reshape(-1, 2).to(torch.float32)
    distance = torch.cdist(coords, coords, p=2)
    if distance.numel() > 0 and distance.max() > 0:
        distance = distance / distance.max()

    if prefix_tokens > 0:
        total = prefix_tokens + distance.shape[0]
        padded = torch.zeros(total, total, dtype=distance.dtype)
        padded[prefix_tokens:, prefix_tokens:] = distance
        distance = padded

    return distance.unsqueeze(0).unsqueeze(0)


def _init_concat_identity(linear: nn.Linear) -> None:
    if linear.in_features != 2 * linear.out_features:
        raise ValueError("Shortcut projection must be Linear(2d, d).")
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[:, : linear.out_features].copy_(
            torch.eye(linear.out_features, dtype=linear.weight.dtype, device=linear.weight.device)
        )
        if linear.bias is not None:
            linear.bias.zero_()


class BottleneckPatchEmbed(nn.Module):
    """Image to patch embedding with optional timestep-adaptive channel masking and multiscale fusion."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True,
        t_embed_dim: Optional[int] = None,
        adaptive_bottleneck: bool = False,
        bottleneck_mask_temperature: float = 12.0,
        adaptive_bottleneck_init_bias: float = 4.0,
        multiscale_patch_embed: bool = False,
        fine_patch_size: Optional[int] = None,
        multiscale_init_bias: float = -4.0,
        **_: object,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.embed_dim = embed_dim

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

        self.adaptive_bottleneck = adaptive_bottleneck
        self.multiscale_patch_embed = multiscale_patch_embed
        self.bottleneck_mask_temperature = float(bottleneck_mask_temperature)
        self.adaptive_bottleneck_init_bias = float(adaptive_bottleneck_init_bias)
        self.multiscale_init_bias = float(multiscale_init_bias)

        self.channel_mask = None
        if adaptive_bottleneck:
            if t_embed_dim is None:
                raise ValueError("adaptive_bottleneck requires t_embed_dim")
            self.channel_mask = nn.Linear(t_embed_dim, 1, bias=True)
            channel_positions = torch.arange(embed_dim, dtype=torch.float32).div(max(embed_dim, 1)).view(1, 1, embed_dim)
            self.register_buffer("channel_positions", channel_positions, persistent=False)

        self.fine_proj1 = None
        self.fine_proj2 = None
        self.fine_blend = None
        self.fine_merge_factor = 1
        if multiscale_patch_embed:
            if t_embed_dim is None:
                raise ValueError("multiscale_patch_embed requires t_embed_dim")

            fine_patch_size = fine_patch_size or (patch_size[0] // 2)
            if fine_patch_size <= 0 or patch_size[0] % fine_patch_size != 0:
                raise ValueError("fine_patch_size must evenly divide patch_size")

            self.fine_merge_factor = patch_size[0] // fine_patch_size
            self.fine_proj1 = nn.Conv2d(
                in_chans,
                pca_dim,
                kernel_size=(fine_patch_size, fine_patch_size),
                stride=(fine_patch_size, fine_patch_size),
                bias=False,
            )
            self.fine_proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
            self.fine_blend = nn.Linear(t_embed_dim, 1, bias=True)

    def reset_special_parameters(self) -> None:
        if self.channel_mask is not None:
            nn.init.constant_(self.channel_mask.weight, 0)
            nn.init.constant_(self.channel_mask.bias, self.adaptive_bottleneck_init_bias)
        if self.fine_blend is not None:
            nn.init.constant_(self.fine_blend.weight, 0)
            nn.init.constant_(self.fine_blend.bias, self.multiscale_init_bias)

    def _embed_branch(self, x: torch.Tensor, proj1: nn.Conv2d, proj2: nn.Conv2d) -> torch.Tensor:
        return proj2(proj1(x)).flatten(2).transpose(1, 2)

    def _apply_multiscale(self, x: torch.Tensor, tokens: torch.Tensor, t_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.fine_proj1 is None or self.fine_proj2 is None or self.fine_blend is None:
            return tokens
        if t_emb is None:
            raise ValueError("multiscale patch embedding requires timestep embeddings")

        fine = self.fine_proj2(self.fine_proj1(x))
        fine = F.avg_pool2d(fine, kernel_size=self.fine_merge_factor, stride=self.fine_merge_factor)
        fine_tokens = fine.flatten(2).transpose(1, 2)
        alpha = torch.sigmoid(self.fine_blend(t_emb)).unsqueeze(1)
        return (1.0 - alpha) * tokens + alpha * fine_tokens

    def _apply_adaptive_mask(self, tokens: torch.Tensor, t_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.channel_mask is None:
            return tokens
        if t_emb is None:
            raise ValueError("adaptive bottleneck requires timestep embeddings")

        threshold = torch.sigmoid(self.channel_mask(t_emb)).unsqueeze(1)
        mask = torch.sigmoid(((threshold * 1.1) - self.channel_positions) * self.bottleneck_mask_temperature)
        return tokens * mask

    def forward(self, x, t_emb: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        tokens = self._embed_branch(x, self.proj1, self.proj2)
        tokens = self._apply_multiscale(x, tokens, t_emb)
        tokens = self._apply_adaptive_mask(tokens, t_emb)
        return tokens


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        cond_dim: Optional[int] = None,
        adaptive_temperature: bool = False,
        position_bias: bool = False,
        head_specialization: bool = False,
        time_dependent_qkv: bool = False,
        temperature_logit_clamp: float = 2.0,
        time_qkv_init_scale: float = 1.0,
        **_: object,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.adaptive_temperature = bool(adaptive_temperature)
        self.position_bias = bool(position_bias)
        self.head_specialization = bool(head_specialization)
        self.time_dependent_qkv = bool(time_dependent_qkv)
        self.temperature_logit_clamp = float(temperature_logit_clamp)
        self.time_qkv_init_scale = float(time_qkv_init_scale)

        if (
            self.adaptive_temperature
            or self.position_bias
            or self.head_specialization
            or self.time_dependent_qkv
        ) and cond_dim is None:
            raise ValueError("Timestep-aware attention requires cond_dim")

        self.temperature_proj = nn.Linear(cond_dim, 1, bias=True) if self.adaptive_temperature else None
        self.position_proj = nn.Linear(cond_dim, 1, bias=True) if self.position_bias else None
        self.head_proj = nn.Linear(cond_dim, num_heads, bias=True) if self.head_specialization else None
        self.qkv_t = nn.Linear(cond_dim, 3 * dim, bias=False) if self.time_dependent_qkv else None

    def reset_design_parameters(self) -> None:
        for proj in (self.temperature_proj, self.position_proj, self.head_proj):
            if proj is not None:
                nn.init.constant_(proj.weight, 0)
                nn.init.constant_(proj.bias, 0)
        if self.qkv_t is not None:
            nn.init.xavier_uniform_(self.qkv_t.weight)
            self.qkv_t.weight.data.mul_(self.time_qkv_init_scale)

    def forward(
        self,
        x,
        rope,
        t_emb: Optional[torch.Tensor] = None,
        distance_matrix: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape
        qkv = self.qkv(x)
        if self.time_dependent_qkv:
            if t_emb is None:
                raise ValueError("time_dependent_qkv requires timestep embeddings")
            qkv = qkv + self.qkv_t(t_emb).unsqueeze(1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        scale = 1.0 / math.sqrt(q.size(-1))
        attn_score = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

        if self.adaptive_temperature:
            if t_emb is None:
                raise ValueError("adaptive_temperature requires timestep embeddings")
            temperature = torch.exp(
                self.temperature_proj(t_emb).clamp(-self.temperature_logit_clamp, self.temperature_logit_clamp)
            )
            attn_score = attn_score / temperature.view(B, 1, 1, 1)

        if self.position_bias:
            if t_emb is None or distance_matrix is None:
                raise ValueError("position_bias requires timestep embeddings and distance_matrix")
            distance_strength = F.softplus(self.position_proj(t_emb)) - math.log(2.0)
            distance_strength = distance_strength.clamp_min(0.0)
            attn_score = attn_score - distance_strength.view(B, 1, 1, 1) * distance_matrix.to(attn_score.dtype)

        attn_weight = torch.softmax(attn_score, dim=-1).to(v.dtype)
        attn_weight = self.attn_drop(attn_weight)
        x = torch.matmul(attn_weight, v)

        if self.head_specialization:
            if t_emb is None:
                raise ValueError("head_specialization requires timestep embeddings")
            head_scale = 2.0 * torch.sigmoid(self.head_proj(t_emb)).view(B, self.num_heads, 1, 1)
            x = x * head_scale.to(x.dtype)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaLNModulation(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        shared_linear: Optional[nn.Linear] = None,
        lora_rank: Optional[int] = None,
        mode: str = "full",
    ):
        super().__init__()
        if mode not in {"full", "single", "shift_only"}:
            raise ValueError("AdaLN mode must be one of: full, single, shift_only")
        self.hidden_size = hidden_size
        self.shared_linear = shared_linear
        self.lora_rank = lora_rank
        self.mode = mode
        self.output_features = 2 * hidden_size if mode == "shift_only" else 6 * hidden_size
        self.linear = None
        self.lora_down = None
        self.lora_up = None
        self.offset = None

        if shared_linear is None:
            self.linear = nn.Linear(hidden_size, self.output_features, bias=True)
        else:
            if shared_linear.out_features != self.output_features:
                raise ValueError("shared AdaLN projection has incompatible output size")
            if mode == "single":
                self.offset = nn.Parameter(torch.zeros(self.output_features))
            else:
                if lora_rank is None or int(lora_rank) <= 0:
                    raise ValueError("AdaLN-LoRA requires a positive lora_rank")
                self.lora_down = nn.Linear(hidden_size, int(lora_rank), bias=False)
                self.lora_up = nn.Linear(int(lora_rank), self.output_features, bias=False)

    def zero_init(self) -> None:
        if self.linear is not None:
            nn.init.constant_(self.linear.weight, 0)
            nn.init.constant_(self.linear.bias, 0)
        if self.lora_up is not None:
            nn.init.constant_(self.lora_up.weight, 0)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(c)
        if self.linear is not None:
            return self.linear(hidden)
        out = self.shared_linear(hidden)
        if self.offset is not None:
            return out + self.offset.view(1, -1)
        return out + self.lora_up(self.lora_down(hidden))


class FinalLayer(nn.Module):
    """The final layer of JiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    @maybe_compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class JiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_type="swiglu",
        ffn_kwargs=None,
        attention_kwargs=None,
        adaln_shared_linear: Optional[nn.Linear] = None,
        adaln_lora_rank: Optional[int] = None,
        adaln_mode: str = "full",
    ):
        super().__init__()
        self.adaln_mode = adaln_mode
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            cond_dim=hidden_size,
            **(attention_kwargs or {}),
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = build_ffn(ffn_type, hidden_size, mlp_hidden_dim, drop=proj_drop, **(ffn_kwargs or {}))
        self.adaLN_modulation = AdaLNModulation(
            hidden_size,
            shared_linear=adaln_shared_linear,
            lora_rank=adaln_lora_rank,
            mode=adaln_mode,
        )

    def _adaln_terms(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.adaln_mode == "shift_only":
            shift_msa, shift_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
            zeros = torch.zeros_like(shift_msa)
            ones = torch.ones_like(shift_msa)
            return shift_msa, zeros, ones, shift_mlp, zeros, ones
        return self.adaLN_modulation(c).chunk(6, dim=-1)

    @maybe_compile
    def forward(
        self,
        x,
        c,
        t_emb,
        y_emb=None,
        y_labels=None,
        feat_rope=None,
        distance_matrix=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._adaln_terms(c)
        attn_cond = c if getattr(self.attn, "time_dependent_qkv", False) else t_emb
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
            t_emb=attn_cond,
            distance_matrix=distance_matrix,
        )
        mlp_kwargs = {}
        if getattr(self.mlp, "uses_class_condition", False):
            mlp_kwargs["class_cond"] = y_emb
            mlp_kwargs["class_labels"] = y_labels
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp),
            c,
            **mlp_kwargs,
        )
        return x


class JiT(nn.Module):
    """Just image Transformer."""

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        ffn_type="swiglu",
        ffn_kwargs=None,
        attention_kwargs=None,
        topology_kwargs=None,
        patch_kwargs=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.depth = depth

        topology_kwargs = dict(topology_kwargs or {})
        patch_kwargs = dict(patch_kwargs or {})
        attention_kwargs = dict(attention_kwargs or {})
        ffn_kwargs = dict(ffn_kwargs or {})

        self.use_dense_input_shortcuts = bool(topology_kwargs.get("dense_input_shortcuts", False))
        self.use_long_shortcuts = bool(topology_kwargs.get("long_shortcuts", False))
        self.adaln_mode = str(topology_kwargs.get("adaln_mode", "full")).lower()
        if self.adaln_mode == "lora":
            self.adaln_mode = "full"
            topology_kwargs.setdefault("adaln_lora_rank", "auto")
        if bool(topology_kwargs.get("adaln_single", False)):
            self.adaln_mode = "single"
        if self.adaln_mode not in {"full", "single", "shift_only"}:
            raise ValueError("topology_kwargs.adaln_mode must be one of: full, lora, single, shift_only")

        adaln_lora_rank = topology_kwargs.get("adaln_lora_rank")
        if adaln_lora_rank == "auto":
            adaln_lora_rank = max(1, hidden_size // 16)
        self.adaln_lora_rank = None if adaln_lora_rank is None else int(adaln_lora_rank)
        if self.adaln_lora_rank is not None and self.adaln_lora_rank <= 0:
            raise ValueError("adaln_lora_rank must be positive")
        if self.adaln_mode == "single" and self.adaln_lora_rank is not None:
            raise ValueError("AdaLN-Single and AdaLN-LoRA are mutually exclusive")
        if self.adaln_mode == "shift_only" and self.adaln_lora_rank is not None:
            raise ValueError("shift_only AdaLN does not support adaln_lora_rank")

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        patch_kwargs.setdefault("t_embed_dim", hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
            **patch_kwargs,
        )

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        ffn_kwargs.setdefault("num_spatial_tokens", num_patches)
        ffn_kwargs.setdefault("null_class_id", num_classes)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        self.register_buffer(
            "distance_no_context",
            _build_distance_matrix(self.x_embedder.grid_size, prefix_tokens=0),
            persistent=False,
        )
        self.register_buffer(
            "distance_in_context",
            _build_distance_matrix(self.x_embedder.grid_size, prefix_tokens=self.in_context_len),
            persistent=False,
        )

        self.shared_adaln_linear = None
        if self.adaln_lora_rank is not None or self.adaln_mode == "single":
            self.shared_adaln_linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

        # transformer
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    ffn_type=ffn_type,
                    ffn_kwargs=ffn_kwargs,
                    attention_kwargs=attention_kwargs,
                    adaln_shared_linear=self.shared_adaln_linear,
                    adaln_lora_rank=self.adaln_lora_rank,
                    adaln_mode=self.adaln_mode,
                )
                for i in range(depth)
            ]
        )

        self.encoder_skip_depth = (depth + 1) // 2
        self.decoder_skip_start = depth - (depth // 2)

        self.dense_input_projs = None
        if self.use_dense_input_shortcuts:
            self.dense_input_projs = nn.ModuleList(
                [nn.Linear(2 * hidden_size, hidden_size, bias=True) for _ in range(depth)]
            )

        self.long_shortcut_projs = None
        if self.use_long_shortcuts:
            self.long_shortcut_projs = nn.ModuleDict(
                {
                    str(i): nn.Linear(2 * hidden_size, hidden_size, bias=True)
                    for i in range(self.decoder_skip_start, depth)
                }
            )

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for conv in (self.x_embedder.proj1, self.x_embedder.proj2, self.x_embedder.fine_proj1, self.x_embedder.fine_proj2):
            if conv is None:
                continue
            nn.init.xavier_uniform_(conv.weight.view([conv.weight.shape[0], -1]))
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        self.x_embedder.reset_special_parameters()

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.shared_adaln_linear is not None:
            nn.init.constant_(self.shared_adaln_linear.weight, 0)
            nn.init.constant_(self.shared_adaln_linear.bias, 0)

        for block in self.blocks:
            block.adaLN_modulation.zero_init()
            block.attn.reset_design_parameters()
            if hasattr(block.mlp, "reset_design_parameters"):
                block.mlp.reset_design_parameters()

        if self.dense_input_projs is not None:
            for proj in self.dense_input_projs:
                _init_concat_identity(proj)

        if self.long_shortcut_projs is not None:
            for proj in self.long_shortcut_projs.values():
                _init_concat_identity(proj)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _pad_aux_tokens(self, source: torch.Tensor, target_len: int, dtype: torch.dtype) -> torch.Tensor:
        if source.shape[1] == target_len:
            return source.to(dtype=dtype)
        if source.shape[1] < target_len:
            pad_len = target_len - source.shape[1]
            pad = source.new_zeros(source.shape[0], pad_len, source.shape[2], dtype=dtype)
            return torch.cat([pad, source.to(dtype=dtype)], dim=1)
        return source[:, -target_len:].to(dtype=dtype)

    def _merge_shortcut(self, x: torch.Tensor, aux: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        return proj(torch.cat([x, aux], dim=-1))

    def _distance_for_tokens(self, token_count: int) -> torch.Tensor:
        if token_count == self.distance_in_context.shape[-1]:
            return self.distance_in_context
        return self.distance_no_context

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x, t_emb=t_emb)
        x = x + self.pos_embed
        dense_source = x
        skip_features: list[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)

            if self.dense_input_projs is not None:
                dense_tokens = self._pad_aux_tokens(dense_source, x.shape[1], x.dtype)
                x = self._merge_shortcut(x, dense_tokens, self.dense_input_projs[i])

            if self.long_shortcut_projs is not None and i >= self.decoder_skip_start:
                decoder_index = i - self.decoder_skip_start
                mirror = self.encoder_skip_depth - 1 - decoder_index
                if mirror >= 0:
                    skip = self._pad_aux_tokens(skip_features[mirror], x.shape[1], x.dtype)
                    x = self._merge_shortcut(x, skip, self.long_shortcut_projs[str(i)])

            rope = self.feat_rope if x.shape[1] == self.pos_embed.shape[1] else self.feat_rope_incontext
            x = block(
                x,
                c,
                t_emb,
                y_emb=y_emb,
                y_labels=y,
                feat_rope=rope,
                distance_matrix=self._distance_for_tokens(x.shape[1]),
            )
            if i < self.encoder_skip_depth:
                skip_features.append(x)

        x = x[:, self.in_context_len :]
        x = self.final_layer(x, c)
        return self.unpatchify(x, self.patch_size)


def _build_jit(**defaults):
    def factory(**overrides):
        config = dict(defaults)
        config.update(overrides)
        return JiT(**config)

    return factory


JiT_B_16_full = _build_jit(
    depth=12,
    hidden_size=768,
    num_heads=12,
    bottleneck_dim=768,
    in_context_len=32,
    in_context_start=4,
    patch_size=16,
)

JiT_B_16 = _build_jit(
    depth=12,
    hidden_size=768,
    num_heads=12,
    bottleneck_dim=128,
    in_context_len=32,
    in_context_start=4,
    patch_size=16,
)

JiT_B_32 = _build_jit(
    depth=12,
    hidden_size=768,
    num_heads=12,
    bottleneck_dim=128,
    in_context_len=32,
    in_context_start=4,
    patch_size=32,
)

JiT_L_16 = _build_jit(
    depth=24,
    hidden_size=1024,
    num_heads=16,
    bottleneck_dim=128,
    in_context_len=32,
    in_context_start=8,
    patch_size=16,
)

JiT_L_32 = _build_jit(
    depth=24,
    hidden_size=1024,
    num_heads=16,
    bottleneck_dim=128,
    in_context_len=32,
    in_context_start=8,
    patch_size=32,
)

JiT_H_16 = _build_jit(
    depth=32,
    hidden_size=1280,
    num_heads=16,
    bottleneck_dim=256,
    in_context_len=32,
    in_context_start=10,
    patch_size=16,
)

JiT_H_32 = _build_jit(
    depth=32,
    hidden_size=1280,
    num_heads=16,
    bottleneck_dim=256,
    in_context_len=32,
    in_context_start=10,
    patch_size=32,
)


JiT_models = {
    "JiT-B/16-full": JiT_B_16_full,
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
    "JiT-L/16": JiT_L_16,
    "JiT-L/32": JiT_L_32,
    "JiT-H/16": JiT_H_16,
    "JiT-H/32": JiT_H_32,
}
