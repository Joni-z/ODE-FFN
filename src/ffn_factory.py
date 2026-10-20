from torch import nn

from ffn_blocks import FFN_ALIASES, FFN_REGISTRY


def normalize_ffn_type(ffn_type: str) -> str:
    ffn_key = ffn_type.lower().strip()
    return FFN_ALIASES.get(ffn_key, ffn_key)


def build_ffn(
    ffn_type: str,
    in_features: int,
    hidden_features: int,
    drop: float = 0.0,
    bias: bool = True,
    **ffn_kwargs,
) -> nn.Module:
    ffn_key = normalize_ffn_type(ffn_type)
    if ffn_key not in FFN_REGISTRY:
        available = ", ".join(sorted(FFN_REGISTRY))
        raise ValueError(f"Unknown ffn_type: {ffn_type}. Use one of: {available}")
    ffn_cls = FFN_REGISTRY[ffn_key]
    return ffn_cls(
        dim=in_features,
        hidden_dim=hidden_features,
        drop=drop,
        bias=bias,
        **ffn_kwargs,
    )
