from dataclasses import dataclass
import torch
import torch.nn as nn

# config
@dataclass
class EncoderConfig:
    node_feat_dim: int = 4  # [x, y, profit/p_max, weight/W]
    d_model: int = 128
    n_heads: int = 8        # unused by the MLP; kept so DecoderConfig stays in sync
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )


class GATLayer(nn.Module):

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        # key_padding_mask is accepted for signature parity but unused: nodes do
        # not attend to one another, so padding slots cannot leak into real ones.
        return h + self.ffn(self.ln_ffn(h))


class GATEncoder(nn.Module):

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.node_in = nn.Linear(self.cfg.node_feat_dim, self.cfg.d_model)
        self.layers = nn.ModuleList([GATLayer(self.cfg) for _ in range(self.cfg.n_layers)])
        self.ln_out = nn.LayerNorm(self.cfg.d_model)

    def forward(
        self,
        node_feats: torch.Tensor,                 # (B, N, F)
        node_mask: torch.Tensor | None = None,    # (B, N) True = real node
    ) -> torch.Tensor:
        if node_feats.dim() != 3:
            raise ValueError(f"node_feats must be (B, N, F); got {node_feats.shape}.")
        key_padding_mask = None if node_mask is None else ~node_mask
        h = self.node_in(node_feats)
        for layer in self.layers:
            h = layer(h, key_padding_mask)
        h = self.ln_out(h)
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)
        return h
