#a
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e30 # variable representing negative infinity
TANH_CLIP_C = 10.0

# config
@dataclass
class DecoderConfig:
    d_model: int = 128
    n_heads: int = 8
    # [W_norm, tau_norm, U_frac, R_norm, drone_norm, drone_wait_norm, drone_elapsed_norm, ED_norm, endurance_slack_norm]
    scalar_dim: int = 9
    tanh_clip: float = TANH_CLIP_C

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

# utils
def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=-1)

def _masked_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    contrib = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
    return -contrib.sum(dim=-1)

def _gather_node(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, _, D = h.shape
    return h.gather(1, idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)


class _PointerHead(nn.Module):
    """Multi-head glimpse + tanh-clipped pointer scoring nodes against a context query."""

    def __init__(self, cfg: DecoderConfig, ctx_dim: int):
        super().__init__()
        H = cfg.n_heads
        d = cfg.d_head
        D = cfg.d_model
        self.H = H
        self.d_head = d
        self.d_model = D
        self.tanh_clip = cfg.tanh_clip

        self.ln_q = nn.LayerNorm(ctx_dim)
        self.q_proj = nn.Linear(ctx_dim, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.glimpse_out = nn.Linear(D, D, bias=False)
        self.q_final = nn.Linear(D, D, bias=False)
        self.k_final = nn.Linear(D, D, bias=False)

    def forward(
        self,
        ctx: torch.Tensor,
        h_nodes: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, M, D = h_nodes.shape
        H = self.H
        d = self.d_head

        ctx_n = self.ln_q(ctx)
        q = self.q_proj(ctx_n).view(B, 1, H, d).transpose(1, 2)
        k = self.k_proj(h_nodes).view(B, M, H, d).transpose(1, 2)
        v = self.v_proj(h_nodes).view(B, M, H, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        scores = scores.masked_fill(~mask.view(B, 1, 1, M), NEG_INF)
        attn = F.softmax(scores, dim=-1)
        glimpse = torch.matmul(attn, v).transpose(1, 2).reshape(B, 1, D)
        glimpse = self.glimpse_out(glimpse).squeeze(1)

        q_f = self.q_final(glimpse)
        k_f = self.k_final(h_nodes)
        u = torch.einsum("bd,bnd->bn", q_f, k_f) / (D ** 0.5)
        u = self.tanh_clip * torch.tanh(u)
        return u.masked_fill(~mask, NEG_INF)


# decoder
class StructuredDecoder(nn.Module):
    DUMMY_SLOT_OFFSET = 0

    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        D = self.cfg.d_model

        # Context: [h_graph, h_{c_t}, h_meanU, scalars] -> D
        self.ctx_proj = nn.Sequential(
            nn.Linear(3 * D + self.cfg.scalar_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # drone LAUNCH-target pointer (decided BEFORE the truck move); dummy = "no launch"
        self.head_k = _PointerHead(self.cfg, ctx_dim=D)
        self.dummy_emb = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.dummy_emb, std=0.02)

        # truck next-node pointer, conditioned on the launch outcome (h_k or dummy)
        self.head_j = _PointerHead(self.cfg, ctx_dim=2 * D)

        # binary heads. rejoin & z_drone are conditioned on the in-flight drone target h_D;
        # z_curr on the current node h_c.
        def _binary_head():
            return nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 2))
        self.head_rejoin = _binary_head()   # land the in-flight drone at c_t?
        self.head_z_curr = _binary_head()   # collect the item at c_t (truck pickup)?
        self.head_z_drone = _binary_head()  # collect the item the drone just delivered?

    def _context(
        self,
        h_nodes: torch.Tensor,       
        c_t: torch.Tensor,               
        U_mask: torch.Tensor,              
        scalars: torch.Tensor,            
        node_mask: torch.Tensor | None = None,  
    ) -> torch.Tensor:
        B, N, D = h_nodes.shape

        # mean over real nodes only (padding-safe)
        if node_mask is None:
            h_graph = h_nodes.mean(dim=1)
        else:
            nm_f = node_mask.float().unsqueeze(-1)
            n_real = nm_f.sum(dim=1).clamp(min=1.0)
            h_graph = (h_nodes * nm_f).sum(dim=1) / n_real
        h_c = _gather_node(h_nodes, c_t)                                 
        U_f = U_mask.float()
        n_U = U_f.sum(dim=1, keepdim=True).clamp(min=1.0)              
        h_meanU = (h_nodes * U_f.unsqueeze(-1)).sum(dim=1) / n_U    
        no_unserved = (~U_mask.any(dim=1)).unsqueeze(-1)                 
        h_meanU = torch.where(no_unserved, h_graph, h_meanU)

        return self.ctx_proj(torch.cat([h_graph, h_c, h_meanU, scalars], dim=-1))

    def build_context(
        self,
        h_nodes: torch.Tensor,
        c_t: torch.Tensor,
        U_mask: torch.Tensor,
        scalars: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._context(h_nodes, c_t, U_mask, scalars, node_mask=node_mask)

    def head_k_forward(
        self,
        h_with_dummy: torch.Tensor,
        ctx: torch.Tensor,
        mask_k_ext: torch.Tensor,
    ) -> torch.Tensor:
        return self.head_k(ctx, h_with_dummy, mask_k_ext)

    def head_j_forward(
        self,
        h_nodes: torch.Tensor,
        h_with_dummy: torch.Tensor,
        ctx: torch.Tensor,
        k_slot: torch.Tensor,
        mask_j: torch.Tensor,
    ) -> torch.Tensor:
        # condition the truck move on the launch decision (h of target k, or dummy)
        h_k = _gather_node(h_with_dummy, k_slot)
        q_ctx = torch.cat([ctx, h_k], dim=-1)
        return self.head_j(q_ctx, h_nodes, mask_j)

    def build_h_with_dummy(self, h_nodes: torch.Tensor) -> torch.Tensor:
        B, N, D = h_nodes.shape
        dummy = self.dummy_emb.view(1, 1, D).expand(B, 1, D)
        return torch.cat([h_nodes, dummy], dim=1)

    # binary heads (rejoin / z_curr / z_drone). Each returns masked logits over {0, 1}.
    def head_rejoin_forward(self, ctx, h_D, mask):
        return self.head_rejoin(torch.cat([ctx, h_D], dim=-1)).masked_fill(~mask, NEG_INF)

    def head_z_curr_forward(self, ctx, h_c, mask):
        return self.head_z_curr(torch.cat([ctx, h_c], dim=-1)).masked_fill(~mask, NEG_INF)

    def head_z_drone_forward(self, ctx, h_D, mask):
        return self.head_z_drone(torch.cat([ctx, h_D], dim=-1)).masked_fill(~mask, NEG_INF)
