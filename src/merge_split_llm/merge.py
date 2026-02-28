import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MergeConfig:
    d_model: int
    d_dir: int          # hidden dim for direction projections
    d_sim: int          # hidden dim for similarity projections
    gamma: float = 1.0  # mergedness sharpness
    eps: float = 1e-6   # numerical stabilizer for std / mass
    delta: float = 1e-6 # numerical stabilizer for apportioned step size
    use_cosine_sim: bool = False  # if True, cosine(Q_s t_i, K_s t_j); else scaled dot
    exclude_self_in_sums: bool = True  # exclude diagonal in sums where applicable


class QKVHadamard(nn.Module):
    """
    Direction-module projections: Q_d, K_d, V_d.

    Implements:
        kappa_ij = V_d( (Q_d t_i) ⊙ (K_d t_j) )
    Vectorized over all pairs (i,j).
    """
    def __init__(self, d_model: int, d_dir: int):
        super().__init__()
        self.Q = nn.Linear(d_model, d_dir, bias=False)
        self.K = nn.Linear(d_model, d_dir, bias=False)
        self.V = nn.Linear(d_dir, d_model, bias=False)

    def forward_pairwise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, N, D)
        Returns:
            kappa: (B, N, N, D) where kappa[:, i, j] = V( Q(t_i) ⊙ K(t_j) )
        """
        q = self.Q(t)  # (B, N, d_dir)
        k = self.K(t)  # (B, N, d_dir)
        # Broadcast to (B, N, N, d_dir)
        q_ = q.unsqueeze(2)
        k_ = k.unsqueeze(1)
        had = q_ * k_
        return self.V(had)  # (B, N, N, D)


class QKSimilarity(nn.Module):
    """
    Similarity projections: Q_s, K_s.

    Implements:
        s_ij = <Q_s t_i, K_s t_j>/sqrt(d_sim)  (or cosine if configured)
    """
    def __init__(self, d_model: int, d_sim: int, use_cosine: bool = False):
        super().__init__()
        self.Q = nn.Linear(d_model, d_sim, bias=False)
        self.K = nn.Linear(d_model, d_sim, bias=False)
        self.d_sim = d_sim
        self.use_cosine = use_cosine

    def forward_pairwise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, N, D)
        Returns:
            s: (B, N, N) similarity matrix
        """
        q = self.Q(t)  # (B, N, d_sim)
        k = self.K(t)  # (B, N, d_sim)

        if self.use_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            # (B, N, N): q @ k^T
            return torch.einsum("bnd,bmd->bnm", q, k)
        else:
            scale = 1.0 / math.sqrt(self.d_sim)
            return scale * torch.einsum("bnd,bmd->bnm", q, k)


class TokenMergeLayer(nn.Module):
    """
    Implements the (anti)symmetric formulation with:
      - antisymmetric direction via kappa_ij - kappa_ji
      - symmetric similarity/gating (if cosine or if you symmetrize; here we explicitly symmetrize)
      - global z-score normalization over off-diagonal entries
      - depth scheduling via (a_l, b_l) producing tilde_s
      - A_ij = sigmoid(tilde_s_ij)
      - m_j = ( sum_k exp(gamma * tilde_s_jk) + eps )^{-1}
      - inertia w_ij = m_j * A_ij
      - apportioned step size alpha_ij = A_ij * w_ij / (sum_k w_ik + delta)
      - update: t_i' = t_i + sum_{j!=i} alpha_ij * D_ij
    """
    def __init__(self, cfg: MergeConfig):
        super().__init__()
        self.cfg = cfg
        self.dir_proj = QKVHadamard(cfg.d_model, cfg.d_dir)
        self.sim_proj = QKSimilarity(cfg.d_model, cfg.d_sim, use_cosine=cfg.use_cosine_sim)

    # ---------- helpers (compute variables internally) ----------

    def _pairwise_kappa(self, t: torch.Tensor) -> torch.Tensor:
        return self.dir_proj.forward_pairwise(t)  # (B, N, N, D)

    def _antisymmetric_kappa(self, t: torch.Tensor) -> torch.Tensor:
        kappa = self._pairwise_kappa(t)  # (B, N, N, D)
        # kappa_bar_ij = kappa_ij - kappa_ji
        return kappa - kappa.transpose(1, 2)

    def _pairwise_direction(self, t: torch.Tensor) -> torch.Tensor:
        """
        D_ij = sigmoid(kappa_bar_ij) ⊙ (t_j - t_i)
        Returns: (B, N, N, D)
        """
        kbar = self._antisymmetric_kappa(t)  # (B, N, N, D)
        gate = torch.sigmoid(kbar)           # (B, N, N, D)
        diff = t.unsqueeze(1) - t.unsqueeze(2)  # (B, N, N, D) gives (t_i - t_j)
        # We want (t_j - t_i):
        diff = -diff
        return gate * diff

    def _pairwise_similarity(self, t: torch.Tensor) -> torch.Tensor:
        return self.sim_proj.forward_pairwise(t)  # (B, N, N)

    def _symmetrize_similarity(self, s: torch.Tensor) -> torch.Tensor:
        # enforce s_ij = s_ji (symmetry)
        return 0.5 * (s + s.transpose(1, 2))

    def _global_zscore(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Global normalization over all off-diagonal entries per batch element.

        Returns:
            s_hat: (B, N, N)
            mu:    (B, 1, 1)
            var:   (B, 1, 1)
        """
        B, N, _ = s.shape
        if self.cfg.exclude_self_in_sums:
            mask = ~torch.eye(N, dtype=torch.bool, device=s.device)  # (N, N)
            vals = s[:, mask]  # (B, N*(N-1))
        else:
            vals = s.reshape(B, -1)

        mu = vals.mean(dim=1, keepdim=True)  # (B, 1)
        var = vals.var(dim=1, keepdim=True, unbiased=False)  # (B, 1)
        mu_ = mu.view(B, 1, 1)
        var_ = var.view(B, 1, 1)
        s_hat = (s - mu_) / torch.sqrt(var_ + self.cfg.eps)
        return s_hat, mu_, var_

    def compute_tilde_s(self, t: torch.Tensor, a_l: torch.Tensor, b_l: torch.Tensor) -> torch.Tensor:
        """
        tilde_s_ij = a_l * zscore(s_ij) + b_l

        Args:
            t:   (B, N, D)
            a_l: scalar or (B,) or (B,1,1)
            b_l: scalar or (B,) or (B,1,1)
        Returns:
            tilde_s: (B, N, N)
        """
        s = self._pairwise_similarity(t)
        s = self._symmetrize_similarity(s)

        s_hat, _, _ = self._global_zscore(s)

        # Broadcast a_l, b_l to (B,1,1)
        if not torch.is_tensor(a_l):
            a_l = torch.tensor(a_l, device=t.device, dtype=t.dtype)
        if not torch.is_tensor(b_l):
            b_l = torch.tensor(b_l, device=t.device, dtype=t.dtype)

        if a_l.ndim == 0:
            a_l = a_l.view(1, 1, 1)
        elif a_l.ndim == 1:
            a_l = a_l.view(-1, 1, 1)
        if b_l.ndim == 0:
            b_l = b_l.view(1, 1, 1)
        elif b_l.ndim == 1:
            b_l = b_l.view(-1, 1, 1)

        return a_l * s_hat + b_l

    def _step_gate_A(self, tilde_s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(tilde_s)  # (B, N, N)

    def _mass_m(self, tilde_s: torch.Tensor) -> torch.Tensor:
        """
        m_j = ( sum_k exp(gamma * tilde_s_jk) + eps )^{-1}
        Returns: m of shape (B, N)
        """
        # sum over k (last dim): (B, N)
        rho = torch.exp(self.cfg.gamma * tilde_s).sum(dim=2)
        return 1.0 / (rho + self.cfg.eps)

    def _inertia_w(self, A: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        w_ij = m_j * A_ij
        m: (B, N) -> broadcast to (B, 1, N)
        """
        return A * m.unsqueeze(1)

    def _apportioned_step_size(self, A: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        alpha_ij = A_ij * w_ij / (sum_k w_ik + delta)
        Here sum_k is over j dimension (last dim), per i (row).
        Returns: (B, N, N)
        """
        denom = w.sum(dim=2, keepdim=True) + self.cfg.delta  # (B, N, 1)
        return A * (w / denom)

    def _mask_diagonal_pairs(self, x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        """
        Zeroes the diagonal i=j for pairwise tensors x:
          - if x is (B,N,N): sets x[:,i,i]=value
          - if x is (B,N,N,D): sets x[:,i,i,:]=value
        """
        B = x.shape[0]
        N = x.shape[1]
        eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)
        if x.ndim == 3:
            x = x.masked_fill(eye, value)
        elif x.ndim == 4:
            x = x.masked_fill(eye.unsqueeze(-1), value)
        else:
            raise ValueError(f"Unsupported shape for diagonal masking: {x.shape}")
        return x

    # ---------- main forward ----------

    def forward(
        self,
        t: torch.Tensor,
        a_l: torch.Tensor,
        b_l: torch.Tensor,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            t:   (B, N, D)
            a_l: depth schedule scale
            b_l: depth schedule offset
            return_aux: if True, return a dict of key intermediates
        Returns:
            t_next: (B, N, D)
            aux: optional dict
        """
        # tilde_s is the only intermediate explicitly passed between equations.
        tilde_s = self.compute_tilde_s(t, a_l=a_l, b_l=b_l)

        # gates / masses
        A = self._step_gate_A(tilde_s)     # (B, N, N)
        m = self._mass_m(tilde_s)          # (B, N)
        w = self._inertia_w(A, m)          # (B, N, N)
        alpha = self._apportioned_step_size(A, w)  # (B, N, N)

        # direction and update
        D = self._pairwise_direction(t)    # (B, N, N, D)

        if self.cfg.exclude_self_in_sums:
            alpha = self._mask_diagonal_pairs(alpha, 0.0)
            D = self._mask_diagonal_pairs(D, 0.0)

        # t_i' = t_i + sum_j alpha_ij * D_ij
        t_next = t - torch.einsum("bnm,bnmd->bnd", alpha, D)

        aux = None
        if return_aux:
            aux = {
                "tilde_s": tilde_s,
                "A": A,
                "m": m,
                "w_inertia": w,
                "alpha_apportioned": alpha,
            }
        return t_next, aux


# ------------------ minimal usage example ------------------
if __name__ == "__main__":
    cfg = MergeConfig(d_model=256, d_dir=128, d_sim=128, gamma=1.0, use_cosine_sim=False)
    layer = TokenMergeLayer(cfg)

    B, N, D = 2, 64, cfg.d_model
    t = torch.randn(B, N, D)

    # Example depth schedule parameters for one layer:
    a_l = torch.tensor(1.0)   # scale
    b_l = torch.tensor(-0.5)  # offset

    t_next, aux = layer(t, a_l=a_l, b_l=b_l, return_aux=True)
    print(t_next.shape, aux["A"].shape, aux["m"].shape)
