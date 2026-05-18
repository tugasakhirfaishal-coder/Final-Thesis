import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
    DGL_AVAILABLE = True
except Exception:
    DGL_AVAILABLE = False

from graph_construction import build_ncc_adjacency, build_dgl_graphs


# ======================================================================
# 1. TEMPORAL POSITIONAL ENCODING
# ======================================================================

class TemporalPE(nn.Module):
    """
    Sinusoidal Positional Encoding untuk posisi jendela W (0..14).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ======================================================================
# 2. INPUT EMBEDDING
# ======================================================================

class InputEmbedding(nn.Module):
    """
    Proyeksi fitur DWT statistik ke hidden_dim.

    Input  : (B, C, W, F)  — B=batch, C=18 channel, W=15 jendela, F=21 fitur
    Output : (B, C, hidden_dim)

    Alur:
      1. Reshape → (B*C, W, F)
      2. LayerNorm(F) — normalisasi 21 fitur ke skala seragam
      3. Linear(F → D) per jendela + GELU + Dropout
      4. TemporalPE(W)
      5. Pool across W → (B*C, D)  [mean atau max]
      6. LayerNorm → reshape → (B, C, D)
    """

    def __init__(
        self,
        n_windows   : int,
        n_features  : int,
        hidden_dim  : int,
        dropout     : float = 0.1,
        pool_mode   : str   = 'mean',
    ):
        super().__init__()
        self.n_windows  = n_windows
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.pool_mode  = pool_mode

        # Normalisasi fitur input sebelum proyeksi.
        # Fitur DWT memiliki skala sangat berbeda (Energy 1e6 >> RWE 0-1).
        # Tanpa ini, Linear(21→128) didominasi Energy dan tidak belajar
        # dari fitur lain. Kritis untuk generalisasi lintas pasien (LOPO).
        self.input_norm  = nn.LayerNorm(n_features)
        self.feat_proj   = nn.Linear(n_features, hidden_dim)
        self.temporal_pe = TemporalPE(d_model=hidden_dim, max_len=n_windows + 4)
        self.dropout     = nn.Dropout(dropout)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, W, NF = x.shape
        x_flat = x.reshape(B * C, W, NF)
        x_flat = self.input_norm(x_flat)   # normalisasi 21 fitur ke unit variance
        h = self.feat_proj(x_flat)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.temporal_pe(h)
        if self.pool_mode == 'mean':
            h = h.mean(dim=1)
        else:
            h = h.max(dim=1).values
        h = self.norm(h)
        return h.reshape(B, C, self.hidden_dim)


# ======================================================================
# 3. SPATIAL PE — RWPE
# ======================================================================

class RWPEEncoder(nn.Module):
    """
    Random Walk PE: PE[i,k] = (P^k)[i,i]
    P = row-normalized adjacency (transition matrix)
    """

    def __init__(self, k_steps: int, hidden_dim: int):
        super().__init__()
        self.k_steps = k_steps
        self.linear  = nn.Linear(k_steps, hidden_dim)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        B, C, _ = adj.shape
        P       = adj.clone()
        landing = []
        Pk      = P.clone()
        for _ in range(self.k_steps):
            landing.append(torch.diagonal(Pk, dim1=-2, dim2=-1))
            Pk = torch.bmm(Pk, P)
        pe_raw = torch.stack(landing, dim=-1)        # (B, C, k)
        return self.norm(self.linear(pe_raw))


# ======================================================================
# 4. SPATIAL PE — LPE
# ======================================================================

class LPEEncoder(nn.Module):
    """
    Laplacian PE: eigenvectors dari L = D - A
    """

    def __init__(self, lpe_dim: int, hidden_dim: int):
        super().__init__()
        self.lpe_dim = lpe_dim
        self.linear  = nn.Linear(lpe_dim, hidden_dim)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        B, C, _ = adj.shape
        pe_list  = []
        for b in range(B):
            deg = adj[b].sum(dim=-1)
            L   = (torch.diag(deg) - adj[b]).float()
            try:
                _, eigvecs = torch.linalg.eigh(L)
                k   = min(self.lpe_dim, C - 1)
                lpe = eigvecs[:, 1:k + 1]
                if k < self.lpe_dim:
                    lpe = torch.cat(
                        [lpe, torch.zeros(C, self.lpe_dim - k, device=adj.device)], dim=-1
                    )
                pe_list.append(lpe)
            except Exception:
                pe_list.append(torch.zeros(C, self.lpe_dim, device=adj.device))
        pe_raw = torch.stack(pe_list, dim=0)
        return self.norm(self.linear(pe_raw))


# ======================================================================
# 5. GATED MULTI-HEAD ATTENTION (DGL)
# ======================================================================

def _src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: edges.src[src_field] * edges.dst[dst_field]}
    return func

def _scaling(field, scale):
    def func(edges):
        return {field: edges.data[field] / scale}
    return func

def _imp_exp_attn(implicit, explicit):
    def func(edges):
        return {implicit: edges.data[implicit] * edges.data[explicit]}
    return func

def _exp_clamp(field, out_field='score_exp'):
    def func(edges):
        return {out_field: torch.exp(
            edges.data[field].sum(-1, keepdim=True).clamp(-5, 5)
        )}
    return func

# [FIX 3] Fungsi ini menerima e_gate yang SUDAH sigmoid dari edata,
# sehingga tidak perlu sigmoid lagi di sini — cukup multiply langsung.
def _apply_edge_gate(attn, gate):
    def func(edges):
        return {attn: edges.data[attn] * edges.data[gate]}
    return func


class MultiHeadAttnLayer(nn.Module):
    """
    Gated Multi-Head Attention dengan explicit edge features.

    Alur sesuai paper (pers. 3.5 – 3.9):
      score = (K · Q) / sqrt(d)          [pers. 3.5, bagian 1]
      score = score × proj_e(e)           [pers. 3.5, bagian 2]
      e_out = score                       [pers. 3.6 — diambil SEBELUM gate]
      score = score × sigmoid(G_e)        [pers. 3.8 — edge gate]
      w     = softmax(score) × sigmoid(G_h)  [pers. 3.7, 3.9]
      h_out = sum_j(w_ij * V_j)          [pers. 3.10]
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_heads: int, use_bias: bool = False):
        super().__init__()
        self.out_dim   = out_dim
        self.num_heads = num_heads
        self.Q      = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K      = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V      = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.e_gate = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.n_gate = nn.Linear(in_dim, out_dim * num_heads, bias=True)

    def forward(self, g, h: torch.Tensor, e: torch.Tensor):
        H, D = self.num_heads, self.out_dim

        g.ndata['Q_h']    = self.Q(h).view(-1, H, D)
        g.ndata['K_h']    = self.K(h).view(-1, H, D)
        g.ndata['V_h']    = self.V(h).view(-1, H, D)
        g.ndata['n_gate'] = torch.sigmoid(self.n_gate(h).view(-1, H, D))
        g.edata['proj_e'] = self.proj_e(e).view(-1, H, D)

        # [FIX 1] sigmoid diterapkan di sini saat assign,
        # konsisten dengan kode referensi (Kode 2) dan paper pers. 3.8
        g.edata['e_gate'] = torch.sigmoid(self.e_gate(e).view(-1, H, D))

        # Langkah 1: score = (K · Q) / sqrt(d)
        g.apply_edges(_src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(_scaling('score', math.sqrt(D)))

        # Langkah 2: score = score × proj_e  →  ini adalah ŵ (pers. 3.5)
        g.apply_edges(_imp_exp_attn('score', 'proj_e'))

        # [FIX 2] e_out diambil SEKARANG, sebelum edge gate sigmoid
        # sesuai paper pers. 3.6: ê = e + O_e · f(ŵ)
        g.edata['e_out'] = g.edata['score']

        # Langkah 3: score = score × sigmoid(G_e)  →  edge gate (pers. 3.8)
        # [FIX 3] _apply_edge_gate tidak sigmoid lagi karena e_gate sudah sigmoid
        g.apply_edges(_apply_edge_gate('score', 'e_gate'))

        # Langkah 4: exp+clamp ≈ softmax numerik stabil
        g.apply_edges(_exp_clamp('score', 'score_exp'))

        # Langkah 5: agregasi h dengan node gate (pers. 3.9, 3.10)
        g.update_all(fn.u_mul_e('V_h', 'score_exp', 'msg'), fn.sum('msg', 'wV'))
        g.update_all(fn.copy_e('score_exp', 'score_copy'), fn.sum('score_copy', 'z'))

        wV    = g.ndata['wV']
        z     = g.ndata['z'].clamp(min=1e-6)
        h_out = (wV / z) * g.ndata['n_gate']
        e_out = g.edata['e_out']
        return h_out, e_out


# ======================================================================
# 6. GATED GRAPH TRANSFORMER LAYER (DGL)
# ======================================================================

class GTLayer(nn.Module):
    """
    Satu layer GT sesuai paper persamaan A1–A9.

    Alur per layer:
      [A1]  h̃, ẽ = LN(h), LN(e)          ← pre-norm sebelum attention
      [A2]  ŵ   = (Q·K/√d) · E           ← attention score
      [A3]  ê   = e + O_e(concat(ŵ))     ← update edge (e_out dari ŵ)
      [A4]  g_h  = sigmoid(G_h · h̃)      ← node gate
      [A5]  g_e  = sigmoid(G_e · ẽ)      ← edge gate
      [A6]  w    = softmax_j(ŵ × g_e)    ← attention weight
      [A7]  ĥ   = h + O_h(concat(g_h × w × V·h̃))
      [A8]  h'  = ĥ + FFN(LN(ĥ))        ← pre-norm sebelum FFN
      [A9]  e'  = ê + FFN(LN(ê))        ← pre-norm sebelum FFN
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int,
                 dropout: float = 0.1, layer_norm: bool = False,
                 batch_norm: bool = True, residual: bool = True,
                 use_bias: bool = False):
        super().__init__()
        assert out_dim % num_heads == 0, \
            f"out_dim ({out_dim}) harus habis dibagi num_heads ({num_heads})"
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.num_heads  = num_heads
        self.dropout    = dropout
        self.residual   = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttnLayer(
            in_dim, out_dim // num_heads, num_heads, use_bias
        )
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        # [FIX A1] Pre-norm: LN diterapkan pada h dan e SEBELUM masuk attention
        # Paper A1: h̃_i = LN(h_i), {h̃_j} = LN({h_j}), {ẽ_ij} = LN({e_ij})
        # Selalu pakai LayerNorm untuk pre-norm (batch_norm tidak cocok untuk pre-norm)
        self.pre_ln_h = nn.LayerNorm(in_dim)
        self.pre_ln_e = nn.LayerNorm(in_dim)

        # Post-norm setelah attention+residual (untuk batch_norm / layer_norm pilihan user)
        if layer_norm:
            self.ln1_h = nn.LayerNorm(out_dim)
            self.ln1_e = nn.LayerNorm(out_dim)
        if batch_norm:
            self.bn1_h = nn.BatchNorm1d(out_dim)
            self.bn1_e = nn.BatchNorm1d(out_dim)

        # FFN dengan GeLU sesuai diagram gambar 1
        # [FIX A8/A9] LN masuk ke DALAM FFN sebagai pre-norm (bukan setelah FFN)
        # Paper A8: h' = ĥ + FFN(LN(ĥ))  — LN adalah input ke FFN
        # Paper A9: e' = ê + FFN(LN(ê))
        self.ffn_ln_h = nn.LayerNorm(out_dim)   # LN sebelum FFN (pre-norm)
        self.ffn_ln_e = nn.LayerNorm(out_dim)
        self.ffn_h = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim * 2, out_dim),
        )
        self.ffn_e = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim * 2, out_dim),
        )

    def _norm_post_attn(self, h, e):
        """Post-norm setelah attention+residual (ĥ → norm → ĥ_normed)."""
        if self.layer_norm:
            return self.ln1_h(h), self.ln1_e(e)
        if self.batch_norm:
            return self.bn1_h(h), self.bn1_e(e)
        return h, e

    def forward(self, g, h: torch.Tensor, e: torch.Tensor):
        # [A1] Pre-norm: LN pada h dan e sebelum masuk attention
        h_ln = self.pre_ln_h(h)
        e_ln = self.pre_ln_e(e)

        # [A2–A6] Gated multi-head attention (input sudah ternormalisasi)
        h_att, e_att = self.attention(g, h_ln, e_ln)

        # [A7] ĥ = h + O_h(concat heads)  — residual dari h ASLI (bukan h_ln)
        h_hat = F.dropout(self.O_h(h_att.reshape(-1, self.out_dim)),
                          self.dropout, self.training)
        e_hat = F.dropout(self.O_e(e_att.reshape(-1, self.out_dim)),
                          self.dropout, self.training)
        if self.residual:
            h_hat = h + h_hat    # residual dari h asli (pers. A7)
            e_hat = e + e_hat    # residual dari e asli (pers. A3)

        # Post-norm ĥ (opsional, sesuai config layer_norm/batch_norm)
        h_hat, e_hat = self._norm_post_attn(h_hat, e_hat)

        # [A8] h' = ĥ + FFN(LN(ĥ))  — LN diterapkan SEBELUM FFN (pre-norm)
        h_out = h_hat + F.dropout(
            self.ffn_h(self.ffn_ln_h(h_hat)), self.dropout, self.training
        )
        # [A9] e' = ê + FFN(LN(ê))
        e_out = e_hat + F.dropout(
            self.ffn_e(self.ffn_ln_e(e_hat)), self.dropout, self.training
        )

        return h_out, e_out


# ======================================================================
# 7. GT LAYER FALLBACK (tanpa DGL)
# ======================================================================

class GTLayerNoDGL(nn.Module):
    """Standard multi-head self-attention sebagai fallback DGL."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int,
                 dropout: float = 0.1, residual: bool = True, **kwargs):
        super().__init__()
        assert out_dim % num_heads == 0
        self.out_dim  = out_dim
        self.n_heads  = num_heads
        self.d_head   = out_dim // num_heads
        self.dropout  = dropout
        self.residual = residual
        self.W_Q  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_K  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_V  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_O  = nn.Linear(out_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor = None):
        N, D = h.shape
        H, d = self.n_heads, self.d_head
        Q = self.W_Q(h).view(N, H, d)
        K = self.W_K(h).view(N, H, d)
        V = self.W_V(h).view(N, H, d)
        scores = torch.einsum('ihd,jhd->hij', Q, K) / math.sqrt(d)
        if adj is not None:
            mask   = (adj > 0).unsqueeze(0).expand(H, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = F.dropout(attn, self.dropout, training=self.training)
        out = torch.einsum('hij,jhd->ihd', attn, V).reshape(N, D)
        out = self.W_O(out)
        h = self.norm1(
            h + F.dropout(out, self.dropout, training=self.training)
        ) if self.residual else self.norm1(out)
        h = self.norm2(h + F.dropout(self.ffn(h), self.dropout,
                                     training=self.training))
        return h


# ======================================================================
# 8. CLASSIFIER HEAD
# ======================================================================

class ClassifierHead(nn.Module):
    """FC Head: D → D → GELU → Dropout → LN → n_classes"""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert n_layers >= 1
        layers = []
        in_d   = input_dim
        for i in range(n_layers - 1):
            out_d = max(hidden_dim // (2 ** i), n_classes * 4)
            layers += [
                nn.Linear(in_d, out_d), nn.GELU(),
                nn.Dropout(dropout), nn.LayerNorm(out_d),
            ]
            in_d = out_d
        layers.append(nn.Linear(in_d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# 9. MODEL UTAMA: GT-STAFG
# ======================================================================

class GTSTAFG(nn.Module):
    """
    Graph Transformer — Spatial-Temporal Adaptive Feature Graph.

    Pipeline:
      (B,C,W,F) → InputEmbedding → (B,C,D)
                → NCC adj (B,C,C)
                → RWPE/LPE → h + spe_proj(spe)
                → GTLayer × n → mean pool → ClassifierHead
                → (B,2) logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D        = cfg.HIDDEN_DIM

        self.embedding = InputEmbedding(
            n_windows  = cfg.N_WINDOWS,
            n_features = cfg.N_FEATURES,
            hidden_dim = D,
            dropout    = 0.1,
            pool_mode  = getattr(cfg, 'POOL_MODE', 'mean'),
        )

        if cfg.SPE_TYPE == 'RWPE':
            self.spe = RWPEEncoder(cfg.RWPE_STEPS, D)
        else:
            self.spe = LPEEncoder(cfg.LPE_DIM, D)
        self.spe_proj = nn.Linear(D, D)

        if DGL_AVAILABLE:
            self.gt_layers = nn.ModuleList([
                GTLayer(
                    in_dim=D, out_dim=D, num_heads=cfg.GT_HEADS,
                    dropout=cfg.GT_DROPOUT, layer_norm=cfg.LAYER_NORM,
                    batch_norm=cfg.BATCH_NORM, residual=True,
                    use_bias=cfg.USE_BIAS,
                )
                for _ in range(cfg.GT_LAYERS)
            ])
        else:
            print("DGL tidak tersedia → GTLayerNoDGL (fallback)")
            self.gt_layers = nn.ModuleList([
                GTLayerNoDGL(
                    in_dim=D, out_dim=D, num_heads=cfg.GT_HEADS,
                    dropout=cfg.GT_DROPOUT, residual=True,
                )
                for _ in range(cfg.GT_LAYERS)
            ])

        self.edge_init = nn.Sequential(
            nn.Linear(1, D), nn.GELU(), nn.LayerNorm(D),
        )

        self.classifier = ClassifierHead(
            input_dim=D, hidden_dim=D,
            n_classes=cfg.N_CLASSES,
            n_layers=cfg.FC_LAYERS,
            dropout=cfg.FC_DROPOUT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, W, NF = x.shape
        D           = self.cfg.HIDDEN_DIM
        h   = self.embedding(x)
        adj = build_ncc_adjacency(h, tau=self.cfg.TOP_TAU)
        spe = self.spe(adj)
        h   = h + self.spe_proj(spe)
        h   = self._forward_dgl(h, adj, B, C, D) if DGL_AVAILABLE \
              else self._forward_nodgl(h, adj, B, C, D)
        return self.classifier(h.mean(dim=1))

    def _forward_dgl(self, h, adj, B, C, D):
        graphs     = build_dgl_graphs(adj, device=h.device)
        h_out_list = []
        for b in range(B):
            g   = graphs[b]
            h_b = h[b]
            e_b = self.edge_init(g.edata['w'])
            for gt_layer in self.gt_layers:
                h_b, e_b = gt_layer(g, h_b, e_b)
            h_out_list.append(h_b)
        return torch.stack(h_out_list, dim=0)

    def _forward_nodgl(self, h, adj, B, C, D):
        # Proses tiap sampel dalam batch secara TERPISAH agar:
        # 1. Channel dari sampel berbeda tidak saling attend (no cross-sample contamination)
        # 2. Adjacency NCC per sampel digunakan untuk masking attention
        h_out = []
        for b in range(B):
            h_b = h[b]       # (C, D) — 18 channel satu pasien
            for gt_layer in self.gt_layers:
                h_b = gt_layer(h_b, adj[b])   # adj[b]: (C,C) masking per sampel
            h_out.append(h_b)
        return torch.stack(h_out, dim=0)   # (B, C, D)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        cfg = self.cfg
        return (
            f"GTSTAFG(\n"
            f"  input=({cfg.N_CHANNELS}ch, {cfg.N_WINDOWS}win, "
            f"{cfg.N_FEATURES}feat)\n"
            f"  hidden={cfg.HIDDEN_DIM}  layers={cfg.GT_LAYERS}  "
            f"heads={cfg.GT_HEADS}  pool={getattr(cfg,'POOL_MODE','mean')}\n"
            f"  spe={cfg.SPE_TYPE}  tau={cfg.TOP_TAU}  "
            f"fc={cfg.FC_LAYERS}\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )