import torch

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


# ======================================================================
# NCC ADJACENCY MATRIX
# ======================================================================

def build_ncc_adjacency(
    h_batch : torch.Tensor,
    tau     : int = 3,
) -> torch.Tensor:
    """
    Bangun sparse adjacency matrix berbasis |NCC| per-batch.

    NCC(h_i, h_j) = Σ(h_i-μ_i)(h_j-μ_j) / (||h_i-μ_i|| ||h_j-μ_j||)
                  = cosine similarity dari centered h

    Gunakan nilai absolut |NCC| karena:
      - Korelasi positif kuat (+1): electrode bergerak bersama
      - Korelasi negatif kuat (-1): electrode bergerak berlawanan
      - Keduanya menandakan hubungan fungsional yang kuat

    Args:
        h_batch : (B, C, D)  — node features (batch, channel, hidden_dim)
        tau     : top-τ tetangga per node (otomatis clamp ke C-1)

    Returns:
        adj : (B, C, C)  float32, sparse, undirected, non-negative
              diagonal = 0 (tanpa self-loop)
    """
    B, C, D = h_batch.shape

    # Clamp tau agar tidak melebihi jumlah node - 1
    tau = min(tau, C - 1)

    # ── Step 1: Centering (zero-mean per node) ────────────────────────
    mean = h_batch.mean(dim=-1, keepdim=True)       # (B, C, 1)
    h_c  = h_batch - mean                            # (B, C, D)

    # ── Step 2: L2 normalisasi (untuk cosine similarity) ─────────────
    norm = h_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (B, C, 1)
    h_n  = h_c / norm                                         # (B, C, D)

    # ── Step 3: NCC matrix = cosine similarity ────────────────────────
    ncc  = torch.bmm(h_n, h_n.transpose(1, 2))   # (B, C, C)

    # ── Step 4: Hilangkan self-loop ───────────────────────────────────
    eye  = torch.eye(C, device=h_batch.device, dtype=h_batch.dtype)
    ncc  = ncc * (1.0 - eye.unsqueeze(0))         # (B, C, C)

    # ── Step 5: Gunakan |NCC| untuk top-k ────────────────────────────
    # Anti-korelasi kuat (-1) sama informatifnya dengan korelasi kuat (+1)
    ncc_abs = ncc.abs()

    # ── Step 6: Top-τ sparse per node ────────────────────────────────
    adj = torch.zeros_like(ncc_abs)
    topk_vals, topk_idx = ncc_abs.topk(tau, dim=-1)   # (B, C, τ)
    adj.scatter_(-1, topk_idx, topk_vals)

    # ── Step 7: Jadikan undirected (simetris) ─────────────────────────
    # Jika i menganggap j sebagai tetangga ATAU j menganggap i tetangga
    adj = torch.max(adj, adj.transpose(1, 2))

    # ── Step 8: Normalisasi baris (row-stochastic untuk stabilitas) ───
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    adj     = adj / row_sum

    return adj.float()   # (B, C, C)


# ======================================================================
# DGL GRAPH BUILDER
# ======================================================================

def build_dgl_graphs(
    adj_batch : torch.Tensor,
    device    : torch.device,
) -> list:
    """
    Konversi adjacency matrix batch ke list DGL graph.

    Perbaikan dari versi lama:
      - Simpan bobot (E, 1) saja di g.edata['w']
        Proyeksi ke hidden_dim dilakukan di model (edge_init)
      - Handle isolated nodes per-node (self-loop lokal)
        bukan reset seluruh adjacency jika graph kosong

    Args:
        adj_batch : (B, C, C)  adjacency matrix (non-negatif, simetris)
        device    : torch device

    Returns:
        graphs : list[dgl.DGLGraph] panjang B
                 g.edata['w'] : (E, 1) float32 — bobot NCC
    """
    if not DGL_AVAILABLE:
        raise ImportError(
            "DGL tidak terinstall. Jalankan: pip install dgl dglgo"
        )

    B, C, _ = adj_batch.shape
    graphs   = []

    for b in range(B):
        adj_b = adj_batch[b]   # (C, C)

        # Cari semua edge aktif (bobot > threshold kecil)
        mask = adj_b > 1e-6
        src, dst = mask.nonzero(as_tuple=True)

        if len(src) == 0:
            # Seluruh graph kosong (sangat jarang) — tambah self-loop semua
            src = torch.arange(C, device=device)
            dst = torch.arange(C, device=device)
            g   = dgl.graph((src, dst), num_nodes=C).to(device)
            g.edata['w'] = torch.ones(C, 1, device=device,
                                      dtype=torch.float32)
            graphs.append(g)
            continue

        g = dgl.graph((src, dst), num_nodes=C).to(device)

        # Bobot edge: simpan sebagai (E, 1)
        # Proyeksi ke hidden_dim ada di model.py (edge_init)
        weights      = adj_b[src, dst].unsqueeze(-1).float()  # (E, 1)
        g.edata['w'] = weights

        # Handle isolated nodes: tambah self-loop hanya untuk node
        # yang degree-nya 0 (tidak punya tetangga)
        degree = g.in_degrees()   # (C,)
        isolated = (degree == 0).nonzero(as_tuple=True)[0]

        if len(isolated) > 0:
            # Tambah self-loop hanya untuk node terisolasi
            self_src = isolated
            self_dst = isolated
            g = dgl.add_edges(g, self_src, self_dst)
            # Bobot self-loop = nilai kecil (bukan 1 agar tidak dominan)
            n_new = len(isolated)
            new_w = torch.full((n_new, 1), 0.1,
                               device=device, dtype=torch.float32)
            g.edata['w'] = torch.cat(
                [g.edata['w'][:len(src)], new_w], dim=0
            )

        graphs.append(g)

    return graphs