"""
dashboard.py — Streamlit Dashboard Deteksi Kejang Epilepsi
Model  : GT-STAFG (Gated Graph Transformer – Spatial-Temporal Adaptive Feature Graph)
Dataset: CHB-MIT (24 pasien)
Metode : LOPO (Leave-One-Patient-Out Cross-Validation)
Fitur  : DWT Statistical Features (18 ch × 15 win × 21 feat)

Cara menjalankan:
    streamlit run dashboard.py

Persyaratan file input:
    - features.npy : shape (N, 18, 15, 21)  float32
    - labels.npy   : shape (N,)              int (0/1) — opsional untuk evaluasi
"""

from __future__ import annotations

import io
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP  (model.py & graph_construction.py ada di folder yang sama)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from model import GTSTAFG
    from graph_construction import build_ncc_adjacency
    MODEL_MODULE_OK = True
except Exception as _model_err:
    MODEL_MODULE_OK = False
    _MODEL_ERR_MSG  = str(_model_err)

try:
    import dgl                          # noqa: F401
    DGL_OK = True
except ImportError:
    DGL_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "models"

N_CHANNELS   = 18
N_WINDOWS    = 15
N_FEATURES   = 21
SEGMENT_SEC  = 30   # durasi satu segmen (detik)
BATCH_SIZE   = 64   # batch saat inferensi

CHANNEL_NAMES: list[str] = [
    'FP1-F7', 'F7-T7',  'T7-P7',  'P7-O1',
    'FP1-F3', 'F3-C3',  'C3-P3',  'P3-O1',
    'FP2-F4', 'F4-C4',  'C4-P4',  'P4-O2',
    'FP2-F8', 'F8-T8',  'T8-P8',  'P8-O2',
    'FZ-CZ',  'CZ-PZ',
]

# Nama region per kanal (untuk interpretasi anatomis)
CHANNEL_REGION: dict[str, tuple[str, str]] = {
    'FP1-F7': ('Frontal-Temporal',   'Kiri'),
    'F7-T7':  ('Temporal Anterior',  'Kiri'),
    'T7-P7':  ('Temporal Tengah',    'Kiri'),
    'P7-O1':  ('Temporo-Oksipital',  'Kiri'),
    'FP1-F3': ('Frontal',            'Kiri'),
    'F3-C3':  ('Fronto-Sentral',     'Kiri'),
    'C3-P3':  ('Sentro-Parietal',    'Kiri'),
    'P3-O1':  ('Parieto-Oksipital',  'Kiri'),
    'FP2-F4': ('Frontal',            'Kanan'),
    'F4-C4':  ('Fronto-Sentral',     'Kanan'),
    'C4-P4':  ('Sentro-Parietal',    'Kanan'),
    'P4-O2':  ('Parieto-Oksipital',  'Kanan'),
    'FP2-F8': ('Frontal-Temporal',   'Kanan'),
    'F8-T8':  ('Temporal Anterior',  'Kanan'),
    'T8-P8':  ('Temporal Tengah',    'Kanan'),
    'P8-O2':  ('Temporo-Oksipital',  'Kanan'),
    'FZ-CZ':  ('Sentral',            'Tengah'),
    'CZ-PZ':  ('Parietal',           'Tengah'),
}

# Posisi (x, y) kanal bipolar pada peta kepala 2D (skala 0.0–1.0)
# Kepala: pusat (0.5, 0.5), radius ≈ 0.44
CHANNEL_XY: dict[str, tuple[float, float]] = {
    # Rantai temporal kiri (busur luar kiri)
    'FP1-F7': (0.215, 0.840),  'F7-T7':  (0.095, 0.625),
    'T7-P7':  (0.095, 0.390),  'P7-O1':  (0.215, 0.175),
    # Rantai parasagital kiri
    'FP1-F3': (0.350, 0.865),  'F3-C3':  (0.280, 0.655),
    'C3-P3':  (0.280, 0.360),  'P3-O1':  (0.350, 0.150),
    # Rantai parasagital kanan
    'FP2-F4': (0.650, 0.865),  'F4-C4':  (0.720, 0.655),
    'C4-P4':  (0.720, 0.360),  'P4-O2':  (0.650, 0.150),
    # Rantai temporal kanan (busur luar kanan)
    'FP2-F8': (0.785, 0.840),  'F8-T8':  (0.905, 0.625),
    'T8-P8':  (0.905, 0.390),  'P8-O2':  (0.785, 0.175),
    # Garis tengah
    'FZ-CZ':  (0.500, 0.655),  'CZ-PZ':  (0.500, 0.360),
}

# Warna tema
CLR_SEIZURE    = '#EF4444'   # merah
CLR_NORMAL     = '#22C55E'   # hijau
CLR_PROB_LINE  = '#3B82F6'   # biru
CLR_THRESHOLD  = '#F59E0B'   # kuning/amber
CLR_BG         = '#0F172A'   # biru gelap
CLR_CARD       = '#1E293B'   # biru abu gelap


# ─────────────────────────────────────────────────────────────────────────────
# 1.  KONFIGURASI MODEL (reconstruct dari hyperparams yang tersimpan di .pt)
# ─────────────────────────────────────────────────────────────────────────────

class InferConfig:
    """Config minimal untuk load GTSTAFG dari checkpoint."""

    # Nilai default (akan di-override dari hyperparams checkpoint)
    N_CHANNELS  = N_CHANNELS
    N_WINDOWS   = N_WINDOWS
    N_FEATURES  = N_FEATURES
    N_CLASSES   = 2
    HIDDEN_DIM  = 128
    GT_LAYERS   = 2
    GT_HEADS    = 8
    GT_DROPOUT  = 0.3
    USE_BIAS    = True
    SPE_TYPE    = 'RWPE'
    RWPE_STEPS  = 4
    LPE_DIM     = 8
    TOP_TAU     = 3
    FC_LAYERS   = 3
    FC_DROPOUT  = 0.0
    POOL_MODE   = 'mean'
    LAYER_NORM  = True
    BATCH_NORM  = False
    DEVICE      = torch.device('cpu')

    @classmethod
    def from_dict(cls, hp: dict) -> 'InferConfig':
        cfg = cls()
        for k, v in hp.items():
            setattr(cfg, k, v)
        cfg.N_CHANNELS = N_CHANNELS
        cfg.N_WINDOWS  = N_WINDOWS
        cfg.N_FEATURES = N_FEATURES
        cfg.N_CLASSES  = 2
        cfg.DEVICE     = torch.device('cpu')
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCAN & LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def scan_models(output_dir: str) -> list[dict]:
    """Temukan semua file .pt di output_dir dan kembalikan list metadata."""
    dirpath = Path(output_dir)
    if not dirpath.exists():
        return []
    models = []
    for pt in sorted(dirpath.glob('model_fold_*.pt')):
        # Nama fold dari nama file: model_fold_chb01.pt → chb01
        fold = pt.stem.replace('model_fold_', '')
        models.append({'name': pt.name, 'path': str(pt), 'fold': fold})
    return models


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> tuple:
    """
    Load model GTSTAFG dari file .pt.

    Returns:
        (model, threshold, test_metrics, hyperparams)
    """
    if not MODEL_MODULE_OK:
        raise ImportError(f"Gagal import model: {_MODEL_ERR_MSG}")

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    hp          = ckpt['hyperparams']
    cfg         = InferConfig.from_dict(hp)
    test_met    = ckpt.get('test_metrics', {})
    threshold   = float(test_met.get('threshold', 0.5))

    model = GTSTAFG(cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    return model, threshold, test_met, hp


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFERENSI
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: 'GTSTAFG',
    features: np.ndarray,       # (N, 18, 15, 21)
    threshold: float,
    batch_size: int = BATCH_SIZE,
    progress_bar=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jalankan inferensi per batch.

    Returns:
        probs : (N,) float  — probabilitas seizure tiap segmen
        preds : (N,) int    — prediksi biner (0=normal, 1=seizure)
    """
    model.eval()
    N       = len(features)
    probs   = np.zeros(N, dtype=np.float32)
    n_steps = math.ceil(N / batch_size)

    for i in range(n_steps):
        start = i * batch_size
        end   = min(start + batch_size, N)
        x     = torch.tensor(features[start:end], dtype=torch.float32)

        # Ganti NaN/Inf
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        logits     = model(x)                          # (B, 2)
        batch_prob = F.softmax(logits, dim=-1)[:, 1]   # (B,) seizure prob
        probs[start:end] = batch_prob.numpy()

        if progress_bar is not None:
            progress_bar.progress((i + 1) / n_steps,
                                  text=f"Memproses segmen {end}/{N}...")

    preds = (probs >= threshold).astype(np.int32)
    return probs, preds


def compute_channel_importance(
    features: np.ndarray,   # (N, 18, 15, 21)
    preds:    np.ndarray,   # (N,) — prediksi biner
) -> np.ndarray:
    """
    Hitung kepentingan per kanal berdasarkan rata-rata energi fitur DWT
    pada segmen yang diprediksi seizure.

    Energi = rata-rata |fitur| per kanal, diambil dari segmen seizure saja.
    Nilai digunakan sebagai proxy aktivasi kanal selama kejang.

    Returns:
        importance : (18,) float32, dinormalisasi ke [0, 1]
    """
    seiz_mask = preds == 1
    if seiz_mask.sum() == 0:
        # Tidak ada prediksi seizure → gunakan seluruh data
        seiz_feats = features
    else:
        seiz_feats = features[seiz_mask]   # (K, 18, 15, 21)

    # Rata-rata energi absolut tiap kanal → (18,)
    importance = np.mean(np.abs(seiz_feats), axis=(0, 2, 3))   # (18,)

    # Normalisasi ke [0, 1]
    vmin, vmax = importance.min(), importance.max()
    if vmax > vmin:
        importance = (importance - vmin) / (vmax - vmin)
    else:
        importance = np.zeros_like(importance)

    return importance.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  KALKULASI METRIK
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Hitung metrik klasifikasi jika label tersedia."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, balanced_accuracy_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, matthews_corrcoef,
    )
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy'     : float(accuracy_score(labels, preds)),
        'recall'       : float(recall_score(labels, preds, zero_division=0)),
        'specificity'  : specificity,
        'precision'    : float(precision_score(labels, preds, zero_division=0)),
        'f1'           : float(f1_score(labels, preds, zero_division=0)),
        'balanced_acc' : float(balanced_accuracy_score(labels, preds)),
        'mcc'          : float(matthews_corrcoef(labels, preds)),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT: TIMELINE PROBABILITAS
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeline(
    probs:      np.ndarray,
    preds:      np.ndarray,
    threshold:  float,
    labels:     Optional[np.ndarray] = None,
    start_sec:  float = 0.0,
) -> go.Figure:
    """
    Plotly line chart: probabilitas kejang per segmen.

    - Garis biru       : probabilitas seizure (model output)
    - Batas kuning     : threshold keputusan
    - Blok merah       : prediksi seizure
    - Blok hijau muda  : ground truth seizure (jika label tersedia)
    """
    N       = len(probs)
    times   = [start_sec + i * SEGMENT_SEC for i in range(N)]  # detik
    times_m = [t / 60 for t in times]                           # menit

    fig = make_subplots(rows=1, cols=1)

    # ── Latar prediksi seizure (merah) ───────────────────────────────────────
    for i, (p, t) in enumerate(zip(preds, times_m)):
        if p == 1:
            fig.add_vrect(
                x0=t, x1=t + SEGMENT_SEC / 60,
                fillcolor=CLR_SEIZURE, opacity=0.15,
                layer='below', line_width=0,
            )

    # ── Latar ground truth (hijau muda, opsional) ────────────────────────────
    if labels is not None:
        for i, (lbl, t) in enumerate(zip(labels, times_m)):
            if lbl == 1:
                fig.add_vrect(
                    x0=t, x1=t + SEGMENT_SEC / 60,
                    fillcolor=CLR_NORMAL, opacity=0.10,
                    layer='below', line_width=0,
                )

    # ── Garis probabilitas ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=times_m, y=probs,
        mode='lines+markers',
        name='Probabilitas Kejang',
        line=dict(color=CLR_PROB_LINE, width=2),
        marker=dict(
            size=6,
            color=[CLR_SEIZURE if p == 1 else CLR_NORMAL for p in preds],
            line=dict(width=1, color='white'),
        ),
        hovertemplate=(
            '<b>Waktu</b>: %{x:.2f} menit<br>'
            '<b>Probabilitas</b>: %{y:.4f}<extra></extra>'
        ),
    ))

    # ── Garis threshold ──────────────────────────────────────────────────────
    fig.add_hline(
        y=threshold, line_dash='dash',
        line_color=CLR_THRESHOLD, line_width=2,
        annotation_text=f'  Threshold = {threshold:.3f}',
        annotation_position='top left',
        annotation_font_color=CLR_THRESHOLD,
    )

    # ── Legend manual (vrect tidak masuk legend otomatis) ────────────────────
    legend_items = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   name='Prediksi: Kejang',
                   marker=dict(color=CLR_SEIZURE, size=10, symbol='square')),
        go.Scatter(x=[None], y=[None], mode='markers',
                   name='Prediksi: Normal',
                   marker=dict(color=CLR_NORMAL, size=10, symbol='square')),
    ]
    if labels is not None:
        legend_items.append(
            go.Scatter(x=[None], y=[None], mode='markers',
                       name='Ground Truth: Kejang',
                       marker=dict(color=CLR_NORMAL, size=10, symbol='square',
                                   opacity=0.4))
        )
    for item in legend_items:
        fig.add_trace(item)

    fig.update_layout(
        title=dict(
            text='<b>Timeline Deteksi Kejang</b>',
            x=0.5, font=dict(size=16),
        ),
        xaxis=dict(
            title='Waktu (menit)',
            showgrid=True, gridcolor='#334155', gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            title='Probabilitas Kejang',
            range=[-0.05, 1.05],
            showgrid=True, gridcolor='#334155', gridwidth=1,
            tickformat='.2f',
        ),
        paper_bgcolor=CLR_BG,
        plot_bgcolor='#1E293B',
        font=dict(color='white'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
        ),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=60, b=50),
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT: BAR CHART KEPENTINGAN KANAL
# ─────────────────────────────────────────────────────────────────────────────

def plot_channel_bar(importance: np.ndarray) -> go.Figure:
    """Horizontal bar chart: kepentingan 18 kanal EEG saat kejang."""
    sorted_idx = np.argsort(importance)[::-1]
    ch_names   = [CHANNEL_NAMES[i] for i in sorted_idx]
    ch_vals    = [float(importance[i]) for i in sorted_idx]
    regions    = [CHANNEL_REGION[n] for n in ch_names]
    colors     = [
        CLR_SEIZURE if v >= 0.7 else
        ('#F97316' if v >= 0.4 else '#3B82F6')
        for v in ch_vals
    ]
    hover = [
        f'<b>{ch}</b><br>Region: {r[0]} {r[1]}<br>Kepentingan: {v:.3f}'
        for ch, r, v in zip(ch_names, regions, ch_vals)
    ]

    fig = go.Figure(go.Bar(
        x=ch_vals[::-1],
        y=ch_names[::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f'{v:.3f}' for v in ch_vals[::-1]],
        textposition='outside',
        hovertext=hover[::-1],
        hoverinfo='text',
    ))
    fig.update_layout(
        title=dict(
            text='<b>Kepentingan Kanal EEG Saat Kejang</b>',
            x=0.5, font=dict(size=15),
        ),
        xaxis=dict(title='Skor Kepentingan (dinormalisasi)', range=[0, 1.15],
                   showgrid=True, gridcolor='#334155'),
        yaxis=dict(title=''),
        paper_bgcolor=CLR_BG,
        plot_bgcolor='#1E293B',
        font=dict(color='white'),
        height=550,
        margin=dict(l=100, r=80, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOT: PETA OTAK 2D
# ─────────────────────────────────────────────────────────────────────────────

def plot_brain_map(importance: np.ndarray) -> plt.Figure:
    """
    Gambar heatmap kepentingan kanal di atas siluet kepala 2D.
    Warna merah = aktivitas tinggi (kemungkinan sumber kejang).
    """
    fig, ax = plt.subplots(figsize=(6, 6.5))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Siluet kepala ────────────────────────────────────────────────────────
    head = plt.Circle((0.5, 0.50), 0.43,
                       color='#334155', linewidth=2,
                       ec='#64748B', fill=True, zorder=1)
    ax.add_patch(head)

    # Hidung
    nose_x = [0.47, 0.50, 0.53]
    nose_y = [0.935, 0.980, 0.935]
    ax.plot(nose_x, nose_y, color='#64748B', linewidth=2, zorder=2)

    # Telinga kiri & kanan
    for ear_x, ear_y in [(0.065, 0.50), (0.935, 0.50)]:
        ear = mpatches.Ellipse((ear_x, ear_y), 0.04, 0.08,
                               color='#334155', ec='#64748B',
                               linewidth=1.5, zorder=2)
        ax.add_patch(ear)

    # ── Colormap: biru–kuning–merah ──────────────────────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'eeg', ['#1E3A5F', '#3B82F6', '#F59E0B', '#EF4444']
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # ── Plot tiap kanal ───────────────────────────────────────────────────────
    for i, ch in enumerate(CHANNEL_NAMES):
        x, y    = CHANNEL_XY[ch]
        val     = float(importance[i])
        color   = cmap(norm(val))
        radius  = 0.032 + val * 0.025   # radius proporsional dengan kepentingan

        circle = plt.Circle((x, y), radius, color=color,
                             ec='white', linewidth=0.8, zorder=4, alpha=0.92)
        ax.add_patch(circle)

        # Label kanal (font kecil)
        ax.text(x, y - radius - 0.025, ch,
                ha='center', va='top', fontsize=5.5,
                color='#CBD5E1', zorder=5)

        # Nilai kepentingan di dalam lingkaran (hanya jika > 0.3)
        if val >= 0.3:
            ax.text(x, y, f'{val:.2f}',
                    ha='center', va='center', fontsize=5,
                    color='white', fontweight='bold', zorder=5)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02,
                        orientation='vertical')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Kepentingan Kanal', color='white', fontsize=9)
    cbar.outline.set_edgecolor('#64748B')

    ax.set_title('Peta Aktivasi Kanal EEG\n(Merah = Aktivitas Tinggi Saat Kejang)',
                 color='white', fontsize=11, pad=10)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(tp, tn, fp, fn) -> go.Figure:
    z    = [[tn, fp], [fn, tp]]
    text = [[f'TN\n{tn}', f'FP\n{fp}'], [f'FN\n{fn}', f'TP\n{tp}']]
    fig  = go.Figure(go.Heatmap(
        z=z,
        x=['Prediksi Normal', 'Prediksi Kejang'],
        y=['Label Normal', 'Label Kejang'],
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=18, color='white'),
        colorscale=[[0, '#1E3A5F'], [0.5, '#3B82F6'], [1, '#EF4444']],
        showscale=False,
        hovertemplate='%{y} → %{x}: %{z}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='<b>Confusion Matrix</b>', x=0.5),
        paper_bgcolor=CLR_BG, plot_bgcolor=CLR_BG,
        font=dict(color='white', size=12),
        height=280, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOT: REGION BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def plot_region_importance(importance: np.ndarray) -> go.Figure:
    """Pie chart: rata-rata kepentingan per region otak."""
    region_scores: dict[str, list[float]] = {}
    for i, ch in enumerate(CHANNEL_NAMES):
        reg, side = CHANNEL_REGION[ch]
        key = f"{reg} ({side})"
        region_scores.setdefault(key, []).append(float(importance[i]))

    labels = list(region_scores.keys())
    values = [np.mean(v) for v in region_scores.values()]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.35,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Skor rata-rata: %{value:.3f}<extra></extra>',
        marker=dict(
            colors=px.colors.qualitative.Bold[:len(labels)],
            line=dict(color='#0F172A', width=1.5),
        ),
    ))
    fig.update_layout(
        title=dict(text='<b>Kontribusi Region Otak Saat Kejang</b>', x=0.5),
        paper_bgcolor=CLR_BG,
        font=dict(color='white', size=10),
        height=400,
        legend=dict(font=dict(size=9)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. HELPER UI
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = '',
                color: str = CLR_NORMAL) -> str:
    """HTML card untuk satu metrik di dashboard."""
    delta_html = (
        f'<span style="font-size:0.75rem;color:#94A3B8">{delta}</span>'
        if delta else ''
    )
    return f"""
    <div style="
        background:{CLR_CARD};border-radius:12px;padding:16px 20px;
        border-left:4px solid {color};margin-bottom:8px;
    ">
        <div style="font-size:0.8rem;color:#94A3B8;margin-bottom:4px">{label}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{color}">{value}</div>
        {delta_html}
    </div>
    """


def results_to_df(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: Optional[np.ndarray] = None,
    start_sec: float = 0.0,
) -> pd.DataFrame:
    """Buat DataFrame hasil deteksi untuk ditampilkan dan diunduh."""
    N  = len(probs)
    df = pd.DataFrame({
        'Segmen'         : np.arange(1, N + 1),
        'Waktu Mulai'    : [f'{int((start_sec + i*SEGMENT_SEC)//60):02d}:'
                            f'{int((start_sec + i*SEGMENT_SEC)%60):02d}'
                            for i in range(N)],
        'Waktu Selesai'  : [f'{int((start_sec + (i+1)*SEGMENT_SEC)//60):02d}:'
                            f'{int((start_sec + (i+1)*SEGMENT_SEC)%60):02d}'
                            for i in range(N)],
        'Prob. Kejang'   : np.round(probs, 4),
        'Prediksi'       : ['⚠️ Kejang' if p == 1 else '✅ Normal'
                            for p in preds],
    })
    if labels is not None:
        df['Label GT'] = ['⚠️ Kejang' if l == 1 else '✅ Normal'
                          for l in labels]
        df['Benar']    = ['✓' if p == l else '✗'
                          for p, l in zip(preds, labels)]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 11. HALAMAN UTAMA STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import math  # pastikan tersedia di scope ini

    st.set_page_config(
        page_title   = 'Dashboard Deteksi Kejang Epilepsi',
        page_icon    = '',
        layout       = 'wide',
        initial_sidebar_state = 'expanded',
    )

    # CSS global
    st.markdown("""
    <style>
        .main { background-color: #0F172A; color: white; }
        .stTabs [data-baseweb="tab-list"] { gap: 12px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E293B;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            color: #94A3B8;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3B82F6 !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #1E293B;
        }
        .stProgress > div > div { background-color: #3B82F6; }
        h1, h2, h3 { color: white; }
        .stDataFrame { background-color: #1E293B; }
    </style>
    """, unsafe_allow_html=True)

    # ── JUDUL ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:24px 0 8px 0">
        <h1 style="font-size:2rem;font-weight:800;color:white">
            🧠 Dashboard Deteksi Kejang Epilepsi
        </h1>
        <p style="color:#94A3B8;font-size:1rem">
            GT-STAFG · DWT Statistical Features · LOPO Cross-Validation · CHB-MIT
        </p>
        <hr style="border-color:#334155;margin-top:12px">
    </div>
    """, unsafe_allow_html=True)

    # ── CEK DEPENDENSI ────────────────────────────────────────────────────────
    if not MODEL_MODULE_OK:
        st.error(f"❌ **Gagal import model.py:** {_MODEL_ERR_MSG}\n\n"
                 "Pastikan `model.py` dan `graph_construction.py` ada di folder yang sama "
                 "dengan `dashboard.py`.")
        st.stop()
    if not DGL_OK:
        st.warning("⚠️ **DGL tidak terinstall.** Inferensi menggunakan fallback "
                   "(GTLayerNoDGL). Hasil mungkin berbeda dari model yang dilatih dengan DGL.\n\n"
                   "Install DGL: `pip install dgl -f https://data.dgl.ai/wheels/repo.html`")

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Konfigurasi")
        st.markdown("---")

        # Direktori output
        output_dir = st.text_input(
            "📁 Direktori Model (.pt)",
            value=DEFAULT_OUTPUT_DIR,
            help="Folder tempat file model_fold_*.pt tersimpan",
        )

        # Scan model yang tersedia
        available_models = scan_models(output_dir)
        if not available_models:
            st.error(f"Tidak ada file `model_fold_*.pt` di:\n`{output_dir}`")
            st.stop()

        model_names  = [m['name'] for m in available_models]
        selected_idx = st.selectbox(
            "🤖 Pilih Model (Fold LOPO)",
            range(len(model_names)),
            format_func=lambda i: model_names[i],
            help="Tiap model dilatih pada 22 pasien, diuji pada 1 pasien.",
        )
        selected_model = available_models[selected_idx]

        st.markdown("---")
        st.markdown("### 🎛️ Parameter Deteksi")

        # Load model untuk dapatkan threshold default
        with st.spinner("Memuat model..."):
            try:
                model, default_thresh, test_metrics, hp = load_model(
                    selected_model['path']
                )
                model_loaded = True
            except Exception as e:
                st.error(f"Gagal memuat model: {e}")
                model_loaded = False

        threshold = st.slider(
            "Threshold Keputusan",
            min_value=0.0, max_value=1.0,
            value=float(default_thresh) if model_loaded else 0.5,
            step=0.01,
            help="Probabilitas di atas nilai ini → prediksi Kejang.\n"
                 "Nilai default dari threshold optimal saat training.",
        )

        start_time = st.number_input(
            "Waktu Mulai Rekaman (detik)",
            min_value=0, value=0, step=30,
            help="Offset waktu untuk label sumbu X pada grafik timeline.",
        )

        st.markdown("---")
        if model_loaded:
            fold_str = selected_model['fold']
            st.markdown(f"### 📊 Metrik Pelatihan ({fold_str})")
            st.markdown(f"""
            | Metrik | Nilai |
            |--------|-------|
            | F1-Score | `{test_metrics.get('f1', 0):.4f}` |
            | Recall (Sensitivitas) | `{test_metrics.get('recall', 0):.4f}` |
            | Specificity | `{test_metrics.get('specificity', 0):.4f}` |
            | Precision | `{test_metrics.get('precision', 0):.4f}` |
            | AUC-ROC | `{test_metrics.get('auc_roc', 0):.4f}` |
            | PR-AUC | `{test_metrics.get('pr_auc', 0):.4f}` |
            | Balanced Acc | `{test_metrics.get('balanced_acc', 0):.4f}` |
            | Threshold | `{test_metrics.get('threshold', 0.5):.4f}` |
            """)

            tp = test_metrics.get('tp', 0)
            tn = test_metrics.get('tn', 0)
            fp = test_metrics.get('fp', 0)
            fn = test_metrics.get('fn', 0)
            st.caption(f"TP={tp} TN={tn} FP={fp} FN={fn}")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#64748B;text-align:center">
            GT-STAFG · CHB-MIT · Skripsi S1<br>
            © 2024 — Universitas
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB UTAMA
    # ──────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Deteksi",
        "📈 Hasil Deteksi",
        "🧠 Lokasi Kejang",
        "🔬 Info Model",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: UPLOAD & DETEKSI
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 📤 Unggah Data EEG")

        col_info, col_fmt = st.columns([2, 1])
        with col_info:
            st.markdown("""
            Upload file fitur DWT yang sudah diekstrak dari sinyal EEG.
            Dashboard akan menjalankan model GT-STAFG untuk mendeteksi
            segmen kejang dan menampilkan lokasi anatomis aktivitas tertinggi.
            """)
        with col_fmt:
            st.markdown("""
            **Format yang diterima:**
            - `features.npy` → shape **(N, 18, 15, 21)**
            - `labels.npy`   → shape **(N,)** — opsional
            """)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            feat_file = st.file_uploader(
                "📂 features.npy (wajib)",
                type=['npy'],
                help="Array fitur DWT shape (N, 18, 15, 21) — N segmen × "
                     "18 kanal × 15 jendela × 21 fitur",
            )
        with c2:
            label_file = st.file_uploader(
                "📂 labels.npy (opsional, untuk evaluasi)",
                type=['npy'],
                help="Label biner shape (N,) — 0=normal, 1=kejang. "
                     "Diperlukan untuk menghitung metrik akurasi.",
            )

        # ── Validasi & preview file ───────────────────────────────────────────
        if feat_file is not None:
            try:
                features = np.load(io.BytesIO(feat_file.read()))
            except Exception as e:
                st.error(f"Gagal membaca features.npy: {e}")
                st.stop()

            # Validasi shape
            expected = (None, N_CHANNELS, N_WINDOWS, N_FEATURES)
            if (features.ndim != 4 or
                    features.shape[1] != N_CHANNELS or
                    features.shape[2] != N_WINDOWS or
                    features.shape[3] != N_FEATURES):
                st.error(
                    f"❌ **Shape tidak sesuai.**\n\n"
                    f"Diharapkan: `(N, {N_CHANNELS}, {N_WINDOWS}, {N_FEATURES})`\n"
                    f"Diterima  : `{features.shape}`\n\n"
                    f"Pastikan fitur sudah diekstrak dengan pipeline DWT yang benar."
                )
                st.stop()

            N_seg = features.shape[0]
            dur   = N_seg * SEGMENT_SEC

            # Ganti NaN/Inf
            n_bad = int(np.sum(~np.isfinite(features)))
            if n_bad > 0:
                features = np.nan_to_num(features,
                                         nan=0.0, posinf=0.0, neginf=0.0)
                st.warning(f"⚠️ Ditemukan {n_bad} nilai NaN/Inf → diganti 0.")

            # Info file
            st.success(f"✅ **features.npy** berhasil dimuat")
            mi, ma = float(features.min()), float(features.max())
            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.metric("Jumlah Segmen", f"{N_seg}")
            ci2.metric("Total Durasi", f"{dur//60:.0f} mnt {dur%60:.0f} dtk")

            # Load label (opsional)
            labels = None
            if label_file is not None:
                try:
                    labels = np.load(io.BytesIO(label_file.read())).astype(np.int32)
                    if labels.shape[0] != N_seg:
                        st.error(f"❌ Panjang labels.npy ({labels.shape[0]}) "
                                 f"tidak sama dengan features.npy ({N_seg}).")
                        labels = None
                    else:
                        n_seiz = int(np.sum(labels == 1))
                        n_norm = int(np.sum(labels == 0))
                        st.success(
                            f"✅ **labels.npy** berhasil dimuat — "
                            f"Kejang: {n_seiz} segmen | Normal: {n_norm} segmen"
                        )
                except Exception as e:
                    st.error(f"Gagal membaca labels.npy: {e}")

            st.markdown("---")

            # ── TOMBOL DETEKSI ────────────────────────────────────────────────
            if not model_loaded:
                st.error("Model belum berhasil dimuat. Periksa path model di sidebar.")
            else:
                col_btn, col_help = st.columns([1, 3])
                with col_btn:
                    run_btn = st.button(
                        "🔍 Jalankan Deteksi",
                        type="primary",
                        use_container_width=True,
                    )
                with col_help:
                    st.caption(
                        f"Model: **{selected_model['name']}** · "
                        f"Threshold: **{threshold:.3f}** · "
                        f"Segmen: **{N_seg}**"
                    )

                if run_btn:
                    # Progress bar
                    prog = st.progress(0, text="Memulai inferensi...")
                    with st.spinner(""):
                        try:
                            probs, preds = run_inference(
                                model, features, threshold,
                                batch_size=BATCH_SIZE,
                                progress_bar=prog,
                            )
                            prog.progress(1.0, text="Selesai!")
                        except Exception as e:
                            prog.empty()
                            st.error(f"❌ Inferensi gagal: {e}")
                            st.exception(e)
                            st.stop()

                    # Simpan ke session_state
                    st.session_state['probs']    = probs
                    st.session_state['preds']    = preds
                    st.session_state['features'] = features
                    st.session_state['labels']   = labels
                    st.session_state['N_seg']    = N_seg
                    st.session_state['start_sec']= float(start_time)
                    st.session_state['model_name']= selected_model['name']
                    st.session_state['threshold']= threshold

                    n_seiz_pred = int(np.sum(preds == 1))
                    st.success(
                        f"✅ **Deteksi selesai!** "
                        f"Ditemukan **{n_seiz_pred}** segmen kejang "
                        f"dari {N_seg} total segmen "
                        f"({n_seiz_pred/N_seg*100:.1f}%).\n\n"
                        "👉 Lihat hasil di tab **Hasil Deteksi** dan **Lokasi Kejang**."
                    )
        else:
            # Placeholder ketika belum upload
            st.info(
                "📌 **Langkah penggunaan:**\n"
                "1. Upload `features.npy` (wajib) — hasil ekstraksi DWT\n"
                "2. Upload `labels.npy` (opsional) — untuk evaluasi performa\n"
                "3. Klik **Jalankan Deteksi**\n"
                "4. Lihat hasil di tab **Hasil Deteksi** dan **Lokasi Kejang**\n\n"
                "---\n"
                "**Cara mendapatkan features.npy:**\n"
                "```\n"
                "# Jalankan pipeline DWT ekstraksi terlebih dahulu:\n"
                "# (2. TRANSFORMASI WAVELET) → output: features.npy, labels.npy\n"
                "```"
            )

            # Demo dengan data sintetis
            st.markdown("#### 🧪 Atau gunakan data demo sintetis")
            if st.button("Generate Data Demo (50 segmen)"):
                rng  = np.random.default_rng(42)
                N_d  = 50
                feat_demo  = rng.normal(0, 1,
                                        (N_d, N_CHANNELS, N_WINDOWS, N_FEATURES)
                                        ).astype(np.float32)
                # Simulasi energi tinggi di segmen ke-15–20 dan 35–40
                for seiz_seg in list(range(15, 21)) + list(range(35, 41)):
                    feat_demo[seiz_seg] *= 3.5
                label_demo = np.zeros(N_d, dtype=np.int32)
                label_demo[15:21] = 1
                label_demo[35:41] = 1

                buf_f = io.BytesIO()
                buf_l = io.BytesIO()
                np.save(buf_f, feat_demo);  buf_f.seek(0)
                np.save(buf_l, label_demo); buf_l.seek(0)

                st.download_button("⬇️ Unduh features_demo.npy",
                                   buf_f, "features_demo.npy")
                st.download_button("⬇️ Unduh labels_demo.npy",
                                   buf_l, "labels_demo.npy")
                st.info("Upload kedua file di atas ke kolom upload untuk mencoba dashboard.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: HASIL DETEKSI
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        if 'probs' not in st.session_state:
            st.info("⏳ Jalankan deteksi di tab **Upload & Deteksi** terlebih dahulu.")
        else:
            probs     = st.session_state['probs']
            preds     = st.session_state['preds']
            labels    = st.session_state['labels']
            N_seg     = st.session_state['N_seg']
            start_sec = st.session_state['start_sec']
            thr       = st.session_state['threshold']

            n_seiz_pred = int(np.sum(preds == 1))
            n_norm_pred = N_seg - n_seiz_pred
            dur_seiz    = n_seiz_pred * SEGMENT_SEC

            # ── Kartu ringkasan ───────────────────────────────────────────────
            st.markdown("### 📊 Ringkasan Hasil Deteksi")
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.markdown(
                    metric_card("Total Segmen", str(N_seg),
                                f"{N_seg * SEGMENT_SEC // 60:.0f} menit rekaman",
                                '#3B82F6'),
                    unsafe_allow_html=True,
                )
            with rc2:
                st.markdown(
                    metric_card("Segmen Kejang",
                                str(n_seiz_pred),
                                f"{n_seiz_pred/N_seg*100:.1f}% dari total",
                                CLR_SEIZURE),
                    unsafe_allow_html=True,
                )
            with rc3:
                st.markdown(
                    metric_card("Segmen Normal",
                                str(n_norm_pred),
                                f"{n_norm_pred/N_seg*100:.1f}% dari total",
                                CLR_NORMAL),
                    unsafe_allow_html=True,
                )
            with rc4:
                st.markdown(
                    metric_card("Durasi Kejang",
                                f"{dur_seiz // 60:.0f}m {dur_seiz % 60:.0f}s",
                                f"Threshold: {thr:.3f}",
                                '#F59E0B'),
                    unsafe_allow_html=True,
                )

            # ── Metrik evaluasi (jika label tersedia) ────────────────────────
            if labels is not None:
                st.markdown("---")
                st.markdown("### 🎯 Metrik Evaluasi (vs. Ground Truth)")
                metrics = compute_metrics(preds, labels)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("F1-Score",    f"{metrics['f1']:.4f}")
                mc2.metric("Sensitivity", f"{metrics['recall']:.4f}",
                           help="Recall/Sensitivity = TP/(TP+FN)")
                mc3.metric("Specificity", f"{metrics['specificity']:.4f}",
                           help="Specificity = TN/(TN+FP)")
                mc4.metric("Balanced Acc", f"{metrics['balanced_acc']:.4f}")

                mc5, mc6, mc7, mc8 = st.columns(4)
                mc5.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
                mc6.metric("Precision", f"{metrics['precision']:.4f}")
                mc7.metric("MCC",       f"{metrics['mcc']:.4f}")
                mc8.metric("",          "")

                cf_col, _ = st.columns([1, 2])
                with cf_col:
                    st.plotly_chart(
                        plot_confusion_matrix(
                            metrics['tp'], metrics['tn'],
                            metrics['fp'], metrics['fn'],
                        ),
                        use_container_width=True,
                    )

            # ── Timeline ─────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📈 Timeline Probabilitas Kejang")
            st.plotly_chart(
                plot_timeline(probs, preds, thr, labels, start_sec),
                use_container_width=True,
            )
            st.caption(
                "🔴 Blok merah = segmen diprediksi **kejang** | "
                "🟢 Blok hijau = ground truth **kejang** (jika label tersedia) | "
                "🟡 Garis putus = threshold"
            )

            # ── Tabel hasil ───────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Tabel Segmen")

            df = results_to_df(probs, preds, labels, start_sec)

            # Filter tampilan
            filter_opt = st.radio(
                "Tampilkan:",
                ["Semua", "Hanya Kejang", "Hanya Normal"],
                horizontal=True,
            )
            if filter_opt == "Hanya Kejang":
                df_show = df[preds == 1]
            elif filter_opt == "Hanya Normal":
                df_show = df[preds == 0]
            else:
                df_show = df

            st.dataframe(
                df_show,
                use_container_width=True,
                hide_index=True,
                height=min(400, 38 * (len(df_show) + 1)),
            )

            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Unduh Hasil sebagai CSV",
                data=csv,
                file_name=f"hasil_deteksi_{selected_model['fold']}.csv",
                mime='text/csv',
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: LOKASI KEJANG
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        if 'probs' not in st.session_state:
            st.info("⏳ Jalankan deteksi di tab **Upload & Deteksi** terlebih dahulu.")
        else:
            probs    = st.session_state['probs']
            preds    = st.session_state['preds']
            features = st.session_state['features']

            st.markdown("### 🧠 Analisis Lokasi Kejang")
            st.markdown("""
            Visualisasi di bawah menunjukkan **kanal EEG mana** yang memiliki
            aktivitas tertinggi pada segmen yang diprediksi **kejang**.
            Nilai kepentingan dihitung dari rata-rata energi absolut fitur DWT
            tiap kanal selama segmen kejang.
            """)

            # Hitung kepentingan kanal
            importance = compute_channel_importance(features, preds)

            n_seiz_pred = int(np.sum(preds == 1))
            if n_seiz_pred == 0:
                st.warning("⚠️ Tidak ada segmen yang diprediksi kejang. "
                           "Coba turunkan threshold di sidebar.")
            else:
                st.info(f"Analisis berdasarkan **{n_seiz_pred}** segmen yang diprediksi kejang.")

            # ── Kanal paling aktif ─────────────────────────────────────────────
            top3_idx  = np.argsort(importance)[::-1][:3]
            top3_ch   = [CHANNEL_NAMES[i] for i in top3_idx]
            top3_reg  = [f"{CHANNEL_REGION[c][0]} ({CHANNEL_REGION[c][1]})"
                         for c in top3_ch]
            top3_val  = [importance[i] for i in top3_idx]

            st.markdown("#### 🔝 3 Kanal dengan Aktivitas Tertinggi Saat Kejang")
            lt1, lt2, lt3 = st.columns(3)
            for col, ch, reg, val in zip([lt1, lt2, lt3], top3_ch, top3_reg, top3_val):
                col.markdown(
                    metric_card(ch, f"{val:.3f}", reg, CLR_SEIZURE),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Layout 2 kolom: bar chart + brain map ─────────────────────────
            col_bar, col_brain = st.columns([1, 1])

            with col_bar:
                st.markdown("#### 📊 Kepentingan Semua Kanal")
                st.plotly_chart(
                    plot_channel_bar(importance),
                    use_container_width=True,
                )

                st.markdown("#### 🗂️ Kontribusi Per Region Otak")
                st.plotly_chart(
                    plot_region_importance(importance),
                    use_container_width=True,
                )

            with col_brain:
                st.markdown("#### 🗺️ Peta Aktivasi Kepala")
                brain_fig = plot_brain_map(importance)
                st.pyplot(brain_fig, use_container_width=True)
                plt.close(brain_fig)

                st.markdown("#### 📋 Tabel Kepentingan Kanal")
                imp_df = pd.DataFrame({
                    'Kanal'    : CHANNEL_NAMES,
                    'Region'   : [f"{CHANNEL_REGION[c][0]}" for c in CHANNEL_NAMES],
                    'Sisi'     : [CHANNEL_REGION[c][1] for c in CHANNEL_NAMES],
                    'Kepentingan': np.round(importance, 4),
                }).sort_values('Kepentingan', ascending=False).reset_index(drop=True)
                imp_df.index = imp_df.index + 1

                st.dataframe(
                    imp_df,
                    use_container_width=True,
                    hide_index=False,
                    height=380,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: INFO MODEL
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 🔬 Informasi Model GT-STAFG")

        if not model_loaded:
            st.error("Model belum berhasil dimuat.")
        else:
            # ── Metrik training ───────────────────────────────────────────────
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### 📊 Metrik Test (dari Training)")
                metrics_data = {
                    'Metrik'     : ['Accuracy', 'F1-Score', 'Recall',
                                    'Specificity', 'Precision',
                                    'Balanced Accuracy', 'MCC',
                                    'AUC-ROC', 'PR-AUC', 'Threshold'],
                    'Nilai'      : [
                        f"{test_metrics.get('accuracy', 0):.6f}",
                        f"{test_metrics.get('f1', 0):.6f}",
                        f"{test_metrics.get('recall', 0):.6f}",
                        f"{test_metrics.get('specificity', 0):.6f}",
                        f"{test_metrics.get('precision', 0):.6f}",
                        f"{test_metrics.get('balanced_acc', 0):.6f}",
                        f"{test_metrics.get('mcc', 0):.6f}",
                        f"{test_metrics.get('auc_roc', 0):.6f}",
                        f"{test_metrics.get('pr_auc', 0):.6f}",
                        f"{test_metrics.get('threshold', 0.5):.6f}",
                    ],
                }
                st.dataframe(pd.DataFrame(metrics_data),
                             use_container_width=True, hide_index=True)

                # Confusion matrix dari training
                st.markdown("#### 🔲 Confusion Matrix Training")
                st.plotly_chart(
                    plot_confusion_matrix(
                        test_metrics.get('tp', 0),
                        test_metrics.get('tn', 0),
                        test_metrics.get('fp', 0),
                        test_metrics.get('fn', 0),
                    ),
                    use_container_width=True,
                )

            with col_m2:
                st.markdown("#### ⚙️ Hyperparameter Model")
                hp_items = [(k, str(v)) for k, v in hp.items()]
                hp_df    = pd.DataFrame(hp_items,
                                        columns=['Parameter', 'Nilai'])
                st.dataframe(hp_df, use_container_width=True, hide_index=True,
                             height=500)

            # ── Arsitektur model ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🏗️ Arsitektur GT-STAFG")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                n_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
                st.markdown(f"""
                | Komponen | Konfigurasi |
                |----------|-------------|
                | Input Shape | `(B, 18, 15, 21)` |
                | Hidden Dim | `{hp.get('HIDDEN_DIM', 128)}` |
                | GT Layers | `{hp.get('GT_LAYERS', 2)}` |
                | Attention Heads | `{hp.get('GT_HEADS', 8)}` |
                | Dropout | `{hp.get('GT_DROPOUT', 0.3)}` |
                | Spatial PE | `{hp.get('SPE_TYPE', 'RWPE')}` |
                | RWPE Steps | `{hp.get('RWPE_STEPS', 4)}` |
                | Graph τ (NCC) | `{hp.get('TOP_TAU', 3)}` |
                | FC Layers | `{hp.get('FC_LAYERS', 3)}` |
                | **Total Parameter** | **`{n_params:,}`** |
                """)
            with col_a2:
                st.markdown(f"""
                | Setting | Nilai |
                |---------|-------|
                | Normalisasi | `{'LayerNorm' if hp.get('LAYER_NORM') else 'BatchNorm'}` |
                | Pool Mode | `{hp.get('POOL_MODE', 'mean')}` |
                | LR Scheduler | `{hp.get('LR_SCHEDULER', 'cosine')}` |
                | Learning Rate | `{hp.get('LEARNING_RATE', 1e-4)}` |
                | Weight Decay | `{hp.get('WEIGHT_DECAY', 1e-4)}` |
                | Batch Size | `{hp.get('BATCH_SIZE', 32)}` |
                | Focal γ | `{hp.get('FOCAL_GAMMA', 0.5)}` |
                | Oversampling | `{'ROS' if hp.get('USE_ROS') else 'SMOTE' if hp.get('USE_SMOTE') else 'Downsampling'}` |
                | Monitor Metric | `{hp.get('MONITOR_METRIC', 'val_pr_auc')}` |
                | DGL | `{'✓ Tersedia' if DGL_OK else '✗ Tidak tersedia'}` |
                """)

            # ── Tentang pipeline ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📖 Pipeline Lengkap")
            st.markdown("""
            ```
            Raw EEG (EDF)
                ↓ [1. Preprocessing]
                  Resampling 256 Hz → Bandpass 0.5–70 Hz → Notch 50/60 Hz
                  Segmentasi 30 detik → 18 Kanal Bipolar
                ↓ [2. DWT Feature Extraction]
                  DWT 5-level (db4) → cD5, cD4, cD3 (3 sub-band)
                  7 Fitur Statistik per sub-band × 15 jendela 2 detik
                  Output: features.npy (N, 18, 15, 21)
                ↓ [3. GT-STAFG Model]
                  InputEmbedding: (B,18,15,21) → (B,18,D)
                  NCC Adjacency: (B,18,D) → (B,18,18)
                  RWPE/LPE Spatial PE: (B,18,18) → (B,18,D)
                  GT Layers × 2: Graph Attention + FFN
                  Mean Pool + Classifier Head → (B,2) logits
                ↓ [4. Dashboard — Dashboard ini]
                  Upload features.npy → Inference → Timeline + Brain Map
            ```
            """)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
import math   # diperlukan di plot_timeline dan run_inference

if __name__ == '__main__':
    main()
