from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_USE_PIN   = torch.cuda.is_available()
_N_WORKERS = 2 if torch.cuda.is_available() else 0


# ======================================================================
# SCAN RECORDS
# ======================================================================

def scan_records(data_dir: str) -> list:
    """
    Scan semua folder *_dwt_stat di bawah data_dir.

    Returns:
        records : list of dict, tiap dict berisi:
                    'feat_path'  : str   — path features.npy
                    'label_path' : str   — path labels.npy
                    'patient_id' : str   — misal 'chb01'
                    'folder'     : str   — path folder
    """
    records  = []
    base     = Path(data_dir)

    if not base.exists():
        raise FileNotFoundError(f"DATA_DIR tidak ditemukan: {base}")

    for patient_dir in sorted(base.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name

        for folder in sorted(patient_dir.iterdir()):
            if not folder.is_dir():
                continue
            feat  = folder / 'features.npy'
            label = folder / 'labels.npy'
            if feat.exists() and label.exists():
                records.append({
                    'feat_path'  : str(feat),
                    'label_path' : str(label),
                    'patient_id' : pid,
                    'folder'     : str(folder),
                })

    patients = sorted(set(r['patient_id'] for r in records))
    _print_scan_summary(records, patients)
    return records


def _print_scan_summary(records: list, patients: list) -> None:
    W = 72
    print(f"\nScan selesai: {len(records)} record dari {len(patients)} pasien")
    print(f"\n{'─'*W}")
    print(f"  {'Pasien':<10} {'Sesi':>5} {'Seg Kejang':>11} "
          f"{'Seg Non-Kejang':>15} {'Rasio':>7}")
    print(f"{'─'*W}")

    total_sz = total_ns = 0

    for pid in patients:
        recs = [r for r in records if r['patient_id'] == pid]
        n_sz = n_ns = 0
        for rec in recs:
            lb    = np.load(rec['label_path'])
            n_sz += int(np.sum(lb == 1))
            n_ns += int(np.sum(lb == 0))
        ratio = f"{n_ns/max(n_sz,1):.1f}:1"
        print(f"  {pid:<10} {len(recs):>5} {n_sz:>11} {n_ns:>15} "
              f"{ratio:>7}")
        total_sz += n_sz
        total_ns += n_ns

    r_total = f"{total_ns/max(total_sz,1):.1f}:1"
    print(f"{'─'*W}")
    print(f"  {'TOTAL':<10} {len(records):>5} {total_sz:>11} "
          f"{total_ns:>15} {r_total:>7}")
    print(f"{'─'*W}\n")


def group_by_patient(records: list) -> dict:
    """Kelompokkan records per patient_id."""
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec['patient_id']].append(rec)
    return dict(grouped)


def get_seizure_sessions(records: list) -> list:
    """
    Filter records yang mengandung ≥1 jendela kejang.

    Digunakan untuk LOSO (Leave-One-Session-Out):
      hanya sesi dengan seizure yang menjadi kandidat test fold.
    Non-seizure sessions tetap dipakai sebagai training data.

    Returns:
        List of record dicts, urutan sama dengan input (sorted by patient).
    """
    result = []
    for rec in records:
        labels = np.load(rec['label_path'])
        if np.any(labels == 1):
            result.append(rec)
    print(f"  Sesi dengan kejang : {len(result)} dari {len(records)} total sesi "
          f"→ {len(result)} fold LOSO")
    return result


# ======================================================================
# PENYEIMBANGAN DATA LATIH — per pasien (bukan per sesi)
# ======================================================================

def balance_train_samples(records: list, rng_seed: int = 42) -> list:
    """
    Penyeimbangan data latih dengan random downsampling per pasien.

    Algoritma:
      Untuk setiap pasien pada set latih:
        1. Kumpulkan semua segmen dari seluruh sesi pasien tersebut
        2. Pisahkan menjadi seiz_samples dan non_seiz_samples
        3. Jika n_non > n_seiz: random downsample non_seiz → n_seiz
        4. Jika n_non <= n_seiz: pakai semua non_seiz (tidak upsample)

    Hasil: data latih seimbang 1:1 (seizure vs non-seizure) per pasien.

    Args:
        records  : list of dict dari scan_records() — hanya data LATIH
        rng_seed : seed random untuk reprodusibilitas

    Returns:
        balanced : list of dict, tiap dict:
                     'features' : np.memmap (N, C, W, F)
                     'idx'      : int — indeks baris dalam features
                     'label'    : int — 0 atau 1
                     'pid'      : str — patient id
    """
    rng = np.random.default_rng(rng_seed)

    # Kelompokkan per pasien
    patient_records = defaultdict(list)
    for rec in records:
        patient_records[rec['patient_id']].append(rec)

    balanced = []

    print("\n  [Balancing] Random downsample non-seizure per pasien:")
    print(f"  {'Pasien':<10} {'Seiz (asli)':>12} {'Non-Seiz (asli)':>16} "
          f"{'Seiz (bal)':>11} {'Non-Seiz (bal)':>15}")
    print(f"  {'─'*68}")

    for pid, pat_recs in sorted(patient_records.items()):

        # Kumpulkan semua sampel pasien ini dari semua sesi
        seiz_samples = []
        non_samples  = []

        for rec in pat_recs:
            feat   = np.load(rec['feat_path'],  mmap_mode='r')
            labels = np.load(rec['label_path'])

            for i, lbl in enumerate(labels):
                entry = {
                    'features': feat,
                    'idx'     : i,
                    'label'   : int(lbl),
                    'pid'     : pid,
                }
                if lbl == 1:
                    seiz_samples.append(entry)
                else:
                    non_samples.append(entry)

        n_seiz_orig = len(seiz_samples)
        n_non_orig  = len(non_samples)

        if n_seiz_orig == 0:
            # Pasien tanpa kejang — tidak masukkan ke training
            # (seharusnya tidak terjadi jika data sudah difilter)
            print(f"  {pid:<10} SKIP — tidak ada segmen kejang")
            continue

        # Random downsample non-seizure → jumlah = n_seiz
        if n_non_orig > n_seiz_orig:
            chosen_idx = rng.choice(n_non_orig,
                                    size=n_seiz_orig,
                                    replace=False)
            non_chosen = [non_samples[i] for i in chosen_idx]
        else:
            # Non-seizure sudah lebih sedikit dari seizure — pakai semua
            non_chosen = non_samples

        balanced.extend(seiz_samples)
        balanced.extend(non_chosen)

        print(f"  {pid:<10} {n_seiz_orig:>12} {n_non_orig:>16} "
              f"{len(seiz_samples):>11} {len(non_chosen):>15}")

    n_total = len(balanced)
    n_seiz  = sum(1 for s in balanced if s['label'] == 1)
    n_non   = n_total - n_seiz
    print(f"  {'─'*68}")
    print(f"  Total setelah balancing: {n_total} sampel  "
          f"(seiz={n_seiz}  non={n_non}  "
          f"rasio={n_non/max(n_seiz,1):.2f}:1)\n")

    return balanced


def collect_raw_samples(records: list) -> list:
    """
    Kumpulkan semua sampel TANPA balancing.
    Digunakan untuk validasi dan test.

    Returns:
        samples : list of dict (format sama dengan balance_train_samples)
    """
    samples = []
    for rec in records:
        feat   = np.load(rec['feat_path'],  mmap_mode='r')
        labels = np.load(rec['label_path'])
        for i, lbl in enumerate(labels):
            samples.append({
                'features': feat,
                'idx'     : i,
                'label'   : int(lbl),
                'pid'     : rec['patient_id'],
            })

    n_total = len(samples)
    n_seiz  = sum(1 for s in samples if s['label'] == 1)
    print(f"    Dataset (tanpa balancing): {n_total} sampel  "
          f"(seiz={n_seiz}  non={n_total-n_seiz})")
    return samples


# ======================================================================
# DATASET
# ======================================================================

class EEGDataset(Dataset):
    """
    Dataset EEG untuk fitur statistik DWT.

    Input per sampel: features (C, W, F) = (18, 15, 21)

    Output __getitem__:
      x     : (C, W, F)  float32 tensor
      label : int (0 atau 1)
      pid   : str
    """

    def __init__(self, samples: list):
        """
        Args:
            samples : list of dict dari balance_train_samples()
                      atau collect_raw_samples()
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        i = s['idx']

        # Baca satu segmen dari memmap → copy ke RAM
        # features shape: (N, C, W, F) → ambil baris ke-i → (C, W, F)
        x = np.array(s['features'][i], dtype=np.float32)  # (C, W, F)

        # Cek NaN/Inf dan ganti dengan 0
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(x),
            torch.tensor(s['label'], dtype=torch.long),
            s['pid'],
        )


# ======================================================================
# COLLATE & LOADERS
# ======================================================================

def collate_fn(batch):
    """Custom collate untuk tuple (x, label, pid)."""
    xs, ys, pids = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(pids)


def make_train_loader(
    train_records : list,
    batch_size    : int,
    rng_seed      : int = 42,
) -> DataLoader:
    """
    DataLoader untuk data LATIH dengan balancing per pasien.

    Strategi: random DOWNSAMPLING non-seizure → rasio 1:1 per pasien.
    Data seizure yang sedikit dipertahankan semua; non-seizure dikurangi.

    Args:
        train_records : list record pasien training
        batch_size    : ukuran batch
        rng_seed      : seed untuk reprodusibilitas

    Returns:
        DataLoader (shuffle=True)
    """
    print("\n  [Train Loader] Menyeimbangkan data latih per pasien...")
    balanced = balance_train_samples(train_records, rng_seed=rng_seed)
    dataset  = EEGDataset(balanced)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers         = _N_WORKERS,
        pin_memory          = _USE_PIN,
        persistent_workers  = (_N_WORKERS > 0),
    )


def make_weighted_train_loader(
    train_records : list,
    batch_size    : int,
    rng_seed      : int = 42,
) -> DataLoader:
    """
    DataLoader untuk data LATIH dengan WeightedRandomSampler.

    Perbedaan dengan make_train_loader (downsampling):
      - Semua data dipertahankan (tidak ada data yang dibuang)
      - Tiap epoch, seizure di-oversample via probabilitas sampling
      - Bobot per sampel = n_total / (2 * n_kelas)
        → seizure mendapat bobot ~n_non/n_seiz kali lebih besar
      - replacement=True: satu seizure bisa muncul beberapa kali per epoch

    Contoh (dataset ini: seiz=227, non=108378):
      n_total      = 108605
      weight_seiz  = 108605 / (2 * 227)    = 239.22  (tinggi)
      weight_non   = 108605 / (2 * 108378) = 0.501   (rendah)
      total_w_seiz = 227    × 239.22 = 54,303  ← sama
      total_w_non  = 108378 × 0.501  = 54,303  ← sama → p(seiz) = 50%
      Expected seizure/batch (32) ≈ 16  (50%)

    Args:
        train_records : list record pasien training
        batch_size    : ukuran batch
        rng_seed      : seed untuk reprodusibilitas

    Returns:
        DataLoader dengan WeightedRandomSampler (tanpa shuffle)
    """
    from torch.utils.data import WeightedRandomSampler

    print("\n  [Train Loader] WeightedRandomSampler (semua data, oversample seizure)...")
    samples = collect_raw_samples(train_records)

    labels  = np.array([s['label'] for s in samples])
    n_total = len(labels)
    n_seiz  = int(np.sum(labels == 1))
    n_non   = int(np.sum(labels == 0))

    if n_seiz == 0:
        print("    PERINGATAN: tidak ada seizure — fallback ke shuffle biasa")
        dataset = EEGDataset(samples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=_N_WORKERS, pin_memory=_USE_PIN, persistent_workers=(_N_WORKERS > 0))

    # Bobot terbalik frekuensi: kelas langka → bobot besar
    weight_seiz = n_total / (2.0 * n_seiz)
    weight_non  = n_total / (2.0 * n_non)

    sample_weights = np.where(labels == 1, weight_seiz, weight_non)

    # ✅ DIPERBAIKI: p_seiz dihitung dari TOTAL bobot per kelas, bukan per-sampel
    total_w_seiz = n_seiz * weight_seiz    # = n_total / 2 (selalu)
    total_w_non  = n_non  * weight_non     # = n_total / 2 (selalu)
    total_w_all  = total_w_seiz + total_w_non
    p_seiz       = total_w_seiz / total_w_all   # selalu ≈ 50%
    exp_seiz_per_batch = batch_size * p_seiz

    # ✅ DIPERBAIKI: num_samples = n_seiz * 2 agar epoch tidak terlalu panjang
    # n_total=108605 → 3394 batch/epoch (terlalu lama)
    # n_seiz*2=454   →   14 batch/epoch (efisien, tetap balanced 50:50)
    num_samples = n_seiz * 2

    print(f"    Dataset (tanpa balancing): {n_total:,} sampel  "
          f"(seiz={n_seiz}  non={n_non:,})")
    print(f"    Seizure       : {n_seiz:,}  → weight = {weight_seiz:.2f}")
    print(f"    Non-seizure   : {n_non:,} → weight = {weight_non:.4f}")
    print(f"    Rasio weight  : {weight_seiz/weight_non:.1f}x")
    print(f"    p(seizure)    : {p_seiz*100:.1f}%  ← dijamin ~50% oleh formula")
    print(f"    Expected seizure/batch ({batch_size}) ≈ {exp_seiz_per_batch:.0f}  "
          f"({p_seiz*100:.0f}%)")
    print(f"    num_samples   : {num_samples:,}  ({num_samples//batch_size} batch/epoch)")

    generator = torch.Generator()
    generator.manual_seed(rng_seed)

    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = num_samples,   # ✅ n_seiz*2, bukan n_total
        replacement = True,
        generator   = generator,
    )

    dataset = EEGDataset(samples)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        collate_fn  = collate_fn,
        num_workers         = _N_WORKERS,
        pin_memory          = _USE_PIN,
        persistent_workers  = (_N_WORKERS > 0),
    )

# ======================================================================
# RANDOM OVERSAMPLING TRAIN LOADER
# ======================================================================

def make_ros_train_loader(
    train_records : list,
    batch_size    : int,
    rng_seed      : int = 42,
) -> DataLoader:
    """
    DataLoader untuk data LATIH dengan Random Oversampling (ROS).

    ROS menduplikasi sampel seizure ASLI secara acak sampai jumlahnya
    setara dengan non-seizure. Berbeda dari SMOTE yang membuat sampel BARU,
    ROS hanya mengulangi sampel yang sudah ada.

    Keunggulan vs SMOTE:
      ✓ Tidak perlu pip install apapun
      ✓ Tidak ada data yang dimuat ke RAM — semua tetap sebagai referensi memmap
      ✓ Setup sangat cepat (hanya duplikasi pointer, bukan data)
      ✓ Tidak ada risiko artefak sintetis

    Kelemahan vs SMOTE:
      ✗ Model melihat sampel seizure yang PERSIS SAMA berulang kali
        → risiko overfitting lebih tinggi dari SMOTE
      ✗ Tidak menambah variasi ke seizure (SMOTE membuat variasi baru)

    Perbedaan vs WeightedRandomSampler:
      WeightedRS : sampling dengan penggantian tiap step (non-deterministik per epoch)
      ROS        : dataset FIXED dengan duplikat eksplisit, tiap epoch identik
                   → lebih deterministik, lebih mudah direproduksi

    RAM usage:
      0 MB ekstra — semua referensi memmap (tidak ada data yang disalin ke RAM)

    Args:
        train_records : list record dari scan_records() — hanya data training
        batch_size    : ukuran batch
        rng_seed      : seed random untuk reprodusibilitas

    Returns:
        DataLoader (shuffle=True) dengan dataset seimbang 1:1
    """
    rng = np.random.default_rng(rng_seed)

    print("\n  [Train Loader] Random Oversampling (ROS) — "
          "duplikasi seizure ke memmap (tanpa load ke RAM)...")

    all_samples  = collect_raw_samples(train_records)
    seiz_samples = [s for s in all_samples if s['label'] == 1]
    non_samples  = [s for s in all_samples if s['label'] == 0]

    n_seiz = len(seiz_samples)
    n_non  = len(non_samples)

    print(f"    Distribusi asli  : seizure={n_seiz:,}  "
          f"non-seizure={n_non:,}  rasio={n_non/max(n_seiz,1):.1f}:1")

    if n_seiz == 0:
        print("    PERINGATAN: tidak ada seizure — fallback ke shuffle biasa")
        return DataLoader(EEGDataset(all_samples), batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=_N_WORKERS, pin_memory=_USE_PIN, persistent_workers=(_N_WORKERS > 0))

    if n_seiz >= n_non:
        # Seizure sudah lebih banyak atau sama — tidak perlu oversample
        print("    Seizure >= Non-seizure — tidak perlu ROS, pakai semua data")
        dataset = EEGDataset(all_samples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=_N_WORKERS, pin_memory=_USE_PIN, persistent_workers=(_N_WORKERS > 0))

    # Duplikasi seizure secara acak sampai n_seiz = n_non
    n_needed    = n_non - n_seiz
    dup_indices = rng.choice(n_seiz, size=n_needed, replace=True)
    duplicated  = [seiz_samples[i] for i in dup_indices]

    # Dataset seimbang: non-seizure + seizure asli + seizure duplikat
    # Semua entry adalah dict dengan referensi memmap — tidak ada data di RAM
    balanced = non_samples + seiz_samples + duplicated

    n_seiz_bal = n_seiz + n_needed
    print(f"    Sampel duplikat  : {n_needed:,} seizure "
          f"(dari {n_seiz} asli, dengan penggantian)")
    print(f"    Sesudah ROS      : seizure={n_seiz_bal:,}  "
          f"non-seizure={n_non:,}  rasio=1.00:1")
    print(f"    RAM extra        : 0 MB (hanya referensi memmap)")

    dataset = EEGDataset(balanced)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers         = _N_WORKERS,
        pin_memory          = _USE_PIN,
        persistent_workers  = (_N_WORKERS > 0),
    )


# ======================================================================
# DATASET CLASSES UNTUK OUTPUT SMOTE
# ======================================================================

class EEGArrayDataset(Dataset):
    """
    Dataset dari array numpy di RAM — digunakan setelah SMOTEENN / SMOTETomek.

    SMOTEENN dan SMOTETomek mengubah jumlah non-seizure (cleaning step
    membuang beberapa sampel). Karena struktur memmap tidak bisa disesuaikan
    secara dinamis, seluruh hasil resampling disimpan sebagai array numpy.

    Args:
        X     : (N, C*W*F) float32 — seluruh data setelah resampling
        y     : (N,) int64          — label
        shape : (C, W, F)           — shape per sampel untuk reshape
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, shape: tuple):
        self.X     = X.astype(np.float32)
        self.y     = y.astype(np.int64)
        self.shape = shape

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].reshape(self.shape)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.from_numpy(x.copy()),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
            'synthetic',
        )


class HybridEEGDataset(Dataset):
    """
    Dataset hybrid untuk vanilla SMOTE — memory-efficient.

    Menggabungkan dua sumber data:
      - Sampel asli (non-seizure + seizure asli) : referensi memmap (lazy, tidak di RAM)
      - Seizure sintetis SMOTE                  : numpy array di RAM

    Mengapa bisa memory-efficient untuk vanilla SMOTE:
      SMOTE hanya butuh seizure untuk generate sintetis.
      Non-seizure tidak diubah → bisa tetap di memmap.
      Hanya hasil sintetis yang perlu disimpan di RAM.

    RAM usage vs EEGArrayDataset:
      EEGArrayDataset  : (N_non + N_seiz + N_syn) × 5670 × 4B  ← semua di RAM
      HybridEEGDataset : N_syn × 5670 × 4B                      ← hanya sintetis
      Penghematan ~50%

    Args:
        memmap_samples : list of dict (format EEGDataset) — semua sampel asli
        synthetic_X    : (N_syn, C*W*F) float32 — seizure sintetis saja
        sample_shape   : (C, W, F)
    """

    def __init__(
        self,
        memmap_samples : list,
        synthetic_X    : np.ndarray,
        sample_shape   : tuple,
    ):
        self.memmap_samples = memmap_samples
        self.synthetic_X    = synthetic_X.astype(np.float32)
        self.sample_shape   = sample_shape
        self.n_memmap       = len(memmap_samples)
        self.n_synthetic    = len(synthetic_X)

    def __len__(self) -> int:
        return self.n_memmap + self.n_synthetic

    def __getitem__(self, idx: int):
        if idx < self.n_memmap:
            s = self.memmap_samples[idx]
            x = np.array(s['features'][s['idx']], dtype=np.float32)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return (
                torch.from_numpy(x),
                torch.tensor(s['label'], dtype=torch.long),
                s['pid'],
            )
        else:
            syn_idx = idx - self.n_memmap
            x = self.synthetic_X[syn_idx].reshape(self.sample_shape)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return (
                torch.from_numpy(x.copy()),
                torch.tensor(1, dtype=torch.long),
                'synthetic',
            )


# ======================================================================
# HELPER INTERNAL SMOTE
# ======================================================================

def _check_smote_import():
    import importlib.util
    if importlib.util.find_spec('imblearn') is None:
        raise ImportError(
            "Package 'imbalanced-learn' belum terpasang.\n"
            "Install dengan: pip install imbalanced-learn"
        )


def _fallback_weighted_sampler(all_samples, n_seiz, n_non, batch_size, rng_seed):
    """Fallback ke WeightedRandomSampler jika seizure terlalu sedikit untuk k-NN."""
    from torch.utils.data import WeightedRandomSampler
    n_total = n_seiz + n_non
    weight_seiz = n_total / (2.0 * n_seiz)
    weight_non  = n_total / (2.0 * n_non)
    weights = np.array(
        [weight_seiz if s['label'] == 1 else weight_non for s in all_samples],
        dtype=np.float64,
    )
    generator = torch.Generator()
    generator.manual_seed(rng_seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights), num_samples=n_total,
        replacement=True, generator=generator,
    )
    return DataLoader(EEGDataset(all_samples), batch_size=batch_size,
                      sampler=sampler, collate_fn=collate_fn,
                      num_workers=_N_WORKERS, pin_memory=_USE_PIN, persistent_workers=(_N_WORKERS > 0))


def _load_all_to_ram(samples: list) -> tuple:
    """Load seluruh samples (seizure + non-seizure) ke array numpy di RAM."""
    X_list, y_list = [], []
    for s in samples:
        x = np.array(s['features'][s['idx']], dtype=np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(x.flatten())
        y_list.append(s['label'])
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


# ======================================================================
# SMOTE TRAIN LOADER — pilih metode: 'smote', 'smoteenn', 'smotetomek'
# ======================================================================

def make_smote_train_loader(
    train_records : list,
    batch_size    : int,
    rng_seed      : int = 42,
    k_neighbors   : int = 5,
    smote_method  : str = 'smote',
) -> DataLoader:
    """
    DataLoader untuk data LATIH dengan oversampling berbasis SMOTE.

    smote_method pilihan:
    ┌─────────────┬────────────────────────────────────────────────┬──────────┐
    │ 'smote'     │ Vanilla SMOTE — interpolasi k-NN, tanpa       │ ~99 MB   │
    │             │ cleaning. Memory-efficient: non-seizure tetap  │ (sintetis│
    │             │ di memmap, hanya seizure sintetis di RAM.      │ saja)    │
    ├─────────────┼────────────────────────────────────────────────┼──────────┤
    │ 'smoteenn'  │ SMOTE + ENN cleaning — setelah SMOTE, buang   │ ~200 MB  │
    │             │ sampel yang misklasifikasi oleh 3-NN. Hasil   │ (semua   │
    │             │ lebih bersih tapi dataset bisa mengecil.       │ di RAM)  │
    │             │ Butuh non-seizure asli → semua dimuat ke RAM. │          │
    ├─────────────┼────────────────────────────────────────────────┼──────────┤
    │ 'smotetomek'│ SMOTE + Tomek Links cleaning — buang pasang   │ ~200 MB  │
    │             │ (seizure, non-seizure) yang saling paling     │ (semua   │
    │             │ dekat. Lebih ringan dari ENN, konservatif.    │ di RAM)  │
    │             │ Butuh non-seizure asli → semua dimuat ke RAM. │          │
    └─────────────┴────────────────────────────────────────────────┴──────────┘

    Batasan umum:
      - Memerlukan: pip install imbalanced-learn
      - SMOTE bekerja di feature space 5670 dim — valid matematis tapi sintetis
        tidak selalu mencerminkan pola EEG fisiologis yang realistis
      - Tidak cocok dikombinasi dengan WeightedRandomSampler

    Args:
        train_records : list record dari scan_records() — hanya data training
        batch_size    : ukuran batch
        rng_seed      : seed random untuk reprodusibilitas
        k_neighbors   : jumlah tetangga k-NN untuk interpolasi SMOTE
        smote_method  : 'smote' | 'smoteenn' | 'smotetomek'

    Returns:
        DataLoader (shuffle=True) dengan dataset seimbang
    """
    _check_smote_import()

    valid_methods = ('smote', 'smoteenn', 'smotetomek')
    if smote_method not in valid_methods:
        raise ValueError(
            f"smote_method '{smote_method}' tidak valid. "
            f"Pilih: {valid_methods}"
        )

    # ── Kumpulkan referensi sampel (belum load data ke RAM) ───────────
    all_samples = collect_raw_samples(train_records)
    if not all_samples:
        raise ValueError("Tidak ada sampel di train_records.")

    sample_shape = all_samples[0]['features'].shape[1:]   # (C, W, F)
    seiz_samples = [s for s in all_samples if s['label'] == 1]
    non_samples  = [s for s in all_samples if s['label'] == 0]
    n_seiz = len(seiz_samples)
    n_non  = len(non_samples)

    label_str = {'smote': 'SMOTE', 'smoteenn': 'SMOTE-ENN',
                 'smotetomek': 'SMOTE-Tomek'}[smote_method]
    print(f"\n  [Train Loader] {label_str} oversampling...")
    print(f"    Distribusi asli : seizure={n_seiz:,}  "
          f"non-seizure={n_non:,}  rasio={n_non/max(n_seiz,1):.1f}:1")

    # ── Edge case: tidak ada seizure ──────────────────────────────────
    if n_seiz == 0:
        print("    PERINGATAN: tidak ada seizure — fallback ke shuffle biasa")
        return DataLoader(EEGDataset(all_samples), batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=_N_WORKERS, pin_memory=_USE_PIN, persistent_workers=(_N_WORKERS > 0))

    # ── Edge case: seizure terlalu sedikit untuk k-NN ─────────────────
    k_safe = min(k_neighbors, n_seiz - 1)
    if k_safe < 1:
        print(f"    PERINGATAN: n_seiz={n_seiz} < 2 → tidak bisa k-NN, "
              f"fallback ke WeightedRandomSampler")
        return _fallback_weighted_sampler(
            all_samples, n_seiz, n_non, batch_size, rng_seed)

    if k_safe != k_neighbors:
        print(f"    k_neighbors: {k_neighbors} → {k_safe} "
              f"(disesuaikan karena n_seiz={n_seiz})")

    # ══════════════════════════════════════════════════════════════════
    # PATH A — Vanilla SMOTE (memory-efficient)
    # ══════════════════════════════════════════════════════════════════
    if smote_method == 'smote':
        from imblearn.over_sampling import SMOTE

        # Load HANYA seizure ke RAM
        print(f"    RAM dimuat   : {n_seiz} seizure saja "
              f"({n_seiz * int(np.prod(sample_shape)) * 4 / 1024:.1f} KB)")

        X_seiz = np.stack([
            np.nan_to_num(
                np.array(s['features'][s['idx']], dtype=np.float32).flatten(),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for s in seiz_samples
        ], axis=0)                                   # (n_seiz, D)

        # Trick: kirim 1 dummy non-seizure (zeros) ke SMOTE.
        # SMOTE tidak menggunakan majority class untuk generate sintetis
        # → dummy aman dipakai. sampling_strategy={1: n_non} atur target.
        dummy_non = np.zeros((1, X_seiz.shape[1]), dtype=np.float32)
        X_input   = np.vstack([X_seiz, dummy_non])  # (n_seiz+1, D)
        y_input   = np.array([1]*n_seiz + [0], dtype=np.int64)

        resampler = SMOTE(
            sampling_strategy = {1: n_non},
            k_neighbors       = k_safe,
            random_state      = rng_seed,
        )
        X_res, y_res = resampler.fit_resample(X_input, y_input)

        # Ambil hanya sampel sintetis (indeks n_seiz ke atas di kelas 1)
        X_all_seiz  = X_res[y_res == 1]
        X_synthetic = X_all_seiz[n_seiz:]
        n_syn       = len(X_synthetic)

        print(f"    Seizure sintetis dibuat : {n_syn:,}  "
              f"(RAM: {X_synthetic.nbytes/1024**2:.1f} MB)")
        print(f"    Non-seizure ({n_non:,}) tetap di memmap — tidak dimuat ke RAM")
        print(f"    Total dataset : {n_non + n_seiz + n_syn:,} sampel  "
              f"(rasio ≈ {n_non/max(n_seiz+n_syn,1):.2f}:1)")

        dataset = HybridEEGDataset(
            memmap_samples = all_samples,   # lazy memmap
            synthetic_X    = X_synthetic,   # RAM
            sample_shape   = sample_shape,
        )

    # ══════════════════════════════════════════════════════════════════
    # PATH B — SMOTE-ENN atau SMOTE-Tomek (semua data ke RAM)
    # ══════════════════════════════════════════════════════════════════
    else:
        # ENN dan Tomek Links butuh non-seizure asli untuk cleaning step
        # → tidak bisa pakai trick dummy → semua harus dimuat ke RAM
        ram_est_mb = (n_seiz + n_non) * int(np.prod(sample_shape)) * 4 / 1024**2
        print(f"    RAM dimuat   : semua {n_seiz+n_non:,} sampel "
              f"(estimasi ~{ram_est_mb:.0f} MB)")

        X_all, y_all = _load_all_to_ram(all_samples)

        if smote_method == 'smoteenn':
            from imblearn.combine import SMOTEENN
            resampler = SMOTEENN(
                sampling_strategy = {1: n_non},
                smote             = None,   # pakai default SMOTE internal
                enn               = None,   # pakai default ENN (n_neighbors=3)
                random_state      = rng_seed,
            )
        else:  # smotetomek
            from imblearn.combine import SMOTETomek
            resampler = SMOTETomek(
                sampling_strategy = {1: n_non},
                random_state      = rng_seed,
            )

        X_res, y_res = resampler.fit_resample(X_all, y_all)

        n_seiz_res = int(np.sum(y_res == 1))
        n_non_res  = int(np.sum(y_res == 0))
        n_removed  = (n_seiz + n_non) - len(y_res)
        n_syn      = n_seiz_res - n_seiz

        print(f"    Sesudah resampling:")
        print(f"      Seizure     : {n_seiz_res:,}  (+{n_syn} sintetis)")
        print(f"      Non-seizure : {n_non_res:,}  "
              f"({n_non-n_non_res} dihapus oleh cleaning)")
        print(f"      Total hapus : {n_removed:,} sampel (ENN/Tomek cleaning)")
        print(f"      Rasio baru  : {n_non_res/max(n_seiz_res,1):.2f}:1")

        dataset = EEGArrayDataset(X_res, y_res, shape=sample_shape)

    # ── Buat DataLoader ───────────────────────────────────────────────
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers         = _N_WORKERS,
        pin_memory          = _USE_PIN,
        persistent_workers  = (_N_WORKERS > 0),
    )


def make_eval_loader(
    records    : list,
    batch_size : int,
) -> DataLoader:
    """
    DataLoader untuk VALIDASI atau TEST (tanpa balancing).

    Args:
        records    : list record pasien
        batch_size : ukuran batch

    Returns:
        DataLoader (shuffle=False)
    """
    samples = collect_raw_samples(records)
    dataset = EEGDataset(samples)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers         = _N_WORKERS,
        pin_memory          = _USE_PIN,
        persistent_workers  = (_N_WORKERS > 0),
    )