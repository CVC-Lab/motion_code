# datasets_rowwise.py
import math
import random
import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



# ---------------------------
# Utility
# ---------------------------
def exists(x) -> bool:
    return x is not None


def _estimate_hatch_from_centers(img_meta: Dict, W: int) -> float:
    centers = img_meta.get("clustered_center", None)
    if centers is None or len(centers) == 0:
        return max(2.0, W / 8.0)
    xs = np.sort(np.array(centers)[:, 0])
    if xs.shape[0] < 2:
        return max(2.0, W / 8.0)
    return float(np.median(np.diff(xs)))


def decode_row_params(img_meta: Dict, row_idx: int, W: int) -> Tuple[float, float, float]:
    """
    Return (orientation_deg, track_width_px, hatch_spacing_px) for a given row.
    Falls back to simple heuristics if metadata is missing.
    """
    phases = img_meta.get("phases", [])
    n_phases = len(phases)
    # keep simple alternating schedule fallback; if phases provided, index safely
    arr = phases[int(row_idx) % max(1, n_phases)] if n_phases > 0 else np.array([])
    def from_arr(i, default):
        try:
            return float(arr[i])
        except Exception:
            return float(default)
    # orientation index 2, track width 1, hatch spacing 0 (matches provided metas)
    orientation_deg = from_arr(2, 0.0 if (row_idx % 2 == 0) else 90.0)
    hatch_spacing_px = from_arr(0, _estimate_hatch_from_centers(img_meta, W))
    track_width_px = from_arr(1, 2.0 * float(img_meta.get("radius", 8.0)))
    return orientation_deg, track_width_px, hatch_spacing_px


# ---------------------------
# MultiPhaseDataset (image + meta)
# ---------------------------
class MultiPhaseDataset(Dataset):
    """
    Loads grayscale images and a companion .txt meta file per image.
    Returns:
        gray_img_tensor: [1, H, W] in [-1, 1] (float32)
        meta_data: dict with keys:
            - "phases": list[np.ndarray]
            - "num_clustered_grains": int
            - "radius": float
            - "clustered_center": np.ndarray [M, 2]
            - "path": str
    """
    def __init__(
        self,
        folder: Union[str, Path],
        image_size: Optional[int] = None,   # kept for API compatibility
        exts: Sequence[str] = ('jpg', 'jpeg', 'png', 'tiff'),
        n_phases: int = 2,
        n_images: Optional[int] = None,
        parse_filename_meta: bool = True,        
    ):
        super().__init__()
        folder = Path(folder)
        paths: List[Path] = []
        for ext in exts:
            paths.extend(folder.glob(f'**/*.{ext}'))
        paths = sorted(paths)
        if n_images is not None:
            paths = paths[:n_images]
        self.paths = paths
        self.image_size = image_size
        self.transform = T.Compose([
            T.Grayscale(),
            T.ToTensor(),
        ])
        self.n_phases = n_phases
        self.parse_filename_meta = parse_filename_meta

    def _parse_image_fname_meta(self, pathlike: Union[str, Path]) -> Dict[str, Optional[float]]:
        """
        Parse patterns like:
        sample_6_867_IntensityPerturbationWeight_0.45.txt
        sample-6-867-IPW-0.45.png
        Returns dict with keys: param_id, image_no, grf_noise_scale (floats/ints or None).
        """
        name = Path(pathlike).name

        # sample_<param>_<image>
        m_ids = re.search(r"sample[_-](\d+)[_-](\d+)", name, flags=re.IGNORECASE)
        param_id = int(m_ids.group(1)) if m_ids else None
        image_no = int(m_ids.group(2)) if m_ids else None

        # IntensityPerturbationWeight or IPW, accept decimal or scientific notation
        num = r"([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)"
        m_w = re.search(r"Intensity(?:Perturbation)?Weight[_-]?"+num, name, flags=re.IGNORECASE)
        if not m_w:
            m_w = re.search(r"IPW[_-]?"+num, name, flags=re.IGNORECASE)
        grf_noise_scale = float(m_w.group(1)) if m_w else None

        return {
            "param_id": param_id,
            "image_no": image_no,
            "grf_noise_scale": grf_noise_scale,
        }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path)
        gray_img_tensor = self.transform(img)  # [1,H,W]
        min_val = gray_img_tensor.min()
        max_val = gray_img_tensor.max()
        gray_img_tensor = 2 * (gray_img_tensor - min_val) / (max_val - min_val + 1e-12) - 1

        meta_path = str(path)[: str(path).rfind('.')] + '.txt'
        with open(meta_path, "r") as f:
            lines = f.readlines()

        phase_meta: List[np.ndarray] = []
        for i in range(self.n_phases):
            phase_meta.append(np.array(list(map(float, lines[i].split()))))

        num_points = int(lines[self.n_phases])
        data_points = np.array([
            list(map(float, line.split()))
            for line in lines[(self.n_phases + 1):(self.n_phases + 1 + num_points)]
        ])

        meta_data = {
            "phases": phase_meta,
            "num_clustered_grains": num_points,
            "radius": 25.0,
            "clustered_center": data_points,
            "path": str(path),
            "meta_txt_path": meta_path,
        }

        # --- NEW: parse filename-derived meta (robust, optional) ---
        if self.parse_filename_meta:
            fname_meta = self._parse_image_fname_meta(meta_path)
            # also expose flat keys for convenience
            meta_data.update(fname_meta)
            meta_data["filename_meta"] = fname_meta

        return gray_img_tensor, meta_data


# ---------------------------
# RowDataset with windowed slicing
# ---------------------------

class RowDataset(Dataset):
    """
    Row-wise window sampler with per-window CLEAN/NOISY labels.

    Labeling rule:
      - From image-level meta: meta["clustered_center"] gives (cx,cy) points.
      - A window [s:e] on row r is labeled 'noisy' if ANY x in [s..e-1] lies within
        distance <= radius_px (default: 25.0 or meta['radius']) of ANY (cx,cy).
      - Otherwise the window is 'clean'.

    Views returned (no artificial noise):
      - 'raw'     : original row slice
      - 'aligned' : row shifted (rolled) by per-row lag derived from phase_meta[0]
                    (interpreted as [pitch_height, pitch_width, orientation_deg, lag0_px]).

    Output:
      returns (x, meta) where:
        x:        [C, win] tensor == primary_view ('raw' or 'aligned')
        meta:     dict with keys:
                    'label'            -> 'clean' or 'noisy'
                    'is_noisy_window'  -> bool
                    'noisy_frac'       -> fraction of pixels in window flagged noisy
                    'noisy_mask'       -> BoolTensor [win]
                    'views'            -> {'raw':..., 'aligned':...}  (if return_all_views=True)
                    + usual fields (image_idx,row_idx,start,end,window_size, QoIs, etc.)
    """
    def __init__(
        self,
        image_dataset: Dataset,
        rows_per_image: Union[int, str] = "all",
        window_size: Optional[Union[int, Sequence[int]]] = None,
        stride: Optional[int] = None,
        restrict_to_images: Optional[Sequence[int]] = None,
        sample_across_images: bool = True,
        precompute: bool = True,
        seed: Optional[int] = None,

        # NEW/kept knobs
        primary_view: str = "aligned",          # what to return as x ('raw' or 'aligned')
        duty_cycle: float = 0.5,                # used only for alignment visualization (not for labels)
        radius_px: Optional[float] = None,      # override; else meta['radius'] or 25.0
        lag_rule: Union[str, Callable[[Dict, int, int], float]] = "periodic",
        return_all_views: bool = True,
    ):
        super().__init__()
        self.image_dataset = image_dataset
        self.rows_per_image = rows_per_image
        self.window_size = window_size
        self.stride = stride
        self.sample_across_images = sample_across_images
        self.precompute = precompute
        self.rng = random.Random(seed) if seed is not None else random

        assert primary_view in {"raw", "aligned"}
        self.primary_view = primary_view
        self.return_all_views = return_all_views

        self.duty_cycle = float(np.clip(duty_cycle, 1e-6, 1.0))
        self.radius_px_override = radius_px
        self.lag_rule = lag_rule  # 'periodic' | 'constant' | callable(meta,row_idx,W)->float

        # choose images
        all_ids = list(range(len(image_dataset)))
        self.image_ids = list(restrict_to_images) if restrict_to_images is not None else all_ids
        self.image_ids = [i for i in self.image_ids if 0 <= i < len(image_dataset)]

        # cache sizes
        self._sizes: Dict[int, Tuple[int, int, int]] = {}
        def get_HW(i: int) -> Tuple[int, int, int]:
            if i in self._sizes:
                return self._sizes[i]
            img, _ = self.image_dataset[i]
            C, H, W = img.shape
            self._sizes[i] = (C, H, W)
            return C, H, W
        self._get_HW = get_HW

        # build index maps
        self._index: List[Tuple[int, int, int, int, int]] = []  # (img_idx,row_idx,s,e,win)
        self._row_map: List[Tuple[int, int]] = []

        if self.precompute:
            sizes = [window_size] if isinstance(window_size, int) or window_size is None else list(window_size)
            for img_idx in tqdm.tqdm(self.image_ids):
                C, H, W = get_HW(img_idx)
                if self.rows_per_image == "all":
                    row_indices = list(range(H))
                else:
                    rp = int(self.rows_per_image)
                    if rp >= H:
                        row_indices = list(range(H))
                    else:
                        step = max(1, math.floor(H / rp))
                        row_indices = list(range(0, H, step))[:rp]

                for r in row_indices:
                    if sizes == [None] or sizes == [W] or (window_size is None):
                        self._index.append((img_idx, r, 0, W, W))
                    else:
                        for win in sizes:
                            win = int(win)
                            use_stride = int(self.stride) if self.stride is not None else win
                            if win >= W:
                                self._index.append((img_idx, r, 0, W, W))
                                continue
                            s = 0
                            while s < W:
                                e = min(W, s + win)
                                self._index.append((img_idx, r, s, e, win))
                                if e == W:
                                    break
                                s += use_stride
        else:
            for img_idx in self.image_ids:
                C, H, W = get_HW(img_idx)
                if self.rows_per_image == "all":
                    row_indices = list(range(H))
                else:
                    rp = int(self.rows_per_image)
                    if rp >= H:
                        row_indices = list(range(H))
                    else:
                        step = max(1, math.floor(H / rp))
                        row_indices = list(range(0, H, step))[:rp]
                for r in row_indices:
                    self._row_map.append((img_idx, r))

        def _choose_window(W: int) -> Tuple[int, int, int]:
            if self.window_size is None:
                return W, 0, W
            if isinstance(self.window_size, Sequence):
                win = int(self.rng.choice(list(self.window_size)))
            else:
                win = int(self.window_size)
            win = max(1, min(win, W))
            s = 0 if (win == W) else self.rng.randint(0, W - win)
            return win, s, s + win
        self._choose_window = _choose_window

    # ------ helpers ------
    @staticmethod
    def _parse_phase0(meta: Dict) -> Tuple[float, float, float, float]:
        # phase_meta[0] := [pitch_height, pitch_width, orientation_deg, lag0_px]
        phases = meta.get("phases", [])
        arr = np.array(phases[0]).astype(float) if len(phases) > 0 else np.zeros(4, dtype=float)
        pitch_h = float(arr[0]) if arr.size > 0 else 1.0
        pitch_w = float(arr[1]) if arr.size > 1 else 16.0
        ori_deg = float(arr[2]) if arr.size > 2 else 0.0
        lag0   = float(arr[3]) if arr.size > 3 else 0.0
        return pitch_h, pitch_w, ori_deg, lag0

    def _compute_lag_px(self, meta: Dict, row_idx: int, W: int) -> float:
        """
        Compute horizontal lag (in pixels) for this row.

        Supported self.lag_rule:
        - 'constant'                : return lag0
        - 'periodic'                : linear advance across each vertical pitch (no flip)
        - 'periodic_flip'           : SAME as 'periodic' but flips direction every other layer
        - 'periodic_from_orientation': sign determined by orientation parity (≈ every other layer)

        Notes:
        layer_idx := floor(row_idx / round(pitch_height))
        frac      := (row_idx % round(pitch_height)) / pitch_height ∈ [0,1)
        progress  := frac * pitch_width  (how much to advance within current layer)
        """
        # callable override
        if callable(self.lag_rule):
            return float(self.lag_rule(meta, row_idx, W))

        pitch_h, pitch_w, ori_deg, lag0 = self._parse_phase0(meta)
        ph = max(1.0, float(pitch_h))
        pw = float(pitch_w)
        lag0 = float(lag0) % W

        if self.lag_rule == "constant":
            return lag0

        # common terms
        ph_int = max(1, int(round(ph)))
        frac = (row_idx % ph_int) / ph
        progress = frac * pw
        
        if self.lag_rule == "periodic":
            lag = lag0 + progress

        elif self.lag_rule == "periodic_flip":
            # flip direction every other *layer* (0-based indexing)
            layer_idx = row_idx // ph_int
            direction = 1.0 if (layer_idx % 2 == 0) else -1.0
            lag = lag0 + direction * progress

        else:
            # fallback to old periodic
            lag = lag0 + progress

        return float(lag % W)

    @staticmethod
    def _noisy_mask_for_row(W: int, row_idx: int, centers_xy: np.ndarray, radius_px: float) -> np.ndarray:
        if centers_xy is None or len(centers_xy) == 0 or radius_px <= 0:
            return np.zeros(W, dtype=bool)
        mask = np.zeros(W, dtype=bool)
        r2 = float(radius_px) ** 2
        xs = np.arange(W, dtype=float)
        for cy, cx in centers_xy:
            dy = float(row_idx) - float(cy)
            if abs(dy) > radius_px:
                continue
            half = math.sqrt(max(0.0, r2 - dy * dy))
            x_lo = int(max(0, math.floor(cx - half)))
            x_hi = int(min(W - 1, math.ceil(cx + half)))
            if x_hi >= x_lo:
                mask[x_lo:(x_hi + 1)] = True
        return mask

    # ------ dataset API ------
    def __len__(self) -> int:
        return len(self._index) if self.precompute else len(self._row_map)

    def __getitem__(self, idx: int):
        # choose a window
        if self.precompute:
            img_idx, row_idx, s, e, win = self._index[idx]
        else:
            img_idx, row_idx = self._row_map[idx]
            _, H, W = self._get_HW(img_idx)
            win, s, e = self._choose_window(W)

        # fetch image + meta
        img, img_meta = self.image_dataset[img_idx]    # img in [-1,1], [C,H,W]
        C, H, W = img.shape
        row_full = img[:, row_idx, :]                  # [C, W]

        # alignment
        lag_px = self._compute_lag_px(img_meta, row_idx, W)
        lag_shift = int(round(lag_px)) % W
        #row_full_aligned = torch.roll(row_full, shifts=lag_shift, dims=-1)

        # slice window
        raw_win     = row_full[:, s:e]                 # [C, win]
        win_shift = (-lag_shift) % max(1, win)
        aligned_win = torch.roll(raw_win, shifts=win_shift, dims=-1)
        #aligned_win = row_full_aligned[:, s:e]         # [C, win]

        # label by clustered centers
        centers = img_meta.get("clustered_center", None)
        centers = np.asarray(centers, dtype=float) if centers is not None else None
        radius = float(self.radius_px_override) if self.radius_px_override is not None else float(img_meta.get("radius", 25.0))
        mask_full = self._noisy_mask_for_row(W, row_idx, centers, radius)  # [W] bool
        mask_win_np = mask_full[s:e]                                       # [win] bool
        mask_win = torch.from_numpy(mask_win_np).to(dtype=torch.bool)

        noisy_pixels = int(mask_win_np.sum())
        win_len = int(e - s)
        noisy_frac = float(noisy_pixels / max(1, win_len))
        is_noisy = noisy_pixels > 0
        label = "noisy" if is_noisy else "clean"

        # QoIs from phase0
        pitch_h, pitch_w, ori_deg, lag0 = self._parse_phase0(img_meta)

        # meta pack
        meta = {
            "image_idx": int(img_idx),
            "row_idx": int(row_idx),
            "start": int(s),
            "end": int(e),
            "window_size": int(win_len),
            "W_full": int(W),
            "H_full": int(H),
            "path": img_meta.get("path", None),

            # QoIs
            "pitch_height": float(pitch_h),
            "pitch_width": float(pitch_w),
            "orientation_deg": float(ori_deg),
            "lag0_px": float(lag0),
            "lag_px_used": float(lag_px),

            # labels
            "label": label,
            "is_noisy_window": bool(is_noisy),
            "noisy_frac": noisy_frac,
            "noisy_mask": mask_win,    # [win] bool
        }

        if self.return_all_views:
            meta["views"] = {"raw": raw_win, "aligned": aligned_win}

        x = aligned_win if self.primary_view == "aligned" else raw_win
        return x, meta

    def __repr__(self) -> str:
        mode = "precompute" if self.precompute else "online"
        scope = f"{len(self.image_ids)} imgs"
        if not self.precompute:
            return f"RowDataset[{mode}, {scope}, primary={self.primary_view}]"
        if len(self._index) == 0:
            return f"RowDataset[{mode}, {scope}, len=0]"
        wins = sorted({it[4] for it in self._index})
        return f"RowDataset[{mode}, {scope}, windows={wins}, primary={self.primary_view}, len={len(self)}]"

# ---------------------------
# Collate functions
# ---------------------------
def row_collate_fn(batch: List[Tuple[torch.Tensor, Dict]]):
    """
    No padding. All windows in the batch must share the same width.
    Input items are [C, W_i]; output is [B, W, C].
    """
    rows, metas = zip(*batch)
    rows = torch.stack(rows, dim=0)    # [B, C, W]
    rows = rows.permute(0, 2, 1).contiguous()  # [B, W, C]
    return rows, list(metas)


def row_collate_pad(batch: List[Tuple[torch.Tensor, Dict]], pad_value: float = 0.0):
    """
    Pads variable-width windows to max width in the batch.
    Returns:
        rows_padded: [B, W_max, C]
        attn_mask  : [B, W_max] where 1 marks valid samples
        metas      : list of dicts
    """
    rows, metas = zip(*batch)
    B = len(rows)
    C = rows[0].shape[0]
    widths = [r.shape[1] for r in rows]
    Wmax = max(widths)
    rows_padded = rows[0].new_full((B, C, Wmax), fill_value=pad_value)
    attn = torch.zeros(B, Wmax, dtype=torch.bool)
    for i, r in enumerate(rows):
        w = r.shape[1]
        rows_padded[i, :, :w] = r
        attn[i, :w] = True
    rows_padded = rows_padded.permute(0, 2, 1).contiguous()  # [B, Wmax, C]
    return rows_padded, attn, list(metas)

# --- ADD THIS HELPER ---
@torch.no_grad()
def plot_image_and_sampled_signals(
    row_ds: RowDataset,
    image_idx: int,
    n_samples: Optional[int] = 24,
    pick_every_k: Optional[int] = None,   # stride selection within this image
    aligned: bool = True,                 # if False, bottom row will mirror raw plots
    rng_seed: Optional[int] = 123,
    savepath: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 9),
    lw: float = 1.2,
    alpha: float = 0.85,
):
    """
    Layout:
      ┌───────────┬───────────────────────────────┬───────────────────────────────┐
      │  IMAGE    │ Raw – Clean (green)           │ Raw – Noisy (red)            │
      ├───────────┼───────────────────────────────┼───────────────────────────────┤
      │           │ Aligned – Clean (green)       │ Aligned – Noisy (red)        │
      └───────────┴───────────────────────────────┴───────────────────────────────┘
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    rng = random.Random(rng_seed) if rng_seed is not None else random
    k = int(pick_every_k) if (pick_every_k is not None and pick_every_k > 1) else 1
    items: List[Tuple[torch.Tensor, Dict]] = []

    # ---------- collect items (true stride for precompute; best-effort for online) ----------
    if row_ds.precompute:
        idxs = [i for i, (img_i, *_rest) in enumerate(row_ds._index) if img_i == image_idx]
        if len(idxs) == 0:
            print(f"[plot] No precomputed windows for image_idx={image_idx}")
            return
        idxs = idxs[::k]
        if n_samples is not None:
            idxs = idxs[: n_samples]
        for i in idxs:
            x, meta = row_ds.__getitem__(i)
            if "views" not in meta:
                meta["views"] = {"raw": x, "aligned": x}
            items.append((x, meta))
    else:
        want = n_samples if n_samples is not None else 50
        N = len(row_ds)
        seen_for_img = 0
        tried = 0
        max_tries = max(5000, 10 * (want or 1))
        while len(items) < want and tried < max_tries:
            j = rng.randrange(N)
            x, meta = row_ds[j]
            if meta.get("image_idx", -1) == image_idx:
                seen_for_img += 1
                if (seen_for_img - 1) % k == 0:
                    if "views" not in meta:
                        meta["views"] = {"raw": x, "aligned": x}
                    items.append((x, meta))
            tried += 1
        if len(items) == 0:
            print(f"[plot] No samples found for image_idx={image_idx} (online mode).")
            return

    # ---------- partition by label ----------
    clean_items = [(x, m) for (x, m) in items if not m["is_noisy_window"]]
    noisy_items = [(x, m) for (x, m) in items if m["is_noisy_window"]]

    # ---------- fetch image ----------
    img, _img_meta = row_ds.image_dataset[image_idx]
    C, H, W = img.shape
    img_show = img[0].cpu().numpy() if C >= 1 else img.squeeze(0).cpu().numpy()

    # ---------- figure with 1 (left) + 2×2 (right) ----------
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0])

    ax_img       = fig.add_subplot(gs[:, 0])  # spans both rows
    ax_raw_clean = fig.add_subplot(gs[0, 1])
    ax_raw_noisy = fig.add_subplot(gs[0, 2])
    ax_al_clean  = fig.add_subplot(gs[1, 1])
    ax_al_noisy  = fig.add_subplot(gs[1, 2])

    # ---- left image with overlays ----
    ax_img.imshow((img_show + 1) * 0.5, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title(f"Image {image_idx}: windows (green=clean, red=noisy)")
    ax_img.set_axis_off()

    def _plot_stack(ax, series: List[np.ndarray], color: str, title: str):
        if len(series) == 0:
            ax.set_title(f"{title}\n( none )")
            ax.set_xlabel("x (window)")
            ax.set_ylabel("intensity")
            ax.set_ylim(-1.05, 1.05)
            return
        for y in series:
            ax.plot(np.arange(len(y)), y, lw=lw, alpha=alpha, color=color)
        ax.set_title(f"{title}  (n={len(series)})")
        ax.set_xlabel("x (window)")
        ax.set_ylabel("intensity")
        ax.set_ylim(-1.05, 1.05)

    # ---- overlay rectangles on image ----
    for _, meta in clean_items:
        s, e, r, w = meta["start"], meta["end"], meta["row_idx"], meta["window_size"]
        ax_img.add_patch(Rectangle((s, r - 0.4), w, 0.8, linewidth=1.0, edgecolor="#2ecc71", facecolor="none", alpha=0.95))
    for _, meta in noisy_items:
        s, e, r, w = meta["start"], meta["end"], meta["row_idx"], meta["window_size"]
        ax_img.add_patch(Rectangle((s, r - 0.4), w, 0.8, linewidth=1.0, edgecolor="#e74c3c", facecolor="none", alpha=0.95))

    # ---- gather 1D series per panel ----
    def _extract(view_key: str, pairs: List[Tuple[torch.Tensor, Dict]]) -> List[np.ndarray]:
        out = []
        for x, meta in pairs:
            v = meta.get("views", {}).get(view_key, x)
            y = v[0].detach().cpu().numpy()
            out.append(y)
        return out

    # Raw panels
    raw_clean_series = _extract("raw", clean_items)
    raw_noisy_series = _extract("raw", noisy_items)

    # Aligned panels (or mirror raw if aligned=False)
    if aligned:
        al_clean_series = _extract("aligned", clean_items)
        al_noisy_series = _extract("aligned", noisy_items)
    else:
        al_clean_series = raw_clean_series
        al_noisy_series = raw_noisy_series

    # ---- draw right 2×2 ----
    _plot_stack(ax_raw_clean, raw_clean_series, "#2ecc71", "Raw – Clean")
    _plot_stack(ax_raw_noisy, raw_noisy_series, "#e74c3c", "Raw – Noisy")
    _plot_stack(ax_al_clean,  al_clean_series,  "#2ecc71", "Aligned – Clean")
    _plot_stack(ax_al_noisy,  al_noisy_series,  "#e74c3c", "Aligned – Noisy")

    # ---- save or show ----
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=160)
        plt.close(fig)
    else:
        plt.show()



if __name__=="__main__":
    # Build image dataset
    #data_path = '/mnt/data/yiwang/code/RandomMaterial/clustered_grain/massive_05_testify_bilevel_sampling_2phases_with_clustered_grains/train'
    data_path = '/mnt/data/yiwang/code/RandomMaterial/clustered_grain/debug_2phases_no_cluster_GRF_noise_0/train'
    img_ds = MultiPhaseDataset(folder=data_path, n_phases=2, n_images=100, parse_filename_meta=True)
    x, meta = img_ds[0]
    print(meta["param_id"], meta["image_no"], meta["grf_noise_scale"])

    # Windows, labeled clean/noisy by clustered_center coverage
    row_ds = RowDataset(
        image_dataset=img_ds,
        rows_per_image="all",
        window_size=128,     # or [128,256] etc.
        stride=64,
        precompute=True,
        primary_view="aligned",   # your trainer gets aligned slices by default
        lag_rule = "periodic_flip",
        radius_px=None,           # use meta['radius'] or 25.0
    )

    #loader = torch.utils.data.DataLoader(row_ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=row_collate_fn)

    # Visualization for a specific image
    plot_image_and_sampled_signals(row_ds, image_idx=0, n_samples=100, pick_every_k=127, aligned=True, savepath="debug/row_windows_img0.png")