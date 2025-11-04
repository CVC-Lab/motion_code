# --- plotting helpers (no QoIs) ---
import math
from typing import Dict, List, Optional, Tuple
from model.signal_models import SignalEncoder1D, SignalDecoder1D
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_lagged_posterior_samples_window(
    mean_win: torch.Tensor,          # (1, win, C) or (1, win, 1)
    cov_win: torch.Tensor,           # (1, win, win) [ch0]
    y_aligned_win: Optional[torch.Tensor],  # (win,) or (win,1) or (1, win) or (1, win, 1)
    out_png: str,
    title: str,
    n_samples: int = 8,
    lw: float = 1.5,
):
    """
    Plot a single window: mean ±2σ and posterior samples vs ALIGNED observation for channel 0.
    No x-warping here: we already applied the roll to both GP and obs.
    """
    # select ch0
    m = mean_win[..., 0:1] if (mean_win.dim() == 3 and mean_win.shape[-1] > 1) else mean_win  # (1,win,1)
    C = cov_win
    N = m.shape[1]

    # numerical stabilizer for Cholesky
    jitter = 1e-9
    C = C.clone()
    C[:, torch.arange(N), torch.arange(N)] += jitter

    # diag std
    sd = torch.sqrt(torch.clamp(torch.diagonal(C, dim1=-2, dim2=-1), min=0.0)).unsqueeze(-1)  # (1,win,1)

    # cholesky
    try:
        L = torch.linalg.cholesky(C.squeeze(0))   # (win,win)
    except Exception:
        # fallback eigen
        evals, evecs = torch.linalg.eigh(C.squeeze(0))
        evals = torch.clamp(evals, min=1e-9)
        L = evecs @ torch.diag(torch.sqrt(evals))

    # samples
    xx = np.arange(N)
    mu = m.squeeze(0).squeeze(-1).cpu().numpy()
    sd_np = sd.squeeze(0).squeeze(-1).cpu().numpy()

    plt.figure(figsize=(8,3))
    plt.plot(xx, mu, linewidth=lw, label="GP mean")
    plt.fill_between(xx, mu - 2*sd_np, mu + 2*sd_np, alpha=0.25, label="±2σ")

    for _ in range(n_samples):
        eps = torch.randn(N, 1, dtype=torch.float64, device=L.device)
        samp = (m.squeeze(0) + (L @ eps)).squeeze(-1).cpu().numpy()
        plt.plot(xx, samp, alpha=0.6, linewidth=1)

    if y_aligned_win is not None:
        yo = y_aligned_win
        if yo.dim() == 3:
            yo = yo.squeeze(0)
        if yo.dim() == 2 and yo.shape[-1] > 1:
            yo = yo[:, 0:1]
        yo = yo.squeeze(-1).cpu().numpy()
        plt.scatter(xx, yo, s=10, alpha=0.8, label="aligned obs")

    plt.xlabel("window coordinate")
    plt.ylabel("signal (ch0)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

@torch.no_grad()
def plot_lagged_sampling_from_qois(
    trainer,
    row_ds,
    rows_to_plot: List[int],
    out_prefix: str,
    image_idx: Optional[int] = None,
    n_samples: int = 8,
):
    """
    Unified plotter:
      • If trainer has .decoder -> latent GP + decoder path (MC in latent).
      • Else -> legacy signal-space GP path.

    Saves one PNG per (row, window). Overlays predicted mean/±2σ with ALIGNED obs (ch0).
    """
    device = trainer.device
    has_decoder = hasattr(trainer, "decoder")

    # choose image
    image_ids = getattr(row_ds, "image_ids", list(range(len(row_ds.image_dataset))))
    img_id = image_ids[0] if image_idx is None else image_idx
    img_tensor, img_meta = row_ds.image_dataset[img_id]  # [C,H,W], dict
    C_img, H, W_full = img_tensor.shape

    # window stepping consistent with training
    win = int(trainer.window_size)
    step = max(1, win - int(trainer.window_overlap))

    for r in rows_to_plot:
        if r < 0 or r >= H: 
            continue

        # per-row lag shift from dataset rule
        lag_px = float(row_ds._compute_lag_px(img_meta, r, W_full))
        lag_shift = int(round(lag_px)) % W_full

        row_full = img_tensor[:, r, :]  # [C, W_full]

        s = 0; k = 0
        while s < W_full:
            e = min(W_full, s + win); wlen = e - s
            if wlen < 2: break

            raw_win = row_full[:, s:e]             # [C, wlen]
            win_shift = (-lag_shift) % max(1, wlen)
            aligned_win = torch.roll(raw_win, shifts=win_shift, dims=-1)  # [C,wlen]
            y_aligned_ch0 = aligned_win[0:1, :].T  # (wlen,1)

            # title/path
            out_png = f"{out_prefix}_img{img_id:03d}_row{r:04d}_win{k:03d}_{s:05d}-{e:05d}.png"
            title = f"img={img_id}, row={r}, win={k} [{s}:{e}), lag_px={lag_px:.2f}, shift={win_shift}"

            if has_decoder:
                # latent query at spatial x (row, window center, layer)
                meta = dict(img_meta)
                meta.update({"row_index": r, "H": H, "W": W_full})
                xq = trainer._spatial_x(meta, s, e, W_full)       # (1,1,d)
                mu_lat, var_lat = trainer.latent_posterior(xq)    # (1,1,D), (1,1,D)

                mu = mu_lat.squeeze(0).squeeze(0)                 # (D,)
                std = torch.sqrt(torch.clamp(var_lat.squeeze(0).squeeze(0), min=0))  # (D,)

                # Monte Carlo in latent -> decode to signal
                samples = []
                for _ in range(n_samples):
                    z = torch.normal(mean=mu, std=std).unsqueeze(0)          # (1,D)
                    y_hat = trainer.decode_latent(z, wlen=wlen)              # (1,wlen,C)
                    # roll the prediction the same way as obs (decoder trained on aligned windows already;
                    # rolling by win_shift keeps strict overlay equivalence)
                    y_hat = torch.roll(y_hat, shifts=win_shift, dims=1)
                    samples.append(y_hat[..., 0:1].squeeze(0))               # (wlen,1)

                Ys = torch.stack(samples, dim=0)                              # (S, wlen, 1)
                mean_sig = Ys.mean(dim=0).squeeze(-1).cpu().numpy()          # (wlen,)
                std_sig  = Ys.std(dim=0, unbiased=False).squeeze(-1).cpu().numpy()

                # plot
                xx = np.arange(wlen)
                plt.figure(figsize=(8,3))
                plt.plot(xx, mean_sig, label="recon mean", linewidth=1.6)
                plt.fill_between(xx, mean_sig - 2*std_sig, mean_sig + 2*std_sig, alpha=0.25, label="±2σ")
                yo = y_aligned_ch0.squeeze(-1).cpu().numpy()
                plt.scatter(xx, yo, s=10, alpha=0.8, label="aligned obs")
                plt.xlabel("window coordinate"); plt.ylabel("signal (ch0)")
                plt.title(title); plt.legend(); plt.tight_layout()
                plt.savefig(out_png, dpi=160); plt.close()

            else:
                # ---- legacy signal-space GP fallback (from earlier reply) ----
                # Use mean_full/cov_full path with rolling P C P^T and plot
                if k == 0:   # cache full prediction once per row set
                    mean_full, cov_full = trainer.predict_row(W=W_full)
                    mean_full = mean_full.to(torch.float64)
                m_win = mean_full[:, s:e, :]
                C_win = cov_full[:, s:e, s:e].clone()
                m_win = torch.roll(m_win, shifts=win_shift, dims=1)
                idx = torch.arange(wlen, device=C_win.device); idx = (idx - win_shift) % wlen
                C_win = C_win[:, idx][:, :, idx]
                plot_lagged_posterior_samples_window(
                    mean_win=m_win, cov_win=C_win, y_aligned_win=y_aligned_ch0,
                    out_png=out_png, title=title, n_samples=n_samples
                )

            k += 1
            if e == W_full: break
            s += step


# Purpose: Minimal row-wise trainer that uses lag-aligned signals from AM_rowwise_dataset
#          and a Gaussian row model provided elsewhere. No QoI code, no validation.

# ---------------------------------------------------------------------
# Expected model interface (duck-typed)
# ---------------------------------------------------------------------
# model.compute_loss(x_list, y_list) -> torch.Tensor (VFE/ELBO; higher is better)
# model.update_online(x_list, y_list) -> None (optional; safe to call)
# model(x) -> (means: List[torch.Tensor], covs: List[torch.Tensor], extras: Dict)
#   where x is shaped (1, N, 1); each mean is (1, N, 1); each cov is (1, N, N)




class GaussianRowTrainer(nn.Module):
    """
    Latent-embedding GP trainer.
    Each (image, row, window) becomes one datapoint at spatial input x_spatial,
    with target the encoder embedding z = Enc(y_aligned_window).

    Assumes your Gaussian model is a multi-output GP over ℝ^d -> ℝ^D with:
      - compute_loss(x_list, y_list)  (ELBO/VFE; maximize)
      - __call__(x) -> (means: List[(1,N,1)], covs: List[(1,N,N)], extras)
    """
    def __init__(
        self,
        model: nn.Module,        # GP over spatial coords -> latent ℝ^D
        W: int,
        C: int,
        *,
        window_size: int = 256,
        window_overlap: int = 128,
        device: str = "cuda",
        embedding_dim: int = 8,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        use_window_center: bool = True,
        use_row_index: bool = True,
        use_layer_index: bool = True,
        reg_align: float = 1e-2,     # GP mean vs encoder embedding
        reg_recon: float = 1e-2,     # decoder(ŷ) vs y reconstruction
    ):
        super().__init__()
        self.model = model
        self.W, self.C = int(W), int(C)
        self.window_size = int(max(8, window_size))
        self.window_overlap = int(max(0, min(window_overlap, self.window_size - 1)))
        self.device = torch.device(device)

        self.D = int(embedding_dim)
        self.encoder = encoder if encoder is not None else SignalEncoder1D(C_in=C, D=self.D)
        self.decoder = decoder if decoder is not None else SignalDecoder1D(D=self.D, C_out=C, window_size=self.window_size)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.use_row_index = bool(use_row_index)
        self.use_window_center = bool(use_window_center)
        self.use_layer_index = bool(use_layer_index)

        params = list(self.model.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=1e-2)
        self.to(self.device)


        # weights & schedules
        self.w_gp_max    = 0.3     # target weight for -ELBO
        self.w_rec       = float(reg_recon)     # reconstruction weight
        self.w_align     = float(reg_align)    # alignment weight
        self.anneal_steps = 2000   # steps to ramp w_gp from 0 -> w_gp_max
        self.global_step  = 0

        # control flow
        self.detach_Z_for_gp   = True   # use Z.detach() in GP loss early on
        self.freeze_encoder_after = 500   # steps to keep encoder frozen (Phase B)
        self.freeze_gp_to      = 500      # optional: keep GP frozen in very early AE warmup

        # optional noise schedule (if your GP exposes sigma_y)
        self.sigma_y_start = 0.20
        self.sigma_y_end   = 0.05
        self.sigma_sched_steps = 2000

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters(): 
            p.requires_grad = flag

    def _linear_ramp(self, t, T, a, b):
        if T <= 0: return b
        alpha = min(1.0, max(0.0, t / float(T)))
        return a + (b - a) * alpha

    def _update_schedules(self):
        # β-GP ramp
        self.w_gp = self._linear_ramp(self.global_step, self.anneal_steps, 0.0, self.w_gp_max)

        # noise anneal (best-effort; skip if attribute missing)
        # try:
        #     if hasattr(self.model, "sigma_y"):
        #         val = self._linear_ramp(self.global_step, self.sigma_sched_steps,
        #                                 self.sigma_y_start, self.sigma_y_end)
        #         # if sigma_y is a tensor/nn.Parameter
        #         if isinstance(self.model.sigma_y, torch.Tensor):
        #             with torch.no_grad():
        #                 self.model.sigma_y[...] = torch.as_tensor(val, dtype=self.model.sigma_y.dtype, device=self.model.sigma_y.device)
        #         else:
        #             # or setter
        #             setattr(self.model, "sigma_y", float(val))
        # except Exception:
        #     pass

        # freeze/unfreeze phases
        enc_on = self.global_step <= self.freeze_encoder_after
        gp_on  = self.global_step >= self.freeze_gp_to
        self._set_requires_grad(self.encoder, enc_on)
        # usually keep decoder on to preserve recon
        self._set_requires_grad(self.decoder, True)
        self._set_requires_grad(self.model,   gp_on)

        # detaching policy: stop detaching when GP is active and w_gp sizeable
        self._use_detach = self.detach_Z_for_gp and (self.global_step < self.freeze_encoder_after or self.w_gp < 0.25*self.w_gp_max)


    # ----------------------
    # window iteration
    # ----------------------
    def _iter_windows(self, N: int):
        if N <= self.window_size:
            yield 0, N
            return
        step = max(1, self.window_size - self.window_overlap)
        s = 0
        while s < N:
            e = min(N, s + self.window_size)
            yield s, e
            if e == N:
                break
            s += step

    # ----------------------
    # spatial feature builder
    # ----------------------
    def _spatial_x(
        self,
        meta: Dict,
        s: int,
        e: int,
        W_full: int
    ) -> torch.Tensor:
        """
        Build spatial coordinate x \in R^d (1,d) using meta + window indices.
        Normalized to [0,1].
        """
        xs: List[float] = []
        if self.use_row_index:
            r = float(meta.get("row_index", 0))
            H = float(meta.get("H", max(1, meta.get("height", 1))))
            xs.append(r / max(1., H - 1.))
        if self.use_window_center:
            cen = 0.5 * (s + e)
            xs.append(float(cen) / max(1., W_full))
        if self.use_layer_index:
            layer = float(meta.get("layer_index", 0.0))
            Ltot = float(meta.get("num_layers", 1.0))
            xs.append(layer / max(1., Ltot - 1.))
        if not xs:
            xs = [0.0]
        x = torch.tensor(xs, dtype=torch.float64, device=self.device).view(1, 1, -1)  # (1,1,d)
        return x

    # ----------------------
    # one training step (batch of rows) in embedding space
    # ----------------------
    def step_batch(self, rows_bwc: torch.Tensor, metas: List[Dict]) -> Dict[str, float]:
        self.model.train(); self.encoder.train(); self.decoder.train()
        rows_bwc = rows_bwc.to(self.device, dtype=torch.float64)
        B, W, C = rows_bwc.shape
        assert C == self.C

        X_feat, Z_obs = [], []
        recon_losses = []
        for b in range(B):
            meta = metas[b]
            W_full = int(meta.get("W", W))
            # If the collator already provides a window with [s,e] (or 'window_start','window_end'), use it:
            s = int(meta.get("s", meta.get("window_start", 0)))
            e = int(meta.get("e", meta.get("window_end", min(W_full, s + self.window_size))))
            s = max(0, s); e = max(s+1, e)
            y_win = rows_bwc[b:b+1, :min(W, e-s), :]  # (1, w', C) from batch
            wlen = y_win.shape[1]

            # Encode -> z
            z = self.encoder(y_win)                    # (1, D)

            # Recon loss: decoder(z) vs y_win
            y_hat = self.decoder(z, wlen=wlen)         # (1, w', C)
            recon_losses.append((y_hat - y_win).pow(2).mean())

            # Spatial input for this window
            x = self._spatial_x(meta, s, e, W_full)    # (1,1,d)
            X_feat.append(x)
            Z_obs.append(z)

        if not X_feat:
            return {"loss": 0.0, "n": 0}

        # --- stack & losses (as you already have) ---
        X = torch.cat(X_feat, dim=1)               # (1, N, d)
        Z = torch.cat(Z_obs, dim=0).unsqueeze(0)   # (1, N, D)

        # schedules
        self._update_schedules()

        # GP ELBO on latent targets (optionally on detached codes)
        Z_for_gp = Z.detach() if self._use_detach else Z
        x_list = [X for _ in range(self.D)]
        y_list = [Z_for_gp[..., i:i+1] for i in range(self.D)]
        vfe = self.model.compute_loss(x_list, y_list)  # maximize

        # GP mean vs encoder latent (alignment)
        means, _, _ = self.model(X)                 # List[(1,N,1)]
        M = torch.cat(means, dim=-1)                # (1, N, D)
        align = (M - Z).pow(2).mean()

        # Reconstruction
        recon = torch.stack(recon_losses).mean()

        # weighted total
        loss = self.w_gp * (-vfe) + self.w_align * align + self.w_rec * recon

        # different clipping for GP vs AE (optional)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip AE stronger than GP?
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=1.0)
        self.opt.step()

        self.global_step += 1
        return {
            "loss":  float(loss.detach().cpu()),
            "elbo":  float(vfe.detach().cpu()),
            "align": float(align.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "w_gp":  float(self.w_gp),
            "n":     int(X.shape[1]),
}


    # -------------
    # prediction in latent space at arbitrary spatial points
    # -------------
    @torch.no_grad()
    def predict_latent(self, X_query: torch.Tensor) -> torch.Tensor:
        """
        X_query: (1, N, d)
        returns latent mean (1, N, D)
        """
        self.model.eval()
        means, _, _ = self.model(X_query.to(self.device, dtype=torch.float64))
        return torch.cat(means, dim=-1)

    # helper to build X_query for a grid of rows in one image
    @torch.no_grad()
    def make_query_grid(self, rows: List[int], img_meta: Dict) -> torch.Tensor:
        Xq = []
        W_full = int(img_meta.get("W", self.W))
        for r in rows:
            # center each query at the middle of the row; you can also sweep window centers
            s, e = 0, W_full
            meta = dict(img_meta)
            meta["row_index"] = r
            Xq.append(self._spatial_x(meta, s, e, W_full))  # (1,1,d)
        return torch.cat(Xq, dim=1)  # (1, len(rows), d)
    
    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor, wlen: Optional[int] = None) -> torch.Tensor:
        """z: (1,D) -> (1,wlen,C)"""
        self.decoder.eval()
        return self.decoder(z.to(self.device, dtype=torch.float64), wlen=wlen)

    @torch.no_grad()
    def latent_posterior(self, Xq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xq: (1,N,d) -> (mu: (1,N,D), var_diag: (1,N,D))
        Works for independent-output GP layers.
        """
        self.model.eval()
        means, covs, _ = self.model(Xq.to(self.device, dtype=torch.float64))
        mu = torch.cat(means, dim=-1)                              # (1,N,D)
        var = torch.stack([torch.clamp(c[:, torch.arange(c.shape[1]), torch.arange(c.shape[2])], min=0.0)
                        for c in covs], dim=-1)                 # (1,N,D)
        return mu, var




# ---------------------------------------------------------------------
# Example wiring (kept in this file for convenience; remove if embedding elsewhere)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset: use lag-aligned signals from AM_rowwise_dataset
    from dataset.AM_rowwise_dataset import MultiPhaseDataset, RowDataset, row_collate_fn

    # Gaussian row model: import your implementation
    # Example uses your MOSKGP components; replace if you have another Gaussian model.
    from model.moskgp import (
        RBFKernel, PeriodicKernel, KernelDictionary,
        MultiOutputSparseGPLayer,
    )

    import torch
    from torch.utils.data import DataLoader

    # --- build dataset & loader (note primary_view="aligned")
    #folder = "/mnt/data/yiwang/code/RandomMaterial/clustered_grain/debug_2phases_no_cluster_GRF_noise_0/train"   # <- change me
    folder = "/mnt/data/yiwang/code/motion_code/data/test"
    img_ds = MultiPhaseDataset(folder=folder, n_phases=2, n_images=1)
    row_ds = RowDataset(
        image_dataset=img_ds,
        rows_per_image="all",
        window_size=128,
        stride=None,
        primary_view="aligned",      # <- use the lag-aligned view
        return_all_views=False,      # <- just the aligned tensor as x
        lag_rule="periodic_flip",
        precompute=True
    )
    print(len(row_ds))
    loader = DataLoader(row_ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=row_collate_fn)

    # --- infer shapes
    sample_x, sample_meta = row_ds[0]
    C = int(sample_x.shape[0])      # [C, win] per item -> collate -> [B, W, C]
    W = int(sample_meta["window_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- build a simple additive kernel & Gaussian model
    D = 2; d_in = 1
    kdict = KernelDictionary({
        "rbf": RBFKernel(lengthscale=0.6, variance=0.6),
        "per": PeriodicKernel(lengthscale=0.4, variance=0.5, period=0.2),
    })
    model = MultiOutputSparseGPLayer(
        input_dim=d_in,
        num_inducing_points=10,
        num_outputs=D,
        kernel=kdict,
        sigma_y=0.05,
        share_kernel=True,
    ).to(device)

    # --- trainer
    trainer = GaussianRowTrainer(
        model=model, W=W, C=C, window_size=128, window_overlap=0, device=device,
        embedding_dim=D, reg_align=1.0, reg_recon=1e2,
        use_row_index=True, use_window_center=False, use_layer_index=False
    )

    # --- training loop (no QoI; pure ELBO/VFE on aligned signals)
    iters = 40
    steps = 0
    for _ in range(iters):
        for rows_bwc, metas in loader:
            logs = trainer.step_batch(rows_bwc, metas)    # rows_bwc: [B, W, C], already aligned
            steps += 1
            if steps % 50 == 0:
                print(f"[step {steps}] loss={logs['loss']:.4f} elbo={logs['elbo']:.4f} align={logs['align']:.4f} recon={logs['recon']:.4f}")

    # Example prediction on a uniform grid
    #mean, cov = trainer.predict_row(W=W)
    #print("Predicted mean:", tuple(mean.shape), "cov:", tuple(cov.shape))

    rows_to_plot = [10, 42, 128]
    plot_lagged_sampling_from_qois(
        trainer=trainer,
        row_ds=row_ds,
        rows_to_plot=rows_to_plot,
        out_prefix="runs/latent_recon",
        n_samples=8
    )
