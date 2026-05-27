"""
Training loop for BleedDemucs drum stem separation model.

Supports multiple loss functions and automatic stem export at listening checkpoints
so you can hear separation quality evolve without waiting for full training.

Usage (diagnostic overfit on debug dataset):
    python python/training/train.py \\
        --data_dir data/debug \\
        --overfit_one \\
        --epochs 200 \\
        --depth 3 --channels 32 --chunk_sec 0.5 \\
        --loss hybrid \\
        --export_epochs 5,10,20,50,100,200

Usage (full training):
    python python/training/train.py \\
        --data_dir data/raw \\
        --epochs 300 --batch_size 4 \\
        --loss hybrid
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.data.dataset import DrumStemDataset
from python.models.demucs import BleedDemucs, STEMS
from python.training.losses import get_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, path: Path,
                    depth: int, channels: int, loss_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
        # Store model config so checkpoints are self-describing
        "depth":                depth,
        "channels":             channels,
        "loss_name":            loss_name,
    }, path)
    print(f"  [ckpt] Saved -> {path}")


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


# ---------------------------------------------------------------------------
# Listening checkpoint: export separated stems to output/checkpoints/epoch_N/
# ---------------------------------------------------------------------------

def _load_take_mixture(take_dir: Path, sample_rate: int):
    """Sum all 4 stems to build the mixture from a take folder."""
    mixture = None
    for s in STEMS:
        p = take_dir / f"{s}.wav"
        if not p.exists():
            continue
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)   # (C, T)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        if sr != sample_rate:
            wav = F.interpolate(wav.unsqueeze(0),
                                size=int(wav.shape[-1] * sample_rate / sr),
                                mode="linear", align_corners=False).squeeze(0)
        mixture = wav if mixture is None else mixture + wav
    if mixture is not None:
        mx = mixture.abs().max()
        if mx > 1.0:
            mixture = mixture / mx
    return mixture


def export_stems(model: BleedDemucs, dataset: DrumStemDataset | Subset,
                 epoch: int, output_dir: str, sample_rate: int,
                 chunk_sec: float, device: torch.device):
    """
    Run inference on the first take and write stems to:
        output/<output_dir>/epoch_<N>/kick.wav  etc.
    Also writes the input mixture for A/B reference.
    """
    # Retrieve the first take directory
    inner = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not inner.takes:
        return

    take_dir = inner.takes[0]
    mixture  = _load_take_mixture(take_dir, sample_rate)
    if mixture is None:
        return

    chunk_len  = int(chunk_sec * sample_rate)
    hop_len    = int(chunk_len * 0.25)
    T          = mixture.shape[-1]
    window     = torch.hann_window(chunk_len)

    out_acc = {s: torch.zeros(2, T) for s in STEMS}
    weight  = torch.zeros(T)

    model.eval()
    with torch.no_grad():
        pos = 0
        while pos < T:
            end   = min(pos + chunk_len, T)
            chunk = mixture[:, pos:end]
            if chunk.shape[-1] < chunk_len:
                chunk = F.pad(chunk, (0, chunk_len - chunk.shape[-1]))
            mono  = chunk.mean(0, keepdim=True).unsqueeze(0).to(device)
            preds = model(mono)
            w     = window[:end - pos]
            for s in STEMS:
                out_acc[s][:, pos:end] += preds[s][0, :, :end - pos].cpu() * w
            weight[pos:end] += w
            pos += hop_len

    weight = weight.clamp(min=1e-8)

    epoch_dir = Path(output_dir) / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Write mixture reference
    mix_np = mixture.mean(0).numpy()
    stereo = np.stack([mix_np, mix_np], axis=1).astype(np.float32)
    sf.write(str(epoch_dir / "_mixture.wav"), stereo, sample_rate, subtype="PCM_24")

    for s in STEMS:
        wav   = out_acc[s] / weight.unsqueeze(0)
        wav   = wav.clamp(-1, 1)
        arr   = wav.numpy().T.astype(np.float32)  # (T, 2)
        sf.write(str(epoch_dir / f"{s}.wav"), arr, sample_rate, subtype="PCM_24")

    print(f"  [listen] Stems exported -> {epoch_dir}/")
    model.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Loss     : {args.loss}")
    print(f"Data     : {args.data_dir}")

    # Parse export epochs
    export_epochs = set()
    if args.export_epochs:
        for tok in args.export_epochs.split(","):
            tok = tok.strip()
            if tok:
                export_epochs.add(int(tok))

    # --- Dataset ---
    dataset = DrumStemDataset(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        chunk_length_sec=args.chunk_sec,
        stems=STEMS,
        augment=not args.overfit_one,
    )

    if len(dataset.takes) == 0:
        print("ERROR: No valid takes found.")
        return

    if args.overfit_one:
        print(f"[OVERFIT MODE] {len(dataset.takes)} take(s), clamped to first 10 items.")
        dataset = Subset(dataset, list(range(min(10, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = BleedDemucs(
        in_channels=args.in_channels,
        out_channels=2,
        num_stems=len(STEMS),
        depth=args.depth,
        channels=args.channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model    : BleedDemucs depth={args.depth} ch={args.channels}  "
          f"params={n_params/1e6:.2f}M")

    # --- Loss & Optimiser ---
    criterion = get_loss(args.loss, stems=STEMS, sample_rate=args.sample_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5
    )

    # --- Resume ---
    start_epoch = 0
    ckpt_path   = Path(args.checkpoint_dir) / "latest.pt"
    if ckpt_path.exists() and not args.overfit_one:
        print(f"Resuming from {ckpt_path}")
        start_epoch, _ = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch += 1

    # Export epoch-0 baseline (untrained reference)
    if export_epochs:
        export_stems(model, dataset, 0, args.output_dir,
                     args.sample_rate, args.chunk_sec, device)

    # --- Training loop ---
    avg_loss = 0.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for mixture, targets in loader:
            mixture = mixture.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            x = mixture.mean(dim=1, keepdim=True) if args.in_channels == 1 else mixture

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        scheduler.step(avg_loss)
        lr_now   = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {avg_loss:+.4f} | "
              f"lr={lr_now:.1e} | {elapsed:.1f}s")

        # Checkpoint save
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path,
                            args.depth, args.channels, args.loss)

        # Listening checkpoint stem export
        ep1 = epoch + 1
        if ep1 in export_epochs:
            export_stems(model, dataset, ep1, args.output_dir,
                         args.sample_rate, args.chunk_sec, device)

    # Final save
    final_path = Path(args.checkpoint_dir) / "final.pt"
    save_checkpoint(model, optimizer, args.epochs - 1, avg_loss, final_path,
                    args.depth, args.channels, args.loss)
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BleedDemucs drum separator")

    # Data
    parser.add_argument("--data_dir",     type=str,   default="data/raw")
    parser.add_argument("--sample_rate",  type=int,   default=44100)
    parser.add_argument("--chunk_sec",    type=float, default=4.0)
    parser.add_argument("--in_channels",  type=int,   default=1)

    # Model
    parser.add_argument("--depth",        type=int,   default=6)
    parser.add_argument("--channels",     type=int,   default=64)

    # Loss
    parser.add_argument("--loss",         type=str,   default="hybrid",
                        choices=["si_sdr", "l1", "mse", "mrstft", "hybrid"],
                        help="Loss function (default: hybrid = MRSTFT + SI-SDR)")

    # Training
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--save_every",   type=int,   default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")

    # Listening checkpoints
    parser.add_argument("--export_epochs", type=str,  default="5,10,20,50",
                        help="Comma-separated epochs to export stems for listening. "
                             "e.g. '5,10,20,50,100'")
    parser.add_argument("--output_dir",   type=str,   default="output/checkpoints",
                        help="Where to write listening checkpoint stems")

    # Debug
    parser.add_argument("--overfit_one",  action="store_true")

    args = parser.parse_args()
    train(args)
