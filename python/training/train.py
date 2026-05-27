"""
Training loop for BleedDemucs drum stem separation model.

Phase 1: Overfitting test — train on a single take to validate
          that the architecture and loss are sound before scaling up.

Usage (overfitting sanity check):
    python python/training/train.py \
        --data_dir data/raw \
        --overfit_one \
        --epochs 200

Usage (full training):
    python python/training/train.py \
        --data_dir data/raw \
        --epochs 100 \
        --batch_size 4
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# Allow running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.data.dataset import DrumStemDataset
from python.models.demucs import BleedDemucs, STEMS
from python.training.loss import SISNRLoss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"  [OK] Saved checkpoint -> {path}")


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    dataset = DrumStemDataset(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        chunk_length_sec=args.chunk_sec,
        stems=STEMS,
        augment=not args.overfit_one,  # No augment during overfit test
    )

    if len(dataset.takes) == 0:
        print("ERROR: No valid takes found. Prepare data first with prepare_cmt_dataset.py")
        return

    if args.overfit_one:
        print("[OVERFIT MODE] Training on a single take to validate the model.")
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
    print(f"BleedDemucs parameters: {n_params / 1e6:.2f}M")

    # --- Loss & Optimiser ---
    criterion = SISNRLoss(stems=STEMS)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # --- Resume ---
    start_epoch = 0
    ckpt_path = Path(args.checkpoint_dir) / "latest.pt"
    if ckpt_path.exists() and not args.overfit_one:
        print(f"Resuming from {ckpt_path}")
        start_epoch, _ = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch += 1

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (mixture, targets) in enumerate(loader):
            mixture = mixture.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            # Model expects (B, in_channels, T) — mixture is (B, 2, T)
            # If mono input mode, mix to mono
            if args.in_channels == 1:
                x = mixture.mean(dim=1, keepdim=True)
            else:
                x = mixture

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {avg_loss:+.4f} dB | {elapsed:.1f}s")

        # Save every N epochs
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

    # Final save
    save_checkpoint(model, optimizer, args.epochs - 1, avg_loss, Path(args.checkpoint_dir) / "final.pt")
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BleedDemucs drum separator")

    # Data
    parser.add_argument("--data_dir",     type=str,   default="data/raw",  help="Path to prepared takes directory")
    parser.add_argument("--sample_rate",  type=int,   default=44100)
    parser.add_argument("--chunk_sec",    type=float, default=4.0,         help="Audio chunk length in seconds")
    parser.add_argument("--in_channels",  type=int,   default=1,           help="1=mono input, 2=stereo input")

    # Model
    parser.add_argument("--depth",        type=int,   default=6,           help="Encoder/decoder depth")
    parser.add_argument("--channels",     type=int,   default=64,          help="Base channel count")

    # Training
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--num_workers",  type=int,   default=0,           help="DataLoader workers (0 = main process, safer on Windows)")
    parser.add_argument("--save_every",   type=int,   default=5,           help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")

    # Debug
    parser.add_argument("--overfit_one",  action="store_true",             help="Overfit on one take (sanity check)")

    args = parser.parse_args()
    train(args)
