"""
Batch-processes all downloaded Cambridge Multitrack sessions into the standard
4-stem format (kick.wav, snare.wav, toms.wav, overheads.wav) required by
the BleedDemucs training pipeline.

For each session folder found in --source_dir, it calls the logic from
prepare_cmt_dataset.py and writes merged stems into --dest_dir/<session_name>/.

Usage:
    python python/scripts/batch_prepare_cmt.py \
        --source_dir data/cambridge_raw \
        --dest_dir data/raw \
        --drums_only

The --drums_only flag skips any session folder where no drum files
(kick/snare/tom/overhead keywords) are detected at all.
"""

import argparse
from pathlib import Path
import sys

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.scripts.prepare_cmt_dataset import process_directory, stem_keywords_has_drums


def batch_prepare(source_dir: Path, dest_dir: Path, drums_only: bool, overwrite: bool):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Each direct subdirectory of source_dir is treated as a session
    sessions = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    if not sessions:
        # The cambridge tool may nest one level deeper, search recursively up to depth 2
        sessions = sorted([d for d in source_dir.rglob("*") if d.is_dir() and d.parent != source_dir])

    if not sessions:
        print(f"No session directories found in {source_dir}")
        return

    print(f"Found {len(sessions)} session folder(s) to process.\n")
    processed, skipped, errors = 0, 0, 0

    for session in sessions:
        name = session.name
        out_path = dest_dir / name

        if out_path.exists() and not overwrite:
            existing = list(out_path.glob("*.wav"))
            if existing:
                print(f"[SKIP] {name} — already processed ({len(existing)} wav files found)")
                skipped += 1
                continue

        if drums_only:
            # Quick pre-check: does this folder contain any drum-related files?
            all_files = [f.name.lower() for f in session.rglob("*.wav")]
            drum_kws = ["kick", "kck", "snare", "snr", "tom", "oh", "overhead", "cymb", "hat", "room", "b.d", "s.d"]
            has_drums = any(any(kw in f for kw in drum_kws) for f in all_files)
            if not has_drums:
                print(f"[SKIP] {name} — no drum files detected (use without --drums_only to force)")
                skipped += 1
                continue

        print(f"[PROCESS] {name}")
        try:
            process_directory(session, out_path)
            processed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Done.  Processed: {processed}  Skipped: {skipped}  Errors: {errors}")
    print(f"Prepared dataset saved to: {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prepare Cambridge MT sessions")
    parser.add_argument("--source_dir", type=str, default="data/cambridge_raw",
                        help="Directory containing raw downloaded session folders")
    parser.add_argument("--dest_dir",   type=str, default="data/raw",
                        help="Output directory for processed 4-stem takes")
    parser.add_argument("--drums_only", action="store_true",
                        help="Skip sessions with no drum-related file names")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-process sessions that already have output")
    args = parser.parse_args()

    batch_prepare(
        source_dir=Path(args.source_dir),
        dest_dir=Path(args.dest_dir),
        drums_only=args.drums_only,
        overwrite=args.overwrite,
    )
