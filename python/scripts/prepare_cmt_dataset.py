import os
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def merge_audio_files(input_files, output_file):
    """Mixes multiple audio files into one."""
    if not input_files:
        return
        
    print(f"Mixing {len(input_files)} files to {output_file}...")
    mixed_audio = None
    sr_out = None
    
    for f in input_files:
        audio, sr = librosa.load(f, sr=None, mono=False)
        if sr_out is None:
            sr_out = sr
            mixed_audio = audio
        else:
            if sr != sr_out:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out)
            
            # Pad if length mismatch
            if audio.shape[-1] > mixed_audio.shape[-1]:
                pad = audio.shape[-1] - mixed_audio.shape[-1]
                if mixed_audio.ndim == 1:
                    mixed_audio = np.pad(mixed_audio, (0, pad))
                else:
                    mixed_audio = np.pad(mixed_audio, ((0,0), (0, pad)))
            elif audio.shape[-1] < mixed_audio.shape[-1]:
                pad = mixed_audio.shape[-1] - audio.shape[-1]
                if audio.ndim == 1:
                    audio = np.pad(audio, (0, pad))
                else:
                    audio = np.pad(audio, ((0,0), (0, pad)))
                    
            mixed_audio += audio
            
    sf.write(output_file, mixed_audio.T if mixed_audio.ndim > 1 else mixed_audio, sr_out)

def process_directory(source_dir, dest_dir):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Keyword matching for stems (handles common Cambridge MT naming conventions)
    stem_keywords = {
        'kick': ['kick', 'kck', 'b.d'],
        'snare': ['snare', 'snr', 'sn', 's.d'],
        'toms': ['tom', 't1', 't2', 't3', 'floor', 'rack'],
        'overheads': ['oh', 'overhead', 'cymb', 'ride', 'crash', 'hat', 'hh', 'room', 'amb'] 
    }
    
    stem_files = {k: [] for k in stem_keywords.keys()}
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith(('.wav', '.aif', '.aiff', '.flac')):
                continue
                
            filepath = os.path.join(root, file)
            filename = file.lower()
            
            matched = False
            for stem, keywords in stem_keywords.items():
                if any(kw in filename for kw in keywords):
                    stem_files[stem].append(filepath)
                    matched = True
                    break
            
            if not matched:
                print(f"Skipping {file} - not identified as drum stem.")
                
    for stem, files in stem_files.items():
        if files:
            merge_audio_files(files, dest_dir / f"{stem}.wav")
        else:
            print(f"WARNING: No files found for stem '{stem}' in {source_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Cambridge Multitrack stems")
    parser.add_argument("--source", type=str, required=True, help="Directory containing raw stems")
    parser.add_argument("--dest", type=str, required=True, help="Destination directory for processed stems (e.g. data/raw/Take_01)")
    args = parser.parse_args()
    
    process_directory(args.source, args.dest)
    print(f"Done processing to {args.dest}")
