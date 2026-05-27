"""
Download Cambridge Multitrack stems from https://www.cambridge-mt.com/ms/mtk/

Scrapes the MTK page for download links, filters to drum-heavy genres,
downloads ZIP archives, and extracts them into a local directory.

Usage:
    # Download a small batch (5 songs) from Acoustic genre for testing:
    python python/scripts/download_cmt.py --output_dir data/cambridge_raw --genre Acoustic --limit 5

    # Download all Pop + Acoustic (large — many GB):
    python python/scripts/download_cmt.py --output_dir data/cambridge_raw --genre Pop Acoustic
"""

import argparse
import os
import re
import zipfile
from pathlib import Path

import cloudscraper
from bs4 import BeautifulSoup
from tqdm import tqdm

# Shared cloudscraper session (handles Cloudflare JS challenges automatically)
_scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})

CMT_URL = "https://www.cambridge-mt.com/ms/mtk/"

# Genre IDs as they appear in the page HTML
# Acoustic and Pop tend to contain real drum kit recordings
DRUM_GENRES = ["Acoustic", "Pop", "HipHop", "Electronica"]


def fetch_page(url: str) -> BeautifulSoup:
    resp = _scraper.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")


def get_download_links(page: BeautifulSoup, genres: list[str]) -> list[dict]:
    """
    Extract 'Full Multitrack' download links filtered by genre.
    Returns list of dicts: {title, genre, url}
    """
    results = []

    for genre_div in page.find_all("div", class_="c-mtk__genre"):
        h3 = genre_div.find("h3")
        if h3 is None:
            continue
        genre_id = h3.get("id", "")

        # Match any supplied genre (case-insensitive partial match)
        if not any(g.lower() in genre_id.lower() for g in genres):
            continue

        for entry in genre_div.find_all("div", class_="m-mtk-download__content"):
            try:
                dl_type = entry.find("div", class_="m-mtk-download__type")
                if dl_type is None or "Full" not in dl_type.text:
                    continue
                link_tag = entry.find("a")
                if link_tag is None:
                    continue
                href = link_tag["href"]
                # Try to extract song title from a nearby heading
                parent = entry.find_parent("div", class_="m-mtk-item")
                title = parent.find("h4").text.strip() if parent and parent.find("h4") else Path(href).stem
                results.append({"title": title, "genre": genre_id, "url": href})
            except Exception:
                continue

    return results


def download_file(url: str, dest_path: Path) -> bool:
    """Stream-download a file with a progress bar."""
    try:
        resp = _scraper.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name, total=total, unit="B", unit_scale=True, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_zip(zip_path: Path, out_dir: Path):
    """Extract a ZIP and delete it afterwards."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        zip_path.unlink()
    except zipfile.BadZipFile as e:
        print(f"  ERROR extracting {zip_path}: {e}")
        zip_path.unlink()


def safe_folder_name(title: str) -> str:
    """Convert a song title to a safe directory name."""
    name = re.sub(r'[\\/:*?"<>|]', "_", title)
    return name.strip()[:80]


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Cambridge MT catalogue from {CMT_URL} ...")
    page = fetch_page(CMT_URL)
    links = get_download_links(page, args.genre)

    if not links:
        print("No links found. The page structure may have changed.")
        return

    print(f"Found {len(links)} entries matching genres: {args.genre}")

    if args.limit:
        links = links[: args.limit]
        print(f"Limiting to first {args.limit} entries.")

    for i, entry in enumerate(links, 1):
        title = entry["title"]
        url = entry["url"]
        folder_name = safe_folder_name(title)
        song_dir = output_dir / folder_name

        print(f"\n[{i}/{len(links)}] {title}")

        if song_dir.exists() and any(song_dir.iterdir()):
            print(f"  Already downloaded, skipping.")
            continue

        song_dir.mkdir(parents=True, exist_ok=True)

        # Download ZIP
        zip_name = Path(url).name
        zip_path = song_dir / zip_name
        print(f"  Downloading {url} ...")
        ok = download_file(url, zip_path)
        if not ok:
            continue

        # Extract
        print(f"  Extracting ...")
        extract_zip(zip_path, song_dir)
        print(f"  Done → {song_dir}")

    print(f"\n✓ All downloads complete. Raw sessions saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Cambridge Multitrack stems")
    parser.add_argument(
        "--output_dir", type=str, default="data/cambridge_raw",
        help="Directory to save downloaded raw stems"
    )
    parser.add_argument(
        "--genre", nargs="+", default=["Acoustic"],
        choices=DRUM_GENRES,
        help="Genres to download (default: Acoustic)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of songs to download (useful for testing)"
    )
    args = parser.parse_args()
    main(args)
