#!/usr/bin/env python3
"""
Merge split zip files downloaded from Google Drive.

When you download large folders from Google Drive, it splits each folder
into multiple zips: FolderName-001.zip, FolderName-002.zip, ...
Each zip is independent (not a split archive) with a subset of files.

This script auto-groups zips by folder name, extracts each group into
its own directory, removes unnecessary visualization MP4s, and re-zips
into a single archive to save disk space.

Usage:
    # Extract all zips, cleanup, re-zip (default)
    python merge_gdrive_zips.py ~/Downloads/

    # Extract to a specific output directory
    python merge_gdrive_zips.py ~/Downloads/ -o /path/to/output/

    # Keep zips after extracting (don't delete)
    python merge_gdrive_zips.py ~/Downloads/ --keep

    # Skip re-zip step (just extract and cleanup)
    python merge_gdrive_zips.py ~/Downloads/ --no-rezip
"""

import argparse
import os
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from shutil import disk_usage


def group_zips(zip_dir: Path) -> dict[str, list[Path]]:
    """
    Group zip files by their base folder name.

    Google Drive naming examples:
      BDL049_Star_front_cam-20260220T152735Z-3-001.zip
      BDL049_Star_front_cam-20260220T152735Z-3-002.zip
    Groups by everything before the last -NNN.

    Returns: dict mapping group_name -> sorted list of zip paths
    """
    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        return {}

    groups = defaultdict(list)
    for zf in zip_files:
        name = zf.stem
        # Match trailing -NNN (3-digit part number)
        m = re.match(r'^(.+)-(\d{3})$', name)
        if m:
            base_name = m.group(1)
        else:
            base_name = name
        groups[base_name].append(zf)

    for key in groups:
        groups[key] = sorted(groups[key])

    return dict(groups)


def clean_folder_name(group_name: str) -> str:
    """
    Extract a clean folder name from the gdrive group name.

    'BDL049_Star_front_cam-20260220T152735Z-3' -> 'BDL049_Star_front_cam'
    'BDL207_Ed_front-20260220T152752Z-3'       -> 'BDL207_Ed_front'
    """
    # Strip the timestamp+suffix: -YYYYMMDDTHHMMSSZ-N
    m = re.match(r'^(.+?)-\d{8}T\d{6}Z.*$', group_name)
    if m:
        return m.group(1)
    return group_name


def extract_group(zip_files: list[Path], output_dir: Path, delete: bool = True):
    """Extract a group of zips into one directory, deleting each after extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    freed = 0

    for zf in zip_files:
        if not zf.exists():
            print(f"    SKIP (already gone): {zf.name}")
            continue
        file_size = zf.stat().st_size
        size_mb = file_size / (1024**2)
        print(f"    Extracting: {zf.name} ({size_mb:.0f} MB) ... ", end="", flush=True)
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(output_dir)
        print("done", end="")

        if delete:
            freed += file_size
            zf.unlink()
            os.sync()  # flush to disk so space is freed immediately
            print(" [deleted]", end="")
        print()

    total_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
    print(f"    -> {total_files} files in {output_dir}")
    if delete and freed > 0:
        print(f"    -> freed {freed / (1024**3):.2f} GB")
    return total_files


# Files to remove from the study-level directory (visualization-only, not needed)
UNNECESSARY_STUDY_FILES = ["Depth.mp4", "Depth_Color.mp4", "Color.mp4", "output.mp4"]


def cleanup_study(study_dir: Path) -> int:
    """
    Remove unnecessary visualization MP4s from the study-level directory.

    The study folder structure is: study_dir/StudyName/  (inner dir with the data)
    Inside the inner dir, Depth.mp4 and Depth_Color.mp4 are visualization-only.

    Returns bytes freed.
    """
    freed = 0
    # The inner directory has the same name as the study folder
    inner_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    for inner_dir in inner_dirs:
        for fname in UNNECESSARY_STUDY_FILES:
            fpath = inner_dir / fname
            if fpath.exists():
                size = fpath.stat().st_size
                freed += size
                fpath.unlink()
                print(f"    [cleanup] Removed {inner_dir.name}/{fname} "
                      f"({size / (1024**2):.0f} MB)")
    return freed


def rezip_folder(folder: Path, zip_path: Path):
    """
    Zip a folder into a single archive using store mode (no compression),
    then remove the original folder.

    Uses the system `zip` command for speed on large directories with
    many files. Falls back to Python zipfile if `zip` is not available.
    """
    print(f"    [rezip] Creating {zip_path.name} ... ", end="", flush=True)

    # Use system zip -r -0 (store mode) for speed
    try:
        result = subprocess.run(
            ["zip", "-r", "-0", "-q", str(zip_path), folder.name],
            cwd=str(folder.parent),
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    except (FileNotFoundError, RuntimeError):
        # Fallback: Python zipfile (slower but always available)
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            for fpath in sorted(folder.rglob("*")):
                arcname = str(fpath.relative_to(folder.parent))
                zf.write(fpath, arcname)

    zip_size = zip_path.stat().st_size
    print(f"done ({zip_size / (1024**3):.1f} GB)")

    # Verify and remove original
    try:
        subprocess.run(
            ["zip", "-T", "-q", str(zip_path)],
            capture_output=True, check=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # If system zip not available or test fails, do a basic Python check
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                if zf.testzip() is not None:
                    print(f"    [rezip] ERROR: zip verification failed, keeping folder")
                    zip_path.unlink()
                    return
        except zipfile.BadZipFile:
            print(f"    [rezip] ERROR: bad zip file, keeping folder")
            zip_path.unlink()
            return

    print(f"    [rezip] Removing extracted folder ... ", end="", flush=True)
    shutil.rmtree(folder)
    os.sync()
    print("done")


def main():
    parser = argparse.ArgumentParser(
        description="Merge split Google Drive zip downloads into separate folders")
    parser.add_argument("zip_dir", help="Directory containing zip files")
    parser.add_argument("-o", "--output",
                        help="Output parent directory (default: <zip_dir>)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep zip files after extraction (default: delete to save space)")
    parser.add_argument("--no-rezip", action="store_true",
                        help="Skip re-zipping (just extract, cleanup, and leave as folders)")
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir)
    out_parent = Path(args.output) if args.output else zip_dir
    delete = not args.keep
    rezip = not args.no_rezip

    groups = group_zips(zip_dir)
    if not groups:
        print(f"No zip files found in {zip_dir}")
        return

    total_zips = sum(len(v) for v in groups.values())
    total_size = sum(z.stat().st_size for zlist in groups.values() for z in zlist if z.exists())
    free_space = disk_usage(str(zip_dir)).free
    print(f"Found {total_zips} zip(s) in {len(groups)} group(s) "
          f"({total_size / (1024**3):.1f} GB total)")
    print(f"Disk free: {free_space / (1024**3):.1f} GB")
    mode_parts = []
    if delete:
        mode_parts.append("extract + delete source zips")
    else:
        mode_parts.append("extract only (--keep)")
    mode_parts.append("cleanup viz MP4s")
    if rezip:
        mode_parts.append("re-zip")
    print(f"Mode: {', '.join(mode_parts)}\n")

    for i, (group_name, zip_files) in enumerate(sorted(groups.items()), 1):
        folder_name = clean_folder_name(group_name)
        group_size = sum(z.stat().st_size for z in zip_files if z.exists())
        print(f"[{i}/{len(groups)}] {folder_name} "
              f"({len(zip_files)} zip(s), {group_size / (1024**3):.1f} GB)")
        out_dir = out_parent / folder_name
        extract_group(zip_files, out_dir, delete=delete)

        # Remove unnecessary visualization MP4s
        freed = cleanup_study(out_dir)
        if freed > 0:
            print(f"    [cleanup] Freed {freed / (1024**2):.0f} MB")

        # Re-zip into a single archive and remove the extracted folder
        if rezip:
            final_zip = out_parent / f"{folder_name}.zip"
            if final_zip.exists():
                print(f"    [rezip] SKIP: {final_zip.name} already exists")
            else:
                rezip_folder(out_dir, final_zip)

        free_now = disk_usage(str(out_parent)).free
        print(f"    Disk free: {free_now / (1024**3):.1f} GB")
        print()

    print("All done!")


if __name__ == "__main__":
    main()
