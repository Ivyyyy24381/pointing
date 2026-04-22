#!/usr/bin/env python3
"""
Process CCD Pointing Comprehension data.

Extracts baby pose CSVs from zipped/unzipped old pipeline data and copies
them to an organized output directory.

For each subject:
  - Reads from zip (or already-extracted folder) on the SSD
  - Copies per-trial CSVs and the combined CSV to output dir
  - No need to extract full images/videos (saves disk space)

Output structure:
  output_dir/
    CCD0194_PVPT_008E_side/
      combined_result.csv          (all trials combined)
      trial_1/
        processed_subject_result_table.csv
      trial_2/
        processed_subject_result_table.csv
      ...

Usage:
    python process_comprehension_CCD.py /path/to/source_dir -o /path/to/output_dir
    python process_comprehension_CCD.py /path/to/source_dir -o /path/to/output_dir --subject CCD0194
"""

import argparse
import io
import os
import re
import shutil
import zipfile
from pathlib import Path


def find_csvs_in_zip(zip_path: Path) -> dict:
    """
    Find all CSVs inside a zip file.

    Returns dict mapping relative path -> True for CSVs we care about.
    """
    csvs = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('processed_subject_result_table.csv'):
                csvs[name] = 'trial_csv'
            elif name.endswith('_combined_result.csv'):
                csvs[name] = 'combined_csv'
            elif name.endswith('auto_split.csv'):
                csvs[name] = 'auto_split'
    return csvs


def extract_csvs_from_zip(zip_path: Path, output_dir: Path, subject_name: str):
    """Extract just the CSVs from a zip file to the output directory."""
    csvs = find_csvs_in_zip(zip_path)
    if not csvs:
        print(f"  WARNING: No CSVs found in {zip_path.name}")
        return 0

    extracted = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for csv_path, csv_type in csvs.items():
            parts = Path(csv_path).parts

            if csv_type == 'trial_csv':
                # e.g., "1/processed_subject_result_table.csv" or "CCD0194/1/processed_..."
                # Find the trial number (a numeric folder name)
                trial_num = None
                for part in parts:
                    if part.isdigit():
                        trial_num = part
                        break
                if trial_num is None:
                    continue

                out_path = output_dir / f"trial_{trial_num}" / "processed_subject_result_table.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                data = zf.read(csv_path)
                out_path.write_bytes(data)
                extracted += 1

            elif csv_type == 'combined_csv':
                out_path = output_dir / "combined_result.csv"
                data = zf.read(csv_path)
                out_path.write_bytes(data)
                extracted += 1

            elif csv_type == 'auto_split':
                out_path = output_dir / "auto_split.csv"
                data = zf.read(csv_path)
                out_path.write_bytes(data)
                extracted += 1

    return extracted


def copy_csvs_from_folder(folder_path: Path, output_dir: Path, subject_name: str):
    """Copy CSVs from an already-extracted folder to the output directory."""
    extracted = 0

    # Find the inner data folder (e.g., CCD0194_PVPT.../CCD0194/)
    inner_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    data_dir = None
    for d in inner_dirs:
        # Check if it has numbered trial folders
        if any(sd.name.isdigit() for sd in d.iterdir() if sd.is_dir()):
            data_dir = d
            break

    if data_dir is None:
        # Maybe the folder itself has trial folders
        if any(d.name.isdigit() for d in folder_path.iterdir() if d.is_dir()):
            data_dir = folder_path

    if data_dir is None:
        print(f"  WARNING: No trial folders found in {folder_path}")
        return 0

    # Copy per-trial CSVs
    for trial_dir in sorted(data_dir.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.isdigit():
            continue

        csv_file = trial_dir / "processed_subject_result_table.csv"
        if csv_file.exists():
            out_path = output_dir / f"trial_{trial_dir.name}" / "processed_subject_result_table.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_file, out_path)
            extracted += 1

    # Copy combined CSV
    for f in data_dir.glob("*_combined_result.csv"):
        out_path = output_dir / "combined_result.csv"
        shutil.copy2(f, out_path)
        extracted += 1
        break

    # Also check parent for combined CSV
    if not (output_dir / "combined_result.csv").exists():
        for f in folder_path.glob("*_combined_result.csv"):
            out_path = output_dir / "combined_result.csv"
            shutil.copy2(f, out_path)
            extracted += 1
            break

    # Copy auto_split.csv
    for f in [data_dir / "auto_split.csv", folder_path / "auto_split.csv"]:
        if f.exists():
            out_path = output_dir / "auto_split.csv"
            shutil.copy2(f, out_path)
            extracted += 1
            break

    return extracted


def process_subject(subject_dir: Path, output_parent: Path) -> bool:
    """Process a single subject directory."""
    subject_name = subject_dir.name
    output_dir = output_parent / subject_name

    print(f"\n  {subject_name}")

    # Check if already processed
    if output_dir.exists():
        existing_trials = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
        if existing_trials:
            print(f"    Already processed ({len(existing_trials)} trials) - SKIP")
            return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find zip file inside subject dir
    zips = list(subject_dir.glob("*.zip"))

    # Check if data is already extracted (has numbered folders inside a subfolder)
    has_extracted = False
    for d in subject_dir.iterdir():
        if d.is_dir():
            if any(sd.name.isdigit() for sd in d.iterdir() if sd.is_dir()):
                has_extracted = True
                break

    extracted = 0
    if has_extracted:
        print(f"    Source: extracted folder")
        extracted = copy_csvs_from_folder(subject_dir, output_dir, subject_name)
    elif zips:
        print(f"    Source: {len(zips)} zip file(s)")
        for zf in sorted(zips):
            n = extract_csvs_from_zip(zf, output_dir, subject_name)
            extracted += n
    else:
        print(f"    WARNING: No data found (no zips, no extracted folders)")
        return False

    # Count output trials
    trial_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')])
    has_combined = (output_dir / "combined_result.csv").exists()

    print(f"    Extracted: {len(trial_dirs)} trial CSVs" +
          (f" + combined CSV" if has_combined else ""))

    return len(trial_dirs) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract baby pose CSVs from CCD comprehension data")
    parser.add_argument("source_dir",
                        help="Directory containing subject folders (with zips or extracted data)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for extracted CSVs")
    parser.add_argument("--subject", help="Process only this subject (folder name prefix)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subject directories
    subjects = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    if args.subject:
        subjects = [d for d in subjects if d.name.startswith(args.subject)]

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Subjects: {len(subjects)}")

    ok = 0
    fail = 0
    skip = 0

    for i, subject_dir in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}]", end="")
        result = process_subject(subject_dir, output_dir)
        if result:
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"  DONE: {ok} OK, {fail} FAILED / {len(subjects)} total")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
