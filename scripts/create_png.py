#!/usr/bin/env python3
"""
Convert .npz files under INPUT_DIR/All to PNGs placed under OUTPUT_DIR/png,
organized into test_slo_fundus, training_slo_fundus, val_slo_fundus folders
based on the 'use' column in data_summary.csv. Optionally filter files listed
in filter_file.txt (one filename per line, without path).
"""

import argparse
import os
import csv
import numpy as np
from PIL import Image

def read_filter(filter_path):
    if not filter_path or not os.path.isfile(filter_path):
        return set()
    with open(filter_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def read_summary_csv(csv_path):
    """
    Returns a dict: filename -> use_value (e.g., 'training', 'test', 'val' or None)
    Accepts CSVs that have a 'filename' column and optional 'use' column.
    """
    mapping = {}
    if not csv_path or not os.path.isfile(csv_path):
        return mapping
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # fall back to positional reading if 'filename' not in header
        if 'filename' not in reader.fieldnames:
            # read rows as lists
            csvfile.seek(0)
            for row in csv.reader(csvfile):
                if not row:
                    continue
                filename = row[0].strip()
                use_val = row[1].strip() if len(row) > 1 else None
                mapping[filename] = use_val
            return mapping
        for row in reader:
            filename = str(row.get('filename', '')).strip()
            if not filename:
                continue
            use_val = row.get('use', None)
            if use_val is not None:
                use_val = str(use_val).strip().lower()
            mapping[filename] = use_val
    return mapping

def pick_image_from_npz(npz_path):
    """
    Load .npz and heuristically pick an array suitable for saving as image.
    Returns a numpy array of shape (H, W) or (H, W, 3) with dtype uint8 or
    convertible to uint8.
    """

    with np.load(npz_path, allow_pickle=True) as data:
        # Only read the 'slo_fundus' entry if present.
        # If the .npz was saved as a naked ndarray (unlikely here), do not use it.

        if 'slo_fundus' in data.files:
            arr = data['slo_fundus']
            if isinstance(arr, np.ndarray):
                return arr


def normalize_and_convert(arr):
    """
    Convert array to uint8 image array.
    - If float, scale 0-1 or min-max to 0-255.
    - If single channel, convert to RGB.
    """
    if arr is None:
        return None
    # squeeze possible singleton channel
    arr = np.array(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    # If values are floats, normalize
    if np.issubdtype(arr.dtype, np.floating):
        minv = np.nanmin(arr)
        maxv = np.nanmax(arr)
        if np.isfinite(minv) and np.isfinite(maxv) and maxv > minv:
            arr = (arr - minv) / (maxv - minv)
        else:
            arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        # if bigger than 8-bit, downscale
        if arr.dtype != np.uint8:
            # scale to 0-255 based on min/max
            minv = np.min(arr)
            maxv = np.max(arr)
            if maxv > minv:
                arr = ((arr - minv) / (maxv - minv) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            # drop alpha
            arr = arr[:, :, :3]
        elif arr.shape[2] == 2:
            # unlikely: replicate channels
            arr = np.repeat(arr[:, :, :1], 3, axis=2)
    return arr

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main(args):
    input_dir = args.input_dir
    output_base = args.output_dir
    csv_path = args.summary_csv
    filter_path = args.filter_file

    # mapping of 'use' values -> output subdir names
    use_to_subdir = {
        'training': 'training_slo_fundus',
        'train': 'training_slo_fundus',
        'testing': 'test_slo_fundus',
        'test': 'test_slo_fundus',
        'val': 'val_slo_fundus',
        'validation': 'val_slo_fundus'
    }

    filters = read_filter(filter_path)
    summary = read_summary_csv(csv_path)

    all_npz_dir = os.path.join(input_dir, "All")
    if not os.path.isdir(all_npz_dir):
        print(f"Input directory not found: {all_npz_dir}")
        return

    png_root = os.path.join(output_base, "png")
    ensure_dir(png_root)
    for sub in set(use_to_subdir.values()):
        ensure_dir(os.path.join(png_root, sub))

    processed = 0
    skipped = 0
    errors = 0

    for root, _, files in os.walk(all_npz_dir):
        for fname in files:
            if not fname.lower().endswith('.npz'):
                continue
            if (fname.replace('.npz', '.png') not in filters):
                skipped += 1
                continue

            rel_name = fname
            use_val = summary.get(rel_name)
            # sometimes csv stores names without extension
            if use_val is None:
                use_val = summary.get(os.path.splitext(rel_name)[0])

            out_sub = use_to_subdir.get(use_val)
            if out_sub is None:
                # default: put into training if unknown
                out_sub = 'training_slo_fundus'

            in_path = os.path.join(root, fname)
            try:
                arr = pick_image_from_npz(in_path)
                img_arr = normalize_and_convert(arr)
                if img_arr is None:
                    errors += 1
                    print(f"Warning: no suitable array found in {in_path}")
                    continue
                img = Image.fromarray(img_arr)
                out_dir = os.path.join(png_root, out_sub)
                ensure_dir(out_dir)
                out_name = os.path.splitext(fname)[0] + ".png"
                out_path = os.path.join(out_dir, out_name)
                img.save(out_path)
                processed += 1
            except Exception as e:
                errors += 1
                print(f"Error processing {in_path}: {e}")

    print(f"Done. processed={processed}, skipped={skipped}, errors={errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz images to PNG and organize by split.")
    parser.add_argument("--input-dir", default="/root/autodl-tmp/data/10k", help="Base dataset dir (expects 'All' under it).")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/data/10k", help="Base output dir; 'png' subdir will be created under this path.")
    parser.add_argument("--summary-csv", default="/root/autodl-tmp/data/10k/data_summary.csv", help="CSV file with 'filename' and 'use' columns.")
    parser.add_argument("--filter-file", default="/root/autodl-tmp/data/10k/filter_file.txt", help="Optional filter file with filenames to skip.")
    args = parser.parse_args()
    main(args)