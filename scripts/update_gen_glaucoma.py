#!/usr/bin/env python3
"""
Scan generated .npz files and set glaucoma / glaucoma_label based on filename ratio.

Usage:
    python scripts/update_gen_glaucoma.py /path/to/gen_data --min-ratio 0.6

This will only modify files whose filenames contain a `ratio_<float>` fragment.
If the ratio > min_ratio the script sets glaucoma=1, else glaucoma=0.
Both 'glaucoma' and 'glaucoma_label' keys will be written.
"""
import argparse
import os
import re
import tempfile
from glob import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Update glaucoma label in generated npz files based on filename ratio")
    parser.add_argument("--gen_data_dir",default="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data", help="Directory containing generated .npz files")
    parser.add_argument("--min-ratio", type=float, default=0.7, help="Threshold above which glaucoma=1 (default: 0.6)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes, only print what would be changed")
    return parser.parse_args()


RATIO_RE = re.compile(r"ratio_([0-9]*\.?[0-9]+)")


def extract_ratio_from_name(filename):
    m = RATIO_RE.search(filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def update_file(npz_path, value, ratio=None, dry_run=False):
    # load existing npz into a dict
    try:
        arrays = dict(np.load(npz_path, allow_pickle=True))
    except Exception as e:
        print(f"[ERROR] Failed to load '{npz_path}': {e}")
        return False

    # set both keys
    arrays["glaucoma"] = np.array([int(value)])
    # 写入杯盘比到 npz，键名为 cup_disc_ratio
    if ratio is not None:
        arrays["cup_disc_ratio"] = np.array(float(ratio))

    if dry_run:
        if ratio is not None:
            print(f"[DRY] Would set glaucoma={int(value)}, cup_disc_ratio={ratio} in {npz_path}")
        else:
            print(f"[DRY] Would set glaucoma={int(value)} in {npz_path}")
        return True

    # write to a temporary file then atomically replace
    dirpath = os.path.dirname(npz_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=dirpath)
    os.close(fd)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, npz_path)
        print(f"[OK] Updated {npz_path} -> glaucoma={int(value)}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write '{npz_path}': {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def main():
    args = parse_args()
    gen_dir = args.gen_data_dir
    if not os.path.isdir(gen_dir):
        print(f"Directory not found: {gen_dir}")
        return 1

    files = sorted(glob(os.path.join(gen_dir, "*.npz")))
    if not files:
        print(f"No .npz files found in {gen_dir}")
        return 0

    total = 0
    updated = 0
    for p in files:
        fname = os.path.basename(p)
        ratio = extract_ratio_from_name(fname)
        if ratio is None:
            # skip files without ratio in the name
            continue
        total += 1
        val = 1 if ratio > args.min_ratio else 0
        ok = update_file(p, val, ratio=ratio, dry_run=args.dry_run)
        if ok:
            updated += 1

    print(f"Processed {total} files with ratio; updated {updated} files (dry_run={args.dry_run})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


