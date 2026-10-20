#!/usr/bin/env python3
"""Generate FID reference statistics for ImageNet (256) used by configs.

Usage (from repo root):
  python src/util/generate_fid_stats_imagenet.py
  python src/util/generate_fid_stats_imagenet.py --image-dir /path/to/imagenet/val --output fid_stats/jit_in256_stats.npz

Uses train by default; if you have val extracted (e.g. from imagenet-val.sqf), pass --image-dir .../val.
"""
import argparse
import os
import sys

# repo root (script lives in src/util/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_script_dir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from util.fid import save_fid_stats

IMAGENET_DEFAULT = "/projects/work/public/ml-datasets/imagenet"
OUTPUT_DEFAULT = "fid_stats/jit_in256_stats.npz"


def main():
    p = argparse.ArgumentParser(description="Generate fid_stats/jit_in256_stats.npz from ImageNet images")
    p.add_argument(
        "--image-dir",
        type=str,
        default=os.path.join(IMAGENET_DEFAULT, "train"),
        help="Directory of reference images (e.g. train or val); default: %(default)s",
    )
    p.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DEFAULT,
        help="Output .npz path; default: %(default)s",
    )
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--dims", type=int, default=2048)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    args = p.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: image dir not found: {args.image_dir}")
        sys.exit(1)

    out_path = args.output
    if os.path.isabs(out_path):
        out_dir = os.path.dirname(out_path)
    else:
        out_dir = os.path.join(REPO_ROOT, os.path.dirname(out_path))
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        if not args.overwrite:
            print(f"Output exists: {out_path}. Use --overwrite to regenerate.")
            sys.exit(0)
        os.remove(out_path)
        print(f"Removed existing {out_path}")

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.num_workers is None:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 0
        num_workers = min(n, 8) if n else 0
    else:
        num_workers = args.num_workers

    save_fid_stats(
        [args.image_dir, out_path],
        batch_size=args.batch_size,
        device=device,
        dims=args.dims,
        num_workers=num_workers,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
