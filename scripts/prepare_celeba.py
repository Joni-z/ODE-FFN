"""Download via torchvision's CelebA implementation and verify the raw dataset.

No training, image resizing, split changes or generation metrics are performed.
Original download IDs and MD5s come from the installed torchvision release.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
            "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset parent; images go under ROOT/celeba/img_align_celeba")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    folder = args.root.resolve() / "celeba"
    status_path = folder / "preparation-status.json"
    started = time.time()

    def status(phase, **details):
        value = {"phase": phase, "elapsed_seconds": round(time.time() - started, 2),
                 "dataset": str(folder), **details}
        write_json(status_path, value)
        print(json.dumps(value), flush=True)

    try:
        status("importing")
        from PIL import Image
        import torch
        import torchvision
        from torchvision.datasets import CelebA

        torch.set_num_threads(1)
        status("official_download" if not args.verify_only else "loading")
        dataset = CelebA(args.root, split="all", target_type="attr",
                         download=not args.verify_only)
        if len(dataset) != 202599 or tuple(dataset.attr.shape) != (202599, 40):
            raise RuntimeError("Unexpected official CelebA image/attribute count")

        status("hashing_source_files")
        files = {}
        for file_id, expected_md5, name in CelebA.file_list:
            path = folder / name
            if not path.is_file():
                raise RuntimeError(f"Missing original dataset file: {path}")
            digest = hashes(path)
            if digest["md5"] != expected_md5:
                raise RuntimeError(f"Official checksum mismatch for {name}")
            files[name] = {**digest, "expected_md5": expected_md5,
                           "google_drive_id": file_id}

        status("checking_archive_crc")
        with zipfile.ZipFile(folder / "img_align_celeba.zip") as archive:
            broken = archive.testzip()
            if broken is not None:
                raise RuntimeError(f"ZIP CRC mismatch: {broken}")
            archived_names = {Path(item.filename).name for item in archive.infolist()
                              if not item.is_dir() and item.filename.endswith(".jpg")}

        status("checking_images_and_splits")
        expected_names = {f"{i:06d}.jpg" for i in range(1, 202600)}
        image_folder = folder / "img_align_celeba"
        actual_names = {path.name for path in image_folder.iterdir()
                        if path.suffix == ".jpg"}
        if archived_names != expected_names or actual_names != expected_names:
            raise RuntimeError("Archive/extracted filenames do not match all 202599 images")
        if set(dataset.filename) != expected_names:
            raise RuntimeError("Official metadata filenames disagree with image files")
        splits = Counter()
        with (folder / "list_eval_partition.txt").open() as stream:
            for line in stream:
                _, split = line.split()
                splits[int(split)] += 1
        if dict(splits) != {0: 162770, 1: 19867, 2: 19962}:
            raise RuntimeError(f"Unexpected official split counts: {dict(splits)}")

        # Fixed positions spanning the dataset; this is a decode sample, not a
        # claim that every extracted JPEG was independently decoded twice.
        indices = [i * (len(dataset) - 1) // 31 for i in range(32)]
        decode_checks = []
        for index in indices:
            path = image_folder / dataset.filename[index]
            with Image.open(path) as image:
                image.verify()
            image, target = dataset[index]
            image.load()
            if image.size != (178, 218) or image.mode != "RGB" or tuple(target.shape) != (40,):
                raise RuntimeError(f"Unexpected decoded image/target: {path.name}")
            decode_checks.append({"filename": path.name, "size": list(image.size),
                                  "mode": image.mode})

        receipt = {"status": "complete", "dataset": "CelebA aligned and cropped",
                   "root": str(folder), "images": len(dataset),
                   "splits": {"train": splits[0], "valid": splits[1], "test": splits[2]},
                   "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "python": sys.version, "files": files, "archive_crc": "passed",
                   "filename_sets": "passed", "decode_sample": decode_checks,
                   "preprocessing": "none; original aligned JPEGs retained",
                   "elapsed_seconds": round(time.time() - started, 2)}
        write_json(folder / "preparation.json", receipt)
        status("complete", images=len(dataset), receipt=str(folder / "preparation.json"))
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
