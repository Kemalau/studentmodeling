#!/usr/bin/env python3

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    files = [path for path in args.source.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"Copying {len(files)} model files ({total_bytes / 2**30:.1f} GiB) "
        f"with {args.workers} workers",
        flush=True,
    )

    def copy_file(path):
        target = args.destination / path.relative_to(args.source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return path.stat().st_size, path.name

    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(copy_file, path) for path in files]
        for future in as_completed(futures):
            size, name = future.result()
            copied_bytes += size
            print(
                f"Copied {copied_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB: {name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
