#!/usr/bin/env python3
"""Check whether a checkpoint contains optimizer/scheduler states."""

import argparse
import sys
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect checkpoint keys")
    parser.add_argument("path", help="Path to checkpoint .pth file")
    args = parser.parse_args()

    checkpoint = torch.load(args.path, map_location="cpu")

    print(f"Loaded: {args.path}")
    if isinstance(checkpoint, dict):
        keys = sorted(checkpoint.keys())
        print("Top-level keys:")
        for key in keys:
            print(f"- {key}")

        print("\nPresence checks:")
        print(f"optimizer: {'optimizer' in checkpoint}")
        print(f"scheduler: {'scheduler' in checkpoint}")
    else:
        print(f"Unexpected checkpoint type: {type(checkpoint)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
