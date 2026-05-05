"""Run every (task, backbone) pair end-to-end. Cross-platform launcher."""
import subprocess
import sys
import time

TASKS = ["cladding", "stories"]
MODELS = ["resnet50", "effnetv2s", "convnext_tiny", "vit_b16"]


def main():
    failed = []
    start = time.time()
    for task in TASKS:
        for model in MODELS:
            label = f"{task} / {model}"
            print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}", flush=True)
            cmd = [sys.executable, "-m", "src.cli", "train", "--task", task, "--model", model]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"!!! FAILED: {label}", flush=True)
                failed.append(label)
    dt = time.time() - start
    print(f"\nTotal wall-clock: {dt / 60:.1f} min")
    if failed:
        print(f"Failed runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
