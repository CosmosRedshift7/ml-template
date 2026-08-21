import argparse
import shutil
from pathlib import Path

TARGETS = {
    "runs": (Path("local/aim"),),
    "checkpoints": (Path("local/checkpoints"),),
    "figures": (Path("local/figures"),),
    "all": (
        Path("local/aim"),
        Path("local/checkpoints"),
        Path("local/figures"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated project outputs.")
    parser.add_argument("target", choices=TARGETS)
    args = parser.parse_args()

    for path in TARGETS[args.target]:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
