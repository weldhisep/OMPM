# src/main.py

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model1_deterministic import solve_model1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["1"],
        default="1",
        help="Which model to run (1=deterministic)."
    )
    args = parser.parse_args()

    if args.model == "1":
        res = solve_model1()
    else:
        raise NotImplementedError("Only model 1 is implemented.")

    print("Results:", res)


if __name__ == "__main__":
    main()
