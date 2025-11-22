# src/main.py

import argparse
from src.models.model1_deterministic import build_and_solve_model1
from src.models.model2_multi_period import build_and_solve_model2
from src.models.model3_stochastic import build_and_solve_model3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["1", "2", "3"],
        default="1",
        help="Which model to run (1=deterministic, 2=multi-period, 3=stochastic)."
    )
    args = parser.parse_args()

    if args.model == "1":
        res = build_and_solve_model1()
    elif args.model == "2":
        res = build_and_solve_model2()
    else:
        res = build_and_solve_model3()

    print("Results:", res)

if __name__ == "__main__":
    main()
