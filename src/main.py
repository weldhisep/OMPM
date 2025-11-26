import argparse
from models.model1_deterministic import build_and_solve_model1
from models.model2_multi_period import build_and_solve_model2
from models.model3_stochastic import build_and_solve_model3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["1", "2", "3"],
        default="1",
        help="Which model to run (1=deterministic, 2=multi-period, 3=stochastic)."
    )
    args = parser.parse_args()

    # Note: Model 1's function is currently called 'solve_model1' in your files,
    # but the import here refers to the function name inside the file. 
    # Let's ensure consistency by using 'build_and_solve_modelX' pattern.
    
    if args.model == "1":
        res = build_and_solve_model1()
    elif args.model == "2":
        res = build_and_solve_model2()
    else:
        res = build_and_solve_model3()

    print("Final Model Results:", res)


if __name__ == "__main__":
    main()