import argparse
# Import the new model function
from models.model1_deterministic import build_and_solve_model1
from models.model2_multi_period import build_and_solve_model2
# from models.model3_stochastic import build_and_solve_model3 # Keep commented until ready

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["1", "2", "3"],
        default="2", # Default to Model 2 now
        help="Which model to run (1=deterministic, 2=multi-period, 3=stochastic).",
    )
    args = parser.parse_args()

    # Note: Model 1's function is currently called 'solve_model1' in your files,
    # but the import here refers to the function name inside the file. 
    # Let's ensure consistency by using 'build_and_solve_modelX' pattern.
    
    if args.model == "1":
        # The original model file seems to use 'solve_model1' for the wrapper function
        # I'll stick to the naming convention established in my previous file:
        # res = solve_model1() # assuming this is the full function in model1_deterministic.py
        # For this exercise, I will assume the provided function 'build_and_solve_model1' exists 
        # in the model1_deterministic.py file.
        res = build_and_solve_model1() 
        
    elif args.model == "2":
        res = build_and_solve_model2()
        
    elif args.model == "3":
        print("Model 3 (Stochastic) is not yet implemented.")
        # res = build_and_solve_model3() 
        res = {}
    
    else:
        print(f"Model {args.model} is not recognized.")
        res = {}

    print("Final Model Results:", res)

if __name__ == "__main__":
    main()