from models.model2_multi_period import build_and_solve_model2

def main():
    """
    Executes the Multi-Period Optimization Model (Model 2) and prints the key results.
    """
    print("--- Starting Model 2: Multi-Period Wind Farm Optimization ---")
    
    # Run the model with default inputs
    results = build_and_solve_model2()
    


if __name__ == "__main__":
    main()