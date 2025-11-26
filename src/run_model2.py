from models.model2_multi_period import build_and_solve_model2

def main():
    """
    Executes the Multi-Period Optimization Model (Model 2) and prints the key results.
    """
    print("--- Starting Model 2: Multi-Period Wind Farm Optimization ---")
    
    # Run the model with default inputs
    results = build_and_solve_model2()
    
    # Check if the optimization was successful and extract results
    C_opt = results.get('C_opt_mw', 0.0)
    NPV_eur = results.get('NPV_eur', 0.0)
    
    print("\n--- Final Results Summary ---")
    print(f"Optimal Installed Capacity (C*): {C_opt:,.2f} MW")
    print(f"Project Net Present Value (NPV): {NPV_eur:,.0f} EUR")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()