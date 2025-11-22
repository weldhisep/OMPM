# src/run_model1.py

from models.model1_deterministic import solve_model1

def main():
    results = solve_model1()
    print(f"Optimal capacity K* = {results['K_opt_mw']:.2f} MW")
    print(f"Turbine size S*      = {results['S_opt_mw']:.2f} MW")
    print(f"Number of turbines N* = {results['N_opt']:.2f}")
    print(f"Project NPV          = {results['NPV_eur']:.2f} EUR")

if __name__ == "__main__":
    main()
