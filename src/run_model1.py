# src/run_model1.py

from src.models.model1_deterministic import solve_model1

def main():
    results = solve_model1(solver_name="gurobi")
    print(f"Optimal capacity K* = {results['K_opt_mw']:.2f} MW")
    print(f"Project NPV       = {results['NPV_eur']:.2f} EUR")

if __name__ == "__main__":
    main()
