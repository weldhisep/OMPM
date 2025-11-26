# src/run_model3.py

from models.model3_stochastic import solve_model3
from data_loader import load_inputs, ModelType

def main():

    # Load inputs for Model 3
    inputs = load_inputs(ModelType.MODEL3_STOCHASTIC)

    # Solve the model
    results = solve_model3(inputs)

    print("\n====== Model 3 Stochastic Results ======\n")
    print(f"Optimal capacity K*     = {results['K_opt_mw']:.2f} MW")
    print(f"Optimal turbine size S* = {results['S_opt_mw']:.2f} MW")
    print(f"Optimal number N*       = {results['N_opt']:.2f}")
    print(f"Project NPV             = {results['NPV_eur']:.2f} EUR")

    # CVaR / risk metrics
    print(f"VaR (zeta)              = {results['zeta']:.2f}")

if __name__ == "__main__":
    main()
