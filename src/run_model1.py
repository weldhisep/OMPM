# src/run_model1.py

from pathlib import Path
import sys

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model1_deterministic import solve_model1


def main():
    results = solve_model1()
    print(f"Optimal capacity K* = {results['K_opt_mw']:.2f} MW")
    print(f"Turbine size S*      = {results['S_opt_mw']:.2f} MW")
    print(f"Number of turbines N* = {results['N_opt']:.2f}")
    print(f"Project NPV          = {results['NPV_eur']:.2f} EUR")


if __name__ == "__main__":
    main()
