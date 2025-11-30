# src/models/model1_deterministic.py

from typing import Dict, Any, Optional
from pathlib import Path
import sys

import gurobipy as gp
from gurobipy import GRB

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_inputs, ModelType
from src.economics import present_value_factor
# src/models/model1_deterministic.py


def build_model1(inputs: Optional[Dict[str, Any]] = None) -> gp.Model:
    """
    MILP deterministic offshore wind model:

    - S chosen from a discrete set of turbine sizes (binary selection)
    - N = integer number of turbines
    - K = S * N (linear because S is defined as sum(s * y_s))

    Includes:
        - Wake losses (CF decreases with N)
        - Turbine-dependent CAPEX (per MW depends on chosen S)
        - Budget constraint
        - Linear NPV objective
    """

    if inputs is None:
        inputs = load_inputs(ModelType.MODEL1_DETERMINISTIC)

    econ = inputs["economics"]
    market = inputs["market"]

    base_cf = float(market["capacity_factor"])
    base_price = float(market["price_da"])
    pv_factor = present_value_factor(econ.discount_rate, econ.lifetime_years)

    fixed_opex = float(econ.fixed_opex_per_mw_year)
    variable_opex = float(econ.variable_opex_per_mwh)
    budget = float(econ.budget_eur)

    # ---- Available turbine sizes ----
    turbine_sizes = [12, 15, 18, 20]  # MW

    # CAPEX per MW depending on turbine size (higher S = cheaper)
    capex_map = {
        12: 3.0e6,
        15: 2.8e6,
        18: 2.6e6,
        20: 2.5e6,
    }

    wake_coeff = 0.0004  # CF loss per turbine

    m = gp.Model("model1_MILP")
    m.Params.MIPGap = 0.0

    # ---- Variables ----
    y = m.addVars(turbine_sizes, vtype=GRB.BINARY, name="y")  # size choice
    N = m.addVar(vtype=GRB.INTEGER, lb=0, name="N")           # number of turbines
    K = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="K")      # total MW

    # ---- Turbine size S derived from y ----
    S = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="S")

    m.addConstr(S == gp.quicksum(size * y[size] for size in turbine_sizes), "S_definition")

    # Must choose EXACTLY one turbine size
    m.addConstr(gp.quicksum(y[size] for size in turbine_sizes) == 1, "OneSize")

    # ---- Link capacity ----
    m.addConstr(K == N * S, "CapacityDefinition")

    # ---- Wake losses ----
    CF = m.addVar(lb=0.0, ub=1.0, name="CF")
    m.addConstr(CF == base_cf - wake_coeff * N, "WakeLoss")
    m.addConstr(CF >= 0.05, "MinCF")  # Prevent going too low

    # Annual MWh per MW
    annual_mwh = CF * 8760.0

    # ---- CAPEX ----
    capex_per_mw = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="CAPEX_per_MW")
    m.addConstr(
        capex_per_mw == gp.quicksum(capex_map[size] * y[size] for size in turbine_sizes),
        "CAPEX_selection"
    )

    # Budget: total CAPEX
    m.addConstr(capex_per_mw * K <= budget, "Budget")

    # ---- Revenue and OPEX ----
    annual_revenue = annual_mwh * base_price * K
    annual_opex = (fixed_opex + annual_mwh * variable_opex) * K
    total_capex = capex_per_mw * K

    # ---- NPV objective (linear because everything is linear in y, N, K) ----
    NPV = (annual_revenue - annual_opex) * pv_factor - total_capex
    m.setObjective(NPV, GRB.MAXIMIZE)

    return m


def solve_model1(inputs=None, solver_options=None):
    m = build_model1(inputs)

    if solver_options:
        for k, v in solver_options.items():
            m.setParam(k, v)

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed with status {m.Status}")

    # Extract chosen turbine size
    chosen_size = None
    for size in [12, 15, 18, 20]:
        if m.getVarByName(f"y[{size}]").X > 0.5:
            chosen_size = size

    return {
        "model": m,
        "K_opt_mw": m.getVarByName("K").X,
        "N_opt": m.getVarByName("N").X,
        "S_opt_mw": chosen_size,
        "CF_opt": m.getVarByName("CF").X,
        "NPV_eur": m.ObjVal,
    }
