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



def build_model1(inputs: Optional[Dict[str, Any]] = None) -> gp.Model:
    """
    High-realism deterministic offshore wind model (quadratic, Gurobi-compatible):

    Includes:
        - Wake losses:      CF = base_cf - wake_coeff * N
        - Turbine CAPEX:    capex_per_mw = base_cost + slope*(20 - S)
        - Capacity link:    K = N * S
        - Budget constraint
        - NPV objective (quadratic)

    Excludes:
        - Cable cost
        - Price depression (removed to avoid cubic terms)
    """

    # ----------------------------
    # Load inputs
    # ----------------------------
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

    # ----------------------------
    # Create model
    # ----------------------------
    m = gp.Model("model1_high_realism_quadratic")
    m.Params.NonConvex = 2     # enable bilinear/quadratic constraints

    # ----------------------------
    # Turbine size bounds (10–25 MW)
    # ----------------------------
    S_min = 10.0
    S_max = 25.0

    # ----------------------------
    # Decision variables
    # ----------------------------
    S = m.addVar(lb=S_min, ub=S_max, vtype=GRB.CONTINUOUS, name="S")  # turbine size [MW]
    N = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="N")              # number of turbines
    K = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K")              # total capacity [MW]

    # ----------------------------
    # Constraint: K = N * S
    # ----------------------------
    m.addQConstr(K == N * S, name="CapacityDefinition")

    # ----------------------------
    # Wake losses: CF(N)
    # CF = base_cf - wake_coeff * N
    # ----------------------------
    wake_coeff = 0.0004      # ~0.04% loss per turbine
    CF = base_cf - wake_coeff * N

    # prevent CF < 0
    m.addConstr(CF >= 0.05, name="MinCF")

    # annual production per MW
    annual_mwh = CF * 8760.0

    # ----------------------------
    # Turbine-dependent CAPEX per MW
    # larger S → cheaper per MW
    # ----------------------------
    base_capex = 2.8e6       # cost at 20 MW turbine
    capex_slope = 1.4e5      # cost difference per MW relative to 20 MW

    # CAPEX per MW is linear in S → ok
    capex_per_mw = base_capex + capex_slope * (20.0 - S)

    # Budget: total CAPEX <= Budget
    m.addConstr(capex_per_mw * K <= budget, name="Budget")

    # ----------------------------
    # Annual OPEX (per MW)
    # ----------------------------
    annual_opex_per_mw = fixed_opex + annual_mwh * variable_opex

    # ----------------------------
    # Annual revenue:
    #    revenue = annual_mwh * base_price * K
    #
    # This keeps the objective quadratic (CF*N*K is bilinear)
    # ----------------------------
    annual_revenue = annual_mwh * base_price * K

    # ----------------------------
    # Total OPEX (and total CAPEX)
    # ----------------------------
    total_opex = annual_opex_per_mw * K
    total_capex = capex_per_mw * K

    # ----------------------------
    # Objective: maximize NPV
    #
    # NPV = PV(annual_revenue - annual_opex) - CAPEX
    # ----------------------------
    NPV = (annual_revenue - total_opex) * pv_factor - total_capex

    m.setObjective(NPV, GRB.MAXIMIZE)

    return m


def solve_model1(
    inputs: Optional[Dict[str, Any]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    m = build_model1(inputs)

    if solver_options:
        for key, val in solver_options.items():
            m.setParam(key, val)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi solver ended with status {m.Status}")

    return {
        "model": m,
        "K_opt_mw": m.getVarByName("K").X,
        "S_opt_mw": m.getVarByName("S").X,
        "N_opt": m.getVarByName("N").X,
        "NPV_eur": m.ObjVal,
    }
