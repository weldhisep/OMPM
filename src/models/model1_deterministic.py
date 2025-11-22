# src/models/model1_deterministic.py

from typing import Dict, Any, Optional

import gurobipy as gp
from gurobipy import GRB

from data_loader import load_inputs, ModelType
from economics import present_value_factor


def build_model1(inputs: Optional[Dict[str, Any]] = None) -> gp.Model:
    """
    Deterministic Model 1 with turbine size and number of turbines in Gurobi.

    VARIABLES:
        S: turbine size (MW per turbine)
        N: number of turbines (continuous for now)
        K: installed capacity (MW)

    CONSTRAINTS:
        1) K = N * S
        2) CAPEX_per_MW * K <= Budget

    OBJECTIVE:
        Maximize NPV = PV(annual margin * K) - CAPEX_per_MW * K
    """

    # ---- Load inputs ----
    if inputs is None:
        inputs = load_inputs(ModelType.MODEL1_DETERMINISTIC)

    econ = inputs["economics"]
    market = inputs["market"]

    pv_factor = present_value_factor(econ.discount_rate, econ.lifetime_years)

    capex_per_mw = float(econ.capex_per_mw)
    fixed_opex_per_mw_year = float(econ.fixed_opex_per_mw_year)
    variable_opex_per_mwh = float(econ.variable_opex_per_mwh)
    budget_eur = float(econ.budget_eur)
    annual_mwh_per_mw = float(market["annual_mwh_per_mw"])
    price_da = float(market["price_da"])

    # --- Create model ---
    m = gp.Model("model1_deterministic")

    # Allow non-convex quadratic due to K = N * S
    m.Params.NonConvex = 2

    # ---- Parameters for turbine size bounds (can move to config later) ----
    S_min = 8.0   # MW
    S_max = 25.0  # MW

    # ---- Decision variables ----
    S = m.addVar(lb=S_min, ub=S_max, vtype=GRB.CONTINUOUS, name="S")  # turbine size [MW]
    N = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N")              # number of turbines
    K = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K")              # total capacity [MW]

    # ---- Constraints ----

    # Capacity definition: K = N * S  (quadratic)
    m.addQConstr(K == N * S, name="CapacityDefinition")

    # Budget constraint: CAPEX * K <= Budget
    m.addConstr(capex_per_mw * K <= budget_eur, name="Budget")

    # ---- Objective ----

    # Annual margin per MW
    annual_revenue_per_mw = annual_mwh_per_mw * price_da
    annual_opex_per_mw = (
        fixed_opex_per_mw_year +
        annual_mwh_per_mw * variable_opex_per_mwh
    )
    annual_margin_per_mw = annual_revenue_per_mw - annual_opex_per_mw

    # NPV per MW over lifetime
    npv_per_mw = annual_margin_per_mw * pv_factor - capex_per_mw

    # Total NPV
    m.setObjective(npv_per_mw * K, GRB.MAXIMIZE)

    return m


def solve_model1(
    inputs: Optional[Dict[str, Any]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build and solve Model 1 with Gurobi.
    Returns a dict with optimal values and the Gurobi model object.
    """

    m = build_model1(inputs)

    # Extra options if needed
    if solver_options:
        for k, v in solver_options.items():
            m.setParam(k, v)

    m.optimize()

    # Basic status check
    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi ended with status {m.Status}")

    K_opt = m.getVarByName("K").X
    S_opt = m.getVarByName("S").X
    N_opt = m.getVarByName("N").X
    obj_val = m.ObjVal

    return {
        "model": m,
        "K_opt_mw": K_opt,
        "S_opt_mw": S_opt,
        "N_opt": N_opt,
        "NPV_eur": obj_val,
    }
