# src/models/model2_multi_period.py

import sys
from typing import Dict, Any, Optional
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_inputs, ModelType
from src.economics import present_value_factor


def build_model2(inputs: Optional[Dict[str, Any]] = None) -> gp.Model:
    # ---- Load inputs ----
    # This check is crucial for running 'build_model2' directly
    if inputs is None:
        inputs = load_inputs(ModelType.MODEL2_MULTI_PERIOD)

    econ = inputs["economics"]
    market = inputs["market"]

    # Economic Parameters (Constants)
    r = econ.discount_rate
    base_cf = float(market["capacity_factor"])

    
    T_periods = market["T_periods"]
    
    fixed_opex_per_mw_year = float(econ.fixed_opex_per_mw_year)
    variable_opex_per_mwh = float(econ.variable_opex_per_mwh)
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
    
    # Market Data (Time-indexed dictionaries)
    W_F = market["W_F"]     # Forecasted MWh/MW/period (for DA bid)
    W_R = market["W_R"]     # Realized MWh/MW/period (for Imbalance)
    P_DA = market["P_DA"]   # Day-Ahead Price (EUR/MWh)
    P_BAL = market["P_BAL"] # Balancing Price (EUR/MWh)

    # ---- Model Setup ----
    m = gp.Model("Model2_Multi_Period")
    m.setParam('OutputFlag', 0) 

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


    # ---- Constraints ----
    # 1. Budget constraint: CAPEX * C <= Budget
    capex_per_mw = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="CAPEX_per_MW")
    m.addConstr(
        capex_per_mw == gp.quicksum(capex_map[size] * y[size] for size in turbine_sizes),
        "CAPEX_selection"
    )
    # Budget: total CAPEX
    total_capex = capex_per_mw * K
    m.addConstr(capex_per_mw * K <= budget, "Budget")

    # ---- Objective Calculation (NPV) ----

    npv_sum = 0.0

    # Iterate over all periods (t=1 to T_periods)
    for t in range(1, T_periods + 1):
        scaling_factor = CF / base_cf  # Adjust production for wake losses
        annual_mwh_per_mw_adj_F = W_F[t] * scaling_factor # Adjusted forecasted production
        annual_mwh_per_mw_adj_R = W_R[t] * scaling_factor # Adjusted realized production
        discount_factor = 1.0 / ((1.0 + r) ** t) # Discount factor for period t        
        revenue_da = P_DA[t] * annual_mwh_per_mw_adj_F * K # Revenue from Day-Ahead market
        revenue_bal = P_BAL[t] * (annual_mwh_per_mw_adj_R - annual_mwh_per_mw_adj_F) * K # Revenue from Balancing market
        opex_fixed = fixed_opex_per_mw_year * K # Fixed OPEX is based on installed capacity
        opex_variable = variable_opex_per_mwh * annual_mwh_per_mw_adj_R * K # Variable OPEX is based on realized MWh
        net_cash_flow_t = revenue_da + revenue_bal - (opex_fixed + opex_variable) # Net cash flow for period t
        npv_sum += net_cash_flow_t * discount_factor # Add discounted cash flow to NPV sum


    # Total NPV = SUM(Discounted Cash Flows) - CAPEX Cost
    total_npv = npv_sum - total_capex 
    
    # Set Objective
    m.setObjective(total_npv, GRB.MAXIMIZE)

    return m


def build_and_solve_model2(
    inputs: Optional[Dict[str, Any]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build and solve Model 2 with Gurobi.
    Returns a dict with optimal values and the Gurobi model object.
    """
    if inputs is None:
        inputs = load_inputs(ModelType.MODEL2_MULTI_PERIOD)
    
#   If data_loader.py fails to return a dictionary, we must stop here.
    if inputs is None or not isinstance(inputs, dict):
        raise RuntimeError(
            "Data loading failed: 'inputs' is None or not a dictionary. "
            "Please check the implementation of load_inputs() in data_loader.py "
            "to ensure it correctly returns a dictionary object."
        )


    m = build_model2(inputs)

    if solver_options:
        for k, v in solver_options.items():
            m.setParam(k, v)
    
    m.optimize()
    results = {}
    if m.status == GRB.OPTIMAL:
        # Get optimal capacity
        C_opt = m.getVarByName("K").X
        N_opt = m.getVarByName("N").X
        CF_opt = m.getVarByName("CF").X
        
        turbine_size = [12, 15, 18, 20]
        chosen_size = None
        for size in turbine_size:
            if m.getVarByName(f"y[{size}]").X > 0.5:
                chosen_size = size
                break
        
        npv_eur = m.objVal
        capex_per_mw = m.getVarByName("CAPEX_per_MW").X 
        capex_used = C_opt * capex_per_mw
        
        # Print Summary
        print(f"\n--- Model 2 Solution ({inputs['market']['T_periods']} periods) ---")
        print(f"Status: {m.status} (OPTIMAL)")
        print(f"Optimal Installed Capacity (K*): {C_opt:,.2f} MW")
        print(f"Optimal Number of Turbines (N*): {N_opt:,.2f} units")
        print(f"Optimal Turbine Size (S*): {chosen_size} MW")
        print(f"Optimal Capacity Factor (CF*): {CF_opt:.4f}")
        print(f"Project NPV (Max Profit): {npv_eur:,.0f} EUR")
        print(f"Total CAPEX used: {capex_used:,.0f} EUR")
        print(f"Budget: {inputs['economics'].budget_eur:,.0f} EUR")
        print("------------------------------------------\n")


        results = {
            "K_opt_mw": C_opt,
            "N_opt": N_opt,
            "S_opt_mw": chosen_size,
            "CF_opt": CF_opt,
            "NPV_eur": npv_eur,
            "Gurobi_Model": m
        }
    else:
        print(f"Optimization failed with status {m.Status}. See Gurobi documentation for status codes.")
        results = {
            "K_opt_mw": 0.0,
            "NPV_eur": 0.0,
            "Gurobi_Model": m
        }

    return results

# If running this file directly (e.g., for quick testing)
if __name__ == "__main__":
    # Test the model with default inputs
    results = build_and_solve_model2()
    print(results)