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
    """
    Multi-Period Model 2: Optimizes the one-time installed capacity (C)
    based on annual time-series data for wind production (W_F, W_R) and prices (P_DA, P_BAL).
    
    The objective maximizes the Net Present Value (NPV) over the project lifetime.
    
    NOTE: This model assumes the Day-Ahead commitment is implicitly equal to the 
    forecasted production (C * W_F[t]).

    VARIABLES:
        C: Installed Capacity (MW) - continuous decision variable
    
    CONSTRAINTS:
        1) CAPEX * C <= Budget
        
    OBJECTIVE (Maximize NPV):
        MAX: SUM_t [ (R^DA_t + R^B_t - K_t) / (1 + r)^t ] - COST^CAPEX
    """

    # ---- Load inputs ----
    # This check is crucial for running 'build_model2' directly
    if inputs is None:
        inputs = load_inputs(ModelType.MODEL2_MULTI_PERIOD)

    # Sanity check for data loading success
    if not isinstance(inputs, dict):
         raise RuntimeError("Data loading failed: 'inputs' is not a dictionary. Check load_inputs() in data_loader.py.")

    econ = inputs["economics"]
    market = inputs["market"]

    # Economic Parameters (Constants)
    r = econ.discount_rate
    
    # Check for T_periods to ensure market data is complete
    if "T_periods" not in market:
        # If the market data stub is incomplete, this will be a useful error
        raise KeyError("Market data is missing required key 'T_periods'. Please complete _load_market_data_model2() in data_loader.py.")

    T_periods = market["T_periods"]
    
    capex_per_mw = float(econ.capex_per_mw)
    fixed_opex_per_mw_year = float(econ.fixed_opex_per_mw_year)
    variable_opex_per_mwh = float(econ.variable_opex_per_mwh)
    budget_eur = float(econ.budget_eur)

    # Market Data (Time-indexed dictionaries)
    W_F = market["W_F"]     # Forecasted MWh/MW/period (for DA bid)
    W_R = market["W_R"]     # Realized MWh/MW/period (for Imbalance)
    P_DA = market["P_DA"]   # Day-Ahead Price (EUR/MWh)
    P_BAL = market["P_BAL"] # Balancing Price (EUR/MWh)

    # ---- Model Setup ----
    m = gp.Model("Model2_Multi_Period")
    # Silence output for cleaner console
    m.setParam('OutputFlag', 0) 

    # ---- Decision Variables ----
    # C: Installed Capacity (MW)
    # The capacity is a continuous variable, fixed for all time periods.
    C = m.addVar(lb=0, name="Installed_Capacity_MW", vtype=GRB.CONTINUOUS)

    # ---- Constraints ----
    # 1. Budget constraint: CAPEX * C <= Budget
    m.addConstr(capex_per_mw * C <= budget_eur, name="Budget_Limit")

    # ---- Objective Calculation (NPV) ----

    # 1. CAPEX Cost (occurs at t=0, no discounting)
    capex_cost = capex_per_mw * C

    # 2. Annual Operating Margin (Per Period Cash Flow)
    npv_sum = 0.0

    # Iterate over all periods (t=1 to T_periods)
    for t in range(1, T_periods + 1):
        # A. Revenues from Day-Ahead (DA) Market (Assuming commitment = forecast)
        # R^DA_t = P^DA_t * C * W^F_t
        revenue_da = P_DA[t] * C * W_F[t]

        # B. Revenue/Cost from Balancing (B) Market
        # R^B_t = P^B_t * C * (W^R_t - W^F_t)
        # Note: (W^R_t - W^F_t) is the imbalance per MW. If negative, R^B_t is a cost.
        imbalance_per_mw = W_R[t] - W_F[t]
        revenue_bal = P_BAL[t] * C * imbalance_per_mw

        # C. Operating Costs (OPEX)
        # Fixed OPEX (per MW of capacity) * C
        fixed_opex = fixed_opex_per_mw_year * C
        # Variable OPEX (per MWh of realized production) * C * W_R_t
        variable_opex = variable_opex_per_mwh * C * W_R[t]
        
        # Total OPEX for period t
        operating_cost = fixed_opex + variable_opex
        
        # Net Cash Flow for Period t
        net_cash_flow_t = revenue_da + revenue_bal - operating_cost

        # Discount Factor for Period t: 1 / (1 + r)^t
        discount_factor_t = 1.0 / ((1.0 + r) ** t)
        
        # Add Present Value of Cash Flow to the NPV sum
        npv_sum += net_cash_flow_t * discount_factor_t

    # Total NPV = SUM(Discounted Cash Flows) - CAPEX Cost
    total_npv = npv_sum - capex_cost
    
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
    
    The 'inputs' dict is loaded here if not provided, to ensure data is available 
    for the results summary.
    """
    
    # 1. Load Inputs 
    # We load the inputs here so they are available for printing the results at the end.
    if inputs is None:
        inputs = load_inputs(ModelType.MODEL2_MULTI_PERIOD)
    
    # CRITICAL: If data_loader.py fails to return a dictionary, we must stop here.
    if inputs is None or not isinstance(inputs, dict):
        raise RuntimeError(
            "Data loading failed: 'inputs' is None or not a dictionary. "
            "Please check the implementation of load_inputs() in data_loader.py "
            "to ensure it correctly returns a dictionary object."
        )


    # 2. Build Model (pass loaded inputs)
    m = build_model2(inputs)

    # 3. Extra options if needed
    if solver_options:
        for k, v in solver_options.items():
            m.setParam(k, v)
    
    # 4. Solve the model
    m.optimize()

    # ---- Process Results ----
    
    if m.status == GRB.OPTIMAL:
        # Get optimal capacity
        C_opt = m.getVarByName("Installed_Capacity_MW").X
        
        # Total NPV (Objective value)
        npv_eur = m.objVal
        
        # Calculate the CAPEX used
        # This access is now safe because we checked 'inputs' is a dict above.
        capex_per_mw = inputs["economics"].capex_per_mw 
        capex_used = C_opt * capex_per_mw
        
        # Print Summary
        print(f"\n--- Model 2 Solution ({inputs['market']['T_periods']} periods) ---")
        print(f"Status: {m.status} (OPTIMAL)")
        print(f"Optimal Installed Capacity (C*): {C_opt:,.2f} MW")
        print(f"Project NPV (Max Profit): {npv_eur:,.0f} EUR")
        print(f"Total CAPEX used: {capex_used:,.0f} EUR")
        print(f"Budget: {inputs['economics'].budget_eur:,.0f} EUR")
        print("------------------------------------------\n")


        results = {
            "C_opt_mw": C_opt,
            "NPV_eur": npv_eur,
            "Gurobi_Model": m
        }
    else:
        print(f"Optimization failed with status {m.status}. See Gurobi documentation for status codes.")
        results = {
            "C_opt_mw": 0.0,
            "NPV_eur": 0.0,
            "Gurobi_Model": m
        }

    return results

# If running this file directly (e.g., for quick testing)
if __name__ == "__main__":
    # Test the model with default inputs
    results = build_and_solve_model2()
    print(results)