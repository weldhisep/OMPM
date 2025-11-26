# src/models/model3_stochastic.py

from typing import Dict, Any, Optional

import gurobipy as gp
from gurobipy import GRB

from data_loader import load_inputs, ModelType
from economics import present_value_factor


def build_model3(inputs: Optional[Dict[str, Any]] = None) -> gp.Model:
    """
    Stochastic Model 3 with turbine size and number of turbines (N & S)
    and CVaR-based risk aversion.

    VARIABLES:
        S: turbine size (MW per turbine)
        N: number of turbines (continuous)
        K: installed capacity (MW)
        g[t,s]: realized generation (MW)
        xi[s]: scenario profit (EUR/year)
        zeta: VaR threshold
        eta[s]: CVaR tail variable

    OBJECTIVE:
        Maximize:  sum_s pi[s] * xi[s] 
                   - CAPEX*K
                   - lambda * CVaR_alpha(loss)

    CONSTRAINTS:
        K = N * S  (nonconvex quadratic)
        CAPEX * K <= Budget
        0 <= g[t,s] <= K * rho[t,s]
        eta_s >= -xi_s - zeta
        eta_s >= 0
    """

    # -------------------------------------------------------
    # Load inputs
    # -------------------------------------------------------
    if inputs is None:
        inputs = load_inputs(ModelType.MODEL3_STOCHASTIC)

    econ = inputs["economics"]
    market = inputs["market"]
    scenarios = market["scenarios"]   # scenario dict with arrays


    # scenario time series
    rho = scenarios["rho"]                # capacity factors [T x S]
    price_da = scenarios["price_da"]      # day-ahead price [T x S]
    price_bal = scenarios["price_bal"]    # balancing price [T x S]
    pi = scenarios["probabilities"]       # probabilities [S]

    T = rho.shape[0]
    S_count = rho.shape[1]

    # economics
    pv_factor = present_value_factor(econ.discount_rate, econ.lifetime_years)

    capex_per_mw = float(econ.capex_per_mw)
    fixed_opex_per_mw_year = float(econ.fixed_opex_per_mw_year)
    variable_opex_per_mwh = float(econ.variable_opex_per_mwh)
    budget_eur = float(econ.budget_eur)

    lambda_risk = float(econ.lambda_risk)      # CVaR weight
    alpha = float(econ.cvar_alpha)

    # for annual OPEX
    # fixed OPEX per MW-year
    # variable OPEX handled via generation * variable_opex_per_mwh

    # bidding scheme: forecast-based bid
    rho_forecast = scenarios["rho_forecast"]    # T x S forecasted CF used for DA bid

    # -------------------------------------------------------
    # Create model
    # -------------------------------------------------------
    m = gp.Model("model3_stochastic")

    # Allow quadratic non-convex K = N * S
    m.Params.NonConvex = 2

    # Turbine size bounds
    S_min = 8.0
    S_max = 25.0

    # -------------------------------------------------------
    # Variables
    # -------------------------------------------------------
    S_var = m.addVar(lb=S_min, ub=S_max, vtype=GRB.CONTINUOUS, name="S")
    N_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N")
    K_var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K")

    # Scenario generation variables
    g = {}
    xi = {}
    eta = {}

    for s in range(S_count):
        xi[s] = m.addVar(lb=-GRB.INFINITY, name=f"xi_s{s}")
        eta[s] = m.addVar(lb=0.0, name=f"eta_s{s}")
        for t in range(T):
            g[(t, s)] = m.addVar(lb=0.0, name=f"g_t{t}_s{s}")

    # VaR variable
    zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")

    m.update()

    # -------------------------------------------------------
    # Constraints
    # -------------------------------------------------------

    # K = N * S (nonconvex quadratic)
    m.addQConstr(K_var == N_var * S_var, name="CapacityDefinition")

    # Budget constraint
    m.addConstr(capex_per_mw * K_var <= budget_eur, name="Budget")

    # Generation limits: 0 <= g[t,s] <= K * rho[t,s]
    for s in range(S_count):
        for t in range(T):
            m.addConstr(
                g[(t, s)] <= K_var * rho[t, s],
                name=f"GenLimit_t{t}_s{s}"
            )

    # -------------------------------------------------------
    # Profit Definition (CORRECTED TIME SCALING)
    # -------------------------------------------------------
    
    # Scaling factor to convert simulation horizon (e.g., 24h) to one year (8760h)
    hours_per_year = 8760.0
    time_scale_factor = hours_per_year / T

    # xi_s = (Daily_Operational_Profit * Scale_Factor) - Fixed_Annual_OPEX
    for s in range(S_count):
        op_profit_expr = gp.LinExpr() # Operational profit for the simulated horizon

        for t in range(T):
            # day-ahead bid = K * rho_forecast
            b_coeff = rho_forecast[t, s]     # bid = K_var * b_coeff

            # DA revenue
            op_profit_expr.addTerms(b_coeff * price_da[t, s], K_var)

            # balancing revenue: (g - bid) * price_bal
            op_profit_expr.add(g[(t, s)], price_bal[t, s])
            op_profit_expr.addTerms(-b_coeff * price_bal[t, s], K_var)

            # variable OPEX
            op_profit_expr.add(g[(t, s)], -variable_opex_per_mwh)
        
        # Combine into annual profit:
        # xi[s] = (Operational Profit * Scale Factor) - Annual Fixed OPEX
        total_annual_expr = gp.LinExpr()
        total_annual_expr.add(op_profit_expr, time_scale_factor)
        total_annual_expr.addTerms(-fixed_opex_per_mw_year, K_var)

        m.addConstr(xi[s] == total_annual_expr, name=f"ProfitDef_s{s}")

    # CVaR constraints
    for s in range(S_count):
        # eta_s >= loss_s - zeta = -xi_s - zeta
        m.addConstr(eta[s] >= -xi[s] - zeta, name=f"CVaR_eta_s{s}")

    # -------------------------------------------------------
    # Objective
    # -------------------------------------------------------

    expected_profit = gp.quicksum(pi[s] * xi[s] for s in range(S_count))

    cvar_term = zeta + gp.quicksum(pi[s] * eta[s] / (1 - alpha) for s in range(S_count))

    total_npv = (
        pv_factor * expected_profit
        - capex_per_mw * K_var          # CAPEX
        - lambda_risk * cvar_term
    )

    m.setObjective(total_npv, GRB.MAXIMIZE)

    return m


def solve_model3(
    inputs: Optional[Dict[str, Any]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    m = build_model3(inputs)

    if solver_options:
        for k, v in solver_options.items():
            m.setParam(k, v)

    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi ended with status {m.Status}")

    return {
        "model": m,
        "K_opt_mw": m.getVarByName("K").X,
        "S_opt_mw": m.getVarByName("S").X,
        "N_opt": m.getVarByName("N").X,
        "NPV_eur": m.ObjVal,
        "zeta": m.getVarByName("zeta").X,
    }