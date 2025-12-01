# src/models/model3_stochastic.py

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


def build_model3(inputs: Optional[Dict[str, Any]] = None) -> gp.Model:
    """
    Stochastic Model 3 (MILP with scenario-based operational variables)

    - Turbine size chosen from discrete set (binary y[size])
    - N integer number of turbines
    - K = N * S (where S = sum(size * y[size]))  -> linear in variables because S uses binaries
    - Scenario operational variables g[t,s] (realized generation with possible curtailment)
    - CVaR risk-penalty on losses (loss = -xi: negative operational profit)
    - Wake losses: capacity factor reduced with more turbines (wake_coeff * N) applied inside g limit
      (implemented as quadratic constraint because K*N appears)
    """

    if inputs is None:
        inputs = load_inputs(ModelType.MODEL3_STOCHASTIC)

    econ = inputs["economics"]
    market = inputs["market"]

    # scenarios loaded under market["scenarios"]
    scenarios = market.get("scenarios", None)
    if scenarios is None:
        raise RuntimeError("No scenarios found in inputs['market']['scenarios']")

    rho = scenarios["rho"]                 # shape (T, S)
    rho_forecast = scenarios["rho_forecast"]
    price_da = scenarios["price_da"]
    price_bal = scenarios["price_bal"]
    pi = scenarios["probabilities"]

    # Dimensions
    T = int(rho.shape[0])
    S_count = int(rho.shape[1])

    # Economic params
    pv_factor = present_value_factor(econ.discount_rate, econ.lifetime_years)

    fixed_opex_per_mw_year = float(econ.fixed_opex_per_mw_year)
    variable_opex_per_mwh = float(econ.variable_opex_per_mwh)
    budget = float(econ.budget_eur)

    # Risk parameters (ensure exist in config)
    lambda_risk = float(getattr(econ, "lambda_risk", 0.0))
    alpha = float(getattr(econ, "cvar_alpha", 0.90))

    # Turbine sizes and CAPEX map (same style as model1)
    turbine_sizes = [12, 15, 18, 20]  # MW
    capex_map = {
        12: 3.0e6,
        15: 2.8e6,
        18: 2.6e6,
        20: 2.5e6,
    }

    wake_coeff = 0.0004  # CF loss per turbine (applied to capacity factor)

    # Create model
    m = gp.Model("model3_stochastic")
    # allow nonconvex quadratics (we will use K*N in wake term)
    m.Params.NonConvex = 2
    # tighten MIP tolerance similar to model1 if desired
    m.Params.MIPGap = 1e-6

    # ----- Variables -----
    y = m.addVars(turbine_sizes, vtype=GRB.BINARY, name="y")  # choose turbine size
    N = m.addVar(vtype=GRB.INTEGER, lb=0, name="N")           # number of turbines
    K = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="K")      # total MW

    # S derived from y (continuous)
    S = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="S")
    m.addConstr(S == gp.quicksum(size * y[size] for size in turbine_sizes), "S_definition")
    m.addConstr(gp.quicksum(y[size] for size in turbine_sizes) == 1, "OneSize")

    # Link capacity K = N * S (nonlinear)
    # Keep as a quadratic constraint consistent with Model 1 style
    m.addQConstr(K == N * S, "CapacityDefinition")

    # Select CAPEX per MW according to chosen size
    CAPEX_per_MW = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="CAPEX_per_MW")
    m.addConstr(
        CAPEX_per_MW == gp.quicksum(capex_map[size] * y[size] for size in turbine_sizes),
        "CAPEX_selection"
    )

    # Budget constraint
    m.addConstr(CAPEX_per_MW * K <= budget, "Budget")

    # Scenario operational variables
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

    # ----- Constraints: generation limits with wake loss -----
    # g[t,s] <= K * (rho[t,s] - wake_coeff * N)
    # rearranged to a quadratic form: g + wake_coeff * K * N <= K * rho
    for s in range(S_count):
        for t in range(T):
            # Left side: g + wake_coeff * K * N
            # Right side: K * rho[t,s]
            # Use addQConstr to allow K*N product
            # g[(t,s)] + wake_coeff * K * N <= K * rho[t,s]
            expr_left = gp.QuadExpr()
            expr_left.add(g[(t, s)])
            # add quadratic term wake_coeff * K * N
            expr_left.add(wake_coeff * K * N)
            # addQConstr(expr_left <= K * rho)
            m.addQConstr(expr_left <= K * float(rho[t, s]), name=f"GenLimit_t{t}_s{s}")

    # ----- Profit definition -----
    # We treat the simulation horizon of length T as representative and scale to annual.
    hours_per_year = 8760.0
    time_scale = hours_per_year / float(T)

    for s in range(S_count):
        op_expr = gp.LinExpr()  # operational profit over simulated horizon (not ann.)
        for t in range(T):
            # day-ahead bid = K * rho_forecast[t,s]
            b_coeff = float(rho_forecast[t, s])

            # DA revenue: b * price_da --> (b_coeff * K) * price_da
            op_expr.addTerms(b_coeff * float(price_da[t, s]), K)

            # balancing revenue: (g - b) * price_bal -> g * p_bal - b * p_bal
            op_expr.add(g[(t, s)], float(price_bal[t, s]))
            op_expr.addTerms(-b_coeff * float(price_bal[t, s]), K)

            # variable OPEX: -variable_opex_per_mwh * g
            op_expr.add(g[(t, s)], -variable_opex_per_mwh)

        # scale operational profit to annual
        total_annual_op = gp.LinExpr()
        total_annual_op.add(op_expr, time_scale)

        # subtract fixed annual OPEX: fixed_opex_per_mw_year * K
        total_annual_op.addTerms(-fixed_opex_per_mw_year, K)

        # xi[s] is annual operational profit (excludes CAPEX)
        m.addConstr(xi[s] == total_annual_op, name=f"ProfitDef_s{s}")

    # CVaR constraints: eta_s >= -xi_s - zeta
    for s in range(S_count):
        m.addConstr(eta[s] >= -xi[s] - zeta, name=f"CVaR_eta_s{s}")

    # ----- Objective -----
    expected_operational = gp.quicksum(pi[s] * xi[s] for s in range(S_count))
    cvar_term = zeta + gp.quicksum(pi[s] * eta[s] / (1.0 - alpha) for s in range(S_count))

    # Total NPV: PV(expected operational) - CAPEX - lambda * CVaR
    total_npv = pv_factor * expected_operational - CAPEX_per_MW * K - lambda_risk * cvar_term

    m.setObjective(total_npv, GRB.MAXIMIZE)

    return m


def solve_model3(
    inputs: Optional[Dict[str, Any]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build and solve Model 3. Returns results similar to Model 1.
    """

    m = build_model3(inputs)

    if solver_options:
        for k, v in solver_options.items():
            m.setParam(k, v)

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed with status {m.Status}")

    # Extract chosen turbine size (y[size])
    chosen_size = None
    for size in [12, 15, 18, 20]:
        var = m.getVarByName(f"y[{size}]")
        if var is not None and var.X > 0.5:
            chosen_size = size

    return {
        "model": m,
        "K_opt_mw": m.getVarByName("K").X,
        "N_opt": m.getVarByName("N").X,
        "S_opt_mw": chosen_size,
        "NPV_eur": m.ObjVal,
        "zeta": m.getVarByName("zeta").X,
    }
