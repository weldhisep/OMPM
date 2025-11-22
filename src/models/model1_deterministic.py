# src/models/model1_deterministic.py

from typing import Dict, Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Var,
    NonNegativeReals,
    PositiveReals,
    Constraint,
    Objective,
    Param,
    maximize,
    value,
    SolverFactory,
)

from src.data_loader import load_inputs, ModelType
from src.economics import present_value_factor


def build_model1(inputs: Optional[Dict[str, Any]] = None) -> ConcreteModel:
    """
    Deterministic Model 1 with turbine size and number of turbines.
    
    VARIABLES:
        S: turbine size (MW per turbine)
        N: number of turbines (continuous for now)
        K: installed capacity (MW)

    CONSTRAINT:
        K = N * S

    OBJECTIVE:
        Maximize NPV = PV(annual margin * K) - CAPEX * K

    BUDGET:
        CAPEX * K <= Budget
    """

    if inputs is None:
        inputs = load_inputs(ModelType.MODEL1_DETERMINISTIC)

    econ = inputs["economics"]
    market = inputs["market"]

    pv_factor = present_value_factor(econ.discount_rate, econ.lifetime_years)

    m = ConcreteModel()

    # ----- Parameters -----
    m.capex_per_mw = Param(initialize=float(econ.capex_per_mw))
    m.fixed_opex_per_mw_year = Param(initialize=float(econ.fixed_opex_per_mw_year))
    m.variable_opex_per_mwh = Param(initialize=float(econ.variable_opex_per_mwh))
    m.budget_eur = Param(initialize=float(econ.budget_eur))
    m.annual_mwh_per_mw = Param(initialize=float(market["annual_mwh_per_mw"]))
    m.price_da = Param(initialize=float(market["price_da"]))
    m.pv_factor = Param(initialize=float(pv_factor))

    # TURBINE SIZE bounds
    # Typical offshore: 8–25 MW, can modify later or make config-based
    m.S_min = Param(initialize=8.0)
    m.S_max = Param(initialize=25.0)

    # ----- Variables -----
    m.S = Var(domain=PositiveReals, bounds=lambda m: (m.S_min, m.S_max))  # turbine size (MW)
    m.N = Var(domain=NonNegativeReals)                                    # number of turbines
    m.K = Var(domain=NonNegativeReals)                                    # total capacity (MW)

    # ----- Constraints -----

    # K = N * S
    def capacity_rule(model):
        return model.K == model.N * model.S
    m.CapacityDefinition = Constraint(rule=capacity_rule)

    # Budget: CAPEX * K <= Budget
    def budget_rule(model):
        return model.capex_per_mw * model.K <= model.budget_eur
    m.BudgetConstraint = Constraint(rule=budget_rule)

    # ----- Objective -----
    def objective_rule(model):
        annual_revenue_per_mw = model.annual_mwh_per_mw * model.price_da
        annual_opex_per_mw = (
            model.fixed_opex_per_mw_year +
            model.annual_mwh_per_mw * model.variable_opex_per_mwh
        )
        annual_margin_per_mw = annual_revenue_per_mw - annual_opex_per_mw

        # NPV contribution of 1 MW
        npv_per_mw = annual_margin_per_mw * model.pv_factor - model.capex_per_mw

        return model.K * npv_per_mw

    m.Objective = Objective(rule=objective_rule, sense=maximize)

    return m



def solve_model1(
    solver_name: str = "gurobi",
    solver_options: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    model = build_model1(inputs)

    solver = SolverFactory(solver_name)
    if solver_options:
        for k, v in solver_options.items():
            solver.options[k] = v

    result = solver.solve(model, tee=False)

    return {
        "model": model,
        "solver_result": result,
        "K_opt_mw": value(model.K),
        "S_opt_mw": value(model.S),
        "N_opt": value(model.N),
        "NPV_eur": value(model.Objective),
    }
