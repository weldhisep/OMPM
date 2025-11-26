# src/config.py

from dataclasses import dataclass

@dataclass
class EconomicParams:
    capex_per_mw: float           # EUR/MW
    fixed_opex_per_mw_year: float # EUR/MW/year
    variable_opex_per_mwh: float  # EUR/MWh
    lifetime_years: int
    discount_rate: float          # e.g. 0.06
    budget_eur: float             # total budget
    lambda_risk: float = 0.0    # risk aversion weight (for CVaR)
    cvar_alpha: float = 0.90     # CVaR confidence level

@dataclass
class MarketParamsDeterministic:
    capacity_factor: float        # 0–1
    price_da_eur_per_mwh: float
    price_bal_eur_per_mwh: float  # not really used in Model 1, but good to have


# ---- Default baseline for DK energy-island style offshore wind ----

def get_default_economic_params() -> EconomicParams:
    return EconomicParams(
        capex_per_mw=2.5e6,              # 2.5 M€/MW
        fixed_opex_per_mw_year=70_000,   # €/MW/year
        variable_opex_per_mwh=5.0,       # €/MWh
        lifetime_years=30,
        discount_rate=0.06,
        budget_eur=7.5e9                 # 7.5 billion €
    )


def get_default_market_params_deterministic() -> MarketParamsDeterministic:
    return MarketParamsDeterministic(
        capacity_factor=0.50,
        price_da_eur_per_mwh=75.0,
        price_bal_eur_per_mwh=75.0
    )
