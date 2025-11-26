# src/data_loader.py

from enum import Enum
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from config import (
    get_default_economic_params,
    get_default_market_params_deterministic,
    EconomicParams
)


class ModelType(str, Enum):
    MODEL1_DETERMINISTIC = "model1"
    MODEL2_MULTI_PERIOD   = "model2"
    MODEL3_STOCHASTIC     = "model3"


def load_inputs(
    model_type: ModelType,
    *,
    override_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Central data-loading / preparation function for ALL models.
    Returns a dictionary with:
        - 'economics': EconomicParams
        - 'market': model-specific market/wind data
        - 'meta': metadata (model type, notes, etc.)
    """

    economics = get_default_economic_params()

    # Allow quick overriding (e.g., for sensitivity studies)
    if override_config:
        for key, value in override_config.items():
            if hasattr(economics, key):
                setattr(economics, key, value)

    if model_type == ModelType.MODEL1_DETERMINISTIC:
        market = _load_market_data_model1()

    elif model_type == ModelType.MODEL2_MULTI_PERIOD:
        market = _load_market_data_model2(economics)

    elif model_type == ModelType.MODEL3_STOCHASTIC:
        market = _load_market_data_model3()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return {
        "economics": economics,
        "market": market,
        "meta": {
            "model_type": model_type.value
        }
    }


# ---- internal helpers for each model type ----

def _load_market_data_model1() -> Dict[str, Any]:
    """
    Deterministic averages: capacity factor + constant prices.
    Suitable for Model 1 (no time, no uncertainty).
    """
    market_det = get_default_market_params_deterministic()

    annual_mwh_per_mw = market_det.capacity_factor * 8760.0

    return {
        "type": "deterministic",
        "capacity_factor": market_det.capacity_factor,
        "price_da": market_det.price_da_eur_per_mwh,
        "price_bal": market_det.price_bal_eur_per_mwh,
        "annual_mwh_per_mw": annual_mwh_per_mw,
    }


def _load_market_data_model2(econ_params: EconomicParams) -> Dict[str, Any]:
    """
    For the multi-period model.
    Loads the time horizon and populates stub data for W_F, W_R, P_DA, P_BAL.
    
    T_periods is set to the project's lifetime_years.
    """
    # T_periods is set to the project's lifetime years (e.g., 30)
    T_periods = econ_params.lifetime_years
    
    # --- STUB DATA POPULATION ---
    # This data ensures the model is runnable, but you should replace it 
    # with real time series data later.
    
    market_det = get_default_market_params_deterministic()
    # MWh produced per MW capacity over a single period (assuming a period is a year)
    annual_mwh_per_mw = market_det.capacity_factor * 8760.0
    
    # Create dictionaries indexed from 1 to T_periods
    # W_F (Forecasted Production per MW) - Constant for stub
    W_F_stub = {t: annual_mwh_per_mw for t in range(1, T_periods + 1)}
    
    # W_R (Realized Production per MW)
    # Introducing slight variation (1% max deviation) to create an imbalance profile
    # This just ensures W_R != W_F for some periods.
    W_R_stub = {
        t: annual_mwh_per_mw * (1.0 + 0.01 * (t % 3 == 0) - 0.005) 
        for t in range(1, T_periods + 1)
    }
    
    # Prices (Constant over the periods for this stub)
    P_DA_stub = {t: market_det.price_da_eur_per_mwh for t in range(1, T_periods + 1)}
    P_BAL_stub = {t: market_det.price_bal_eur_per_mwh for t in range(1, T_periods + 1)}
    
    return {
        "type": "multi_period",
        "T_periods": T_periods, # Now correctly set to 30 (or whatever is in config.py)
        "W_F": W_F_stub,     # Forecasted MWh/MW/period
        "W_R": W_R_stub,     # Realized MWh/MW/period
        "P_DA": P_DA_stub,   # Day-Ahead Price (EUR/MWh)
        "P_BAL": P_BAL_stub, # Balancing Price (EUR/MWh)
        
        # Additional metadata
        "time_index": list(range(1, T_periods + 1)),
        "da_prices": P_DA_stub,
        "bal_prices": P_BAL_stub,
    }



def _load_market_data_model3() -> Dict[str, Any]:
    """
    For the stochastic model (stub for now).
    You will later load scenarios/scenario tree here.
    """
    return {
        "type": "stochastic",
        "scenarios": None,
        "scenario_probabilities": None,
    }
