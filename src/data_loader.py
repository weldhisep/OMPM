# src/data_loader.py

from enum import Enum
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
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
    
    # 2. Prices (Implementing Annual Decay)
    
    # Annual Decay Factor: -1.0% per year (simulating price cannibalization)
    price_decay_rate = 0.015 
    
    P_DA_base = market_det.price_da_eur_per_mwh
    P_BAL_base = market_det.price_bal_eur_per_mwh
    
    # Price in period t: Base Price * (1 - decay_rate)^(t-1)
    P_DA_stub = {
        t: P_DA_base * ((1.0 - price_decay_rate) ** (t - 1))
        for t in range(1, T_periods + 1)
    }
    
    P_BAL_stub = {
        t: P_BAL_base * ((1.0 - price_decay_rate) ** (t - 1))
        for t in range(1, T_periods + 1)
    }
    
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
    Loads random but realistic scenario data for Model 3 (stochastic).
    This is where all scenario values (rho, forecast, prices, probabilities)
    should be created or loaded.
    """

    import numpy as np

    # --- dimensions ---
    T = 24              # 24 hours
    S = 6               # number of scenarios

    # --- Generate realistic CF patterns ---
    rho = np.zeros((T, S))
    for s in range(S):
        base = 0.35 + 0.10 * np.sin(np.linspace(0, 2*np.pi, T))
        noise = np.random.normal(0, 0.05, T)
        scale = np.random.normal(1.0, 0.15)
        rho[:, s] = np.clip(scale * (base + noise), 0, 1)

    # --- Forecast CF ---
    rho_forecast = np.clip(rho + np.random.normal(0, 0.06, rho.shape), 0, 1)

    # --- Day-ahead prices ---
    price_da = 120 - 40 * rho + np.random.normal(0, 3, size=(T, S))
    price_da = np.clip(price_da, 20, 200)

    # --- Balancing prices ---
    price_bal = price_da * (1 + np.random.normal(0, 0.05, size=price_da.shape))

    # --- Equal probabilities ---
    pi = np.ones(S) / S

    scenarios = {
        "rho": rho,
        "rho_forecast": rho_forecast,
        "price_da": price_da,
        "price_bal": price_bal,
        "probabilities": pi,
    }

    return {
        "type": "stochastic",
        "scenarios": scenarios,
    }
