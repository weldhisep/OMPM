# src/data_loader.py

from enum import Enum
from typing import Dict, Any, Optional

from config import (
    get_default_economic_params,
    get_default_market_params_deterministic,
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
        market = _load_market_data_model2()

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


def _load_market_data_model2() -> Dict[str, Any]:
    """
    For the multi-period model (stub for now).
    You will later load time series from CSV/DB here.
    """
    return {
        "type": "multi_period",
        "time_index": None,
        "da_prices": None,
        "bal_prices": None,
        "forecast_wind": None,
        "realized_wind": None,
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
