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
    For the stochastic model (stub for now).
    You will later load scenarios/scenario tree here.
    """
    return {
        "type": "stochastic",
        "scenarios": None,
        "scenario_probabilities": None,
    }
