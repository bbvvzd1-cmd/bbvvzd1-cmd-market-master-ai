import numpy as np
from market_master_ai import QuantitativeEngine, MilitaryLogger

def make_synthetic_data(length=200):
    base = np.linspace(100.0, 110.0, length)
    noise = np.random.normal(0, 0.2, length)
    close = base + noise
    high = close * 1.005
    low = close * 0.995
    volume = np.full(length, 1000.0)
    timestamps = np.arange(length)
    return {
        "timestamp": timestamps,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "current_price": float(close[-1])
    }

def test_calculate_advanced_indicators_basic():
    logger = MilitaryLogger()
    engine = QuantitativeEngine(logger)
    data = make_synthetic_data(200)
    indicators = engine.calculate_advanced_indicators(data)
    assert indicators is None or isinstance(indicators, dict)
    if indicators:
        for k in ['rsi', 'volatility', 'bb_upper', 'bb_lower', 'sma_20']:
            assert k in indicators
