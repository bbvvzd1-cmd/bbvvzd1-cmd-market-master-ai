#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTIMATE CRYPTO TRADING AI - MARKET MASTER
Version: 4.0 | AI-Powered Adaptive Trading System
Developer: Elite Quant Team
License: Proprietary Trading Algorithm

PATCHED: safer TA handling, cache bounding & pruning, logger handler guard,
         BB division guard, safer market metrics parsing, fixed symbol rotation,
         NaN/inf checks and other small robustness fixes.
"""

# =============================================================================
# IMPORTS - ABSOLUTE REQUIREMENTS - NO COMPROMISES
# =============================================================================
import numpy as np
import pandas as pd
import requests
import time
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')
import json
import talib
from scipy import stats
import logging
from logging.handlers import RotatingFileHandler
import math

# =============================================================================
# GLOBAL CONFIGURATION - MILITARY GRADE PRECISION
# =============================================================================
class EliteConfig:
    # API ENDPOINTS - PRIMARY & FAILOVER
    BINANCE_ENDPOINTS = [
        'https://api.binance.com',
        'https://api1.binance.com',
        'https://api2.binance.com',
        'https://api3.binance.com'
    ]
    
    # TRADING UNIVERSE - DYNAMIC EXPANSION
    BASE_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
        'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT'
    ]
    
    # ANALYSIS PARAMETERS - OPTIMIZED FOR PERFORMANCE
    ANALYSIS_INTERVAL = 1.5  # MINUTES
    DATA_POINTS = 200
    SIGNAL_CONFIDENCE_THRESHOLD = 68  # PERCENT
    
    # RISK MANAGEMENT - NON-NEGOTIABLE
    MAX_POSITIONS_PER_CYCLE = 8
    MIN_VOLUME_THRESHOLD = 500000  # USD (assumes quote currency is USDT)
    MAX_VOLATILITY_THRESHOLD = 25.0  # PERCENT

# =============================================================================
# ADVANCED LOGGING SYSTEM - COMMAND CENTER GRADE
# =============================================================================
class MilitaryLogger:
    def __init__(self):
        self.logger = logging.getLogger('MarketMasterAI')
        self.logger.setLevel(logging.INFO)
        
        # Prevent adding duplicate handlers if logger already configured
        if not self.logger.handlers:
            # CONSOLE HANDLER
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            
            # FILE HANDLER
            file_handler = RotatingFileHandler(
                'market_master.log', 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
            )
            file_handler.setFormatter(file_format)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False
    
    def log_signal(self, symbol, signal_type, confidence, conditions):
        self.logger.info(
            f"SIGNAL | {symbol} | {signal_type} | Confidence: {confidence}% | "
            f"Conditions: {', '.join(conditions)}"
        )
    
    def log_market_condition(self, condition, value):
        self.logger.info(f"MARKET | {condition}: {value}")

# =============================================================================
# INTELLIGENT DATA ACQUISITION ENGINE
# =============================================================================
class MarketDataEngine:
    def __init__(self, logger):
        self.logger = logger
        self.endpoint_index = 0
        self.failover_count = 0
        self.request_cache = {}
        self.cache_timeout = 2  # SECONDS
        self.cache_max_entries = 500  # bound cache to avoid unbounded growth
    
    def _rotate_endpoint(self):
        """INTELLIGENT ENDPOINT ROTATION WITH FAILOVER"""
        self.endpoint_index = (self.endpoint_index + 1) % len(EliteConfig.BINANCE_ENDPOINTS)
        if self.endpoint_index == 0:
            self.failover_count += 1
            self.logger.logger.warning(f"Endpoint failover cycle #{self.failover_count}")
    
    def _make_api_request(self, endpoint_suffix, params=None):
        """MILITARY-GRADE API REQUEST HANDLER (with 429 handling & backoff)"""
        max_retries = 4
        backoff = 1
        for attempt in range(max_retries):
            try:
                base_url = EliteConfig.BINANCE_ENDPOINTS[self.endpoint_index]
                url = f"{base_url}/api/v3/{endpoint_suffix}"
                
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=10,
                    headers={'User-Agent': 'MarketMasterAI/4.0'}
                )
                
                if response.status_code == 200:
                    # Attempt to parse JSON safely
                    try:
                        return response.json()
                    except Exception:
                        # Return raw text fallback
                        return response.text
                elif response.status_code == 429:
                    # Rate limited — respect Retry-After if provided
                    retry_after = response.headers.get('Retry-After')
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff
                    self.logger.logger.warning(f"Rate limited (429). Sleeping for {wait}s.")
                    time.sleep(wait)
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                    
            except Exception as e:
                self.logger.logger.warning(f"API attempt {attempt + 1} failed: {str(e)}")
                self._rotate_endpoint()
                time.sleep(backoff)
                backoff = min(backoff * 2, 8)
        
        raise Exception("All API endpoints exhausted")
    
    def _prune_cache_if_needed(self):
        if len(self.request_cache) > self.cache_max_entries:
            # remove oldest based on timestamp
            sorted_items = sorted(self.request_cache.items(), key=lambda kv: kv[1]['timestamp'])
            # remove 10% oldest or until under limit
            remove_count = max(1, int(0.1 * len(sorted_items)))
            for i in range(remove_count):
                key_to_remove = sorted_items[i][0]
                try:
                    del self.request_cache[key_to_remove]
                except KeyError:
                    pass
    
    def get_klines_data(self, symbol, interval='5m', limit=100):
        """ADVANCED KLINE DATA ACQUISITION WITH INTELLIGENT CACHING"""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()
        
        # INTELLIGENT CACHE MANAGEMENT
        cached = self.request_cache.get(cache_key)
        if cached and current_time - cached['timestamp'] < self.cache_timeout:
            return cached['data']
        
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            data = self._make_api_request('klines', params)
            
            if not data or len(data) < 20:
                return None
            
            # Validate candle shape
            candles = []
            for candle in data:
                if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                    continue
                candles.append(candle)
            if len(candles) < 20:
                return None
            
            # HIGH-PRECISION DATA EXTRACTION (timestamps normalized to UTC)
            processed_data = {
                'timestamp': [datetime.utcfromtimestamp(candle[0] / 1000) for candle in candles],
                'open': np.array([float(candle[1]) for candle in candles]),
                'high': np.array([float(candle[2]) for candle in candles]),
                'low': np.array([float(candle[3]) for candle in candles]),
                'close': np.array([float(candle[4]) for candle in candles]),
                'volume': np.array([float(candle[5]) for candle in candles]),
                'current_price': float(candles[-1][4])
            }
            
            # UPDATE CACHE (bounded)
            self.request_cache[cache_key] = {
                'data': processed_data,
                'timestamp': current_time
            }
            self._prune_cache_if_needed()
            
            return processed_data
            
        except Exception as e:
            self.logger.logger.error(f"Data acquisition failed for {symbol}: {str(e)}")
            return None
    
    def get_market_metrics(self, symbol):
        """COMPREHENSIVE MARKET METRICS COLLECTION (safe parsing)"""
        try:
            ticker_data = self._make_api_request('ticker/24hr', {'symbol': symbol})
            if not ticker_data or not isinstance(ticker_data, dict):
                return None
            
            def safe_float(val):
                try:
                    return float(val)
                except Exception:
                    return 0.0
            def safe_int(val):
                try:
                    return int(val)
                except Exception:
                    return 0
            
            return {
                'price_change_percent': safe_float(ticker_data.get('priceChangePercent', 0)),
                'volume': safe_float(ticker_data.get('volume', 0)),
                'quote_volume': safe_float(ticker_data.get('quoteVolume', 0)),
                'count': safe_int(ticker_data.get('count', 0))
            }
        except Exception as e:
            self.logger.logger.warning(f"Market metrics failed for {symbol}: {str(e)}")
            return None

# =============================================================================
# QUANTITATIVE ANALYSIS ENGINE - INSTITUTIONAL GRADE
# =============================================================================
class QuantitativeEngine:
    def __init__(self, logger):
        self.logger = logger
        self.technical_indicators = {}
    
    def _is_finite_scalar(self, x):
        return x is not None and np.isfinite(x) and not (isinstance(x, float) and math.isnan(x))
    
    def calculate_advanced_indicators(self, data):
        """INSTITUTIONAL-GRADE TECHNICAL ANALYSIS (with NaN/inf safeguards)"""
        if data is None or len(data['close']) < 50:
            return None
        
        closes = data['close']
        highs = data['high']
        lows = data['low']
        volumes = data['volume']
        
        try:
            indicators = {}
            
            # ===== TREND INDICATORS =====
            sma_20_arr = talib.SMA(closes, timeperiod=20)
            sma_50_arr = talib.SMA(closes, timeperiod=50)
            ema_12_arr = talib.EMA(closes, timeperiod=12)
            ema_26_arr = talib.EMA(closes, timeperiod=26)
            
            indicators['sma_20'] = float(sma_20_arr[-1]) if np.isfinite(sma_20_arr[-1]) else np.nan
            indicators['sma_50'] = float(sma_50_arr[-1]) if np.isfinite(sma_50_arr[-1]) else np.nan
            indicators['ema_12'] = float(ema_12_arr[-1]) if np.isfinite(ema_12_arr[-1]) else np.nan
            indicators['ema_26'] = float(ema_26_arr[-1]) if np.isfinite(ema_26_arr[-1]) else np.nan
            
            # ===== MOMENTUM INDICATORS =====
            rsi_arr = talib.RSI(closes, timeperiod=14)
            indicators['rsi'] = float(rsi_arr[-1]) if np.isfinite(rsi_arr[-1]) else np.nan
            
            stoch_k_arr, stoch_d_arr = talib.STOCH(highs, lows, closes)
            indicators['stoch_k'] = float(stoch_k_arr[-1]) if np.isfinite(stoch_k_arr[-1]) else np.nan
            indicators['stoch_d'] = float(stoch_d_arr[-1]) if np.isfinite(stoch_d_arr[-1]) else np.nan
            
            macd_arr, macd_signal_arr, macd_hist_arr = talib.MACD(closes)
            indicators['macd'] = float(macd_arr[-1]) if np.isfinite(macd_arr[-1]) else np.nan
            indicators['macd_signal'] = float(macd_signal_arr[-1]) if np.isfinite(macd_signal_arr[-1]) else np.nan
            indicators['macd_hist'] = float(macd_hist_arr[-1]) if np.isfinite(macd_hist_arr[-1]) else np.nan
            
            # ===== VOLATILITY INDICATORS =====
            bb_upper_arr, bb_middle_arr, bb_lower_arr = talib.BBANDS(closes)
            indicators['bb_upper'] = float(bb_upper_arr[-1]) if np.isfinite(bb_upper_arr[-1]) else np.nan
            indicators['bb_middle'] = float(bb_middle_arr[-1]) if np.isfinite(bb_middle_arr[-1]) else np.nan
            indicators['bb_lower'] = float(bb_lower_arr[-1]) if np.isfinite(bb_lower_arr[-1]) else np.nan
            atr_arr = talib.ATR(highs, lows, closes, timeperiod=14)
            indicators['atr'] = float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) else np.nan
            
            # ===== VOLUME INDICATORS =====
            vol_sma_arr = talib.SMA(volumes, timeperiod=20)
            indicators['volume_sma'] = float(vol_sma_arr[-1]) if np.isfinite(vol_sma_arr[-1]) else np.nan
            if indicators['volume_sma'] > 0:
                indicators['volume_ratio'] = float(volumes[-1]) / indicators['volume_sma']
            else:
                indicators['volume_ratio'] = 1.0
            
            obv_arr = talib.OBV(closes, volumes)
            indicators['obv'] = float(obv_arr[-1]) if np.isfinite(obv_arr[-1]) else np.nan
            
            # ===== ADVANCED MOMENTUM =====
            willr_arr = talib.WILLR(highs, lows, closes, timeperiod=14)
            indicators['williams_r'] = float(willr_arr[-1]) if np.isfinite(willr_arr[-1]) else np.nan
            cci_arr = talib.CCI(highs, lows, closes, timeperiod=20)
            indicators['cci'] = float(cci_arr[-1]) if np.isfinite(cci_arr[-1]) else np.nan
            mfi_arr = talib.MFI(highs, lows, closes, volumes, timeperiod=14)
            indicators['mfi'] = float(mfi_arr[-1]) if np.isfinite(mfi_arr[-1]) else np.nan
            
            # ===== PRICE ACTION =====
            # Ensure we have enough history for requested offsets
            def safe_pct_change(a, offset):
                try:
                    if len(a) > offset:
                        prev = a[-(offset+1)]
                        if prev != 0:
                            return ((a[-1] - prev) / prev) * 100
                except Exception:
                    pass
                return 0.0
            
            price_changes = {
                '5m': safe_pct_change(closes, 1),
                '15m': safe_pct_change(closes, 4),
                '1h': safe_pct_change(closes, 12),
                '4h': safe_pct_change(closes, 48)
            }
            
            indicators['price_changes'] = price_changes
            
            # volatility based on recent 20 closes, guard division
            recent = closes[-20:]
            mean_recent = np.mean(recent) if np.mean(recent) != 0 else np.nan
            if np.isfinite(mean_recent) and mean_recent != 0:
                indicators['volatility'] = (np.std(recent) / mean_recent) * 100
            else:
                indicators['volatility'] = np.nan
            
            # Final validation: ensure required scalars are finite
            required_scalars = ['rsi', 'volatility', 'bb_upper', 'bb_lower', 'sma_20', 'sma_50', 'ema_12', 'ema_26']
            for key in required_scalars:
                if not self._is_finite_scalar(indicators.get(key)):
                    self.logger.logger.debug(f"Indicator {key} not finite for data set; skipping.")
                    return None
            
            return indicators
            
        except Exception as e:
            self.logger.logger.error(f"Indicator calculation failed: {str(e)}")
            return None
    
    def calculate_market_regime(self, indicators_list):
        """ADAPTIVE MARKET REGIME DETECTION"""
        if not indicators_list:
            return "NEUTRAL"
        
        rsi_values = [ind['rsi'] for ind in indicators_list if ind]
        volatility_values = [ind['volatility'] for ind in indicators_list if ind]
        
        if not rsi_values or not volatility_values:
            return "NEUTRAL"
        
        avg_rsi = np.mean(rsi_values)
        avg_volatility = np.mean(volatility_values)
        
        if avg_rsi < 35 and avg_volatility > 8:
            return "OVERSOLD_HIGH_VOL"
        elif avg_rsi > 65 and avg_volatility > 8:
            return "OVERBOUGHT_HIGH_VOL"
        elif avg_rsi < 30:
            return "STRONGLY_OVERSOLD"
        elif avg_rsi > 70:
            return "STRONGLY_OVERBOUGHT"
        elif avg_volatility < 3:
            return "LOW_VOLATILITY"
        else:
            return "NEUTRAL"

# =============================================================================
# AI-POWERED SIGNAL GENERATION ENGINE
# =============================================================================
class AISignalEngine:
    def __init__(self, logger):
        self.logger = logger
        self.market_regime = "NEUTRAL"
        self.adaptive_thresholds = {
            "OVERSOLD_HIGH_VOL": 65,
            "OVERBOUGHT_HIGH_VOL": 70,
            "STRONGLY_OVERSOLD": 60,
            "STRONGLY_OVERBOUGHT": 75,
            "LOW_VOLATILITY": 72,
            "NEUTRAL": 68
        }
    
    def analyze_symbol(self, symbol, data, indicators, market_metrics):
        """AI-POWERED COMPREHENSIVE ANALYSIS"""
        if not all([data, indicators, market_metrics]):
            return None
        
        try:
            # ===== MULTI-DIMENSIONAL SCORING SYSTEM =====
            score_components = {}
            conditions_met = []
            
            # 1. MOMENTUM SCORING (35 POINTS MAX)
            momentum_score = 0
            if indicators['rsi'] < 30:
                momentum_score += 20
                conditions_met.append("RSI < 30 (Strong Oversold)")
            elif indicators['rsi'] < 35:
                momentum_score += 15
                conditions_met.append("RSI < 35 (Oversold)")
            
            if (indicators.get('stoch_k') is not None and indicators.get('stoch_d') is not None 
                and indicators['stoch_k'] < 20 and indicators['stoch_k'] > indicators['stoch_d']):
                momentum_score += 10
                conditions_met.append("Stochastic Bullish Crossover")
            
            if indicators.get('macd_hist', 0) > 0 and indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                momentum_score += 5
                conditions_met.append("MACD Bullish")
            
            score_components['momentum'] = min(momentum_score, 35)
            
            # 2. TREND SCORING (25 POINTS MAX)
            trend_score = 0
            if indicators['ema_12'] > indicators['ema_26']:
                trend_score += 10
                conditions_met.append("EMA Bullish Alignment")
            
            if data['close'][-1] > indicators['sma_20']:
                trend_score += 8
                conditions_met.append("Price > SMA20")
            
            if indicators['sma_20'] > indicators['sma_50']:
                trend_score += 7
                conditions_met.append("SMA Bullish Alignment")
            
            score_components['trend'] = min(trend_score, 25)
            
            # 3. VOLUME & VOLATILITY SCORING (20 POINTS MAX)
            volume_vol_score = 0
            if indicators['volume_ratio'] > 2.5:
                volume_vol_score += 12
                conditions_met.append("Volume > 2.5x Average")
            elif indicators['volume_ratio'] > 1.8:
                volume_vol_score += 8
                conditions_met.append("Volume > 1.8x Average")
            
            if 3 < indicators['volatility'] < 15:
                volume_vol_score += 8
                conditions_met.append("Optimal Volatility Range")
            
            score_components['volume_vol'] = min(volume_vol_score, 20)
            
            # 4. MARKET STRUCTURE SCORING (20 POINTS MAX)
            structure_score = 0
            denom = (indicators['bb_upper'] - indicators['bb_lower'])
            if denom and denom != 0:
                bb_position = (data['close'][-1] - indicators['bb_lower']) / denom
            else:
                bb_position = 0.5  # fallback neutral position
            
            if bb_position < 0.2:
                structure_score += 10
                conditions_met.append("Near Bollinger Lower Band")
            
            if indicators.get('williams_r', 0) < -80:
                structure_score += 6
                conditions_met.append("Williams %R Oversold")
            
            if indicators.get('cci', 0) < -100:
                structure_score += 4
                conditions_met.append("CCI Oversold")
            
            score_components['structure'] = min(structure_score, 20)
            
            # ===== CONFIDENCE CALCULATION =====
            total_score = sum(score_components.values())
            confidence = min(total_score, 100)
            
            # ===== MARKET REGIME ADAPTATION =====
            adaptive_threshold = self.adaptive_thresholds.get(self.market_regime, 68)
            
            if confidence >= adaptive_threshold and len(conditions_met) >= 2:
                # SIGNAL CLASSIFICATION
                if confidence >= 85:
                    signal_type = "🟢 STRONG BUY"
                    alert_level = "HIGH"
                elif confidence >= 75:
                    signal_type = "🟡 MODERATE BUY"
                    alert_level = "MEDIUM"
                else:
                    signal_type = "🔵 WEAK BUY"
                    alert_level = "LOW"
                
                signal_data = {
                    'symbol': symbol,
                    'signal': signal_type,
                    'alert_level': alert_level,
                    'confidence': confidence,
                    'price': data['current_price'],
                    'rsi': round(indicators['rsi'], 1),
                    'volume_ratio': round(indicators['volume_ratio'], 2),
                    'volatility': round(indicators['volatility'], 2),
                    'price_change_1h': round(indicators['price_changes']['1h'], 2),
                    'market_regime': self.market_regime,
                    'score_breakdown': score_components,
                    'conditions_met': conditions_met,
                    'timestamp': datetime.utcnow()
                }
                
                self.logger.log_signal(symbol, signal_type, confidence, conditions_met)
                return signal_data
            
            return None
            
        except Exception as e:
            self.logger.logger.error(f"Signal analysis failed for {symbol}: {str(e)}")
            return None
    
    def update_market_regime(self, regime):
        """DYNAMIC MARKET REGIME ADAPTATION"""
        self.market_regime = regime
        self.logger.log_market_condition("Market Regime", regime)

# =============================================================================
# EXECUTION ENGINE - PRECISION TRADING SYSTEM
# =============================================================================
class ExecutionEngine:
    def __init__(self, logger):
        self.logger = logger
        self.analysis_count = 0
        self.signals_generated = 0
        self.performance_metrics = defaultdict(list)
        
    def execute_analysis_cycle(self, data_engine, quant_engine, signal_engine, symbols):
        """PRECISION EXECUTION OF ANALYSIS CYCLE"""
        self.analysis_count += 1
        cycle_start = datetime.utcnow()
        
        self.logger.logger.info(f"=== ANALYSIS CYCLE #{self.analysis_count} INITIATED ===")
        
        all_indicators = []
        valid_signals = []
        
        # PHASE 1: DATA COLLECTION & INDICATOR CALCULATION
        for symbol in symbols:
            try:
                # ACQUIRE MARKET DATA
                data = data_engine.get_klines_data(symbol, '5m', EliteConfig.DATA_POINTS)
                market_metrics = data_engine.get_market_metrics(symbol)
                
                if not data or not market_metrics:
                    continue
                
                # FILTER BY LIQUIDITY (using quote_volume)
                if market_metrics.get('quote_volume', 0) < EliteConfig.MIN_VOLUME_THRESHOLD:
                    continue
                
                # CALCULATE INDICATORS
                indicators = quant_engine.calculate_advanced_indicators(data)
                if indicators:
                    all_indicators.append(indicators)
                    
                    # GENERATE SIGNAL
                    signal = signal_engine.analyze_symbol(symbol, data, indicators, market_metrics)
                    if signal:
                        valid_signals.append(signal)
                
                # RATE LIMITING (keep small pause to reduce burst)
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.logger.error(f"Cycle processing failed for {symbol}: {str(e)}")
                continue
        
        # PHASE 2: MARKET REGIME DETECTION
        if all_indicators:
            market_regime = quant_engine.calculate_market_regime(all_indicators)
            signal_engine.update_market_regime(market_regime)
        
        # PHASE 3: SIGNAL PROCESSING & RANKING
        if valid_signals:
            valid_signals.sort(key=lambda x: x['confidence'], reverse=True)
            top_signals = valid_signals[:EliteConfig.MAX_POSITIONS_PER_CYCLE]
            self.signals_generated += len(top_signals)
            
            # UPDATE PERFORMANCE METRICS
            self.performance_metrics['signals_per_cycle'].append(len(top_signals))
            self.performance_metrics['avg_confidence'].append(
                np.mean([s['confidence'] for s in top_signals])
            )
        
        cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
        self.logger.logger.info(
            f"CYCLE COMPLETED | Duration: {cycle_duration:.2f}s | "
            f"Signals: {len(valid_signals)} | Regime: {signal_engine.market_regime}"
        )
        
        return valid_signals
    
    def get_performance_stats(self):
        """REAL-TIME PERFORMANCE ANALYTICS"""
        if not self.performance_metrics or 'signals_per_cycle' not in self.performance_metrics:
            return {}
        
        stats = {
            'total_cycles': self.analysis_count,
            'total_signals': self.signals_generated,
            'signals_per_cycle': np.mean(self.performance_metrics['signals_per_cycle']) if self.performance_metrics['signals_per_cycle'] else 0.0,
            'avg_confidence': np.mean(self.performance_metrics['avg_confidence']) if self.performance_metrics['avg_confidence'] else 0.0,
            'efficiency_ratio': self.signals_generated / max(1, self.analysis_count)
        }
        
        return stats

# =============================================================================
# COMMAND & CONTROL CENTER - MASTER COORDINATOR
# =============================================================================
class MarketMasterAI:
    def __init__(self):
        # INITIALIZE CORE COMPONENTS
        self.logger = MilitaryLogger()
        self.data_engine = MarketDataEngine(self.logger)
        self.quant_engine = QuantitativeEngine(self.logger)
        self.signal_engine = AISignalEngine(self.logger)
        self.execution_engine = ExecutionEngine(self.logger)
        
        # OPERATIONAL STATE
        self.is_running = False
        self.last_analysis = None
        self.current_symbols = EliteConfig.BASE_SYMBOLS.copy()
        
        self.logger.logger.info("=== MARKET MASTER AI INITIALIZED ===")
    
    def dynamic_symbol_expansion(self):
        """INTELLIGENT SYMBOL UNIVERSE EXPANSION (wrap-around selection)"""
        try:
            # IN REAL IMPLEMENTATION, DYNAMICALLY FETCH TOP VOLUME SYMBOLS
            additional_symbols = [
                'NEARUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'ETCUSDT',
                'XLMUSDT', 'HBARUSDT', 'EGLDUSDT', 'FTMUSDT', 'SANDUSDT'
            ]
            
            # ROTATE 5 ADDITIONAL SYMBOLS EACH CYCLE (ensures 5 selected using modulo)
            rotation_index = self.execution_engine.analysis_count % len(additional_symbols)
            rotated_symbols = []
            for i in range(5):
                rotated_symbols.append(additional_symbols[(rotation_index + i) % len(additional_symbols)])
            
            expanded_universe = list(dict.fromkeys(self.current_symbols + rotated_symbols))
            return expanded_universe[:25]  # CAP AT 25 SYMBOLS
            
        except Exception as e:
            self.logger.logger.error(f"Symbol expansion failed: {str(e)}")
            return self.current_symbols
    
    def run_continuous_analysis(self):
        """PERPETUAL ANALYSIS LOOP - PRECISION EXECUTION"""
        self.is_running = True
        self.logger.logger.info("=== CONTINUOUS ANALYSIS ENGAGED ===")
        
        try:
            while self.is_running:
                current_time = datetime.utcnow()
                
                # INTELLIGENT TIMING CONTROL
                if (self.last_analysis and 
                    (current_time - self.last_analysis).total_seconds() < EliteConfig.ANALYSIS_INTERVAL * 60):
                    
                    sleep_time = EliteConfig.ANALYSIS_INTERVAL * 60 - (current_time - self.last_analysis).total_seconds()
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 10))  # MAX SLEEP 10 SECONDS
                    continue
                
                # DYNAMIC SYMBOL SELECTION
                analysis_symbols = self.dynamic_symbol_expansion()
                
                # EXECUTE ANALYSIS CYCLE
                signals = self.execution_engine.execute_analysis_cycle(
                    self.data_engine, 
                    self.quant_engine, 
                    self.signal_engine, 
                    analysis_symbols
                )
                
                # DISPLAY RESULTS
                self._display_cycle_results(signals)
                
                # UPDATE TIMING
                self.last_analysis = datetime.utcnow()
                
        except KeyboardInterrupt:
            self.logger.logger.info("=== ANALYSIS TERMINATED BY USER ===")
        except Exception as e:
            self.logger.logger.critical(f"CRITICAL FAILURE: {str(e)}")
        finally:
            self.is_running = False
            self._display_final_stats()
    
    def _display_cycle_results(self, signals):
        """PRECISION RESULTS DISPLAY"""
        print(f"\n{'='*80}")
        print(f"🎯 MARKET MASTER AI | CYCLE #{self.execution_engine.analysis_count}")
        print(f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"📊 MARKET REGIME: {self.signal_engine.market_regime}")
        print(f"{'='*80}")
        
        if signals:
            print(f"🚀 GENERATED {len(signals)} TRADING SIGNALS:")
            print("-" * 80)
            
            for i, signal in enumerate(signals[:8], 1):  # DISPLAY TOP 8 SIGNALS
                price_fmt = f"{signal['price']:.4f}" if signal['price'] < 1000 else f"{signal['price']:.2f}"
                print(f"{i}. {signal['symbol']} | {signal['signal']}")
                print(f"   CONFIDENCE: {signal['confidence']}% | PRICE: ${price_fmt}")
                print(f"   RSI: {signal['rsi']} | VOLUME: {signal['volume_ratio']}x | VOLATILITY: {signal['volatility']}%")
                print(f"   1H CHANGE: {signal['price_change_1h']}% | REGIME: {signal['market_regime']}")
                print(f"   CONDITIONS: {', '.join(signal['conditions_met'][:3])}")
                print()
        else:
            print("⚠️  NO HIGH-CONFIDENCE SIGNALS DETECTED")
            print("💡 AI IS WAITING FOR OPTIMAL MARKET CONDITIONS")
            print("🎯 ADAPTIVE THRESHOLD:", self.signal_engine.adaptive_thresholds[self.signal_engine.market_regime])
        
        # PERFORMANCE STATS
        stats = self.execution_engine.get_performance_stats()
        if stats:
            print(f"📈 PERFORMANCE: {stats['signals_per_cycle']:.1f} signals/cycle | "
                  f"Avg Confidence: {stats['avg_confidence']:.1f}% | "
                  f"Efficiency: {stats['efficiency_ratio']:.2f}")
        
        # NEXT CYCLE COUNTDOWN
        if self.last_analysis:
            next_analysis = self.last_analysis + timedelta(minutes=EliteConfig.ANALYSIS_INTERVAL)
            time_remaining = (next_analysis - datetime.utcnow()).total_seconds()
            if time_remaining > 0:
                mins = int(time_remaining // 60)
                secs = int(time_remaining % 60)
                print(f"⏰ NEXT ANALYSIS: {mins:02d}:{secs:02d}")
    
    def _display_final_stats(self):
        """COMPREHENSIVE FINAL STATISTICS"""
        stats = self.execution_engine.get_performance_stats() or {}
        
        print(f"\n{'='*80}")
        print("📊 MISSION SUMMARY - MARKET MASTER AI")
        print(f"{'='*80}")
        print(f"✅ TOTAL ANALYSIS CYCLES: {stats.get('total_cycles', 0)}")
        print(f"✅ TOTAL SIGNALS GENERATED: {stats.get('total_signals', 0)}")
        print(f"✅ AVERAGE SIGNALS PER CYCLE: {stats.get('signals_per_cycle', 0.0):.2f}")
        print(f"✅ AVERAGE CONFIDENCE: {stats.get('avg_confidence', 0.0):.2f}%")
        print(f"✅ SYSTEM EFFICIENCY: {stats.get('efficiency_ratio', 0.0):.2f}")
        print(f"✅ FINAL MARKET REGIME: {self.signal_engine.market_regime}")
        print(f"{'='*80}")
        print("🎯 AI TRADING SYSTEM - MISSION ACCOMPLISHED")

# =============================================================================
# EXECUTION COMMAND - ABSOLUTE AUTHORITY
# =============================================================================
if __name__ == "__main__":
    try:
        # INSTANTIATE AND DEPLOY
        ai_system = MarketMasterAI()
        
        # DEPLOYMENT CONFIRMATION
        print("\n" + "="*80)
        print("🚀 MARKET MASTER AI - DEPLOYMENT INITIATED")
        print("⚡ INSTITUTIONAL-GRADE TRADING ALGORITHM")
        print("🎯 ADAPTIVE AI-POWERED SIGNAL GENERATION")
        print("="*80)
        print("💡 SYSTEM STATUS: OPERATIONAL")
        print("🎯 ANALYSIS INTERVAL: 1.5 MINUTES")
        print("📊 SYMBOL UNIVERSE: 15-25 DYNAMIC ASSETS")
        print("⚡ CONFIDENCE THRESHOLD: ADAPTIVE (68-75%)")
        print("="*80)
        print("⏰ INITIALIZING CONTINUOUS ANALYSIS...")
        time.sleep(2)
        
        # ENGAGE PERPETUAL ANALYSIS
        ai_system.run_continuous_analysis()
        
    except Exception as e:
        print(f"💥 CRITICAL DEPLOYMENT FAILURE: {str(e)}")
        print("🚨 SYSTEM OFFLINE - IMMEDIATE ATTENTION REQUIRED")
