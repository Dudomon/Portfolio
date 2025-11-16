import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeaturesAnalyzer:
    """
    🚀 ANALISADOR DE FEATURES AVANÇADAS
    
    Implementa features complementares para completar a arquitetura de 2580 dimensões:
    - Pattern Recognition Avançado (8 features)
    - Regime Detection Refinado (6 features)
    - Risk Metrics Expandidos (4 features)
    - Temporal Context Enhanced (2 features)
    
    Total: 20 features adicionais
    """
    
    def __init__(self, window_size: int = 25):
        self.window_size = window_size
        
    def calculate_pattern_recognition_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🎯 PATTERN RECOGNITION AVANÇADO (8 features)
        
        Identifica padrões complexos de preço e volume:
        - Candlestick patterns
        - Support/resistance levels
        - Breakout patterns
        - Volume patterns
        """
        try:
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 10:
                return np.full(8, 0.4, dtype=np.float32)
            
            window_data = df.iloc[start_idx:end_idx].copy()
            if len(window_data) < 5:
                return np.full(8, 0.4, dtype=np.float32)
            
            # 1. CANDLESTICK PATTERN STRENGTH
            body_size = abs(window_data['close'] - window_data['open']) / window_data['close']
            shadow_size = (window_data['high'] - window_data['low']) / window_data['close'] - body_size
            candlestick_strength = (body_size / (shadow_size + 1e-6)).rolling(3, min_periods=1).mean()
            pattern_1 = max(0.0, min(1.0, np.tanh(candlestick_strength.iloc[-1] * 2) * 0.5 + 0.5))
            
            # 2. SUPPORT/RESISTANCE PROXIMITY
            recent_lows = window_data['low'].rolling(10, min_periods=1).min()
            recent_highs = window_data['high'].rolling(10, min_periods=1).max()
            current_price = window_data['close'].iloc[-1]
            support_distance = (current_price - recent_lows.iloc[-1]) / current_price
            resistance_distance = (recent_highs.iloc[-1] - current_price) / current_price
            pattern_2 = max(0.0, min(1.0, 1 - min(support_distance, resistance_distance) * 10))
            
            # 3. BREAKOUT PATTERN DETECTION
            price_range = window_data['high'] - window_data['low']
            avg_range = price_range.rolling(10, min_periods=1).mean()
            current_range = price_range.iloc[-1]
            volume_spike = window_data['volume'].iloc[-1] / window_data['volume'].rolling(10, min_periods=1).mean().iloc[-1]
            breakout_signal = (current_range / avg_range.iloc[-1]) * np.sqrt(volume_spike)
            pattern_3 = max(0.0, min(1.0, np.tanh(breakout_signal - 1) * 0.4 + 0.5))
            
            # 4. VOLUME PATTERN ANALYSIS
            volume_ma = window_data['volume'].rolling(5, min_periods=1).mean()
            volume_trend = volume_ma.pct_change().rolling(3, min_periods=1).mean()
            price_volume_correlation = np.corrcoef(
                window_data['close'].pct_change().fillna(0).values,
                window_data['volume'].pct_change().fillna(0).values
            )[0, 1] if len(window_data) > 3 else 0.5
            if np.isnan(price_volume_correlation):
                price_volume_correlation = 0.5
            pattern_4 = max(0.0, min(1.0, (price_volume_correlation + 1) * 0.5))
            
            # 5. FIBONACCI RETRACEMENT LEVELS
            recent_high = window_data['high'].rolling(15, min_periods=1).max().iloc[-1]
            recent_low = window_data['low'].rolling(15, min_periods=1).min().iloc[-1]
            price_range_fib = recent_high - recent_low
            if price_range_fib > 0:
                fib_levels = [recent_low + price_range_fib * level for level in [0.236, 0.382, 0.618, 0.786]]
                fib_proximity = min([abs(current_price - level) / current_price for level in fib_levels])
                pattern_5 = max(0.0, min(1.0, 1 - fib_proximity * 20))
            else:
                pattern_5 = 0.5
            
            # 6. MOMENTUM DIVERGENCE PATTERN
            price_momentum = window_data['close'].pct_change().rolling(5, min_periods=1).sum()
            volume_momentum = window_data['volume'].pct_change().rolling(5, min_periods=1).sum()
            momentum_divergence = abs(np.sign(price_momentum.iloc[-1]) - np.sign(volume_momentum.iloc[-1]))
            pattern_6 = max(0.0, min(1.0, momentum_divergence * 0.5 + 0.3))
            
            # 7. GAP ANALYSIS
            gaps = abs(window_data['open'] - window_data['close'].shift(1)) / window_data['close'].shift(1)
            gaps = gaps.fillna(0)
            gap_significance = gaps.rolling(5, min_periods=1).mean()
            pattern_7 = max(0.0, min(1.0, np.tanh(gap_significance.iloc[-1] * 100) * 0.6 + 0.2))
            
            # 8. TREND CHANNEL ANALYSIS
            highs_trend = window_data['high'].rolling(10, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
            )
            lows_trend = window_data['low'].rolling(10, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
            )
            channel_convergence = abs(highs_trend.iloc[-1] - lows_trend.iloc[-1]) if not pd.isna(highs_trend.iloc[-1]) else 0
            pattern_8 = max(0.0, min(1.0, 1 - np.tanh(channel_convergence * 1000) * 0.5))
            
            return np.array([
                pattern_1, pattern_2, pattern_3, pattern_4,
                pattern_5, pattern_6, pattern_7, pattern_8
            ], dtype=np.float32)
            
        except Exception:
            return np.full(8, 0.4, dtype=np.float32)
    
    def calculate_regime_detection_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        📊 REGIME DETECTION REFINADO (6 features)
        
        Detecta regimes de mercado com maior precisão:
        - Trend regime strength
        - Volatility regime classification
        - Volume regime identification
        - Cyclical pattern detection
        """
        try:
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 15:
                return np.full(6, 0.5, dtype=np.float32)
            
            window_data = df.iloc[start_idx:end_idx].copy()
            if len(window_data) < 10:
                return np.full(6, 0.5, dtype=np.float32)
            
            # 1. TREND REGIME STRENGTH
            returns = window_data['close'].pct_change().fillna(0)
            trend_strength = abs(returns.rolling(10, min_periods=1).mean().iloc[-1])
            trend_consistency = (returns > 0).rolling(10, min_periods=1).mean()
            regime_1 = max(0.0, min(1.0, trend_strength * 100 * abs(trend_consistency.iloc[-1] - 0.5) * 2))
            
            # 2. VOLATILITY REGIME CLASSIFICATION
            volatility = returns.rolling(5, min_periods=1).std()
            vol_percentile = (volatility < volatility.iloc[-1]).mean()
            vol_regime = 1 if vol_percentile > 0.8 else (0.5 if vol_percentile > 0.2 else 0)
            regime_2 = vol_regime
            
            # 3. VOLUME REGIME IDENTIFICATION
            volume_ma = window_data['volume'].rolling(10, min_periods=1).mean()
            volume_regime_indicator = window_data['volume'].iloc[-1] / volume_ma.iloc[-1]
            regime_3 = max(0.0, min(1.0, np.tanh(volume_regime_indicator - 1) * 0.5 + 0.5))
            
            # 4. CYCLICAL PATTERN DETECTION
            price_cycles = (window_data['close'] / window_data['close'].rolling(5, min_periods=1).mean() - 1).fillna(0)
            cycle_amplitude = price_cycles.std()
            cycle_frequency = (price_cycles.diff().abs() > cycle_amplitude * 0.5).sum()
            regime_4 = max(0.0, min(1.0, cycle_frequency / len(price_cycles)))
            
            # 5. MEAN REVERSION TENDENCY
            price_deviation = (window_data['close'] - window_data['close'].rolling(15, min_periods=1).mean()).fillna(0)
            reversion_speed = abs(price_deviation.diff()).mean()
            regime_5 = max(0.0, min(1.0, reversion_speed / (price_deviation.std() + 1e-6)))
            
            # 6. MARKET EFFICIENCY MEASURE
            random_walk = np.cumsum(np.random.normal(0, returns.std(), len(returns)))
            actual_path = returns.cumsum()
            efficiency = 1 - np.corrcoef(random_walk, actual_path)[0, 1] if len(returns) > 5 else 0.5
            if np.isnan(efficiency):
                efficiency = 0.5
            regime_6 = max(0.0, min(1.0, (efficiency + 1) * 0.5))
            
            return np.array([
                regime_1, regime_2, regime_3,
                regime_4, regime_5, regime_6
            ], dtype=np.float32)
            
        except Exception:
            return np.full(6, 0.5, dtype=np.float32)
    
    def calculate_risk_metrics_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        ⚠️ RISK METRICS EXPANDIDOS (4 features)
        
        Métricas de risco avançadas:
        - Value at Risk (VaR)
        - Expected Shortfall
        - Risk-adjusted returns
        - Tail risk measures
        """
        try:
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 10:
                return np.full(4, 0.3, dtype=np.float32)
            
            window_data = df.iloc[start_idx:end_idx].copy()
            if len(window_data) < 8:
                return np.full(4, 0.3, dtype=np.float32)
            
            returns = window_data['close'].pct_change().fillna(0)
            
            # 1. VALUE AT RISK (95% confidence)
            var_95 = np.percentile(returns, 5) if len(returns) > 5 else -0.02
            risk_1 = max(0.0, min(1.0, abs(var_95) * 50))
            
            # 2. EXPECTED SHORTFALL (conditional VaR)
            tail_returns = returns[returns <= var_95]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_95
            risk_2 = max(0.0, min(1.0, abs(expected_shortfall) * 40))
            
            # 3. RISK-ADJUSTED RETURNS (Sharpe-like ratio)
            mean_return = returns.mean()
            return_volatility = returns.std()
            risk_adj_return = mean_return / (return_volatility + 1e-6)
            risk_3 = max(0.0, min(1.0, np.tanh(risk_adj_return * 5) * 0.5 + 0.5))
            
            # 4. TAIL RISK MEASURE (kurtosis proxy)
            if len(returns) > 8:
                returns_centered = returns - returns.mean()
                fourth_moment = (returns_centered ** 4).mean()
                second_moment = (returns_centered ** 2).mean()
                kurtosis = fourth_moment / (second_moment ** 2 + 1e-6) - 3  # Excess kurtosis
                tail_risk = max(0.0, min(1.0, kurtosis / 10 + 0.5))
            else:
                tail_risk = 0.5
            risk_4 = tail_risk
            
            return np.array([risk_1, risk_2, risk_3, risk_4], dtype=np.float32)
            
        except Exception:
            return np.full(4, 0.3, dtype=np.float32)
    
    def calculate_temporal_context_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        ⏰ TEMPORAL CONTEXT ENHANCED (2 features)
        
        Contexto temporal avançado:
        - Session momentum
        - Intraday position
        """
        try:
            # 1. SESSION MOMENTUM (momentum within trading session)
            if current_idx >= 20:
                session_start = max(0, current_idx - 20)  # Aproximadamente uma sessão
                session_data = df.iloc[session_start:current_idx + 1]
                session_return = (session_data['close'].iloc[-1] / session_data['close'].iloc[0] - 1)
                temporal_1 = max(0.0, min(1.0, np.tanh(session_return * 20) * 0.5 + 0.5))
            else:
                temporal_1 = 0.5
            
            # 2. INTRADAY POSITION (position within the day cycle)
            # Simular posição intraday baseada no índice atual
            intraday_cycle = (current_idx % 288) / 288.0  # 288 = 24h * 60min / 5min
            intraday_position = np.sin(2 * np.pi * intraday_cycle) * 0.3 + 0.5
            temporal_2 = max(0.0, min(1.0, intraday_position))
            
            return np.array([temporal_1, temporal_2], dtype=np.float32)
            
        except Exception:
            return np.full(2, 0.5, dtype=np.float32)
    
    def get_all_enhanced_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🎯 COMBINAR TODAS AS FEATURES AVANÇADAS
        
        Returns:
            np.ndarray: 20 features (8 pattern + 6 regime + 4 risk + 2 temporal)
        """
        try:
            pattern_features = self.calculate_pattern_recognition_features(df, current_idx)
            regime_features = self.calculate_regime_detection_features(df, current_idx)
            risk_features = self.calculate_risk_metrics_features(df, current_idx)
            temporal_features = self.calculate_temporal_context_features(df, current_idx)
            
            return np.concatenate([
                pattern_features,
                regime_features,
                risk_features,
                temporal_features
            ])
            
        except Exception:
            # Valores padrão seguros em caso de erro
            return np.full(20, 0.4, dtype=np.float32)


def create_enhanced_analyzer() -> EnhancedFeaturesAnalyzer:
    """Factory function para criar analisador de features avançadas"""
    return EnhancedFeaturesAnalyzer(window_size=25)