import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeMomentumAnalyzer:
    """
    ⚡ ANALISADOR DE MOMENTUM MULTI-TIMEFRAME
    
    Implementa análise de confluência de momentum entre diferentes timeframes
    para capturar dinâmicas temporais ausentes no sistema atual.
    
    Features implementadas:
    - 1min-5min Momentum Confluence (1 feature)
    - 5min-15min Momentum Confluence (1 feature)
    - 15min-1h Momentum Confluence (1 feature)
    - Momentum Divergence Strength (1 feature)
    - Acceleration Pattern (1 feature)
    - Momentum Sustainability (1 feature)
    """
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.momentum_history = []
        
        # Timeframe multipliers para simulação
        self.tf_1m_factor = 0.2  # 1min é mais volátil
        self.tf_5m_factor = 1.0  # Base (nossos dados)
        self.tf_15m_factor = 3.0  # 15min é mais suave
        self.tf_1h_factor = 12.0  # 1h é muito mais suave
        
    def calculate_momentum_1m_5m_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        ⚡ 1MIN-5MIN MOMENTUM CONFLUENCE
        
        Calcula confluência de momentum entre timeframes ultra-curtos usando:
        - Short-term momentum alignment
        - Intrabar momentum patterns
        - Micro-trend consistency
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 10:
                return 0.5  # Confluence neutra
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 5:
                return 0.5
            
            # 5min momentum (nossos dados base)
            returns_5m = window_data['close'].pct_change().fillna(0)
            momentum_5m = returns_5m.rolling(3, min_periods=1).sum()  # 3-period momentum
            
            # 1min momentum simulado (mais granular, mais volátil)
            # Simular intrabar movements usando high-low ranges
            intrabar_range = (window_data['high'] - window_data['low']) / window_data['close']
            intrabar_momentum = (window_data['close'] - window_data['open']) / window_data['close']
            
            # 1min proxy: combinar intrabar momentum com noise
            momentum_1m_proxy = (
                intrabar_momentum * self.tf_1m_factor +  # Intrabar component
                returns_5m * 0.3 +  # Base trend
                (intrabar_range.pct_change().fillna(0)) * 0.2 +  # Volatility component
                np.random.normal(0, 0.005, len(returns_5m)) * 0.1  # 1min noise
            )
            
            # Smooth 1min proxy para simular múltiplas 1min bars dentro de 5min bar
            momentum_1m = momentum_1m_proxy.rolling(2, min_periods=1).mean()
            
            # Calcular confluência (correlação entre momentums)
            if len(momentum_5m) >= 5 and len(momentum_1m) >= 5:
                confluence = np.corrcoef(momentum_5m.values, momentum_1m.values)[0, 1]
                if np.isnan(confluence):
                    confluence = 0.5
            else:
                confluence = 0.5
            
            # Normalizar para [0,1] onde 0.5 = sem confluência
            confluence = max(-0.98, min(0.98, confluence))
            normalized_confluence = (confluence + 1) * 0.5
            
            return float(normalized_confluence)
            
        except Exception:
            return 0.5
    
    def calculate_momentum_5m_15m_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        📊 5MIN-15MIN MOMENTUM CONFLUENCE
        
        Calcula confluência entre timeframes de curto prazo usando:
        - Medium-term trend alignment
        - Momentum persistence patterns
        - Trend confirmation signals
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 15:
                return 0.6  # Confluência ligeiramente positiva default
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 8:
                return 0.6
            
            # 5min momentum (nossos dados)
            returns_5m = window_data['close'].pct_change().fillna(0)
            momentum_5m = returns_5m.rolling(5, min_periods=1).sum()  # 5-period momentum
            
            # 15min momentum simulado (cada 15min = 3 barras de 5min)
            # Usar rolling windows maiores para simular timeframe maior
            returns_15m_proxy = returns_5m.rolling(3, min_periods=1).sum()  # 3x5min = 15min
            momentum_15m = returns_15m_proxy.rolling(3, min_periods=1).mean()  # Smoother
            
            # Adicionar características de 15min timeframe
            volume_15m_proxy = window_data['volume'].rolling(3, min_periods=1).mean()
            volume_momentum = volume_15m_proxy.pct_change().fillna(0)
            
            # 15min momentum refinado
            momentum_15m_refined = momentum_15m + volume_momentum * 0.1
            
            # Calcular confluência
            if len(momentum_5m) >= 8 and len(momentum_15m_refined) >= 8:
                confluence = np.corrcoef(momentum_5m.values, momentum_15m_refined.values)[0, 1]
                if np.isnan(confluence):
                    confluence = 0.6
            else:
                confluence = 0.6
            
            # Adicionar trend strength factor (confluência mais forte quando ambos trends são fortes)
            trend_strength_5m = abs(momentum_5m.iloc[-1]) if len(momentum_5m) > 0 else 0.01
            trend_strength_15m = abs(momentum_15m_refined.iloc[-1]) if len(momentum_15m_refined) > 0 else 0.01
            combined_strength = np.sqrt(trend_strength_5m * trend_strength_15m)
            
            # Ajustar confluência pela força combinada
            confluence_adjusted = confluence * (1 + combined_strength * 10)
            confluence_adjusted = max(-0.98, min(0.98, confluence_adjusted))
            
            # Normalizar para [0,1]
            normalized_confluence = (confluence_adjusted + 1) * 0.5
            
            return float(normalized_confluence)
            
        except Exception:
            return 0.6
    
    def calculate_momentum_15m_1h_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        📈 15MIN-1H MOMENTUM CONFLUENCE
        
        Calcula confluência entre timeframes de médio prazo usando:
        - Long-term trend alignment
        - Structural momentum patterns
        - Major trend confirmation
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 20:
                return 0.7  # Confluência positiva default para timeframes maiores
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 12:
                return 0.7
            
            # 15min momentum simulado (rolling window médio)
            returns_15m_proxy = window_data['close'].pct_change().rolling(3, min_periods=1).sum()
            momentum_15m = returns_15m_proxy.rolling(5, min_periods=1).mean()
            
            # 1h momentum simulado (cada 1h = 12 barras de 5min)
            returns_1h_proxy = window_data['close'].pct_change().rolling(12, min_periods=1).sum()
            momentum_1h = returns_1h_proxy.rolling(3, min_periods=1).mean()  # Very smooth
            
            # Adicionar características estruturais de 1h
            structural_trend = (window_data['close'] / window_data['close'].rolling(20, min_periods=1).mean() - 1).fillna(0)
            momentum_1h_refined = momentum_1h + structural_trend * 0.1
            
            # Volume pattern para 1h (institutional activity proxy)
            volume_1h_proxy = window_data['volume'].rolling(12, min_periods=1).mean()
            volume_trend_1h = volume_1h_proxy.pct_change().fillna(0)
            momentum_1h_refined += volume_trend_1h * 0.05
            
            # Calcular confluência
            if len(momentum_15m) >= 12 and len(momentum_1h_refined) >= 12:
                confluence = np.corrcoef(momentum_15m.values, momentum_1h_refined.values)[0, 1]
                if np.isnan(confluence):
                    confluence = 0.7
            else:
                confluence = 0.7
            
            # Major trend confirmation bonus (quando ambos trends apontam mesma direção)
            direction_15m = 1 if momentum_15m.iloc[-1] > 0 else -1 if momentum_15m.iloc[-1] < 0 else 0
            direction_1h = 1 if momentum_1h_refined.iloc[-1] > 0 else -1 if momentum_1h_refined.iloc[-1] < 0 else 0
            
            direction_alignment = 1 if direction_15m == direction_1h and direction_15m != 0 else 0.8
            confluence_final = confluence * direction_alignment
            
            # Normalizar para [0,1]
            confluence_final = max(-0.98, min(0.98, confluence_final))
            normalized_confluence = (confluence_final + 1) * 0.5
            
            return float(normalized_confluence)
            
        except Exception:
            return 0.7
    
    def calculate_momentum_divergence_strength(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        🔄 MOMENTUM DIVERGENCE STRENGTH
        
        Mede força das divergências entre timeframes usando:
        - Cross-timeframe momentum conflicts
        - Divergence persistence
        - Reversal signal strength
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 15:
                return 0.2  # Low divergence default
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 10:
                return 0.2
            
            # Calcular momentums para diferentes timeframes
            returns = window_data['close'].pct_change().fillna(0)
            
            # Short-term momentum (5min equivalent)
            momentum_short = returns.rolling(3, min_periods=1).sum()
            
            # Medium-term momentum (15min equivalent) 
            momentum_medium = returns.rolling(9, min_periods=1).sum()
            
            # Long-term momentum (1h equivalent)
            momentum_long = returns.rolling(20, min_periods=1).sum()
            
            # Calcular divergências entre timeframes
            divergence_short_medium = abs(momentum_short - momentum_medium).rolling(5, min_periods=1).mean()
            divergence_medium_long = abs(momentum_medium - momentum_long).rolling(5, min_periods=1).mean()
            divergence_short_long = abs(momentum_short - momentum_long).rolling(5, min_periods=1).mean()
            
            # Divergence strength combinada
            current_div_sm = divergence_short_medium.iloc[-1] if len(divergence_short_medium) > 0 else 0.01
            current_div_ml = divergence_medium_long.iloc[-1] if len(divergence_medium_long) > 0 else 0.01
            current_div_sl = divergence_short_long.iloc[-1] if len(divergence_short_long) > 0 else 0.01
            
            # Weighted divergence (short-long divergence é mais significativa)
            divergence_strength = (
                current_div_sm * 0.2 +
                current_div_ml * 0.3 +
                current_div_sl * 0.5
            )
            
            # Normalize to [0,1]
            divergence_strength = min(1.0, divergence_strength * 20)  # Scale up
            
            return float(divergence_strength)
            
        except Exception:
            return 0.2
    
    def calculate_acceleration_pattern(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        🚀 ACCELERATION PATTERN
        
        Detecta padrões de aceleração/desaceleração usando:
        - Momentum acceleration (2nd derivative)
        - Velocity changes across timeframes
        - Trend strength evolution
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 10:
                return 0.4  # Neutral acceleration
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 8:
                return 0.4
            
            # Calcular momentum velocity (1st derivative of price)
            returns = window_data['close'].pct_change().fillna(0)
            momentum_velocity = returns.rolling(3, min_periods=1).mean()
            
            # Calcular momentum acceleration (2nd derivative of price)
            momentum_acceleration = momentum_velocity.diff().fillna(0)
            
            # Smooth acceleration to reduce noise
            smooth_acceleration = momentum_acceleration.rolling(3, min_periods=1).mean()
            
            # Current acceleration strength
            current_acceleration = smooth_acceleration.iloc[-1] if len(smooth_acceleration) > 0 else 0.0
            
            # Volume acceleration (confirma price acceleration)
            volume_velocity = window_data['volume'].pct_change().fillna(0).rolling(3, min_periods=1).mean()
            volume_acceleration = volume_velocity.diff().fillna(0)
            current_vol_accel = volume_acceleration.iloc[-1] if len(volume_acceleration) > 0 else 0.0
            
            # Combined acceleration pattern
            # Positive when price and volume accelerate together
            combined_acceleration = current_acceleration + current_vol_accel * 0.3
            
            # Trend consistency factor (acceleration mais significativa em trends fortes)
            trend_strength = abs(momentum_velocity.iloc[-1]) if len(momentum_velocity) > 0 else 0.01
            acceleration_significance = combined_acceleration * (1 + trend_strength * 10)
            
            # Transform to [0,1] where 0.5 = no acceleration
            acceleration_pattern = np.tanh(acceleration_significance * 100) * 0.4 + 0.5
            acceleration_pattern = max(0.0, min(1.0, acceleration_pattern))
            
            return float(acceleration_pattern)
            
        except Exception:
            return 0.4
    
    def calculate_momentum_sustainability(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        🔄 MOMENTUM SUSTAINABILITY
        
        Avalia sustentabilidade do momentum atual usando:
        - Momentum persistence measures
        - Energy depletion indicators
        - Reversal probability estimation
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 15:
                return 0.5  # Neutral sustainability
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 10:
                return 0.5
            
            # Calcular momentum e sua persistência
            returns = window_data['close'].pct_change().fillna(0)
            momentum = returns.rolling(5, min_periods=1).sum()
            
            # Momentum persistence (how long momentum maintains direction)
            momentum_direction = np.sign(momentum)
            direction_changes = (momentum_direction.diff() != 0).sum()
            total_periods = len(momentum_direction)
            persistence_ratio = 1 - (direction_changes / total_periods) if total_periods > 0 else 0.5
            
            # Momentum strength evolution (is momentum getting stronger or weaker?)
            momentum_strength = abs(momentum)
            strength_trend = momentum_strength.diff().rolling(3, min_periods=1).mean()
            current_strength_trend = strength_trend.iloc[-1] if len(strength_trend) > 0 else 0.0
            
            # Volume sustainability (momentum needs volume support)
            volume_trend = window_data['volume'].rolling(5, min_periods=1).mean()
            volume_support = (volume_trend.iloc[-1] / volume_trend.mean()) if len(volume_trend) > 0 and volume_trend.mean() > 0 else 1.0
            volume_sustainability = min(2.0, volume_support) / 2.0  # Normalize to [0,1]
            
            # Volatility sustainability (excessive volatility reduces sustainability)
            volatility = returns.rolling(5, min_periods=1).std()
            current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0.01
            avg_vol = volatility.mean() if len(volatility) > 0 else 0.01
            vol_ratio = current_vol / (avg_vol + 1e-8)
            vol_sustainability = max(0.2, min(1.0, 2.0 / (1 + vol_ratio)))  # Lower vol = more sustainable
            
            # Combined sustainability score
            sustainability = (
                persistence_ratio * 0.4 +
                (np.tanh(current_strength_trend * 100) * 0.5 + 0.5) * 0.3 +  # Strength trend
                volume_sustainability * 0.2 +
                vol_sustainability * 0.1
            )
            
            sustainability = max(0.0, min(1.0, sustainability))
            
            return float(sustainability)
            
        except Exception:
            return 0.5
    
    def get_all_momentum_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🎯 COMBINAR TODAS AS FEATURES DE MOMENTUM MULTI-TIMEFRAME
        
        Returns:
            np.ndarray: 6 features (1m-5m confluence + 5m-15m confluence + 15m-1h confluence + divergence + acceleration + sustainability)
        """
        try:
            # Calcular todas as features
            confluence_1m_5m = self.calculate_momentum_1m_5m_confluence(df, current_idx)
            confluence_5m_15m = self.calculate_momentum_5m_15m_confluence(df, current_idx)
            confluence_15m_1h = self.calculate_momentum_15m_1h_confluence(df, current_idx)
            divergence_strength = self.calculate_momentum_divergence_strength(df, current_idx)
            acceleration_pattern = self.calculate_acceleration_pattern(df, current_idx)
            momentum_sustainability = self.calculate_momentum_sustainability(df, current_idx)
            
            return np.array([
                confluence_1m_5m,
                confluence_5m_15m,
                confluence_15m_1h,
                divergence_strength,
                acceleration_pattern,
                momentum_sustainability
            ], dtype=np.float32)
            
        except Exception:
            # Valores padrão seguros em caso de erro
            return np.array([0.5, 0.6, 0.7, 0.2, 0.4, 0.5], dtype=np.float32)


def create_momentum_analyzer() -> MultiTimeframeMomentumAnalyzer:
    """Factory function para criar analisador de momentum multi-timeframe"""
    return MultiTimeframeMomentumAnalyzer(window_size=30)