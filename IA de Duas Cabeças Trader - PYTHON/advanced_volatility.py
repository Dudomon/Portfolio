import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedVolatilityAnalyzer:
    """
    📈 ANALISADOR DE VOLATILIDADE AVANÇADA
    
    Implementa análise GARCH e clustering de volatilidade para capturar
    dinâmicas estruturais ausentes no sistema atual.
    
    Features implementadas:
    - GARCH Signal (1 feature)
    - Volatility Breakout Indicator (1 feature)
    - Volatility Clustering Strength (1 feature)
    - Realized vs Implied Volatility (1 feature)
    - Volatility Surface Skew (1 feature)
    """
    
    def __init__(self, window_size: int = 20, garch_window: int = 50):
        self.window_size = window_size
        self.garch_window = garch_window
        self.volatility_history = []
        self.returns_history = []
        
    def calculate_garch_signal(self, returns: np.ndarray) -> float:
        """
        📊 GARCH(1,1) SIGNAL
        
        Calcula sinal GARCH usando aproximação simplificada baseada em:
        - Heterocedasticidade condicional
        - Clustering de volatilidade
        - Persistência de volatilidade
        """
        try:
            if len(returns) < 10:
                return 0.5
            
            # Calcular squared returns (proxy para volatilidade)
            squared_returns = returns ** 2
            
            # GARCH(1,1) simplificado: sigma²_t = α₀ + α₁*ε²_{t-1} + β₁*σ²_{t-1}
            # Usar moving averages como proxy
            
            # α₀ (long-term variance)
            long_term_var = np.mean(squared_returns)
            
            # α₁ (reaction to news - ARCH effect)
            arch_effect = np.corrcoef(squared_returns[1:], squared_returns[:-1])[0, 1] if len(squared_returns) > 1 else 0.1
            arch_effect = max(0.0, min(0.3, arch_effect))  # Bound between 0-0.3
            
            # β₁ (persistence - GARCH effect)
            volatility_ma = pd.Series(squared_returns).rolling(5, min_periods=1).mean()
            garch_persistence = np.corrcoef(volatility_ma[1:], volatility_ma[:-1])[0, 1] if len(volatility_ma) > 1 else 0.7
            garch_persistence = max(0.3, min(0.95, garch_persistence))  # Bound between 0.3-0.95
            
            # Current conditional variance
            recent_shock = squared_returns[-1] if len(squared_returns) > 0 else long_term_var
            recent_volatility = volatility_ma.iloc[-1] if len(volatility_ma) > 0 else long_term_var
            
            # GARCH signal: how far current volatility deviates from expected
            expected_variance = long_term_var * (1 - arch_effect - garch_persistence) + arch_effect * recent_shock + garch_persistence * recent_volatility
            current_variance = recent_volatility
            
            garch_signal = current_variance / (expected_variance + 1e-8)
            garch_signal = max(0.0, min(1.0, np.tanh(garch_signal - 1) * 0.5 + 0.5))
            
            return float(garch_signal)
            
        except Exception:
            return 0.5
    
    def calculate_volatility_breakout(self, returns: np.ndarray, window: int = 20) -> float:
        """
        🚨 VOLATILITY BREAKOUT INDICATOR
        
        Detecta quebras estruturais na volatilidade usando:
        - Rolling volatility percentiles
        - Sudden volatility regime changes
        - Volatility expansion patterns
        """
        try:
            if len(returns) < window:
                return 0.3
            
            # Calculate rolling volatility
            squared_returns = returns ** 2
            rolling_vol = pd.Series(squared_returns).rolling(5, min_periods=1).mean()
            
            # Current volatility percentile
            current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else np.mean(squared_returns)
            historical_vol = rolling_vol.iloc[:-5] if len(rolling_vol) > 5 else rolling_vol
            
            if len(historical_vol) > 0:
                vol_percentile = (historical_vol < current_vol).mean()
            else:
                vol_percentile = 0.5
            
            # Volatility acceleration (rate of change in volatility)
            if len(rolling_vol) >= 3:
                vol_acceleration = (rolling_vol.iloc[-1] - rolling_vol.iloc[-3]) / (rolling_vol.iloc[-3] + 1e-8)
                vol_acceleration = np.tanh(vol_acceleration * 10) * 0.3  # Scale and bound
            else:
                vol_acceleration = 0.0
            
            # Volatility clustering strength (how persistent high vol periods are)
            high_vol_threshold = np.quantile(rolling_vol, 0.7) if len(rolling_vol) > 3 else current_vol
            high_vol_periods = (rolling_vol > high_vol_threshold).astype(int)
            
            if len(high_vol_periods) >= 5:
                clustering_strength = high_vol_periods.rolling(5, min_periods=1).sum().iloc[-1] / 5.0
            else:
                clustering_strength = 0.5
            
            # Combine indicators
            breakout_signal = vol_percentile * 0.5 + clustering_strength * 0.3 + (vol_acceleration + 0.3) * 0.2
            breakout_signal = max(0.0, min(1.0, breakout_signal))
            
            return float(breakout_signal)
            
        except Exception:
            return 0.3
    
    def calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """
        🔄 VOLATILITY CLUSTERING STRENGTH
        
        Mede a força do clustering de volatilidade usando:
        - Autocorrelation of squared returns
        - Persistence of high/low volatility regimes
        - Volatility regime switching probability
        """
        try:
            if len(returns) < 10:
                return 0.4
            
            squared_returns = returns ** 2
            
            # Autocorrelation of squared returns (Ljung-Box style)
            autocorrs = []
            for lag in range(1, min(6, len(squared_returns))):
                if len(squared_returns) > lag:
                    corr = np.corrcoef(squared_returns[lag:], squared_returns[:-lag])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))
            
            avg_autocorr = np.mean(autocorrs) if autocorrs else 0.2
            
            # Volatility regime persistence
            vol_median = np.median(squared_returns)
            high_vol_regime = (squared_returns > vol_median).astype(int)
            
            # Count regime switches
            regime_switches = np.sum(np.diff(high_vol_regime) != 0)
            max_switches = len(high_vol_regime) - 1
            switch_rate = regime_switches / max_switches if max_switches > 0 else 0.5
            
            # Lower switch rate = higher clustering
            regime_persistence = 1.0 - switch_rate
            
            # Volatility of volatility (vol clustering creates vol-of-vol)
            if len(squared_returns) >= 5:
                rolling_vol = pd.Series(squared_returns).rolling(3, min_periods=1).mean()
                vol_of_vol = rolling_vol.std() / (rolling_vol.mean() + 1e-8)
                vol_of_vol = np.tanh(vol_of_vol * 5) * 0.5 + 0.25  # Scale appropriately
            else:
                vol_of_vol = 0.4
            
            # Combine clustering measures
            clustering_strength = avg_autocorr * 0.4 + regime_persistence * 0.4 + vol_of_vol * 0.2
            clustering_strength = max(0.0, min(1.0, clustering_strength))
            
            return float(clustering_strength)
            
        except Exception:
            return 0.4
    
    def calculate_realized_vs_implied(self, returns: np.ndarray, timeframe_minutes: int = 5) -> float:
        """
        📊 REALIZED VS IMPLIED VOLATILITY
        
        Compara volatilidade realizada com implied usando:
        - Realized volatility from returns
        - Implied volatility proxy from option-like behavior
        - Volatility risk premium estimation
        """
        try:
            if len(returns) < 10:
                return 0.5
            
            # Realized volatility (annualized)
            returns_clean = returns[~np.isnan(returns)]
            if len(returns_clean) == 0:
                return 0.5
            
            # Calculate realized vol (annualized from 5-minute returns)
            periods_per_day = 288  # 24*60/5 = 288 five-minute periods per day
            periods_per_year = periods_per_day * 252  # 252 trading days
            
            realized_vol = np.std(returns_clean) * np.sqrt(periods_per_year)
            
            # Implied volatility proxy using option-like measures
            # Use extreme movements and skewness as implied vol proxy
            price_changes = returns_clean
            
            # Tail risk premium (extreme movements suggest higher implied vol)
            tail_threshold = np.quantile(np.abs(price_changes), 0.9) if len(price_changes) > 5 else np.std(price_changes)
            tail_events = np.sum(np.abs(price_changes) > tail_threshold)
            tail_risk_premium = tail_events / len(price_changes)
            
            # Skewness premium (asymmetry suggests vol risk premium)
            skewness = pd.Series(price_changes).skew() if len(price_changes) > 3 else 0.0
            skew_premium = abs(skewness) * 0.1  # Scale skewness to vol premium
            
            # Forward-looking volatility estimation (using recent volatility trends)
            if len(price_changes) >= 10:
                recent_vol = np.std(price_changes[-10:]) * np.sqrt(periods_per_year)
                vol_trend = recent_vol / (realized_vol + 1e-8)
            else:
                vol_trend = 1.0
            
            # Implied vol proxy
            implied_vol_proxy = realized_vol * (1 + tail_risk_premium + skew_premium) * vol_trend
            
            # Volatility ratio (realized vs implied)
            vol_ratio = realized_vol / (implied_vol_proxy + 1e-8)
            
            # Transform to [0,1] range where 0.5 = realized == implied
            vol_comparison = np.tanh((vol_ratio - 1) * 2) * 0.4 + 0.5
            vol_comparison = max(0.0, min(1.0, vol_comparison))
            
            return float(vol_comparison)
            
        except Exception:
            return 0.5
    
    def calculate_volatility_skew(self, returns: np.ndarray) -> float:
        """
        📈 VOLATILITY SURFACE SKEW
        
        Estima assimetria da superfície de volatilidade usando:
        - Return skewness (put-call skew proxy)
        - Volatility term structure
        - Risk reversal patterns
        """
        try:
            if len(returns) < 10:
                return 0.5
            
            returns_clean = returns[~np.isnan(returns)]
            if len(returns_clean) < 5:
                return 0.5
            
            # Return skewness (proxy for volatility skew)
            skewness = pd.Series(returns_clean).skew()
            if np.isnan(skewness):
                skewness = 0.0
            
            # Normalize skewness to [0,1] where 0.5 = no skew
            normalized_skew = np.tanh(skewness) * 0.4 + 0.5
            
            # Volatility term structure slope
            if len(returns_clean) >= 15:
                short_term_vol = np.std(returns_clean[-5:])   # Last 5 periods
                medium_term_vol = np.std(returns_clean[-10:]) # Last 10 periods  
                long_term_vol = np.std(returns_clean[-15:])   # Last 15 periods
                
                # Term structure slope (contango vs backwardation)
                if long_term_vol > 0:
                    term_structure = (short_term_vol - long_term_vol) / long_term_vol
                    term_structure = np.tanh(term_structure * 3) * 0.3 + 0.5
                else:
                    term_structure = 0.5
            else:
                term_structure = 0.5
            
            # Risk reversal proxy (put-call asymmetry)
            # Use asymmetry in positive vs negative returns
            positive_returns = returns_clean[returns_clean > 0]
            negative_returns = returns_clean[returns_clean < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                pos_vol = np.std(positive_returns)
                neg_vol = np.std(negative_returns)
                
                # Risk reversal: negative moves typically have higher volatility
                risk_reversal = neg_vol / (pos_vol + 1e-8)
                risk_reversal = np.tanh((risk_reversal - 1) * 2) * 0.3 + 0.5
            else:
                risk_reversal = 0.5
            
            # Combine skew measures
            volatility_skew = normalized_skew * 0.4 + term_structure * 0.3 + risk_reversal * 0.3
            volatility_skew = max(0.0, min(1.0, volatility_skew))
            
            return float(volatility_skew)
            
        except Exception:
            return 0.5
    
    def get_all_volatility_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🎯 COMBINAR TODAS AS FEATURES DE VOLATILIDADE AVANÇADA
        
        Returns:
            np.ndarray: 5 features (GARCH + breakout + clustering + realized_vs_implied + skew)
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.garch_window)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 10:
                return np.array([0.5, 0.3, 0.4, 0.5, 0.5], dtype=np.float32)
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 5:
                return np.array([0.5, 0.3, 0.4, 0.5, 0.5], dtype=np.float32)
            
            # Calcular returns
            returns = window_data['close'].pct_change().fillna(0).values
            
            # Calcular todas as features
            garch_signal = self.calculate_garch_signal(returns)
            vol_breakout = self.calculate_volatility_breakout(returns)
            vol_clustering = self.calculate_volatility_clustering(returns)
            realized_vs_implied = self.calculate_realized_vs_implied(returns)
            vol_skew = self.calculate_volatility_skew(returns)
            
            return np.array([
                garch_signal,
                vol_breakout,
                vol_clustering,
                realized_vs_implied,
                vol_skew
            ], dtype=np.float32)
            
        except Exception:
            # Valores padrão seguros em caso de erro
            return np.array([0.5, 0.3, 0.4, 0.5, 0.5], dtype=np.float32)


def create_volatility_analyzer() -> AdvancedVolatilityAnalyzer:
    """Factory function para criar analisador de volatilidade avançada"""
    return AdvancedVolatilityAnalyzer(window_size=20, garch_window=50)