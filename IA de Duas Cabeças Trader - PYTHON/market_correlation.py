import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MarketCorrelationAnalyzer:
    """
    🌐 ANALISADOR DE CORRELAÇÃO INTER-MERCADOS
    
    Implementa análise de correlação com mercados amplos para capturar
    contexto de mercado ausente no sistema atual.
    
    Features implementadas:
    - SPY Correlation (1 feature)
    - Sector Correlation (1 feature)  
    - VIX Divergence (1 feature)
    - Dollar Strength Impact (1 feature)
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.correlation_history = []
        
        # 🎯 MOCK DATA: Simulação de índices de mercado
        # Em produção, estes dados viriam de APIs reais (Yahoo Finance, Alpha Vantage, etc.)
        self.spy_proxy_multiplier = 1.0  # Proxy para correlação SPY
        self.sector_proxy_multiplier = 0.8  # Proxy para correlação setorial
        self.vix_proxy_base = 20.0  # Proxy para VIX base
        self.dxy_proxy_multiplier = -0.3  # Proxy para correlação dólar
        
    def calculate_spy_correlation(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        📈 SPY CORRELATION (50 períodos)
        
        Calcula correlação com mercado geral usando proxy baseado em:
        - Market regime alignment
        - Broad market momentum patterns
        - Systematic risk exposure
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 20:
                return 0.6  # Correlação média default
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 10:
                return 0.6
            
            # Calcular returns do ativo
            asset_returns = window_data['close'].pct_change().fillna(0)
            
            # 🎯 SPY PROXY: Simular returns do SPY baseado no comportamento do ativo
            # Em produção: usar dados reais do SPY
            
            # Broad market proxy usando trend e volume patterns
            market_trend = asset_returns.rolling(5, min_periods=1).mean()
            volume_trend = window_data['volume'].pct_change().fillna(0).rolling(5, min_periods=1).mean()
            
            # SPY proxy: combinar trend geral com ruído de mercado
            spy_proxy_returns = (
                market_trend * self.spy_proxy_multiplier +  # Trend alignment
                np.random.normal(0, 0.01, len(market_trend)) * 0.3 +  # Market noise
                volume_trend * 0.1  # Volume influence
            )
            
            # Adicionar regime de mercado (bull/bear cycles)
            cumulative_returns = (1 + asset_returns).cumprod()
            market_regime = (cumulative_returns / cumulative_returns.rolling(20, min_periods=1).mean() - 1).fillna(0)
            spy_proxy_returns += market_regime * 0.2
            
            # Calcular correlação
            if len(asset_returns) >= 10 and len(spy_proxy_returns) >= 10:
                correlation = np.corrcoef(asset_returns.values, spy_proxy_returns.values)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.6
            else:
                correlation = 0.6
            
            # Normalizar para [0,1] onde 0.5 = sem correlação
            correlation = max(-0.95, min(0.95, correlation))  # Bound correlation
            normalized_corr = (correlation + 1) * 0.5  # Transform to [0,1]
            
            return float(normalized_corr)
            
        except Exception:
            return 0.6
    
    def calculate_sector_correlation(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        🏢 SECTOR CORRELATION
        
        Calcula correlação com setor específico usando:
        - Sector-specific patterns
        - Industry momentum
        - Sector rotation effects
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 20:
                return 0.7  # Correlação setorial tipicamente maior
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 10:
                return 0.7
            
            # Calcular returns do ativo
            asset_returns = window_data['close'].pct_change().fillna(0)
            
            # 🎯 SECTOR PROXY: Simular comportamento setorial
            # Em produção: usar dados reais do setor (XLF, XLK, XLE, etc.)
            
            # Sector proxy com características específicas
            intraday_range = (window_data['high'] - window_data['low']) / window_data['close']
            sector_momentum = asset_returns.rolling(10, min_periods=1).mean()
            
            # Sector proxy returns (mais correlacionado que mercado geral)
            sector_proxy_returns = (
                asset_returns * self.sector_proxy_multiplier +  # Higher sector correlation
                sector_momentum * 0.4 +  # Sector momentum
                intraday_range.pct_change().fillna(0) * 0.2 +  # Volatility patterns
                np.random.normal(0, 0.008, len(asset_returns)) * 0.2  # Sector-specific noise
            )
            
            # Adicionar sector rotation effects
            volume_regime = (window_data['volume'] / window_data['volume'].rolling(20, min_periods=1).mean() - 1).fillna(0)
            sector_proxy_returns += volume_regime * 0.15
            
            # Calcular correlação
            if len(asset_returns) >= 10 and len(sector_proxy_returns) >= 10:
                correlation = np.corrcoef(asset_returns.values, sector_proxy_returns.values)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.7
            else:
                correlation = 0.7
            
            # Normalizar para [0,1]
            correlation = max(-0.95, min(0.95, correlation))
            normalized_corr = (correlation + 1) * 0.5
            
            return float(normalized_corr)
            
        except Exception:
            return 0.7
    
    def calculate_vix_divergence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        😰 VIX DIVERGENCE
        
        Calcula divergência com índice de volatilidade usando:
        - Realized vs implied volatility divergence
        - Fear/greed sentiment proxy
        - Market stress indicators
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 20:
                return 0.3  # Low divergence default
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 10:
                return 0.3
            
            # Calcular volatilidade realizada
            asset_returns = window_data['close'].pct_change().fillna(0)
            realized_vol = asset_returns.rolling(10, min_periods=1).std()
            
            # 🎯 VIX PROXY: Simular comportamento do VIX
            # Em produção: usar dados reais do VIX
            
            # VIX proxy baseado em market stress indicators
            price_gaps = np.abs(window_data['open'] - window_data['close'].shift(1)) / window_data['close'].shift(1)
            price_gaps = price_gaps.fillna(0)
            
            volume_spikes = window_data['volume'] / window_data['volume'].rolling(10, min_periods=1).mean()
            volume_spikes = volume_spikes.fillna(1)
            
            # VIX proxy (inversely related to market confidence)
            vix_proxy = (
                self.vix_proxy_base +  # Base VIX level
                realized_vol * 100 +  # Realized vol component
                price_gaps * 50 +  # Gap risk
                (volume_spikes - 1) * 5 +  # Volume stress
                np.random.normal(0, 2, len(realized_vol))  # VIX noise
            )
            
            # Market returns (for divergence calculation)
            market_returns = asset_returns.rolling(5, min_periods=1).mean()
            
            # VIX divergence: when VIX high but market stable (or vice versa)
            current_vix = vix_proxy.iloc[-1] if len(vix_proxy) > 0 else self.vix_proxy_base
            current_market_stress = abs(market_returns.iloc[-1]) if len(market_returns) > 0 else 0.01
            
            # Expected VIX based on market movement
            expected_vix = self.vix_proxy_base + current_market_stress * 50
            
            # Divergence calculation
            vix_divergence = abs(current_vix - expected_vix) / (expected_vix + 1e-6)
            vix_divergence = min(1.0, vix_divergence / 2)  # Normalize to [0,1]
            
            return float(vix_divergence)
            
        except Exception:
            return 0.3
    
    def calculate_dollar_strength_impact(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        💵 DOLLAR STRENGTH IMPACT
        
        Calcula impacto da força do dólar usando:
        - Currency correlation patterns
        - Commodity vs dollar relationships
        - International capital flows proxy
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < 20:
                return 0.4  # Neutral dollar impact
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 10:
                return 0.4
            
            # Calcular returns do ativo
            asset_returns = window_data['close'].pct_change().fillna(0)
            
            # 🎯 DXY PROXY: Simular comportamento do Dollar Index
            # Em produção: usar dados reais do DXY
            
            # Dollar strength proxy baseado em market patterns
            # Dollar forte tipicamente: high rates, low risk appetite, flight to quality
            
            # Flight to quality proxy (large moves suggest risk-off)
            risk_off_indicator = asset_returns.abs().rolling(5, min_periods=1).mean()
            
            # Interest rate proxy (volatility patterns suggest rate changes)
            rate_proxy = window_data['close'].pct_change().rolling(20, min_periods=1).std()
            
            # Dollar strength proxy
            dxy_proxy_returns = (
                -asset_returns * abs(self.dxy_proxy_multiplier) +  # Inverse correlation with risk assets
                risk_off_indicator * 2 +  # Flight to quality
                rate_proxy * 0.5 +  # Rate differential proxy
                np.random.normal(0, 0.005, len(asset_returns)) * 0.3  # DXY noise
            )
            
            # Commodity/dollar relationship (if asset acts like commodity)
            volume_intensity = window_data['volume'] / window_data['volume'].rolling(20, min_periods=1).mean()
            volume_intensity = volume_intensity.fillna(1)
            
            # High volume often coincides with dollar moves
            dxy_proxy_returns += (volume_intensity - 1) * 0.1
            
            # Calculate correlation/impact
            if len(asset_returns) >= 10 and len(dxy_proxy_returns) >= 10:
                dollar_correlation = np.corrcoef(asset_returns.values, dxy_proxy_returns.values)[0, 1]
                if np.isnan(dollar_correlation):
                    dollar_correlation = self.dxy_proxy_multiplier
            else:
                dollar_correlation = self.dxy_proxy_multiplier
            
            # Transform correlation to impact measure
            dollar_impact = abs(dollar_correlation)  # Strength of relationship
            dollar_impact = max(0.0, min(1.0, dollar_impact))
            
            return float(dollar_impact)
            
        except Exception:
            return 0.4
    
    def get_all_correlation_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🎯 COMBINAR TODAS AS FEATURES DE CORRELAÇÃO INTER-MERCADOS
        
        Returns:
            np.ndarray: 4 features (SPY correlation + sector correlation + VIX divergence + dollar impact)
        """
        try:
            # Calcular todas as features
            spy_correlation = self.calculate_spy_correlation(df, current_idx)
            sector_correlation = self.calculate_sector_correlation(df, current_idx)
            vix_divergence = self.calculate_vix_divergence(df, current_idx)
            dollar_impact = self.calculate_dollar_strength_impact(df, current_idx)
            
            return np.array([
                spy_correlation,
                sector_correlation,
                vix_divergence,
                dollar_impact
            ], dtype=np.float32)
            
        except Exception:
            # Valores padrão seguros em caso de erro
            return np.array([0.6, 0.7, 0.3, 0.4], dtype=np.float32)


def create_correlation_analyzer() -> MarketCorrelationAnalyzer:
    """Factory function para criar analisador de correlação inter-mercados"""
    return MarketCorrelationAnalyzer(window_size=50)