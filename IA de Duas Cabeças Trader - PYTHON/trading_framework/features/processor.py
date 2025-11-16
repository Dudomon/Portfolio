#!/usr/bin/env python3
"""
🔧 PROCESSADOR DE FEATURES
Processamento de features técnicas para trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import ta


class FeatureProcessor:
    """Processador de features técnicas"""
    
    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [5, 10, 20, 50]
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa features técnicas do DataFrame"""
        df = df.copy()
        
        # Features básicas
        df = self._add_basic_features(df)
        
        # Features técnicas
        df = self._add_technical_features(df)
        
        # Features de volume
        df = self._add_volume_features(df)
        
        # Features de volatilidade
        df = self._add_volatility_features(df)
        
        # Limpar NaN
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features básicas"""
        # Retornos
        df['returns'] = df['close_5m'].pct_change()
        df['log_returns'] = np.log(df['close_5m'] / df['close_5m'].shift(1))
        
        # High-Low spread
        df['hl_spread'] = (df['high_5m'] - df['low_5m']) / df['close_5m']
        
        # Price position
        df['price_position'] = (df['close_5m'] - df['low_5m']) / (df['high_5m'] - df['low_5m'])
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features técnicas"""
        # Médias móveis
        for window in self.window_sizes:
            df[f'sma_{window}'] = ta.trend.sma_indicator(df['close_5m'], window=window)
            df[f'ema_{window}'] = ta.trend.ema_indicator(df['close_5m'], window=window)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close_5m'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close_5m'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close_5m'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_position'] = (df['close_5m'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high_5m'], df['low_5m'], df['close_5m'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volume"""
        if 'volume_5m' in df.columns:
            # Volume médio
            for window in self.window_sizes:
                df[f'volume_sma_{window}'] = ta.volume.volume_sma(df['close_5m'], df['volume_5m'], window=window)
            
            # Volume ratio
            df['volume_ratio'] = df['volume_5m'] / df['volume_5m'].rolling(20).mean()
            
            # OBV
            df['obv'] = ta.volume.on_balance_volume(df['close_5m'], df['volume_5m'])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volatilidade"""
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high_5m'], df['low_5m'], df['close_5m'])
        
        # Volatilidade histórica
        for window in self.window_sizes:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        # Volatilidade implícita (simulada)
        df['implied_volatility'] = df['atr'] / df['close_5m']
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Retorna lista de colunas de features"""
        return [
            'returns', 'log_returns', 'hl_spread', 'price_position',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_position', 'stoch_k', 'stoch_d',
            'volume_ratio', 'atr', 'implied_volatility'
        ] + [f'sma_{w}' for w in self.window_sizes] + \
           [f'ema_{w}' for w in self.window_sizes] + \
           [f'volume_sma_{w}' for w in self.window_sizes] + \
           [f'volatility_{w}' for w in self.window_sizes]
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normaliza features"""
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if method == 'zscore':
            for col in available_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[f'{col}_norm'] = (df[col] - mean_val) / std_val
                else:
                    df[f'{col}_norm'] = 0
        
        elif method == 'minmax':
            for col in available_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_norm'] = 0
        
        return df 