import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MicrostructureAnalyzer:
    """
    🏛️ ANALISADOR DE MICROESTRUTURA DE MERCADO
    
    Implementa análise de order flow e tick-level analytics para capturar
    dinâmicas de microestrutura ausentes no sistema atual.
    
    Features implementadas:
    - Order Flow Dynamics (8 features)
    - Tick-Level Analytics (6 features)
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_history = []
        self.volume_history = []
        self.tick_history = []
        
    def calculate_order_flow_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🔄 ORDER FLOW DYNAMICS (8 features)
        
        Analisa o fluxo de ordens e desequilíbrios de liquidez baseado nos dados disponíveis.
        Simula características de microestrutura usando price action e volume.
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < self.window_size:
                return np.full(8, 0.3, dtype=np.float32)
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 5:
                return np.full(8, 0.3, dtype=np.float32)
            
            # 1. BID-ASK IMBALANCE ESTIMADO
            # Usando spread implícito baseado em volatilidade e volume
            price_changes = window_data['close'].pct_change().fillna(0)
            volume_weighted_changes = price_changes * window_data['volume']
            bid_ask_imbalance = np.tanh(volume_weighted_changes.sum() / (window_data['volume'].sum() + 1e-6))
            bid_ask_imbalance = max(0.0, min(1.0, (bid_ask_imbalance + 1) * 0.5))
            
            # 2. ORDER BOOK DEPTH L1 ESTIMADO
            # Baseado em volatilidade e volume - profundidade nível 1
            recent_vol = window_data['volume'].tail(5).mean()
            recent_price_vol = price_changes.tail(5).std()
            depth_l1 = recent_vol / (recent_price_vol * window_data['close'].iloc[-1] + 1e-6)
            depth_l1 = max(0.0, min(1.0, np.tanh(depth_l1 / 10000) * 0.7 + 0.3))
            
            # 3. ORDER BOOK DEPTH L5 ESTIMADO  
            # Profundidade estimada nos 5 níveis superiores
            volume_momentum = window_data['volume'].pct_change().fillna(0).tail(5).mean()
            depth_l5 = depth_l1 * (1 + np.tanh(volume_momentum) * 0.3)
            depth_l5 = max(0.0, min(1.0, depth_l5))
            
            # 4. MARKET IMPACT ESTIMADO
            # Impacto de mercado baseado em relação volume/volatilidade
            avg_volume = window_data['volume'].mean()
            price_impact = price_changes.abs().mean()
            market_impact = price_impact * avg_volume / (window_data['close'].iloc[-1] + 1e-6)
            market_impact = max(0.0, min(1.0, np.tanh(market_impact * 1000) * 0.6 + 0.2))
            
            # 5. LIQUIDITY CLUSTERING NEAR
            # Agrupamento de liquidez em níveis próximos ao preço atual
            current_price = window_data['close'].iloc[-1]
            price_levels = np.abs(window_data['close'] - current_price) / current_price
            volume_near = window_data['volume'][price_levels <= 0.001].sum()
            total_volume = window_data['volume'].sum()
            liquidity_near = volume_near / (total_volume + 1e-6)
            liquidity_near = max(0.0, min(1.0, liquidity_near))
            
            # 6. LIQUIDITY CLUSTERING FAR  
            # Agrupamento de liquidez em níveis distantes
            volume_far = window_data['volume'][price_levels > 0.005].sum()
            liquidity_far = volume_far / (total_volume + 1e-6)
            liquidity_far = max(0.0, min(1.0, liquidity_far))
            
            # 7. ORDER FLOW MOMENTUM
            # Momentum do fluxo de ordens baseado em sequência de trades
            up_moves = (price_changes > 0).sum()
            down_moves = (price_changes < 0).sum()
            total_moves = up_moves + down_moves
            flow_momentum = (up_moves - down_moves) / (total_moves + 1e-6)
            flow_momentum = max(0.0, min(1.0, (flow_momentum + 1) * 0.5))
            
            # 8. INSTITUTIONAL FLOW ESTIMADO
            # Fluxo institucional baseado em volume blocks e padrões de preço
            large_volume_threshold = window_data['volume'].quantile(0.8)
            institutional_volume = window_data['volume'][window_data['volume'] >= large_volume_threshold].sum()
            institutional_flow = institutional_volume / (total_volume + 1e-6)
            institutional_flow = max(0.0, min(1.0, institutional_flow))
            
            return np.array([
                bid_ask_imbalance,
                depth_l1, 
                depth_l5,
                market_impact,
                liquidity_near,
                liquidity_far,
                flow_momentum,
                institutional_flow
            ], dtype=np.float32)
            
        except Exception as e:
            # Valores padrão seguros em caso de erro
            return np.array([0.5, 0.4, 0.4, 0.3, 0.6, 0.2, 0.5, 0.3], dtype=np.float32)
    
    def calculate_tick_analytics_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        📊 TICK-LEVEL ANALYTICS (6 features)
        
        Analisa características tick-a-tick do mercado usando dados de preço e volume.
        """
        try:
            # Garantir dados suficientes
            start_idx = max(0, current_idx - self.window_size)
            end_idx = current_idx + 1
            
            if end_idx <= start_idx or current_idx < self.window_size:
                return np.full(6, 0.4, dtype=np.float32)
            
            # Extrair dados da janela temporal
            window_data = df.iloc[start_idx:end_idx].copy()
            
            if len(window_data) < 5:
                return np.full(6, 0.4, dtype=np.float32)
            
            # 1. TICK DIRECTION MOMENTUM
            # Momentum direcional baseado em sequência de upticks/downticks
            price_changes = window_data['close'].diff().fillna(0)
            upticks = (price_changes > 0).astype(int)
            downticks = (price_changes < 0).astype(int)
            
            # Calcular momentum direcional
            tick_sequence = upticks - downticks
            tick_momentum = tick_sequence.rolling(5, min_periods=1).mean().iloc[-1]
            tick_momentum = max(0.0, min(1.0, (tick_momentum + 1) * 0.5))
            
            # 2. VOLUME AT PRICE CONCENTRATION
            # Concentração de volume em níveis específicos de preço
            price_bins = pd.qcut(window_data['close'], q=5, duplicates='drop')
            volume_by_price = window_data.groupby(price_bins)['volume'].sum()
            volume_concentration = volume_by_price.max() / (volume_by_price.sum() + 1e-6)
            volume_concentration = max(0.0, min(1.0, volume_concentration))
            
            # 3. TRADE VELOCITY
            # Velocidade de execução de trades (estimada por frequência de mudanças)
            non_zero_changes = (price_changes != 0).sum()
            trade_velocity = non_zero_changes / len(window_data)
            trade_velocity = max(0.0, min(1.0, trade_velocity))
            
            # 4. PRINT SIZE DISTRIBUTION  
            # Distribuição do tamanho das transações (baseada em volume)
            volume_changes = window_data['volume'].pct_change().fillna(0).abs()
            print_size_variation = volume_changes.std()
            print_size_dist = np.tanh(print_size_variation * 10) * 0.7 + 0.3
            print_size_dist = max(0.0, min(1.0, print_size_dist))
            
            # 5. UPTICK DOWNTICK RATIO
            # Razão entre upticks e downticks
            total_directional = upticks.sum() + downticks.sum()
            if total_directional > 0:
                uptick_ratio = upticks.sum() / total_directional
            else:
                uptick_ratio = 0.5
            uptick_ratio = max(0.0, min(1.0, uptick_ratio))
            
            # 6. AGGRESSIVE PASSIVE RATIO
            # Razão entre trades agressivos e passivos (estimada por volume vs price impact)
            volume_weighted_impact = (price_changes.abs() * window_data['volume']).sum()
            total_volume = window_data['volume'].sum()
            aggressive_ratio = volume_weighted_impact / (total_volume * window_data['close'].iloc[-1] + 1e-6)
            aggressive_ratio = max(0.0, min(1.0, np.tanh(aggressive_ratio * 1000) * 0.6 + 0.4))
            
            return np.array([
                tick_momentum,
                volume_concentration,
                trade_velocity,
                print_size_dist,
                uptick_ratio,
                aggressive_ratio
            ], dtype=np.float32)
            
        except Exception as e:
            # Valores padrão seguros em caso de erro
            return np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.4], dtype=np.float32)
    
    def get_all_microstructure_features(self, df: pd.DataFrame, current_idx: int) -> np.ndarray:
        """
        🎯 COMBINAR TODAS AS FEATURES DE MICROESTRUTURA
        
        Returns:
            np.ndarray: 14 features (8 order flow + 6 tick analytics)
        """
        order_flow = self.calculate_order_flow_features(df, current_idx)
        tick_analytics = self.calculate_tick_analytics_features(df, current_idx)
        
        return np.concatenate([order_flow, tick_analytics])


def create_microstructure_analyzer() -> MicrostructureAnalyzer:
    """Factory function para criar analisador de microestrutura"""
    return MicrostructureAnalyzer(window_size=20)