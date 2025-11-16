#!/usr/bin/env python3
"""
Volatility Enhancement Progressivo - Dataset Yahoo Gold
Implementa scaling adaptativo preservando estrutura de correlação
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolatilityEnhancer:
    def __init__(self, base_factor=1.5, max_factor=3.0, volatility_window=100):
        """
        Inicializa o enhancer de volatilidade
        
        Args:
            base_factor: Fator mínimo de scaling (1.5x)
            max_factor: Fator máximo de scaling (3.0x)
            volatility_window: Janela para calcular volatilidade local
        """
        self.base_factor = base_factor
        self.max_factor = max_factor  
        self.volatility_window = volatility_window
        
    def enhance_volatility(self, df):
        """
        Aplica enhancement de volatilidade preservando estrutura
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com preços enhanced
        """
        print(f"🚀 VOLATILITY SCALING PROGRESSIVO")
        print(f"   Base factor: {self.base_factor}x")
        print(f"   Max factor: {self.max_factor}x") 
        print(f"   Window: {self.volatility_window} barras")
        
        df_enhanced = df.copy()
        
        # Calcular returns originais
        returns = df['close'].pct_change()
        df_enhanced['returns_original'] = returns
        
        # Calcular volatilidade rolling
        vol_rolling = returns.rolling(self.volatility_window, min_periods=10).std()
        vol_rolling = vol_rolling.fillna(vol_rolling.mean())
        
        # Calcular percentil de volatilidade (0 = mais baixa, 1 = mais alta)
        vol_percentile = vol_rolling.rank(pct=True)
        
        # Scaling adaptativo: mais scaling onde volatilidade é baixa
        # Fórmula: scale = base + (max-base) * (1 - percentile)
        # Volatilidade baixa (percentile ~0) -> scale próximo de max_factor
        # Volatilidade alta (percentile ~1) -> scale próximo de base_factor
        scale_factor = self.base_factor + (self.max_factor - self.base_factor) * (1 - vol_percentile)
        
        # Aplicar scaling preservando direção
        enhanced_returns = returns * scale_factor
        df_enhanced['returns_enhanced'] = enhanced_returns
        df_enhanced['scale_factor'] = scale_factor
        
        # Reconstruir preços mantendo continuidade
        # Usar cumprod para evitar drift
        price_multiplier = (1 + enhanced_returns).cumprod()
        
        # Normalizar para manter mesmo preço inicial
        first_close = df['close'].iloc[0]
        df_enhanced['close_enhanced'] = price_multiplier * first_close
        
        # Ajustar OHLC mantendo proporções relativas
        close_ratio = df_enhanced['close_enhanced'] / df['close']
        close_ratio = close_ratio.fillna(1.0)  # Handle NaN ratios
        
        df_enhanced['open_enhanced'] = df['open'] * close_ratio
        df_enhanced['high_enhanced'] = df['high'] * close_ratio  
        df_enhanced['low_enhanced'] = df['low'] * close_ratio
        
        # Fix OHLC consistency issues (vectorized)
        df_enhanced['high_enhanced'] = np.maximum.reduce([
            df_enhanced['high_enhanced'], 
            df_enhanced['open_enhanced'], 
            df_enhanced['close_enhanced']
        ])
        df_enhanced['low_enhanced'] = np.minimum.reduce([
            df_enhanced['low_enhanced'], 
            df_enhanced['open_enhanced'], 
            df_enhanced['close_enhanced']
        ])
        
        # Verificar sanidade dos dados
        self._validate_enhanced_data(df, df_enhanced)
        
        return df_enhanced
        
    def _validate_enhanced_data(self, df_original, df_enhanced):
        """Valida dados enhanced"""
        print(f"\n✅ VALIDAÇÃO DOS DADOS ENHANCED:")
        
        # Comparar volatilidades
        vol_orig = df_original['close'].pct_change().std()
        vol_enh = df_enhanced['close_enhanced'].pct_change().std()
        vol_ratio = vol_enh / vol_orig
        
        print(f"   Volatilidade original: {vol_orig*100:.3f}%")
        print(f"   Volatilidade enhanced: {vol_enh*100:.3f}%")
        print(f"   Ratio de aumento: {vol_ratio:.2f}x")
        
        # Verificar preservação de tendência geral
        trend_orig = df_original['close'].iloc[-1] / df_original['close'].iloc[0]
        trend_enh = df_enhanced['close_enhanced'].iloc[-1] / df_enhanced['close_enhanced'].iloc[0]
        
        print(f"   Tendência original: {(trend_orig-1)*100:+.2f}%")
        print(f"   Tendência enhanced: {(trend_enh-1)*100:+.2f}%")
        
        # Verificar ausência de NaNs/infs
        nan_count = df_enhanced[['close_enhanced', 'open_enhanced', 'high_enhanced', 'low_enhanced']].isna().sum().sum()
        inf_count = np.isinf(df_enhanced[['close_enhanced', 'open_enhanced', 'high_enhanced', 'low_enhanced']]).sum().sum()
        
        print(f"   NaNs encontrados: {nan_count}")
        print(f"   Infs encontrados: {inf_count}")
        
        # Verificar OHLC consistency
        ohlc_issues = 0
        for i in range(len(df_enhanced)):
            o, h, l, c = (df_enhanced.iloc[i]['open_enhanced'], 
                         df_enhanced.iloc[i]['high_enhanced'],
                         df_enhanced.iloc[i]['low_enhanced'], 
                         df_enhanced.iloc[i]['close_enhanced'])
            
            if not (l <= min(o, c) <= max(o, c) <= h):
                ohlc_issues += 1
                
        print(f"   OHLC inconsistencies: {ohlc_issues}")
        
        if ohlc_issues == 0 and nan_count == 0 and inf_count == 0:
            print(f"   ✅ Dados enhanced são válidos!")
        else:
            print(f"   ⚠️ Encontrados problemas nos dados enhanced")

def create_enhanced_dataset():
    """Cria dataset enhanced com volatilidade aumentada"""
    
    print("🔧 CRIANDO DATASET YAHOO ENHANCED")
    print("=" * 50)
    
    # Carregar dados originais
    df = pd.read_csv('data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"📊 Dados originais carregados:")
    print(f"   Barras: {len(df):,}")
    print(f"   Período: {df['time'].min()} - {df['time'].max()}")
    
    # Aplicar enhancement em fases progressivas
    enhancer = VolatilityEnhancer(base_factor=1.5, max_factor=2.5, volatility_window=100)
    df_enhanced = enhancer.enhance_volatility(df)
    
    # Preparar dataset final com colunas necessárias
    df_final = df_enhanced[['time', 'open_enhanced', 'high_enhanced', 'low_enhanced', 'close_enhanced']].copy()
    df_final.columns = ['time', 'open', 'high', 'low', 'close']
    
    # Adicionar volume (manter original)
    df_final['tick_volume'] = df['tick_volume']
    df_final['spread'] = df['spread'] 
    df_final['real_volume'] = df['real_volume']
    
    # Recalcular indicadores básicos
    df_final['returns'] = df_final['close'].pct_change()
    df_final['volatility_20'] = df_final['returns'].rolling(20).std()
    df_final['sma_20'] = df_final['close'].rolling(20).mean()
    df_final['sma_50'] = df_final['close'].rolling(50).mean()
    
    # RSI
    delta = df_final['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_final['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df_final['high'] - df_final['low']
    hc = abs(df_final['high'] - df_final['close'].shift())
    lc = abs(df_final['low'] - df_final['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_final['atr_14'] = tr.rolling(14).mean()
    
    # Bollinger Bands position
    bb_mid = df_final['sma_20']
    bb_std = df_final['close'].rolling(20).std()
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)
    df_final['bb_position'] = (df_final['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Preencher NaNs
    df_final = df_final.fillna(method='ffill').fillna(method='bfill')
    
    # Adicionar colunas restantes para compatibilidade
    df_final['trend_strength'] = 0.0
    df_final['stoch_k'] = 50.0
    df_final['volume_ratio'] = df_final['tick_volume']
    df_final['var_99'] = df_final['close'] * 0.95  # Aproximação simples
    
    # Salvar dataset enhanced
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/GC_YAHOO_VOLATILITY_ENHANCED_{timestamp}.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n💾 DATASET ENHANCED SALVO:")
    print(f"   Arquivo: {output_file}")
    print(f"   Barras: {len(df_final):,}")
    
    # Estatísticas finais
    vol_orig = df['close'].pct_change().std()
    vol_enh = df_final['close'].pct_change().std()
    
    print(f"\n📈 RESULTADO FINAL:")
    print(f"   Volatilidade original: {vol_orig*100:.3f}%")
    print(f"   Volatilidade enhanced: {vol_enh*100:.3f}%")
    print(f"   Aumento de volatilidade: {vol_enh/vol_orig:.2f}x")
    
    return output_file, df_final

if __name__ == "__main__":
    try:
        output_file, df_enhanced = create_enhanced_dataset()
        print(f"\n✅ VOLATILITY ENHANCEMENT CONCLUÍDO!")
        print(f"   Arquivo pronto: {output_file}")
        
    except Exception as e:
        print(f"❌ Erro no enhancement: {e}")
        import traceback
        traceback.print_exc()