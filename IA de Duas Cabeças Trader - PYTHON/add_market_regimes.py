#!/usr/bin/env python3
"""
Market Regime Augmentation - Dataset Yahoo Enhanced & Compressed
Adiciona ciclos bull/bear sintéticos e transições mais frequentes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeAugmenter:
    def __init__(self, trend_acceleration=0.6, volatility_boost=1.3, regime_frequency=0.7):
        """
        Inicializa augmentador de regimes de mercado
        
        Args:
            trend_acceleration: Fator de aceleração de tendências (0.6 = 40% mais rápido)
            volatility_boost: Multiplicador de volatilidade em regimes ativos
            regime_frequency: Frequência de mudanças de regime (0.7 = 30% mais mudanças)
        """
        self.trend_acceleration = trend_acceleration
        self.volatility_boost = volatility_boost
        self.regime_frequency = regime_frequency
        
    def augment_regimes(self, df):
        """
        Aplica augmentação de regimes de mercado
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com regimes augmentados
        """
        print(f"🌊 MARKET REGIME AUGMENTATION")
        print(f"   Aceleração de tendência: {(1-self.trend_acceleration)*100:.0f}% mais rápido")
        print(f"   Boost de volatilidade: {self.volatility_boost:.1f}x")
        print(f"   Frequência de mudanças: +{(self.regime_frequency-0.5)*100:.0f}%")
        
        df_augmented = df.copy()
        
        # Detectar regimes atuais
        regimes = self._detect_market_regimes(df)
        df_augmented['original_regime'] = regimes
        
        # Criar regimes sintéticos com transições mais frequentes
        synthetic_regimes = self._create_synthetic_regimes(df, regimes)
        df_augmented['synthetic_regime'] = synthetic_regimes
        
        # Aplicar augmentação baseada nos regimes
        df_augmented = self._apply_regime_augmentation(df_augmented)
        
        # Validar resultado
        self._validate_augmentation(df, df_augmented)
        
        return df_augmented
        
    def _detect_market_regimes(self, df):
        """Detecta regimes de mercado usando multiple timeframes"""
        print(f"   🔍 Detectando regimes originais...")
        
        returns = df['close'].pct_change()
        
        # Múltiplos timeframes para regime detection
        short_trend = returns.rolling(50).mean()  # ~4h trend
        medium_trend = returns.rolling(200).mean()  # ~16h trend
        long_trend = returns.rolling(800).mean()  # ~3d trend
        
        # Volatilidade para identificar consolidação
        volatility = returns.rolling(50).std()
        vol_threshold = volatility.quantile(0.4)  # Bottom 40% = consolidação
        
        # Classificar regimes
        regimes = []
        for i in range(len(df)):
            st = short_trend.iloc[i] if not pd.isna(short_trend.iloc[i]) else 0
            mt = medium_trend.iloc[i] if not pd.isna(medium_trend.iloc[i]) else 0
            lt = long_trend.iloc[i] if not pd.isna(long_trend.iloc[i]) else 0
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else vol_threshold
            
            # Lógica de classificação
            if vol < vol_threshold:
                regime = 'consolidation'
            elif st > 0 and mt > 0:
                regime = 'bull_strong'
            elif st > 0 and mt <= 0:
                regime = 'bull_weak'  
            elif st < 0 and mt < 0:
                regime = 'bear_strong'
            elif st < 0 and mt >= 0:
                regime = 'bear_weak'
            else:
                regime = 'neutral'
                
            regimes.append(regime)
        
        regimes_series = pd.Series(regimes)
        regime_counts = regimes_series.value_counts()
        print(f"   Regimes detectados: {dict(regime_counts)}")
        
        return regimes_series
        
    def _create_synthetic_regimes(self, df, original_regimes):
        """Cria regimes sintéticos com transições mais frequentes"""
        print(f"   🎭 Criando regimes sintéticos...")
        
        synthetic = original_regimes.copy()
        
        # Identificar pontos de mudança de regime
        regime_changes = original_regimes != original_regimes.shift(1)
        change_points = regime_changes[regime_changes].index.tolist()
        
        print(f"   Mudanças originais: {len(change_points)}")
        
        # Adicionar mudanças sintéticas baseadas na frequência desejada
        additional_changes = int(len(change_points) * self.regime_frequency)
        
        # Encontrar segmentos longos para subdividir
        long_segments = []
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            length = end - start
            
            if length > 500:  # Segmentos > ~2 dias
                # Dividir segmento longo
                n_divisions = max(2, int(length / 300))  # Divisões a cada ~1.25 dias
                division_points = np.linspace(start, end, n_divisions + 1, dtype=int)[1:-1]
                long_segments.extend(division_points)
        
        # Aplicar mudanças sintéticas
        for change_point in long_segments[:additional_changes]:
            current_regime = synthetic.iloc[change_point]
            
            # Escolher novo regime baseado no atual
            if current_regime == 'bull_strong':
                new_regime = np.random.choice(['bull_weak', 'consolidation'], p=[0.6, 0.4])
            elif current_regime == 'bull_weak':
                new_regime = np.random.choice(['bull_strong', 'neutral', 'consolidation'], p=[0.4, 0.3, 0.3])
            elif current_regime == 'bear_strong':
                new_regime = np.random.choice(['bear_weak', 'consolidation'], p=[0.6, 0.4])
            elif current_regime == 'bear_weak':
                new_regime = np.random.choice(['bear_strong', 'neutral', 'consolidation'], p=[0.4, 0.3, 0.3])
            elif current_regime == 'consolidation':
                new_regime = np.random.choice(['bull_weak', 'bear_weak', 'neutral'], p=[0.35, 0.35, 0.3])
            else:  # neutral
                new_regime = np.random.choice(['bull_weak', 'bear_weak', 'consolidation'], p=[0.35, 0.35, 0.3])
            
            # Aplicar mudança por um período
            change_duration = np.random.randint(50, 200)  # 20min - 1h
            end_point = min(change_point + change_duration, len(synthetic))
            synthetic.iloc[change_point:end_point] = new_regime
            
        synthetic_changes = (synthetic != synthetic.shift(1)).sum()
        print(f"   Mudanças sintéticas: {synthetic_changes} (+{synthetic_changes - len(change_points)})")
        
        return synthetic
        
    def _apply_regime_augmentation(self, df):
        """Aplica augmentação baseada nos regimes sintéticos"""
        print(f"   🚀 Aplicando augmentação por regime...")
        
        returns = df['close'].pct_change()
        augmented_returns = returns.copy()
        
        # Aplicar diferentes augmentações por regime
        for regime in df['synthetic_regime'].unique():
            mask = df['synthetic_regime'] == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) == 0:
                continue
                
            if regime == 'bull_strong':
                # Acelerar tendência de alta + aumentar volatilidade
                trend_boost = 1.2
                vol_boost = self.volatility_boost
            elif regime == 'bull_weak':  
                # Acelerar levemente + volatilidade moderada
                trend_boost = 1.1
                vol_boost = 1.1
            elif regime == 'bear_strong':
                # Acelerar tendência de baixa + aumentar volatilidade
                trend_boost = 1.2  # Para baixa, mantém magnitude
                vol_boost = self.volatility_boost
            elif regime == 'bear_weak':
                # Acelerar levemente + volatilidade moderada  
                trend_boost = 1.1
                vol_boost = 1.1
            elif regime == 'consolidation':
                # Reduzir tendência + aumentar micro-volatilidade
                trend_boost = 0.8
                vol_boost = 1.3
            else:  # neutral
                # Manter neutro mas com mais ação
                trend_boost = 1.0
                vol_boost = 1.2
            
            # Aplicar boost de tendência
            augmented_returns[mask] = regime_returns * trend_boost
            
            # Aplicar boost de volatilidade via noise injection
            if vol_boost > 1.0:
                noise_std = regime_returns.std() * (vol_boost - 1.0) * 0.5
                noise = np.random.normal(0, noise_std, len(regime_returns))
                augmented_returns[mask] += noise
        
        # Reconstruir preços OHLC
        price_multiplier = (1 + augmented_returns).cumprod()
        first_close = df['close'].iloc[0]
        df['close_augmented'] = price_multiplier * first_close
        
        # Ajustar OHLC mantendo proporções
        close_ratio = df['close_augmented'] / df['close']
        close_ratio = close_ratio.fillna(1.0)
        
        df['open_augmented'] = df['open'] * close_ratio
        df['high_augmented'] = df['high'] * close_ratio
        df['low_augmented'] = df['low'] * close_ratio
        
        # Fix OHLC consistency
        df['high_augmented'] = np.maximum.reduce([
            df['high_augmented'], df['open_augmented'], df['close_augmented']
        ])
        df['low_augmented'] = np.minimum.reduce([
            df['low_augmented'], df['open_augmented'], df['close_augmented']
        ])
        
        # Atualizar returns augmented
        df['returns_augmented'] = augmented_returns
        
        return df
        
    def _validate_augmentation(self, df_original, df_augmented):
        """Valida augmentação de regimes"""
        print(f"\n✅ VALIDAÇÃO MARKET REGIME AUGMENTATION:")
        
        # Comparar volatilidades
        vol_orig = df_original['close'].pct_change().std()
        vol_aug = df_augmented['close_augmented'].pct_change().std()
        
        print(f"   Volatilidade original: {vol_orig*100:.3f}%")
        print(f"   Volatilidade augmented: {vol_aug*100:.3f}%")
        print(f"   Aumento: {vol_aug/vol_orig:.2f}x")
        
        # Comparar tendências
        trend_orig = df_original['close'].iloc[-1] / df_original['close'].iloc[0]
        trend_aug = df_augmented['close_augmented'].iloc[-1] / df_augmented['close_augmented'].iloc[0]
        
        print(f"   Tendência original: {(trend_orig-1)*100:+.2f}%")
        print(f"   Tendência augmented: {(trend_aug-1)*100:+.2f}%")
        
        # Verificar regimes
        regime_changes_orig = (df_augmented['original_regime'] != df_augmented['original_regime'].shift(1)).sum()
        regime_changes_synth = (df_augmented['synthetic_regime'] != df_augmented['synthetic_regime'].shift(1)).sum()
        
        print(f"   Mudanças regime original: {regime_changes_orig}")
        print(f"   Mudanças regime sintético: {regime_changes_synth}")
        print(f"   Aumento de dinamismo: +{regime_changes_synth - regime_changes_orig} mudanças")

def create_final_augmented_dataset():
    """Cria dataset final com augmentação de regimes"""
    
    print("🌊 CRIANDO DATASET COM MARKET REGIME AUGMENTATION")
    print("=" * 60)
    
    # Carregar dataset comprimido
    input_file = 'data/GC_YAHOO_ENHANCED_COMPRESSED_20250804_181454.csv'
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"📊 Dataset comprimido carregado:")
    print(f"   Arquivo: {input_file}")
    print(f"   Barras: {len(df):,}")
    
    # Aplicar augmentação de regimes
    augmenter = MarketRegimeAugmenter(
        trend_acceleration=0.7,  # 30% mais rápido
        volatility_boost=1.4,    # 40% mais volatilidade
        regime_frequency=0.8     # 30% mais mudanças
    )
    
    df_augmented = augmenter.augment_regimes(df)
    
    # Preparar dataset final
    df_final = df_augmented[['time', 'open_augmented', 'high_augmented', 'low_augmented', 'close_augmented']].copy()
    df_final.columns = ['time', 'open', 'high', 'low', 'close']
    
    # Manter outras colunas necessárias
    df_final['tick_volume'] = df['tick_volume']
    df_final['spread'] = df['spread']
    df_final['real_volume'] = df['real_volume']
    
    # Recalcular indicadores para compatibilidade
    df_final['returns'] = df_final['close'].pct_change()
    df_final['volatility_20'] = df_final['returns'].rolling(20).std()
    df_final['sma_20'] = df_final['close'].rolling(20).mean()
    df_final['sma_50'] = df_final['close'].rolling(50).mean()
    
    # RSI
    delta = df_final['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_final['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df_final['high'] - df_final['low']
    hc = abs(df_final['high'] - df_final['close'].shift())
    lc = abs(df_final['low'] - df_final['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_final['atr_14'] = tr.rolling(14).mean()
    
    # Bollinger position
    bb_mid = df_final['sma_20']
    bb_std = df_final['close'].rolling(20).std()
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)
    df_final['bb_position'] = (df_final['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Preencher NaNs e adicionar colunas de compatibilidade
    df_final = df_final.fillna(method='ffill').fillna(method='bfill')
    df_final['trend_strength'] = 0.0
    df_final['stoch_k'] = 50.0
    df_final['volume_ratio'] = df_final['tick_volume']
    df_final['var_99'] = df_final['close'] * 0.95
    
    # Salvar dataset final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/GC_YAHOO_FINAL_AUGMENTED_{timestamp}.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n💾 DATASET FINAL SALVO:")
    print(f"   Arquivo: {output_file}")
    print(f"   Barras: {len(df_final):,}")
    
    # Estatísticas finais vs original
    return output_file, df_final

if __name__ == "__main__":
    try:
        output_file, df_final = create_final_augmented_dataset()
        print(f"\n✅ MARKET REGIME AUGMENTATION CONCLUÍDA!")
        print(f"   Dataset final pronto: {output_file}")
        
    except Exception as e:
        print(f"❌ Erro na augmentação: {e}")
        import traceback
        traceback.print_exc()