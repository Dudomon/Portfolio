#!/usr/bin/env python3
"""
Análise de Volatilidade - Dataset Yahoo Gold 15+ anos
Estabelece baseline para enhancement de volatilidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_yahoo_volatility():
    """Análise completa da volatilidade do dataset Yahoo"""
    
    print("🔍 ANÁLISE BASELINE - Dataset Yahoo Gold")
    print("=" * 50)
    
    # Carregar dados
    df = pd.read_csv('data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"📊 Dataset Info:")
    print(f"   Período: {df['time'].min()} - {df['time'].max()}")
    print(f"   Barras: {len(df):,}")
    print(f"   Anos: {(df['time'].max() - df['time'].min()).days / 365.25:.1f}")
    
    # Calcular returns
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()
    
    # Volatilidades em diferentes timeframes
    volatilities = {}
    windows = [6, 12, 48, 288]  # 30min, 1h, 4h, 24h (5min bars)
    window_names = ['30min', '1h', '4h', '24h']
    
    for window, name in zip(windows, window_names):
        vol = df['returns'].rolling(window).std() * np.sqrt(window * 288)  # Annualized
        volatilities[name] = vol
        df[f'vol_{name}'] = vol
    
    # Estatísticas gerais
    print(f"\n📈 Estatísticas de Returns:")
    print(f"   Média: {df['returns'].mean()*100:.4f}%")
    print(f"   Std: {df['returns'].std()*100:.4f}%")
    print(f"   Min: {df['returns'].min()*100:.2f}%")
    print(f"   Max: {df['returns'].max()*100:.2f}%")
    print(f"   Skewness: {df['returns'].skew():.2f}")
    print(f"   Kurtosis: {df['returns'].kurtosis():.2f}")
    
    # Análise de volatilidade por período
    print(f"\n🌊 Volatilidade Anualizada por Período:")
    for name in window_names:
        vol = volatilities[name]
        print(f"   {name:>5}: {vol.mean()*100:.1f}% ± {vol.std()*100:.1f}%")
        print(f"          Min: {vol.min()*100:.1f}% | Max: {vol.max()*100:.1f}%")
    
    # Identificar períodos "mortos" (baixa volatilidade)
    vol_1h = volatilities['1h']
    low_vol_threshold = vol_1h.quantile(0.25)  # Bottom 25%
    low_vol_periods = vol_1h < low_vol_threshold
    
    print(f"\n🔇 Períodos de Baixa Volatilidade (Bottom 25%):")
    print(f"   Threshold: {low_vol_threshold*100:.1f}% anual")
    print(f"   Ocorrência: {low_vol_periods.sum():,} barras ({low_vol_periods.mean()*100:.1f}%)")
    
    # Identificar movimentos extremos
    large_moves = df['abs_returns'] > df['abs_returns'].quantile(0.99)
    print(f"\n🚀 Movimentos Extremos (Top 1%):")
    print(f"   Threshold: {df['abs_returns'].quantile(0.99)*100:.2f}%")
    print(f"   Ocorrência: {large_moves.sum():,} barras")
    print(f"   Max movimento: {df['abs_returns'].max()*100:.2f}%")
    
    # Análise por regime de mercado
    df['returns_cumsum'] = df['returns'].cumsum()
    df['trend_diff'] = df['returns_cumsum'].rolling(288).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 288 else 0)
    df['trend'] = np.where(df['trend_diff'] > 0, 'Bull', 'Bear')
    
    bull_periods = df['trend'] == 'Bull'
    bear_periods = df['trend'] == 'Bear'
    
    print(f"\n🐂🐻 Análise por Regime:")
    print(f"   Bull markets: {bull_periods.sum():,} barras ({bull_periods.mean()*100:.1f}%)")
    print(f"   Bear markets: {bear_periods.sum():,} barras ({bear_periods.mean()*100:.1f}%)")
    
    if bull_periods.sum() > 0:
        bull_vol = df[bull_periods]['vol_1h'].mean()
        print(f"   Volatilidade Bull: {bull_vol*100:.1f}%")
    
    if bear_periods.sum() > 0:
        bear_vol = df[bear_periods]['vol_1h'].mean()
        print(f"   Volatilidade Bear: {bear_vol*100:.1f}%")
    
    # Distribuição de returns
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n📊 Distribuição de Returns (percentis):")
    for p in percentiles:
        val = df['returns'].quantile(p/100)
        print(f"   P{p:2d}: {val*100:+6.2f}%")
    
    # Clustering temporal de volatilidade
    df['vol_regime'] = pd.cut(df['vol_1h'], bins=3, labels=['Low', 'Med', 'High'])
    vol_regime_counts = df['vol_regime'].value_counts()
    
    print(f"\n🎯 Regimes de Volatilidade (1h):")
    for regime in ['Low', 'Med', 'High']:
        if regime in vol_regime_counts:
            count = vol_regime_counts[regime]
            pct = count / len(df) * 100
            print(f"   {regime}: {count:,} barras ({pct:.1f}%)")
    
    # Recommendations para enhancement
    print(f"\n💡 RECOMENDAÇÕES PARA ENHANCEMENT:")
    print(f"   1. VOLATILITY SCALING:")
    low_vol_pct = (vol_1h < vol_1h.quantile(0.3)).mean() * 100
    print(f"      - {low_vol_pct:.1f}% do tempo em baixa volatilidade")
    print(f"      - Sugestão: Scale 2.0x em períodos < P30 volatilidade")
    
    print(f"   2. TIME COMPRESSION:")
    consecutive_low = identify_consecutive_low_vol(df, vol_1h, low_vol_threshold)
    print(f"      - {len(consecutive_low)} sequências de baixa volatilidade")
    print(f"      - Sugestão: Comprimir 30% das barras em sequências > 2h")
    
    print(f"   3. REGIME AUGMENTATION:")
    avg_trend_length = analyze_trend_lengths(df)
    print(f"      - Tendências médias: {avg_trend_length:.0f} barras ({avg_trend_length/12:.1f}h)")
    print(f"      - Sugestão: Criar transições mais frequentes (50% mais rápidas)")
    
    return {
        'df': df,
        'volatilities': volatilities,
        'low_vol_threshold': low_vol_threshold,
        'large_moves_threshold': df['abs_returns'].quantile(0.99),
        'recommendations': {
            'volatility_scaling': 2.0,
            'time_compression': 0.3,
            'trend_acceleration': 0.5
        }
    }

def identify_consecutive_low_vol(df, vol_series, threshold):
    """Identifica sequências consecutivas de baixa volatilidade"""
    low_vol = vol_series < threshold
    sequences = []
    current_seq = []
    
    for i, is_low in enumerate(low_vol):
        if is_low:
            current_seq.append(i)
        else:
            if len(current_seq) > 24:  # Mais de 2h de baixa vol
                sequences.append(current_seq)
            current_seq = []
    
    if len(current_seq) > 24:
        sequences.append(current_seq)
    
    return sequences

def analyze_trend_lengths(df):
    """Analisa duração média das tendências"""
    if 'trend' not in df.columns:
        return 0
    
    trend_changes = df['trend'] != df['trend'].shift(1)
    trend_lengths = []
    current_length = 1
    
    for changed in trend_changes[1:]:
        if changed:
            trend_lengths.append(current_length)
            current_length = 1
        else:
            current_length += 1
    
    return np.mean(trend_lengths) if trend_lengths else 0

if __name__ == "__main__":
    try:
        results = analyze_yahoo_volatility()
        print(f"\n✅ Análise baseline concluída!")
        print(f"   Dados prontos para enhancement de volatilidade")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()