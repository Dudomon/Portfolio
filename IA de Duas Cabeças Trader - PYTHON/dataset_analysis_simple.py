#!/usr/bin/env python3
"""
ANÁLISE SIMPLES DO DATASET - SEM EMOJIS
"""

import pandas as pd
import numpy as np
from scipy import stats

def analyze_dataset_simple():
    dataset_path = 'data/GOLD_TRADING_READY_2M_20250803_222334.csv'
    
    print("ANALISE COMPLETA DO DATASET")
    print("="*50)
    
    df = pd.read_csv(dataset_path)
    
    # ANÁLISE DE RETURNS
    df['returns'] = df['close'].pct_change()
    
    print("\\nRETURNS ANALYSIS:")
    vol_daily = df['returns'].std()
    vol_annual = vol_daily * np.sqrt(252 * 1440)
    skew = df['returns'].skew()
    kurt = df['returns'].kurtosis()
    autocorr1 = df['returns'].autocorr(lag=1)
    
    print(f"  Volatilidade diaria: {vol_daily:.4f} ({vol_daily*100:.2f}%)")
    print(f"  Volatilidade anualizada: {vol_annual:.1f}%")
    print(f"  Skewness: {skew:.3f}")
    print(f"  Kurtosis: {kurt:.3f}")
    print(f"  Autocorrelacao lag-1: {autocorr1:.4f}")
    
    # TESTE DE NORMALIDADE
    returns_clean = df['returns'].dropna()
    ks_stat, ks_p = stats.kstest(returns_clean, 'norm')
    print(f"  Teste normalidade p-value: {ks_p:.6f}")
    
    # REGIMES
    print("\\nREGIME ANALYSIS:")
    if 'regime' in df.columns:
        regime_counts = df['regime'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"  {regime}: {count:,} ({pct:.1f}%)")
        
        # Performance por regime
        regime_perf = df.groupby('regime')['returns'].agg(['mean', 'std'])
        print(f"\\nPerformance por regime:")
        for regime in regime_perf.index:
            mean_ret = regime_perf.loc[regime, 'mean']
            std_ret = regime_perf.loc[regime, 'std']
            print(f"  {regime}: mean={mean_ret:.6f}, std={std_ret:.6f}")
    
    # SPREADS
    df['spread'] = (df['high'] - df['low']) / df['close']
    avg_spread = df['spread'].mean()
    std_spread = df['spread'].std()
    
    print(f"\\nSPREAD ANALYSIS:")
    print(f"  Spread medio: {avg_spread:.4f} ({avg_spread*100:.2f}%)")
    print(f"  Spread std: {std_spread:.4f}")
    
    # VOLUME
    vol_mean = df['volume'].mean()
    vol_std = df['volume'].std()
    vol_cv = vol_std / vol_mean
    
    print(f"\\nVOLUME ANALYSIS:")
    print(f"  Volume medio: {vol_mean:,.0f}")
    print(f"  Volume CV: {vol_cv:.3f}")
    
    # BUY AND HOLD
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    sharpe_bh = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252 * 1440)
    
    # DRAWDOWN
    cumulative = (1 + df['returns']).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max) - 1
    max_dd = drawdown.min()
    
    print(f"\\nPERFORMANCE ANALYSIS:")
    print(f"  Buy-hold return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  Buy-hold Sharpe: {sharpe_bh:.3f}")
    print(f"  Max drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    
    # SCORE DE DIFICULDADE
    print(f"\\nDIFFICULTY SCORE:")
    difficulty = 0
    
    if vol_daily > 0.015:  # > 1.5%
        print("  [+] Alta volatilidade")
        difficulty += 1
    else:
        print("  [-] Baixa volatilidade")
    
    if abs(autocorr1) < 0.05:
        print("  [+] Baixa autocorrelacao (imprevisivel)")
        difficulty += 1
    else:
        print("  [-] Alta autocorrelacao (previsivel)")
    
    if ks_p < 0.01:  # Não normal
        print("  [+] Distribuicao nao-normal (realista)")
        difficulty += 1
    else:
        print("  [-] Distribuicao muito normal")
    
    if abs(sharpe_bh) < 1.5:
        print("  [+] Buy-hold nao muito bom")
        difficulty += 1
    else:
        print("  [-] Buy-hold muito bom (facil)")
    
    if abs(max_dd) > 0.05:  # > 5%
        print("  [+] Drawdown significativo")
        difficulty += 1
    else:
        print("  [-] Drawdown pequeno")
    
    print(f"\\nSCORE FINAL: {difficulty}/5")
    
    if difficulty >= 4:
        print("VEREDICTO: DATASET MUITO DESAFIADOR")
    elif difficulty >= 3:
        print("VEREDICTO: DATASET ADEQUADO")
    elif difficulty >= 2:
        print("VEREDICTO: DATASET FACIL")
    else:
        print("VEREDICTO: DATASET MUITO FACIL")
    
    # DIAGNÓSTICO ESPECÍFICO PARA CLIP_FRACTION = 0
    print(f"\\nDIAGNOSTICO CLIP_FRACTION = 0:")
    
    if vol_daily < 0.01:
        print("PROBLEMA: Volatilidade muito baixa")
        print("SOLUCAO: Aumentar volatilidade do dataset")
    
    if abs(autocorr1) > 0.1:
        print("PROBLEMA: Dataset muito previsivel")
        print("SOLUCAO: Adicionar mais aleatoriedade")
    
    if difficulty < 3:
        print("PROBLEMA: Dataset muito facil")
        print("SOLUCAO: Criar dataset mais desafiador")
    else:
        print("Dataset OK - problema pode ser:")
        print("- Hiperparametros ainda muito conservadores")
        print("- Modelo já em estado otimo inicial")
        print("- Normalizacao muito agressiva")

if __name__ == '__main__':
    analyze_dataset_simple()