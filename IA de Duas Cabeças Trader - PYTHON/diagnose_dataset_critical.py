#!/usr/bin/env python3
"""
DIAGNÓSTICO CRÍTICO DO DATASET DESAFIADOR
Análise minuciosa do problema de convergência
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def critical_dataset_diagnosis():
    """Diagnóstico crítico para descobrir por que o modelo não converge"""
    
    dataset_path = 'data/GOLD_SAFE_CHALLENGING_2M_20250801_203251.csv'
    
    print("🔍 DIAGNÓSTICO CRÍTICO DO DATASET DESAFIADOR")
    print("="*80)
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset carregado: {len(df):,} barras")
    except FileNotFoundError:
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return
    
    # === ANÁLISE DETALHADA DE RETURNS ===
    df['returns'] = df['close'].pct_change()
    returns_clean = df['returns'].dropna()
    
    print(f"\n📊 ANÁLISE DETALHADA DE RETURNS:")
    print(f"   Mean return: {returns_clean.mean():.8f}")
    print(f"   Std return: {returns_clean.std():.6f}")
    print(f"   Min return: {returns_clean.min():.6f}")
    print(f"   Max return: {returns_clean.max():.6f}")
    print(f"   Skewness: {returns_clean.skew():.4f}")
    print(f"   Kurtosis: {returns_clean.kurtosis():.4f}")
    
    # === PROBLEMA CRÍTICO 1: ZERO RETURNS ===
    zero_returns = np.sum(np.abs(returns_clean) < 1e-8)
    tiny_returns = np.sum(np.abs(returns_clean) < 1e-6)
    small_returns = np.sum(np.abs(returns_clean) < 1e-4)
    
    print(f"\n⚠️  ANÁLISE DE RETURNS EXTREMAMENTE PEQUENOS:")
    print(f"   Zero returns (< 1e-8): {zero_returns:,} ({zero_returns/len(returns_clean)*100:.2f}%)")
    print(f"   Tiny returns (< 1e-6): {tiny_returns:,} ({tiny_returns/len(returns_clean)*100:.2f}%)")
    print(f"   Small returns (< 1e-4): {small_returns:,} ({small_returns/len(returns_clean)*100:.2f}%)")
    
    if zero_returns > len(returns_clean) * 0.01:  # Mais de 1%
        print("   🚨 PROBLEMA CRÍTICO: Muitos returns zero! Modelo não aprende!")
    
    # === PROBLEMA CRÍTICO 2: OHLC SPREAD ANÁLISE ===
    df['ohlc_spread'] = (df['high'] - df['low']) / df['close']
    df['oc_spread'] = np.abs(df['close'] - df['open']) / df['open']
    
    zero_ohlc_spread = np.sum(df['ohlc_spread'] < 1e-6)
    zero_oc_spread = np.sum(df['oc_spread'] < 1e-6)
    
    print(f"\n📏 ANÁLISE DE SPREADS OHLC:")
    print(f"   OHLC spread médio: {df['ohlc_spread'].mean():.6f}")
    print(f"   OC spread médio: {df['oc_spread'].mean():.6f}")
    print(f"   Zero OHLC spreads: {zero_ohlc_spread:,} ({zero_ohlc_spread/len(df)*100:.2f}%)")
    print(f"   Zero OC spreads: {zero_oc_spread:,} ({zero_oc_spread/len(df)*100:.2f}%)")
    
    if zero_ohlc_spread > len(df) * 0.001:  # Mais de 0.1%
        print("   🚨 PROBLEMA CRÍTICO: Barras com spread zero! Sem movimento!")
    
    # === PROBLEMA CRÍTICO 3: REGIME TRANSITIONS ===
    if 'regime' in df.columns:
        regime_changes = np.sum(df['regime'] != df['regime'].shift(1))
        avg_regime_duration = len(df) / regime_changes if regime_changes > 0 else len(df)
        
        print(f"\n🔄 ANÁLISE DE REGIMES:")
        print(f"   Mudanças de regime: {regime_changes:,}")
        print(f"   Duração média de regime: {avg_regime_duration:.1f} barras")
        
        # Análise de performance por regime
        regime_stats = df.groupby('regime')['returns'].agg(['mean', 'std', 'count'])
        print(f"   Performance por regime:")
        for regime in regime_stats.index:
            mean_ret = regime_stats.loc[regime, 'mean']
            std_ret = regime_stats.loc[regime, 'std']
            count = regime_stats.loc[regime, 'count']
            print(f"     {regime}: mean={mean_ret:.8f}, std={std_ret:.6f}, count={count:,}")
        
        # PROBLEMA: Regimes com performance muito similar
        regime_means = regime_stats['mean'].values
        if np.std(regime_means) < 1e-6:
            print("   🚨 PROBLEMA CRÍTICO: Regimes têm performance idêntica! Sem sinal!")
    
    # === PROBLEMA CRÍTICO 4: AUTOCORRELAÇÃO TEMPORAL ===
    autocorrs = [returns_clean.autocorr(lag=i) for i in range(1, 11)]
    max_autocorr = max(np.abs(autocorrs))
    
    print(f"\n📈 ANÁLISE DE AUTOCORRELAÇÃO:")
    print(f"   Autocorr lag-1: {autocorrs[0]:.6f}")
    print(f"   Max autocorr (lags 1-10): {max_autocorr:.6f}")
    
    if max_autocorr < 0.01:
        print("   🚨 PROBLEMA CRÍTICO: Zero autocorrelação! Puramente aleatório!")
    
    # === PROBLEMA CRÍTICO 5: VOLUME CORRELATION ===
    if 'volume' in df.columns:
        volume_price_corr = np.corrcoef(df['volume'], np.abs(df['returns'].fillna(0)))[0,1]
        volume_volatility_corr = np.corrcoef(df['volume'], df['ohlc_spread'])[0,1]
        
        print(f"\n📊 ANÁLISE DE VOLUME:")
        print(f"   Volume-Return correlation: {volume_price_corr:.6f}")
        print(f"   Volume-Volatility correlation: {volume_volatility_corr:.6f}")
        
        if abs(volume_price_corr) < 0.05 and abs(volume_volatility_corr) < 0.05:
            print("   🚨 PROBLEMA CRÍTICO: Volume não correlaciona com preço/volatilidade!")
    
    # === DIAGNÓSTICO DE NORMALIZAÇÃO ===
    print(f"\n🔧 ANÁLISE DE NORMALIZAÇÃO:")
    
    # Simular normalização que o modelo faria
    price_values = df[['open', 'high', 'low', 'close']].values
    price_normalized = (price_values - price_values.mean()) / price_values.std()
    
    extreme_values = np.sum(np.abs(price_normalized) > 3)
    zero_values = np.sum(np.abs(price_normalized) < 1e-6)
    
    print(f"   Valores extremos (|z| > 3): {extreme_values:,}")
    print(f"   Valores próximos de zero: {zero_values:,}")
    
    # === TESTE DE PREDIBILIDADE ===
    print(f"\n🎯 TESTE DE PREDIBILIDADE:")
    
    # Correlação com próximo return
    future_returns = df['returns'].shift(-1)
    current_features = df[['returns', 'ohlc_spread', 'volume']].fillna(0)
    
    correlations = {}
    for col in current_features.columns:
        if col in df.columns:
            corr = np.corrcoef(current_features[col], future_returns.fillna(0))[0,1]
            correlations[col] = corr
            print(f"   {col} -> future_return: {corr:.6f}")
    
    max_pred_corr = max(np.abs(list(correlations.values())))
    if max_pred_corr < 0.01:
        print("   🚨 PROBLEMA CRÍTICO: Dataset não tem predibilidade!")
    
    # === SUMMARY DO DIAGNÓSTICO ===
    print(f"\n{'='*80}")
    print("🔍 RESUMO DO DIAGNÓSTICO:")
    print("="*80)
    
    problems = []
    
    if zero_returns > len(returns_clean) * 0.01:
        problems.append("MUITOS RETURNS ZERO")
    
    if zero_ohlc_spread > len(df) * 0.001:
        problems.append("BARRAS SEM MOVIMENTO")
    
    if 'regime' in df.columns and np.std(regime_stats['mean'].values) < 1e-6:
        problems.append("REGIMES IDÊNTICOS")
    
    if max_autocorr < 0.01:
        problems.append("ZERO AUTOCORRELAÇÃO")
    
    if max_pred_corr < 0.01:
        problems.append("ZERO PREDIBILIDADE")
    
    if len(problems) == 0:
        print("✅ Dataset tecnicamente correto")
        print("💡 Problema pode ser:")
        print("   - Hiperparâmetros muito conservadores")
        print("   - Learning rate muito baixo")
        print("   - Clipping muito agressivo")
        print("   - Normalização muito forte")
    else:
        print("🚨 PROBLEMAS CRÍTICOS ENCONTRADOS:")
        for i, problem in enumerate(problems, 1):
            print(f"   {i}. {problem}")
        
        print(f"\n💡 SOLUÇÕES RECOMENDADAS:")
        if "MUITOS RETURNS ZERO" in problems:
            print("   - Aumentar volatilidade mínima no dataset")
            print("   - Evitar arredondamentos excessivos")
        
        if "BARRAS SEM MOVIMENTO" in problems:
            print("   - Garantir spread mínimo em todas as barras")
            print("   - Revisar lógica de geração OHLC")
        
        if "REGIMES IDÊNTICOS" in problems:
            print("   - Aumentar diferença entre regimes")
            print("   - Revisar parâmetros de drift por regime")
        
        if "ZERO AUTOCORRELAÇÃO" in problems:
            print("   - Adicionar componente de momentum")
            print("   - Introduzir persistence nos returns")
        
        if "ZERO PREDIBILIDADE" in problems:
            print("   - Adicionar features preditivas")
            print("   - Introduzir padrões identificáveis")
    
    # === CRIAR DATASET CORRIGIDO ===
    if len(problems) > 0:
        print(f"\n🔧 GERANDO DATASET CORRIGIDO...")
        create_corrected_dataset(df, problems)

def create_corrected_dataset(original_df, problems):
    """Criar versão corrigida do dataset baseado nos problemas encontrados"""
    
    print("🔧 CRIANDO DATASET CORRIGIDO...")
    
    df = original_df.copy()
    
    # Correção 1: Garantir movimento mínimo
    if "BARRAS SEM MOVIMENTO" in problems:
        min_spread = 0.0001  # 0.01% mínimo
        for idx in range(len(df)):
            current_spread = (df.iloc[idx]['high'] - df.iloc[idx]['low']) / df.iloc[idx]['close']
            if current_spread < min_spread:
                mid_price = (df.iloc[idx]['high'] + df.iloc[idx]['low']) / 2
                df.iloc[idx, df.columns.get_loc('high')] = mid_price * (1 + min_spread/2)
                df.iloc[idx, df.columns.get_loc('low')] = mid_price * (1 - min_spread/2)
        print("   ✅ Spread mínimo aplicado")
    
    # Correção 2: Ajustar regimes para ter diferenças claras
    if "REGIMES IDÊNTICOS" in problems and 'regime' in df.columns:
        # Aplicar drifts mais distintivos por regime
        regime_adjustments = {
            'bull': 0.0002,    # +0.02% drift
            'bear': -0.0002,   # -0.02% drift  
            'sideways': 0.0    # zero drift
        }
        
        for regime, drift in regime_adjustments.items():
            mask = df['regime'] == regime
            if mask.any():
                # Aplicar drift cumulativo
                regime_indices = df[mask].index
                cumulative_drift = np.cumsum([drift] * len(regime_indices))
                df.loc[regime_indices, 'close'] *= (1 + cumulative_drift)
                # Ajustar OHLC proporcionalmente
                df.loc[regime_indices, 'open'] *= (1 + cumulative_drift)
                df.loc[regime_indices, 'high'] *= (1 + cumulative_drift)
                df.loc[regime_indices, 'low'] *= (1 + cumulative_drift)
        
        print("   ✅ Regimes diferenciados aplicados")
    
    # Correção 3: Adicionar autocorrelação
    if "ZERO AUTOCORRELAÇÃO" in problems:
        # Aplicar suavização para criar autocorrelação
        df['returns_raw'] = df['close'].pct_change()
        returns_smoothed = df['returns_raw'].rolling(window=3, center=True).mean().fillna(df['returns_raw'])
        
        # Reconstruir preços com returns suavizados
        df['close_corrected'] = df['close'].iloc[0] * (1 + returns_smoothed).cumprod()
        df['close'] = df['close_corrected']
        
        print("   ✅ Autocorrelação adicionada")
    
    # Salvar dataset corrigido
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corrected_filename = f"data/GOLD_CORRECTED_2M_{timestamp}.csv"
    
    # Remover colunas auxiliares
    cols_to_remove = ['returns_raw', 'close_corrected']
    df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
    
    df.to_csv(corrected_filename, index=False)
    print(f"   ✅ Dataset corrigido salvo: {corrected_filename}")
    
    return corrected_filename

if __name__ == '__main__':
    critical_dataset_diagnosis()