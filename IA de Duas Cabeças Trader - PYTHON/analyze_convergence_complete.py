#!/usr/bin/env python3
"""
Análise completa e robusta do dataset GOLD_TRADING_READY_2M
para identificar problemas de convergência de RL
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_complete():
    """Análise completa do dataset"""
    
    print("="*90)
    print("ANÁLISE COMPLETA DO DATASET GOLD_TRADING_READY_2M - PROBLEMAS DE CONVERGÊNCIA RL")
    print("="*90)
    
    # Carregar dataset
    filepath = r"D:\Projeto\data\GOLD_TRADING_READY_2M_20250803_222334.csv"
    df = pd.read_csv(filepath)
    
    # Converter timestamp e calcular returns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    print(f"\n📊 ESTRUTURA BÁSICA:")
    print(f"   Tamanho: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print(f"   Período: {df.index[0]} até {df.index[-1]}")
    print(f"   Colunas: {list(df.columns)}")
    
    # ========== ANÁLISES CRÍTICAS ==========
    
    print(f"\n🔍 1. ANÁLISE DE REGIMES (CRÍTICO)")
    print("="*50)
    
    if 'regime' in df.columns:
        regime_stats = df.groupby('regime')['returns'].agg(['count', 'mean', 'std']).round(8)
        print("Performance por regime:")
        print(regime_stats)
        
        # Verificar se regimes são distintivos
        mean_diff = regime_stats['mean'].max() - regime_stats['mean'].min()
        std_diff = regime_stats['std'].max() - regime_stats['std'].min()
        
        print(f"\nDiferença entre regimes:")
        print(f"   Returns médios: {mean_diff:.8f}")
        print(f"   Volatilidade: {std_diff:.8f}")
        
        if mean_diff < 0.0001:  # < 0.01%
            print("🚨 PROBLEMA CRÍTICO: Regimes têm performance idêntica - ZERO predibilidade!")
        elif mean_diff < 0.001:  # < 0.1%
            print("⚠️  PROBLEMA: Regimes pouco distintivos - dificulta aprendizado RL")
        else:
            print("✅ Regimes suficientemente distintivos")
    else:
        print("❌ Coluna 'regime' não encontrada")
    
    print(f"\n🔍 2. DISTRIBUIÇÃO DOS RETURNS")
    print("="*50)
    
    returns = df['returns'].dropna()
    
    # Estatísticas básicas
    print(f"Estatísticas dos returns:")
    print(f"   Média: {returns.mean():.8f}")
    print(f"   Desvio padrão: {returns.std():.6f}")
    print(f"   Assimetria: {stats.skew(returns):.6f}")
    print(f"   Curtose: {stats.kurtosis(returns):.6f}")
    
    # Outliers
    Q1, Q3 = returns.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = returns[(returns < Q1 - 1.5*IQR) | (returns > Q3 + 1.5*IQR)]
    outlier_pct = len(outliers) / len(returns) * 100
    
    print(f"   Outliers: {len(outliers):,} ({outlier_pct:.2f}%)")
    print(f"   Range outliers: [{outliers.min():.6f}, {outliers.max():.6f}]")
    
    # Returns zerados
    zero_returns = (returns == 0).sum()
    zero_pct = zero_returns / len(returns) * 100
    print(f"   Returns zerados: {zero_returns:,} ({zero_pct:.3f}%)")
    
    if outlier_pct > 10:
        print("⚠️  PROBLEMA: Muitos outliers podem dificultar aprendizado")
    if zero_pct > 1:
        print("⚠️  PROBLEMA: Muitos returns zerados - possível problema de dados")
    
    print(f"\n🔍 3. AUTOCORRELAÇÃO E PREDIBILIDADE")
    print("="*50)
    
    # Autocorrelação
    lags = [1, 5, 10, 20, 50]
    autocorrs = [returns.autocorr(lag=lag) for lag in lags]
    
    print("Autocorrelação dos returns:")
    for lag, autocorr in zip(lags, autocorrs):
        print(f"   Lag {lag:2d}: {autocorr:.6f}")
    
    # Autocorrelação significativa
    significant_autocorr = [abs(ac) > 0.05 for ac in autocorrs]
    
    if autocorrs[0] < -0.1:
        print("🚨 PROBLEMA CRÍTICO: Forte autocorrelação negativa - dados artificiais?")
    elif any(significant_autocorr):
        print("✅ Autocorrelação detectada - há padrões para RL aprender")
    else:
        print("⚠️  Baixa autocorrelação - dificulta predição")
    
    # Volatilidade clustering
    returns_sq = returns ** 2
    vol_autocorr = [returns_sq.autocorr(lag=lag) for lag in lags[:3]]
    print(f"\nVolatilidade clustering (returns²):")
    for lag, autocorr in zip(lags[:3], vol_autocorr):
        print(f"   Lag {lag:2d}: {autocorr:.6f}")
    
    if vol_autocorr[0] > 0.1:
        print("✅ Clustering de volatilidade detectado - padrão realista")
    else:
        print("⚠️  Pouco clustering de volatilidade")
    
    print(f"\n🔍 4. QUALIDADE DOS DADOS OHLC")
    print("="*50)
    
    # Consistência OHLC
    high_issues = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
    low_issues = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
    
    print(f"Inconsistências OHLC:")
    print(f"   High < max(Open,Close): {high_issues:,}")
    print(f"   Low > min(Open,Close): {low_issues:,}")
    
    if high_issues > 0 or low_issues > 0:
        print("🚨 PROBLEMA CRÍTICO: Dados OHLC inconsistentes!")
    else:
        print("✅ Dados OHLC consistentes")
    
    # Missing values
    missing_total = df.isnull().sum().sum()
    print(f"   Missing values total: {missing_total:,}")
    
    # Gaps extremos
    price_changes = df['close'].pct_change().abs()
    extreme_gaps = (price_changes > 0.1).sum()  # > 10%
    print(f"   Gaps extremos (>10%): {extreme_gaps:,}")
    
    if extreme_gaps > len(df) * 0.001:  # > 0.1%
        print("⚠️  PROBLEMA: Muitos gaps extremos")
    
    print(f"\n🔍 5. VOLUME E CORRELAÇÕES")
    print("="*50)
    
    # Volume statistics
    print(f"Volume:")
    print(f"   Média: {df['volume'].mean():,.0f}")
    print(f"   Desvio padrão: {df['volume'].std():,.0f}")
    print(f"   Min/Max: {df['volume'].min():,.0f} / {df['volume'].max():,.0f}")
    
    # Volume constante (problema sintético)
    volume_unique_pct = df['volume'].nunique() / len(df) * 100
    print(f"   Valores únicos: {df['volume'].nunique():,} ({volume_unique_pct:.2f}%)")
    
    if volume_unique_pct < 1:
        print("🚨 PROBLEMA: Volume muito repetitivo - dados sintéticos mal construídos")
    
    # Correlações importantes
    vol_ret_corr = df['volume'].corr(returns.abs())
    vol_range_corr = df['volume'].corr((df['high'] - df['low']) / df['close'])
    
    print(f"\nCorrelações:")
    print(f"   Volume vs |Returns|: {vol_ret_corr:.6f}")
    print(f"   Volume vs Range: {vol_range_corr:.6f}")
    
    if abs(vol_ret_corr) < 0.01 and abs(vol_range_corr) < 0.01:
        print("🚨 PROBLEMA: Volume não correlacionado - elimina informação técnica")
    
    print(f"\n🔍 6. INDICADORES TÉCNICOS")
    print("="*50)
    
    # Verificar indicadores existentes
    tech_indicators = [col for col in df.columns if any(ind in col.lower() 
                      for ind in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'momentum', 'stoch'])]
    
    print(f"Indicadores técnicos encontrados: {len(tech_indicators)}")
    print(f"   {tech_indicators}")
    
    if len(tech_indicators) == 0:
        print("🚨 PROBLEMA CRÍTICO: Nenhum indicador técnico - agente RL precisa de features!")
        
        # Criar indicadores básicos para análise
        print("\nCriando indicadores básicos para análise...")
        df['sma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = returns.rolling(20).std()
        
        # RSI simplificado
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        tech_indicators = ['sma_20', 'volatility', 'rsi']
        print(f"   Criados: {tech_indicators}")
    
    # Análise dos indicadores
    for indicator in tech_indicators[:3]:  # Analisar primeiros 3
        if indicator in df.columns:
            ind_data = df[indicator].dropna()
            if len(ind_data) > 100:
                ind_std = ind_data.std()
                ind_range = ind_data.max() - ind_data.min()
                
                print(f"\n{indicator.upper()}:")
                print(f"   Range: [{ind_data.min():.4f}, {ind_data.max():.4f}]")
                print(f"   Std: {ind_std:.6f}")
                
                if ind_std < 1e-6:
                    print(f"   🚨 PROBLEMA: {indicator} é praticamente constante!")
    
    print(f"\n🔍 7. ANÁLISE TEMPORAL E REGIMES")
    print("="*50)
    
    # Análise por períodos
    df_temp = df.copy()
    df_temp.reset_index(inplace=True)
    df_temp['period'] = df_temp.index // 100000  # Períodos de 100k
    
    period_stats = df_temp.groupby('period')['returns'].agg(['mean', 'std']).round(8)
    print("Estatísticas por período (100k obs):")
    print(period_stats.head(10))
    
    # Verificar drift temporal
    period_means = period_stats['mean']
    mean_trend = np.corrcoef(range(len(period_means)), period_means)[0,1]
    print(f"\nTendência temporal dos returns: {mean_trend:.6f}")
    
    if abs(mean_trend) > 0.3:
        print("⚠️  PROBLEMA: Forte tendência temporal - dados não estacionários")
    
    # Estabilidade da volatilidade
    period_stds = period_stats['std']
    vol_cv = period_stds.std() / period_stds.mean()
    print(f"Coeficiente de variação da volatilidade: {vol_cv:.4f}")
    
    if vol_cv > 0.5:
        print("⚠️  PROBLEMA: Volatilidade muito instável entre períodos")
    
    print(f"\n🔍 8. RESUMO DE PROBLEMAS CRÍTICOS PARA RL")
    print("="*60)
    
    problems = []
    
    # 1. Regimes não distintivos
    if 'regime' in df.columns:
        regime_stats = df.groupby('regime')['returns'].agg(['mean', 'std'])
        mean_diff = regime_stats['mean'].max() - regime_stats['mean'].min()
        if mean_diff < 0.0001:
            problems.append("CRÍTICO: Regimes com performance idêntica - zero predibilidade")
        elif mean_diff < 0.001:
            problems.append("GRAVE: Regimes pouco distintivos - dificulta aprendizado")
    
    # 2. Falta de features técnicas
    if len(tech_indicators) == 0:
        problems.append("CRÍTICO: Nenhum indicador técnico no dataset")
    elif len(tech_indicators) < 5:
        problems.append("GRAVE: Poucos indicadores técnicos - features insuficientes")
    
    # 3. Dados OHLC inconsistentes
    if high_issues > 0 or low_issues > 0:
        problems.append(f"CRÍTICO: {high_issues + low_issues} inconsistências OHLC")
    
    # 4. Volume não correlacionado
    if abs(vol_ret_corr) < 0.01:
        problems.append("GRAVE: Volume não correlacionado com price action")
    
    # 5. Autocorrelação problemática
    if autocorrs[0] < -0.1:
        problems.append("CRÍTICO: Autocorrelação negativa extrema - dados artificiais")
    elif all(abs(ac) < 0.02 for ac in autocorrs):
        problems.append("GRAVE: Ausência de autocorrelação - dificulta predição")
    
    # 6. Returns zerados excessivos
    if zero_pct > 1:
        problems.append(f"GRAVE: {zero_pct:.2f}% returns zerados - problema de dados")
    
    # 7. Outliers excessivos
    if outlier_pct > 10:
        problems.append(f"MODERADO: {outlier_pct:.1f}% outliers - pode dificultar treinamento")
    
    # 8. Volume sintético
    if volume_unique_pct < 1:
        problems.append("GRAVE: Volume muito repetitivo - sintético mal construído")
    
    # 9. Volatilidade instável
    if vol_cv > 0.5:
        problems.append("MODERADO: Volatilidade muito instável entre períodos")
    
    # Resumo final
    print(f"TOTAL DE PROBLEMAS IDENTIFICADOS: {len(problems)}")
    print()
    
    for i, problem in enumerate(problems, 1):
        severity = problem.split(':')[0]
        if severity == "CRÍTICO":
            print(f"🚨 {i:2d}. {problem}")
        elif severity == "GRAVE":
            print(f"⚠️  {i:2d}. {problem}")
        else:
            print(f"📋 {i:2d}. {problem}")
    
    # Diagnóstico final
    critical_count = sum(1 for p in problems if p.startswith("CRÍTICO"))
    grave_count = sum(1 for p in problems if p.startswith("GRAVE"))
    
    print(f"\n" + "="*60)
    print("DIAGNÓSTICO FINAL DE CONVERGÊNCIA RL")
    print("="*60)
    
    if critical_count > 0:
        print("🚨 CONVERGÊNCIA: IMPOSSÍVEL")
        print(f"   {critical_count} problemas críticos impedem qualquer aprendizado")
        print("   AÇÃO: Recriar dataset completamente")
    elif grave_count >= 3:
        print("⚠️  CONVERGÊNCIA: MUITO IMPROVÁVEL") 
        print(f"   {grave_count} problemas graves dificultam severamente o aprendizado")
        print("   AÇÃO: Corrigir problemas graves antes do treinamento")
    elif grave_count >= 1:
        print("📋 CONVERGÊNCIA: POSSÍVEL MAS DIFÍCIL")
        print(f"   {grave_count} problemas graves podem atrasar convergência")
        print("   AÇÃO: Corrigir se possível, monitorar treinamento")
    else:
        print("✅ CONVERGÊNCIA: PROVÁVEL")
        print("   Dataset adequado para treinamento RL")
    
    print(f"\nTotal de observações analisadas: {len(df):,}")
    print(f"Período de análise: {df.index[0].strftime('%Y-%m-%d')} até {df.index[-1].strftime('%Y-%m-%d')}")
    
    print("\n" + "="*90)
    print("ANÁLISE CONCLUÍDA")
    print("="*90)
    
    return df, problems

if __name__ == "__main__":
    df, problems = analyze_dataset_complete()