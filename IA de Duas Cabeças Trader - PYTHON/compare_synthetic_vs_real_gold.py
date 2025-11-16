#!/usr/bin/env python3
"""
COMPARAÇÃO: Dataset Sintético vs Dados Reais do Ouro (Yahoo)
Análise de coerência dos padrões
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_real_gold_data():
    """Carregar dados reais do ouro do Yahoo"""
    # Tentar carregar o dataset mais recente do Yahoo
    yahoo_files = [
        'data/GC=F_YAHOO_DAILY_5MIN_20250711_041924.csv',
        'data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv',
        'data/GOLD_5m_20250704_122438.csv',
        'data/GOLD_COMPLETE_20250704_122438.csv'
    ]
    
    for file_path in yahoo_files:
        try:
            print(f"Tentando carregar: {file_path}")
            df = pd.read_csv(file_path)
            
            # Padronizar nomes das colunas
            df.columns = df.columns.str.lower()
            column_mapping = {
                'datetime': 'timestamp',
                'date': 'timestamp', 
                'time': 'timestamp'
            }
            df = df.rename(columns=column_mapping)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Verificar se tem colunas OHLCV
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                print(f"✅ Dataset real carregado: {file_path}")
                print(f"   Período: {df.index.min()} até {df.index.max()}")
                print(f"   Barras: {len(df):,}")
                return df
                
        except Exception as e:
            print(f"❌ Erro ao carregar {file_path}: {e}")
            continue
    
    raise FileNotFoundError("❌ Nenhum dataset real do ouro encontrado!")

def load_synthetic_data():
    """Carregar nosso dataset sintético"""
    synthetic_path = 'data/GOLD_COHERENT_FIXED_2M_20250803_211100.csv'
    
    df = pd.read_csv(synthetic_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print(f"✅ Dataset sintético carregado: {synthetic_path}")
    print(f"   Período: {df.index.min()} até {df.index.max()}")
    print(f"   Barras: {len(df):,}")
    
    return df

def analyze_returns_patterns(df_real, df_synthetic):
    """Comparar padrões de returns"""
    print("\n📈 ANÁLISE DE RETURNS:")
    print("="*60)
    
    # Calcular returns
    df_real['returns'] = df_real['close'].pct_change()
    df_synthetic['returns'] = df_synthetic['close'].pct_change()
    
    # Remover outliers extremos para comparação justa
    real_returns = df_real['returns'].dropna()
    real_returns = real_returns[np.abs(real_returns) < real_returns.std() * 5]
    
    synthetic_returns = df_synthetic['returns'].dropna()
    synthetic_returns = synthetic_returns[np.abs(synthetic_returns) < synthetic_returns.std() * 5]
    
    # Estatísticas comparativas
    stats_comparison = {
        'Métrica': ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max'],
        'Real': [
            real_returns.mean(),
            real_returns.std(),
            real_returns.skew(),
            real_returns.kurtosis(),
            real_returns.min(),
            real_returns.max()
        ],
        'Sintético': [
            synthetic_returns.mean(),
            synthetic_returns.std(),
            synthetic_returns.skew(),
            synthetic_returns.kurtosis(),
            synthetic_returns.min(),
            synthetic_returns.max()
        ]
    }
    
    for i, metric in enumerate(stats_comparison['Métrica']):
        real_val = stats_comparison['Real'][i]
        synth_val = stats_comparison['Sintético'][i]
        diff_pct = abs(real_val - synth_val) / abs(real_val) * 100 if real_val != 0 else 0
        
        print(f"{metric:12} | Real: {real_val:10.6f} | Sintético: {synth_val:10.6f} | Diff: {diff_pct:6.1f}%")
    
    # Teste de normalidade
    real_ks_stat, real_ks_p = stats.kstest(real_returns, 'norm')
    synth_ks_stat, synth_ks_p = stats.kstest(synthetic_returns, 'norm')
    
    print(f"\nTeste Normalidade:")
    print(f"Real p-value:      {real_ks_p:.6f}")
    print(f"Sintético p-value: {synth_ks_p:.6f}")
    
    return real_returns, synthetic_returns

def analyze_volatility_clustering(df_real, df_synthetic):
    """Analisar clustering de volatilidade"""
    print("\n📊 ANÁLISE DE VOLATILITY CLUSTERING:")
    print("="*60)
    
    # Calcular volatilidade rolling
    window = 20
    df_real['vol_rolling'] = df_real['returns'].rolling(window).std()
    df_synthetic['vol_rolling'] = df_synthetic['returns'].rolling(window).std()
    
    # Autocorrelação da volatilidade
    real_vol_autocorr = []
    synth_vol_autocorr = []
    
    for lag in range(1, 11):
        real_autocorr = df_real['vol_rolling'].dropna().autocorr(lag=lag)
        synth_autocorr = df_synthetic['vol_rolling'].dropna().autocorr(lag=lag)
        
        real_vol_autocorr.append(real_autocorr)
        synth_vol_autocorr.append(synth_autocorr)
        
        if lag <= 5:
            print(f"Lag-{lag:2d} | Real: {real_autocorr:7.4f} | Sintético: {synth_autocorr:7.4f}")
    
    # Média da autocorrelação (indicador de clustering)
    real_clustering = np.mean(real_vol_autocorr[:5])  # Primeiros 5 lags
    synth_clustering = np.mean(synth_vol_autocorr[:5])
    
    print(f"\nClustering Score (média lag 1-5):")
    print(f"Real:      {real_clustering:.4f}")
    print(f"Sintético: {synth_clustering:.4f}")
    print(f"Diferença: {abs(real_clustering - synth_clustering):.4f}")
    
    return real_clustering, synth_clustering

def analyze_price_patterns(df_real, df_synthetic):
    """Analisar padrões de preço"""
    print("\n💰 ANÁLISE DE PADRÕES DE PREÇO:")
    print("="*60)
    
    # Spreads OHLC
    df_real['ohlc_spread'] = (df_real['high'] - df_real['low']) / df_real['close']
    df_synthetic['ohlc_spread'] = (df_synthetic['high'] - df_synthetic['low']) / df_synthetic['close']
    
    df_real['oc_spread'] = abs(df_real['close'] - df_real['open']) / df_real['open']
    df_synthetic['oc_spread'] = abs(df_synthetic['close'] - df_synthetic['open']) / df_synthetic['open']
    
    print("SPREADS MÉDIOS:")
    print(f"OHLC Spread | Real: {df_real['ohlc_spread'].mean():.4f} | Sintético: {df_synthetic['ohlc_spread'].mean():.4f}")
    print(f"OC Spread   | Real: {df_real['oc_spread'].mean():.4f} | Sintético: {df_synthetic['oc_spread'].mean():.4f}")
    
    # Tendências de longo prazo
    window_trend = 100
    df_real['trend_100'] = df_real['close'].rolling(window_trend).mean()
    df_synthetic['trend_100'] = df_synthetic['close'].rolling(window_trend).mean()
    
    df_real['trend_signal'] = (df_real['close'] > df_real['trend_100']).astype(int)
    df_synthetic['trend_signal'] = (df_synthetic['close'] > df_synthetic['trend_100']).astype(int)
    
    # Frequência de mudanças de tendência
    real_trend_changes = (df_real['trend_signal'].diff() != 0).sum()
    synth_trend_changes = (df_synthetic['trend_signal'].diff() != 0).sum()
    
    real_trend_freq = real_trend_changes / len(df_real) * 1000
    synth_trend_freq = synth_trend_changes / len(df_synthetic) * 1000
    
    print(f"\nMUDANÇAS DE TENDÊNCIA (por 1000 barras):")
    print(f"Real:      {real_trend_freq:.2f}")
    print(f"Sintético: {synth_trend_freq:.2f}")
    
    return df_real, df_synthetic

def analyze_volume_patterns(df_real, df_synthetic):
    """Analisar padrões de volume"""
    print("\n📊 ANÁLISE DE VOLUME:")
    print("="*60)
    
    if 'volume' not in df_real.columns:
        print("❌ Volume não disponível nos dados reais")
        return
    
    # Correlação volume vs volatilidade
    df_real['abs_returns'] = df_real['returns'].abs()
    df_synthetic['abs_returns'] = df_synthetic['returns'].abs()
    
    real_vol_corr = df_real['volume'].corr(df_real['abs_returns'])
    synth_vol_corr = df_synthetic['volume'].corr(df_synthetic['abs_returns'])
    
    print(f"Correlação Volume-Volatilidade:")
    print(f"Real:      {real_vol_corr:.4f}")
    print(f"Sintético: {synth_vol_corr:.4f}")
    print(f"Diferença: {abs(real_vol_corr - synth_vol_corr):.4f}")
    
    # Estatísticas de volume
    real_vol_cv = df_real['volume'].std() / df_real['volume'].mean()
    synth_vol_cv = df_synthetic['volume'].std() / df_synthetic['volume'].mean()
    
    print(f"\nCoeficiente de Variação do Volume:")
    print(f"Real:      {real_vol_cv:.3f}")
    print(f"Sintético: {synth_vol_cv:.3f}")

def analyze_regime_identification(df_real, df_synthetic):
    """Identificar regimes nos dados reais e comparar"""
    print("\n🎯 ANÁLISE DE REGIMES:")
    print("="*60)
    
    # Identificar regimes nos dados reais usando volatilidade e returns
    window = 50
    df_real['vol_regime'] = df_real['returns'].rolling(window).std()
    df_real['return_regime'] = df_real['returns'].rolling(window).mean()
    
    # Classificar regimes baseado em quartis
    vol_quartiles = df_real['vol_regime'].quantile([0.33, 0.67]).values
    return_quartiles = df_real['return_regime'].quantile([0.33, 0.67]).values
    
    def classify_regime(row):
        if pd.isna(row['vol_regime']) or pd.isna(row['return_regime']):
            return 'unknown'
        
        vol = row['vol_regime']
        ret = row['return_regime']
        
        if vol > vol_quartiles[1]:  # Alta volatilidade
            return 'volatile'
        elif ret > return_quartiles[1]:  # Alto return
            return 'bull'
        elif ret < return_quartiles[0]:  # Baixo return
            return 'bear'
        else:
            return 'sideways'
    
    df_real['regime_detected'] = df_real.apply(classify_regime, axis=1)
    
    # Comparar distribuição de regimes
    real_regime_dist = df_real['regime_detected'].value_counts(normalize=True)
    synth_regime_dist = df_synthetic['regime'].value_counts(normalize=True)
    
    print("DISTRIBUIÇÃO DE REGIMES:")
    print(f"{'Regime':<10} | {'Real %':<8} | {'Sintético %':<12} | {'Diferença'}")
    print("-" * 50)
    
    all_regimes = set(real_regime_dist.index) | set(synth_regime_dist.index)
    
    for regime in sorted(all_regimes):
        real_pct = real_regime_dist.get(regime, 0) * 100
        synth_pct = synth_regime_dist.get(regime, 0) * 100
        diff = abs(real_pct - synth_pct)
        
        print(f"{regime:<10} | {real_pct:6.1f}%  | {synth_pct:8.1f}%    | {diff:6.1f}%")
    
    # Analisar performance por regime nos dados reais
    if len(real_regime_dist) > 1:
        print(f"\nPERFORMANCE POR REGIME (dados reais):")
        real_regime_stats = df_real.groupby('regime_detected')['returns'].agg(['mean', 'std']).round(6)
        for regime in real_regime_stats.index:
            if regime != 'unknown':
                mean_ret = real_regime_stats.loc[regime, 'mean']
                std_ret = real_regime_stats.loc[regime, 'std']
                print(f"  {regime}: mean={mean_ret:.6f}, std={std_ret:.6f}")

def generate_coherence_report(df_real, df_synthetic):
    """Gerar relatório final de coerência"""
    print("\n" + "="*80)
    print("🎯 RELATÓRIO DE COERÊNCIA FINAL")
    print("="*80)
    
    # Calcular score de coerência
    coherence_scores = []
    
    # 1. Volatilidade similar
    real_vol = df_real['returns'].std()
    synth_vol = df_synthetic['returns'].std()
    vol_diff = abs(real_vol - synth_vol) / real_vol
    vol_score = max(0, 1 - vol_diff * 2)  # Penalizar diferenças > 50%
    coherence_scores.append(vol_score)
    print(f"1. Volatilidade:      {vol_score:.3f} (diff: {vol_diff*100:.1f}%)")
    
    # 2. Skewness similar
    real_skew = df_real['returns'].skew()
    synth_skew = df_synthetic['returns'].skew()
    skew_diff = abs(real_skew - synth_skew) / abs(real_skew) if real_skew != 0 else 0
    skew_score = max(0, 1 - skew_diff)
    coherence_scores.append(skew_score)
    print(f"2. Skewness:          {skew_score:.3f} (diff: {skew_diff*100:.1f}%)")
    
    # 3. Spread patterns
    real_spread = df_real['ohlc_spread'].mean()
    synth_spread = df_synthetic['ohlc_spread'].mean()
    spread_diff = abs(real_spread - synth_spread) / real_spread
    spread_score = max(0, 1 - spread_diff * 2)
    coherence_scores.append(spread_score)
    print(f"3. OHLC Spreads:      {spread_score:.3f} (diff: {spread_diff*100:.1f}%)")
    
    # 4. Autocorrelação
    real_autocorr = df_real['returns'].autocorr(lag=1)
    synth_autocorr = df_synthetic['returns'].autocorr(lag=1)
    autocorr_diff = abs(real_autocorr - synth_autocorr)
    autocorr_score = max(0, 1 - autocorr_diff * 5)  # Autocorr pequena, penalizar mais
    coherence_scores.append(autocorr_score)
    print(f"4. Autocorrelação:    {autocorr_score:.3f} (diff: {autocorr_diff:.4f})")
    
    # Score final
    final_score = np.mean(coherence_scores)
    
    print(f"\n🎯 SCORE DE COERÊNCIA FINAL: {final_score:.3f}")
    
    if final_score >= 0.8:
        print("✅ EXCELENTE: Dataset sintético altamente coerente com dados reais")
    elif final_score >= 0.6:
        print("✅ BOM: Dataset sintético adequadamente coerente")
    elif final_score >= 0.4:
        print("⚠️  MODERADO: Algumas diferenças significativas detectadas")
    else:
        print("❌ BAIXO: Dataset sintético precisa de ajustes")
    
    return final_score

def main():
    print("🔍 COMPARAÇÃO: DATASET SINTÉTICO vs DADOS REAIS DO OURO")
    print("="*80)
    
    try:
        # Carregar datasets
        df_real = load_real_gold_data()
        df_synthetic = load_synthetic_data()
        
        # Limitar dados reais ao mesmo período de análise (últimas barras)
        if len(df_real) > len(df_synthetic):
            df_real = df_real.tail(min(len(df_synthetic), 500000))  # Máximo 500k barras
            print(f"📊 Dados reais limitados a {len(df_real):,} barras para comparação")
        
        # Executar análises
        real_returns, synth_returns = analyze_returns_patterns(df_real, df_synthetic)
        analyze_volatility_clustering(df_real, df_synthetic)
        analyze_price_patterns(df_real, df_synthetic)
        analyze_volume_patterns(df_real, df_synthetic)
        analyze_regime_identification(df_real, df_synthetic)
        
        # Relatório final
        coherence_score = generate_coherence_report(df_real, df_synthetic)
        
        print(f"\n💾 Análise concluída com score de coerência: {coherence_score:.3f}")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == '__main__':
    main()