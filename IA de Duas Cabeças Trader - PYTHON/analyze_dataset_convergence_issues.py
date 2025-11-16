#!/usr/bin/env python3
"""
Análise completa do dataset GOLD_TRADING_READY_2M para identificar problemas
que possam impedir convergência de RL.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_examine_dataset(filepath):
    """Carrega e examina estrutura básica do dataset"""
    print("="*80)
    print("1. ESTRUTURA BÁSICA DO DATASET")
    print("="*80)
    
    # Carregar dataset
    df = pd.read_csv(filepath)
    
    print(f"Tamanho do dataset: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print(f"Período de tempo: {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]}")
    
    # Converter timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"\nColunas disponíveis:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col} ({df[col].dtype})")
    
    # Estatísticas básicas de tamanho
    print(f"\nMemória utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def calculate_descriptive_stats(df):
    """Calcula estatísticas descritivas dos returns e features principais"""
    print("\n" + "="*80)
    print("2. ESTATÍSTICAS DESCRITIVAS")
    print("="*80)
    
    # Calcular returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Estatísticas básicas dos preços
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    print("\nEstatísticas dos Preços:")
    print(df[price_cols].describe())
    
    # Estatísticas dos returns
    print("\nEstatísticas dos Returns:")
    returns_stats = df[['returns', 'log_returns']].describe()
    print(returns_stats)
    
    # Estatísticas avançadas dos returns
    returns = df['returns'].dropna()
    log_returns = df['log_returns'].dropna()
    
    print(f"\nEstatísticas Avançadas dos Returns:")
    print(f"  Assimetria (Skewness): {stats.skew(returns):.6f}")
    print(f"  Curtose (Kurtosis): {stats.kurtosis(returns):.6f}")
    print(f"  Teste Jarque-Bera: {stats.jarque_bera(returns)}")
    
    # Volatilidade
    print(f"\nVolatilidade:")
    print(f"  Returns std: {returns.std():.6f}")
    print(f"  Log returns std: {log_returns.std():.6f}")
    print(f"  Volatilidade anualizada: {returns.std() * np.sqrt(252*288):.6f}")  # 288 períodos de 5min por dia
    
    return df

def analyze_returns_distribution(df):
    """Analisa distribuição dos returns, outliers e anomalias"""
    print("\n" + "="*80)
    print("3. ANÁLISE DA DISTRIBUIÇÃO DOS RETURNS")
    print("="*80)
    
    returns = df['returns'].dropna()
    
    # Outliers usando IQR
    Q1 = returns.quantile(0.25)
    Q3 = returns.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
    
    print(f"Outliers identificados:")
    print(f"  Total de outliers: {len(outliers):,} ({len(outliers)/len(returns)*100:.2f}%)")
    print(f"  Range normal: [{lower_bound:.6f}, {upper_bound:.6f}]")
    print(f"  Outliers extremos: min={outliers.min():.6f}, max={outliers.max():.6f}")
    
    # Percentis extremos
    percentiles = [0.1, 1, 5, 95, 99, 99.9]
    print(f"\nPercentis dos returns:")
    for p in percentiles:
        print(f"  {p:4.1f}%: {returns.quantile(p/100):.6f}")
    
    # Teste de normalidade
    shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Shapiro limitado a 5000 amostras
    ks_stat, ks_p = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
    
    print(f"\nTestes de Normalidade:")
    print(f"  Shapiro-Wilk: stat={shapiro_stat:.6f}, p-value={shapiro_p:.2e}")
    print(f"  Kolmogorov-Smirnov: stat={ks_stat:.6f}, p-value={ks_p:.2e}")
    
    # Análise de zeros
    zero_returns = (returns == 0).sum()
    print(f"\nReturns zerados: {zero_returns:,} ({zero_returns/len(returns)*100:.2f}%)")
    
    return returns

def analyze_autocorrelation(df):
    """Analisa autocorrelação dos returns"""
    print("\n" + "="*80)
    print("4. ANÁLISE DE AUTOCORRELAÇÃO")
    print("="*80)
    
    returns = df['returns'].dropna()
    
    # Autocorrelação em diferentes lags
    lags = [1, 5, 10, 20, 50, 100, 288]  # 288 = 1 dia em períodos de 5min
    print("Autocorrelação dos returns:")
    for lag in lags:
        if lag < len(returns):
            autocorr = returns.autocorr(lag=lag)
            print(f"  Lag {lag:3d}: {autocorr:.6f}")
    
    # Análise manual de autocorrelação significativa
    print(f"\nAnálise de autocorrelação significativa:")
    significant_lags = []
    for lag in lags:
        if lag < len(returns):
            autocorr = returns.autocorr(lag=lag)
            if abs(autocorr) > 0.05:  # Threshold para significância
                significant_lags.append((lag, autocorr))
    
    if significant_lags:
        print("  Lags com autocorrelação significativa (>0.05):")
        for lag, autocorr in significant_lags:
            print(f"    Lag {lag}: {autocorr:.6f}")
    else:
        print("  Nenhuma autocorrelação significativa detectada")
    
    # Autocorrelação dos returns ao quadrado (heterocedasticidade)
    returns_sq = returns ** 2
    print(f"\nAutocorrelação dos returns ao quadrado (volatilidade clustering):")
    for lag in lags[:5]:  # Apenas primeiros lags
        if lag < len(returns_sq):
            autocorr_sq = returns_sq.autocorr(lag=lag)
            print(f"  Lag {lag:3d}: {autocorr_sq:.6f}")

def test_stationarity(df):
    """Testa estacionariedade dos dados usando métodos simples"""
    print("\n" + "="*80)
    print("5. ANÁLISE DE ESTACIONARIEDADE")
    print("="*80)
    
    price_series = df['close'].dropna()
    returns_series = df['returns'].dropna()
    
    # Análise visual de estacionariedade - preços
    print("Análise PREÇOS:")
    rolling_mean = price_series.rolling(window=1000).mean()
    rolling_std = price_series.rolling(window=1000).std()
    
    mean_variation = (rolling_mean.max() - rolling_mean.min()) / price_series.mean() * 100
    std_variation = (rolling_std.max() - rolling_std.min()) / rolling_std.mean() * 100
    
    print(f"  Variação da média móvel: {mean_variation:.2f}%")
    print(f"  Variação do desvio padrão móvel: {std_variation:.2f}%")
    print(f"  Resultado: {'Não-estacionário (esperado)' if mean_variation > 10 else 'Possivelmente estacionário'}")
    
    # Análise visual de estacionariedade - returns
    print(f"\nAnálise RETURNS:")
    returns_rolling_mean = returns_series.rolling(window=1000).mean()
    returns_rolling_std = returns_series.rolling(window=1000).std()
    
    returns_mean_variation = abs(returns_rolling_mean.max() - returns_rolling_mean.min()) / abs(returns_series.mean()) * 100 if returns_series.mean() != 0 else 0
    returns_std_variation = (returns_rolling_std.max() - returns_rolling_std.min()) / returns_rolling_std.mean() * 100
    
    print(f"  Variação da média móvel: {returns_mean_variation:.2f}%")
    print(f"  Variação do desvio padrão móvel: {returns_std_variation:.2f}%")
    print(f"  Média dos returns: {returns_series.mean():.6f}")
    
    if abs(returns_series.mean()) < 1e-4 and returns_std_variation < 50:
        print(f"  Resultado: Provavelmente estacionário")
    else:
        print(f"  Resultado: Possivelmente não-estacionário")

def analyze_technical_indicators(df):
    """Analisa qualidade dos indicadores técnicos (se existirem)"""
    print("\n" + "="*80)
    print("6. ANÁLISE DE INDICADORES TÉCNICOS")
    print("="*80)
    
    # Verificar se existem colunas de indicadores técnicos
    possible_indicators = ['sma', 'ema', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'momentum']
    existing_indicators = [col for col in df.columns if any(ind in col.lower() for ind in possible_indicators)]
    
    print(f"Colunas disponíveis: {list(df.columns)}")
    print(f"Indicadores técnicos encontrados: {existing_indicators}")
    
    if not existing_indicators:
        print("PROBLEMA: Nenhum indicador técnico encontrado no dataset!")
        print("Isso pode ser um problema crítico para RL - agente precisa de features técnicas")
        
        # Criar alguns indicadores básicos para análise
        print("\nCriando indicadores básicos para análise:")
        
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI simplificado
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatilidade
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        existing_indicators = ['sma_20', 'sma_50', 'rsi', 'volatility']
        print(f"Indicadores criados: {existing_indicators}")
    
    # Analisar qualidade dos indicadores
    for indicator in existing_indicators:
        if indicator in df.columns:
            ind_data = df[indicator].dropna()
            if len(ind_data) > 0:
                print(f"\n{indicator.upper()}:")
                print(f"  Missing values: {df[indicator].isna().sum():,} ({df[indicator].isna().sum()/len(df)*100:.2f}%)")
                print(f"  Range: [{ind_data.min():.4f}, {ind_data.max():.4f}]")
                print(f"  Mean ± Std: {ind_data.mean():.4f} ± {ind_data.std():.4f}")
                
                # Verificar valores extremos ou constantes
                if ind_data.std() == 0:
                    print(f"  ⚠️  PROBLEMA: {indicator} é constante!")
                elif ind_data.std() < 1e-6:
                    print(f"  ⚠️  PROBLEMA: {indicator} tem variação muito baixa!")

def analyze_temporal_patterns(df):
    """Identifica padrões temporais"""
    print("\n" + "="*80)
    print("7. ANÁLISE DE PADRÕES TEMPORAIS")
    print("="*80)
    
    # Adicionar componentes temporais
    df_temp = df.copy()
    df_temp.reset_index(inplace=True)
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    df_temp['day_of_week'] = df_temp['timestamp'].dt.dayofweek
    df_temp['day_of_month'] = df_temp['timestamp'].dt.day
    
    # Análise por hora
    hourly_stats = df_temp.groupby('hour')['returns'].agg(['count', 'mean', 'std']).round(6)
    print("Estatísticas por Hora do Dia:")
    print(hourly_stats)
    
    # Verificar se há padrões óbvios
    hourly_mean_range = hourly_stats['mean'].max() - hourly_stats['mean'].min()
    print(f"\nRange de returns médios por hora: {hourly_mean_range:.6f}")
    
    # Análise por dia da semana
    daily_stats = df_temp.groupby('day_of_week')['returns'].agg(['count', 'mean', 'std']).round(6)
    print(f"\nEstatísticas por Dia da Semana:")
    print(daily_stats)
    
    # Verificar tendências
    df_temp['period'] = df_temp.index // 10000  # Dividir em períodos
    period_stats = df_temp.groupby('period')['returns'].agg(['mean', 'std']).round(6)
    print(f"\nEstatísticas por Período (cada ~10k observações):")
    print(period_stats.head(10))
    
    # Verificar se há drift nos returns
    df_temp['index_num'] = range(len(df_temp))
    trend_corr = df_temp['returns'].corr(df_temp['index_num'])
    print(f"\nCorrelação returns vs tempo (drift): {trend_corr:.6f}")
    if abs(trend_corr) > 0.1:
        print("⚠️  PROBLEMA: Possível trend/drift nos returns!")

def check_data_quality(df):
    """Verifica missing values e dados corrompidos"""
    print("\n" + "="*80)
    print("8. VERIFICAÇÃO DE QUALIDADE DOS DADOS")
    print("="*80)
    
    # Missing values
    print("Missing Values por coluna:")
    missing_stats = df.isnull().sum()
    for col, missing_count in missing_stats.items():
        pct = missing_count / len(df) * 100
        print(f"  {col}: {missing_count:,} ({pct:.2f}%)")
    
    # Valores infinitos
    print(f"\nValores Infinitos:")
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"  {col}: {inf_count:,} valores infinitos")
    
    # OHLC consistency
    print(f"\nConsistência OHLC:")
    ohlc_issues = 0
    
    # High >= max(Open, Close) and Low <= min(Open, Close)
    high_issues = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
    low_issues = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
    
    print(f"  High < max(Open,Close): {high_issues:,}")
    print(f"  Low > min(Open,Close): {low_issues:,}")
    
    if high_issues > 0 or low_issues > 0:
        print("⚠️  PROBLEMA: Inconsistências em dados OHLC!")
        ohlc_issues = high_issues + low_issues
    
    # Verificar gaps extremos
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        price_changes = df[col].pct_change().abs()
        extreme_changes = (price_changes > 0.1).sum()  # Mudanças > 10%
        if extreme_changes > 0:
            print(f"  {col}: {extreme_changes:,} mudanças extremas (>10%)")
    
    # Verificar volume zero
    zero_volume = (df['volume'] == 0).sum()
    print(f"  Volume zero: {zero_volume:,} ({zero_volume/len(df)*100:.2f}%)")
    
    # Verificar duplicatas de timestamp
    if 'timestamp' in df.index.names or 'timestamp' in df.columns:
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
        else:
            duplicates = df.index.duplicated().sum()
        print(f"  Timestamps duplicados: {duplicates:,}")
    
    return ohlc_issues

def identify_rl_convergence_issues(df, returns, ohlc_issues):
    """Identifica problemas específicos que podem impedir convergência de RL"""
    print("\n" + "="*80)
    print("9. PROBLEMAS ESPECÍFICOS PARA CONVERGÊNCIA DE RL")
    print("="*80)
    
    issues = []
    
    # 1. Falta de features técnicas
    tech_indicators = [col for col in df.columns if any(ind in col.lower() 
                      for ind in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'momentum'])]
    if len(tech_indicators) < 3:
        issues.append("CRÍTICO: Poucos indicadores técnicos - agente RL precisa de features ricas")
    
    # 2. Distribuição de returns problemática
    returns_clean = returns.dropna()
    if abs(stats.skew(returns_clean)) > 2:
        issues.append(f"PROBLEMA: Returns muito assimétricos (skew={stats.skew(returns_clean):.3f})")
    
    if stats.kurtosis(returns_clean) > 10:
        issues.append(f"PROBLEMA: Returns com curtose extrema (kurt={stats.kurtosis(returns_clean):.3f})")
    
    # 3. Volatilidade extrema
    vol_rolling = returns_clean.rolling(window=100).std()
    vol_changes = vol_rolling.pct_change().abs()
    extreme_vol_changes = (vol_changes > 2).sum()  # Mudanças de volatilidade > 200%
    if extreme_vol_changes > len(returns_clean) * 0.01:  # > 1% das observações
        issues.append(f"PROBLEMA: Volatilidade muito instável ({extreme_vol_changes:,} mudanças extremas)")
    
    # 4. Returns zerados excessivos
    zero_returns_pct = (returns_clean == 0).sum() / len(returns_clean) * 100
    if zero_returns_pct > 5:
        issues.append(f"PROBLEMA: Muitos returns zerados ({zero_returns_pct:.2f}%)")
    
    # 5. Autocorrelação forte (não random walk)
    autocorr_1 = returns_clean.autocorr(lag=1)
    if abs(autocorr_1) > 0.1:
        issues.append(f"ALERTA: Autocorrelação forte lag-1 ({autocorr_1:.4f}) - não é random walk")
    
    # 6. Dados OHLC inconsistentes
    if ohlc_issues > 0:
        issues.append(f"CRÍTICO: {ohlc_issues:,} inconsistências em dados OHLC")
    
    # 7. Regime único
    if 'regime' in df.columns:
        regime_counts = df['regime'].value_counts()
        if len(regime_counts) == 1:
            issues.append(f"PROBLEMA: Apenas um regime ({regime_counts.index[0]}) - falta diversidade")
        elif regime_counts.min() / regime_counts.max() < 0.1:
            issues.append(f"PROBLEMA: Distribuição de regimes muito desbalanceada")
    
    # 8. Falta de variabilidade nos preços
    price_range = (df['high'].max() - df['low'].min()) / df['close'].mean() * 100
    if price_range < 5:  # Menos de 5% de range total
        issues.append(f"PROBLEMA: Range de preços muito pequeno ({price_range:.2f}%)")
    
    # 9. Volume patterns
    if (df['volume'] == df['volume'].iloc[0]).sum() / len(df) > 0.5:
        issues.append("PROBLEMA: Volume muito constante - pode ser sintético de forma inadequada")
    
    # 10. Timeframe issues
    time_diffs = df.index.to_series().diff()[1:]
    expected_diff = pd.Timedelta(minutes=5)
    irregular_intervals = (time_diffs != expected_diff).sum()
    if irregular_intervals > len(df) * 0.01:  # > 1%
        issues.append(f"PROBLEMA: {irregular_intervals:,} intervalos irregulares no timeframe")
    
    # Resumo
    print(f"TOTAL DE PROBLEMAS IDENTIFICADOS: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"{i:2d}. {issue}")
    
    if len(issues) == 0:
        print("✅ Nenhum problema crítico identificado para RL")
    elif len(issues) <= 3:
        print("⚠️  Alguns problemas identificados - podem afetar convergência")
    else:
        print("🚨 MUITOS PROBLEMAS - convergência de RL provavelmente comprometida")
    
    return issues

def main():
    """Função principal de análise"""
    filepath = r"D:\Projeto\data\GOLD_TRADING_READY_2M_20250803_222334.csv"
    
    print("ANÁLISE COMPLETA DO DATASET PARA IDENTIFICAÇÃO DE PROBLEMAS DE CONVERGÊNCIA DE RL")
    print("="*90)
    
    try:
        # 1. Carregar e examinar estrutura
        df = load_and_examine_dataset(filepath)
        
        # 2. Estatísticas descritivas
        df = calculate_descriptive_stats(df)
        
        # 3. Análise de distribuição
        returns = analyze_returns_distribution(df)
        
        # 4. Autocorrelação
        analyze_autocorrelation(df)
        
        # 5. Estacionariedade
        test_stationarity(df)
        
        # 6. Indicadores técnicos
        analyze_technical_indicators(df)
        
        # 7. Padrões temporais
        analyze_temporal_patterns(df)
        
        # 8. Qualidade dos dados
        ohlc_issues = check_data_quality(df)
        
        # 9. Problemas específicos de RL
        issues = identify_rl_convergence_issues(df, returns, ohlc_issues)
        
        print("\n" + "="*90)
        print("ANÁLISE CONCLUÍDA")
        print("="*90)
        print(f"Dataset analisado: {len(df):,} observações")
        print(f"Problemas identificados: {len(issues)}")
        
        return df, issues
        
    except Exception as e:
        print(f"ERRO na análise: {e}")
        import traceback
        traceback.print_exc()
        return None, []

if __name__ == "__main__":
    df, issues = main()