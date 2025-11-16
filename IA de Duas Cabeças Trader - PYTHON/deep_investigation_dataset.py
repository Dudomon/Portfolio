#!/usr/bin/env python3
"""
🔬 INVESTIGAÇÃO PROFUNDA DO DATASET V2
Análise extremamente detalhada dos problemas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta

def deep_investigation():
    """Investigação MUITO MAIS PROFUNDA do dataset V4"""
    print("🔬 INVESTIGAÇÃO PROFUNDA - DATASET V4")
    print("="*70)
    
    dataset_path = "data/GC=F_REALISTIC_V4_20250911_235945.csv"
    
    # Carregar dataset completo para análise profunda
    print(f"📂 Carregando dataset COMPLETO...")
    df = pd.read_csv(dataset_path, nrows=500000)  # 500k linhas para análise profunda
    print(f"✅ Carregado: {len(df):,} linhas para análise profunda")
    
    df['returns'] = df['close'].pct_change().fillna(0)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    
    # 1. ANÁLISE MICROSCÓPICA DA VOLATILIDADE
    print(f"\n🔬 ANÁLISE 1: VOLATILIDADE MICROSCÓPICA")
    print("-"*50)
    
    # Volatilidade em múltiplos timeframes
    for window in [10, 30, 60, 240, 1440]:  # 10min a 24h
        df[f'vol_{window}'] = df['returns'].rolling(window).std() * 100
        vol_data = df[f'vol_{window}'].dropna()
        
        print(f"\nVolatilidade {window}min:")
        print(f"   Média: {vol_data.mean():.6f}%")
        print(f"   Mediana: {vol_data.median():.6f}%")
        print(f"   P95: {vol_data.quantile(0.95):.6f}%")
        print(f"   P99: {vol_data.quantile(0.99):.6f}%")
        print(f"   Zeros: {(vol_data == 0).sum():,} ({(vol_data == 0).mean()*100:.2f}%)")
        
        # PROBLEMA: Muitos zeros de volatilidade
        if (vol_data == 0).mean() > 0.1:  # >10% zeros
            print(f"   ❌ PROBLEMA: {(vol_data == 0).mean()*100:.1f}% com volatilidade ZERO")
    
    # 2. ANÁLISE DE MICROESTRUTURA DOS PREÇOS
    print(f"\n🔬 ANÁLISE 2: MICROESTRUTURA DOS PREÇOS")
    print("-"*50)
    
    # Verificar se preços seguem tick size correto
    price_diffs = df['close'].diff().dropna()
    unique_diffs = price_diffs[price_diffs != 0].unique()
    
    print(f"Diferenças de preço únicas (primeiras 20):")
    print(f"   {sorted(unique_diffs[:20])}")
    
    # Verificar se há tick size consistente
    min_positive_diff = price_diffs[price_diffs > 0].min()
    print(f"   Menor movimento positivo: {min_positive_diff:.6f}")
    
    # Contar movimentos por múltiplos de 0.01 (tick normal do ouro)
    tick_multiples = price_diffs[price_diffs != 0] / 0.01
    clean_multiples = tick_multiples[np.abs(tick_multiples - np.round(tick_multiples)) < 1e-10]
    
    print(f"   Movimentos em múltiplos de 0.01: {len(clean_multiples)}/{len(price_diffs[price_diffs != 0])} ({len(clean_multiples)/len(price_diffs[price_diffs != 0])*100:.1f}%)")
    
    # 3. ANÁLISE DE SEQUÊNCIAS E PADRÕES TEMPORAIS
    print(f"\n🔬 ANÁLISE 3: SEQUÊNCIAS E PADRÕES TEMPORAIS")
    print("-"*50)
    
    # Analisar sequências estáticas (mesmo preço)
    static_sequences = []
    current_price = None
    current_length = 0
    
    for price in df['close']:
        if price == current_price:
            current_length += 1
        else:
            if current_length > 1:
                static_sequences.append(current_length)
            current_price = price
            current_length = 1
    
    if current_length > 1:
        static_sequences.append(current_length)
    
    static_sequences = np.array(static_sequences)
    
    print(f"Sequências estáticas (mesmo preço):")
    print(f"   Total: {len(static_sequences):,}")
    print(f"   Média: {static_sequences.mean():.1f} barras")
    print(f"   Mediana: {np.median(static_sequences):.1f} barras")
    print(f"   P95: {np.percentile(static_sequences, 95):.1f} barras")
    print(f"   Máxima: {static_sequences.max()} barras")
    
    # PROBLEMA: Sequências muito longas
    long_sequences = static_sequences[static_sequences > 100]
    if len(long_sequences) > 0:
        print(f"   ❌ PROBLEMA: {len(long_sequences)} sequências >100 barras")
        print(f"      Sequências ultra-longas: {long_sequences[long_sequences > 500]}")
    
    # 4. ANÁLISE DE DISTRIBUIÇÃO DOS RETORNOS
    print(f"\n🔬 ANÁLISE 4: DISTRIBUIÇÃO DOS RETORNOS")
    print("-"*50)
    
    returns_clean = df['returns'][df['returns'] != 0]
    
    print(f"Estatísticas dos retornos (não-zeros):")
    print(f"   Total retornos: {len(df['returns']):,}")
    print(f"   Retornos zero: {(df['returns'] == 0).sum():,} ({(df['returns'] == 0).mean()*100:.1f}%)")
    print(f"   Retornos não-zero: {len(returns_clean):,}")
    print(f"   Média: {returns_clean.mean():.8f}")
    print(f"   Std: {returns_clean.std():.8f}")
    print(f"   Skew: {stats.skew(returns_clean):.4f}")
    print(f"   Kurtosis: {stats.kurtosis(returns_clean):.4f}")
    
    # Teste de normalidade
    stat, p_value = stats.jarque_bera(returns_clean)
    print(f"   Jarque-Bera p-value: {p_value:.8f}")
    if p_value > 0.05:
        print(f"   ✅ Distribuição aproximadamente normal")
    else:
        print(f"   ⚠️ Distribuição NÃO normal")
    
    # 5. ANÁLISE DE HORÁRIOS E PADRÕES INTRADAY
    print(f"\n🔬 ANÁLISE 5: PADRÕES INTRADAY DETALHADOS")
    print("-"*50)
    
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Volatilidade por hora
    hourly_stats = df.groupby('hour').agg({
        'returns': ['count', 'std', 'mean'],
        'vol_60': 'mean'
    }).round(8)
    
    print(f"Volatilidade por hora (primeiras 6 horas):")
    for hour in range(6):
        if hour in hourly_stats.index:
            vol_std = hourly_stats.loc[hour, ('returns', 'std')]
            vol_60 = hourly_stats.loc[hour, ('vol_60', 'mean')]
            count = hourly_stats.loc[hour, ('returns', 'count')]
            print(f"   {hour:02d}h: Vol={vol_std:.8f}, Vol60={vol_60:.8f}, N={count}")
    
    # Verificar se volatilidade varia realisticamente por hora
    hourly_vols = [hourly_stats.loc[h, ('returns', 'std')] for h in hourly_stats.index if not np.isnan(hourly_stats.loc[h, ('returns', 'std')])]
    vol_cv = np.std(hourly_vols) / np.mean(hourly_vols)  # Coeficiente de variação
    
    print(f"   Coeficiente variação horária: {vol_cv:.4f}")
    if vol_cv < 0.1:
        print(f"   ❌ PROBLEMA: Volatilidade muito uniforme por hora")
    
    # 6. ANÁLISE DE OUTLIERS E EVENTOS EXTREMOS
    print(f"\n🔬 ANÁLISE 6: OUTLIERS E EVENTOS EXTREMOS")
    print("-"*50)
    
    # Definir thresholds para ouro (mais rigorosos)
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%
    
    abs_returns = np.abs(df['returns'])
    
    for threshold in thresholds:
        count = (abs_returns > threshold).sum()
        pct = count / len(df) * 100
        expected_pct = stats.norm.sf(threshold/df['returns'].std()) * 200  # Bilateral
        
        print(f"   |Retorno| > {threshold*100:.1f}%: {count:,} ({pct:.3f}%) - Esperado: ~{expected_pct:.3f}%")
        
        if threshold == 0.01 and pct < 0.1:  # Muito poucos movimentos >1%
            print(f"      ❌ PROBLEMA: Muito poucos movimentos extremos")
    
    # 7. ANÁLISE DE MOMENTUM E REVERSÃO
    print(f"\n🔬 ANÁLISE 7: MOMENTUM E REVERSÃO")
    print("-"*50)
    
    # Analisar padrões de momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_30'] = df['close'].pct_change(30)
    
    # Correlação entre retornos consecutivos
    for lag in [1, 2, 5, 10]:
        corr = df['returns'].corr(df['returns'].shift(lag))
        print(f"   Correlação lag-{lag}: {corr:.6f}")
        
        if abs(corr) > 0.05:  # Correlação significativa
            if corr > 0:
                print(f"      ⚠️ Momentum detectado (correlação positiva)")
            else:
                print(f"      ⚠️ Reversão detectada (correlação negativa)")
    
    # 8. COMPARAÇÃO COM DADOS REAIS
    print(f"\n🔬 ANÁLISE 8: COMPARAÇÃO COM EXPECTATIVAS REAIS")
    print("-"*50)
    
    # Parâmetros esperados para ouro 1min
    real_gold_params = {
        'daily_vol_pct': 1.0,  # ~1% volatilidade diária
        'minute_vol_pct': 1.0 / np.sqrt(1440),  # ~0.026% por minuto
        'extreme_moves_per_day': 5,  # ~5 movimentos >0.1% por dia
        'static_ratio_max': 0.1,  # Máximo 10% de barras estáticas
    }
    
    # Nossa volatilidade por minuto
    our_minute_vol = df['returns'].std() * 100
    expected_minute_vol = real_gold_params['minute_vol_pct']
    
    print(f"Volatilidade por minuto:")
    print(f"   Nossa: {our_minute_vol:.6f}%")
    print(f"   Esperada: {expected_minute_vol:.6f}%")
    print(f"   Ratio: {our_minute_vol/expected_minute_vol:.2f}x")
    
    if our_minute_vol < expected_minute_vol * 0.5:
        print(f"   ❌ PROBLEMA: Volatilidade {our_minute_vol/expected_minute_vol:.2f}x menor que o esperado")
    
    # Taxa de barras estáticas
    static_ratio = (df['returns'] == 0).mean()
    print(f"Taxa de barras estáticas:")
    print(f"   Nossa: {static_ratio*100:.1f}%")
    print(f"   Máxima aceitável: {real_gold_params['static_ratio_max']*100:.1f}%")
    
    if static_ratio > real_gold_params['static_ratio_max']:
        print(f"   ❌ PROBLEMA: {static_ratio*100:.1f}% barras estáticas (muito alto)")
    
    # RESUMO EXECUTIVO DOS PROBLEMAS
    print(f"\n" + "="*70)
    print(f"🎯 RESUMO EXECUTIVO - PROBLEMAS IDENTIFICADOS")
    print(f"="*70)
    
    critical_issues = []
    moderate_issues = []
    
    # Avaliar cada problema
    if static_ratio > 0.2:  # >20% estáticas
        critical_issues.append(f"Barras estáticas: {static_ratio*100:.1f}% (crítico)")
    elif static_ratio > 0.1:
        moderate_issues.append(f"Barras estáticas: {static_ratio*100:.1f}% (alto)")
    
    if our_minute_vol < expected_minute_vol * 0.3:
        critical_issues.append(f"Volatilidade: {our_minute_vol/expected_minute_vol:.2f}x menor (crítico)")
    elif our_minute_vol < expected_minute_vol * 0.7:
        moderate_issues.append(f"Volatilidade: {our_minute_vol/expected_minute_vol:.2f}x menor")
    
    extreme_001_pct = (abs_returns > 0.001).mean() * 100
    if extreme_001_pct < 1.0:  # <1% de movimentos >0.1%
        critical_issues.append(f"Movimentos extremos: {extreme_001_pct:.2f}% >0.1% (muito baixo)")
    
    if len(long_sequences) > 100:
        critical_issues.append(f"Sequências longas: {len(long_sequences)} sequências >100 barras")
    
    if vol_cv < 0.05:
        moderate_issues.append(f"Volatilidade horária: CV={vol_cv:.4f} (muito uniforme)")
    
    # Veredito final
    print(f"\n🚨 PROBLEMAS CRÍTICOS ({len(critical_issues)}):")
    for issue in critical_issues:
        print(f"   ❌ {issue}")
    
    print(f"\n⚠️ PROBLEMAS MODERADOS ({len(moderate_issues)}):")
    for issue in moderate_issues:
        print(f"   ⚠️ {issue}")
    
    if len(critical_issues) >= 2:
        print(f"\n🔥 VEREDITO: DATASET ARTIFICIALMENTE FÁCIL")
        print(f"   Win rate 92% é EXPLICADO pelos problemas críticos")
        print(f"   Dataset precisa ser REFEITO com parâmetros realistas")
        return "CRITICAL"
    elif len(critical_issues) + len(moderate_issues) >= 3:
        print(f"\n⚠️ VEREDITO: DATASET SUSPEITO")
        print(f"   Performance pode estar inflada")
        return "SUSPICIOUS"  
    else:
        print(f"\n✅ VEREDITO: DATASET ACEITÁVEL")
        return "ACCEPTABLE"

if __name__ == "__main__":
    result = deep_investigation()
    print(f"\n🏁 RESULTADO FINAL: {result}")