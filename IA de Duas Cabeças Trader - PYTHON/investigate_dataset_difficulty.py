#!/usr/bin/env python3
"""
🔍 INVESTIGADOR DE DIFICULDADE DO DATASET V2
Analisa se o dataset está artificialmente fácil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def investigate_dataset_difficulty():
    """Investigar dificuldade real do dataset V2"""
    print("🔍 INVESTIGAÇÃO: DATASET V2 MUITO FÁCIL?")
    print("="*60)
    
    dataset_path = "data/GC=F_HYBRID_V2_3Y_1MIN_20250911_200306.csv"
    
    print(f"📂 Carregando dataset: {len(dataset_path)} chars")
    df = pd.read_csv(dataset_path, nrows=100000)  # Primeiras 100k para velocidade
    print(f"✅ Carregado: {len(df):,} linhas")
    
    # 1. ANÁLISE DE PADRÕES PREVISÍVEIS
    print(f"\n🎯 TESTE 1: PADRÕES PREVISÍVEIS")
    print("-"*40)
    
    # Calcular retornos
    df['returns'] = df['close'].pct_change().fillna(0)
    
    # Sequências de mesmo sinal (tendência)
    df['return_sign'] = np.sign(df['returns'])
    
    # Contar sequências longas de mesmo sinal
    sequences = []
    current_sign = 0
    current_length = 0
    
    for sign in df['return_sign']:
        if sign == current_sign and sign != 0:
            current_length += 1
        else:
            if current_length > 5:  # Sequências > 5 são suspeitas
                sequences.append(current_length)
            current_sign = sign
            current_length = 1
    
    if sequences:
        avg_sequence = np.mean(sequences)
        max_sequence = max(sequences)
        print(f"⚠️ SUSPEITO: {len(sequences)} sequências longas (>5)")
        print(f"   Sequência média: {avg_sequence:.1f} barras")
        print(f"   Sequência máxima: {max_sequence} barras")
        if avg_sequence > 15:
            print(f"❌ MUITO PREVISÍVEL: Sequências muito longas")
    else:
        print(f"✅ OK: Sem sequências suspeitas")
    
    # 2. ANÁLISE DE VOLATILIDADE ARTIFICIAL
    print(f"\n📊 TESTE 2: VOLATILIDADE REALÍSTICA")
    print("-"*40)
    
    # Volatilidade rolling
    df['vol_60'] = df['returns'].rolling(60).std() * 100
    
    vol_stats = df['vol_60'].dropna().describe()
    print(f"Volatilidade 60min:")
    print(f"   Média: {vol_stats['mean']:.4f}%")
    print(f"   Std: {vol_stats['std']:.4f}%")
    print(f"   Min: {vol_stats['min']:.4f}%")
    print(f"   Max: {vol_stats['max']:.4f}%")
    
    # Volatilidade muito baixa é suspeita
    if vol_stats['mean'] < 0.01:
        print(f"❌ SUSPEITO: Volatilidade muito baixa ({vol_stats['mean']:.4f}%)")
    elif vol_stats['std'] < 0.005:
        print(f"❌ SUSPEITO: Volatilidade muito constante (std: {vol_stats['std']:.4f}%)")
    else:
        print(f"✅ VOLATILIDADE OK")
    
    # 3. TESTE DE AUTOCORRELAÇÃO (Padrões repetitivos)
    print(f"\n🔄 TESTE 3: AUTOCORRELAÇÃO (PADRÕES CÍCLICOS)")
    print("-"*40)
    
    # Autocorrelação dos retornos
    returns_clean = df['returns'].dropna()
    
    autocorrs = []
    for lag in [1, 5, 10, 30, 60]:
        if len(returns_clean) > lag:
            corr = returns_clean.corr(returns_clean.shift(lag))
            autocorrs.append((lag, corr))
            print(f"   Lag {lag:2d}: {corr:.4f}")
    
    # Autocorrelação alta = padrões previsíveis
    high_corrs = [corr for lag, corr in autocorrs if abs(corr) > 0.1]
    if high_corrs:
        print(f"❌ SUSPEITO: {len(high_corrs)} autocorrelações altas (>0.1)")
    else:
        print(f"✅ OK: Autocorrelações baixas")
    
    # 4. TESTE DE GAPS E SALTOS IRREAIS
    print(f"\n⚡ TESTE 4: GAPS E SALTOS IRREAIS")
    print("-"*40)
    
    # Calcular mudanças percentuais extremas
    abs_returns = np.abs(df['returns'])
    extreme_moves = abs_returns[abs_returns > 0.01]  # >1%
    
    print(f"Movimentos >1%: {len(extreme_moves)} ({len(extreme_moves)/len(df)*100:.2f}%)")
    
    if len(extreme_moves) == 0:
        print(f"❌ SUSPEITO: Nenhum movimento >1% em {len(df):,} barras")
    elif len(extreme_moves) < len(df) * 0.001:  # <0.1%
        print(f"❌ SUSPEITO: Pouquíssimos movimentos grandes ({len(extreme_moves)/len(df)*100:.3f}%)")
    else:
        print(f"✅ OK: Movimentos grandes presentes")
    
    # 5. TESTE DE SPREAD BID-ASK (Realismo)
    print(f"\n💰 TESTE 5: SPREAD E REALISMO DE TRADING")
    print("-"*40)
    
    # Verificar se spread existe e é realista
    if 'spread' in df.columns:
        spread_stats = df['spread'].describe()
        print(f"Spread médio: {spread_stats['mean']:.4f}")
        
        if spread_stats['mean'] == 0:
            print(f"❌ IRREAL: Spread sempre 0 (sem custos de trading)")
        else:
            print(f"✅ OK: Spread presente")
    
    # 6. PADRÕES HORÁRIOS ARTIFICIAIS
    print(f"\n⏰ TESTE 6: PADRÕES HORÁRIOS ARTIFICIAIS")
    print("-"*40)
    
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    
    hourly_vol = df.groupby('hour')['returns'].std()
    vol_range = hourly_vol.max() - hourly_vol.min()
    
    print(f"Range volatilidade horária: {vol_range:.6f}")
    
    if vol_range < 0.0001:
        print(f"❌ SUSPEITO: Volatilidade muito uniforme por hora")
    else:
        print(f"✅ OK: Variação horária presente")
    
    # 7. TESTE DE REGIME ÚNICO (Falta de bear markets)
    print(f"\n📈 TESTE 7: REGIMES DE MERCADO")
    print("-"*40)
    
    # Tendências de longo prazo
    df['ma_200'] = df['close'].rolling(200).mean()
    df['above_ma200'] = df['close'] > df['ma_200']
    
    pct_above_ma200 = df['above_ma200'].mean() * 100
    print(f"% tempo acima MA200: {pct_above_ma200:.1f}%")
    
    if pct_above_ma200 > 80:
        print(f"❌ SUSPEITO: Sempre bull market ({pct_above_ma200:.1f}%)")
    elif pct_above_ma200 < 20:
        print(f"❌ SUSPEITO: Sempre bear market ({pct_above_ma200:.1f}%)")
    else:
        print(f"✅ OK: Regimes variados")
    
    # RESUMO FINAL
    print(f"\n" + "="*60)
    print(f"🎯 RESUMO DA INVESTIGAÇÃO:")
    print(f"="*60)
    
    issues = []
    if sequences and avg_sequence > 15:
        issues.append(f"Sequências previsíveis (média {avg_sequence:.1f})")
    if vol_stats['mean'] < 0.01:
        issues.append(f"Volatilidade baixa ({vol_stats['mean']:.4f}%)")
    if high_corrs:
        issues.append(f"{len(high_corrs)} autocorrelações altas")
    if len(extreme_moves) < len(df) * 0.001:
        issues.append("Poucos movimentos extremos")
    if 'spread' in df.columns and df['spread'].mean() == 0:
        issues.append("Spread zero (irreal)")
    if vol_range < 0.0001:
        issues.append("Volatilidade horária uniforme")
    if pct_above_ma200 > 80 or pct_above_ma200 < 20:
        issues.append(f"Regime único ({pct_above_ma200:.1f}% acima MA200)")
    
    if not issues:
        print(f"🎉 DATASET REALISTA: Nenhuma irregularidade encontrada")
        return "REALISTIC"
    elif len(issues) <= 2:
        print(f"⚠️ DATASET SUSPEITO: {len(issues)} problemas encontrados")
        for issue in issues:
            print(f"   - {issue}")
        return "SUSPICIOUS"
    else:
        print(f"❌ DATASET ARTIFICIAL: {len(issues)} problemas sérios")
        for issue in issues:
            print(f"   - {issue}")
        print(f"\n🔥 CONCLUSÃO: Dataset V2 está FACILITANDO o treinamento!")
        print(f"   Win rate 92% é explicado por dataset previsível/artificial")
        return "ARTIFICIAL"

if __name__ == "__main__":
    result = investigate_dataset_difficulty()
    print(f"\n🏁 VEREDICTO FINAL: {result}")