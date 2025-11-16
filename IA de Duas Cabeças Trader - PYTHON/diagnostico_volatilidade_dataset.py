#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE VOLATILIDADE DO DATASET
Analisar se o problema é simplesmente falta de volatilidade
"""

import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict

# Adicionar paths
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

def diagnose_dataset_volatility():
    """Diagnosticar volatilidade do dataset"""
    print("DIAGNOSTICO DE VOLATILIDADE DO DATASET")
    print("="*50)
    
    # Tentar carregar o dataset real
    data_files = [
        "data/GC=F_YAHOO_DAILY_5MIN_20250711_041924.csv",
        "data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv", 
        "data/GOLD_5m_20250704_122438.csv",
        "data/GOLD_5m_20250513_125132.csv"
    ]
    
    dataset_found = None
    for data_file in data_files:
        if os.path.exists(data_file):
            dataset_found = data_file
            break
    
    if dataset_found:
        print(f"DATASET ENCONTRADO: {dataset_found}")
        df = pd.read_csv(dataset_found)
        print(f"TAMANHO: {len(df)} barras")
        
        # Assumir que tem coluna 'close' ou similar
        price_col = None
        for col in ['close', 'Close', 'close_5m', 'CLOSE']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            prices = df[price_col].values
            print(f"COLUNA DE PRECO: {price_col}")
        else:
            print("COLUNAS DISPONÍVEIS:", list(df.columns))
            return
    else:
        print("DATASET NAO ENCONTRADO - SIMULANDO DADOS")
        # Simular diferentes tipos de mercado
        dates = pd.date_range('2023-01-01', periods=5000, freq='5min')
        
        # 80% baixa volatilidade, 20% alta volatilidade (como mercado real)
        low_vol_periods = int(0.8 * len(dates))
        high_vol_periods = len(dates) - low_vol_periods
        
        # Mercado de baixa volatilidade (problema!)
        low_vol_changes = np.random.normal(0, 0.003, low_vol_periods)  # 0.3% std
        # Mercado de alta volatilidade 
        high_vol_changes = np.random.normal(0, 0.015, high_vol_periods)  # 1.5% std
        
        # Combinar (maioria baixa volatilidade)
        price_changes = np.concatenate([low_vol_changes, high_vol_changes])
        np.random.shuffle(price_changes)
        
        base_price = 2000
        prices = [base_price]
        for change in price_changes:
            prices.append(prices[-1] * (1 + change))
        prices = np.array(prices[1:])
    
    # ANÁLISE DE VOLATILIDADE
    print(f"\nANALISE DE VOLATILIDADE:")
    print("-" * 50)
    
    # Calcular retornos
    returns = np.diff(prices) / prices[:-1]
    
    # Estatísticas básicas
    volatilidade_total = np.std(returns) * 100
    volatilidade_media_movel = pd.Series(returns).rolling(50).std().mean() * 100
    
    print(f"Volatilidade total: {volatilidade_total:.3f}%")
    print(f"Volatilidade média móvel: {volatilidade_media_movel:.3f}%")
    
    # Percentis de volatilidade
    vol_percentiles = np.percentile(np.abs(returns) * 100, [25, 50, 75, 90, 95, 99])
    print(f"\nPERCENTIS DE VOLATILIDADE:")
    print(f"  25%: {vol_percentiles[0]:.3f}%")
    print(f"  50%: {vol_percentiles[1]:.3f}%") 
    print(f"  75%: {vol_percentiles[2]:.3f}%")
    print(f"  90%: {vol_percentiles[3]:.3f}%")
    print(f"  95%: {vol_percentiles[4]:.3f}%")
    print(f"  99%: {vol_percentiles[5]:.3f}%")
    
    # Contar períodos de baixa/alta volatilidade
    threshold_baixa = 0.005  # 0.5%
    threshold_alta = 0.015   # 1.5%
    
    baixa_vol = np.sum(np.abs(returns) < threshold_baixa)
    media_vol = np.sum((np.abs(returns) >= threshold_baixa) & (np.abs(returns) < threshold_alta))
    alta_vol = np.sum(np.abs(returns) >= threshold_alta)
    
    total_periods = len(returns)
    
    print(f"\nDISTRIBUICAO DE VOLATILIDADE:")
    print(f"  Baixa vol (<0.5%): {baixa_vol:4d} ({baixa_vol/total_periods*100:.1f}%)")
    print(f"  Media vol (0.5-1.5%): {media_vol:4d} ({media_vol/total_periods*100:.1f}%)")
    print(f"  Alta vol (>1.5%): {alta_vol:4d} ({alta_vol/total_periods*100:.1f}%)")
    
    # PROBLEMA IDENTIFICADO?
    problema_volatilidade = baixa_vol / total_periods > 0.7  # >70% baixa volatilidade
    
    print(f"\nDIAGNOSTICO:")
    print("-" * 50)
    
    if problema_volatilidade:
        print("PROBLEMA IDENTIFICADO!")
        print(f"  {baixa_vol/total_periods*100:.1f}% do dataset tem BAIXA volatilidade")
        print("  V7 Intuition não consegue identificar oportunidades")
        print("  Composite scores ficam sempre baixos")
        print("  Max positions é atingido nas poucas oportunidades")
        
        print(f"\nIMPLICACAO PARA V7:")
        print("  - Modelo 'super treinado' aprendeu a detectar micro oportunidades")
        print("  - Em dados de baixa volatilidade, ainda consegue gerar scores altos")
        print("  - Threshold 0.75 pode não ser suficiente!")
        print("  - Solução: Threshold 0.9+ ou filtro de volatilidade mínima")
        
    else:
        print("Dataset com volatilidade adequada")
        print("Problema deve estar em outro lugar")
    
    # SIMULAÇÃO DE SOLUÇÃO
    print(f"\nTESTE DE SOLUÇÕES:")
    print("-" * 50)
    
    if problema_volatilidade:
        # Simular filtro de volatilidade
        vol_window = 20
        rolling_vol = pd.Series(returns).rolling(vol_window).std()
        vol_threshold = 0.008  # 0.8% volatilidade mínima
        
        high_vol_periods = np.sum(rolling_vol > vol_threshold)
        vol_filtered_ratio = high_vol_periods / len(rolling_vol)
        
        print(f"FILTRO DE VOLATILIDADE MINIMA (0.8%):")
        print(f"  Períodos válidos: {high_vol_periods}/{len(rolling_vol)}")
        print(f"  Taxa de aprovação: {vol_filtered_ratio*100:.1f}%")
        print(f"  Estimativa de trades: ~{vol_filtered_ratio * 18:.1f}/dia")
        
        if vol_filtered_ratio < 0.3:
            print("  ⚠️ Filtro muito restritivo - poucos trades")
        elif vol_filtered_ratio > 0.7:
            print("  ⚠️ Filtro insuficiente - ainda muitos períodos")
        else:
            print("  ✅ Filtro balanceado - boa taxa de trades")

if __name__ == "__main__":
    diagnose_dataset_volatility()