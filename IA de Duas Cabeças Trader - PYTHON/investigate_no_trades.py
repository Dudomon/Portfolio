#!/usr/bin/env python3
"""
Investigação - Por que muitos episódios sem trades após aumento de volatilidade
Analisa o dataset V2 e sistema de recompensas para identificar problema
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_v2_volatility():
    """Analisa as características do dataset V2"""
    
    print("🔍 INVESTIGAÇÃO - EPISÓDIOS SEM TRADES")
    print("=" * 50)
    
    # Carregar dataset V2
    df_v2 = pd.read_csv('data/GC_YAHOO_ENHANCED_V2_TARGET_20250804_184852.csv')
    df_v2['time'] = pd.to_datetime(df_v2['time'])
    
    # Carregar dataset original para comparação
    df_orig = pd.read_csv('data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv')
    df_orig['time'] = pd.to_datetime(df_orig['time'])
    
    print(f"📊 DATASETS CARREGADOS:")
    print(f"   Original: {len(df_orig):,} barras")
    print(f"   V2: {len(df_v2):,} barras")
    
    # Analisar volatilidade
    returns_orig = df_orig['close'].pct_change()
    returns_v2 = df_v2['close'].pct_change()
    
    vol_orig = returns_orig.std()
    vol_v2 = returns_v2.std()
    
    print(f"\n📈 VOLATILIDADE COMPARADA:")
    print(f"   Original: {vol_orig*100:.3f}%")
    print(f"   V2: {vol_v2*100:.3f}%")
    print(f"   Multiplicador real: {vol_v2/vol_orig:.2f}x")
    
    # Analisar distribuição de returns
    print(f"\n📊 DISTRIBUIÇÃO DE RETURNS:")
    
    # Percentis para comparação
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    print(f"   {'Percentil':<10} {'Original':<12} {'V2':<12} {'Ratio':<8}")
    print(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    
    for p in percentiles:
        orig_val = returns_orig.quantile(p/100)
        v2_val = returns_v2.quantile(p/100)
        ratio = abs(v2_val / orig_val) if orig_val != 0 else float('inf')
        
        print(f"   P{p:2d}        {orig_val*100:+8.3f}%    {v2_val*100:+8.3f}%    {ratio:6.1f}x")
    
    # Analisar movimentos extremos
    large_moves_orig = abs(returns_orig) > returns_orig.quantile(0.95)
    large_moves_v2 = abs(returns_v2) > returns_v2.quantile(0.95)
    
    print(f"\n🚀 MOVIMENTOS EXTREMOS (Top 5%):")
    print(f"   Original: {large_moves_orig.sum():,} barras ({large_moves_orig.mean()*100:.2f}%)")
    print(f"   V2: {large_moves_v2.sum():,} barras ({large_moves_v2.mean()*100:.2f}%)")
    
    # Threshold para trading
    orig_95 = abs(returns_orig).quantile(0.95)
    v2_95 = abs(returns_v2).quantile(0.95)
    
    print(f"   Threshold P95 original: {orig_95*100:.3f}%")
    print(f"   Threshold P95 V2: {v2_95*100:.3f}%")
    print(f"   Ratio: {v2_95/orig_95:.1f}x")
    
    return {
        'vol_orig': vol_orig,
        'vol_v2': vol_v2,
        'threshold_orig': orig_95,
        'threshold_v2': v2_95,
        'large_moves_orig': large_moves_orig.sum(),
        'large_moves_v2': large_moves_v2.sum()
    }

def analyze_potential_trading_issues():
    """Analisa possíveis problemas que impedem trading"""
    
    print(f"\n🎯 ANÁLISE DE PROBLEMAS POTENCIAIS:")
    print("-" * 40)
    
    # 1. Problema de scaling nos indicadores
    print(f"1. 📊 SCALING DE INDICADORES:")
    
    df_v2 = pd.read_csv('data/GC_YAHOO_ENHANCED_V2_TARGET_20250804_184852.csv')
    
    # Verificar se indicadores acompanharam o scaling
    indicators = ['rsi_14', 'bb_position', 'volatility_20', 'atr_14']
    
    for indicator in indicators:
        if indicator in df_v2.columns:
            values = df_v2[indicator].dropna()
            print(f"   {indicator}: range [{values.min():.3f}, {values.max():.3f}]")
            
            # Verificar valores anômalos
            if indicator == 'rsi_14':
                anomalous = (values < 0) | (values > 100)
                print(f"     RSI fora de [0,100]: {anomalous.sum()} barras")
            elif indicator == 'bb_position':
                anomalous = (values < -0.5) | (values > 1.5)
                print(f"     BB_position anômalo: {anomalous.sum()} barras")
    
    # 2. Problema de normalização extrema
    print(f"\n2. 🔢 NORMALIZAÇÃO EXTREMA:")
    
    prices = df_v2['close']
    price_range = prices.max() - prices.min()
    price_std = prices.std()
    
    print(f"   Preços: [{prices.min():.0f}, {prices.max():.0f}]")
    print(f"   Range: {price_range:.0f}")
    print(f"   Std: {price_std:.0f}")
    print(f"   Coef. Variação: {price_std/prices.mean()*100:.1f}%")
    
    # Se coef. variação muito alto, pode quebrar normalização
    if price_std/prices.mean() > 0.5:
        print(f"   ⚠️ ALERTA: Coeficiente de variação muito alto!")
        print(f"   Isso pode quebrar a normalização do modelo")
    
    # 3. Problema de action space scaling
    print(f"\n3. 🎮 ACTION SPACE SCALING:")
    
    # Simular como o modelo "vê" os dados
    returns = df_v2['close'].pct_change().dropna()
    
    # Movimento típico que o modelo precisa detectar
    typical_move = returns.std()
    large_move = abs(returns).quantile(0.95)
    extreme_move = abs(returns).quantile(0.99)
    
    print(f"   Movimento típico: {typical_move*100:.3f}%")
    print(f"   Movimento grande: {large_move*100:.3f}%")
    print(f"   Movimento extremo: {extreme_move*100:.3f}%")
    
    # Se movimentos são muito grandes, modelo pode ficar "assustado"
    if typical_move > 0.01:  # 1%
        print(f"   ⚠️ ALERTA: Movimentos típicos muito grandes!")
        print(f"   Modelo pode estar evitando trades por medo")
    
    return {
        'price_volatility': price_std/prices.mean(),
        'typical_move': typical_move,
        'large_move': large_move
    }

def check_reward_system_compatibility():
    """Verifica se sistema de rewards está compatível com alta volatilidade"""
    
    print(f"\n💰 SISTEMA DE REWARDS E ALTA VOLATILIDADE:")
    print("-" * 40)
    
    df_v2 = pd.read_csv('data/GC_YAHOO_ENHANCED_V2_TARGET_20250804_184852.csv')
    returns = df_v2['close'].pct_change().dropna()
    
    # Simular alguns cenários de reward
    # Assumindo SL/TP típicos do sistema
    sl_points = 5.0  # 5 pontos de SL típico
    tp_points = 10.0  # 10 pontos de TP típico
    
    close_prices = df_v2['close']
    avg_price = close_prices.mean()
    
    sl_percentage = sl_points / avg_price
    tp_percentage = tp_points / avg_price
    
    print(f"   Preço médio: ${avg_price:.0f}")
    print(f"   SL típico: {sl_points} pontos = {sl_percentage*100:.3f}%")
    print(f"   TP típico: {tp_points} pontos = {tp_percentage*100:.3f}%")
    
    # Comparar com volatilidade atual
    typical_move = returns.std()
    print(f"   Movimento típico: {typical_move*100:.3f}%")
    
    # Ratio crítico
    sl_ratio = typical_move / sl_percentage
    tp_ratio = typical_move / tp_percentage
    
    print(f"\n   🎯 RATIOS CRÍTICOS:")
    print(f"   SL ratio: {sl_ratio:.2f} (movimento/SL)")
    print(f"   TP ratio: {tp_ratio:.2f} (movimento/TP)")
    
    if sl_ratio > 0.5:
        print(f"   ⚠️ ALERTA: Movimento típico é {sl_ratio:.1f}x do SL!")
        print(f"   SL sendo atingido muito facilmente")
    
    if tp_ratio > 0.3:
        print(f"   ⚠️ ALERTA: Movimento típico é {tp_ratio:.1f}x do TP!")
        print(f"   TP também sendo atingido facilmente")
    
    # Recomendações
    print(f"\n   💡 RECOMENDAÇÕES:")
    if sl_ratio > 0.5 or tp_ratio > 0.3:
        print(f"   1. Aumentar SL/TP proporcionalmente à volatilidade")
        print(f"   2. Ajustar reward scaling para alta volatilidade")
        print(f"   3. Modificar thresholds de entrada")
    
    return {
        'sl_ratio': sl_ratio,
        'tp_ratio': tp_ratio,
        'needs_adjustment': sl_ratio > 0.5 or tp_ratio > 0.3
    }

def generate_recommendations():
    """Gera recomendações baseadas na análise"""
    
    print(f"\n🚀 RECOMENDAÇÕES PARA RESOLVER 'SEM TRADES':")
    print("=" * 50)
    
    print(f"1. 🎯 AJUSTAR THRESHOLDS DE ENTRADA:")
    print(f"   - Aumentar confidence threshold para evitar trades em ruído")
    print(f"   - Calibrar entry signals para alta volatilidade")
    
    print(f"\n2. 📊 AJUSTAR SL/TP DINÂMICOS:")
    print(f"   - SL: Multiplicar por fator de volatilidade (~2-3x)")
    print(f"   - TP: Idem, manter ratio risk/reward")
    
    print(f"\n3. 🔧 REVISAR NORMALIZAÇÃO:")
    print(f"   - Verificar se features estão em ranges apropriados")
    print(f"   - Aplicar robust scaling se necessário")
    
    print(f"\n4. 💰 AJUSTAR SISTEMA DE REWARDS:")
    print(f"   - Reduzir penalidades por volatilidade")
    print(f"   - Aumentar rewards por trades bem-sucedidos")
    
    print(f"\n5. 🎮 REBALANCEAR ACTION SPACE:")
    print(f"   - Verificar se confidence/risk_appetite adequados")
    print(f"   - Ajustar ranges de SL/TP actions")

if __name__ == "__main__":
    try:
        # Análise do dataset
        dataset_analysis = analyze_dataset_v2_volatility()
        
        # Análise de problemas potenciais
        trading_issues = analyze_potential_trading_issues()
        
        # Análise do sistema de rewards
        reward_analysis = check_reward_system_compatibility()
        
        # Gerar recomendações
        generate_recommendations()
        
        print(f"\n📋 RESUMO DA INVESTIGAÇÃO:")
        print(f"   Volatilidade V2: {dataset_analysis['vol_v2']/dataset_analysis['vol_orig']:.1f}x original")
        print(f"   Price volatility: {trading_issues['price_volatility']*100:.1f}%")
        print(f"   SL ratio crítico: {reward_analysis['sl_ratio']:.2f}")
        print(f"   Ajustes necessários: {'SIM' if reward_analysis['needs_adjustment'] else 'NÃO'}")
        
    except Exception as e:
        print(f"❌ Erro na investigação: {e}")
        import traceback
        traceback.print_exc()