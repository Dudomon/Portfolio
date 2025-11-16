#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO COMPOSITE SCORES
Analisara EXATAMENTE que scores o V7 está gerando
"""

import numpy as np
import pandas as pd
import sys
import os
import torch

# Adicionar paths
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

from daytrader import TradingEnv, TRADING_CONFIG, TRIAL_2_TRADING_PARAMS

def diagnose_composite_scores():
    """Investigar composite scores do V7"""
    print("🔍 DIAGNÓSTICO COMPOSITE SCORES V7")
    print("="*50)
    
    # Criar dados mock
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    base_price = 4000
    price_changes = np.random.normal(0, 0.01, 100)
    
    prices = [base_price]
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))
    prices = prices[1:]
    
    df = pd.DataFrame({
        'close_5m': prices,
        'high_5m': [p * 1.001 for p in prices],
        'low_5m': [p * 0.999 for p in prices],
        'volume_5m': [5000] * len(prices),
    }, index=dates)
    
    # Criar ambiente
    env = TradingEnv(df=df, window_size=20, is_training=True)
    obs = env.reset()
    
    print(f"📊 Dados: {len(df)} barras")
    
    # HACK: Acessar o modelo V7 diretamente
    # Vamos criar ações e ver o que acontece
    print(f"\n🎯 TESTANDO AÇÕES DIRETAMENTE")
    print("-" * 50)
    
    # Gerar diferentes tipos de ações
    test_actions = [
        ("Super Bullish", np.array([1.0, 0.95, 0.9, 0.8, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0])),
        ("Moderate Long", np.array([1.0, 0.6, 0.5, 0.5, 0.0, 2.0, 0.0, 0.0, 1.5, 0.0, 0.0])),
        ("Weak Long", np.array([1.0, 0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
        ("Hold", np.array([0.4, 0.5, 0.5, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Weak Short", np.array([0.0, 0.3, 0.0, 0.4, 0.6, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])),
        ("Super Bearish", np.array([0.0, 0.1, 0.0, 0.2, 0.95, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0])),
    ]
    
    scores_log = []
    
    for name, action in test_actions:
        positions_before = len(env.positions)
        current_price = df['close_5m'].iloc[env.current_step]
        
        print(f"\n🧪 TESTE: {name}")
        print(f"  Ação: {action[:5]}")
        print(f"  Posições antes: {positions_before}")
        
        # TENTAR INTERCEPTAR O COMPOSITE SCORE
        # Vamos usar um hook para capturar o que acontece dentro do step
        
        obs, reward, done, info = env.step(action)
        
        positions_after = len(env.positions)
        positions_created = positions_after - positions_before
        
        print(f"  Posições depois: {positions_after} ({positions_created:+d})")
        print(f"  Max positions: {len(env.positions) >= env.max_positions}")
        
        scores_log.append({
            'name': name,
            'action': action.copy(),
            'positions_before': positions_before,
            'positions_after': positions_after,
            'created': positions_created,
            'blocked_by_max': len(env.positions) >= env.max_positions,
            'raw_decision': float(action[0])
        })
        
        # Se criou posição, mostrar detalhes
        if positions_created > 0:
            pos = env.positions[-1]
            print(f"  ✅ POSIÇÃO CRIADA:")
            print(f"    Tipo: {pos.get('type', 'unknown')}")
            print(f"    Entry: ${pos.get('entry_price', 0):.2f}")
            print(f"    SL: ${pos.get('sl', 0):.2f}")
            print(f"    TP: ${pos.get('tp', 0):.2f}")
        
        if done:
            obs = env.reset()
    
    # Análise dos resultados
    print(f"\n📊 ANÁLISE DOS RESULTADOS")
    print("-" * 50)
    
    if scores_log:
        df_scores = pd.DataFrame(scores_log)
        
        total_tests = len(df_scores)
        created_count = df_scores['created'].sum()
        blocked_by_max = df_scores['blocked_by_max'].sum()
        
        print(f"Total de testes: {total_tests}")
        print(f"Posições criadas: {created_count}")
        print(f"Bloqueado por max: {blocked_by_max}")
        print(f"Taxa de criação: {(created_count/total_tests)*100:.1f}%")
        
        # Analisar por tipo de ação
        print(f"\n🔍 DETALHES POR TESTE:")
        for _, row in df_scores.iterrows():
            status = "CRIOU" if row['created'] > 0 else "BLOQUEOU"
            reason = "MAX_POS" if row['blocked_by_max'] else "THRESHOLD?"
            
            print(f"  {row['name']:15} | {status:8} | {reason:10} | Decision: {row['raw_decision']:.2f}")
    
    print(f"\n🚨 PROBLEMA IDENTIFICADO:")
    print(f"Se mesmo com threshold 0.85 ainda está criando posições,")
    print(f"então o V7 Intuition pode estar:")
    print(f"1. Ignorando o composite gate")
    print(f"2. Tendo scores sempre > 0.85")
    print(f"3. Usando um bypass oculto")

if __name__ == "__main__":
    diagnose_composite_scores()