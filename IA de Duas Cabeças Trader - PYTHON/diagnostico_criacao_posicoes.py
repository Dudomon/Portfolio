#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE CRIAÇÃO DE POSIÇÕES
Analisa por que poucas posições estão sendo criadas
"""

import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict
import torch

# Adicionar paths
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

from daytrader import TradingEnv, TRADING_CONFIG, TRIAL_2_TRADING_PARAMS

def diagnose_position_creation():
    """Diagnosticar problema de criação de posições"""
    print("🔍 DIAGNÓSTICO DE CRIAÇÃO DE POSIÇÕES")
    print("="*50)
    
    # Criar dados mock com oportunidades claras
    dates = pd.date_range('2023-01-01', periods=200, freq='5min')
    base_price = 4000
    
    # Criar padrões óbvios para trading
    price_changes = np.concatenate([
        np.random.normal(0, 0.005, 50),      # Baixa volatilidade
        np.linspace(0, 0.15, 50),            # Trend forte para cima
        np.random.normal(0, 0.02, 50),       # Alta volatilidade  
        np.linspace(0, -0.15, 50),           # Trend forte para baixo
    ])
    
    prices = [base_price]
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))
    prices = prices[1:]
    
    df = pd.DataFrame({
        'close_5m': prices,
        'high_5m': [p * 1.002 for p in prices],
        'low_5m': [p * 0.998 for p in prices],
        'volume_5m': [10000] * len(prices),
    }, index=dates)
    
    # Criar ambiente
    env = TradingEnv(df=df, window_size=20, is_training=True)
    obs = env.reset()
    
    print(f"📊 Dados criados: {len(df)} barras")
    print(f"💰 Variação total: {((prices[-1] - prices[0]) / prices[0]) * 100:.2f}%")
    
    # Testar diferentes tipos de ações
    print(f"\n🎯 TESTE 1: AÇÕES AGRESSIVAS")
    print("-" * 50)
    
    action_tests = [
        ("Super Bullish", np.array([1.0, 0.9, 0.8, 0.8, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0])),
        ("Bullish Normal", np.array([1.0, 0.7, 0.6, 0.6, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0])),
        ("Super Bearish", np.array([0.0, 0.1, 0.0, 0.2, 0.8, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0])),
        ("Bearish Normal", np.array([0.0, 0.3, 0.0, 0.4, 0.6, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0])),
        ("Mixed Aggressive", np.array([0.8, 0.8, 0.8, 0.2, 0.8, 4.0, 4.0, 0.0, 2.5, 2.5, 0.0])),
    ]
    
    total_attempts = 0
    total_created = 0
    detailed_log = []
    
    # Reset para começar limpo
    obs = env.reset()
    
    for name, action in action_tests:
        positions_before = len(env.positions)
        current_price = df['close_5m'].iloc[env.current_step]
        
        # Executar ação
        obs, reward, done, info = env.step(action)
        
        positions_after = len(env.positions)
        positions_created = positions_after - positions_before
        
        total_attempts += 1
        total_created += positions_created
        
        print(f"  {name}:")
        print(f"    Ação: {action[:5]}")  # Mostrar só parte da ação
        print(f"    Preço: ${current_price:.2f}")
        print(f"    Posições: {positions_before} -> {positions_after} ({positions_created:+d})")
        
        # Log detalhado
        detailed_log.append({
            'name': name,
            'action': action.copy(),
            'price': current_price,
            'positions_before': positions_before,
            'positions_after': positions_after,
            'created': positions_created,
            'step': env.current_step
        })
        
        if done:
            obs = env.reset()
    
    print(f"\n📊 TESTE 2: SEQUÊNCIA LONGA COM VARIAÇÕES")
    print("-" * 50)
    
    # Reset ambiente
    obs = env.reset()
    
    sequence_log = []
    max_steps = min(100, len(df) - 30)  # Deixar margem
    
    for i in range(max_steps):
        # Variar ações baseado no step
        if i % 10 < 3:  # Primeiros 30% - Bullish
            action = np.array([1.0, 0.8, 0.7, 0.6, 0.0, 4.0, 0.0, 0.0, 2.5, 0.0, 0.0])
        elif i % 10 < 6:  # Próximos 30% - Bearish  
            action = np.array([0.0, 0.2, 0.0, 0.4, 0.8, 0.0, 4.0, 0.0, 0.0, 2.5, 0.0])
        elif i % 10 < 8:  # 20% - Neutro
            action = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        else:  # 20% - Agressivo misto
            action = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0])
        
        positions_before = len(env.positions)
        current_price = df['close_5m'].iloc[env.current_step]
        step = env.current_step
        
        obs, reward, done, info = env.step(action)
        
        positions_after = len(env.positions)
        positions_created = positions_after - positions_before
        
        total_attempts += 1
        if positions_created > 0:
            total_created += positions_created
            print(f"  Step {step}: CRIOU {positions_created} posição(s) - Preço: ${current_price:.2f}")
        
        sequence_log.append({
            'step': step,
            'price': current_price,
            'positions_before': positions_before,
            'positions_after': positions_after,
            'created': positions_created,
            'action_type': 'bullish' if action[0] > 0.7 else 'bearish' if action[4] > 0.7 else 'mixed'
        })
        
        # Parar se max positions ou done
        if positions_after >= 3:
            print(f"    ⚠️ Max positions atingido: {positions_after}")
            break
            
        if done:
            print(f"    ✅ Episódio terminou no step {step}")
            break
    
    # Análise final
    print(f"\n📊 ANÁLISE FINAL")
    print("-" * 50)
    
    print(f"Total de tentativas: {total_attempts}")
    print(f"Total de posições criadas: {total_created}")
    print(f"Taxa de criação: {(total_created/total_attempts)*100:.1f}%")
    print(f"Posições finais: {len(env.positions)}")
    
    # Análise por tipo de ação
    if sequence_log:
        df_seq = pd.DataFrame(sequence_log)
        
        print(f"\n🎯 ANÁLISE POR TIPO DE AÇÃO:")
        for action_type in ['bullish', 'bearish', 'mixed']:
            subset = df_seq[df_seq['action_type'] == action_type]
            if len(subset) > 0:
                created = subset['created'].sum()
                attempts = len(subset)
                rate = (created/attempts)*100 if attempts > 0 else 0
                print(f"  {action_type.title()}: {created}/{attempts} ({rate:.1f}%)")
    
    # Verificar se max_positions é o limitador
    max_pos_reached = any(log['positions_after'] >= 3 for log in sequence_log)
    print(f"\nMax positions (3) atingido: {'SIM' if max_pos_reached else 'NÃO'}")
    
    if total_created < 5:
        print(f"\n❌ PROBLEMA CONFIRMADO: Muito poucas posições criadas!")
        print(f"   Possíveis causas:")
        print(f"   1. V7 Intuition composite threshold muito alto (0.5)")
        print(f"   2. Filtros ocultos no V7 que não identificamos")
        print(f"   3. Problema na interpretação das ações")
    else:
        print(f"\n✅ Taxa de criação OK: {total_created} posições em {total_attempts} tentativas")

if __name__ == "__main__":
    diagnose_position_creation()