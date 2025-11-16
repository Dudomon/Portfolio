#!/usr/bin/env python3
"""
Debug Simples do Sistema de Episódios SILUS
==========================================
Usar dataset sintético para testar episódios
"""

import sys
import os
sys.path.append('D:/Projeto')

import numpy as np
import pandas as pd
import time

def create_synthetic_dataset(length=5000):
    """Criar dataset sintético para teste"""
    np.random.seed(42)
    
    # Preço base (simular ouro ~$2000)
    base_price = 2000.0
    prices = [base_price]
    
    # Gerar preços com random walk
    for i in range(length):
        change = np.random.normal(0, 0.5)  # Volatilidade realista
        new_price = max(prices[-1] + change, 1800)  # Não deixar muito baixo
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Criar DataFrame com estrutura esperada (todos arrays de tamanho length)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
        'open_5m': prices[:-1],
        'high_5m': prices[:-1] + np.abs(np.random.normal(0, 0.3, length)),
        'low_5m': prices[:-1] - np.abs(np.random.normal(0, 0.3, length)),
        'close_5m': prices[1:],
        'volume_5m': np.random.uniform(100, 1000, length),
        
        # 1h timeframe (aproximado)
        'open_1h': prices[:-1],
        'high_1h': prices[:-1] + np.abs(np.random.normal(0, 0.5, length)),
        'low_1h': prices[:-1] - np.abs(np.random.normal(0, 0.5, length)),
        'close_1h': prices[1:],
        'volume_1h': np.random.uniform(500, 2000, length),
    })
    
    return df

def debug_episode_basic():
    """Debug básico sem importar silus completo"""
    
    print("="*80)
    print("🔍 DEBUG BÁSICO DO SISTEMA DE EPISÓDIOS")
    print("="*80)
    
    # Criar dataset sintético
    print("\n⏳ Criando dataset sintético...")
    df = create_synthetic_dataset(5000)
    print(f"   Dataset criado: {len(df)} barras")
    
    # Simular parâmetros do SILUS
    MAX_STEPS = 3000
    initial_balance = 500.0
    window_size = 20
    
    print(f"\n📊 PARÂMETROS SIMULADOS:")
    print(f"   MAX_STEPS: {MAX_STEPS}")
    print(f"   Dataset size: {len(df)}")
    print(f"   Window size: {window_size}")
    print(f"   Initial balance: {initial_balance}")
    
    # Simular execução de episódio
    print(f"\n🚀 SIMULANDO EXECUÇÃO DE EPISÓDIO...")
    
    current_step = window_size  # Step inicial
    episode_steps = 0
    episode_rewards = []
    portfolio_resets = []
    
    portfolio_value = initial_balance
    episode_reward = 0
    episodes_completed = 0
    
    max_test_steps = 20000
    
    for step in range(max_test_steps):
        # Simular step
        current_step += 1
        episode_steps += 1
        
        # Simular reward (pode ser zero na maioria das vezes)
        reward = np.random.choice([0, 0, 0, 0.001, -0.001], p=[0.9, 0.02, 0.02, 0.03, 0.03])
        episode_reward += reward
        
        # Simular mudança no portfolio (pequenas flutuações)
        portfolio_change = np.random.normal(0, 2)
        portfolio_value += portfolio_change
        
        # Detectar reset de portfolio para 500
        if abs(portfolio_value - 500.0) < 0.1:
            portfolio_resets.append(step)
        
        # Verificar condições de done (igual ao SILUS)
        done = False
        
        # Condição 1: Dados acabaram
        if current_step >= len(df) - 1:
            done = True
            print(f"   Done por dados acabarem: step {step}")
        
        # Condição 2: MAX_STEPS atingido
        if episode_steps >= MAX_STEPS:
            done = True
            print(f"   Done por MAX_STEPS: step {step}")
        
        if done:
            episodes_completed += 1
            episode_rewards.append(episode_reward)
            
            print(f"\n✅ EPISÓDIO {episodes_completed} SIMULADO!")
            print(f"   Episode steps: {episode_steps}")
            print(f"   Episode reward: {episode_reward:.6f}")
            print(f"   Portfolio final: {portfolio_value:.2f}")
            print(f"   Current step: {current_step}")
            
            # Reset para próximo episódio
            current_step = window_size
            episode_steps = 0
            portfolio_value = initial_balance
            episode_reward = 0
            
            # Parar após 3 episódios
            if episodes_completed >= 3:
                break
        
        # Log de progresso
        if step % 2000 == 0 and step > 0:
            print(f"   Step {step}: Episode {episode_steps}, Portfolio {portfolio_value:.2f}")
    
    # Análise
    print(f"\n" + "="*80)
    print("📊 ANÁLISE DO TESTE SIMULADO")
    print("="*80)
    
    print(f"\n🔢 RESULTADOS:")
    print(f"   Total steps testados: {step + 1}")
    print(f"   Episódios completados: {episodes_completed}")
    print(f"   Portfolio resets: {len(portfolio_resets)}")
    
    if episodes_completed > 0:
        print(f"   Episode reward médio: {np.mean(episode_rewards):.6f}")
        print(f"   Rewards são sempre zero: {all(r == 0 for r in episode_rewards)}")
        
        print(f"\n✅ EPISÓDIOS FUNCIONARIAM CORRETAMENTE")
        print(f"   Problema REAL deve estar no:")
        print(f"   1. Reward system (retornando sempre 0)")
        print(f"   2. Condições de done não sendo atendidas")
        print(f"   3. MAX_STEPS muito alto vs dados disponíveis")
        
    else:
        print(f"\n❌ NENHUM EPISÓDIO COMPLETADO (PROBLEMA SIMULADO)")
        print(f"   Condições testadas:")
        print(f"   - current_step >= len(df): {current_step >= len(df) - 1}")
        print(f"   - episode_steps >= MAX_STEPS: {episode_steps >= MAX_STEPS}")
    
    # Cálculos específicos do SILUS
    print(f"\n🔍 ANÁLISE ESPECÍFICA PARA SILUS:")
    print(f"   Se dataset tem {len(df)} barras:")
    print(f"   - Episódios possíveis com MAX_STEPS=3000: {len(df) // MAX_STEPS}")
    print(f"   - Se MAX_STEPS > len(df): Apenas 1 episódio possível")
    print(f"   - Portfolio resets a cada 30 steps = {len(df) // 30} resets esperados")
    
    return episodes_completed > 0

if __name__ == "__main__":
    success = debug_episode_basic()
    
    print(f"\n" + "="*80)
    print("💡 DIAGNÓSTICO FINAL")
    print("="*80)
    
    if success:
        print("""
✅ LÓGICA DE EPISÓDIOS DEVERIA FUNCIONAR

O problema no SILUS provavelmente é:

1. REWARD SYSTEM: calculate_reward retornando sempre 0
2. DATASET MUITO PEQUENO: Se dataset < 3000 barras, apenas 1 episódio
3. RESET PREMATURO: Portfolio resetando antes de completar episódio
4. CONDIÇÕES DE DONE: Não sendo atendidas corretamente

INVESTIGAR PRÓXIMO:
- Tamanho real do dataset
- Sistema de rewards V4 INNO
- Lógica de reset do portfolio
""")
    else:
        print("""
❌ PROBLEMA NA LÓGICA BÁSICA

Verificar:
1. Condições de done
2. MAX_STEPS vs dataset size
3. current_step progression
""")