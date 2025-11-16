#!/usr/bin/env python3
"""
Debug do Sistema de Episódios SILUS
===================================
Investigar por que episódios não estão terminando corretamente
"""

import sys
import os
sys.path.append('D:/Projeto')

import numpy as np
import pandas as pd
from pathlib import Path
import time

# Importar o ambiente SILUS
from silus import TradingEnv, load_optimized_data

def debug_episode_system():
    """Testar o sistema de episódios manualmente"""
    
    print("="*80)
    print("🔍 DEBUG DO SISTEMA DE EPISÓDIOS SILUS")
    print("="*80)
    
    # Carregar dados
    print("\n⏳ Carregando dataset...")
    df = load_optimized_data()
    print(f"   Dataset carregado: {len(df)} barras")
    
    # Criar ambiente
    print("\n🏗️ Criando ambiente...")
    env = TradingEnv(df, window_size=20, is_training=True, initial_balance=500)
    
    print(f"   MAX_STEPS configurado: {env.MAX_STEPS}")
    print(f"   Dataset size: {len(env.df)}")
    print(f"   Initial balance: {env.initial_balance}")
    print(f"   Current step inicial: {env.current_step}")
    print(f"   Episode steps inicial: {env.episode_steps}")
    
    # Testar reset
    print("\n🔄 Testando reset...")
    obs = env.reset()
    print(f"   Após reset:")
    print(f"     Current step: {env.current_step}")
    print(f"     Episode steps: {env.episode_steps}")
    print(f"     Portfolio value: {env.portfolio_value}")
    print(f"     Observation shape: {obs.shape}")
    
    # Simular alguns steps
    print(f"\n🚀 Simulando steps até encontrar episódio completo...")
    
    episode_count = 0
    step_count = 0
    max_test_steps = 10000
    
    episode_rewards = []
    episode_lengths = []
    portfolio_resets = []
    
    current_episode_reward = 0
    current_episode_length = 0
    
    for step in range(max_test_steps):
        # Ação aleatória
        action = env.action_space.sample()
        
        # Step no ambiente
        obs, reward, done, info = env.step(action)
        
        current_episode_reward += reward
        current_episode_length += 1
        step_count += 1
        
        # Verificar reset de portfolio
        if abs(env.portfolio_value - 500.0) < 0.01:
            portfolio_resets.append(step)
        
        # Verificar se episódio terminou
        if done:
            episode_count += 1
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            print(f"\n✅ EPISÓDIO {episode_count} COMPLETADO!")
            print(f"   Steps no episódio: {current_episode_length}")
            print(f"   Reward total: {current_episode_reward:.6f}")
            print(f"   Portfolio final: {env.portfolio_value:.2f}")
            print(f"   Trades no episódio: {len(env.trades)}")
            print(f"   Current step: {env.current_step}")
            print(f"   Episode steps: {env.episode_steps}")
            
            # Reset para próximo episódio
            obs = env.reset()
            current_episode_reward = 0
            current_episode_length = 0
            
            print(f"   Após reset automático:")
            print(f"     Current step: {env.current_step}")
            print(f"     Episode steps: {env.episode_steps}")
            print(f"     Portfolio value: {env.portfolio_value}")
            
            # Parar após alguns episódios para análise
            if episode_count >= 3:
                break
        
        # Log de progresso
        if step % 1000 == 0 and step > 0:
            print(f"   Step {step}: Episode steps: {env.episode_steps}, Portfolio: {env.portfolio_value:.2f}")
    
    # Análise final
    print(f"\n" + "="*80)
    print("📊 RESULTADO DA ANÁLISE")
    print("="*80)
    
    print(f"\n🔢 ESTATÍSTICAS:")
    print(f"   Total steps testados: {step_count}")
    print(f"   Episódios completados: {episode_count}")
    print(f"   Portfolio resets detectados: {len(portfolio_resets)}")
    
    if episode_count > 0:
        print(f"\n✅ EPISÓDIOS FUNCIONANDO:")
        print(f"   Reward médio por episódio: {np.mean(episode_rewards):.6f}")
        print(f"   Length médio por episódio: {np.mean(episode_lengths):.1f}")
        print(f"   Steps entre resets de portfolio: {np.diff(portfolio_resets).mean():.1f}")
        
        # Verificar se rewards são sempre zero
        if all(r == 0 for r in episode_rewards):
            print(f"   ❌ PROBLEMA: Todos os episode rewards são ZERO!")
        else:
            print(f"   ✅ Episode rewards variando corretamente")
    
    else:
        print(f"\n❌ PROBLEMA CRÍTICO: NENHUM EPISÓDIO COMPLETADO!")
        print(f"   Max episode length atual: {current_episode_length}")
        print(f"   Current step final: {env.current_step}")
        print(f"   Episode steps final: {env.episode_steps}")
        
        # Verificar condições de done
        print(f"\n🔍 VERIFICANDO CONDIÇÕES DE DONE:")
        print(f"   current_step >= len(df) - 1: {env.current_step >= len(env.df) - 1}")
        print(f"   episode_steps >= MAX_STEPS: {env.episode_steps >= env.MAX_STEPS}")
        print(f"   Valores: current_step={env.current_step}, len(df)={len(env.df)}, episode_steps={env.episode_steps}, MAX_STEPS={env.MAX_STEPS}")
    
    if portfolio_resets:
        print(f"\n🔄 PORTFOLIO RESETS:")
        print(f"   Frequência média: a cada {step_count / len(portfolio_resets):.1f} steps")
        print(f"   Primeiros resets em: {portfolio_resets[:10]}")
    
    # Analisar system reward
    print(f"\n🎯 ANÁLISE DO REWARD SYSTEM:")
    if hasattr(env, 'reward_system'):
        print(f"   Reward system: {type(env.reward_system).__name__}")
        print(f"   Initial balance: {env.reward_system.initial_balance}")
    else:
        print(f"   ❌ Reward system não encontrado!")
    
    return {
        'episodes_completed': episode_count,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'portfolio_resets': len(portfolio_resets),
        'total_steps': step_count
    }

if __name__ == "__main__":
    result = debug_episode_system()
    
    print(f"\n" + "="*80)
    print("💡 CONCLUSÕES")
    print("="*80)
    
    if result['episodes_completed'] == 0:
        print("""
❌ PROBLEMA CONFIRMADO: Sistema de episódios QUEBRADO

POSSÍVEIS CAUSAS:
1. MAX_STEPS muito alto (3000) vs dataset pequeno
2. Condição de done não sendo atendida
3. current_step não avançando corretamente
4. episode_steps não sendo resetado

SOLUÇÕES SUGERIDAS:
1. Reduzir MAX_STEPS para 252 (1 dia de trading)
2. Verificar lógica de done no step()
3. Garantir reset correto de episode_steps
4. Adicionar done por timeout absoluto
""")
    else:
        print(f"""
✅ Sistema funcionando parcialmente
- {result['episodes_completed']} episódios completados
- Reward médio: {np.mean(result['episode_rewards']) if result['episode_rewards'] else 0:.6f}

Se rewards são zero, problema é no reward system, não nos episódios.
""")