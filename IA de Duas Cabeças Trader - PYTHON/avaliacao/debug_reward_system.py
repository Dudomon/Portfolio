#!/usr/bin/env python3
"""
Debug do Sistema de Rewards V4 INNO
===================================
Testar se o reward system está retornando sempre 0
"""

import sys
import os
sys.path.append('D:/Projeto')

import numpy as np
from pathlib import Path

# Importar o reward system diretamente
from trading_framework.rewards.reward_daytrade_v4_inno import create_innovative_daytrade_reward_system

class MockEnv:
    """Ambiente simulado para testar rewards"""
    
    def __init__(self, initial_balance=500):
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.peak_portfolio_value = initial_balance
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.positions = []
        self.trades = []
        self.current_step = 0
        self.episode_steps = 0
        
    def simulate_profitable_trade(self):
        """Simular trade lucrativo"""
        self.total_realized_pnl += 10.0  # $10 lucro
        self.portfolio_value = self.initial_balance + self.total_realized_pnl
        
    def simulate_loss_trade(self):
        """Simular trade perdedor"""
        self.total_realized_pnl -= 5.0  # $5 perda
        self.portfolio_value = self.initial_balance + self.total_realized_pnl
        
    def simulate_unrealized_profit(self):
        """Simular lucro não realizado"""
        self.total_unrealized_pnl = 15.0  # $15 não realizado
        
    def simulate_no_activity(self):
        """Simular sem atividade"""
        pass  # Nada muda

def debug_reward_system():
    """Testar o sistema de rewards V4 INNO"""
    
    print("="*80)
    print("🔍 DEBUG DO SISTEMA DE REWARDS V4 INNO")
    print("="*80)
    
    # Criar reward system
    print("\n⏳ Criando reward system V4 INNO...")
    reward_system = create_innovative_daytrade_reward_system(initial_balance=500)
    print(f"   Reward system criado: {type(reward_system).__name__}")
    
    # Criar mock environment
    env = MockEnv(initial_balance=500)
    print(f"   Mock env criado: portfolio ${env.portfolio_value}")
    
    # Testar diferentes cenários
    scenarios = [
        ("Sem atividade", lambda: env.simulate_no_activity()),
        ("Trade lucrativo", lambda: env.simulate_profitable_trade()),
        ("Trade perdedor", lambda: env.simulate_loss_trade()),
        ("Lucro não realizado", lambda: env.simulate_unrealized_profit()),
    ]
    
    print(f"\n🧪 TESTANDO CENÁRIOS:")
    print("-" * 60)
    
    for name, scenario_func in scenarios:
        print(f"\n🔸 {name}:")
        
        # Reset inicial
        env.portfolio_value = 500
        env.total_realized_pnl = 0
        env.total_unrealized_pnl = 0
        
        # Aplicar cenário
        scenario_func()
        
        print(f"   Portfolio: ${env.portfolio_value:.2f}")
        print(f"   Realized PnL: ${env.total_realized_pnl:.2f}")
        print(f"   Unrealized PnL: ${env.total_unrealized_pnl:.2f}")
        
        # Testar reward
        action = np.array([0.5, 0.3, 0.2])  # Ação qualquer
        old_state = {}
        
        try:
            reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
            
            print(f"   ➜ REWARD: {reward:.6f}")
            print(f"   ➜ DONE: {done}")
            
            # Mostrar componentes do reward
            if 'pnl_reward' in info:
                print(f"     PnL Component: {info['pnl_reward']:.6f}")
            if 'shaping_reward' in info:
                print(f"     Shaping Component: {info['shaping_reward']:.6f}")
            if 'activity_bonus' in info:
                print(f"     Activity Component: {info['activity_bonus']:.6f}")
            if 'total_reward' in info:
                print(f"     Total Calculated: {info['total_reward']:.6f}")
                
            # Verificar se reward é sempre zero
            if reward == 0.0:
                print(f"   ❌ REWARD É ZERO!")
            else:
                print(f"   ✅ Reward não é zero")
                
        except Exception as e:
            print(f"   ❌ ERRO: {str(e)}")
    
    # Teste de stress - múltiplos steps
    print(f"\n🔥 TESTE DE STRESS - 100 STEPS:")
    print("-" * 60)
    
    env.portfolio_value = 500
    env.total_realized_pnl = 0
    env.total_unrealized_pnl = 0
    
    rewards_collected = []
    
    for step in range(100):
        # Simular variações aleatórias
        if step % 10 == 0:
            env.simulate_profitable_trade()
        elif step % 15 == 0:
            env.simulate_loss_trade()
        
        action = np.random.random(3)
        
        try:
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            rewards_collected.append(reward)
            
            if step < 5 or step % 20 == 0:
                print(f"   Step {step}: Reward {reward:.6f}, Portfolio ${env.portfolio_value:.2f}")
                
        except Exception as e:
            print(f"   Step {step}: ERRO - {str(e)}")
            rewards_collected.append(0.0)
    
    # Análise final
    print(f"\n" + "="*80)
    print("📊 ANÁLISE DOS RESULTADOS")
    print("="*80)
    
    rewards_array = np.array(rewards_collected)
    zero_rewards = (rewards_array == 0).sum()
    
    print(f"\n🔢 ESTATÍSTICAS:")
    print(f"   Total rewards testados: {len(rewards_collected)}")
    print(f"   Rewards = 0: {zero_rewards} ({zero_rewards/len(rewards_collected)*100:.1f}%)")
    print(f"   Reward médio: {np.mean(rewards_array):.6f}")
    print(f"   Reward min: {np.min(rewards_array):.6f}")
    print(f"   Reward max: {np.max(rewards_array):.6f}")
    print(f"   Reward std: {np.std(rewards_array):.6f}")
    
    if zero_rewards == len(rewards_collected):
        print(f"\n❌ PROBLEMA CONFIRMADO: TODOS OS REWARDS SÃO ZERO!")
        print(f"   O sistema de rewards V4 INNO não está funcionando")
        
        return False
    else:
        print(f"\n✅ REWARD SYSTEM FUNCIONANDO")
        print(f"   {len(rewards_collected) - zero_rewards} rewards não-zero encontrados")
        
        return True

if __name__ == "__main__":
    working = debug_reward_system()
    
    print(f"\n" + "="*80)
    print("💡 CONCLUSÃO")
    print("="*80)
    
    if not working:
        print("""
❌ REWARD SYSTEM V4 INNO QUEBRADO

INVESTIGAR:
1. Método _calculate_pure_pnl_reward_v4
2. Propriedades total_realized_pnl e total_unrealized_pnl do env
3. Condições de erro que retornam 0.0
4. Cache de rewards que pode estar interferindo

PRÓXIMOS PASSOS:
1. Verificar se env.total_realized_pnl existe
2. Testar reward system em ambiente real SILUS
3. Adicionar debug logs no reward system
""")
    else:
        print("""
✅ REWARD SYSTEM FUNCIONA EM TESTE

Se funciona aqui mas não no treinamento:
1. Problema na conexão env ↔ reward_system
2. Propriedades do env não estão sendo atualizadas
3. Episode termination interferindo
4. Cache problemático

INVESTIGAR env.total_realized_pnl durante treinamento real.
""")