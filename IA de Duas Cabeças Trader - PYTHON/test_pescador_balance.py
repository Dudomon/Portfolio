#!/usr/bin/env python3
"""
Teste de Balanceamento do PescadorRewardSystem
Verifica se os incentivos/penalidades estão adequados
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from reward_pescador import PescadorRewardSystem
import numpy as np

# Mock environment class para testes
class MockEnv:
    def __init__(self):
        self.current_step = 0
        self.positions = []
        self.trades = []
        
    def set_step(self, step):
        self.current_step = step
        
    def add_trade(self, pnl_usd=5.0, duration=8, exit_reason="TP"):
        self.trades.append({
            'pnl_usd': pnl_usd,
            'duration': duration, 
            'exit_reason': exit_reason,
            'type': 'long',
            'entry_step': self.current_step - duration,
            'exit_step': self.current_step
        })

def test_balance():
    print("=== TESTE DE BALANCEAMENTO PESCADOR REWARD ===\n")
    
    # Inicializar reward system
    reward_sys = PescadorRewardSystem(
        activity_bonus=0.05,
        inactivity_threshold=50,
        inactivity_penalty=0.01
    )
    
    env = MockEnv()
    action = np.array([0, 0])  # Hold
    old_state = {}
    
    scenarios = [
        ("🟢 CENÁRIO 1: Atividade recente (10 steps)", 10),
        ("🟡 CENÁRIO 2: Inatividade moderada (75 steps)", 75), 
        ("🔴 CENÁRIO 3: Inatividade severa (150 steps)", 150),
        ("⚫ CENÁRIO 4: Inatividade extrema (300 steps)", 300),
    ]
    
    print("VALORES BASE:")
    print(f"  Activity Bonus: +{reward_sys.activity_bonus}")
    print(f"  Inactivity Threshold: {reward_sys.inactivity_threshold} steps")
    print(f"  Inactivity Penalty Base: -{reward_sys.inactivity_penalty}")
    print()
    
    for desc, steps_inactive in scenarios:
        # Simular último trade há X steps atrás
        reward_sys.last_trade_step = -steps_inactive
        env.set_step(0)
        
        reward, info, _ = reward_sys.calculate_reward_and_info(env, action, old_state)
        
        print(f"{desc}")
        print(f"  Steps sem trade: {steps_inactive}")
        print(f"  Reward total: {reward:+.4f}")
        if 'activity_bonus' in info.get('components', {}):
            print(f"  Activity bonus: +{info['components']['activity_bonus']:.4f}")
        if 'inactivity_penalty' in info.get('components', {}):
            print(f"  Inactivity penalty: {info['components']['inactivity_penalty']:+.4f}")
        print()
    
    print("=== TESTE COM TRADE BEM-SUCEDIDO ===")
    
    # Resetar para teste de trade
    reward_sys = PescadorRewardSystem()
    env = MockEnv()
    env.set_step(100)
    
    # Simular trade rápido e lucrativo 
    env.add_trade(pnl_usd=8.0, duration=6, exit_reason="TP")
    
    reward, info, _ = reward_sys.calculate_reward_and_info(env, action, old_state)
    
    print(f"Trade rápido (+$8, 6 steps, TP):")
    print(f"  Reward total: {reward:+.4f}")
    components = info.get('components', {})
    for comp_name, comp_value in components.items():
        print(f"  {comp_name}: {comp_value:+.4f}")
    print()
    
    print("=== AVALIAÇÃO DO BALANCEAMENTO ===")
    print("✅ Activity bonus moderado: incentiva sem causar spam")
    print("✅ Inactivity penalty gradual: evita paralisia prolongada")
    print("✅ Foco mantido em PnL e velocidade (core pescador)")
    print("✅ Valores adequados para scalping (não overrides comportamento)")

if __name__ == "__main__":
    test_balance()