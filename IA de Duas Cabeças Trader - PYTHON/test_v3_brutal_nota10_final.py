#!/usr/bin/env python3
"""
🏆 V3 BRUTAL NOTA 10 - TESTE FINAL
Sistema PnL puro sem qualquer noise - versão otimizada
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
from trading_framework.rewards.reward_daytrade_v3_brutal_nota10 import BrutalMoneyRewardNota10

class PurePnLEnvironment:
    """Environment PURO para teste final"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.peak_portfolio_value = initial_balance
    
    def set_pnl_percent(self, pnl_percent: float):
        """Set PnL direto"""
        self.portfolio_value = self.initial_balance * (1 + pnl_percent)
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

def test_brutal_nota_10():
    """🏆 TESTE FINAL NOTA 10 - V3 BRUTAL PURO"""
    
    print("🏆 V3 BRUTAL NOTA 10 - TESTE FINAL")
    print("=" * 50)
    
    # Sistema V3 Brutal NOTA 10
    reward_system = BrutalMoneyRewardNota10(initial_balance=10000)
    
    # Cenários puros de PnL (iguais ao V3 Elegance)
    test_cases = [
        -0.10,  # -10%
        -0.05,  # -5%
        0.0,    # 0%
        0.05,   # +5%
        0.10,   # +10%
        0.20,   # +20%
    ]
    
    print(f"📊 TESTE PNL PURO (sem noise):")
    
    rewards = []
    pnl_percents = []
    
    for pnl_pct in test_cases:
        # Ambiente puro
        env = PurePnLEnvironment()
        reward_system.reset()
        env.set_pnl_percent(pnl_pct)
        
        # Ação dummy
        dummy_action = np.zeros(8)
        
        # Reward calculation
        reward, info, done = reward_system.calculate_reward_and_info(env, dummy_action, {})
        
        rewards.append(reward)
        pnl_percents.append(pnl_pct * 100)
        
        print(f"   PnL: {pnl_pct*100:+.0f}% → Reward: {reward:+.6f}")
    
    # Análise final
    correlation = np.corrcoef(pnl_percents, rewards)[0, 1]
    
    # Perfect linearity check - escala 10x como no V3 Elegance
    expected_rewards = [p * 0.1 for p in pnl_percents]  # Scale 10x
    linearity_error = np.mean(np.abs(np.array(rewards) - np.array(expected_rewards)))
    
    print(f"\n🎯 ANÁLISE FINAL:")
    print(f"   PnL-Reward Correlation: {correlation:.6f}")
    print(f"   Expected Scale: 0.1 (10x)")
    print(f"   Linearity Error: {linearity_error:.6f}")
    print(f"   Perfect Scaling: {all(abs(r - e) < 0.001 for r, e in zip(rewards, expected_rewards))}")
    
    # Noise analysis
    residuals = np.array(rewards) - np.array(expected_rewards)
    noise_level = np.std(residuals)
    signal_level = np.std(expected_rewards)
    snr = signal_level / max(noise_level, 0.001)
    
    print(f"   Signal-to-Noise: {snr:.2f}:1")
    print(f"   Noise Level: {noise_level:.6f}")
    
    # Componentes analysis
    print(f"\n📊 BREAKDOWN DOS COMPONENTES:")
    for pnl_pct in test_cases:
        env = PurePnLEnvironment()
        reward_system.reset()
        env.set_pnl_percent(pnl_pct)
        
        reward, info, done = reward_system.calculate_reward_and_info(env, np.zeros(8), {})
        
        pnl_component = info.get('pnl_component', 0)
        risk_component = info.get('risk_component', 0)
        shaping_component = info.get('shaping_component', 0)
        
        print(f"   PnL {pnl_pct*100:+.0f}%: Total={reward:+.6f} | PnL={pnl_component:+.6f} | Risk={risk_component:+.6f} | Shaping={shaping_component:+.6f}")
    
    # Grade calculation
    if correlation > 0.9999 and linearity_error < 0.001 and snr > 100:
        grade = "A+"
        nota = 10
    elif correlation > 0.999 and linearity_error < 0.01 and snr > 50:
        grade = "A"
        nota = 9
    elif correlation > 0.99 and snr > 10:
        grade = "B"
        nota = 8
    else:
        grade = "C"
        nota = 7
    
    print(f"\n🏆 NOTA FINAL: {nota}/10 - {grade}")
    
    if nota == 10:
        print("🎉 PERFEIÇÃO ALCANÇADA!")
        print("   ✅ Correlação perfeita")
        print("   ✅ Linearidade perfeita") 
        print("   ✅ Zero noise")
        print("   ✅ Signal-to-noise > 100:1")
    else:
        print(f"❌ Ainda não nota 10:")
        print(f"   Correlation: {correlation:.6f} (need > 0.9999)")
        print(f"   Linearity: {linearity_error:.6f} (need < 0.001)")
        print(f"   SNR: {snr:.2f}:1 (need > 100:1)")
    
    return nota

if __name__ == "__main__":
    final_grade = test_brutal_nota_10()
    
    if final_grade == 10:
        print("\n🚀 V3 BRUTAL NOTA 10 APROVADO!")
    else:
        print(f"\n🔧 Precisa ajustes para nota 10 (atual: {final_grade})")