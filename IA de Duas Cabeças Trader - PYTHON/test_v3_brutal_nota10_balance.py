#!/usr/bin/env python3
"""
🏆 V3 BRUTAL BALANCE TEST - NOTA 10
Teste de balanceamento igual ao aplicado no V3 Elegance
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class PurePnLEnvironment:
    """Environment PURO para teste final"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.peak_portfolio_value = initial_balance
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.trades = []
        self.positions = []
        self.current_step = 0
    
    def set_pnl_percent(self, pnl_percent: float):
        """Set PnL direto"""
        self.total_realized_pnl = self.initial_balance * pnl_percent
        self.portfolio_value = self.initial_balance * (1 + pnl_percent)
        
        # Update peak se for lucro
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

def test_v3_brutal_nota_10():
    """🏆 TESTE FINAL NOTA 10 - V3 BRUTAL"""
    
    print("🏆 V3 BRUTAL - TESTE NOTA 10 (BALANCE)")
    print("=" * 50)
    
    # Sistema V3 Brutal
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Cenários puros de PnL (iguais ao V3 Elegance)
    test_cases = [
        -0.10,  # -10%
        -0.05,  # -5%
        0.0,    # 0%
        0.05,   # +5%
        0.10,   # +10%
        0.20,   # +20%
    ]
    
    print(f"📊 TESTE PNL PURO (comparação com V3 Elegance):")
    
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
    
    # Perfect linearity check - adaptar para escala do V3 Brutal
    # V3 Brutal usa scaling diferente, então vamos calcular a escala esperada
    non_zero_rewards = [r for r in rewards if abs(r) > 0.001]
    non_zero_pnls = [p for p, r in zip(pnl_percents, rewards) if abs(r) > 0.001]
    
    if non_zero_rewards and non_zero_pnls:
        # Calcular scale factor baseado nos dados não-zero
        scale_factor = np.mean([r/p for r, p in zip(non_zero_rewards, non_zero_pnls) if abs(p) > 0.1])
        expected_rewards = [p * scale_factor for p in pnl_percents]
    else:
        expected_rewards = [0.0] * len(pnl_percents)
    
    linearity_error = np.mean(np.abs(np.array(rewards) - np.array(expected_rewards)))
    
    print(f"\n🎯 ANÁLISE FINAL:")
    print(f"   PnL-Reward Correlation: {correlation:.6f}")
    print(f"   Scale Factor: {scale_factor:.6f}")
    print(f"   Linearity Error: {linearity_error:.6f}")
    
    # Análise de componentes
    print(f"\n📊 BREAKDOWN DOS COMPONENTES:")
    for i, pnl_pct in enumerate(test_cases):
        env = PurePnLEnvironment()
        reward_system.reset()
        env.set_pnl_percent(pnl_pct)
        
        reward, info, done = reward_system.calculate_reward_and_info(env, np.zeros(8), {})
        
        pnl_component = info.get('pnl_reward', 0)
        risk_component = info.get('risk_reward', 0) 
        shaping_component = info.get('shaping_reward', 0)
        
        print(f"   PnL {pnl_pct*100:+.0f}%:")
        print(f"     Total: {reward:+.6f}")
        print(f"     └─ PnL: {pnl_component:+.6f} ({abs(pnl_component)/max(abs(reward), 0.001)*100:.1f}%)")
        print(f"     └─ Risk: {risk_component:+.6f} ({abs(risk_component)/max(abs(reward), 0.001)*100:.1f}%)")
        print(f"     └─ Shaping: {shaping_component:+.6f} ({abs(shaping_component)/max(abs(reward), 0.001)*100:.1f}%)")
    
    # Noise analysis
    residuals = np.array(rewards) - np.array(expected_rewards)
    noise_level = np.std(residuals)
    signal_level = np.std(expected_rewards) if np.std(expected_rewards) > 0 else np.std(rewards)
    snr = signal_level / max(noise_level, 0.001)
    
    print(f"\n📈 SIGNAL-TO-NOISE ANALYSIS:")
    print(f"   Signal Level: {signal_level:.6f}")
    print(f"   Noise Level: {noise_level:.6f}")
    print(f"   Signal-to-Noise: {snr:.2f}:1")
    
    # Componentes dominantes analysis
    all_rewards = []
    all_pnl_components = []
    all_risk_components = []
    all_shaping_components = []
    
    for pnl_pct in test_cases:
        env = PurePnLEnvironment()
        reward_system.reset()
        env.set_pnl_percent(pnl_pct)
        reward, info, done = reward_system.calculate_reward_and_info(env, np.zeros(8), {})
        
        all_rewards.append(abs(reward))
        all_pnl_components.append(abs(info.get('pnl_reward', 0)))
        all_risk_components.append(abs(info.get('risk_reward', 0)))
        all_shaping_components.append(abs(info.get('shaping_reward', 0)))
    
    total_signal = sum(all_rewards)
    pnl_dominance = sum(all_pnl_components) / max(total_signal, 0.001) * 100
    risk_dominance = sum(all_risk_components) / max(total_signal, 0.001) * 100
    shaping_dominance = sum(all_shaping_components) / max(total_signal, 0.001) * 100
    
    print(f"\n⚖️ COMPONENT DOMINANCE:")
    print(f"   PnL Component: {pnl_dominance:.1f}%")
    print(f"   Risk Component: {risk_dominance:.1f}%") 
    print(f"   Shaping Component: {shaping_dominance:.1f}%")
    
    # Grade calculation
    if correlation > 0.999 and linearity_error < 0.1 and snr > 50 and pnl_dominance > 80:
        grade = "A+"
        nota = 10
    elif correlation > 0.99 and linearity_error < 0.2 and snr > 20 and pnl_dominance > 70:
        grade = "A"
        nota = 9
    elif correlation > 0.95 and snr > 10 and pnl_dominance > 60:
        grade = "B"
        nota = 8
    else:
        grade = "C"
        nota = 7
    
    print(f"\n🏆 NOTA FINAL: {nota}/10 - {grade}")
    
    if nota == 10:
        print("🎉 PERFEIÇÃO ALCANÇADA!")
        print("   ✅ Correlação excelente")
        print("   ✅ Linearidade adequada") 
        print("   ✅ Low noise")
        print("   ✅ PnL dominante > 80%")
    else:
        print(f"❌ Ainda não nota 10:")
        print(f"   Correlation: {correlation:.6f} (need > 0.999)")
        print(f"   Linearity: {linearity_error:.6f} (need < 0.1)")
        print(f"   SNR: {snr:.2f}:1 (need > 50:1)")
        print(f"   PnL Dominance: {pnl_dominance:.1f}% (need > 80%)")
        
        # Sugestões para melhoria
        print(f"\n🔧 SUGESTÕES PARA NOTA 10:")
        if pnl_dominance < 80:
            print(f"   • Aumentar peso PnL, reduzir risk/shaping")
        if snr < 50:
            print(f"   • Reduzir componentes ruidosos (risk/shaping)")
        if linearity_error > 0.1:
            print(f"   • Simplificar cálculo reward para maior linearidade")
        if correlation < 0.999:
            print(f"   • Eliminar componentes não-lineares")
    
    return nota

if __name__ == "__main__":
    final_grade = test_v3_brutal_nota_10()
    
    if final_grade == 10:
        print("\n🚀 V3 BRUTAL APROVADO COM NOTA 10!")
    else:
        print(f"\n🔧 Precisa ajustes para nota 10 (atual: {final_grade})")