#!/usr/bin/env python3
"""
📊 ANÁLISE RÁPIDA: V2 vs V3 BRUTAL
"""

import numpy as np

def analyze_rewards():
    # Sistema V2 (atual)
    def v2_reward(pnl_pct):
        pnl_component = pnl_pct * 5.0
        if pnl_pct > 0:
            bonus = 0.2 * pnl_pct
            total = pnl_component + bonus
        else:
            penalty = -0.3 * abs(pnl_pct)
            total = pnl_component + penalty
        return total / 5.0  # Normalização que mata
    
    # Sistema V3 Brutal
    def v3_brutal_reward(pnl_pct):
        base_reward = pnl_pct * 100.0
        if pnl_pct < -0.05:
            base_reward *= 4.0  # PAIN MULTIPLICATION
        elif pnl_pct > 0.03:
            base_reward *= 1.2
        return np.clip(base_reward, -50, 50)
    
    print("🔥 ANÁLISE MATEMÁTICA BRUTAL")
    print("=" * 50)
    
    test_cases = [-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10]
    
    print("PnL%     V2 Reward    V3 Brutal    Amplificação")
    print("-" * 50)
    
    for pnl in test_cases:
        v2_r = v2_reward(pnl)
        v3_r = v3_brutal_reward(pnl)
        amp = abs(v3_r / v2_r) if v2_r != 0 else "INF"
        
        print(f"{pnl*100:+5.0f}%    {v2_r:+8.2f}    {v3_r:+8.2f}    {amp}")
    
    # Casos críticos
    print(f"\n🎯 CASOS CRÍTICOS:")
    loss_10_v2 = v2_reward(-0.10)
    loss_10_v3 = v3_brutal_reward(-0.10)
    print(f"Perda 10%: V2 = {loss_10_v2:+.2f}, V3 = {loss_10_v3:+.2f}")
    print(f"V3 aplica {abs(loss_10_v3/loss_10_v2):.1f}x mais DOR!")
    
    gain_5_v2 = v2_reward(0.05)
    gain_5_v3 = v3_brutal_reward(0.05)
    print(f"Ganho 5%: V2 = {gain_5_v2:+.2f}, V3 = {gain_5_v3:+.2f}")
    print(f"V3 incentiva {gain_5_v3/gain_5_v2:.1f}x mais!")

if __name__ == "__main__":
    analyze_rewards()