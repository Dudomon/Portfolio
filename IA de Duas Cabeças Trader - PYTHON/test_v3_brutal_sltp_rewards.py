#!/usr/bin/env python3
"""
🧪 TESTE dos novos rewards SL/TP + Trailing Stop do V3 Brutal
"""

import sys
import os
sys.path.append('.')

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

def main():
    """Testar os novos componentes de reward para SL/TP dinâmico"""
    print("🚀 TESTE V3 BRUTAL - Novos Rewards SL/TP + Trailing Stop")
    print("=" * 60)
    
    # Inicializar reward system
    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    
    # Executar teste dos novos rewards
    reward_system.test_trailing_sltp_rewards()
    
    print("\n" + "=" * 60)
    print("🎯 RESUMO DOS NOVOS REWARDS:")
    print("✅ Trailing Stop Activation: +0.01 a +0.04 (baseado no timing)")
    print("✅ Trailing Stop Protection: +0.025 (protegeu lucros)")
    print("✅ Trailing Stop Movement: +0.005 por movimento")
    print("✅ SL Adjustment Defense: +0.015 (preservou resultado)")
    print("✅ TP Adjustment Expansion: +0.01 a +0.04 (expandiu alvos)")
    print("✅ Combo Reward: +0.015 (trailing + TP juntos)")
    print("❌ Missed Opportunity: -0.01 (não usou trailing)")
    print("\n🎯 O modelo agora aprenderá a usar SL/TP dinâmico e trailing stops!")

if __name__ == "__main__":
    main()