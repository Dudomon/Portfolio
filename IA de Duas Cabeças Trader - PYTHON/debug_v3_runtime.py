#!/usr/bin/env python3
"""
🔍 DEBUG V3 BRUTAL - RUNTIME
Verifica o que está acontecendo com o V3 brutal durante o treinamento real
"""

import numpy as np
import sys
import os
import time
sys.path.append(os.getcwd())

# Monkey patch do V3 brutal para debug
from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

# Salvar método original
original_calculate = BrutalMoneyReward.calculate_reward_and_info

def debug_calculate_reward_and_info(self, env, action, old_state):
    """Wrapper com debug para o V3 brutal"""
    
    # Chamar método original
    reward, info, done = original_calculate(self, env, action, old_state)
    
    # Debug apenas a cada 100 steps para não spammar
    if hasattr(self, '_debug_counter'):
        self._debug_counter += 1
    else:
        self._debug_counter = 1
    
    if self._debug_counter % 500 == 0 or abs(reward) > 0.001:  # Debug frequente se reward != 0
        # Extrair dados do environment
        realized_pnl = getattr(env, 'total_realized_pnl', 0.0)
        unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)
        portfolio_value = getattr(env, 'portfolio_value', getattr(env, 'current_balance', 1000))
        
        total_pnl = realized_pnl + (unrealized_pnl * 0.5)
        pnl_percent = (total_pnl / self.initial_balance) * 100
        
        print(f"\n🔍 [V3 BRUTAL DEBUG] Step {self._debug_counter}")
        print(f"   💰 Realized PnL: ${realized_pnl:+.2f}")
        print(f"   💰 Unrealized PnL: ${unrealized_pnl:+.2f}")
        print(f"   💰 Total PnL: ${total_pnl:+.2f} ({pnl_percent:+.2f}%)")
        print(f"   💰 Portfolio: ${portfolio_value:.2f}")
        print(f"   🎯 Final Reward: {reward:+.6f}")
        print(f"   📊 PnL Reward: {info.get('pnl_reward', 0):+.6f}")
        print(f"   📊 Shaping Reward: {info.get('shaping_reward', 0):+.6f}")
        
        if abs(reward) < 0.001 and abs(total_pnl) > 1.0:
            print(f"   🚨 PROBLEMA: PnL significativo (${total_pnl:.2f}) mas reward microscópico ({reward:.6f})")
        
        if reward == 0.0:
            print(f"   ❌ REWARD ZERO - Possível problema na normalização")
    
    return reward, info, done

# Aplicar monkey patch
BrutalMoneyReward.calculate_reward_and_info = debug_calculate_reward_and_info

print("🔍 DEBUG V3 BRUTAL ATIVADO - Monitor de reward runtime instalado")
print("   - Debug a cada 500 steps ou quando reward > 0.001")
print("   - Analisa PnL vs reward para detectar problemas")