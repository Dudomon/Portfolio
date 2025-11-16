#!/usr/bin/env python3
"""
🔍 DEBUG REWARD FLOW MANUAL - Simular exatamente o que acontece
"""

import numpy as np
import sys
sys.path.append('.')

from trading_framework.rewards.reward_daytrade_v2 import create_balanced_daytrading_reward_system

class MockEnv:
    def __init__(self):
        self.trades = [
            {'pnl_usd': 10.0, 'duration_steps': 30, 'position_size': 0.01},
            {'pnl_usd': 5.0, 'duration_steps': 25, 'position_size': 0.01},
            {'pnl_usd': -3.0, 'duration_steps': 40, 'position_size': 0.01},
            {'pnl_usd': 8.0, 'duration_steps': 35, 'position_size': 0.01},
            {'pnl_usd': 2.0, 'duration_steps': 50, 'position_size': 0.01}
        ]
        self.current_balance = 1000.0
        self.peak_balance = 1000.0

def debug_manual_flow():
    print("🔍 DEBUG MANUAL FLOW - Simulando exatamente o fluxo do reward")
    print("=" * 60)
    
    reward_system = create_balanced_daytrading_reward_system(1000.0)
    env = MockEnv()
    
    # Configurar
    reward_system.step_count = 50
    old_state = {
        "portfolio_total_value": 1000.0,
        "current_drawdown": 0.0,
        "trades_count": 5
    }
    action = np.array([0, 0.5, 0, 0, 0, 0, 0, 0, 0])
    
    # SIMULAR O FLUXO MANUALMENTE
    print("🔍 Simulando o fluxo step-by-step:")
    
    # Step 1: Early exit check
    old_trades_count = old_state.get('trades_count', 0)
    current_trades_count = len(getattr(env, 'trades', []))
    
    print(f"1. Early exit check:")
    print(f"   old_trades: {old_trades_count}, current_trades: {current_trades_count}")
    print(f"   step_count % 50: {reward_system.step_count % 50}")
    print(f"   Early exit condition: {current_trades_count == old_trades_count and reward_system.step_count % 50 != 0}")
    
    if current_trades_count == old_trades_count and reward_system.step_count % 50 != 0:
        print("   → EARLY EXIT!")
        return
    else:
        print("   → Continue to main logic")
    
    # Step 2: Initialize
    reward = 0.0
    components = {}
    print(f"\n2. Initialize: reward = {reward}")
    
    # Step 3: New trade check
    print(f"\n3. New trade check:")
    print(f"   current_trades_count > old_trades_count: {current_trades_count > old_trades_count}")
    
    if current_trades_count > old_trades_count:
        print("   → Calculate trade reward")
        # Não vamos simular isso aqui
    else:
        print("   → Skip trade reward (no new trades)")
    
    # Step 4: Periodic consistency
    print(f"\n4. Periodic consistency check:")
    print(f"   step_count % 50 == 0: {reward_system.step_count % 50 == 0}")
    
    if reward_system.step_count % 50 == 0:
        print("   → Calculate consistency reward")
        consistency_reward = reward_system._calculate_consistency_rewards(env)
        reward += consistency_reward
        components['consistency'] = consistency_reward
        print(f"   → consistency_reward = {consistency_reward}")
        print(f"   → total reward = {reward}")
    else:
        print("   → Skip consistency")
    
    # Step 5: Gaming penalties
    print(f"\n5. Gaming penalties:")
    gaming_penalty = reward_system._calculate_gaming_penalties(env)
    reward += gaming_penalty
    components['gaming_penalty'] = gaming_penalty
    print(f"   → gaming_penalty = {gaming_penalty}")
    print(f"   → total reward = {reward}")
    
    # Step 6: Normalization
    print(f"\n6. Normalization:")
    normalized_reward = reward_system._normalize_reward(reward)
    print(f"   → raw reward = {reward}")
    print(f"   → normalized reward = {normalized_reward}")
    
    print(f"\n📊 RESULTADO FINAL:")
    print(f"   Reward: {normalized_reward}")
    print(f"   Components: {components}")

if __name__ == "__main__":
    debug_manual_flow()