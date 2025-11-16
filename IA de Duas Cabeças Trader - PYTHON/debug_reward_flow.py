#!/usr/bin/env python3
"""
🔍 DEBUG REWARD FLOW - Rastrear exatamente onde o reward está sendo perdido
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

def debug_reward_flow():
    print("🔍 DEBUG REWARD FLOW - Rastreando onde reward é perdido")
    print("=" * 60)
    
    reward_system = create_balanced_daytrading_reward_system(1000.0)
    env = MockEnv()
    
    # Configurar para step múltiplo de 50
    reward_system.step_count = 50
    
    old_state = {
        "portfolio_total_value": 1000.0,
        "current_drawdown": 0.0,
        "trades_count": 5  # Mesmo número de trades
    }
    
    action = np.array([0, 0.5, 0, 0, 0, 0, 0, 0, 0])
    
    print(f"📊 Configuração:")
    print(f"   Trades no env: {len(env.trades)}")
    print(f"   Trades no old_state: {old_state['trades_count']}")
    print(f"   Step count: {reward_system.step_count}")
    print(f"   Step % 50 == 0: {reward_system.step_count % 50 == 0}")
    
    # Verificar condições do early exit
    old_trades_count = old_state.get('trades_count', 0)
    current_trades_count = len(getattr(env, 'trades', []))
    
    print(f"\n🔍 Verificação Early Exit:")
    print(f"   old_trades_count: {old_trades_count}")
    print(f"   current_trades_count: {current_trades_count}")
    print(f"   trades iguais: {current_trades_count == old_trades_count}")
    print(f"   step % 50 != 0: {reward_system.step_count % 50 != 0}")
    print(f"   Early exit condition: {current_trades_count == old_trades_count and reward_system.step_count % 50 != 0}")
    
    reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
    
    print(f"\n📊 Resultado:")
    print(f"   Reward: {reward}")
    print(f"   Components: {info.get('components', {})}")
    print(f"   Info keys: {list(info.keys())}")

if __name__ == "__main__":
    debug_reward_flow()