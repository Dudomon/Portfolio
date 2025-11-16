#!/usr/bin/env python3
"""
🔍 DEBUG CONSISTENCY ESPECÍFICO - Testar cada componente individualmente
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

def debug_consistency():
    print("🔍 DEBUG COMPONENTES - Testando cada função individualmente")
    print("=" * 60)
    
    reward_system = create_balanced_daytrading_reward_system(1000.0)
    env = MockEnv()
    
    print(f"📊 Setup:")
    print(f"   Trades: {len(env.trades)}")
    
    # Testar cada componente individualmente
    print(f"\n🧪 Teste 1: _calculate_consistency_rewards")
    consistency = reward_system._calculate_consistency_rewards(env)
    print(f"   Resultado: {consistency}")
    
    print(f"\n🧪 Teste 2: _calculate_gaming_penalties")
    gaming = reward_system._calculate_gaming_penalties(env)
    print(f"   Resultado: {gaming}")
    
    print(f"\n🧪 Teste 3: Win rate dos trades")
    wins = sum(1 for t in env.trades if t.get('pnl_usd', 0) > 0)
    win_rate = wins / len(env.trades)
    optimal_win_rate = reward_system.optimal_win_rate
    print(f"   Wins: {wins}/{len(env.trades)} = {win_rate:.2f}")
    print(f"   Optimal: {optimal_win_rate}")
    print(f"   Atende critério: {win_rate >= optimal_win_rate}")
    
    print(f"\n🧪 Teste 4: Sharpe ratio simulado")
    pnls = [t.get('pnl_usd', 0) for t in env.trades]
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    if std_pnl > 0:
        pseudo_sharpe = mean_pnl / std_pnl
        print(f"   Mean PnL: {mean_pnl:.2f}")
        print(f"   Std PnL: {std_pnl:.2f}")
        print(f"   Pseudo Sharpe: {pseudo_sharpe:.2f}")
        print(f"   Atende critério (>0.5): {pseudo_sharpe > 0.5}")
    else:
        print(f"   Std PnL é 0, não pode calcular Sharpe")

if __name__ == "__main__":
    debug_consistency()