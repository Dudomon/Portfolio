#!/usr/bin/env python3
"""
🧪 TESTE: Verificar se micro-farming não recebe quality bonus
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator

class MockEnv:
    def __init__(self, trades=None):
        self.trades = trades or []
        self.current_step = 0
        self.balance = 1000
        self.realized_balance = 1000 + sum(t.get('pnl_usd', 0) for t in self.trades)
        self.portfolio_value = self.realized_balance
        self.initial_balance = 1000
        self.peak_portfolio = max(1000, self.portfolio_value)
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.current_positions = 0
        self.reward_history_size = 100
        self.recent_rewards = []

def test_micro_vs_quality():
    print("🧪 TESTE: Micro-farming vs Quality Trading")
    print("=" * 50)
    
    calc = BalancedDayTradingRewardCalculator()
    
    scenarios = {
        "micro_extreme": [{'pnl_usd': 0.5} for _ in range(200)],  # 200 x $0.50
        "micro_moderate": [{'pnl_usd': 2.0} for _ in range(100)], # 100 x $2.00
        "small_trades": [{'pnl_usd': 10.0} for _ in range(20)],   # 20 x $10 (borda)
        "quality_trades": [{'pnl_usd': 50.0} for _ in range(8)]   # 8 x $50
    }
    
    for name, trades in scenarios.items():
        env = MockEnv(trades)
        action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
        old_state = {'trades_count': 0}
        
        reward, info, _ = calc.calculate_reward_and_info(env, action, old_state)
        components = info.get('reward_components', {})
        
        total_pnl = sum(t['pnl_usd'] for t in trades)
        quality_component = components.get('trade_quality', 0)
        reward_per_dollar = reward / total_pnl if total_pnl > 0 else 0
        
        print(f"{name:15}: Total=${total_pnl:4.0f}, Quality={quality_component:.6f}, R/$ ratio={reward_per_dollar:.8f}")
    
    # Teste específico: micro-farming deve ter quality = 0
    micro_trades = [{'pnl_usd': 0.5} for _ in range(10)]
    env_micro = MockEnv(micro_trades)
    
    quality_score = calc._calculate_v3_quality_component(env_micro)
    print(f"\n🔍 Debug micro-farming quality score: {quality_score:.6f}")
    print(f"   Expected: 0.0 (no quality bonus for micro-trades)")

if __name__ == "__main__":
    test_micro_vs_quality()