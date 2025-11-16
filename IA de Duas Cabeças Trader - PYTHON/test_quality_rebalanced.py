#!/usr/bin/env python3
"""
🧪 TESTE RÁPIDO: Quality component rebalanceado
Verificar se quality não domina mais
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

def test_component_balance():
    print("🧪 TESTE: Balanceamento com Quality Reabilitado")
    print("=" * 50)
    
    calc = BalancedDayTradingRewardCalculator()
    
    # Cenário padrão
    trades = [{'pnl_usd': 40.0} for _ in range(8)]
    env = MockEnv(trades)
    action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
    old_state = {'trades_count': 0}
    
    reward, info, _ = calc.calculate_reward_and_info(env, action, old_state)
    components = info.get('reward_components', {})
    
    print("COMPONENTES:")
    total_abs = 0
    for name, value in components.items():
        if abs(value) > 0.001:
            print(f"  {name:20}: {value:.6f}")
            total_abs += abs(value)
    
    print(f"\nTOTAL ABSOLUTE: {total_abs:.6f}")
    
    # Verificar balanceamento
    pnl_pct = abs(components.get('pnl', 0)) / total_abs * 100 if total_abs > 0 else 0
    quality_pct = abs(components.get('trade_quality', 0)) / total_abs * 100 if total_abs > 0 else 0
    
    print(f"\nBALANCEAMENTO:")
    print(f"  PnL: {pnl_pct:.1f}%")
    print(f"  Quality: {quality_pct:.1f}%")
    
    if pnl_pct > 50:
        print(f"  ✅ PnL is dominant ({pnl_pct:.1f}%)")
    else:
        print(f"  ❌ PnL not dominant ({pnl_pct:.1f}%)")
        
    if quality_pct < 30:
        print(f"  ✅ Quality balanced ({quality_pct:.1f}%)")
    else:
        print(f"  ⚠️ Quality too high ({quality_pct:.1f}%)")

if __name__ == "__main__":
    test_component_balance()