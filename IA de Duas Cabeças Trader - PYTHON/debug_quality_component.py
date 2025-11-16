#!/usr/bin/env python3
"""
🔍 DEBUG: Quality component values
Identificar por que está retornando valores altos
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator

class MockEnv:
    def __init__(self, trades=None):
        self.trades = trades or []

def debug_quality():
    print("🔍 DEBUG: Quality Component Calculation")
    print("=" * 50)
    
    calc = BalancedDayTradingRewardCalculator()
    
    # Cenário padrão
    trades = [{'pnl_usd': 40.0} for _ in range(8)]
    env = MockEnv(trades)
    
    # Debug manual do cálculo
    recent_trades = trades[-5:] if len(trades) >= 5 else trades
    pnls = [trade.get('pnl_usd', 0) for trade in recent_trades]
    avg_pnl = np.mean([abs(p) for p in pnls])
    profitable_trades = len([p for p in pnls if p > 0])
    win_rate = profitable_trades / len(pnls)
    
    print(f"Recent trades: {len(recent_trades)}")
    print(f"PnLs: {pnls}")
    print(f"Avg PnL (abs): {avg_pnl}")
    print(f"Win rate: {win_rate}")
    
    quality_score = 0.0
    
    # 1. Size component
    if avg_pnl > 5:
        size_component = min(0.3, avg_pnl / 200)
        quality_score += size_component
        print(f"Size component: {size_component:.6f} (avg_pnl={avg_pnl:.1f})")
    
    # 2. Win rate component
    if win_rate > 0.4:
        wr_component = min(0.15, (win_rate - 0.4) * 0.5)
        quality_score += wr_component
        print(f"WR component: {wr_component:.6f} (win_rate={win_rate:.1f})")
    
    # 3. Consistency component
    if len(pnls) > 3 and np.std(pnls) > 0:
        mean_pnl = np.mean(pnls)
        if mean_pnl > 0:
            consistency_ratio = mean_pnl / np.std(pnls)
            if consistency_ratio > 1:
                consistency_component = min(0.05, consistency_ratio * 0.02)
                quality_score += consistency_component
                print(f"Consistency component: {consistency_component:.6f} (ratio={consistency_ratio:.1f})")
    
    final_score = min(0.5, quality_score)
    
    print(f"\nFinal quality score: {final_score:.6f}")
    
    # Testar o método real
    real_score = calc._calculate_v3_quality_component(env)
    print(f"Real method score: {real_score:.6f}")
    
    # Calcular com peso
    weight = calc.base_weights.get('trade_size_quality', 0.0)
    weighted_score = real_score * weight
    print(f"Weight: {weight}")
    print(f"Weighted score: {weighted_score:.6f}")

if __name__ == "__main__":
    debug_quality()