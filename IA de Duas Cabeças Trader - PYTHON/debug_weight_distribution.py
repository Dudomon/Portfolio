#!/usr/bin/env python3
"""
🔍 DEBUG WEIGHT DISTRIBUTION
Analyze why the weight distribution is not working as expected
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockTradingEnvironment:
    """Simple mock environment for debugging"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.trades = []
        self.positions = []
        self.current_step = 0
        self.last_action = np.zeros(8)
        
    def set_scenario(self, pnl_realized: float = 0, pnl_unrealized: float = 0, action: np.ndarray = None):
        self.total_realized_pnl = pnl_realized
        self.total_unrealized_pnl = pnl_unrealized
        self.portfolio_value = self.initial_balance + pnl_realized + pnl_unrealized
        self.peak_portfolio_value = max(self.portfolio_value, self.initial_balance)
        
        if action is not None:
            self.last_action = action

def debug_single_scenario():
    """Debug a single scenario in detail"""
    
    print("🔍 DETAILED WEIGHT DISTRIBUTION DEBUG")
    print("=" * 60)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    env = MockTradingEnvironment(initial_balance=10000)
    
    # Test with 5% profit scenario
    env.reset()
    env.set_scenario(pnl_realized=500)
    action = np.array([0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Decisive action
    
    print(f"Test Scenario: +5% profit ({500} USD)")
    print(f"Action: Decisive buy")
    print("-" * 60)
    
    # Step through calculation manually
    reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
    
    print(f"📊 CALCULATION BREAKDOWN:")
    print(f"   PnL Reward (raw): {info.get('pnl_reward', 0):+.6f}")
    print(f"   Raw Shaping: {info.get('raw_shaping', 0):+.6f}")
    print(f"   Scaled Shaping: {info.get('proportional_shaping', 0):+.6f}")
    print(f"   Pure PnL Component: {info.get('pure_pnl_component', 0):+.6f}")
    print(f"   Shaping Component: {info.get('shaping_component', 0):+.6f}")
    print(f"   Total Reward: {info.get('total_reward', 0):+.6f}")
    
    print(f"\n🧮 MANUAL PERCENTAGE CALCULATION:")
    pnl_comp = info.get('pure_pnl_component', 0)
    shap_comp = info.get('shaping_component', 0)
    total = pnl_comp + shap_comp
    
    if abs(total) > 1e-8:
        pnl_pct_manual = (abs(pnl_comp) / abs(total)) * 100
        shap_pct_manual = (abs(shap_comp) / abs(total)) * 100
        
        print(f"   |PnL Component| = {abs(pnl_comp):+.6f}")
        print(f"   |Shaping Component| = {abs(shap_comp):+.6f}")
        print(f"   |Total| = {abs(total):+.6f}")
        print(f"   PnL Percentage = {abs(pnl_comp):+.6f} / {abs(total):+.6f} * 100 = {pnl_pct_manual:.2f}%")
        print(f"   Shaping Percentage = {abs(shap_comp):+.6f} / {abs(total):+.6f} * 100 = {shap_pct_manual:.2f}%")
        
        print(f"\n🎯 TARGET vs ACTUAL:")
        print(f"   Target PnL: 85.0%, Actual: {pnl_pct_manual:.1f}% (Error: {abs(pnl_pct_manual - 85.0):.1f}%)")
        print(f"   Target Shaping: 15.0%, Actual: {shap_pct_manual:.1f}% (Error: {abs(shap_pct_manual - 15.0):.1f}%)")
        
        # Analyze why the distribution is wrong
        print(f"\n🔧 DISTRIBUTION ANALYSIS:")
        expected_shaping_magnitude = abs(pnl_comp) * (15.0 / 85.0)  # Should be ~17.6% of PnL
        actual_shaping_magnitude = abs(shap_comp)
        
        print(f"   Expected |Shaping|: {expected_shaping_magnitude:.6f}")
        print(f"   Actual |Shaping|: {actual_shaping_magnitude:.6f}")
        print(f"   Shaping Scale Factor Needed: {expected_shaping_magnitude / (actual_shaping_magnitude + 1e-10):.2f}")
        
        # Test theoretical correct distribution
        print(f"\n✅ THEORETICAL CORRECT DISTRIBUTION:")
        theoretical_pnl = pnl_comp
        theoretical_shaping = expected_shaping_magnitude * (1.0 if shap_comp >= 0 else -1.0)
        theoretical_total = abs(theoretical_pnl) + abs(theoretical_shaping)
        
        if theoretical_total > 1e-8:
            theo_pnl_pct = (abs(theoretical_pnl) / theoretical_total) * 100
            theo_shap_pct = (abs(theoretical_shaping) / theoretical_total) * 100
            
            print(f"   Theoretical PnL: {theoretical_pnl:.6f} ({theo_pnl_pct:.1f}%)")
            print(f"   Theoretical Shaping: {theoretical_shaping:.6f} ({theo_shap_pct:.1f}%)")
            print(f"   Theoretical Total: {theoretical_pnl + theoretical_shaping:.6f}")
    
    else:
        print("   Total reward too small for meaningful percentage calculation")

def test_mathematical_approach():
    """Test the mathematical approach for correct distribution"""
    
    print(f"\n🧮 TESTING MATHEMATICAL APPROACH")
    print("=" * 60)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    env = MockTradingEnvironment(initial_balance=10000)
    
    # Test different PnL levels
    test_pnls = [100, 500, 1000, -300, -800]  # Various PnL amounts
    
    print(f"PnL (USD)   PnL%    Expected 85%/15%     Actual Result")
    print("-" * 60)
    
    for pnl_usd in test_pnls:
        env.reset()
        env.set_scenario(pnl_realized=pnl_usd)
        action = np.zeros(8)
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        pnl_comp = info.get('pure_pnl_component', 0)
        shap_comp = info.get('shaping_component', 0)
        
        # Calculate what 85%/15% should look like
        pnl_pct = (pnl_usd / 10000) * 100
        
        # Expected components for perfect 85%/15% split
        if abs(pnl_comp + shap_comp) > 1e-8:
            actual_pnl_pct = (abs(pnl_comp) / abs(pnl_comp + shap_comp)) * 100
            actual_shap_pct = (abs(shap_comp) / abs(pnl_comp + shap_comp)) * 100
            
            print(f"{pnl_usd:+8.0f} {pnl_pct:+6.1f}%    85% / 15%           {actual_pnl_pct:5.1f}% / {actual_shap_pct:4.1f}%")
        else:
            print(f"{pnl_usd:+8.0f} {pnl_pct:+6.1f}%    85% / 15%           0.0% / 0.0%")

if __name__ == "__main__":
    debug_single_scenario()
    test_mathematical_approach()