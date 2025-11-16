#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE V3 BRUTAL REWARD BALANCE TEST
Advanced analysis of V3 Brutal reward system balance and component weights
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockTradingEnvironment:
    """Enhanced mock trading environment for comprehensive testing"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset to initial state"""
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.trades = []
        self.positions = []
        self.current_step = 0
        self.last_action = np.zeros(8)
        
    def apply_scenario(self, pnl_realized: float = 0, pnl_unrealized: float = 0, 
                      drawdown_pct: float = 0, trades_history: List = None,
                      positions_open: List = None, step: int = 0):
        """Apply a specific test scenario"""
        self.total_realized_pnl = pnl_realized
        self.total_unrealized_pnl = pnl_unrealized
        self.portfolio_value = self.initial_balance + pnl_realized + pnl_unrealized
        self.current_step = step
        
        if drawdown_pct > 0:
            # Calculate peak based on drawdown
            self.peak_portfolio_value = self.portfolio_value / (1 - drawdown_pct)
        else:
            self.peak_portfolio_value = max(self.portfolio_value, self.initial_balance)
            
        self.trades = trades_history or []
        self.positions = positions_open or []

def test_reward_component_breakdown():
    """Test detailed breakdown of reward components"""
    
    print("🔬 COMPREHENSIVE V3 BRUTAL REWARD COMPONENT ANALYSIS")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    env = MockTradingEnvironment(initial_balance=10000)
    
    # Define comprehensive test scenarios
    scenarios = [
        # Basic PnL scenarios
        {"name": "Break Even", "pnl_realized": 0, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Small Profit +1%", "pnl_realized": 100, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Medium Profit +3%", "pnl_realized": 300, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Large Profit +7%", "pnl_realized": 700, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Huge Profit +15%", "pnl_realized": 1500, "pnl_unrealized": 0, "drawdown_pct": 0},
        
        # Loss scenarios (testing pain multiplier)
        {"name": "Small Loss -2%", "pnl_realized": -200, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Medium Loss -4%", "pnl_realized": -400, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Large Loss -8%", "pnl_realized": -800, "pnl_unrealized": 0, "drawdown_pct": 0},
        {"name": "Huge Loss -15%", "pnl_realized": -1500, "pnl_unrealized": 0, "drawdown_pct": 0},
        
        # Mixed scenarios
        {"name": "Mixed: +5% Real, -2% Unreal", "pnl_realized": 500, "pnl_unrealized": -200, "drawdown_pct": 0},
        {"name": "Mixed: -3% Real, +2% Unreal", "pnl_realized": -300, "pnl_unrealized": 200, "drawdown_pct": 0},
        
        # Drawdown scenarios
        {"name": "10% Drawdown (no new trades)", "pnl_realized": 0, "pnl_unrealized": 0, "drawdown_pct": 0.10},
        {"name": "20% Drawdown (severe)", "pnl_realized": 0, "pnl_unrealized": 0, "drawdown_pct": 0.20},
        {"name": "30% Drawdown (critical)", "pnl_realized": 0, "pnl_unrealized": 0, "drawdown_pct": 0.30},
        
        # Unrealized scenarios
        {"name": "Large Unrealized +10%", "pnl_realized": 0, "pnl_unrealized": 1000, "drawdown_pct": 0},
        {"name": "Large Unrealized -10%", "pnl_realized": 0, "pnl_unrealized": -1000, "drawdown_pct": 0},
    ]
    
    results = []
    
    print("\n📊 DETAILED COMPONENT BREAKDOWN:")
    print("-" * 80)
    
    for scenario in scenarios:
        env.reset()
        scenario_name = scenario.pop('name')  # Remove name before passing to apply_scenario
        env.apply_scenario(**scenario)
        
        # Test with neutral action
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        try:
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            
            # Extract components
            pnl_reward = info.get('pnl_reward', 0.0)
            risk_reward = info.get('risk_reward', 0.0)
            shaping_reward = info.get('shaping_reward', 0.0)
            pure_pnl_component = info.get('pure_pnl_component', 0.0)
            proportional_shaping = info.get('proportional_shaping', 0.0)
            
            # Calculate actual PnL percentage
            total_pnl = scenario['pnl_realized'] + (scenario['pnl_unrealized'] * 0.5)
            pnl_percent = (total_pnl / 10000) * 100
            
            result = {
                'scenario': scenario_name,
                'pnl_percent': pnl_percent,
                'total_reward': reward,
                'pnl_reward': pnl_reward,
                'risk_reward': risk_reward,
                'shaping_reward': shaping_reward,
                'pure_pnl_component': pure_pnl_component,
                'proportional_shaping': proportional_shaping,
                'pain_applied': info.get('pain_applied', False),
                'portfolio_drawdown': info.get('portfolio_drawdown', 0.0),
                'done': done
            }
            
            results.append(result)
            
            # Display result
            print(f"\n🔸 {scenario_name}")
            print(f"   PnL: {pnl_percent:+6.2f}% | Total Reward: {reward:+8.4f}")
            print(f"   ├─ PnL Component: {pure_pnl_component:+8.4f} ({(abs(pure_pnl_component)/abs(reward)*100 if reward != 0 else 0):5.1f}%)")
            print(f"   ├─ Shaping Component: {proportional_shaping:+8.4f} ({(abs(proportional_shaping)/abs(reward)*100 if reward != 0 else 0):5.1f}%)")
            print(f"   ├─ Raw PnL Reward: {pnl_reward:+8.4f}")
            print(f"   ├─ Risk Penalty: {risk_reward:+8.4f}")
            print(f"   ├─ Pain Applied: {'YES' if result['pain_applied'] else 'NO'}")
            print(f"   └─ Episode Done: {'YES' if done else 'NO'}")
            
        except Exception as e:
            print(f"❌ Error in {scenario_name}: {str(e)}")
            
    return results

def analyze_pain_threshold_behavior(results):
    """Analyze the behavior of the pain threshold"""
    
    print(f"\n💥 PAIN THRESHOLD ANALYSIS")
    print("=" * 80)
    
    pain_cases = [r for r in results if r['pain_applied']]
    no_pain_cases = [r for r in results if not r['pain_applied'] and r['pnl_percent'] < 0]
    
    print(f"\nCases with PAIN applied: {len(pain_cases)}")
    print(f"Loss cases without PAIN: {len(no_pain_cases)}")
    
    if pain_cases:
        print(f"\n🔥 PAIN THRESHOLD BEHAVIOR:")
        print("PnL%      Total Reward    PnL Reward    Multiplier")
        print("-" * 55)
        
        for case in pain_cases:
            base_reward = case['pnl_percent'] * 0.05  # Base calculation
            actual_pnl_reward = case['pnl_reward']
            multiplier = abs(actual_pnl_reward / base_reward) if base_reward != 0 else 1.0
            
            print(f"{case['pnl_percent']:+6.2f}%    {case['total_reward']:+10.4f}    {case['pnl_reward']:+10.4f}    {multiplier:6.2f}x")
    
    # Find the pain threshold
    loss_cases = sorted([r for r in results if r['pnl_percent'] < 0], key=lambda x: x['pnl_percent'], reverse=True)
    
    pain_threshold = None
    for case in loss_cases:
        if case['pain_applied']:
            pain_threshold = abs(case['pnl_percent'])
            break
    
    if pain_threshold:
        print(f"\n🎯 IDENTIFIED PAIN THRESHOLD: ~{pain_threshold:.1f}%")
    else:
        print(f"\n⚠️ PAIN THRESHOLD NOT CLEARLY IDENTIFIED")

def test_reward_linearity_and_scaling():
    """Test the linearity and scaling of rewards"""
    
    print(f"\n📈 REWARD LINEARITY AND SCALING ANALYSIS")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    env = MockTradingEnvironment(initial_balance=10000)
    
    # Test range of PnL values
    pnl_range = np.arange(-15, 16, 1)  # -15% to +15% in 1% increments
    
    rewards = []
    pnl_rewards = []
    
    print(f"\nPnL%     Total Reward    PnL Reward    Linearity Check")
    print("-" * 60)
    
    for pnl_pct in pnl_range:
        env.reset()
        pnl_usd = (pnl_pct / 100) * 10000
        env.apply_scenario(pnl_realized=pnl_usd)
        
        action = np.zeros(8)
        
        try:
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            
            rewards.append(reward)
            pnl_rewards.append(info.get('pnl_reward', 0.0))
            
            # Check linearity (compare with expected linear reward)
            expected_linear = pnl_pct * 0.01  # Simple linear expectation
            linearity_ratio = reward / expected_linear if expected_linear != 0 else float('inf')
            
            print(f"{pnl_pct:+4.0f}%    {reward:+10.4f}    {info.get('pnl_reward', 0):+10.4f}    {linearity_ratio:8.2f}x")
            
        except Exception as e:
            print(f"{pnl_pct:+4.0f}%    ERROR: {str(e)}")
            rewards.append(0)
            pnl_rewards.append(0)
    
    # Calculate correlation
    valid_indices = [i for i, r in enumerate(rewards) if r != 0 or pnl_range[i] == 0]
    if len(valid_indices) > 2:
        valid_pnl = [pnl_range[i] for i in valid_indices]
        valid_rewards = [rewards[i] for i in valid_indices]
        
        correlation = np.corrcoef(valid_pnl, valid_rewards)[0, 1]
        print(f"\n📊 PnL-Reward Correlation: {correlation:.4f}")
        
        if correlation > 0.95:
            print("✅ EXCELLENT linearity")
        elif correlation > 0.85:
            print("✅ GOOD linearity")
        elif correlation > 0.70:
            print("⚠️ MODERATE linearity")
        else:
            print("❌ POOR linearity")

def test_action_sensitivity():
    """Test how sensitive rewards are to different actions"""
    
    print(f"\n🎮 ACTION SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    env = MockTradingEnvironment(initial_balance=10000)
    
    # Set up a scenario with some PnL
    env.reset()
    env.apply_scenario(pnl_realized=300)  # 3% profit scenario
    
    # Test different action types
    test_actions = {
        'Hold (all zeros)': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'Conservative buy': np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'Aggressive buy': np.array([0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'Conservative sell': np.array([0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]),
        'Aggressive sell': np.array([0.0, 0.0, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0]),
        'Mixed strategy': np.array([0.3, 0.2, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0]),
        'High activity': np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        'Random': np.random.rand(8)
    }
    
    print(f"\nTesting with 3% profit scenario:")
    print("Action Type          Total Reward    PnL Reward      Shaping")
    print("-" * 65)
    
    action_results = {}
    
    for action_name, action in test_actions.items():
        try:
            env.last_action = action  # Set last action for shaping calculation
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            
            action_results[action_name] = {
                'reward': reward,
                'pnl_reward': info.get('pnl_reward', 0.0),
                'shaping': info.get('proportional_shaping', 0.0)
            }
            
            print(f"{action_name:20} {reward:+10.4f}    {info.get('pnl_reward', 0):+10.4f}    {info.get('proportional_shaping', 0):+8.4f}")
            
        except Exception as e:
            print(f"{action_name:20} ERROR: {str(e)}")
    
    # Analyze variance in rewards due to actions
    rewards = [r['reward'] for r in action_results.values()]
    reward_variance = np.var(rewards)
    reward_range = max(rewards) - min(rewards)
    
    print(f"\n📊 ACTION SENSITIVITY METRICS:")
    print(f"   Reward Variance: {reward_variance:.8f}")
    print(f"   Reward Range: {reward_range:.6f}")
    print(f"   Coefficient of Variation: {np.std(rewards)/np.mean(rewards)*100:.2f}%")
    
    if reward_range < 0.001:
        print("   ⚠️ Very low action sensitivity - actions barely affect rewards")
    elif reward_range < 0.01:
        print("   ✅ Good action sensitivity - moderate effect from actions")
    else:
        print("   📈 High action sensitivity - actions significantly affect rewards")

def generate_balance_report(results):
    """Generate a comprehensive balance report"""
    
    print(f"\n📋 COMPREHENSIVE BALANCE REPORT")
    print("=" * 80)
    
    # Separate results by profit/loss
    profit_results = [r for r in results if r['pnl_percent'] > 0]
    loss_results = [r for r in results if r['pnl_percent'] < 0]
    breakeven_results = [r for r in results if r['pnl_percent'] == 0]
    
    print(f"\n📊 SCENARIO DISTRIBUTION:")
    print(f"   Profit scenarios: {len(profit_results)}")
    print(f"   Loss scenarios: {len(loss_results)}")
    print(f"   Breakeven scenarios: {len(breakeven_results)}")
    
    # Component dominance analysis
    print(f"\n🏆 COMPONENT DOMINANCE ANALYSIS:")
    
    total_reward_magnitude = sum(abs(r['total_reward']) for r in results if r['total_reward'] != 0)
    pnl_magnitude = sum(abs(r['pure_pnl_component']) for r in results if r['pure_pnl_component'] != 0)
    shaping_magnitude = sum(abs(r['proportional_shaping']) for r in results if r['proportional_shaping'] != 0)
    
    if total_reward_magnitude > 0:
        pnl_dominance = (pnl_magnitude / total_reward_magnitude) * 100
        shaping_dominance = (shaping_magnitude / total_reward_magnitude) * 100
        
        print(f"   PnL Component Dominance: {pnl_dominance:.1f}%")
        print(f"   Shaping Component Dominance: {shaping_dominance:.1f}%")
        
        if pnl_dominance > 80:
            print("   ✅ PnL properly dominates (>80%)")
        elif pnl_dominance > 60:
            print("   ⚠️ PnL moderately dominates (60-80%)")
        else:
            print("   ❌ PnL insufficiently dominates (<60%)")
    
    # Reward scaling analysis
    if profit_results and loss_results:
        avg_profit_reward = np.mean([r['total_reward'] for r in profit_results])
        avg_loss_reward = np.mean([r['total_reward'] for r in loss_results])
        
        print(f"\n⚖️ PROFIT/LOSS BALANCE:")
        print(f"   Average profit reward: {avg_profit_reward:+.4f}")
        print(f"   Average loss reward: {avg_loss_reward:+.4f}")
        print(f"   Loss/Profit ratio: {abs(avg_loss_reward/avg_profit_reward):.2f}x")
        
        if abs(avg_loss_reward/avg_profit_reward) > 1.2:
            print("   📈 Loss penalties are stronger (good for risk management)")
        elif abs(avg_loss_reward/avg_profit_reward) < 0.8:
            print("   ⚠️ Loss penalties are weaker than profit rewards")
        else:
            print("   ✅ Balanced profit/loss scaling")
    
    # Risk management effectiveness
    risk_cases = [r for r in results if r['portfolio_drawdown'] > 0]
    print(f"\n🚨 RISK MANAGEMENT ANALYSIS:")
    print(f"   Scenarios with drawdown: {len(risk_cases)}")
    
    if risk_cases:
        avg_drawdown = np.mean([r['portfolio_drawdown'] for r in risk_cases])
        avg_risk_penalty = np.mean([r['risk_reward'] for r in risk_cases])
        
        print(f"   Average drawdown in risk cases: {avg_drawdown:.1f}%")
        print(f"   Average risk penalty: {avg_risk_penalty:+.4f}")
        
    # Early termination analysis
    done_cases = [r for r in results if r['done']]
    print(f"\n🛑 EARLY TERMINATION:")
    print(f"   Scenarios triggering termination: {len(done_cases)}")
    
    if done_cases:
        min_termination_loss = min([r['pnl_percent'] for r in done_cases])
        print(f"   Minimum loss for termination: {min_termination_loss:.1f}%")
    
    # Overall balance score
    balance_score = 0
    max_score = 5
    
    # Score factors
    if pnl_dominance > 80:
        balance_score += 1
        
    if len(profit_results) > 0 and len(loss_results) > 0:
        balance_score += 1
        
    pain_cases = [r for r in results if r['pain_applied']]
    if pain_cases:
        balance_score += 1
        
    if total_reward_magnitude > 0:
        balance_score += 1
        
    # Check for reasonable reward ranges
    all_rewards = [r['total_reward'] for r in results]
    if all_rewards and abs(max(all_rewards)) < 2.0 and abs(min(all_rewards)) < 2.0:
        balance_score += 1
        
    print(f"\n🏆 OVERALL BALANCE SCORE: {balance_score}/{max_score}")
    
    if balance_score >= 4:
        print("   ✅ EXCELLENT balance")
    elif balance_score >= 3:
        print("   ✅ GOOD balance")
    elif balance_score >= 2:
        print("   ⚠️ MODERATE balance - some issues")
    else:
        print("   ❌ POOR balance - needs attention")
        
    return balance_score

def save_results_to_file(results):
    """Save test results to a JSON file for further analysis"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v3_brutal_balance_test_{timestamp}.json"
    
    # Prepare data for JSON serialization
    json_data = {
        'timestamp': timestamp,
        'test_type': 'V3 Brutal Reward Balance Test',
        'total_scenarios': len(results),
        'results': results
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {str(e)}")

def main():
    """Main test execution"""
    
    print("🧪 STARTING COMPREHENSIVE V3 BRUTAL BALANCE TEST")
    print("=" * 80)
    
    try:
        # Run all tests
        results = test_reward_component_breakdown()
        
        if results:
            analyze_pain_threshold_behavior(results)
            test_reward_linearity_and_scaling()
            test_action_sensitivity()
            balance_score = generate_balance_report(results)
            save_results_to_file(results)
            
            print(f"\n🎯 FINAL CONCLUSION:")
            print("=" * 80)
            
            if balance_score >= 4:
                print("✅ V3 Brutal reward system is WELL BALANCED")
                print("   The system properly emphasizes PnL over other components")
                print("   Pain thresholds are working as intended")
                print("   Ready for production use")
            elif balance_score >= 3:
                print("✅ V3 Brutal reward system has GOOD balance")
                print("   Minor issues may exist but system is functional")
                print("   Consider minor adjustments for optimization")
            else:
                print("⚠️ V3 Brutal reward system needs ATTENTION")
                print("   Significant balance issues detected")
                print("   Review component weights and thresholds")
                
        else:
            print("❌ No results obtained - check reward system implementation")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()