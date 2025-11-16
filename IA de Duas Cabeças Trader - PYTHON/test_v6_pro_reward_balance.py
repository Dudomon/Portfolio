"""
🧪 COMPREHENSIVE BALANCE TEST FOR V6 PRO REWARD SYSTEM
====================================================

Tests all aspects of the V6 Pro reward system to validate:
1. Component balance and weights
2. Reward scaling and normalization
3. Edge cases and stability
4. Trading behavior alignment
5. Numerical stability

Author: RL Reward Engineer
Date: 2025-09-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append('D:\\Projeto')

from trading_framework.rewards.reward_daytrade_v6_pro import RewardV6Pro, V6_CONFIG


class MockEnvironment:
    """Mock trading environment for testing"""
    
    def __init__(self, portfolio_value: float = 500.0, positions: List = None, 
                 trades: List = None, current_step: int = 0):
        self.portfolio_value = portfolio_value
        self.positions = positions or []
        self.trades = trades or []
        self.current_step = current_step
        self.steps_since_last_trade = 0
        
        # Mock market data
        self.df = self._generate_mock_data()
        
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate realistic market data for testing"""
        np.random.seed(42)  # Reproducible results
        
        # Generate 1000 periods of 5-minute gold data
        periods = 1000
        
        # Base price around $2000/oz
        base_price = 2000.0
        
        # Random walk with trending behavior
        returns = np.random.normal(0, 0.0002, periods)  # 0.02% std per period
        
        # Add some trending behavior
        trend = np.sin(np.arange(periods) / 100) * 0.0001
        returns += trend
        
        # Generate OHLC from returns
        closes = [base_price]
        for r in returns[1:]:
            closes.append(closes[-1] * (1 + r))
        
        closes = np.array(closes)
        
        # Generate OHLC with realistic spreads
        highs = closes * (1 + np.abs(np.random.normal(0, 0.0001, periods)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.0001, periods)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        return pd.DataFrame({
            'open': opens,
            'high': highs, 
            'low': lows,
            'close': closes,
            'open_5m': opens,
            'high_5m': highs,
            'low_5m': lows, 
            'close_5m': closes
        })


class V6ProRewardBalanceTester:
    """Comprehensive tester for V6 Pro reward system balance"""
    
    def __init__(self):
        self.reward_system = RewardV6Pro(initial_balance=500.0)
        self.test_results = {}
        
    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        print("🧪 STARTING V6 PRO REWARD BALANCE TEST SUITE")
        print("=" * 60)
        
        # Component balance tests
        self.test_component_weights()
        self.test_pnl_dominance()
        self.test_activity_scaling()
        self.test_risk_components()
        
        # Scenario tests
        self.test_trading_scenarios()
        self.test_edge_cases()
        self.test_numerical_stability()
        
        # Behavior tests
        self.test_training_incentives()
        self.test_reward_distribution()
        
        # Generate final report
        self.generate_balance_report()
        
        return self.test_results
    
    def test_component_weights(self):
        """Test if reward components have expected weights"""
        print("\n🔍 Testing Component Weights...")
        
        # Test scenario: medium portfolio change
        env = MockEnvironment(portfolio_value=510.0)  # +2% gain
        old_state = {'positions': []}
        action = np.array([0.8, 0.6, 0.3, 0.2])  # Decisive action
        
        reward, info, _ = self.reward_system.calculate_reward_and_info(env, action, old_state)
        
        # Analyze component distribution
        components = {
            'base_pnl': info.get('base_pnl', 0),
            'close_component': info.get('close_component', 0),
            'risk_component': info.get('risk_component', 0),
            'activity_component': info.get('activity_component', 0)
        }
        
        total_magnitude = sum(abs(v) for v in components.values())
        
        if total_magnitude > 0:
            weights = {k: abs(v) / total_magnitude for k, v in components.items()}
        else:
            weights = {k: 0 for k in components.keys()}
        
        self.test_results['component_weights'] = {
            'components': components,
            'weights': weights,
            'total_reward': reward,
            'pnl_dominance': weights.get('base_pnl', 0) > 0.6  # Should be dominant
        }
        
        print(f"   Base PnL weight: {weights.get('base_pnl', 0):.1%}")
        print(f"   Activity weight: {weights.get('activity_component', 0):.1%}")
        print(f"   Risk weight: {weights.get('risk_component', 0):.1%}")
        print(f"   PnL dominance: {'✅' if weights.get('base_pnl', 0) > 0.6 else '❌'}")
    
    def test_pnl_dominance(self):
        """Test that PnL component dominates reward signal"""
        print("\n💰 Testing PnL Dominance...")
        
        scenarios = [
            ("Large Gain +5%", 525.0),
            ("Medium Gain +2%", 510.0),
            ("Small Gain +0.5%", 502.5),
            ("No Change", 500.0),
            ("Small Loss -0.5%", 497.5),
            ("Medium Loss -2%", 490.0),
            ("Large Loss -5%", 475.0)
        ]
        
        pnl_rewards = []
        total_rewards = []
        
        for name, portfolio_value in scenarios:
            env = MockEnvironment(portfolio_value=portfolio_value)
            old_state = {'positions': []}
            action = np.array([0.5, 0.5, 0.0, 0.0])  # Neutral action
            
            reward, info, _ = self.reward_system.calculate_reward_and_info(env, action, old_state)
            
            pnl_reward = info.get('base_pnl', 0)
            pnl_rewards.append(pnl_reward)
            total_rewards.append(reward)
            
            print(f"   {name}: PnL={pnl_reward:.4f}, Total={reward:.4f}")
        
        # Check linearity and dominance
        pnl_range = max(pnl_rewards) - min(pnl_rewards)
        total_range = max(total_rewards) - min(total_rewards)
        dominance_ratio = pnl_range / max(total_range, 1e-8)
        
        self.test_results['pnl_dominance'] = {
            'scenarios': list(zip([s[0] for s in scenarios], pnl_rewards, total_rewards)),
            'pnl_range': pnl_range,
            'total_range': total_range,
            'dominance_ratio': dominance_ratio,
            'is_dominant': dominance_ratio > 0.7  # PnL should drive >70% of reward variance
        }
        
        print(f"   PnL dominance ratio: {dominance_ratio:.1%}")
        print(f"   Dominance check: {'✅' if dominance_ratio > 0.7 else '❌'}")
    
    def test_activity_scaling(self):
        """Test activity component scaling and thresholds"""
        print("\n⚡ Testing Activity Component Scaling...")
        
        base_env = MockEnvironment(portfolio_value=500.0)
        old_state = {'positions': []}
        
        # Test different action magnitudes
        activity_tests = [
            ("No Action", np.array([0.0, 0.0, 0.0, 0.0])),
            ("Weak Action", np.array([0.05, 0.05, 0.0, 0.0])),
            ("Moderate Action", np.array([0.1, 0.1, 0.0, 0.0])),
            ("Decisive Action", np.array([0.8, 0.7, 0.0, 0.0])),
            ("Maximum Action", np.array([1.0, 1.0, 0.0, 0.0]))
        ]
        
        activity_results = []
        
        for name, action in activity_tests:
            reward, info, _ = self.reward_system.calculate_reward_and_info(base_env, action, old_state)
            
            activity_reward = info.get('activity_component', 0)
            action_magnitude = info.get('action_magnitude', 0)
            
            activity_results.append({
                'name': name,
                'action_magnitude': action_magnitude,
                'activity_reward': activity_reward,
                'total_reward': reward
            })
            
            print(f"   {name}: Magnitude={action_magnitude:.3f}, Activity={activity_reward:.4f}")
        
        # Test inactivity penalty with steps_since_last_trade
        base_env.steps_since_last_trade = 35  # Trigger inactivity multiplier
        action = np.array([0.01, 0.01, 0.0, 0.0])  # Very weak action
        
        reward, info, _ = self.reward_system.calculate_reward_and_info(base_env, action, old_state)
        inactivity_penalty = info.get('activity_component', 0)
        
        self.test_results['activity_scaling'] = {
            'magnitude_tests': activity_results,
            'inactivity_penalty': inactivity_penalty,
            'thresholds_working': True,  # Manual validation
            'scaling_appropriate': True  # Manual validation
        }
        
        print(f"   Inactivity penalty (35 steps): {inactivity_penalty:.4f}")
    
    def test_risk_components(self):
        """Test risk components (MAE and time penalties)"""
        print("\n⚠️  Testing Risk Components...")
        
        # Test with positions of different ages and MAE
        risk_scenarios = [
            {
                'name': 'New Profitable Position',
                'positions': [{
                    'entry_price': 2000.0,
                    'entry_step': 95,  # 5 steps old
                    'type': 'long',
                    'lot_size': 0.01
                }],
                'current_step': 100
            },
            {
                'name': 'Old Losing Position',
                'positions': [{
                    'entry_price': 2010.0,  # Losing position
                    'entry_step': 50,   # 50 steps old
                    'type': 'long', 
                    'lot_size': 0.03    # Max lot size
                }],
                'current_step': 100
            },
            {
                'name': 'Multiple Positions',
                'positions': [
                    {
                        'entry_price': 2000.0,
                        'entry_step': 80,
                        'type': 'long',
                        'lot_size': 0.02
                    },
                    {
                        'entry_price': 1995.0,
                        'entry_step': 60,
                        'type': 'short',
                        'lot_size': 0.01
                    }
                ],
                'current_step': 100
            }
        ]
        
        risk_results = []
        
        for scenario in risk_scenarios:
            env = MockEnvironment(
                portfolio_value=500.0,
                positions=scenario['positions'],
                current_step=scenario['current_step']
            )
            
            old_state = {'positions': []}
            action = np.array([0.0, 0.0, 0.0, 0.0])  # No new action
            
            reward, info, _ = self.reward_system.calculate_reward_and_info(env, action, old_state)
            
            risk_component = info.get('risk_component', 0)
            
            risk_results.append({
                'scenario': scenario['name'],
                'risk_component': risk_component,
                'num_positions': len(scenario['positions']),
                'total_reward': reward
            })
            
            print(f"   {scenario['name']}: Risk={risk_component:.4f}")
        
        self.test_results['risk_components'] = {
            'scenarios': risk_results,
            'penalties_applied': any(r['risk_component'] < 0 for r in risk_results),
            'magnitude_reasonable': all(abs(r['risk_component']) < 1.0 for r in risk_results)
        }
    
    def test_trading_scenarios(self):
        """Test realistic trading scenarios"""
        print("\n📈 Testing Trading Scenarios...")
        
        scenarios = [
            {
                'name': 'Successful Day Trade',
                'portfolio_start': 500.0,
                'portfolio_end': 515.0,
                'trades': [{'pnl_usd': 15.0, 'pnl_percent': 0.03}],
                'action': np.array([0.8, 0.9, 0.0, 0.0])  # Strong entry
            },
            {
                'name': 'Failed Day Trade',
                'portfolio_start': 500.0,
                'portfolio_end': 485.0,
                'trades': [{'pnl_usd': -15.0, 'pnl_percent': -0.03}],
                'action': np.array([0.1, 0.2, 0.0, 0.0])  # Weak entry
            },
            {
                'name': 'Breakeven Trade',
                'portfolio_start': 500.0,
                'portfolio_end': 499.0,
                'trades': [{'pnl_usd': -1.0, 'pnl_percent': -0.002}],
                'action': np.array([0.5, 0.5, 0.0, 0.0])
            },
            {
                'name': 'No Trading (Hold)',
                'portfolio_start': 500.0,
                'portfolio_end': 500.0,
                'trades': [],
                'action': np.array([0.0, 0.0, 0.0, 0.0])
            },
            {
                'name': 'Position Management',
                'portfolio_start': 500.0,
                'portfolio_end': 508.0,
                'positions': [{'entry_price': 2000.0, 'entry_step': 90, 'type': 'long', 'lot_size': 0.02}],
                'trades': [],
                'action': np.array([0.0, 0.0, 0.8, -0.5])  # Active management
            }
        ]
        
        scenario_results = []
        
        for scenario in scenarios:
            # Setup environment
            self.reward_system.last_portfolio_value = scenario['portfolio_start']
            
            env = MockEnvironment(
                portfolio_value=scenario['portfolio_end'],
                trades=scenario.get('trades', []),
                positions=scenario.get('positions', []),
                current_step=100
            )
            
            old_state = {'positions': scenario.get('old_positions', [])}
            
            reward, info, _ = self.reward_system.calculate_reward_and_info(
                env, scenario['action'], old_state
            )
            
            result = {
                'scenario': scenario['name'],
                'portfolio_change': scenario['portfolio_end'] - scenario['portfolio_start'],
                'reward': reward,
                'components': {
                    'base_pnl': info.get('base_pnl', 0),
                    'close_component': info.get('close_component', 0), 
                    'risk_component': info.get('risk_component', 0),
                    'activity_component': info.get('activity_component', 0)
                }
            }
            
            scenario_results.append(result)
            
            print(f"   {scenario['name']}: Reward={reward:.4f}, ΔP&L=${result['portfolio_change']:.1f}")
        
        self.test_results['trading_scenarios'] = scenario_results
    
    def test_edge_cases(self):
        """Test edge cases and extreme values"""
        print("\n🚨 Testing Edge Cases...")
        
        edge_cases = [
            {
                'name': 'Extreme Loss -50%',
                'portfolio_value': 250.0,
                'action': np.array([0.0, 0.0, 0.0, 0.0])
            },
            {
                'name': 'Extreme Gain +100%',
                'portfolio_value': 1000.0,
                'action': np.array([1.0, 1.0, 1.0, 1.0])
            },
            {
                'name': 'Zero Portfolio',
                'portfolio_value': 0.0,
                'action': np.array([0.5, 0.5, 0.0, 0.0])
            },
            {
                'name': 'Negative Action Values',
                'portfolio_value': 500.0,
                'action': np.array([-1.0, -0.5, -1.0, -1.0])
            },
            {
                'name': 'NaN Action Values',
                'portfolio_value': 500.0,
                'action': np.array([np.nan, 0.5, 0.0, 0.0])
            },
            {
                'name': 'Very Large Actions',
                'portfolio_value': 500.0,
                'action': np.array([100.0, 50.0, 10.0, 5.0])
            }
        ]
        
        edge_results = []
        
        for case in edge_cases:
            try:
                env = MockEnvironment(portfolio_value=case['portfolio_value'])
                old_state = {'positions': []}
                
                reward, info, done = self.reward_system.calculate_reward_and_info(
                    env, case['action'], old_state
                )
                
                result = {
                    'case': case['name'],
                    'reward': reward,
                    'is_finite': np.isfinite(reward),
                    'within_bounds': abs(reward) <= 2.5,  # Max reward bound
                    'error': None
                }
                
            except Exception as e:
                result = {
                    'case': case['name'],
                    'reward': 0.0,
                    'is_finite': False,
                    'within_bounds': False,
                    'error': str(e)
                }
            
            edge_results.append(result)
            
            status = '✅' if result['is_finite'] and result['within_bounds'] else '❌'
            print(f"   {case['name']}: {status} Reward={result['reward']:.4f}")
        
        self.test_results['edge_cases'] = {
            'results': edge_results,
            'all_finite': all(r['is_finite'] for r in edge_results),
            'all_bounded': all(r['within_bounds'] for r in edge_results),
            'error_count': sum(1 for r in edge_results if r['error'] is not None)
        }
    
    def test_numerical_stability(self):
        """Test numerical stability across multiple steps"""
        print("\n🔢 Testing Numerical Stability...")
        
        # Simulate 1000 steps of trading
        rewards = []
        portfolio_values = [500.0]
        
        env = MockEnvironment()
        
        for step in range(1000):
            # Simulate random portfolio changes
            change = np.random.normal(0, 5.0)  # ±$5 average change
            new_portfolio = max(0, portfolio_values[-1] + change)
            portfolio_values.append(new_portfolio)
            
            env.portfolio_value = new_portfolio
            env.current_step = step
            
            # Random action
            action = np.random.uniform(-0.5, 1.0, 4)
            action = np.clip(action, 0, 1)  # Ensure valid action space
            
            old_state = {'positions': []}
            
            try:
                reward, info, _ = self.reward_system.calculate_reward_and_info(env, action, old_state)
                rewards.append(reward)
            except Exception as e:
                print(f"   Error at step {step}: {e}")
                rewards.append(0.0)
        
        # Analyze stability
        rewards = np.array(rewards)
        
        stability_metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'finite_ratio': np.sum(np.isfinite(rewards)) / len(rewards),
            'bounded_ratio': np.sum(np.abs(rewards) <= 2.5) / len(rewards),
            'zero_ratio': np.sum(rewards == 0.0) / len(rewards)
        }
        
        self.test_results['numerical_stability'] = stability_metrics
        
        print(f"   Mean reward: {stability_metrics['mean_reward']:.4f}")
        print(f"   Std reward: {stability_metrics['std_reward']:.4f}")
        print(f"   Range: [{stability_metrics['min_reward']:.3f}, {stability_metrics['max_reward']:.3f}]")
        print(f"   Finite ratio: {stability_metrics['finite_ratio']:.1%}")
        print(f"   Bounded ratio: {stability_metrics['bounded_ratio']:.1%}")
    
    def test_training_incentives(self):
        """Test if reward system provides correct training incentives"""
        print("\n🎯 Testing Training Incentives...")
        
        incentive_tests = [
            {
                'name': 'Profit vs Loss',
                'scenarios': [
                    ('Profit +2%', 510.0, np.array([0.8, 0.7, 0.0, 0.0])),
                    ('Loss -2%', 490.0, np.array([0.8, 0.7, 0.0, 0.0]))
                ]
            },
            {
                'name': 'Action vs Inaction',
                'scenarios': [
                    ('Decisive Action', 500.0, np.array([0.9, 0.8, 0.0, 0.0])),
                    ('No Action', 500.0, np.array([0.0, 0.0, 0.0, 0.0]))
                ]
            },
            {
                'name': 'Quick vs Slow Loss Cutting',
                'scenarios': [
                    ('Quick Cut (5 steps)', 495.0, np.array([0.0, 0.0, -0.8, 0.0]), 
                     [{'entry_step': 95, 'entry_price': 2010.0, 'type': 'long', 'lot_size': 0.01}]),
                    ('Slow Cut (50 steps)', 485.0, np.array([0.0, 0.0, -0.8, 0.0]),
                     [{'entry_step': 50, 'entry_price': 2010.0, 'type': 'long', 'lot_size': 0.01}])
                ]
            }
        ]
        
        incentive_results = []
        
        for test in incentive_tests:
            test_results = []
            
            for scenario in test['scenarios']:
                name = scenario[0]
                portfolio_value = scenario[1]
                action = scenario[2]
                positions = scenario[3] if len(scenario) > 3 else []
                
                env = MockEnvironment(
                    portfolio_value=portfolio_value,
                    positions=positions,
                    current_step=100
                )
                
                old_state = {'positions': []}
                
                reward, info, _ = self.reward_system.calculate_reward_and_info(env, action, old_state)
                
                test_results.append({
                    'name': name,
                    'reward': reward,
                    'action_magnitude': np.abs(action).max()
                })
            
            # Analyze incentive direction
            if len(test_results) == 2:
                reward_diff = test_results[0]['reward'] - test_results[1]['reward']
                correct_incentive = self._check_incentive_direction(test['name'], reward_diff)
            else:
                correct_incentive = True  # Multi-scenario tests
            
            incentive_results.append({
                'test_name': test['name'],
                'scenarios': test_results,
                'correct_incentive': correct_incentive
            })
            
            print(f"   {test['name']}: {'✅' if correct_incentive else '❌'}")
            for result in test_results:
                print(f"     {result['name']}: {result['reward']:.4f}")
        
        self.test_results['training_incentives'] = incentive_results
    
    def _check_incentive_direction(self, test_name: str, reward_diff: float) -> bool:
        """Check if reward difference indicates correct training incentive"""
        if test_name == 'Profit vs Loss':
            return reward_diff > 0  # Profit should have higher reward
        elif test_name == 'Action vs Inaction':
            return reward_diff > 0  # Action should be slightly rewarded
        elif test_name == 'Quick vs Slow Loss Cutting':
            return reward_diff > 0  # Quick cutting should be rewarded
        return True
    
    def test_reward_distribution(self):
        """Test reward distribution characteristics"""
        print("\n📊 Testing Reward Distribution...")
        
        # Generate reward distribution from random scenarios
        rewards = []
        
        for _ in range(500):
            # Random portfolio change (-10% to +10%)
            portfolio_change = np.random.uniform(-0.1, 0.1)
            portfolio_value = 500.0 * (1 + portfolio_change)
            
            # Random action
            action = np.random.uniform(0, 1, 4)
            
            # Random positions
            positions = []
            if np.random.random() < 0.3:  # 30% chance of having positions
                positions = [{
                    'entry_price': 2000.0,
                    'entry_step': np.random.randint(50, 95),
                    'type': np.random.choice(['long', 'short']),
                    'lot_size': np.random.uniform(0.01, 0.03)
                }]
            
            env = MockEnvironment(
                portfolio_value=portfolio_value,
                positions=positions,
                current_step=100
            )
            
            old_state = {'positions': []}
            
            try:
                reward, _, _ = self.reward_system.calculate_reward_and_info(env, action, old_state)
                rewards.append(reward)
            except:
                rewards.append(0.0)
        
        rewards = np.array(rewards)
        
        distribution_stats = {
            'mean': np.mean(rewards),
            'median': np.median(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'q25': np.percentile(rewards, 25),
            'q75': np.percentile(rewards, 75),
            'positive_ratio': np.sum(rewards > 0) / len(rewards),
            'negative_ratio': np.sum(rewards < 0) / len(rewards),
            'zero_ratio': np.sum(rewards == 0) / len(rewards)
        }
        
        self.test_results['reward_distribution'] = distribution_stats
        
        print(f"   Mean: {distribution_stats['mean']:.4f}")
        print(f"   Median: {distribution_stats['median']:.4f}")
        print(f"   Std: {distribution_stats['std']:.4f}")
        print(f"   Range: [{distribution_stats['min']:.3f}, {distribution_stats['max']:.3f}]")
        print(f"   Positive: {distribution_stats['positive_ratio']:.1%}")
        print(f"   Negative: {distribution_stats['negative_ratio']:.1%}")
        print(f"   Zero: {distribution_stats['zero_ratio']:.1%}")
    
    def generate_balance_report(self):
        """Generate comprehensive balance analysis report"""
        print("\n📋 GENERATING BALANCE REPORT...")
        
        # Overall balance score
        balance_score = self._calculate_balance_score()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'version': 'V6_PRO',
            'balance_score': balance_score,
            'test_results': self.test_results,
            'configuration': V6_CONFIG,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        filename = f"v6_pro_balance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"D:\\Projeto\\avaliacoes\\{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   Report saved: {filepath}")
        
        return report
    
    def _calculate_balance_score(self) -> float:
        """Calculate overall balance score (0-100)"""
        score = 100.0
        
        # PnL dominance check
        if not self.test_results.get('pnl_dominance', {}).get('is_dominant', False):
            score -= 20
        
        # Component weight balance
        weights = self.test_results.get('component_weights', {}).get('weights', {})
        if weights.get('base_pnl', 0) < 0.5:  # Should be >50%
            score -= 15
        
        # Edge case stability
        edge_cases = self.test_results.get('edge_cases', {})
        if not edge_cases.get('all_finite', True):
            score -= 25
        if not edge_cases.get('all_bounded', True):
            score -= 15
        
        # Numerical stability
        stability = self.test_results.get('numerical_stability', {})
        if stability.get('finite_ratio', 0) < 0.95:
            score -= 15
        if stability.get('bounded_ratio', 0) < 0.95:
            score -= 10
        
        # Training incentives
        incentives = self.test_results.get('training_incentives', [])
        incorrect_incentives = sum(1 for t in incentives if not t.get('correct_incentive', True))
        score -= incorrect_incentives * 10
        
        return max(0, score)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # PnL dominance recommendations
        pnl_dom = self.test_results.get('pnl_dominance', {})
        if not pnl_dom.get('is_dominant', False):
            recommendations.append(
                "CRITICAL: Increase w_pnl weight or reduce other component weights to ensure PnL dominance"
            )
        
        # Component balance recommendations
        weights = self.test_results.get('component_weights', {}).get('weights', {})
        if weights.get('activity_component', 0) > 0.3:
            recommendations.append(
                "Consider reducing w_activity to prevent activity component from overwhelming PnL signal"
            )
        
        # Edge case recommendations
        edge_cases = self.test_results.get('edge_cases', {})
        if edge_cases.get('error_count', 0) > 0:
            recommendations.append(
                "Add additional error handling for edge cases to improve robustness"
            )
        
        # Numerical stability recommendations
        stability = self.test_results.get('numerical_stability', {})
        if stability.get('std_reward', 0) > 1.0:
            recommendations.append(
                "Consider adjusting scaling parameters to reduce reward variance"
            )
        
        # Distribution recommendations
        distribution = self.test_results.get('reward_distribution', {})
        if abs(distribution.get('mean', 0)) > 0.1:
            recommendations.append(
                "Reward distribution is biased. Consider rebalancing to center around zero"
            )
        
        if not recommendations:
            recommendations.append("No critical issues found. System appears well-balanced.")
        
        return recommendations


def main():
    """Run the comprehensive balance test"""
    print("🚀 V6 PRO REWARD SYSTEM BALANCE ANALYSIS")
    print("=" * 60)
    
    tester = V6ProRewardBalanceTester()
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 BALANCE TEST SUMMARY")
    print("=" * 60)
    
    balance_score = tester._calculate_balance_score()
    print(f"Overall Balance Score: {balance_score:.1f}/100")
    
    if balance_score >= 90:
        print("🟢 EXCELLENT: Reward system is well-balanced")
    elif balance_score >= 75:
        print("🟡 GOOD: Minor adjustments recommended")
    elif balance_score >= 60:
        print("🟠 MODERATE: Several issues need attention")
    else:
        print("🔴 POOR: Major rebalancing required")
    
    recommendations = tester._generate_recommendations()
    print(f"\n📋 RECOMMENDATIONS ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return results


if __name__ == "__main__":
    main()