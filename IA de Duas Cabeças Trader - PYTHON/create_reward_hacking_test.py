#!/usr/bin/env python3
"""
🕵️ TESTE DE REWARD HACKING
Sistema completo para identificar e testar exploits no reward system
"""

import sys
sys.path.append("D:/Projeto")

import numpy as np
from typing import Dict, List, Tuple
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator

class MockTradeEnvironment:
    """Environment mockado para testes de gaming"""
    
    def __init__(self):
        self.initial_balance = 1000.0
        self.balance = 1000.0
        self.realized_balance = 1000.0
        self.trades = []
        self.current_step = 0
        self.positions = []
        self.current_positions = 0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.recent_rewards = []
        self.reward_history_size = 100
        self.portfolio_value = 1000.0
        self.peak_portfolio = 1000.0
        
    def reset(self):
        """Reset environment"""
        self.balance = self.initial_balance
        self.realized_balance = self.initial_balance
        self.trades = []
        self.current_step = 0
        self.positions = []
        self.current_positions = 0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.recent_rewards = []
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        
    def add_trade(self, pnl: float, duration: int = 10, position_size: float = 0.02):
        """Adiciona trade com parâmetros controlados"""
        trade = {
            'pnl_usd': pnl,
            'pnl': pnl,
            'duration_steps': duration,
            'position_size': position_size,
            'entry_price': 2000.0,
            'exit_price': 2000.0 + pnl,
            'side': 'long' if pnl > 0 else 'short',
            'exit_reason': 'manual',
            'entry_step': self.current_step,
            'exit_step': self.current_step + duration
        }
        
        self.trades.append(trade)
        self.balance += pnl
        self.realized_balance += pnl
        self.portfolio_value += pnl
        
        # Update drawdown
        if self.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
        self.current_step += duration
        return trade

class RewardHackingTester:
    """Tester para identificar exploits no reward system"""
    
    def __init__(self):
        self.reward_calc = BalancedDayTradingRewardCalculator()
        
    def test_micro_farming(self) -> Dict:
        """Teste: muitos trades pequenos e lucrativos"""
        print("\n🕵️ TESTE: MICRO FARMING")
        print("=" * 50)
        
        env = MockTradeEnvironment()
        
        # Strategy: 100 trades de +$1 cada
        total_reward = 0
        for i in range(100):
            env.add_trade(pnl=1.0, duration=1, position_size=0.01)
            
            # Mock state/action
            state = np.random.randn(2580)
            action = np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0])  # Simple long
            next_state = np.random.randn(2580)
            
            reward, _, _ = self.reward_calc.calculate_reward_and_info(env, action, {})
            total_reward += reward
            
        avg_reward_per_trade = total_reward / 100
        total_pnl = env.portfolio_value - env.initial_balance
        
        result = {
            'strategy': 'micro_farming',
            'trades_count': 100,
            'total_pnl': total_pnl,
            'total_reward': total_reward,
            'avg_reward_per_trade': avg_reward_per_trade,
            'reward_per_dollar': total_reward / total_pnl if total_pnl > 0 else 0,
            'win_rate': 1.0,
            'is_exploit': avg_reward_per_trade > 0.01  # Threshold
        }
        
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Avg Reward/Trade: {avg_reward_per_trade:.6f}")
        print(f"   Reward/$ ratio: {result['reward_per_dollar']:.6f}")
        print(f"   Is Exploit: {'🚨 YES' if result['is_exploit'] else '✅ NO'}")
        
        return result
    
    def test_big_win_farming(self) -> Dict:
        """Teste: poucos trades enormes"""
        print("\n🕵️ TESTE: BIG WIN FARMING")  
        print("=" * 50)
        
        env = MockTradeEnvironment()
        
        # Strategy: 5 trades de +$200 cada
        total_reward = 0
        for i in range(5):
            env.add_trade(pnl=200.0, duration=20, position_size=0.05)
            
            state = np.random.randn(2580)
            action = np.array([1.0, 0.8, 0, 0, 0, 0, 0, 0])  # Big long
            next_state = np.random.randn(2580)
            
            reward, _, _ = self.reward_calc.calculate_reward_and_info(env, action, {})
            total_reward += reward
            
        avg_reward_per_trade = total_reward / 5
        total_pnl = env.portfolio_value - env.initial_balance
        
        result = {
            'strategy': 'big_win_farming',
            'trades_count': 5,
            'total_pnl': total_pnl,
            'total_reward': total_reward,
            'avg_reward_per_trade': avg_reward_per_trade,
            'reward_per_dollar': total_reward / total_pnl if total_pnl > 0 else 0,
            'win_rate': 1.0,
            'is_exploit': avg_reward_per_trade > 0.1  # Threshold maior
        }
        
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Avg Reward/Trade: {avg_reward_per_trade:.6f}")
        print(f"   Reward/$ ratio: {result['reward_per_dollar']:.6f}")
        print(f"   Is Exploit: {'🚨 YES' if result['is_exploit'] else '✅ NO'}")
        
        return result
        
    def test_mixed_strategy(self) -> Dict:
        """Teste: mix de wins/losses realista"""
        print("\n🕵️ TESTE: MIXED REALISTIC STRATEGY")
        print("=" * 50)
        
        env = MockTradeEnvironment()
        
        # Strategy: 60% win rate, wins maiores que losses
        total_reward = 0
        wins = 0
        losses = 0
        
        np.random.seed(42)  # Reproducible
        
        for i in range(50):
            if np.random.random() < 0.6:  # 60% win rate
                pnl = np.random.uniform(10, 100)  # Win
                wins += 1
            else:
                pnl = np.random.uniform(-80, -5)  # Loss
                losses += 1
                
            env.add_trade(pnl=pnl, duration=np.random.randint(5, 30))
            
            state = np.random.randn(2580)
            action = np.array([1.0 if pnl > 0 else 0.0, 0.6, 0, 0, 0, 0, 0, 0])
            next_state = np.random.randn(2580)
            
            reward, _, _ = self.reward_calc.calculate_reward_and_info(env, action, {})
            total_reward += reward
            
        avg_reward_per_trade = total_reward / 50
        total_pnl = env.portfolio_value - env.initial_balance
        win_rate = wins / (wins + losses)
        
        result = {
            'strategy': 'mixed_realistic',
            'trades_count': 50,
            'wins': wins,
            'losses': losses,
            'total_pnl': total_pnl,
            'total_reward': total_reward,
            'avg_reward_per_trade': avg_reward_per_trade,
            'reward_per_dollar': total_reward / total_pnl if total_pnl > 0 else float('inf'),
            'win_rate': win_rate,
            'is_exploit': False  # Baseline
        }
        
        print(f"   Wins/Losses: {wins}/{losses} ({win_rate:.1%} win rate)")
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Avg Reward/Trade: {avg_reward_per_trade:.6f}")
        print(f"   Reward/$ ratio: {result['reward_per_dollar']:.6f}")
        
        return result
        
    def test_hold_strategy(self) -> Dict:
        """Teste: estratégia de só fazer HOLD"""
        print("\n🕵️ TESTE: HOLD ONLY STRATEGY")
        print("=" * 50)
        
        env = MockTradeEnvironment()
        
        # Strategy: não fazer trades, só HOLD
        total_reward = 0
        
        for i in range(100):  # 100 steps de HOLD
            state = np.random.randn(2580)
            action = np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0])  # HOLD
            next_state = np.random.randn(2580)
            
            reward, _, _ = self.reward_calc.calculate_reward_and_info(env, action, {})
            total_reward += reward
            env.current_step += 1
            
        avg_reward_per_step = total_reward / 100
        total_pnl = env.portfolio_value - env.initial_balance
        
        result = {
            'strategy': 'hold_only',
            'trades_count': 0,
            'total_pnl': total_pnl,
            'total_reward': total_reward,
            'avg_reward_per_step': avg_reward_per_step,
            'is_exploit': avg_reward_per_step > 0.005  # Se HOLD dá muito reward
        }
        
        print(f"   Trades: {len(env.trades)}")
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Avg Reward/Step: {avg_reward_per_step:.6f}")
        print(f"   Is Exploit: {'🚨 YES' if result['is_exploit'] else '✅ NO'}")
        
        return result
        
    def run_all_tests(self) -> Dict:
        """Executa todos os testes de hacking"""
        print("🕵️ TESTANDO REWARD HACKING")
        print("=" * 70)
        
        results = {}
        
        results['micro_farming'] = self.test_micro_farming()
        results['big_win_farming'] = self.test_big_win_farming() 
        results['mixed_strategy'] = self.test_mixed_strategy()
        results['hold_strategy'] = self.test_hold_strategy()
        
        # Análise comparativa
        print("\n📊 ANÁLISE COMPARATIVA")
        print("=" * 50)
        
        exploits_found = []
        
        for strategy, result in results.items():
            if result.get('is_exploit', False):
                exploits_found.append(strategy)
                print(f"🚨 EXPLOIT: {strategy}")
            else:
                print(f"✅ SAFE: {strategy}")
                
        print(f"\n🎯 EXPLOITS ENCONTRADOS: {len(exploits_found)}")
        
        if len(exploits_found) == 0:
            print("✅ REWARD SYSTEM SEGURO CONTRA GAMING")
        else:
            print("⚠️ REWARD SYSTEM VULNERÁVEL A GAMING")
            for exploit in exploits_found:
                print(f"   - {exploit}")
                
        return {
            'test_results': results,
            'exploits_found': exploits_found,
            'is_vulnerable': len(exploits_found) > 0
        }

def main():
    tester = RewardHackingTester()
    results = tester.run_all_tests()
    
    # Save results
    import json
    with open("D:/Projeto/reward_hacking_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Resultados salvos em: reward_hacking_results.json")

if __name__ == "__main__":
    main()