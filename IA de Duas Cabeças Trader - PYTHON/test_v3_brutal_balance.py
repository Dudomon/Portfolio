#!/usr/bin/env python3
"""
🧪 TESTE DE BALANCE - V3 BRUTAL REWARD
Simula cenários reais para verificar se os rewards estão balanceados
"""

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from trading_framework.rewards.reward_daytrade_v3_brutal import create_brutal_daytrade_reward_system

class MockTradingEnv:
    """Mock environment para simular cenários de trading"""
    
    def __init__(self, realized_pnl=0, unrealized_pnl=0, current_balance=1000, portfolio_value=1000):
        self.total_realized_pnl = realized_pnl
        self.total_unrealized_pnl = unrealized_pnl
        self.current_balance = current_balance
        self.portfolio_value = portfolio_value
        self.initial_balance = 1000
        
        # Histórico simulado
        self.trades_history = []
        self.portfolio_history = [portfolio_value]
        
    def get_current_price(self):
        return 4500.0  # Preço mock do ES

def test_reward_balance():
    """Testa diferentes cenários de PnL para verificar balance"""
    
    print("🧪 TESTE DE BALANCE - V3 BRUTAL REWARD CORRIGIDO")
    print("=" * 60)
    
    reward_system = create_brutal_daytrade_reward_system(initial_balance=1000)
    
    # Cenários de teste
    test_scenarios = [
        # (nome, realized_pnl, unrealized_pnl, portfolio_value)
        ("Breakeven", 0, 0, 1000),
        ("Pequeno lucro +1%", 10, 0, 1010),
        ("Lucro médio +3%", 30, 0, 1030),  # Como no seu log
        ("Lucro bom +5%", 50, 0, 1050),
        ("Pequena perda -1%", -10, 0, 990),
        ("Perda média -3%", -30, 0, 970),
        ("Perda alta -5%", -50, 0, 950),
        ("Posição aberta +2%", 0, 20, 1020),
        ("Posição aberta -2%", 0, -20, 980),
        ("Mix: Real +2%, Aberto +1%", 20, 10, 1030),
    ]
    
    print("\n📊 RESULTADOS DOS CENÁRIOS:")
    print("-" * 60)
    
    results = []
    
    for scenario_name, realized, unrealized, portfolio in test_scenarios:
        env = MockTradingEnv(realized, unrealized, portfolio, portfolio)
        
        # Ação neutra
        action = np.array([0.0, 0.0, 0.0, 0.0])
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        pnl_total = realized + (unrealized * 0.5)
        pnl_percent = (pnl_total / 1000) * 100
        
        results.append({
            'scenario': scenario_name,
            'pnl_percent': pnl_percent,
            'reward': reward,
            'pnl_reward': info.get('pnl_reward', 0),
            'shaping_reward': info.get('shaping_reward', 0),
            'done': done
        })
        
        status = "🚨 DONE" if done else "✅ CONT"
        
        print(f"{status} {scenario_name:20s} | PnL: {pnl_percent:+6.2f}% | Reward: {reward:+8.4f} | PnL: {info.get('pnl_reward', 0):+6.3f}")
    
    print("\n🎯 ANÁLISE DE BALANCE:")
    print("-" * 40)
    
    # Verifica se rewards são proporcionais
    positive_rewards = [r for r in results if r['pnl_percent'] > 0]
    negative_rewards = [r for r in results if r['pnl_percent'] < 0]
    
    if positive_rewards:
        avg_pos_reward = np.mean([r['reward'] for r in positive_rewards])
        print(f"📈 Reward médio lucros: {avg_pos_reward:+.4f}")
    
    if negative_rewards:
        avg_neg_reward = np.mean([r['reward'] for r in negative_rewards])
        print(f"📉 Reward médio perdas: {avg_neg_reward:+.4f}")
    
    # Verifica se reward de +3% é detectável
    plus_3_scenario = next((r for r in results if "média +3%" in r['scenario']), None)
    if plus_3_scenario:
        reward_3pct = plus_3_scenario['reward']
        print(f"\n🎯 CENÁRIO DO SEU LOG (+3% = +$30):")
        print(f"   Reward: {reward_3pct:+.4f}")
        if abs(reward_3pct) < 0.001:
            print("   ❌ PROBLEMA: Reward muito baixo (< 0.001)")
        else:
            print("   ✅ OK: Reward detectável pelo PPO")
    
    # Teste de linearidade
    print(f"\n📐 TESTE DE LINEARIDADE:")
    linear_scenarios = [(r['pnl_percent'], r['reward']) for r in results if -5 <= r['pnl_percent'] <= 5]
    if len(linear_scenarios) > 2:
        pnls = [s[0] for s in linear_scenarios]
        rewards = [s[1] for s in linear_scenarios]
        correlation = np.corrcoef(pnls, rewards)[0,1] if len(set(rewards)) > 1 else 0
        print(f"   Correlação PnL-Reward: {correlation:.3f}")
        if correlation > 0.8:
            print("   ✅ BOA linearidade")
        else:
            print("   ⚠️  Linearidade baixa")

def test_extreme_scenarios():
    """Testa cenários extremos"""
    
    print(f"\n🔥 TESTE DE CENÁRIOS EXTREMOS:")
    print("-" * 40)
    
    reward_system = create_brutal_daytrade_reward_system(initial_balance=1000)
    
    extreme_scenarios = [
        ("Lucro extremo +20%", 200, 0, 1200),
        ("Perda extrema -20%", -200, 0, 800),
        ("Drawdown catastrófico -60%", -600, 0, 400),
        ("Posição gigante +10%", 0, 100, 1100),
    ]
    
    for scenario_name, realized, unrealized, portfolio in extreme_scenarios:
        env = MockTradingEnv(realized, unrealized, portfolio, portfolio)
        action = np.array([0.0, 0.0, 0.0, 0.0])
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        pnl_total = realized + (unrealized * 0.5)
        pnl_percent = (pnl_total / 1000) * 100
        
        status = "🚨 DONE" if done else "✅ CONT"
        clipped = "📎" if abs(reward) >= 0.99 else "  "
        
        print(f"{status} {clipped} {scenario_name:25s} | PnL: {pnl_percent:+6.1f}% | Reward: {reward:+8.4f}")

if __name__ == "__main__":
    test_reward_balance()
    test_extreme_scenarios()
    
    print(f"\n🏁 CONCLUSÃO:")
    print("Se o reward de +3% está > 0.01, o sistema deve funcionar corretamente")
    print("Se ainda estiver ~0, há outro problema na arquitetura")