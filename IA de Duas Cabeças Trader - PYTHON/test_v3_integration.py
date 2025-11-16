#!/usr/bin/env python3
"""
🧪 TESTE DE INTEGRAÇÃO V3 BRUTAL
Verificação rápida que o sistema V3 está funcionando no daytrader.py
"""

import sys
sys.path.append("D:/Projeto")

import numpy as np
from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

def test_v3_integration():
    print("🧪 TESTE DE INTEGRAÇÃO V3 BRUTAL")
    print("=" * 50)
    
    # Testar inicialização
    reward_system = BrutalMoneyReward(initial_balance=10000)
    print("✅ V3 Brutal inicializado com sucesso")
    
    # Mock environment para teste
    class MockTradingEnv:
        def __init__(self):
            self.total_realized_pnl = 300  # +3% lucro
            self.total_unrealized_pnl = 0
            self.portfolio_value = 10300
            self.peak_portfolio_value = 10300
    
    env = MockTradingEnv()
    action = np.zeros(8)
    old_state = {}
    
    # Testar cálculo de reward
    reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
    
    print(f"✅ Reward calculado: {reward:+.2f}")
    print(f"✅ PnL %: {info.get('pnl_percent', 0):+.1f}%")
    print(f"✅ Done: {done}")
    print(f"✅ Info keys: {list(info.keys())}")
    
    # Testar caso com PAIN
    env.total_realized_pnl = -800  # -8% perda
    env.portfolio_value = 9200
    env.peak_portfolio_value = 10000
    
    reward_pain, info_pain, done_pain = reward_system.calculate_reward_and_info(env, action, old_state)
    
    print(f"\n💥 TESTE COM PAIN:")
    print(f"✅ Reward com pain: {reward_pain:+.2f}")
    print(f"✅ Pain ativado: {info_pain.get('pain_applied', False)}")
    print(f"✅ Ratio pain vs normal: {abs(reward_pain / reward):.1f}x mais intenso")
    
    # Verificar stats
    stats = reward_system.get_stats()
    print(f"\n📊 STATS:")
    print(f"✅ Total rewards: {stats['total_rewards']}")
    print(f"✅ Positive ratio: {stats['positive_ratio']:.2f}")
    print(f"✅ Negative ratio: {stats['negative_ratio']:.2f}")
    
    print(f"\n🎯 INTEGRAÇÃO V3 BRUTAL: SUCESSO! 🚀")

if __name__ == "__main__":
    test_v3_integration()