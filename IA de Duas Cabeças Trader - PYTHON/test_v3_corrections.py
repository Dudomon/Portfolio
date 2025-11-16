#!/usr/bin/env python3
"""
🧪 TESTE FINAL DAS CORREÇÕES V3.0
Validar que micro-farming foi corrigido e quality trading é premiado
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator

class MockEnvV3:
    def __init__(self):
        self.trades = []
        self.current_step = 0
        self.balance = 1000
        self.realized_balance = 1000
        self.portfolio_value = 1000
        self.initial_balance = 1000
        self.peak_portfolio = 1000
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.current_positions = 0
        self.reward_history_size = 100
        self.recent_rewards = []
        
    def add_trade(self, pnl):
        self.trades.append({
            'pnl_usd': pnl,
            'pnl': pnl,
            'duration_steps': 10,
            'position_size': 0.02,
            'entry_price': 2000.0,
            'exit_price': 2000.0 + pnl,
            'side': 'long' if pnl > 0 else 'short',
            'exit_reason': 'manual'
        })
        self.portfolio_value += pnl
        
def test_v3_corrections():
    """Testar as correções V3.0 implementadas"""
    print("🔧 TESTANDO CORREÇÕES V3.0 DO REWARD SYSTEM")
    print("=" * 60)
    
    calc = BalancedDayTradingRewardCalculator()
    
    # TESTE 1: Cenário Micro-farming (DEVE SER PENALIZADO)
    print("\n🧪 TESTE 1: MICRO-FARMING (50 trades de $2)")
    env1 = MockEnvV3()
    
    for i in range(50):
        env1.add_trade(2.0)
        
    action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
    old_state = {'trades_count': 0}
    
    reward1, info1, _ = calc.calculate_reward_and_info(env1, action, old_state)
    
    print(f"   Total PnL: ${50 * 2}")
    print(f"   Reward: {reward1:.6f}")
    print(f"   Componentes: {info1.get('reward_components', {})}")
    print(f"   Status: {'❌ PENALIZADO' if reward1 < 0.1 else '⚠️ AINDA ALTO'}")
    
    # TESTE 2: Cenário Quality Trading (DEVE SER PREMIADO)
    print("\n🧪 TESTE 2: QUALITY TRADING (5 trades de $80)")
    env2 = MockEnvV3()
    
    for i in range(5):
        env2.add_trade(80.0)
        
    reward2, info2, _ = calc.calculate_reward_and_info(env2, action, old_state)
    
    print(f"   Total PnL: ${5 * 80}")
    print(f"   Reward: {reward2:.6f}")
    print(f"   Componentes: {info2.get('reward_components', {})}")
    print(f"   Status: {'✅ PREMIADO' if reward2 > 0.1 else '❌ AINDA BAIXO'}")
    
    # TESTE 3: Cenário Overtrading (DEVE SER PENALIZADO)
    print("\n🧪 TESTE 3: OVERTRADING (100 trades de $5)")
    env3 = MockEnvV3()
    
    for i in range(100):
        env3.add_trade(5.0)
        
    reward3, info3, _ = calc.calculate_reward_and_info(env3, action, old_state)
    
    print(f"   Total PnL: ${100 * 5}")
    print(f"   Reward: {reward3:.6f}")
    print(f"   Status: {'❌ PENALIZADO' if reward3 < 0.05 else '⚠️ AINDA ALTO'}")
    
    # TESTE 4: Cenário Balanced (DEVE SER NEUTRO)
    print("\n🧪 TESTE 4: BALANCED TRADING (10 trades de $40)")
    env4 = MockEnvV3()
    
    for i in range(10):
        env4.add_trade(40.0)
        
    reward4, info4, _ = calc.calculate_reward_and_info(env4, action, old_state)
    
    print(f"   Total PnL: ${10 * 40}")
    print(f"   Reward: {reward4:.6f}")
    print(f"   Status: {'✅ BALANCEADO' if 0.05 <= reward4 <= 0.3 else '⚠️ DESBALANCEADO'}")
    
    # ANÁLISE COMPARATIVA
    print("\n📊 ANÁLISE COMPARATIVA (V3.0)")
    print("=" * 50)
    
    ratios = {
        "Quality vs Micro": reward2 / reward1 if reward1 > 0 else float('inf'),
        "Quality vs Overtrading": reward2 / reward3 if reward3 > 0 else float('inf'),
        "Balanced vs Micro": reward4 / reward1 if reward1 > 0 else float('inf')
    }
    
    for name, ratio in ratios.items():
        if ratio == float('inf'):
            print(f"   {name}: ∞ (micro-farming completamente penalizado)")
        else:
            print(f"   {name}: {ratio:.1f}x")
            
    # VALIDAÇÃO FINAL
    print("\n🎯 VALIDAÇÃO V3.0")
    print("=" * 30)
    
    corrections_working = 0
    
    # 1. Quality trading deve ser > Micro-farming
    if reward2 > reward1:
        print("   ✅ Quality > Micro-farming")
        corrections_working += 1
    else:
        print("   ❌ Quality ≤ Micro-farming")
        
    # 2. Quality trading deve ser > Overtrading  
    if reward2 > reward3:
        print("   ✅ Quality > Overtrading")
        corrections_working += 1
    else:
        print("   ❌ Quality ≤ Overtrading")
        
    # 3. Micro-farming deve ser severamente penalizado
    if reward1 < 0.05:
        print("   ✅ Micro-farming penalizado")
        corrections_working += 1
    else:
        print("   ❌ Micro-farming não penalizado suficiente")
        
    # 4. Overtrading deve ser penalizado
    if reward3 < reward4:
        print("   ✅ Overtrading penalizado vs Balanced")
        corrections_working += 1
    else:
        print("   ❌ Overtrading não penalizado vs Balanced")
        
    success_rate = corrections_working / 4 * 100
    print(f"\n🏆 TAXA DE SUCESSO: {success_rate:.0f}% ({corrections_working}/4)")
    
    if success_rate >= 75:
        print("✅ CORREÇÕES V3.0 FUNCIONANDO!")
    else:
        print("❌ CORREÇÕES V3.0 PRECISAM AJUSTE")
        
    return {
        'micro_farming_reward': reward1,
        'quality_trading_reward': reward2,
        'overtrading_reward': reward3,
        'balanced_reward': reward4,
        'success_rate': success_rate
    }

if __name__ == "__main__":
    results = test_v3_corrections()