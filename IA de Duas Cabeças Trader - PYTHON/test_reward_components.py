"""
🔍 TESTE DE COMPONENTES DO BRUTAL REWARD SYSTEM
Sistema para identificar qual componente está explodindo gradientes
"""

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockEnv:
    """Mock environment para testar reward components"""
    
    def __init__(self):
        self.current_balance = 1000.0
        self.initial_balance = 1000.0
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.max_balance_reached = 1000.0
        self.current_step = 100
        self.positions = []
        self.closed_positions = []
        
    def set_scenario(self, scenario_name, **kwargs):
        """Define cenários de teste específicos"""
        if scenario_name == "small_loss":
            self.current_balance = 990.0  # -1%
            self.total_realized_pnl = -10.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "medium_loss":
            self.current_balance = 950.0  # -5%
            self.total_realized_pnl = -50.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "large_loss":
            self.current_balance = 900.0  # -10%
            self.total_realized_pnl = -100.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "small_gain":
            self.current_balance = 1020.0  # +2%
            self.total_realized_pnl = 20.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "large_gain":
            self.current_balance = 1100.0  # +10%
            self.total_realized_pnl = 100.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "extreme_loss":
            self.current_balance = 800.0  # -20%
            self.total_realized_pnl = -200.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "with_unrealized":
            self.current_balance = 1000.0
            self.total_realized_pnl = 10.0
            self.total_unrealized_pnl = 30.0  # Unrealized gain
            
        # Update max balance
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.max_balance_reached = max(self.max_balance_reached, self.current_balance)

def test_reward_components():
    """Testa cada componente do reward system separadamente"""
    
    print("🔍 TESTANDO COMPONENTES DO BRUTAL REWARD SYSTEM")
    print("=" * 60)
    
    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    env = MockEnv()
    
    scenarios = [
        "small_loss",
        "medium_loss", 
        "large_loss",
        "small_gain",
        "large_gain",
        "extreme_loss",
        "with_unrealized"
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 CENÁRIO: {scenario.upper()}")
        print("-" * 40)
        
        env.set_scenario(scenario)
        
        # Mock action
        action = np.array([0.3, 0.5, 0.2, 0.1])  # Entry, confidence, pos1, pos2
        
        try:
            # Calcula reward
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            
            # Separa componentes individualmente para análise
            realized_pnl = env.total_realized_pnl
            unrealized_pnl = env.total_unrealized_pnl
            total_pnl = realized_pnl + (unrealized_pnl * 0.5)
            pnl_percent = total_pnl / env.initial_balance
            
            # Componente base (CORRIGIDO conforme código atual)
            pnl_percent_clipped = np.clip(pnl_percent, -0.15, 0.15)
            base_reward_raw = pnl_percent_clipped * 5.0  # Base multiplier atual
            
            # Pain/bonus factors (CORRIGIDO para usar clipped values)
            pain_applied = pnl_percent_clipped < -0.03
            bonus_applied = pnl_percent_clipped > 0.02
            
            if pain_applied:
                pain_factor = 1.0 + (1.5 - 1.0) * np.tanh(abs(pnl_percent_clipped) * 20)
                base_reward_with_multiplier = base_reward_raw * pain_factor
            elif bonus_applied:
                bonus_factor = 1.0 + 0.1 * np.tanh(pnl_percent_clipped * 50)
                base_reward_with_multiplier = base_reward_raw * bonus_factor
            else:
                base_reward_with_multiplier = base_reward_raw
                
            # Portfolio drawdown
            portfolio_drawdown = (env.max_balance_reached - env.current_balance) / env.max_balance_reached
            
            print(f"  💰 Balance: {env.current_balance:.1f} ({pnl_percent*100:+.1f}%)")
            print(f"  📈 PnL Realizado: {realized_pnl:+.2f}")
            print(f"  📊 PnL Não Realizado: {unrealized_pnl:+.2f}")
            print(f"  🎯 Base Reward Raw: {base_reward_raw:.4f}")
            print(f"  ⚡ Pain Applied: {pain_applied}")
            print(f"  🎁 Bonus Applied: {bonus_applied}")
            print(f"  🔥 Base c/ Multiplier: {base_reward_with_multiplier:.4f}")
            print(f"  📉 Portfolio DD: {portfolio_drawdown:.3f}")
            print(f"  ✅ REWARD FINAL: {reward:.6f}")
            print(f"  🚨 Done: {done}")
            
            # Detecta valores extremos
            extreme_values = []
            if abs(base_reward_raw) > 1.0:
                extreme_values.append(f"base_raw={base_reward_raw:.4f}")
            if abs(base_reward_with_multiplier) > 2.0:
                extreme_values.append(f"base_multiplied={base_reward_with_multiplier:.4f}")
            if abs(reward) > 1.0:
                extreme_values.append(f"final_reward={reward:.4f}")
                
            if extreme_values:
                print(f"  🚨 VALORES EXTREMOS: {', '.join(extreme_values)}")
            
            results[scenario] = {
                'reward': reward,
                'base_raw': base_reward_raw,
                'base_multiplied': base_reward_with_multiplier,
                'pnl_percent': pnl_percent,
                'extreme_values': extreme_values
            }
            
        except Exception as e:
            print(f"  ❌ ERRO: {e}")
            results[scenario] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    print("📋 RESUMO DE COMPONENTES PROBLEMÁTICOS")
    print("=" * 60)
    
    problematic_scenarios = []
    for scenario, result in results.items():
        if 'error' in result:
            print(f"🚨 {scenario}: ERRO - {result['error']}")
            problematic_scenarios.append(scenario)
        elif result.get('extreme_values'):
            print(f"⚠️  {scenario}: {', '.join(result['extreme_values'])}")
            problematic_scenarios.append(scenario)
        else:
            print(f"✅ {scenario}: OK (reward={result['reward']:.4f})")
    
    if problematic_scenarios:
        print(f"\n🎯 CENÁRIOS PROBLEMÁTICOS: {', '.join(problematic_scenarios)}")
        print("💡 COMPONENTES SUSPEITOS:")
        
        # Analisa padrões
        large_base_raw = [s for s, r in results.items() if not 'error' in r and abs(r.get('base_raw', 0)) > 1.0]
        large_multiplied = [s for s, r in results.items() if not 'error' in r and abs(r.get('base_multiplied', 0)) > 2.0]
        
        if large_base_raw:
            print(f"   - Base reward raw muito alto em: {', '.join(large_base_raw)}")
        if large_multiplied:
            print(f"   - Multipliers causando explosão em: {', '.join(large_multiplied)}")
            
    else:
        print("\n✅ TODOS OS COMPONENTES ESTÃO ESTÁVEIS!")
    
    return results

if __name__ == "__main__":
    results = test_reward_components()