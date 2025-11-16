"""
🔍 TESTE COMPLETO DO REWARD V3 BRUTAL - DETECTAR INSTABILIDADES
Sistema para testar todos os cenários e identificar o que está quebrando gradientes
"""

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockEnv:
    """Mock environment mais completo para testar reward brutal"""
    
    def __init__(self):
        self.initial_balance = 1000.0
        self.current_balance = 1000.0
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.max_balance_reached = 1000.0
        self.current_step = 100
        self.positions = []
        self.closed_positions = []
        self.portfolio_value = 1000.0
        
    def set_scenario(self, scenario_name, **kwargs):
        """Define cenários específicos de teste"""
        if scenario_name == "tiny_loss":
            self.current_balance = 999.0  # -0.1%
            self.total_realized_pnl = -1.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "medium_loss":
            self.current_balance = 970.0  # -3%
            self.total_realized_pnl = -30.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "large_loss":
            self.current_balance = 950.0  # -5%
            self.total_realized_pnl = -50.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "extreme_loss":
            self.current_balance = 900.0  # -10%
            self.total_realized_pnl = -100.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "catastrophic_loss":
            self.current_balance = 800.0  # -20%
            self.total_realized_pnl = -200.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "tiny_gain":
            self.current_balance = 1001.0  # +0.1%
            self.total_realized_pnl = 1.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "medium_gain":
            self.current_balance = 1030.0  # +3%
            self.total_realized_pnl = 30.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "large_gain":
            self.current_balance = 1100.0  # +10%
            self.total_realized_pnl = 100.0
            self.total_unrealized_pnl = 0.0
            
        elif scenario_name == "with_unrealized_loss":
            self.current_balance = 1000.0
            self.total_realized_pnl = 10.0
            self.total_unrealized_pnl = -50.0  # Large unrealized loss
            
        elif scenario_name == "mixed_complex":
            self.current_balance = 980.0
            self.total_realized_pnl = -30.0
            self.total_unrealized_pnl = 10.0  # Some unrealized gain
            
        # Update dependent values
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.max_balance_reached = max(self.max_balance_reached, self.current_balance)
        self.portfolio_value = self.current_balance

def test_reward_stability():
    """Testa estabilidade do reward em múltiplos cenários"""
    
    print("🔍 TESTE COMPLETO DE ESTABILIDADE DO REWARD V3 BRUTAL")
    print("=" * 70)
    
    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    env = MockEnv()
    
    scenarios = [
        "tiny_loss",
        "medium_loss", 
        "large_loss",
        "extreme_loss",
        "catastrophic_loss",
        "tiny_gain",
        "medium_gain",
        "large_gain",
        "with_unrealized_loss",
        "mixed_complex"
    ]
    
    problematic_rewards = []
    extreme_values = []
    
    for scenario in scenarios:
        print(f"\n📊 CENÁRIO: {scenario.upper()}")
        print("-" * 50)
        
        env.set_scenario(scenario)
        
        # Test multiple action patterns
        action_patterns = [
            np.array([0.1, 0.1, 0.0, 0.0]),   # Conservative
            np.array([0.5, 0.5, 0.0, 0.0]),   # Moderate  
            np.array([0.9, 0.9, 0.5, 0.5]),   # Aggressive
            np.array([1.5, 0.8, -0.5, 0.3]),  # Mixed extreme
        ]
        
        for i, action in enumerate(action_patterns):
            try:
                reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
                
                # Detect problematic values
                if abs(reward) > 1.0:
                    extreme_values.append({
                        'scenario': scenario,
                        'action_pattern': i,
                        'reward': reward,
                        'info': info
                    })
                    
                if not np.isfinite(reward):
                    problematic_rewards.append({
                        'scenario': scenario,
                        'action_pattern': i,
                        'reward': reward,
                        'error': 'Non-finite reward'
                    })
                    
                # Print detailed analysis
                pnl_percent = (env.total_realized_pnl + env.total_unrealized_pnl * 0.5) / env.initial_balance
                
                print(f"  Action {i}: Reward={reward:.6f}")
                print(f"    PnL%: {pnl_percent*100:+.2f}%")
                print(f"    Components: PnL={info.get('pnl_reward', 0):.4f}, Risk={info.get('risk_reward', 0):.4f}, Shape={info.get('shaping_reward', 0):.4f}")
                
                # Check if reward is in dangerous range for gradients
                if abs(reward) > 0.5:
                    print(f"    ⚠️  POTENTIAL GRADIENT RISK: |{reward:.4f}| > 0.5")
                elif abs(reward) > 0.2:
                    print(f"    ⚠️  MODERATE RISK: |{reward:.4f}| > 0.2")
                    
            except Exception as e:
                print(f"  ❌ ERROR in action {i}: {e}")
                problematic_rewards.append({
                    'scenario': scenario,
                    'action_pattern': i,
                    'error': str(e)
                })
    
    print("\n" + "=" * 70)
    print("📋 ANÁLISE DE ESTABILIDADE")
    print("=" * 70)
    
    if extreme_values:
        print(f"\n🚨 VALORES EXTREMOS DETECTADOS ({len(extreme_values)} casos):")
        for case in extreme_values:
            print(f"  - {case['scenario']}: reward={case['reward']:.4f}")
            
    if problematic_rewards:
        print(f"\n💥 ERROS CRÍTICOS ({len(problematic_rewards)} casos):")
        for case in problematic_rewards:
            print(f"  - {case['scenario']}: {case.get('error', 'Unknown error')}")
    
    # Test gradient-critical range
    gradient_safe_count = 0
    total_tests = len(scenarios) * len(action_patterns)
    
    for scenario in scenarios:
        env.set_scenario(scenario)
        for action in action_patterns:
            try:
                reward, _, _ = reward_system.calculate_reward_and_info(env, action, {})
                if abs(reward) <= 0.1:  # Safe range for gradients
                    gradient_safe_count += 1
            except:
                pass
    
    gradient_safety_ratio = gradient_safe_count / total_tests
    
    print(f"\n📊 ESTATÍSTICAS DE ESTABILIDADE:")
    print(f"  - Total de testes: {total_tests}")
    print(f"  - Valores extremos (|reward| > 1.0): {len(extreme_values)}")
    print(f"  - Erros críticos: {len(problematic_rewards)}")
    print(f"  - Taxa de segurança para gradientes (|reward| ≤ 0.1): {gradient_safety_ratio:.1%}")
    
    if gradient_safety_ratio < 0.8:
        print(f"\n🚨 SISTEMA INSTÁVEL: Apenas {gradient_safety_ratio:.1%} dos rewards são seguros para gradientes!")
        print("💡 RECOMENDAÇÃO: Reduzir escalas do reward drasticamente")
    elif extreme_values:
        print(f"\n⚠️  SISTEMA MARGINALMENTE ESTÁVEL: {len(extreme_values)} valores extremos detectados")
        print("💡 RECOMENDAÇÃO: Ajustar normalização e clipping")
    else:
        print(f"\n✅ SISTEMA ESTÁVEL: {gradient_safety_ratio:.1%} dos rewards são seguros")
        
    return {
        'extreme_values': extreme_values,
        'problematic_rewards': problematic_rewards,
        'gradient_safety_ratio': gradient_safety_ratio,
        'total_tests': total_tests
    }

if __name__ == "__main__":
    results = test_reward_stability()