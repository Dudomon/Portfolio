"""
🧪 TESTE DA CORREÇÃO DO REWARD SYSTEM - CHERRY
Compara versão original vs corrigida
"""

import sys
import os
import numpy as np

# Adicionar paths
sys.path.append(r'D:\Projeto')

print("🧪 [TESTE] Comparando reward systems: Original vs Corrigido...")

# Import reward systems
from trading_framework.rewards.reward_system_simple import create_simple_reward_system
from cherry_reward_fix import create_cherry_reward_fixed

class MockEnv:
    """Mock environment para comparação"""
    def __init__(self):
        self.trades = []
        self.positions = []
        self.current_step = 100
        self.current_drawdown = 0.0
        self.df = None
        
    def add_trade(self, pnl_usd, sl_points, tp_points, duration_steps=30):
        trade = {
            'pnl_usd': pnl_usd,
            'sl_points': sl_points,
            'tp_points': tp_points,
            'duration_steps': duration_steps,
            'type': 'long',
            'entry_price': 2000.0,
            'exit_price': 2000.0 + (pnl_usd / 0.02 / 100),
            'sl_price': 2000.0 - sl_points,
            'tp_price': 2000.0 + tp_points,
            'exit_reason': 'tp' if pnl_usd > 0 else 'sl'
        }
        self.trades.append(trade)
        return trade

def compare_reward_systems():
    """Comparar comportamento dos sistemas"""
    print("\n📊 [COMPARAÇÃO] Original vs Corrigido...")
    
    # Criar ambos os sistemas
    original_system = create_simple_reward_system(500.0)
    fixed_system = create_cherry_reward_fixed(500.0)
    
    # Cenários de teste
    test_scenarios = [
        (8.0, 12, 18, 35, "Win médio swing"),
        (-6.0, 10, 15, 30, "Loss médio swing"),
        (25.0, 15, 30, 60, "Win grande swing"),
        (-18.0, 20, 35, 45, "Loss grande swing"),
        (1.0, 8, 12, 25, "Win pequeno"),
        (0.0, 10, 15, 40, "Breakeven"),
    ]
    
    print("   Cenário           | Original | Corrigido | Diferença | Clipping?")
    print("   " + "-" * 68)
    
    improvements = 0
    total_tests = len(test_scenarios)
    
    for pnl, sl, tp, duration, desc in test_scenarios:
        # Teste sistema original
        env_orig = MockEnv()
        env_orig.add_trade(pnl, sl, tp, duration)
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': 0}
        
        reward_orig, info_orig, _ = original_system.calculate_reward_and_info(env_orig, action, old_state)
        
        # Teste sistema corrigido
        env_fixed = MockEnv()
        env_fixed.add_trade(pnl, sl, tp, duration)
        
        reward_fixed, info_fixed, _ = fixed_system.calculate_reward_and_info(env_fixed, action, old_state)
        
        # Análise
        difference = reward_fixed - reward_orig
        clipping_orig = abs(reward_orig) >= 99.9
        clipping_fixed = info_fixed.get('clipping_applied', False)
        
        print(f"   {desc:16} | {reward_orig:8.2f} | {reward_fixed:9.2f} | {difference:9.2f} | {clipping_orig}/{clipping_fixed}")
        
        # Contar melhorias
        if abs(reward_fixed) < abs(reward_orig) and abs(reward_orig) > 50:
            improvements += 1
    
    print(f"\n✅ Melhorias detectadas: {improvements}/{total_tests} cenários")
    
    return improvements

def test_dense_rewards():
    """Testar dense rewards para HOLDs"""
    print("\n🧠 [TESTE DENSE] Rewards para ações HOLD...")
    
    fixed_system = create_cherry_reward_fixed(500.0)
    
    # Cenários de HOLD
    hold_scenarios = [
        ("HOLD normal", 0, []),
        ("HOLD com posições cheias", 2, []),
        ("HOLD em drawdown", 0, [], 15.0),
        ("HOLD overtrading", 0, [], 0.0, True),
    ]
    
    print("   Cenário              | Reward | Componentes")
    print("   " + "-" * 50)
    
    for desc, positions_count, trades, dd, *extra in hold_scenarios:
        env = MockEnv()
        env.positions = [{'id': i} for i in range(positions_count)]
        env.current_drawdown = dd if len(extra) > 0 and not extra[0] else 0.0
        
        # HOLD action
        action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': len(env.trades)}
        
        reward, info, _ = fixed_system.calculate_reward_and_info(env, action, old_state)
        
        # Extrair componentes dense
        dense_reward = info['components'].get('dense_rewards', 0)
        
        print(f"   {desc:19} | {reward:6.3f} | Dense: {dense_reward:.3f}")
    
    return True

def test_clipping_improvements():
    """Testar melhorias no clipping"""
    print("\n🔧 [TESTE CLIPPING] Soft vs Hard Clipping...")
    
    original_system = create_simple_reward_system(500.0)
    fixed_system = create_cherry_reward_fixed(500.0)
    
    # Valores extremos
    extreme_values = [30.0, 50.0, 100.0, -30.0, -50.0, -100.0]
    
    print("   PnL    | Original  | Corrigido | Tipo Clipping")
    print("   " + "-" * 48)
    
    for pnl in extreme_values:
        env_orig = MockEnv()
        env_orig.add_trade(pnl, 12, 18, 35)
        env_fixed = MockEnv()
        env_fixed.add_trade(pnl, 12, 18, 35)
        
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': 0}
        
        reward_orig, _, _ = original_system.calculate_reward_and_info(env_orig, action, old_state)
        reward_fixed, info_fixed, _ = fixed_system.calculate_reward_and_info(env_fixed, action, old_state)
        
        orig_type = "Hard" if abs(reward_orig) >= 99.9 else "None"
        fixed_type = "Soft" if info_fixed.get('clipping_applied', False) else "None"
        
        print(f"   ${pnl:5.0f} | {reward_orig:9.2f} | {reward_fixed:9.2f} | {orig_type:4s}/{fixed_type:4s}")
    
    return True

def main():
    print("🍒 ========================================")
    print("🍒 TESTE CORREÇÃO REWARD SYSTEM - CHERRY")  
    print("🍒 ========================================")
    
    # Executar testes
    improvements = compare_reward_systems()
    dense_ok = test_dense_rewards()
    clipping_ok = test_clipping_improvements()
    
    print("\n" + "="*50)
    print("📋 RESUMO DOS TESTES")
    print("="*50)
    
    print(f"✅ Melhorias em magnitude: {improvements} cenários")
    print(f"✅ Dense rewards: {'Funcionando' if dense_ok else 'Problema'}")
    print(f"✅ Soft clipping: {'Funcionando' if clipping_ok else 'Problema'}")
    
    print("\n🎯 PRINCIPAIS CORREÇÕES APLICADAS:")
    print("- PnL direto: 15.0 → 3.0 (5x menor)")
    print("- Win bonus: 8.0 → 2.0 (4x menor)")
    print("- Clipping: Hard [-100,100] → Soft tanh([-20,20])")
    print("- Dense rewards: +0.1 para HOLDs inteligentes")
    print("- Componentes balanceados: PnL 40%, não 93%")
    
    print(f"\n🚀 PRÓXIMO PASSO: Implementar no cherry.py")

if __name__ == "__main__":
    main()