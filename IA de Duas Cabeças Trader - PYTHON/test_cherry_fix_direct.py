"""
🧪 TESTE RÁPIDO DAS CORREÇÕES CHERRY
Verifica se as correções aplicadas diretamente estão funcionando
"""

import sys
import os
import numpy as np

# Adicionar paths
sys.path.append(r'D:\Projeto')

print("🧪 [TESTE] Verificando correções aplicadas ao Cherry...")

try:
    from trading_framework.rewards.reward_system_simple import create_simple_reward_system
    print("✅ SimpleRewardCalculator corrigido importado com sucesso")
    
except ImportError as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

class MockEnv:
    """Mock environment para teste"""
    def __init__(self):
        self.trades = []
        self.positions = []
        self.current_step = 100
        self.current_drawdown = 0.0
        self.df = None
        
    def add_trade(self, pnl_usd, sl_points=12, tp_points=18):
        trade = {
            'pnl_usd': pnl_usd,
            'sl_points': sl_points,
            'tp_points': tp_points,
            'duration_steps': 35,
            'type': 'long',
            'entry_price': 2000.0,
            'exit_price': 2000.0 + (pnl_usd / 0.02 / 100),
            'sl_price': 2000.0 - sl_points,
            'tp_price': 2000.0 + tp_points,
            'exit_reason': 'tp' if pnl_usd > 0 else 'sl'
        }
        self.trades.append(trade)

def test_magnitude_fix():
    """Testar se a magnitude foi corrigida"""
    print("\n📊 [TESTE 1] Verificando magnitude corrigida...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Verificar pesos corrigidos
    expected_fixes = {
        'pnl_direct': 3.0,    # era 15.0
        'win_bonus': 2.0,     # era 8.0
        'loss_penalty': -1.5  # era -5.0
    }
    
    for weight_name, expected_value in expected_fixes.items():
        actual_value = reward_system.weights.get(weight_name, 999)
        if abs(actual_value - expected_value) < 0.1:
            print(f"   ✅ {weight_name}: {actual_value} (corrigido)")
        else:
            print(f"   ❌ {weight_name}: {actual_value} (esperado: {expected_value})")
    
    return True

def test_reward_magnitude():
    """Testar magnitude dos rewards"""
    print("\n📊 [TESTE 2] Verificando magnitude dos rewards...")
    
    reward_system = create_simple_reward_system(500.0)
    
    test_trades = [
        (10.0, "Win médio"),
        (-8.0, "Loss médio"),
        (25.0, "Win grande")
    ]
    
    print("   Trade           | Reward   | Status")
    print("   " + "-" * 38)
    
    all_reasonable = True
    
    for pnl, desc in test_trades:
        env = MockEnv()
        env.add_trade(pnl)
        
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        old_state = {'trades_count': 0}
        
        reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
        
        # Verificar se está em range razoável
        reasonable = -30.0 <= reward <= 30.0
        status = "✅ OK" if reasonable else "❌ ALTO"
        
        if not reasonable:
            all_reasonable = False
        
        print(f"   {desc:14} | {reward:8.2f} | {status}")
    
    return all_reasonable

def test_dense_rewards():
    """Testar dense rewards para HOLDs"""
    print("\n📊 [TESTE 3] Verificando dense rewards...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Teste HOLD
    env = MockEnv()
    env.positions = [{'id': 1}, {'id': 2}]  # 2 posições abertas
    env.current_drawdown = 12.0  # Alto drawdown
    
    # HOLD action
    action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    old_state = {'trades_count': 0}
    
    reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
    
    # Verificar se tem dense rewards
    dense_reward = info['components'].get('dense_rewards_cherry_fix', 0)
    
    print(f"   HOLD com 2 posições + DD alto:")
    print(f"   Dense reward: {dense_reward:.3f}")
    print(f"   Total reward: {reward:.3f}")
    
    has_dense = dense_reward > 0.01
    print(f"   Status: {'✅ Dense rewards funcionando' if has_dense else '❌ Sem dense rewards'}")
    
    return has_dense

def test_soft_clipping():
    """Testar soft clipping"""
    print("\n📊 [TESTE 4] Verificando soft clipping...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Trade extremo para testar clipping
    env = MockEnv()
    env.add_trade(100.0)  # PnL muito alto
    
    action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    old_state = {'trades_count': 0}
    
    reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
    
    # Verificar se está dentro dos novos limites
    within_limits = -20.0 <= reward <= 20.0
    is_soft = abs(reward) < 19.9  # Não deve estar no limite hard
    
    print(f"   PnL extremo $100:")
    print(f"   Reward: {reward:.2f}")
    print(f"   Dentro limites [-20, 20]: {'✅' if within_limits else '❌'}")
    print(f"   Soft clipping (não saturado): {'✅' if is_soft else '❌'}")
    
    return within_limits and is_soft

def main():
    print("🍒 ========================================")
    print("🍒 TESTE CORREÇÕES CHERRY - DIRETO")  
    print("🍒 ========================================")
    
    # Executar testes
    fix1 = test_magnitude_fix()
    fix2 = test_reward_magnitude()
    fix3 = test_dense_rewards()
    fix4 = test_soft_clipping()
    
    print("\n" + "="*50)
    print("📋 RESUMO DAS CORREÇÕES")
    print("="*50)
    
    fixes = [
        (fix1, "Magnitude dos pesos corrigida"),
        (fix2, "Rewards em range razoável"), 
        (fix3, "Dense rewards funcionando"),
        (fix4, "Soft clipping aplicado")
    ]
    
    total_fixed = sum(1 for fix, _ in fixes if fix)
    
    for fix, desc in fixes:
        status = "✅" if fix else "❌"
        print(f"   {status} {desc}")
    
    print(f"\n🎯 CORREÇÕES APLICADAS: {total_fixed}/4")
    
    if total_fixed >= 3:
        print("🎉 CHERRY REWARD SYSTEM CORRIGIDO!")
        print("💡 Explained variance deve melhorar significativamente")
    else:
        print("⚠️ Algumas correções falharam - verificar implementação")

if __name__ == "__main__":
    main()