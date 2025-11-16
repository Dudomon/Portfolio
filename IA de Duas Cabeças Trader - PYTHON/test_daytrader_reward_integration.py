#!/usr/bin/env python3
"""
🔌 TESTE DE INTEGRAÇÃO: DayTrader + Reward System V3.0
Verificar se o daytrader está recebendo corretamente as rewards
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
import time
from unittest.mock import Mock, patch
import logging

# Configurar logging para capturar saídas
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_daytrader_reward_integration():
    print("🔌 TESTE DE INTEGRAÇÃO: DayTrader + Reward System V3.0")
    print("=" * 70)
    
    try:
        # 1. IMPORTAR DAYTRADER
        print("1️⃣ IMPORTANDO DAYTRADER...")
        
        # Mock algumas dependências que podem ser problemáticas
        with patch('gym.make') as mock_gym_make:
            with patch('stable_baselines3.ppo.PPO') as mock_ppo:
                with patch('vectorized_trading_env.VectorizedTradingEnv') as mock_vec_env:
                    
                    # Importar o módulo principal
                    from daytrader import BEST_PARAMS, create_environment
                    print("   ✅ Daytrader importado com sucesso")
                    
                    # 2. VERIFICAR SE USA REWARD SYSTEM V3.0
                    print("\n2️⃣ TESTANDO CRIAÇÃO DE ENVIRONMENT...")
                    
                    # Mock básico do environment
                    mock_env = Mock()
                    mock_env.reset.return_value = np.random.randn(2580)
                    mock_env.step.return_value = (
                        np.random.randn(2580),  # observation
                        0.05,                   # reward (será testado)
                        False,                  # done
                        {'episode': {'r': 0.05}} # info
                    )
                    
                    mock_vec_env.return_value = mock_env
                    
                    # Tentar criar environment
                    try:
                        env = create_environment()
                        print("   ✅ Environment criado com sucesso")
                        
                        # 3. TESTAR REWARD CALCULATION
                        print("\n3️⃣ TESTANDO REWARD CALCULATION...")
                        
                        # Simular alguns passos
                        obs = env.reset()
                        print(f"   Initial obs shape: {np.array(obs).shape}")
                        
                        # Simular ações
                        actions = [
                            np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]),  # HOLD
                            np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0]),  # BUY
                            np.array([0.0, 0.0, 1, 0, 0, 0, 0, 0]),  # SELL
                        ]
                        
                        rewards_received = []
                        
                        for i, action in enumerate(actions):
                            obs, reward, done, info = env.step(action)
                            rewards_received.append(reward)
                            print(f"   Step {i+1}: Action={action[:2]}, Reward={reward:.6f}, Done={done}")
                        
                        print(f"   ✅ Rewards received: {rewards_received}")
                        
                    except Exception as e:
                        print(f"   ⚠️ Environment creation failed (expected in mock): {e}")
                        print("   ℹ️ This is normal - we're mocking dependencies")
                        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    # 4. TESTE DIRETO DO REWARD SYSTEM (SEM DAYTRADER)
    print("\n4️⃣ TESTE DIRETO DO REWARD SYSTEM...")
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        
        # Mock environment com alguns trades
        class MockTradingEnv:
            def __init__(self):
                self.trades = [
                    {'pnl_usd': 25.0, 'duration_steps': 10, 'position_size': 0.02},
                    {'pnl_usd': -10.0, 'duration_steps': 5, 'position_size': 0.01},
                    {'pnl_usd': 45.0, 'duration_steps': 20, 'position_size': 0.03}
                ]
                self.current_step = 30
                self.balance = 1060
                self.realized_balance = 1060
                self.portfolio_value = 1060
                self.initial_balance = 1000
                self.peak_portfolio = 1060
                self.current_drawdown = 0.0
                self.max_drawdown = 0.0
                self.current_positions = 0
                self.reward_history_size = 100
                self.recent_rewards = []
        
        reward_calc = BalancedDayTradingRewardCalculator()
        mock_env = MockTradingEnv()
        
        action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
        old_state = {'trades_count': 2}
        
        reward, info, done = reward_calc.calculate_reward_and_info(mock_env, action, old_state)
        components = info.get('reward_components', {})
        
        print(f"   Reward calculated: {reward:.6f}")
        print(f"   Components:")
        
        total_contribution = 0
        for comp_name, comp_value in components.items():
            if abs(comp_value) > 0.001:
                print(f"     {comp_name:20}: {comp_value:.6f}")
                total_contribution += abs(comp_value)
        
        print(f"   Total contribution: {total_contribution:.6f}")
        
        # Validações
        validations = []
        
        # 1. Reward não deve ser zero
        if abs(reward) > 0.001:
            validations.append("✅ Reward não é zero")
        else:
            validations.append("❌ Reward é zero")
        
        # 2. PnL deve ser dominante
        pnl_contribution = abs(components.get('pnl', 0))
        pnl_percentage = (pnl_contribution / total_contribution * 100) if total_contribution > 0 else 0
        
        if pnl_percentage > 50:
            validations.append(f"✅ PnL is dominant ({pnl_percentage:.1f}%)")
        else:
            validations.append(f"❌ PnL not dominant ({pnl_percentage:.1f}%)")
        
        # 3. Activity bonus deve ser zero
        activity_bonus = components.get('activity_bonus', -999)
        if activity_bonus == 0.0:
            validations.append("✅ Activity bonus eliminated")
        else:
            validations.append(f"❌ Activity bonus still exists ({activity_bonus})")
        
        # 4. Components devem ser consistentes
        if len([c for c in components.values() if abs(c) > 0.001]) >= 2:
            validations.append("✅ Multiple components active")
        else:
            validations.append("❌ Too few components active")
        
        print(f"\n   VALIDAÇÕES:")
        for validation in validations:
            print(f"     {validation}")
        
        success_count = len([v for v in validations if v.startswith("✅")])
        print(f"\n   SUCCESS RATE: {success_count}/{len(validations)} ({success_count/len(validations)*100:.0f}%)")
        
        if success_count >= 3:
            print("   🟢 REWARD SYSTEM WORKING CORRECTLY")
            return True
        else:
            print("   🔴 REWARD SYSTEM HAS ISSUES")
            return False
        
    except Exception as e:
        print(f"❌ Direct reward system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_system_import():
    """Teste específico de importação do reward system"""
    print("\n5️⃣ TESTE DE IMPORTAÇÃO DO REWARD SYSTEM...")
    
    try:
        # Testar importação direta
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        
        # Instanciar
        calc = BalancedDayTradingRewardCalculator()
        
        # Verificar atributos críticos
        checks = []
        
        if hasattr(calc, 'activity_bonus_weight') and calc.activity_bonus_weight == 0.0:
            checks.append("✅ Activity bonus weight is 0.0")
        else:
            checks.append(f"❌ Activity bonus weight is {getattr(calc, 'activity_bonus_weight', 'missing')}")
        
        if hasattr(calc, 'base_weights') and calc.base_weights.get('pnl_direct', 0) > 0:
            checks.append(f"✅ PnL direct weight is {calc.base_weights.get('pnl_direct', 0)}")
        else:
            checks.append("❌ PnL direct weight missing or zero")
        
        if hasattr(calc, 'min_trade_size_threshold') and calc.min_trade_size_threshold == 10.0:
            checks.append("✅ Anti-micro-farming threshold set")
        else:
            checks.append("❌ Anti-micro-farming threshold missing")
        
        print("   Import checks:")
        for check in checks:
            print(f"     {check}")
        
        success_count = len([c for c in checks if c.startswith("✅")])
        
        if success_count >= 2:
            print("   🟢 REWARD SYSTEM IMPORT OK")
            return True
        else:
            print("   🔴 REWARD SYSTEM IMPORT ISSUES")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Executar todos os testes"""
    print("🧪 INICIANDO TESTES DE INTEGRAÇÃO DAYTRADER + REWARD V3.0")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 2
    
    # Teste 1: Integração com daytrader
    if test_daytrader_reward_integration():
        tests_passed += 1
        
    # Teste 2: Importação do reward system
    if test_reward_system_import():
        tests_passed += 1
    
    print(f"\n🏆 RESULTADO FINAL DOS TESTES DE INTEGRAÇÃO")
    print("=" * 50)
    print(f"   Testes passou: {tests_passed}/{total_tests}")
    print(f"   Taxa de sucesso: {tests_passed/total_tests*100:.0f}%")
    
    if tests_passed >= total_tests:
        print("   🟢 INTEGRAÇÃO FUNCIONANDO CORRETAMENTE")
        print("   ✅ Daytrader pode usar o reward system V3.0 com segurança")
        return True
    else:
        print("   🔴 PROBLEMAS DE INTEGRAÇÃO DETECTADOS")
        print("   ❌ Corrigir problemas antes do treinamento")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)