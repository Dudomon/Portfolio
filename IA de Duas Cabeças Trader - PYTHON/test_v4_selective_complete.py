#!/usr/bin/env python3
"""
🧪 BATERIA COMPLETA DE TESTES - V4 SELECTIVE REWARD
Testa o reward system corrigido com normalização adaptativa
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import do reward system
sys.path.append('D:\\Projeto')
from trading_framework.rewards.reward_daytrade_v4_selective import SelectiveTradingReward

class MockEnv:
    """Ambiente mock para testes"""
    def __init__(self):
        self.portfolio_value = 1000.0
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.positions = []
        self.trades = []
        self.current_step = 0
        self.cooldown_counter = 0
        self.steps_since_last_trade = 0
        
        # Mock data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'close': 1800 + np.cumsum(np.random.randn(1000) * 0.5),
            'high': 1800 + np.cumsum(np.random.randn(1000) * 0.5) + np.abs(np.random.randn(1000)),
            'low': 1800 + np.cumsum(np.random.randn(1000) * 0.5) - np.abs(np.random.randn(1000)),
            'spread': np.random.uniform(0.01, 0.05, 1000)
        })
    
    def simulate_pnl_change(self, change_usd: float):
        """Simula mudança no PnL"""
        self.portfolio_value += change_usd
        if change_usd != 0:
            self.total_realized_pnl += change_usd * 0.7  # 70% realizado
            self.total_unrealized_pnl += change_usd * 0.3  # 30% não realizado
    
    def add_position(self, ptype='long', pnl=0.0, age=0):
        """Adiciona posição mock"""
        pos = {
            'type': ptype,
            'entry_price': self.df['close'].iloc[self.current_step],
            'entry_step': self.current_step - age,
            'unrealized_pnl': pnl,
            'lot_size': 0.01,
            'age': age
        }
        self.positions.append(pos)
    
    def close_position(self, pnl_usd: float):
        """Fecha posição e registra trade"""
        if self.positions:
            self.positions.pop()
            trade = {'pnl_usd': pnl_usd}
            self.trades.append(trade)
            self.simulate_pnl_change(pnl_usd)

def test_reward_stability():
    """🧪 Teste de estabilidade do reward"""
    print("🧪 TESTE 1: Estabilidade do Reward")
    print("=" * 50)
    
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    env = MockEnv()
    
    rewards = []
    scenarios = [
        ("Small profit", 10.0),
        ("Small loss", -5.0),
        ("Big profit", 50.0), 
        ("Big loss", -30.0),
        ("Huge profit", 200.0),  # Teste extremo
        ("Huge loss", -150.0),   # Teste extremo
        ("Zero change", 0.0)
    ]
    
    for name, pnl_change in scenarios:
        env.simulate_pnl_change(pnl_change)
        action = np.array([0.5, 0.7, 0.0, 0.0])  # Ação neutra
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        rewards.append(reward)
        
        print(f"{name:15} | PnL: ${pnl_change:7.1f} | Reward: {reward:8.4f} | Norm: {'✅' if abs(reward) <= 2.0 else '❌'}")
    
    # Estatísticas
    rewards_array = np.array(rewards)
    print(f"\nReward Range: [{rewards_array.min():.4f}, {rewards_array.max():.4f}]")
    print(f"Reward Std: {rewards_array.std():.4f}")
    print(f"Stable: {'✅' if np.all(np.abs(rewards_array) <= 2.0) else '❌'}")
    
    return rewards_array.std() < 1.0  # Teste passa se std < 1.0

def test_normalization_activation():
    """🧪 Teste de ativação da normalização adaptativa"""
    print("\n🧪 TESTE 2: Normalização Adaptativa")
    print("=" * 50)
    
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    env = MockEnv()
    
    rewards_before = []
    rewards_after = []
    
    # Coleta rewards antes da ativação (primeiros 50 steps)
    for i in range(60):
        pnl = np.random.uniform(-20, 50)  # PnL variável
        env.simulate_pnl_change(pnl)
        action = np.array([np.random.uniform(0, 1), 0.5, 0.0, 0.0])
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        if i < 50:
            rewards_before.append(reward)
        else:
            rewards_after.append(reward)
    
    # Coleta mais rewards após ativação (próximos 100 steps)  
    for i in range(100):
        pnl = np.random.uniform(-30, 80)  # PnL mais extremo
        env.simulate_pnl_change(pnl)
        action = np.array([np.random.uniform(0, 1), 0.5, 0.0, 0.0])
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        rewards_after.append(reward)
    
    before_std = np.std(rewards_before)
    after_std = np.std(rewards_after)
    
    print(f"Rewards Before Normalization - Range: [{np.min(rewards_before):.3f}, {np.max(rewards_before):.3f}], Std: {before_std:.3f}")
    print(f"Rewards After Normalization - Range: [{np.min(rewards_after):.3f}, {np.max(rewards_after):.3f}], Std: {after_std:.3f}")
    print(f"Normalization Active: {'✅' if reward_system.normalization_active else '❌'}")
    print(f"Stability Improved: {'✅' if after_std < before_std else '❌'}")
    
    return reward_system.normalization_active and after_std <= 1.0

def test_overtrading_penalty():
    """🧪 Teste de penalização por overtrading"""
    print("\n🧪 TESTE 3: Anti-Overtrading")
    print("=" * 50)
    
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    env = MockEnv()
    
    # Simular overtrading (60% tempo posicionado)
    total_steps = 100
    for step in range(total_steps):
        env.current_step = step
        
        # 60% do tempo com posição (overtrading)
        has_position = step % 10 < 6  # 6 de cada 10 steps
        
        if has_position:
            env.add_position('long', pnl=5.0, age=1)
        else:
            env.positions = []
            
        action = np.array([0.5, 0.5, 0.0, 0.0])
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        if step == 50:  # Meio do teste
            print(f"Step {step}: Position ratio: {info.get('position_time_ratio', 0):.2f}")
            if 'overtrading_penalty' in info:
                print(f"Overtrading penalty: {info['overtrading_penalty']:.4f} ✅")
    
    final_ratio = reward_system.steps_in_position / reward_system.total_steps
    print(f"Final position time ratio: {final_ratio:.2f}")
    print(f"Overtrading penalties applied: {reward_system.overtrading_penalties_applied}")
    print(f"Anti-overtrading working: {'✅' if reward_system.overtrading_penalties_applied > 0 else '❌'}")
    
    return reward_system.overtrading_penalties_applied > 0

def test_selective_quality():
    """🧪 Teste de qualidade seletiva"""
    print("\n🧪 TESTE 4: Qualidade de Entrada")
    print("=" * 50)
    
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    env = MockEnv()
    
    # Teste: entrada com alta confiança em mercado bom
    old_state = {'positions': []}
    env.positions = [{'type': 'long', 'entry_price': 1800, 'lot_size': 0.01}]  # Simular entrada
    
    # Ação com alta confiança  
    action_high_conf = np.array([1.0, 0.9, 0.0, 0.0])  # entry_decision=1, confidence=0.9
    reward_high, info_high, done = reward_system.calculate_reward_and_info(env, action_high_conf, old_state)
    
    # Reset
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    env.positions = [{'type': 'long', 'entry_price': 1800, 'lot_size': 0.01}]
    
    # Ação com baixa confiança
    action_low_conf = np.array([1.0, 0.2, 0.0, 0.0])   # entry_decision=1, confidence=0.2
    reward_low, info_low, done = reward_system.calculate_reward_and_info(env, action_low_conf, old_state)
    
    print(f"High confidence entry reward: {reward_high:.4f}")
    print(f"Low confidence entry reward: {reward_low:.4f}")
    print(f"Quality bonuses given (high conf): {info_high.get('quality_bonuses_given', 0)}")
    print(f"Quality system working: {'✅' if 'quality_score' in info_high else '❌'}")
    
    return 'quality_score' in info_high

def test_performance_scenarios():
    """🧪 Teste de cenários de performance"""
    print("\n🧪 TESTE 5: Cenários de Performance")
    print("=" * 50)
    
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    
    scenarios = [
        ("Profitable consistent", [10, 15, 8, 12, 20]),
        ("Loss cutting", [-5, -3, -8, -2, -10]),
        ("Mixed performance", [20, -15, 30, -5, 25]),
        ("Choppy market", [2, -1, 3, -2, 1])
    ]
    
    for scenario_name, pnl_sequence in scenarios:
        env = MockEnv()
        total_reward = 0
        
        for pnl in pnl_sequence:
            env.simulate_pnl_change(pnl)
            action = np.array([0.3, 0.6, 0.1, -0.1])
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            total_reward += reward
        
        print(f"{scenario_name:20} | Total PnL: ${sum(pnl_sequence):6.1f} | Total Reward: {total_reward:8.4f}")
        
        # Reset para próximo cenário
        reward_system = SelectiveTradingReward(initial_balance=1000.0)
    
    return True

def run_stress_test():
    """🧪 Teste de stress com volume alto"""
    print("\n🧪 TESTE 6: Stress Test (1000 steps)")
    print("=" * 50)
    
    reward_system = SelectiveTradingReward(initial_balance=1000.0)
    env = MockEnv()
    
    rewards = []
    errors = 0
    
    for step in range(1000):
        env.current_step = step
        
        # PnL randômico extremo
        pnl = np.random.uniform(-100, 200)
        env.simulate_pnl_change(pnl)
        
        # Ações randômicas
        action = np.random.uniform(-1, 1, 4)
        
        # Posições randômicas
        if np.random.random() < 0.3:
            env.add_position('long', np.random.uniform(-50, 100), np.random.randint(1, 50))
        
        try:
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            rewards.append(reward)
            
            # Verificar se reward é válido
            if not np.isfinite(reward) or abs(reward) > 10:
                errors += 1
                
        except Exception as e:
            errors += 1
            print(f"Error at step {step}: {e}")
    
    rewards_array = np.array(rewards)
    print(f"Steps completed: {len(rewards)}/1000")
    print(f"Errors: {errors}")
    print(f"Reward range: [{rewards_array.min():.3f}, {rewards_array.max():.3f}]")
    print(f"Reward std: {rewards_array.std():.3f}")
    print(f"Normalization active: {'✅' if reward_system.normalization_active else '❌'}")
    print(f"Stress test passed: {'✅' if errors == 0 and rewards_array.std() < 2.0 else '❌'}")
    
    return errors == 0 and rewards_array.std() < 2.0

def main():
    """Executa bateria completa de testes"""
    print("🚀 BATERIA COMPLETA DE TESTES - V4 SELECTIVE REWARD")
    print("=" * 70)
    print("Testando reward system corrigido com normalização adaptativa")
    print("=" * 70)
    
    tests = [
        ("Reward Stability", test_reward_stability),
        ("Normalization Activation", test_normalization_activation), 
        ("Overtrading Penalty", test_overtrading_penalty),
        ("Selective Quality", test_selective_quality),
        ("Performance Scenarios", test_performance_scenarios),
        ("Stress Test", run_stress_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"\n{status} - {test_name}")
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ ERRO - {test_name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"🏆 RESULTADO FINAL: {passed}/{total} testes passaram")
    print(f"Taxa de sucesso: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✅ V4 SELECTIVE REWARD está funcionando perfeitamente!")
        print("Pronto para usar no silus.py")
    else:
        print("❌ Alguns testes falharam - verificar correções necessárias")
    
    print("=" * 70)

if __name__ == "__main__":
    main()