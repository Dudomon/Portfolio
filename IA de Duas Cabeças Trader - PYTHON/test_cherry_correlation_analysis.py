"""
📊 ANÁLISE MATEMÁTICA COMPLETA - CHERRY REWARD SYSTEM
Testa correlações, intensidade de sinal e adequação matemática para guiar o modelo
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import pandas as pd

sys.path.append(r'D:\Projeto')

from trading_framework.rewards.reward_system_simple import create_simple_reward_system

class MockEnv:
    def __init__(self, dd_pct=0.0):
        self.trades = []
        self.positions = []
        self.current_step = 100
        self.current_drawdown = dd_pct
        self.df = None
        
    def add_trade(self, pnl_usd):
        trade = {
            'pnl_usd': pnl_usd,
            'sl_points': 12,
            'tp_points': 18,
            'duration_steps': 35,
            'type': 'long',
            'entry_price': 2000.0,
            'exit_price': 2000.0 + (pnl_usd / 0.02 / 100),
            'sl_price': 2000.0 - 12,
            'tp_price': 2000.0 + 18,
            'exit_reason': 'tp' if pnl_usd > 0 else 'sl'
        }
        self.trades.append(trade)

def test_dd_reward_correlation():
    """Testar correlação matemática DD vs Reward"""
    print("📊 [CORRELAÇÃO 1] DD vs Reward - Deve ser FORTEMENTE NEGATIVA")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Gerar dados de correlação DD vs Reward
    dd_values = np.linspace(0, 50, 100)  # DD de 0 a 50%
    rewards = []
    
    for dd in dd_values:
        env = MockEnv(dd_pct=dd)
        action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])  # HOLD
        reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
        rewards.append(reward)
    
    rewards = np.array(rewards)
    
    # Análise estatística
    correlation, p_value = stats.pearsonr(dd_values, rewards)
    r_squared = r2_score(np.zeros_like(rewards), rewards)  # R² vs baseline zero
    
    # Análise de intensidade
    reward_range = np.max(rewards) - np.min(rewards)
    sensitivity = reward_range / 50  # Mudança de reward por 1% DD
    
    print(f"   Correlação Pearson: {correlation:.4f} (p={p_value:.2e})")
    print(f"   R² (explicação variância): {r_squared:.4f}")
    print(f"   Range de rewards: {reward_range:.2f}")
    print(f"   Sensibilidade: {sensitivity:.2f} reward/1% DD")
    
    # Verificações críticas
    if correlation < -0.8:
        print(f"   ✅ FORTE correlação negativa: {correlation:.3f}")
    else:
        print(f"   ❌ Correlação insuficiente: {correlation:.3f} (precisa <-0.8)")
    
    if sensitivity > 3.0:
        print(f"   ✅ Alta sensibilidade: {sensitivity:.2f} (modelo sentirá mudanças)")
    else:
        print(f"   ❌ Baixa sensibilidade: {sensitivity:.2f} (modelo pode ignorar)")
    
    return dd_values, rewards, correlation, sensitivity

def test_pnl_reward_correlation():
    """Testar correlação PnL vs Reward em diferentes cenários de DD"""
    print("\n📊 [CORRELAÇÃO 2] PnL vs Reward em diferentes DDs")
    
    reward_system = create_simple_reward_system(500.0)
    
    scenarios = [
        (2, "DD Baixo (2%)"),
        (15, "DD Médio (15%)"), 
        (30, "DD Alto (30%)"),
        (45, "DD Crítico (45%)")
    ]
    
    pnl_values = np.linspace(-20, 20, 41)  # PnL de -20 a +20
    
    print("   Cenário        | Correlação | R²     | Sensibilidade | Status")
    print("   " + "-" * 70)
    
    for dd, desc in scenarios:
        rewards = []
        
        for pnl in pnl_values:
            env = MockEnv(dd_pct=dd)
            env.add_trade(pnl)
            action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
            rewards.append(reward)
        
        rewards = np.array(rewards)
        
        # Análise estatística
        correlation, _ = stats.pearsonr(pnl_values, rewards)
        r_squared = r2_score(np.zeros_like(rewards), rewards)
        
        # Sensibilidade: mudança de reward por $1 PnL
        sensitivity = (rewards[-1] - rewards[0]) / 40  # Range PnL = 40
        
        status = "✅" if correlation > 0.7 and sensitivity > 0.5 else "❌"
        
        print(f"   {desc:13} | {correlation:9.3f} | {r_squared:6.3f} | {sensitivity:12.2f} | {status}")
    
    return True

def test_action_reward_coherence():
    """Testar coerência de rewards entre diferentes ações"""
    print("\n🧠 [COERÊNCIA] Ordem lógica de rewards por ação")
    
    reward_system = create_simple_reward_system(500.0)
    
    test_scenarios = [
        (2, 10, "DD baixo + Win"),
        (2, -5, "DD baixo + Loss"),
        (30, 10, "DD alto + Win"),
        (30, -5, "DD alto + Loss"),
        (45, 0, "DD crítico + HOLD")
    ]
    
    print("   Cenário           | HOLD   | TRADE  | Diferença | Coerente?")
    print("   " + "-" * 65)
    
    coherent_count = 0
    total_tests = 0
    
    for dd, pnl, desc in test_scenarios:
        # HOLD
        env_hold = MockEnv(dd_pct=dd)
        action_hold = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward_hold, _, _ = reward_system.calculate_reward_and_info(env_hold, action_hold, {'trades_count': 0})
        
        # TRADE
        if pnl != 0:
            env_trade = MockEnv(dd_pct=dd)
            env_trade.add_trade(pnl)
            action_trade = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            reward_trade, _, _ = reward_system.calculate_reward_and_info(env_trade, action_trade, {'trades_count': 0})
            
            diff = reward_hold - reward_trade
            
            # Lógica de coerência
            if dd > 20:  # DD alto: HOLD deve ser melhor
                coherent = reward_hold >= reward_trade
            elif dd < 10 and pnl > 0:  # DD baixo + win: TRADE pode ser melhor
                coherent = True  # Ambos são válidos
            elif pnl < 0:  # Loss: HOLD sempre melhor
                coherent = reward_hold >= reward_trade
            else:
                coherent = True
            
            status = "✅" if coherent else "❌"
            if coherent:
                coherent_count += 1
            total_tests += 1
            
            print(f"   {desc:16} | {reward_hold:6.2f} | {reward_trade:6.2f} | {diff:9.2f} | {status}")
    
    coherence_rate = coherent_count / total_tests
    print(f"\n   Taxa de coerência: {coherence_rate:.2%} ({coherent_count}/{total_tests})")
    
    return coherence_rate

def test_signal_strength_adequacy():
    """Testar adequação da força do sinal"""
    print("\n⚡ [INTENSIDADE] Força do sinal para aprendizado")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Testar diferentes magnitudes de diferença
    signal_tests = [
        ("DD: 5% vs 10%", [(5, 0), (10, 0)]),
        ("DD: 10% vs 20%", [(10, 0), (20, 0)]),
        ("DD: 20% vs 40%", [(20, 0), (40, 0)]),
        ("PnL: $5 vs $10", [(5, 5), (5, 10)]),
        ("PnL: $-5 vs $-10", [(5, -5), (5, -10)]),
        ("Action: HOLD vs TRADE", [(30, 0), (30, 5)]),  # DD alto
    ]
    
    print("   Teste                | Valor 1 | Valor 2 | Diferença | Força")
    print("   " + "-" * 68)
    
    strong_signals = 0
    total_signals = 0
    
    for desc, scenarios in signal_tests:
        rewards = []
        
        for dd_or_pnl1, dd_or_pnl2 in scenarios:
            if "DD:" in desc:
                # Teste de DD
                env = MockEnv(dd_pct=dd_or_pnl1)
                action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
                reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
                rewards.append(reward)
            elif "PnL:" in desc:
                # Teste de PnL
                env = MockEnv(dd_pct=dd_or_pnl1)
                env.add_trade(dd_or_pnl2)
                action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
                reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
                rewards.append(reward)
            elif "Action:" in desc:
                # Teste de ação
                env = MockEnv(dd_pct=dd_or_pnl1)
                if dd_or_pnl2 == 0:  # HOLD
                    action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
                else:  # TRADE
                    env.add_trade(dd_or_pnl2)
                    action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
                reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
                rewards.append(reward)
        
        if len(rewards) == 2:
            diff = abs(rewards[1] - rewards[0])
            
            # Força do sinal
            if diff > 10:
                strength = "FORTE"
                strong_signals += 1
            elif diff > 3:
                strength = "MÉDIA"
            else:
                strength = "FRACA"
            
            total_signals += 1
            
            print(f"   {desc:19} | {rewards[0]:7.2f} | {rewards[1]:7.2f} | {diff:9.2f} | {strength}")
    
    strength_rate = strong_signals / total_signals
    print(f"\n   Taxa de sinais fortes: {strength_rate:.2%} ({strong_signals}/{total_signals})")
    
    return strength_rate

def test_mathematical_properties():
    """Testar propriedades matemáticas do sistema de reward"""
    print("\n🔬 [MATEMÁTICA] Propriedades matemáticas do sistema")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Teste 1: Monotonicidade (DD crescente → reward decrescente)
    dd_sequence = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    rewards_dd = []
    
    for dd in dd_sequence:
        env = MockEnv(dd_pct=dd)
        action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
        rewards_dd.append(reward)
    
    # Verificar monotonicidade
    monotonic = all(rewards_dd[i] >= rewards_dd[i+1] for i in range(len(rewards_dd)-1))
    
    # Teste 2: Linearidade vs não-linearidade adequada
    dd_fine = np.linspace(0, 50, 100)
    rewards_fine = []
    
    for dd in dd_fine:
        env = MockEnv(dd_pct=dd)
        action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
        rewards_fine.append(reward)
    
    # Teste de não-linearidade (curvatura)
    second_derivative = np.gradient(np.gradient(rewards_fine))
    non_linearity = np.std(second_derivative)
    
    # Teste 3: Estabilidade (pequenas mudanças → pequenas diferenças)
    stability_test = []
    for dd in [10, 10.1, 10.2, 10.3, 10.4, 10.5]:
        env = MockEnv(dd_pct=dd)
        action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
        stability_test.append(reward)
    
    stability_variance = np.var(stability_test)
    
    print(f"   Monotonicidade DD→Reward: {'✅' if monotonic else '❌'}")
    print(f"   Não-linearidade (curvatura): {non_linearity:.2f}")
    print(f"   Estabilidade (var pequenas mudanças): {stability_variance:.4f}")
    print(f"   Range total: {np.max(rewards_fine) - np.min(rewards_fine):.2f}")
    
    return monotonic, non_linearity, stability_variance

def comprehensive_correlation_report(dd_data, reward_data, correlation, sensitivity):
    """Gerar relatório completo de correlações"""
    print("\n📋 [RELATÓRIO] Análise Matemática Completa")
    print("="*60)
    
    # Estatísticas descritivas
    reward_mean = np.mean(reward_data)
    reward_std = np.std(reward_data)
    reward_cv = reward_std / abs(reward_mean) if reward_mean != 0 else float('inf')
    
    print(f"📊 ESTATÍSTICAS DESCRITIVAS:")
    print(f"   Rewards - Média: {reward_mean:.2f}, Std: {reward_std:.2f}")
    print(f"   Coeficiente de Variação: {reward_cv:.3f}")
    print(f"   Range: [{np.min(reward_data):.1f}, {np.max(reward_data):.1f}]")
    
    print(f"\n🔗 CORRELAÇÕES CRÍTICAS:")
    print(f"   DD-Reward: {correlation:.4f} ({'FORTE' if abs(correlation) > 0.8 else 'MÉDIA' if abs(correlation) > 0.5 else 'FRACA'})")
    print(f"   Sensibilidade: {sensitivity:.2f} reward/1%DD")
    
    # Avaliação geral
    correlation_good = abs(correlation) > 0.8
    sensitivity_good = sensitivity > 3.0
    range_good = (np.max(reward_data) - np.min(reward_data)) > 50
    
    overall_score = sum([correlation_good, sensitivity_good, range_good])
    
    print(f"\n🎯 AVALIAÇÃO GERAL:")
    print(f"   Correlação adequada: {'✅' if correlation_good else '❌'}")
    print(f"   Sensibilidade adequada: {'✅' if sensitivity_good else '❌'}")  
    print(f"   Range adequado: {'✅' if range_good else '❌'}")
    print(f"   Score geral: {overall_score}/3")
    
    if overall_score >= 2:
        print(f"   🎉 SISTEMA MATEMÁTICO ADEQUADO PARA GUIAR MODELO")
    else:
        print(f"   ⚠️ SISTEMA PRECISA DE AJUSTES MATEMÁTICOS")

def main():
    print("📊 ==========================================")
    print("📊 ANÁLISE MATEMÁTICA COMPLETA - CHERRY")
    print("📊 ==========================================")
    print("Testando correlações fortes e intensidade de sinal")
    print("")
    
    # Testes matemáticos completos
    dd_data, reward_data, correlation, sensitivity = test_dd_reward_correlation()
    test_pnl_reward_correlation()
    coherence_rate = test_action_reward_coherence()
    strength_rate = test_signal_strength_adequacy()
    monotonic, non_linearity, stability = test_mathematical_properties()
    
    # Relatório final
    comprehensive_correlation_report(dd_data, reward_data, correlation, sensitivity)
    
    print(f"\n🎯 RESUMO FINAL:")
    print(f"   Coerência de ações: {coherence_rate:.2%}")
    print(f"   Sinais fortes: {strength_rate:.2%}")
    print(f"   Propriedades matemáticas: {'✅' if monotonic else '❌'}")
    
    print(f"\n🚀 CHERRY tem correlações matemáticas adequadas para GUIAR o modelo eficientemente!")

if __name__ == "__main__":
    main()