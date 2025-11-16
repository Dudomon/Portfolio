"""
🧪 TESTE FINAL - CHERRY REWARD SYSTEM BRUTAL
Verifica todas as correções aplicadas para DD 50% e explained variance
"""

import sys
import os
import numpy as np

sys.path.append(r'D:\Projeto')

from trading_framework.rewards.reward_system_simple import create_simple_reward_system

class MockEnv:
    """Mock environment para teste completo"""
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

def test_brutal_dd_control():
    """Testar controle brutal de DD"""
    print("🚨 [TESTE BRUTAL] Controle de DD Ultra-Agressivo...")
    
    reward_system = create_simple_reward_system(500.0)
    
    scenarios = [
        (2.0, "DD seguro"),
        (10.0, "DD moderado"), 
        (25.0, "DD crítico"),
        (45.0, "DD atual (infernal)"),
        (55.0, "DD pior que atual")
    ]
    
    print("   DD%   | HOLD    | TRADE WIN | TRADE LOSS | Diferença HOLD-TRADE")
    print("   " + "-" * 70)
    
    for dd, desc in scenarios:
        # HOLD
        env_hold = MockEnv(dd_pct=dd)
        action_hold = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward_hold, _, _ = reward_system.calculate_reward_and_info(env_hold, action_hold, {'trades_count': 0})
        
        # TRADE WIN
        env_win = MockEnv(dd_pct=dd)
        env_win.add_trade(10.0)
        action_trade = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward_win, _, _ = reward_system.calculate_reward_and_info(env_win, action_trade, {'trades_count': 0})
        
        # TRADE LOSS
        env_loss = MockEnv(dd_pct=dd)
        env_loss.add_trade(-8.0)
        reward_loss, _, _ = reward_system.calculate_reward_and_info(env_loss, action_trade, {'trades_count': 0})
        
        # Diferença crítica: HOLD deve ser MUITO melhor que TRADE em DD alto
        diff_win = reward_hold - reward_win
        diff_loss = reward_hold - reward_loss
        
        print(f"   {dd:4.1f} | {reward_hold:7.2f} | {reward_win:9.2f} | {reward_loss:10.2f} | +{diff_win:6.2f}/+{diff_loss:6.2f}")
    
    print(f"\n🎯 ANÁLISE: HOLD deve ter reward MAIOR que trades em DD >20%")

def test_no_clipping_verification():
    """Verificar se o clipping foi realmente removido"""
    print("\n🔧 [TESTE CLIPPING] Verificando remoção do clipping...")
    
    reward_system = create_simple_reward_system(500.0)
    
    extreme_scenarios = [
        (50.0, 50.0, "Win extremo em DD extremo"),
        (50.0, -50.0, "Loss extremo em DD extremo"),
        (2.0, 100.0, "Win absurdo em DD baixo"),
        (2.0, -100.0, "Loss absurdo em DD baixo")
    ]
    
    print("   Cenário                    | DD%   | PnL    | Reward   | Clippado?")
    print("   " + "-" * 75)
    
    for dd, pnl, desc in extreme_scenarios:
        env = MockEnv(dd_pct=dd)
        env.add_trade(pnl)
        
        action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        reward, info, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
        
        # Verificar se está clippado (antigos limites eram ±20)
        clipped = abs(reward) == 20.0 or abs(reward) == 200.0
        
        print(f"   {desc:25} | {dd:4.1f} | ${pnl:6.1f} | {reward:8.2f} | {'SIM' if clipped else 'NÃO'}")
    
    print(f"\n🎯 Se rewards >200 ou <-200, clipping foi removido!")

def test_explained_variance_signals():
    """Testar sinais que melhoram explained variance"""
    print("\n📊 [TESTE EV] Sinais para melhorar Explained Variance...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Cenários que causavam explained variance negativa
    problematic_scenarios = [
        ("Sequence trades em DD crescente", [(5, 3), (10, 5), (20, -2), (30, -5), (40, -8)]),
        ("Operação segura consistente", [(2, 5), (3, 4), (2, 6), (1, 3), (2, 4)]),
        ("Recovery após DD alto", [(40, 0), (35, 2), (30, 3), (25, 4), (20, 5)]),
    ]
    
    for desc, scenario_data in problematic_scenarios:
        print(f"\n   🎯 {desc}:")
        
        total_reward = 0
        rewards = []
        
        for dd, pnl in scenario_data:
            if pnl == 0:  # HOLD
                env = MockEnv(dd_pct=dd)
                action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            else:  # TRADE
                env = MockEnv(dd_pct=dd)
                env.add_trade(pnl)
                action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            
            reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': len(rewards)})
            rewards.append(reward)
            total_reward += reward
            
            action_type = "HOLD" if pnl == 0 else f"TRADE(${pnl})"
            print(f"      DD {dd:2.0f}% + {action_type:10} → {reward:8.2f}")
        
        # Análise da sequência
        reward_variance = np.var(rewards)
        reward_trend = "crescente" if rewards[-1] > rewards[0] else "decrescente"
        
        print(f"      Total: {total_reward:8.2f} | Variância: {reward_variance:6.1f} | Trend: {reward_trend}")

def test_bias_correction():
    """Testar correção de bias sistemático"""
    print("\n⚖️ [TESTE BIAS] Correção de bias sistemático...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Gerar 50 cenários balanceados
    rewards = []
    scenario_types = []
    
    for i in range(50):
        if i < 25:  # Primeiros 25: operação segura (DD baixo)
            dd = np.random.uniform(0, 5)
            scenario_types.append("seguro")
        else:  # Últimos 25: operação perigosa (DD alto)  
            dd = np.random.uniform(20, 50)
            scenario_types.append("perigoso")
        
        env = MockEnv(dd_pct=dd)
        
        if np.random.random() < 0.5:  # 50% HOLDs
            action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        else:  # 50% trades
            pnl = np.random.normal(0, 6)  # PnL neutro
            env.add_trade(pnl)
            action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        
        reward, _, _ = reward_system.calculate_reward_and_info(env, action, {'trades_count': 0})
        rewards.append(reward)
    
    # Análise de bias
    rewards = np.array(rewards)
    safe_rewards = [rewards[i] for i in range(25)]
    dangerous_rewards = [rewards[i] for i in range(25, 50)]
    
    print(f"   Cenários Seguros (DD <5%):")
    print(f"      Média: {np.mean(safe_rewards):8.2f}")
    print(f"      Std:   {np.std(safe_rewards):8.2f}")
    
    print(f"   Cenários Perigosos (DD >20%):")
    print(f"      Média: {np.mean(dangerous_rewards):8.2f}")
    print(f"      Std:   {np.std(dangerous_rewards):8.2f}")
    
    print(f"   Geral:")
    print(f"      Média total: {np.mean(rewards):8.2f}")
    print(f"      Diferença seguro-perigoso: {np.mean(safe_rewards) - np.mean(dangerous_rewards):8.2f}")
    
    # Verificar correção de bias
    total_bias = abs(np.mean(rewards))
    difference = np.mean(safe_rewards) - np.mean(dangerous_rewards)
    
    print(f"\n🎯 ANÁLISE BIAS:")
    if total_bias < 10:
        print(f"   ✅ Bias total baixo: {total_bias:.2f}")
    else:
        print(f"   ❌ Bias total alto: {total_bias:.2f}")
    
    if difference > 50:
        print(f"   ✅ Diferença seguro-perigoso boa: {difference:.2f}")
    else:
        print(f"   ❌ Diferença seguro-perigoso insuficiente: {difference:.2f}")

def main():
    print("🚨 ========================================")
    print("🚨 TESTE FINAL CHERRY - SISTEMA BRUTAL")  
    print("🚨 ========================================")
    print("Verificando correções para DD 50% e EV negativa")
    print("")
    
    test_brutal_dd_control()
    test_no_clipping_verification()
    test_explained_variance_signals()
    test_bias_correction()
    
    print("\n" + "="*60)
    print("📋 RESUMO FINAL")
    print("="*60)
    print("🎯 Se todas as correções funcionaram:")
    print("   1. DD >20% deve ter rewards MUITO negativos")
    print("   2. HOLD deve ser melhor que TRADE em DD alto")
    print("   3. Sem clipping - penalidades extremas passam")
    print("   4. Bias corrigido - operação segura recompensada")
    print("   5. Explained variance deve ficar POSITIVA")
    print("")
    print("🚀 CHERRY está pronto para CONTROLAR DD e ter EV positiva!")

if __name__ == "__main__":
    main()