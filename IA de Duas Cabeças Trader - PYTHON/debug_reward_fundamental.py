"""
🚨 DEBUG FUNDAMENTAL - REWARD SYSTEM QUEBRADO
Investigar por que DD 50% e explained variance negativo persistem
"""

import sys
import os
import numpy as np

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

def debug_reward_components():
    """Debugar componentes do reward em cenário real"""
    print("🚨 [DEBUG] Analisando componentes detalhados...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Cenário REAL: DD alto + trade perdedor (comum no treino problemático)
    env = MockEnv(dd_pct=45.0)  # DD absurdo
    env.add_trade(-8.0)  # Loss de $8
    
    action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])  # LONG action
    old_state = {'trades_count': 0}
    
    reward, info, done = reward_system.calculate_reward_and_info(env, action, old_state)
    
    print(f"🎯 CENÁRIO: DD 45% + Loss $8")
    print(f"   Total reward: {reward:.2f}")
    print(f"   Done flag: {done}")
    print("")
    
    print("📊 COMPONENTES DETALHADOS:")
    if 'components' in info:
        total_magnitude = 0
        for comp_name, comp_value in info['components'].items():
            print(f"   {comp_name:30} : {comp_value:8.2f}")
            total_magnitude += abs(comp_value)
        print(f"   {'TOTAL MAGNITUDE':30} : {total_magnitude:8.2f}")
    
    print("\n📊 TARGET ANALYSIS:")
    if 'target_analysis' in info:
        for comp_name, comp_value in info['target_analysis'].items():
            if isinstance(comp_value, (int, float)):
                print(f"   {comp_name:30} : {comp_value:8.2f}")
    
    return reward, info

def test_hold_vs_trade_bias():
    """Testar se há bias sistemático HOLD vs TRADE"""
    print("\n🧠 [DEBUG] Testando bias HOLD vs TRADE...")
    
    reward_system = create_simple_reward_system(500.0)
    
    scenarios = [
        # (action, dd, trade_pnl, description)
        ([0.0, 0.8, 0.0, 0.0, 0.0, 0.0], 45.0, None, "HOLD em DD alto"),
        ([1.0, 0.8, 0.0, 0.0, 0.0, 0.0], 45.0, 5.0, "LONG win em DD alto"),
        ([1.0, 0.8, 0.0, 0.0, 0.0, 0.0], 45.0, -5.0, "LONG loss em DD alto"),
        ([0.0, 0.8, 0.0, 0.0, 0.0, 0.0], 2.0, None, "HOLD em DD baixo"),
        ([1.0, 0.8, 0.0, 0.0, 0.0, 0.0], 2.0, 5.0, "LONG win em DD baixo"),
        ([1.0, 0.8, 0.0, 0.0, 0.0, 0.0], 2.0, -5.0, "LONG loss em DD baixo"),
    ]
    
    print("   Ação                | DD%   | Reward | Deveria Incentivar?")
    print("   " + "-" * 60)
    
    hold_high_dd = None
    trade_high_dd_win = None
    trade_high_dd_loss = None
    
    for action_vals, dd, trade_pnl, desc in scenarios:
        env = MockEnv(dd_pct=dd)
        if trade_pnl is not None:
            env.add_trade(trade_pnl)
            
        action = np.array(action_vals)
        old_state = {'trades_count': 0 if trade_pnl is None else 0}
        
        reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
        
        # Determinar o que DEVERIA ser incentivado
        if dd > 20:  # DD alto
            should_incentivize = "HOLD" if action_vals[0] == 0 else "NÃO"
            if desc == "HOLD em DD alto":
                hold_high_dd = reward
            elif "win em DD alto" in desc:
                trade_high_dd_win = reward
            elif "loss em DD alto" in desc:
                trade_high_dd_loss = reward
        else:  # DD baixo
            should_incentivize = "Qualquer ação inteligente"
        
        print(f"   {desc:18} | {dd:4.1f}% | {reward:6.2f} | {should_incentivize}")
    
    print(f"\n🎯 ANÁLISE CRÍTICA:")
    
    # Problema 1: HOLD em DD alto deveria ter reward MUITO maior que trade
    if hold_high_dd is not None and trade_high_dd_loss is not None:
        if hold_high_dd <= trade_high_dd_loss:
            print(f"❌ ERRO 1: HOLD ({hold_high_dd:.2f}) não é mais recompensado que TRADE LOSS ({trade_high_dd_loss:.2f}) em DD alto!")
        else:
            print(f"✅ OK 1: HOLD ({hold_high_dd:.2f}) > TRADE LOSS ({trade_high_dd_loss:.2f}) em DD alto")
    
    # Problema 2: Wins em DD alto não deveriam ser tão recompensados
    if trade_high_dd_win is not None and hold_high_dd is not None:
        if trade_high_dd_win > hold_high_dd:
            print(f"❌ ERRO 2: TRADE WIN ({trade_high_dd_win:.2f}) ainda mais recompensado que HOLD ({hold_high_dd:.2f}) em DD alto!")
        else:
            print(f"✅ OK 2: HOLD ({hold_high_dd:.2f}) >= TRADE WIN ({trade_high_dd_win:.2f}) em DD alto")

def test_reward_variance():
    """Testar variância de rewards - problema de explained variance"""
    print("\n📊 [DEBUG] Testando variância de rewards...")
    
    reward_system = create_simple_reward_system(500.0)
    
    # Gerar 100 cenários típicos
    rewards = []
    
    for i in range(100):
        # Cenários aleatórios típicos de treino
        dd = np.random.uniform(0, 50)  # DD entre 0-50%
        
        if np.random.random() < 0.7:  # 70% HOLDs
            action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            env = MockEnv(dd_pct=dd)
        else:  # 30% trades
            action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            env = MockEnv(dd_pct=dd)
            pnl = np.random.normal(0, 8)  # PnL com média 0, std 8
            env.add_trade(pnl)
        
        old_state = {'trades_count': 0}
        reward, _, _ = reward_system.calculate_reward_and_info(env, action, old_state)
        rewards.append(reward)
    
    rewards = np.array(rewards)
    
    print(f"   Média: {np.mean(rewards):.2f}")
    print(f"   Std Dev: {np.std(rewards):.2f}")
    print(f"   Min: {np.min(rewards):.2f}")
    print(f"   Max: {np.max(rewards):.2f}")
    print(f"   Range: {np.max(rewards) - np.min(rewards):.2f}")
    
    # Problemas de variância
    if abs(np.mean(rewards)) > 5.0:
        print(f"❌ PROBLEMA: Média muito longe de 0 ({np.mean(rewards):.2f}) - bias sistemático!")
    
    if np.std(rewards) > 15.0:
        print(f"❌ PROBLEMA: Std dev muito alta ({np.std(rewards):.2f}) - rewards instáveis!")
    
    if np.max(rewards) - np.min(rewards) > 30.0:
        print(f"❌ PROBLEMA: Range muito grande ({np.max(rewards) - np.min(rewards):.2f}) - extremos demais!")
    
    # Explained variance seria negativa se:
    # 1. Média dos rewards é inconsistente com as ações
    # 2. Variância é muito alta 
    # 3. Há bias sistemático não relacionado ao desempenho real
    
    return rewards

def test_critical_scenarios():
    """Testar cenários críticos que podem quebrar explained variance"""
    print("\n💥 [DEBUG] Testando cenários críticos...")
    
    reward_system = create_simple_reward_system(500.0)
    
    critical_scenarios = [
        # Cenários que podem confundir o modelo
        ("Trade win massivo em DD 50%", 50.0, 25.0),
        ("Trade loss pequeno em DD 50%", 50.0, -2.0),
        ("HOLD prolongado sem trades", 0.0, None),
        ("Sequence de losses", 25.0, [-5, -3, -8, -4]),
        ("Sequence de wins pequenos", 15.0, [2, 1, 3, 2]),
    ]
    
    for desc, dd, pnl_data in critical_scenarios:
        print(f"\n   🎯 {desc}:")
        
        env = MockEnv(dd_pct=dd)
        
        if pnl_data is None:
            # HOLD scenario
            action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            old_state = {'trades_count': 0}
            reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
            print(f"      HOLD reward: {reward:.2f}")
            
        elif isinstance(pnl_data, list):
            # Sequence scenario
            total_reward = 0
            for i, pnl in enumerate(pnl_data):
                env.add_trade(pnl)
                action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
                old_state = {'trades_count': i}
                reward, _, _ = reward_system.calculate_reward_and_info(env, action, old_state)
                total_reward += reward
                print(f"      Trade {i+1} (${pnl}): {reward:.2f}")
            print(f"      Total sequence: {total_reward:.2f}")
            
        else:
            # Single trade scenario
            env.add_trade(pnl_data)
            action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])
            old_state = {'trades_count': 0}
            reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
            print(f"      Trade reward: {reward:.2f}")
            
            # Verificar componentes críticos
            risk_component = info['components'].get('aggressive_risk_control', 0)
            pnl_component = info['components'].get('pnl_direct', 0)
            print(f"      Risk penalty: {risk_component:.2f}")
            print(f"      PnL component: {pnl_component:.2f}")

def main():
    print("🚨 ================================")
    print("🚨 DEBUG FUNDAMENTAL - REWARD SYSTEM")  
    print("🚨 ================================")
    
    print("Problema: DD 50% + Explained Variance negativo")
    print("Hipóteses:")
    print("1. Reward system não está penalizando adequadamente")
    print("2. Bias sistemático HOLD vs TRADE")
    print("3. Variância de rewards muito alta/instável")
    print("4. Componentes conflitantes")
    print("")
    
    # Debug detalhado
    reward, info = debug_reward_components()
    test_hold_vs_trade_bias()
    rewards_array = test_reward_variance()
    test_critical_scenarios()
    
    print(f"\n🎯 CONCLUSÕES:")
    print(f"Se o sistema ainda permite DD 50%, significa que:")
    print(f"1. As penalidades não são suficientemente agressivas")
    print(f"2. O modelo encontra formas de 'contornar' as penalidades")
    print(f"3. Outros componentes do reward estão dominando o sinal de risco")
    print(f"4. O valor function não está aprendendo corretamente")

if __name__ == "__main__":
    main()