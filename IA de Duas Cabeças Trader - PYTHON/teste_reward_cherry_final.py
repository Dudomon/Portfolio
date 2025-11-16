#!/usr/bin/env python3
"""
🍒 TESTE FINAL - VALIDAÇÃO DAS CORREÇÕES REWARD CHERRY
=====================================================

Teste rápido focado nos indicadores principais após correções
"""

import sys
import os
import numpy as np
sys.path.append("D:/Projeto")

def test_reward_balance_final():
    """Teste final - apenas indicadores principais"""
    print("🍒 TESTE FINAL - VALIDAÇÃO CORREÇÕES REWARD CHERRY")
    print("=" * 60)
    
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import load_optimized_data_original, TradingEnv
        from trading_framework.rewards.reward_system_simple import SimpleRewardCalculator
        
        # Setup rápido
        data = load_optimized_data_original()
        data = data.iloc[-1000:].reset_index(drop=True)
        
        env = TradingEnv(df=data, window_size=20, is_training=True, initial_balance=500.0)
        reward_calculator = SimpleRewardCalculator(initial_balance=500.0)
        
        print("✅ Setup completo")
        
        # TESTE 1: Balance de componentes
        print("\n🔍 TESTE BALANCE CORRIGIDO:")
        print("-" * 40)
        
        scenarios = [
            {"name": "Trade +$50", "pnl": 50, "action": [1.0, 0.8, 0, 0]},
            {"name": "Trade -$30", "pnl": -30, "action": [1.5, 0.7, 0, 0]}, 
            {"name": "HOLD Smart", "pnl": 0, "action": [0.2, 0.5, 0, 0]},
        ]
        
        rewards = []
        for scenario in scenarios:
            obs = env.reset()
            old_state = {"portfolio_total_value": env.portfolio_value, "trades_count": len(env.trades)}
            
            if scenario['pnl'] != 0:
                trade_data = {'pnl_usd': scenario['pnl'], 'type': 'long', 'entry_price': 2000.0, 'exit_price': 2000.0 + scenario['pnl']/0.02/100, 'sl_points': 15, 'tp_points': 25, 'duration_hours': 2.0}
                env.trades.append(trade_data)
                env.portfolio_value += scenario['pnl']
            
            action = np.array(scenario['action'])
            reward, info, _ = reward_calculator.calculate_reward_and_info(env, action, old_state)
            
            rewards.append(reward)
            print(f"  {scenario['name']:12}: {reward:7.2f}")
        
        # Calcular balance score
        reward_std = np.std(rewards)
        balance_score = max(0, 1.0 - reward_std/10.0)
        
        print(f"\nBalance Score: {balance_score:.3f} (era -7.109)")
        
        # TESTE 2: Correlação simulada
        print("\n🔍 TESTE CORRELAÇÃO CORRIGIDA:")
        print("-" * 40)
        
        pnls = [50, 20, -10, -40, 30]
        test_rewards = []
        
        for pnl in pnls:
            obs = env.reset()
            reward_calculator.reset()
            old_state = {"portfolio_total_value": env.portfolio_value, "trades_count": len(env.trades)}
            
            if pnl != 0:
                trade_data = {'pnl_usd': pnl, 'type': 'long', 'entry_price': 2000.0, 'exit_price': 2000.0 + pnl/0.02/100}
                env.trades.append(trade_data) 
                env.portfolio_value = 500 + pnl
            
            action = np.array([1.0, 0.8, 0, 0])
            reward, _, _ = reward_calculator.calculate_reward_and_info(env, action, old_state)
            test_rewards.append(reward)
        
        correlation = np.corrcoef(pnls, test_rewards)[0,1]
        print(f"Correlação PnL vs Reward: {correlation:.3f} (era 0.072)")
        
        # TESTE 3: HOLD vs TRADING balance
        print("\n🔍 TESTE HOLD vs TRADING:")
        print("-" * 40)
        
        # HOLD action
        obs = env.reset()
        old_state = {"portfolio_total_value": env.portfolio_value, "trades_count": len(env.trades)}
        action_hold = np.array([0.2, 0.5, 0, 0])
        reward_hold, _, _ = reward_calculator.calculate_reward_and_info(env, action_hold, old_state)
        
        # TRADE action (breakeven)
        obs = env.reset()
        env.portfolio_value = 500  # Breakeven
        old_state = {"portfolio_total_value": env.portfolio_value, "trades_count": len(env.trades)}
        trade_data = {'pnl_usd': 0, 'type': 'long', 'entry_price': 2000.0, 'exit_price': 2000.0}
        env.trades.append(trade_data)
        action_trade = np.array([1.0, 0.8, 0, 0])
        reward_trade, _, _ = reward_calculator.calculate_reward_and_info(env, action_trade, old_state)
        
        print(f"  HOLD reward: {reward_hold:.2f}")
        print(f"  TRADE reward (breakeven): {reward_trade:.2f}")
        print(f"  Diferença: {abs(reward_hold - reward_trade):.2f} (menor = melhor balance)")
        
        # AVALIAÇÃO FINAL
        print("\n📊 AVALIAÇÃO FINAL:")
        print("=" * 40)
        
        # Critérios de aprovação
        balance_ok = balance_score > 0.5  # Era -7.109, agora deve ser > 0.5
        correlation_ok = correlation > 0.5  # Era 0.072, agora deve ser > 0.5
        hold_balance_ok = abs(reward_hold - reward_trade) < 3.0  # Diferença razoável
        
        print(f"✅ Balance: {'APROVADO' if balance_ok else 'REPROVADO'} ({balance_score:.3f})")
        print(f"✅ Correlação: {'APROVADO' if correlation_ok else 'REPROVADO'} ({correlation:.3f})")  
        print(f"✅ HOLD Balance: {'APROVADO' if hold_balance_ok else 'REPROVADO'} ({abs(reward_hold - reward_trade):.2f})")
        
        all_approved = balance_ok and correlation_ok and hold_balance_ok
        
        if all_approved:
            print("\n🎉 TODAS AS CORREÇÕES APROVADAS!")
            print("Sistema reward Cherry está balanceado e pronto para uso.")
        else:
            print("\n⚠️ Algumas correções ainda precisam de ajuste.")
        
        return all_approved
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_reward_balance_final()
    print(f"\n{'✅ SISTEMA CORRIGIDO' if success else '❌ PRECISA AJUSTES'}")