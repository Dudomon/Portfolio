#!/usr/bin/env python3
"""
🔍 DEBUG ESPECÍFICO: IDENTIFICAR FONTE DO REWARD CONSTANTE
Investigar exatamente qual componente gera o reward ~0.845-0.9531
"""
import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd

def debug_constant_reward():
    """Debug específico para encontrar a fonte do reward constante"""
    
    print("🔍 DEBUG: IDENTIFICAR FONTE DO REWARD CONSTANTE")
    print("=" * 60)
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        from daytrader import TradingEnv
        
        # Dataset pequeno para debug específico
        dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
        df = pd.read_csv(dataset_path)
        
        # Processar dataset
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)
            df.drop('time', axis=1, inplace=True)
        
        df = df.rename(columns={
            'open': 'open_5m',
            'high': 'high_5m',
            'low': 'low_5m', 
            'close': 'close_5m',
            'tick_volume': 'volume_5m'
        })
        
        # Dataset mínimo para debug
        test_df = df.head(100).copy()
        print(f"📊 Dataset debug: {len(test_df):,} barras")
        
        # Configurar ambiente
        trading_params = {
            'base_lot_size': 0.02,
            'max_lot_size': 0.03,
            'initial_balance': 500.0,
            'target_trades_per_day': 18,
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        env = TradingEnv(
            test_df,
            window_size=20,
            is_training=True,
            initial_balance=500.0,
            trading_params=trading_params
        )
        
        print("✅ Ambiente debug criado")
        
        # TESTE ESPECÍFICO: FORÇAR TRADE E ANALISAR CADA COMPONENTE
        print("\n🎯 ANÁLISE COMPONENTE POR COMPONENTE")
        print("-" * 50)
        
        obs = env.reset()
        
        # Verificar inicial balance
        print(f"📊 Initial balance: {env.portfolio_value}")
        
        # Tentar acessar reward calculator de diferentes formas
        reward_calc = None
        if hasattr(env, 'reward_calculator'):
            reward_calc = env.reward_calculator
        elif hasattr(env, 'reward_system'):
            reward_calc = env.reward_system
            print(f"📊 Using reward_system: {type(reward_calc)}")
        elif hasattr(env, 'unified_reward_system'):
            reward_calc = env.unified_reward_system 
            print(f"📊 Using unified_reward_system: {type(reward_calc)}")
        else:
            # Ver atributos relacionados a reward
            reward_attrs = [attr for attr in dir(env) if 'reward' in attr.lower()]
            print(f"📋 Reward-related attributes: {reward_attrs}")
        
        if reward_calc:
            print(f"📊 Reward calc type: {type(reward_calc)}")
            if hasattr(reward_calc, 'initial_balance'):
                print(f"📊 Reward calculator initial_balance: {reward_calc.initial_balance}")
            else:
                print(f"📋 Reward calc attributes: {[attr for attr in dir(reward_calc) if not attr.startswith('_')]}")
        
        # TESTE 1: Verificar continuous feedback (sem trade)
        print("\n🧪 TESTE 1: CONTINUOUS FEEDBACK (HOLD)")
        hold_action = np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs, reward_hold, done, info = env.step(hold_action)
        
        print(f"   HOLD reward: {reward_hold:.6f}")
        if 'reward_components' in info:
            print("   HOLD components:")
            for k, v in info['reward_components'].items():
                print(f"      {k}: {v:.6f}")
        
        # TESTE 2: Forçar trade e analisar cada componente
        print("\n🧪 TESTE 2: FORÇAR TRADE ESPECÍFICO")
        
        # Forçar LONG
        long_action = np.array([1, 0.8, 0.3, 0.7, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Capturar estado antes
        old_trades = len(getattr(env, 'trades', []))
        old_portfolio = env.portfolio_value
        
        obs, reward_trade, done, info = env.step(long_action)
        
        new_trades = len(getattr(env, 'trades', []))
        new_portfolio = env.portfolio_value
        
        print(f"   LONG reward: {reward_trade:.6f}")
        print(f"   Portfolio change: ${new_portfolio - old_portfolio:.3f}")
        print(f"   New trades: {new_trades - old_trades}")
        
        if 'reward_components' in info:
            print("   TRADE components:")
            for k, v in info['reward_components'].items():
                print(f"      {k}: {v:.6f}")
        
        # Se houve trade, analisar dados do trade
        if new_trades > old_trades:
            last_trade = env.trades[-1]
            print(f"\n📋 DADOS DO TRADE:")
            for key, value in last_trade.items():
                print(f"      {key}: {value}")
            
            # ANÁLISE ESPECÍFICA DO PnL REWARD CALCULATION
            print(f"\n🔬 ANÁLISE ESPECÍFICA DO CÁLCULO PnL:")
            
            pnl_fields = ['pnl_usd', 'pnl', 'profit_loss', 'pl']
            pnl = None
            for field in pnl_fields:
                if field in last_trade and last_trade[field] is not None:
                    pnl = float(last_trade[field])
                    print(f"      PnL source: {field} = {pnl:.6f}")
                    break
            
            if pnl is None:
                # Calcular PnL manualmente
                entry = last_trade.get('entry_price', 0)
                exit_price = last_trade.get('exit_price', 0) 
                quantity = last_trade.get('quantity', 0.01)
                side = last_trade.get('side', 'buy')
                
                if entry > 0 and exit_price > 0:
                    if side == 'buy':
                        pnl = (exit_price - entry) * quantity
                    else:
                        pnl = (entry - exit_price) * quantity
                    print(f"      PnL calculado: entry={entry}, exit={exit_price}, qty={quantity}, side={side} → PnL={pnl:.6f}")
                else:
                    pnl = 0.0
                    print(f"      PnL = 0.0 (dados insuficientes)")
            
            # Calcular reward esperado baseado apenas em PnL
            initial_balance = reward_calc.initial_balance if reward_calc else 500.0
            pnl_percent = pnl / initial_balance if initial_balance > 0 else 0
            expected_pnl_reward = pnl_percent * 1000.0  # peso pnl_direct
            
            print(f"      PnL percent: {pnl_percent:.8f}")
            print(f"      Expected PnL reward: {expected_pnl_reward:.6f}")
            print(f"      Actual total reward: {reward_trade:.6f}")
            print(f"      Difference: {reward_trade - expected_pnl_reward:.6f}")
        
        # TESTE 3: Análise direta do reward calculator
        print(f"\n🔬 ANÁLISE DIRETA DO REWARD CALCULATOR")
        print("-" * 50)
        
        if reward_calc:
            print(f"   Reward system attributes:")
            attrs = [attr for attr in dir(reward_calc) if not attr.startswith('_')]
            print(f"      {attrs[:10]}...")  # Show first 10 attrs
            
            if hasattr(reward_calc, 'base_weights'):
                print(f"   Base weights:")
                for k, v in reward_calc.base_weights.items():
                    if v != 0.0:
                        print(f"      {k}: {v}")
            else:
                print("   ⚠️ Reward system não tem 'base_weights'")
        else:
            print("   ⚠️ Reward calculator não acessível diretamente")
        
        # TESTE 4: Verificar se há reward constante em continuous feedback
        print(f"\n🧪 TESTE 4: MÚLTIPLOS HOLDS PARA VERIFICAR CONSISTÊNCIA")
        print("-" * 50)
        
        hold_rewards = []
        for i in range(5):
            obs, reward, done, info = env.step(hold_action)
            hold_rewards.append(reward)
            print(f"   HOLD {i+1}: {reward:.6f}")
            
            if 'reward_components' in info:
                non_zero_components = {k: v for k, v in info['reward_components'].items() if abs(v) > 1e-8}
                if non_zero_components:
                    print(f"      Non-zero components: {non_zero_components}")
        
        # Verificar se rewards são constantes
        if len(set([round(r, 6) for r in hold_rewards])) == 1:
            print(f"   ❌ PROBLEMA IDENTIFICADO: Rewards HOLD são constantes = {hold_rewards[0]:.6f}")
        else:
            print(f"   ✅ Rewards HOLD variam: {hold_rewards}")
        
        print(f"\n🎯 DIAGNÓSTICO FINAL:")
        print("-" * 50)
        
        if abs(reward_hold) > 1e-8:
            print(f"   🔍 HOLD rewards não-zero: {reward_hold:.6f}")
            print("   💡 Possível fonte: alive_bonus ou continuous feedback components")
        
        if new_trades > old_trades and abs(reward_trade - expected_pnl_reward) > 0.01:
            print(f"   🔍 Trade reward inconsistente com PnL puro")
            print(f"   💡 Diferença: {reward_trade - expected_pnl_reward:.6f}")
            print("   💡 Possíveis fontes: risk_management, timing, consistency components")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_constant_reward()
    print(f"\n{'✅ DEBUG CONCLUÍDO' if success else '❌ DEBUG FALHOU'}")