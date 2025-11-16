#!/usr/bin/env python3
"""
🔍 DEBUG: POR QUE REWARDS SÃO ZERO?
Investigar se o problema é na execução de trades ou no cálculo de rewards
"""
import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd

def debug_zero_rewards():
    """Debug específico para descobrir por que rewards são zero"""
    
    print("🔍 DEBUG: POR QUE REWARDS SÃO ZERO?")
    print("=" * 60)
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        from daytrader import TradingEnv
        
        # Dataset pequeno para debug
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
        
        # Dataset mínimo
        test_df = df.head(200).copy()  # Mais barras para tentar forçar trades
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
        
        obs = env.reset()
        
        # Acessar reward system diretamente
        reward_calc = env.reward_system
        print(f"📊 Reward system: {type(reward_calc)}")
        
        # TESTE 1: Verificar como o reward system é chamado
        print("\n🧪 TESTE 1: COMO O REWARD SYSTEM É CHAMADO")
        print("-" * 50)
        
        # Simular old_state como o env faria
        old_state = {
            'trades_count': len(getattr(env, 'trades', [])),
            'observation': obs
        }
        
        print(f"   Old trades count: {old_state['trades_count']}")
        
        # Tentar ação de HOLD primeiro
        print("\n   TESTE HOLD DIRETO:")
        hold_action = np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        try:
            # Chamar reward system diretamente
            direct_reward, direct_info, direct_done = reward_calc.calculate_reward_and_info(
                env, hold_action, old_state
            )
            print(f"      Direct reward call: {direct_reward:.6f}")
            print(f"      Direct components: {direct_info.get('reward_components', {})}")
        except Exception as e:
            print(f"      ❌ Direct reward call error: {e}")
        
        # Usar step normal
        obs, step_reward, done, step_info = env.step(hold_action)
        print(f"      Step reward: {step_reward:.6f}")
        print(f"      Step components: {step_info.get('reward_components', {})}")
        
        # TESTE 2: Tentar forçar trade mais agressivo
        print("\n🧪 TESTE 2: FORÇAR TRADE MAIS AGRESSIVO")
        print("-" * 50)
        
        # Ações mais extremas
        extreme_actions = [
            np.array([1, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # LONG máximo
            np.array([2, 1.0, -1.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32), # SHORT máximo
            np.array([1, 0.9, 0.8, 0.9, 0.7, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # LONG alto
            np.array([2, 0.9, -0.8, 0.9, -0.7, 0, 0, 0, 0, 0, 0], dtype=np.float32), # SHORT alto
        ]
        
        for i, action in enumerate(extreme_actions):
            old_trades = len(getattr(env, 'trades', []))
            old_portfolio = env.portfolio_value
            
            print(f"   Tentativa {i+1}: action={action[:5]}")
            
            obs, reward, done, info = env.step(action)
            
            new_trades = len(getattr(env, 'trades', []))
            new_portfolio = env.portfolio_value
            
            print(f"      Reward: {reward:.6f}")
            print(f"      Portfolio: ${old_portfolio:.2f} → ${new_portfolio:.2f} (Δ=${new_portfolio-old_portfolio:.3f})")
            print(f"      Trades: {old_trades} → {new_trades} (Δ={new_trades-old_trades})")
            
            if new_trades > old_trades:
                print(f"      ✅ TRADE EXECUTADO!")
                last_trade = env.trades[-1]
                print(f"         Trade data: {last_trade}")
                break
            else:
                print(f"      ❌ Nenhum trade executado")
            
            if done:
                print(f"      Episode ended")
                break
        
        # TESTE 3: Verificar thresholds e validações do ambiente
        print(f"\n🔬 TESTE 3: VERIFICAR THRESHOLDS DO AMBIENTE")
        print("-" * 50)
        
        print(f"   Current step: {getattr(env, 'current_step', 'N/A')}")
        print(f"   Max steps: {getattr(env, 'MAX_STEPS', 'N/A')}")
        print(f"   Window size: {getattr(env, 'window_size', 'N/A')}")
        print(f"   Can trade: {env.current_step >= env.window_size if hasattr(env, 'current_step') and hasattr(env, 'window_size') else 'N/A'}")
        
        # Verificar posições atuais
        if hasattr(env, 'positions'):
            print(f"   Current positions: {env.positions}")
        if hasattr(env, 'current_positions'):
            print(f"   Current positions (alt): {env.current_positions}")
        
        # TESTE 4: Simular reward calculation manual
        print(f"\n🧪 TESTE 4: SIMULAR REWARD CALCULATION MANUAL")
        print("-" * 50)
        
        # Criar trade fake para testar reward
        fake_trade = {
            'pnl_usd': 0.05,  # $0.05 lucro
            'pnl': 0.05,
            'entry_price': 2000.0,
            'exit_price': 2002.5,
            'quantity': 0.02,
            'side': 'buy',
            'duration_steps': 10,
            'position_size': 0.02
        }
        
        # Simular ambiente com este trade
        env.trades = [fake_trade]  # Forçar trade
        
        # Recalcular reward
        new_old_state = {
            'trades_count': 0,  # Antes não tinha trade
            'observation': obs
        }
        
        try:
            manual_reward, manual_info, manual_done = reward_calc.calculate_reward_and_info(
                env, hold_action, new_old_state
            )
            print(f"   Manual reward com trade fake: {manual_reward:.6f}")
            print(f"   Manual components: {manual_info.get('reward_components', {})}")
            
            # Calcular PnL expected
            pnl = fake_trade['pnl_usd']
            pnl_percent = pnl / 500.0  # initial balance
            expected_pnl_reward = pnl_percent * 1000.0  # pnl_direct weight
            print(f"   Expected PnL reward: PnL={pnl} → {pnl_percent:.6f}% → {expected_pnl_reward:.6f}")
            
        except Exception as e:
            print(f"   ❌ Manual reward error: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_zero_rewards()
    print(f"\n{'✅ DEBUG ZERO REWARDS CONCLUÍDO' if success else '❌ DEBUG ZERO REWARDS FALHOU'}")