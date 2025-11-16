#!/usr/bin/env python3
"""
🔍 DEBUG: INVESTIGAR CORRELAÇÃO REWARD vs PERFORMANCE
Análise profunda do que está causando baixa correlação
"""
import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd

def debug_reward_correlation():
    """Debug específico da correlação reward vs performance"""
    
    print("🔍 DEBUG: CORRELAÇÃO REWARD vs PERFORMANCE")
    print("=" * 60)
    
    try:
        from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator
        from daytrader import TradingEnv
        
        # Dataset pequeno para análise detalhada
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
        
        # Usar subset muito pequeno para debug
        test_df = df.head(1000).copy()
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
        
        # FORÇAR TRADES ESPECÍFICOS para testar reward correlation
        print("\n🎯 TESTE 1: SIMULAÇÃO DE TRADES ESPECÍFICOS")
        print("-" * 50)
        
        obs = env.reset()
        
        # Dados para análise
        rewards_log = []
        portfolio_log = []
        trades_log = []
        actions_log = []
        
        portfolio_log.append(env.portfolio_value)
        
        # Simular sequência específica: alguns holds, depois trades forçados
        test_actions = [
            # Holds iniciais
            *[np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) for _ in range(10)],
            
            # Trades forçados com qualidade alta
            np.array([1, 0.9, 0.5, 0.8, 0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # LONG alto
            np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),      # HOLD
            np.array([2, 0.8, -0.3, 0.7, -0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32), # SHORT alto
            
            # Mais holds
            *[np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) for _ in range(20)],
            
            # Mais trades
            np.array([1, 0.7, 0.3, 0.6, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # LONG médio
            np.array([2, 0.6, -0.4, 0.8, -0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32), # SHORT médio
            
            # Final com holds
            *[np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) for _ in range(50)]
        ]
        
        print(f"   📊 Executando {len(test_actions)} ações específicas...")
        
        for i, action in enumerate(test_actions):
            if i >= 100:  # Limite para não exceder dataset
                break
                
            old_portfolio = env.portfolio_value
            old_trades = len(getattr(env, 'trades', []))
            
            obs, reward, done, info = env.step(action)
            
            new_portfolio = env.portfolio_value
            new_trades = len(getattr(env, 'trades', []))
            
            # Log detalhado
            rewards_log.append(reward)
            portfolio_log.append(new_portfolio)
            
            action_type = "HOLD" if action[0] == 0 else ("LONG" if action[0] == 1 else "SHORT")
            quality = action[1]
            
            actions_log.append({
                'step': i,
                'action_type': action_type,
                'quality': quality,
                'reward': reward,
                'portfolio_before': old_portfolio,
                'portfolio_after': new_portfolio,
                'portfolio_change': new_portfolio - old_portfolio,
                'new_trades': new_trades - old_trades
            })
            
            # Log trades específicos
            if new_trades > old_trades:
                recent_trades = env.trades[-(new_trades - old_trades):]
                for trade in recent_trades:
                    trade_pnl = trade.get('pnl_usd', trade.get('pnl', 0))
                    trades_log.append({
                        'step': i,
                        'pnl': trade_pnl,
                        'reward': reward
                    })
                    print(f"      [TRADE] Step {i}: PnL=${trade_pnl:.3f}, Reward={reward:.4f}")
            
            if done:
                break
        
        # ANÁLISE DETALHADA
        print(f"\n📊 ANÁLISE DETALHADA ({len(rewards_log)} steps)")
        print("-" * 50)
        
        rewards = np.array(rewards_log)
        portfolios = np.array(portfolio_log)
        
        # Portfolio changes
        portfolio_changes = np.diff(portfolios, prepend=portfolios[0])
        
        print(f"   📈 Rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        print(f"   💰 Portfolio changes: mean=${portfolio_changes.mean():.3f}, std=${portfolio_changes.std():.3f}")
        
        # Correlação detalhada
        if len(rewards) > 1 and len(portfolio_changes) > 1:
            correlation = np.corrcoef(rewards[1:], portfolio_changes[1:])[0, 1]
            print(f"   🔗 Correlação Reward vs Portfolio Change: {correlation:.4f}")
            
            # Análise só dos trades
            if trades_log:
                trade_rewards = [t['reward'] for t in trades_log]
                trade_pnls = [t['pnl'] for t in trades_log]
                
                if len(trade_rewards) > 1:
                    trade_correlation = np.corrcoef(trade_rewards, trade_pnls)[0, 1]
                    print(f"   💼 Correlação Trade Rewards vs Trade PnL: {trade_correlation:.4f}")
                    
                    print(f"\n📋 ANÁLISE DE TRADES INDIVIDUAIS:")
                    for i, trade in enumerate(trades_log):
                        expected_reward = trade['pnl'] / 500.0 * 200.0  # Cálculo esperado
                        print(f"      Trade {i+1}: PnL=${trade['pnl']:.3f}, Reward={trade['reward']:.4f}, Expected={expected_reward:.4f}")
            
        # Análise por tipo de ação
        print(f"\n🎮 ANÁLISE POR TIPO DE AÇÃO:")
        
        hold_actions = [a for a in actions_log if a['action_type'] == 'HOLD']
        long_actions = [a for a in actions_log if a['action_type'] == 'LONG'] 
        short_actions = [a for a in actions_log if a['action_type'] == 'SHORT']
        
        print(f"   ⚪ HOLD: {len(hold_actions)} actions")
        if hold_actions:
            hold_rewards = [a['reward'] for a in hold_actions]
            hold_portfolio_changes = [a['portfolio_change'] for a in hold_actions]
            print(f"      Reward médio: {np.mean(hold_rewards):.4f}")
            print(f"      Portfolio change médio: ${np.mean(hold_portfolio_changes):.3f}")
        
        print(f"   🟢 LONG: {len(long_actions)} actions")
        if long_actions:
            long_rewards = [a['reward'] for a in long_actions]
            long_portfolio_changes = [a['portfolio_change'] for a in long_actions]
            print(f"      Reward médio: {np.mean(long_rewards):.4f}")
            print(f"      Portfolio change médio: ${np.mean(long_portfolio_changes):.3f}")
            
        print(f"   🔴 SHORT: {len(short_actions)} actions")
        if short_actions:
            short_rewards = [a['reward'] for a in short_actions]
            short_portfolio_changes = [a['portfolio_change'] for a in short_actions]
            print(f"      Reward médio: {np.mean(short_rewards):.4f}")
            print(f"      Portfolio change médio: ${np.mean(short_portfolio_changes):.3f}")
        
        # DIAGNÓSTICO FINAL
        print(f"\n🔬 DIAGNÓSTICO:")
        print("-" * 50)
        
        if correlation < 0.1:
            print("   ❌ PROBLEMA: Correlação ainda muito baixa")
            print("   💡 POSSÍVEIS CAUSAS:")
            print("      1. Alive bonus ainda dominando")
            print("      2. Componentes não-PnL mascarando signal")
            print("      3. Escala ainda inadequada")
            print("      4. VecNormalize destruindo correlação")
        elif correlation < 0.3:
            print("   ⚠️ MELHORIA: Correlação baixa mas positiva")
            print("   💡 PRÓXIMOS PASSOS: Aumentar ainda mais peso do PnL")
        else:
            print("   ✅ SUCESSO: Correlação boa!")
            
        return correlation > 0.3
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_reward_correlation()
    print(f"\n{'✅ CORRELAÇÃO BOA' if success else '❌ CORRELAÇÃO AINDA BAIXA'}")