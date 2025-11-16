#!/usr/bin/env python3
"""
🔍 DEBUG REWARD & PnL ANALYSIS
Script para investigar o sistema de reward e cálculos de PnL
que podem estar causando a performance ruim do modelo
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append("D:/Projeto")

from daytrader import TradingEnv

def analyze_reward_pnl_system():
    """🔍 Análise detalhada do sistema reward/PnL"""
    
    print("🔍 INICIANDO ANÁLISE DETALHADA REWARD/PnL SYSTEM")
    print("=" * 60)
    
    # Carregar dataset pequeno para teste
    dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return False
    
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
    
    # Pegar apenas 500 barras para análise rápida
    test_df = df.head(500).copy()
    print(f"✅ Dataset teste: {len(test_df)} barras")
    print(f"📅 Período: {test_df.index.min()} até {test_df.index.max()}")
    
    # Criar ambiente
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
        is_training=False,
        initial_balance=500.0,
        trading_params=trading_params
    )
    
    print(f"✅ Ambiente criado")
    print(f"🔍 Action Space: {env.action_space}")
    print(f"💰 Portfolio Inicial: ${env.portfolio_value:.2f}")
    
    # Simular algumas ações manuais para debug
    obs = env.reset()
    step_count = 0
    max_steps = 100
    
    # Arrays para debug
    rewards_log = []
    pnl_log = []
    trades_log = []
    actions_log = []
    
    print("\n🎮 INICIANDO SIMULAÇÃO DE DEBUG...")
    print("=" * 50)
    
    while step_count < max_steps:
        # Criar ação de teste - vamos forçar algumas entradas
        if step_count < 10:
            # Primeiros steps: HOLD
            action = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            action_type = "HOLD"
        elif step_count < 20:
            # Steps 10-19: LONG com confidence alta
            action = np.array([1.5, 0.8, -1.0, 2.0, 0.5, 1.0, -0.5, 1.5])
            action_type = "LONG"
        elif step_count < 30:
            # Steps 20-29: HOLD novamente
            action = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            action_type = "HOLD"
        elif step_count < 40:
            # Steps 30-39: SHORT com confidence média
            action = np.array([2.5, 0.6, 1.0, -1.0, -0.5, -1.0, 0.5, -1.5])
            action_type = "SHORT"
        else:
            # Resto: HOLD
            action = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            action_type = "HOLD"
        
        # Store pre-step data
        pre_trades_count = len(env.trades)
        pre_portfolio = env.portfolio_value
        pre_positions = len(env.positions)
        
        # Execute step
        obs, reward, done, info = env.step(action)
        
        # Store post-step data
        post_trades_count = len(env.trades)
        post_portfolio = env.portfolio_value
        post_positions = len(env.positions)
        
        # Log detalhado
        actions_log.append({
            'step': step_count,
            'action_type': action_type,
            'action': action.copy(),
            'pre_portfolio': pre_portfolio,
            'post_portfolio': post_portfolio,
            'portfolio_change': post_portfolio - pre_portfolio,
            'pre_positions': pre_positions,
            'post_positions': post_positions,
            'reward': reward,
            'new_trades': post_trades_count - pre_trades_count
        })
        
        rewards_log.append(reward)
        
        # Se houve novos trades, analisar
        if post_trades_count > pre_trades_count:
            new_trades = env.trades[-(post_trades_count - pre_trades_count):]
            for trade in new_trades:
                trade_pnl = trade.get('pnl_usd', 0)
                pnl_log.append(trade_pnl)
                trades_log.append({
                    'step': step_count,
                    'type': trade.get('type', 'unknown'),
                    'entry_price': trade.get('entry_price', 0),
                    'exit_price': trade.get('exit_price', 0),
                    'pnl': trade_pnl,
                    'lot_size': trade.get('volume', 0),
                    'duration': trade.get('duration', 0)
                })
                
                trade_type = trade.get('type', 'unknown')
                print(f"  🔹 STEP {step_count}: {action_type} | Trade: {trade_type} | "
                      f"PnL: ${trade_pnl:.2f} | Reward: {reward:.4f} | "
                      f"Portfolio: ${pre_portfolio:.2f} → ${post_portfolio:.2f}")
        
        # Log periódico do estado
        if step_count % 10 == 0:
            print(f"  📊 STEP {step_count}: {action_type} | Portfolio: ${env.portfolio_value:.2f} | "
                  f"Positions: {len(env.positions)} | Trades: {len(env.trades)} | "
                  f"Reward: {reward:.4f}")
        
        step_count += 1
        if done:
            break
    
    # ANÁLISE FINAL
    print("\n" + "=" * 60)
    print("📊 ANÁLISE FINAL DO DEBUG")
    print("=" * 60)
    
    # Estatísticas de rewards
    total_reward = sum(rewards_log)
    avg_reward = np.mean(rewards_log) if rewards_log else 0
    reward_std = np.std(rewards_log) if len(rewards_log) > 1 else 0
    
    print(f"💰 PORTFOLIO:")
    print(f"   Inicial: $500.00")
    print(f"   Final: ${env.portfolio_value:.2f}")
    print(f"   Mudança: ${env.portfolio_value - 500:.2f} ({((env.portfolio_value - 500) / 500 * 100):+.2f}%)")
    
    print(f"\n🎯 REWARDS:")
    print(f"   Total: {total_reward:.4f}")
    print(f"   Média: {avg_reward:.4f}")
    print(f"   Desvio: {reward_std:.4f}")
    print(f"   Min: {min(rewards_log):.4f}" if rewards_log else "   Min: N/A")
    print(f"   Max: {max(rewards_log):.4f}" if rewards_log else "   Max: N/A")
    
    print(f"\n📈 TRADES:")
    print(f"   Total: {len(trades_log)}")
    if trades_log:
        profitable = [t for t in trades_log if t['pnl'] > 0]
        losing = [t for t in trades_log if t['pnl'] < 0]
        total_pnl = sum(t['pnl'] for t in trades_log)
        
        print(f"   Lucrativos: {len(profitable)}")
        print(f"   Perdedores: {len(losing)}")
        print(f"   Win Rate: {(len(profitable) / len(trades_log) * 100):.1f}%")
        print(f"   PnL Total: ${total_pnl:.2f}")
        print(f"   PnL Médio: ${(total_pnl / len(trades_log)):.2f}")
        
        if profitable:
            avg_win = np.mean([t['pnl'] for t in profitable])
            print(f"   Ganho Médio: ${avg_win:.2f}")
        
        if losing:
            avg_loss = np.mean([t['pnl'] for t in losing])
            print(f"   Perda Média: ${avg_loss:.2f}")
    
    # Análise de componentes do reward system
    print(f"\n🔬 ANÁLISE DE REWARD COMPONENTS:")
    if hasattr(env, 'reward_system'):
        reward_system = env.reward_system
        if hasattr(reward_system, 'reward_history') and reward_system.reward_history:
            recent_rewards = reward_system.reward_history[-20:]  # Últimos 20
            print(f"   Rewards Recentes: {recent_rewards}")
            
    # Detectar problemas
    print(f"\n⚠️ DETECÇÃO DE PROBLEMAS:")
    problems_found = []
    
    if avg_reward < -0.01:
        problems_found.append("Reward médio muito negativo")
    
    if len(trades_log) == 0:
        problems_found.append("Nenhum trade executado")
    elif len(trades_log) > 0:
        win_rate = len([t for t in trades_log if t['pnl'] > 0]) / len(trades_log)
        if win_rate < 0.3:
            problems_found.append(f"Win rate muito baixo ({win_rate:.1%})")
            
        total_pnl = sum(t['pnl'] for t in trades_log)
        if total_pnl < -10:
            problems_found.append(f"PnL total muito negativo (${total_pnl:.2f})")
    
    if env.portfolio_value < 450:  # Perda > 10%
        problems_found.append(f"Portfolio perdeu muito valor ({((env.portfolio_value - 500)/500*100):+.1f}%)")
    
    if problems_found:
        print(f"   ❌ Problemas encontrados:")
        for i, problem in enumerate(problems_found, 1):
            print(f"     {i}. {problem}")
    else:
        print(f"   ✅ Nenhum problema crítico detectado")
    
    # Salvar log detalhado
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"D:/Projeto/debug_reward_pnl_log_{timestamp}.txt"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("DEBUG REWARD & PnL ANALYSIS\\n")
        f.write("=" * 60 + "\\n\\n")
        f.write(f"Total Steps: {step_count}\\n")
        f.write(f"Portfolio: $500.00 → ${env.portfolio_value:.2f} ({((env.portfolio_value - 500) / 500 * 100):+.2f}%)\\n")
        f.write(f"Trades: {len(trades_log)}\\n")
        f.write(f"Avg Reward: {avg_reward:.4f}\\n\\n")
        
        f.write("DETAILED ACTIONS LOG:\\n")
        for action_info in actions_log:
            f.write(f"Step {action_info['step']}: {action_info['action_type']} | "
                   f"Reward: {action_info['reward']:.4f} | "
                   f"Portfolio: ${action_info['pre_portfolio']:.2f} → ${action_info['post_portfolio']:.2f}\\n")
        
        f.write("\\nTRADES LOG:\\n")
        for trade in trades_log:
            f.write(f"Step {trade['step']}: {trade['type']} | "
                   f"Entry: {trade['entry_price']:.2f} | Exit: {trade['exit_price']:.2f} | "
                   f"PnL: ${trade['pnl']:.2f} | Duration: {trade['duration']}\\n")
    
    print(f"\\n💾 Log detalhado salvo em: {log_file}")
    
    return True

if __name__ == "__main__":
    print("🔍 INICIANDO DEBUG DO REWARD & PnL SYSTEM")
    success = analyze_reward_pnl_system()
    if success:
        print("\\n✅ ANÁLISE CONCLUÍDA")
    else:
        print("\\n❌ ANÁLISE FALHOU")