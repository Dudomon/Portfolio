#!/usr/bin/env python3
"""
🔍 DEBUG REWARD COMPONENTS SYSTEM
Script para debugar os componentes específicos do reward system
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append("D:/Projeto")

from daytrader import TradingEnv

def debug_reward_components():
    """🔍 Debug detalhado dos componentes do reward"""
    
    print("🔍 DEBUG DETALHADO DOS COMPONENTES DO REWARD SYSTEM")
    print("=" * 60)
    
    # Carregar dataset pequeno
    dataset_path = "D:/Projeto/data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
    df = pd.read_csv(dataset_path)
    
    # Processar dataset
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
        df.set_index("timestamp", inplace=True)
        df.drop("time", axis=1, inplace=True)
    
    df = df.rename(columns={
        "open": "open_5m",
        "high": "high_5m", 
        "low": "low_5m",
        "close": "close_5m",
        "tick_volume": "volume_5m"
    })
    
    # Usar apenas 100 barras
    test_df = df.head(100).copy()
    
    # Criar ambiente
    trading_params = {
        "base_lot_size": 0.02,
        "max_lot_size": 0.03,
        "initial_balance": 500.0,
        "target_trades_per_day": 18,
        "stop_loss_range": (2.0, 8.0),
        "take_profit_range": (3.0, 15.0)
    }
    
    env = TradingEnv(
        test_df,
        window_size=20,
        is_training=False,
        initial_balance=500.0,
        trading_params=trading_params
    )
    
    # Acessar o reward system diretamente
    reward_system = env.reward_system
    print(f"✅ Reward System: {type(reward_system).__name__}")
    print(f"🔍 Initial Balance: ${reward_system.initial_balance}")
    print(f"🔍 Current Phase: {reward_system.current_phase}")
    
    # Simular um caso simples
    obs = env.reset()
    
    # Fazer uma ação HOLD primeiro para detectar o problema
    action = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # HOLD
    
    # Store state before action
    old_state = {
        "portfolio_total_value": env.portfolio_value,
        "current_drawdown": env.current_drawdown,
        "trades_count": len(env.trades)
    }
    
    print(f"\n🎮 EXECUTANDO AÇÃO HOLD...")
    print(f"   Portfolio antes: ${env.portfolio_value:.2f}")
    print(f"   Posições antes: {len(env.positions)}")
    print(f"   Trades antes: {len(env.trades)}")
    
    # Execute action
    obs, reward, done, info = env.step(action)
    
    print(f"   Portfolio depois: ${env.portfolio_value:.2f}")
    print(f"   Posições depois: {len(env.positions)}")
    print(f"   Trades depois: {len(env.trades)}")
    print(f"   Reward final: {reward:.6f}")
    
    # Debug o raw reward
    raw_reward, detailed_info, done_flag = reward_system.calculate_reward_and_info(env, action, old_state)
    print(f"   Raw reward: {raw_reward:.6f}")
    print(f"   Scaled reward (/5): {raw_reward / 5.0:.6f}")
    
    return True

if __name__ == "__main__":
    debug_reward_components()
