#!/usr/bin/env python3
"""
🔍 QUICK POSITION TEST - Testar criação de posições com logs
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from daytrader import TradingEnv

def test_position_creation():
    """Teste rápido da criação de posições"""
    
    print("🔍 TESTANDO CRIAÇÃO DE POSIÇÕES...")
    
    # Criar dados dummy
    dates = pd.date_range('2023-01-01', periods=200, freq='5min')
    df = pd.DataFrame({
        'close_5m': [2000.0 + np.sin(i/10)*20 + np.random.randn()*1 for i in range(200)],
        'open_5m': [2000.0 + np.sin(i/10)*20 + np.random.randn()*1 for i in range(200)], 
        'high_5m': [2000.0 + np.sin(i/10)*20 + np.random.randn()*1 + 3 for i in range(200)],
        'low_5m': [2000.0 + np.sin(i/10)*20 + np.random.randn()*1 - 3 for i in range(200)],
    }, index=dates)
    
    # Criar environment
    try:
        env = TradingEnv(df, window_size=20, is_training=False, initial_balance=500)
        obs = env.reset()
        
        print(f"✅ Environment criado")
        
        # Simular algumas ações que criam posições
        actions = [
            np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # LONG
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # HOLD  
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # HOLD
            np.array([2.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # SHORT
        ]
        
        for i, action in enumerate(actions):
            print(f"\n--- Step {i+1}: Action {action} ---")
            obs, reward, done, info = env.step(action)
            
            print(f"Positions: {len(env.positions)}")
            print(f"Portfolio: ${env.portfolio_value:.2f}")
            
            if done:
                break
                
        print(f"\n✅ Teste concluído")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_position_creation()