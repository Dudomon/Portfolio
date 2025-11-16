#!/usr/bin/env python3
"""
🔍 DEBUG SL BUG - Investigar perdas >300 pontos violando SL máximo
"""

import pandas as pd
import numpy as np
from daytrader import TradingEnv, REALISTIC_SLTP_CONFIG

def test_sl_enforcement():
    """Testar se SL está sendo respeitado corretamente"""
    
    print("🔍 TESTANDO APLICAÇÃO DE SL...")
    
    # Criar dados dummy para teste
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    df = pd.DataFrame({
        'close_5m': [2000.0 + np.sin(i/10)*50 + np.random.randn()*2 for i in range(1000)],
        'open_5m': [2000.0 + np.sin(i/10)*50 + np.random.randn()*2 for i in range(1000)],
        'high_5m': [2000.0 + np.sin(i/10)*50 + np.random.randn()*2 + 5 for i in range(1000)],
        'low_5m': [2000.0 + np.sin(i/10)*50 + np.random.randn()*2 - 5 for i in range(1000)],
    }, index=dates)
    
    # Criar environment
    env = TradingEnv(df, window_size=20, is_training=False, initial_balance=500)
    
    # Reset environment
    obs = env.reset()
    
    print(f"📊 Initial Portfolio: ${env.portfolio_value}")
    print(f"📊 SL Config: {REALISTIC_SLTP_CONFIG['sl_max_points']} pontos máximo")
    
    # Simular uma entrada LONG
    entry_action = np.array([1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Long com confiança alta
    obs, reward, done, info = env.step(entry_action)
    
    if env.positions:
        pos = env.positions[0]
        print(f"\n🔍 POSIÇÃO CRIADA:")
        print(f"  Tipo: {pos['type']}")
        print(f"  Entry Price: {pos['entry_price']:.2f}")
        print(f"  SL Price: {pos.get('sl', 'SEM SL!'):.2f}")
        print(f"  TP Price: {pos.get('tp', 'SEM TP!'):.2f}")
        
        if 'sl' in pos and pos['sl'] > 0:
            sl_points = abs(pos['entry_price'] - pos['sl'])
            print(f"  SL Distance: {sl_points:.1f} pontos")
            
            if sl_points > 15:  # Máximo deveria ser 8
                print(f"🚨 BUG ENCONTRADO! SL > 8 pontos: {sl_points:.1f}")
                return False
        else:
            print("🚨 BUG CRÍTICO! Posição sem SL!")
            return False
    else:
        print("❌ Nenhuma posição foi criada")
        return False
    
    # Simular movimento de preço que deveria ativar SL
    print(f"\n🎯 SIMULANDO MOVIMENTO DE PREÇO...")
    steps_without_sl = 0
    
    for step in range(50):
        # Simular HOLD
        hold_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs, reward, done, info = env.step(hold_action)
        
        current_price = env.df[f'close_{env.base_tf}'].iloc[env.current_step]
        
        if env.positions:
            pos = env.positions[0]
            pnl = env._get_position_pnl(pos, current_price)
            
            # Se perda > 8 pontos * 100 (fator) * 0.01 (lot) = $8
            # Mas para 0.05 lot = $40 máximo
            max_loss_expected = 8 * pos['lot_size'] * 100
            
            print(f"  Step {step}: Price={current_price:.2f}, PnL=${pnl:.2f}, Max=${max_loss_expected:.2f}")
            
            if pnl < -max_loss_expected * 2:  # Se perda > 2x esperado
                print(f"🚨 PERDA EXCESSIVA! PnL=${pnl:.2f} > MaxExpected=${max_loss_expected:.2f}")
                print(f"   SL Price: {pos.get('sl', 'N/A')}, Current: {current_price}")
                return False
                
            steps_without_sl += 1
            
            if steps_without_sl > 30:
                print(f"🚨 POSIÇÃO ABERTA POR MUITO TEMPO SEM SL HIT!")
                return False
        else:
            print(f"  ✅ Posição fechada no step {step}")
            break
    
    print(f"\n✅ TESTE PASSOU - SL sendo respeitado corretamente")
    return True

if __name__ == "__main__":
    success = test_sl_enforcement()
    if not success:
        print("\n🔴 TESTE FALHOU - BUG NO SISTEMA DE SL!")
        exit(1)
    else:
        print("\n🟢 TESTE PASSOU - SL funcionando corretamente")