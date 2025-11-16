#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO ESPECÍFICO DE SL/TP
Analisa por que as posições não estão sendo fechadas
"""

import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict
import json

# Adicionar paths
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

from daytrader import TradingEnv, TRADING_CONFIG, TRIAL_2_TRADING_PARAMS

def diagnose_sltp_issue():
    """Diagnosticar problema específico de SL/TP"""
    print("🔍 DIAGNÓSTICO ESPECÍFICO DE SL/TP")
    print("="*50)
    
    # Criar dados mock
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    # Criar dados com movimento significativo para testar SL/TP
    base_price = 4000
    price_changes = np.concatenate([
        np.random.normal(0, 0.02, 20),    # Movimento normal
        np.linspace(0, 0.1, 20),          # Movimento para cima (TP)
        np.linspace(0.1, -0.1, 20),       # Movimento para baixo (SL)
        np.random.normal(0, 0.02, 40)     # Movimento normal
    ])
    
    prices = [base_price]
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))
    prices = prices[1:]
    
    df = pd.DataFrame({
        'close_5m': prices,
        'high_5m': [p * 1.001 for p in prices],
        'low_5m': [p * 0.999 for p in prices],
        'volume_5m': [5000] * len(prices),
    }, index=dates)
    
    # Criar ambiente
    env = TradingEnv(df=df, window_size=20, is_training=True)
    obs = env.reset()
    
    print(f"📊 Dados criados: {len(df)} barras")
    print(f"💰 Preço inicial: ${prices[0]:.2f}")
    print(f"💰 Preço final: ${prices[-1]:.2f}")
    print(f"📈 Variação total: {((prices[-1] - prices[0]) / prices[0]) * 100:.2f}%")
    
    # Forçar criação de posições com análise detalhada
    print("\n🎯 TESTE 1: CRIAR POSIÇÕES E ANALISAR SL/TP")
    print("-" * 50)
    
    # Ação que força LONG
    long_action = np.array([1.0, 0.8, 0.5, 0.5, 0.0, 2.0, 0.0, 0.0, 1.5, 0.0, 0.0])
    obs, reward, done, info = env.step(long_action)
    
    print(f"Posições após LONG: {len(env.positions)}")
    if env.positions:
        pos = env.positions[0]
        current_price = df['close_5m'].iloc[env.current_step]
        print(f"  Tipo: {pos['type']}")
        print(f"  Entry Price: ${pos['entry_price']:.2f}")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  SL: ${pos.get('sl', 'N/A')}")
        print(f"  TP: ${pos.get('tp', 'N/A')}")
        
        if 'sl' in pos:
            sl_distance = abs(pos['entry_price'] - pos['sl'])
            print(f"  SL Distance: ${sl_distance:.2f}")
            print(f"  SL Trigger: Current <= ${pos['sl']:.2f}")
        
        if 'tp' in pos:
            tp_distance = abs(pos['tp'] - pos['entry_price'])
            print(f"  TP Distance: ${tp_distance:.2f}")
            print(f"  TP Trigger: Current >= ${pos['tp']:.2f}")
    
    # Testar movimento de preços
    print(f"\n🔄 TESTE 2: SIMULAR MOVIMENTO DE PREÇOS")
    print("-" * 50)
    
    position_history = []
    
    for i in range(min(50, len(df) - env.current_step - 1)):
        current_step = env.current_step
        current_price = df['close_5m'].iloc[current_step]
        positions_before = len(env.positions)
        
        # Log posições se existirem
        for j, pos in enumerate(env.positions):
            sl_hit = False
            tp_hit = False
            
            if pos['type'] == 'long':
                if 'sl' in pos and current_price <= pos['sl']:
                    sl_hit = True
                if 'tp' in pos and current_price >= pos['tp']:
                    tp_hit = True
            else:  # short
                if 'sl' in pos and current_price >= pos['sl']:
                    sl_hit = True
                if 'tp' in pos and current_price <= pos['tp']:
                    tp_hit = True
            
            position_history.append({
                'step': current_step,
                'position_id': j,
                'type': pos['type'],
                'entry_price': pos['entry_price'],
                'current_price': current_price,
                'sl': pos.get('sl', None),
                'tp': pos.get('tp', None), 
                'sl_hit': sl_hit,
                'tp_hit': tp_hit,
                'should_close': sl_hit or tp_hit,
                'pnl': env._get_position_pnl(pos, current_price)
            })
        
        # Step neutro para avançar tempo
        neutral_action = np.array([0.3, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs, reward, done, info = env.step(neutral_action)
        
        positions_after = len(env.positions)
        
        if positions_before != positions_after:
            print(f"  Step {current_step}: Posições {positions_before} -> {positions_after}")
            print(f"    Preço: ${current_price:.2f}")
        
        if done:
            break
    
    # Análise dos resultados
    print(f"\n📊 ANÁLISE DOS RESULTADOS")
    print("-" * 50)
    
    if position_history:
        df_pos = pd.DataFrame(position_history)
        
        total_positions = len(df_pos)
        should_close_count = df_pos['should_close'].sum()
        sl_hits = df_pos['sl_hit'].sum()
        tp_hits = df_pos['tp_hit'].sum()
        
        print(f"Total de registros de posição: {total_positions}")
        print(f"Deveriam ter fechado: {should_close_count}")
        print(f"SL hits: {sl_hits}")
        print(f"TP hits: {tp_hits}")
        
        if should_close_count > 0:
            print(f"\n⚠️ PROBLEMA DETECTADO: {should_close_count} posições deveriam ter fechado!")
            
            # Mostrar exemplos
            should_close = df_pos[df_pos['should_close'] == True]
            for _, row in should_close.head(5).iterrows():
                print(f"  Step {row['step']}: {row['type']} - Price: ${row['current_price']:.2f}")
                print(f"    Entry: ${row['entry_price']:.2f}, SL: ${row['sl']}, TP: ${row['tp']}")
                print(f"    SL Hit: {row['sl_hit']}, TP Hit: {row['tp_hit']}")
                print(f"    PnL: ${row['pnl']:.2f}")
        else:
            print(f"✅ SL/TP funcionando corretamente")
    
    print(f"\n🎯 DIAGNÓSTICO FINAL")
    print("-" * 50)
    
    final_positions = len(env.positions)
    print(f"Posições finais: {final_positions}")
    
    if final_positions > 0:
        print(f"⚠️ PROBLEMA: {final_positions} posições ainda abertas")
        for i, pos in enumerate(env.positions):
            current_price = df['close_5m'].iloc[env.current_step]
            duration = env.current_step - pos['entry_step']
            pnl = env._get_position_pnl(pos, current_price)
            
            print(f"  Posição {i}: {pos['type']}")
            print(f"    Duration: {duration} steps")
            print(f"    Entry: ${pos['entry_price']:.2f}")
            print(f"    Current: ${current_price:.2f}")
            print(f"    SL: ${pos.get('sl', 'N/A')}")
            print(f"    TP: ${pos.get('tp', 'N/A')}")
            print(f"    PnL: ${pnl:.2f}")
            
            # Verificar se deveria ter fechado
            if pos['type'] == 'long':
                if 'sl' in pos and current_price <= pos['sl']:
                    print(f"    ❌ DEVERIA TER FECHADO POR SL!")
                if 'tp' in pos and current_price >= pos['tp']:
                    print(f"    ❌ DEVERIA TER FECHADO POR TP!")
            else:  # short
                if 'sl' in pos and current_price >= pos['sl']:
                    print(f"    ❌ DEVERIA TER FECHADO POR SL!")
                if 'tp' in pos and current_price <= pos['tp']:
                    print(f"    ❌ DEVERIA TER FECHADO POR TP!")
    else:
        print(f"✅ Todas as posições foram fechadas corretamente")

if __name__ == "__main__":
    diagnose_sltp_issue()