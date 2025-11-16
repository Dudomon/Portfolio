#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO TIMING DE FECHAMENTO
Analisa quanto tempo posições ficam abertas
"""

import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict

# Adicionar paths
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

from daytrader import TradingEnv, TRADING_CONFIG, TRIAL_2_TRADING_PARAMS

def diagnose_closing_timing():
    """Diagnosticar timing de fechamento de posições"""
    print("🔍 DIAGNÓSTICO TIMING DE FECHAMENTO")
    print("="*50)
    
    # Criar dados com movimento claro para SL/TP
    dates = pd.date_range('2023-01-01', periods=150, freq='5min')
    base_price = 4000
    
    # Movimento que deveria ativar SL/TP rapidamente
    price_pattern = []
    for i in range(150):
        if i < 20:
            price_pattern.append(0)  # Estável 
        elif i < 40:
            price_pattern.append(0.002 * (i-20))  # Subida suave
        elif i < 60:  
            price_pattern.append(0.04 - 0.004 * (i-40))  # Queda abrupta (SL)
        elif i < 80:
            price_pattern.append(-0.04 + 0.005 * (i-60))  # Recuperação (TP)
        else:
            price_pattern.append(np.random.normal(0, 0.005))  # Ruído
    
    prices = [base_price]
    for change in price_pattern:
        prices.append(prices[-1] * (1 + change))
    prices = prices[1:]
    
    df = pd.DataFrame({
        'close_5m': prices,
        'high_5m': [p * 1.001 for p in prices],
        'low_5m': [p * 0.999 for p in prices],
        'volume_5m': [8000] * len(prices),
    }, index=dates)
    
    # Criar ambiente
    env = TradingEnv(df=df, window_size=20, is_training=True)
    obs = env.reset()
    
    print(f"📊 Teste com {len(df)} barras")
    print(f"💰 Variação: {((prices[-1] - prices[0]) / prices[0]) * 100:.2f}%")
    
    # Forçar criação de 3 posições
    print(f"\n🎯 FASE 1: CRIAR 3 POSIÇÕES")
    print("-" * 50)
    
    positions_log = []
    
    # Criar 3 posições LONG
    for i in range(3):
        long_action = np.array([1.0, 0.9, 0.7, 0.7, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        obs, reward, done, info = env.step(long_action)
        
        if len(env.positions) > i:
            pos = env.positions[-1]  # Última posição criada
            current_price = df['close_5m'].iloc[env.current_step]
            
            print(f"  Posição {i+1}: LONG @ ${pos['entry_price']:.2f}")
            print(f"    SL: ${pos.get('sl', 'N/A'):.2f}")
            print(f"    TP: ${pos.get('tp', 'N/A'):.2f}")
            print(f"    Current: ${current_price:.2f}")
            
            positions_log.append({
                'id': i,
                'entry_step': env.current_step,
                'entry_price': pos['entry_price'],
                'sl': pos.get('sl', 0),
                'tp': pos.get('tp', 0),
                'type': pos.get('type', 'long')
            })
    
    print(f"\n🔄 FASE 2: MONITORAR FECHAMENTOS POR {min(100, len(df)-env.current_step-5)} STEPS")
    print("-" * 50)
    
    close_events = []
    step_log = []
    
    for step in range(min(100, len(df) - env.current_step - 5)):
        positions_before = len(env.positions)
        current_price = df['close_5m'].iloc[env.current_step]
        step_info = env.current_step
        
        # Registrar estado atual das posições
        for i, pos in enumerate(env.positions):
            duration = step_info - pos['entry_step']
            pnl = env._get_position_pnl(pos, current_price)
            
            # Verificar condições SL/TP
            sl_should_hit = False
            tp_should_hit = False
            
            if pos['type'] == 'long':
                if 'sl' in pos and current_price <= pos['sl']:
                    sl_should_hit = True
                if 'tp' in pos and current_price >= pos['tp']:
                    tp_should_hit = True
            
            step_log.append({
                'step': step_info,
                'position_id': i,
                'duration': duration,
                'current_price': current_price,
                'entry_price': pos['entry_price'],
                'sl': pos.get('sl', 0),
                'tp': pos.get('tp', 0),
                'pnl': pnl,
                'sl_should_hit': sl_should_hit,
                'tp_should_hit': tp_should_hit,
                'should_close': sl_should_hit or tp_should_hit
            })
        
        # Step neutro para avançar
        neutral_action = np.array([0.3, 0.5, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs, reward, done, info = env.step(neutral_action)
        
        positions_after = len(env.positions)
        
        # Detectar fechamentos
        if positions_before > positions_after:
            positions_closed = positions_before - positions_after
            print(f"  Step {step_info}: {positions_closed} posição(s) FECHADA(s)")
            print(f"    Preço: ${current_price:.2f}")
            print(f"    Posições: {positions_before} -> {positions_after}")
            
            close_events.append({
                'step': step_info,
                'positions_closed': positions_closed,
                'price': current_price
            })
        
        if done:
            break
    
    # Análise dos resultados
    print(f"\n📊 ANÁLISE DOS RESULTADOS")
    print("-" * 50)
    
    print(f"Total de eventos de fechamento: {len(close_events)}")
    print(f"Posições finais: {len(env.positions)}")
    
    if step_log:
        df_steps = pd.DataFrame(step_log)
        
        # Posições que deveriam ter fechado
        should_close = df_steps[df_steps['should_close'] == True]
        total_should_close = len(should_close)
        actual_closes = len(close_events)
        
        print(f"Deveriam ter fechado: {total_should_close}")
        print(f"Realmente fecharam: {actual_closes}")
        
        if total_should_close > 0:
            efficiency = (actual_closes / total_should_close) * 100
            print(f"Eficiência de fechamento: {efficiency:.1f}%")
            
            if efficiency < 90:
                print(f"\n⚠️ PROBLEMA DE TIMING DETECTADO!")
                print(f"Exemplos de falhas:")
                for _, row in should_close.head(3).iterrows():
                    print(f"  Step {row['step']}: Preço ${row['current_price']:.2f}")
                    print(f"    SL: ${row['sl']:.2f}, TP: ${row['tp']:.2f}")
                    print(f"    SL hit: {row['sl_should_hit']}, TP hit: {row['tp_should_hit']}")
        
        # Duração média das posições
        if len(env.positions) > 0:
            current_durations = []
            for pos in env.positions:
                duration = env.current_step - pos['entry_step']
                current_durations.append(duration)
            
            avg_duration = np.mean(current_durations)
            max_duration = max(current_durations)
            
            print(f"\nDuração das posições abertas:")
            print(f"  Média: {avg_duration:.1f} steps ({avg_duration*5:.0f} min)")
            print(f"  Máxima: {max_duration} steps ({max_duration*5:.0f} min)")
            print(f"  Limite do sistema: 576 steps (2880 min = 48h)")
            
            if max_duration > 100:
                print(f"  ⚠️ Posições muito longas! Limite deveria ser menor.")

if __name__ == "__main__":
    diagnose_closing_timing()