#!/usr/bin/env python3
"""
🔍 TEST GAP HYPOTHESIS - Verificar se gaps de preço causam perdas massivas
"""

import pandas as pd
import numpy as np

def analyze_price_gaps():
    """Analisar gaps de preço nos dados de avaliação"""
    
    try:
        # Simular dados similares aos usados na avaliação
        print("🔍 ANALISANDO POSSIBILIDADE DE GAPS DE PREÇO...")
        
        # Criar dados dummy com gaps extremos
        dates = pd.date_range('2025-06-18', periods=100, freq='5min')
        prices = []
        
        base_price = 2000.0
        for i in range(100):
            if i == 50:  # Simular gap no meio
                base_price = 1500.0  # Gap de 500 pontos!
            else:
                base_price += np.random.randn() * 2
            prices.append(base_price)
        
        df = pd.DataFrame({
            'close_5m': prices,
            'open_5m': prices,
            'high_5m': [p + 5 for p in prices],
            'low_5m': [p - 5 for p in prices],
        }, index=dates)
        
        # Analisar gaps
        price_changes = df['close_5m'].diff().abs()
        large_moves = price_changes[price_changes > 50]  # Movimentos > 50 pontos
        
        print(f"📊 ANÁLISE DE GAPS:")
        print(f"   Total de barras: {len(df)}")
        print(f"   Movimentos > 50 pontos: {len(large_moves)}")
        print(f"   Maior movimento: {price_changes.max():.1f} pontos")
        
        if len(large_moves) > 0:
            print(f"🚨 GAPS DETECTADOS:")
            for idx, gap in large_moves.items():
                prev_price = df['close_5m'].loc[:idx].iloc[-2]
                curr_price = df['close_5m'].loc[idx]
                print(f"   {idx}: {prev_price:.1f} → {curr_price:.1f} (gap: {gap:.1f} pontos)")
                
                # Simular posição LONG com SL
                sl_price = prev_price - 8.0  # SL 8 pontos abaixo
                
                if curr_price < sl_price:  # Preço pula SL
                    loss_points = prev_price - curr_price
                    loss_usd = loss_points * 0.05 * 100  # 0.05 lote
                    print(f"   🚨 SL PULADO! SL={sl_price:.1f}, Atual={curr_price:.1f}")
                    print(f"   💸 Perda: {loss_points:.1f} pontos = ${loss_usd:.2f}")
                    
                    if loss_usd > 300:
                        print(f"   ✅ EXPLICAÇÃO ENCONTRADA! Perda > $300 devido a gap")
                        return True
        
        print("❌ Nenhum gap significativo encontrado nos dados dummy")
        return False
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return False

def check_sl_logic():
    """Verificar se lógica SL tem brechas"""
    
    print("\n🔍 VERIFICANDO LÓGICA DE SL...")
    
    # Simular cenários problemáticos
    scenarios = [
        {"name": "SL Normal", "entry": 2000.0, "current": 1992.0, "sl": 1992.0},
        {"name": "Gap Pequeno", "entry": 2000.0, "current": 1990.0, "sl": 1992.0},
        {"name": "Gap Grande", "entry": 2000.0, "current": 1800.0, "sl": 1992.0},
        {"name": "SL Zero", "entry": 2000.0, "current": 1800.0, "sl": 0.0},
        {"name": "SL Ausente", "entry": 2000.0, "current": 1800.0, "sl": None},
    ]
    
    for scenario in scenarios:
        pos = {'type': 'long', 'entry_price': scenario['entry'], 'lot_size': 0.05}
        if scenario['sl'] is not None:
            pos['sl'] = scenario['sl']
        
        current_price = scenario['current']
        should_close = False
        
        # Simular lógica do daytrader.py
        if 'sl' in pos and pos['sl'] > 0:
            if pos['type'] == 'long' and current_price <= pos['sl']:
                should_close = True
        
        # Calcular PnL
        pnl = (current_price - pos['entry_price']) * pos['lot_size'] * 100
        
        print(f"📋 {scenario['name']}:")
        print(f"   Entry: {pos['entry_price']}, Current: {current_price}, SL: {scenario['sl']}")
        print(f"   Should Close: {should_close}, PnL: ${pnl:.2f}")
        
        if not should_close and abs(pnl) > 300:
            print(f"   🚨 BUG! Perda ${abs(pnl):.2f} sem fechamento!")
        print()

if __name__ == "__main__":
    gap_found = analyze_price_gaps()
    check_sl_logic()
    
    if gap_found:
        print("🎯 HIPÓTESE CONFIRMADA: Gaps de preço podem causar perdas massivas!")
    else:
        print("🤔 Hipótese de gap não confirmada, deve haver outro problema...")