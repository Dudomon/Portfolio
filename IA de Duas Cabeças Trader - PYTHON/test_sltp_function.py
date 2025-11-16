#!/usr/bin/env python3
"""
🔍 TEST SLTP FUNCTION - Testar convert_action_to_realistic_sltp para edge cases
"""

import sys
sys.path.append('.')

from daytrader import REALISTIC_SLTP_CONFIG

def convert_action_to_realistic_sltp_test(sltp_action_values, current_price):
    """Cópia da função para teste"""
    sl_adjust = sltp_action_values[0]  # [-3,3] para SL
    tp_adjust = sltp_action_values[1]  # [-3,3] para TP
    
    # Converter para pontos realistas separadamente
    sl_points = REALISTIC_SLTP_CONFIG['sl_min_points'] + \
                (sl_adjust + 3) * (REALISTIC_SLTP_CONFIG['sl_max_points'] - REALISTIC_SLTP_CONFIG['sl_min_points']) / 6
    
    tp_points = REALISTIC_SLTP_CONFIG['tp_min_points'] + \
                (tp_adjust + 3) * (REALISTIC_SLTP_CONFIG['tp_max_points'] - REALISTIC_SLTP_CONFIG['tp_min_points']) / 6
    
    # Arredondar para múltiplos de 0.5 pontos
    sl_points = round(sl_points * 2) / 2
    tp_points = round(tp_points * 2) / 2
    
    # Garantir limites
    sl_points = max(REALISTIC_SLTP_CONFIG['sl_min_points'], min(sl_points, REALISTIC_SLTP_CONFIG['sl_max_points']))
    tp_points = max(REALISTIC_SLTP_CONFIG['tp_min_points'], min(tp_points, REALISTIC_SLTP_CONFIG['tp_max_points']))
    
    return [sl_points, tp_points]

def test_edge_cases():
    """Testar casos extremos"""
    
    print("🔍 TESTANDO convert_action_to_realistic_sltp...")
    print(f"Config: SL {REALISTIC_SLTP_CONFIG['sl_min_points']}-{REALISTIC_SLTP_CONFIG['sl_max_points']}")
    print(f"        TP {REALISTIC_SLTP_CONFIG['tp_min_points']}-{REALISTIC_SLTP_CONFIG['tp_max_points']}")
    
    test_cases = [
        {"name": "Normal", "sl": 0.0, "tp": 0.0},
        {"name": "Min extremo", "sl": -3.0, "tp": -3.0},
        {"name": "Max extremo", "sl": 3.0, "tp": 3.0},
        {"name": "NaN", "sl": float('nan'), "tp": 0.0},
        {"name": "Inf", "sl": float('inf'), "tp": 0.0},
        {"name": "Muito negativo", "sl": -100.0, "tp": 0.0},
        {"name": "Muito positivo", "sl": 100.0, "tp": 0.0},
    ]
    
    current_price = 2000.0
    
    for case in test_cases:
        try:
            result = convert_action_to_realistic_sltp_test([case['sl'], case['tp']], current_price)
            sl_points, tp_points = result
            
            print(f"\n📋 {case['name']}:")
            print(f"   Input: SL={case['sl']}, TP={case['tp']}")
            print(f"   Output: SL={sl_points:.1f}pts, TP={tp_points:.1f}pts")
            
            # Verificar se resultados são válidos
            if sl_points <= 0:
                print(f"   🚨 SL INVÁLIDO: {sl_points}")
            if tp_points <= 0:
                print(f"   🚨 TP INVÁLIDO: {tp_points}")
            if not (0 < sl_points <= 50):
                print(f"   ⚠️ SL fora de range esperado: {sl_points}")
            if not (0 < tp_points <= 100):
                print(f"   ⚠️ TP fora de range esperado: {tp_points}")
                
        except Exception as e:
            print(f"\n❌ {case['name']}: ERRO - {e}")

def test_price_calculation():
    """Testar cálculo de preços SL/TP"""
    
    print("\n🔍 TESTANDO CÁLCULO DE PREÇOS...")
    
    current_price = 2000.0
    sl_points = 5.0
    tp_points = 10.0
    
    # Simular lógica do daytrader.py
    sl_price_diff = sl_points * 1.0
    tp_price_diff = tp_points * 1.0
    
    # LONG position
    long_sl = current_price - sl_price_diff
    long_tp = current_price + tp_price_diff
    
    # SHORT position  
    short_sl = current_price + sl_price_diff
    short_tp = current_price - tp_price_diff
    
    print(f"📊 Current Price: {current_price}")
    print(f"📊 LONG:  SL={long_sl} ({sl_points}pts), TP={long_tp} ({tp_points}pts)")
    print(f"📊 SHORT: SL={short_sl} ({sl_points}pts), TP={short_tp} ({tp_points}pts)")
    
    # Verificar se há problemas
    if long_sl >= current_price:
        print("🚨 LONG SL >= entry price!")
    if long_tp <= current_price:
        print("🚨 LONG TP <= entry price!")
    if short_sl <= current_price:
        print("🚨 SHORT SL <= entry price!")
    if short_tp >= current_price:
        print("🚨 SHORT TP >= entry price!")

if __name__ == "__main__":
    test_edge_cases()
    test_price_calculation()