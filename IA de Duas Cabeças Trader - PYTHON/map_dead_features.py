#!/usr/bin/env python3
"""
MAPEAMENTO DE FEATURES MORTAS
Identifica quais componentes específicos estão sendo zerados
"""

def map_dead_features():
    """Mapeia os índices das features mortas para seus componentes"""
    
    print("=" * 80)
    print("MAPEAMENTO DE FEATURES MORTAS")
    print("=" * 80)
    
    # Baseado no cálculo do daytrader.py
    components = [
        ("Market Features Real", 16),      # 0-15
        ("Position Features", 27),         # 16-42  (3 pos × 9 features)
        ("Intelligent V7", 37),            # 43-79
        ("Microstructure", 14),            # 80-93
        ("Volatility Advanced", 5),        # 94-98
        ("Correlation", 4),                # 99-102
        ("Momentum Multi", 6),             # 103-108
        ("Additional Features", 20)        # 109-128
    ]
    
    dead_indices = [11, 12, 13, 14, 15, 20, 29, 38]  # Novos índices detectados
    
    print("COMPONENTES E SEUS RANGES:")
    current_start = 0
    component_ranges = []
    
    for name, count in components:
        end = current_start + count - 1
        component_ranges.append((name, current_start, end))
        print(f"{name:20s}: indices {current_start:3d}-{end:3d} ({count:2d} features)")
        current_start += count
    
    print(f"\nTOTAL: {current_start} features")
    
    print(f"\nANALISE DAS FEATURES MORTAS:")
    for idx in dead_indices:
        # Encontrar qual componente
        for name, start, end in component_ranges:
            if start <= idx <= end:
                relative_idx = idx - start
                print(f"  Índice {idx:2d}: {name} - posição {relative_idx} (de {end-start+1})")
                break
    
    print(f"\nDETALHAMENTO POR COMPONENTE:")
    
    # Market Features (0-15)
    market_dead = [idx for idx in dead_indices if 0 <= idx <= 15]
    if market_dead:
        print(f"  MARKET FEATURES mortas: {market_dead}")
        market_subcomponents = [
            "close_5m", "high_5m", "low_5m", "volume_5m",      # 0-3
            "rsi_5m", "macd_5m", "bb_upper_5m", "bb_lower_5m", # 4-7  
            "sma_20_5m",                                        # 8 (9th base feature)
            "volume_momentum", "price_position", "breakout_strength",  # 9-11
            "trend_consistency", "support_resistance", "volatility_regime", # 12-14
            "market_structure"                                  # 15 (7th high quality)
        ]
        for idx in market_dead:
            if idx < len(market_subcomponents):
                print(f"    Índice {idx}: {market_subcomponents[idx]}")
    
    # Position Features (16-42) - 3 positions × 9 features each
    position_dead = [idx for idx in dead_indices if 16 <= idx <= 42]
    if position_dead:
        print(f"  POSITION FEATURES mortas: {position_dead}")
        position_subcomponents = [
            "active", "entry_price", "current_price", "pnl", 
            "duration", "risk_ratio", "drawdown", "trailing_stop", "position_size"
        ]
        for idx in position_dead:
            pos_num = (idx - 16) // 9
            feat_num = (idx - 16) % 9
            if feat_num < len(position_subcomponents):
                print(f"    Índice {idx}: Position {pos_num} - {position_subcomponents[feat_num]}")
    
    # Microstructure (80-93)
    micro_dead = [idx for idx in dead_indices if 80 <= idx <= 93]
    if micro_dead:
        print(f"  MICROSTRUCTURE mortas: {micro_dead}")
    
    print(f"\nPADRÕES IDENTIFICADOS:")
    
    # Verificar padrões
    gaps = [dead_indices[i+1] - dead_indices[i] for i in range(len(dead_indices)-1)]
    print(f"  Gaps entre índices: {gaps}")
    
    # Verificar se são múltiplos
    for divisor in [2, 3, 4, 5, 6, 7, 8, 9]:
        multiples = [idx for idx in dead_indices if idx % divisor == 0]
        if len(multiples) > 1:
            print(f"  Múltiplos de {divisor}: {multiples}")
    
    print(f"\nHIPÓTESES:")
    print("1. Features específicas sendo zeradas no preprocessing")
    print("2. Posições inativas sempre zeradas")
    print("3. Componentes opcionais não implementados")
    print("4. Bug no pipeline de features")

if __name__ == "__main__":
    map_dead_features()