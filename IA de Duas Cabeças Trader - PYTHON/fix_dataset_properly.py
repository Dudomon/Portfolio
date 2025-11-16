#!/usr/bin/env python3
"""
CORREÇÃO SUAVE DO DATASET - SEM DRIFT CUMULATIVO EXPLOSIVO
"""

import pandas as pd
import numpy as np
from datetime import datetime

def fix_dataset_properly():
    """Aplicar correção SUAVE, não cumulativa"""
    
    print("🔧 CORRIGINDO DATASET PROPRIAMENTE")
    print("="*50)
    
    # Carregar dataset original
    original_path = 'data/GOLD_SAFE_CHALLENGING_2M_20250801_203251.csv'
    print(f"Carregando dataset original: {original_path}")
    
    df = pd.read_csv(original_path)
    print(f"Dataset carregado: {len(df):,} barras")
    
    # Aplicar correção SUAVE (não cumulativa)
    print("\nAplicando drifts diferenciados por regime...")
    
    regime_adjustments = {
        'bull': 0.0002,    # +0.02% drift SIMPLES
        'bear': -0.0002,   # -0.02% drift SIMPLES
        'sideways': 0.0    # zero drift
    }
    
    for regime, drift in regime_adjustments.items():
        mask = df['regime'] == regime
        if mask.any():
            count = mask.sum()
            print(f"  {regime}: {count:,} barras, drift {drift:+.4f}")
            
            # CORREÇÃO SUAVE: aplicar drift UMA VEZ só, não cumulativo
            df.loc[mask, 'close'] *= (1 + drift)
            df.loc[mask, 'open'] *= (1 + drift)  
            df.loc[mask, 'high'] *= (1 + drift)
            df.loc[mask, 'low'] *= (1 + drift)
    
    print("✅ Drifts aplicados")
    
    # Validar resultado
    print("\nValidando correção...")
    df['returns'] = df['close'].pct_change()
    
    regime_stats = df.groupby('regime')['returns'].agg(['mean', 'std'])
    print("\nPerformance por regime (corrigida):")
    for regime in regime_stats.index:
        mean_ret = regime_stats.loc[regime, 'mean']
        std_ret = regime_stats.loc[regime, 'std']
        print(f"  {regime}: mean={mean_ret:.6f}, std={std_ret:.6f}")
    
    # Verificar se volatilidade está normal
    vol_total = df['returns'].std()
    print(f"\nVolatilidade total: {vol_total:.6f} ({vol_total*100:.2f}%)")
    
    if vol_total > 0.1:  # > 10%
        print("❌ FALHA: Volatilidade ainda muito alta!")
        return None
    else:
        print("✅ Volatilidade normalizada")
    
    # Salvar dataset corrigido
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corrected_filename = f"data/GOLD_FIXED_PROPERLY_2M_{timestamp}.csv"
    
    df.to_csv(corrected_filename, index=False)
    print(f"\n✅ Dataset corrigido salvo: {corrected_filename}")
    
    return corrected_filename

if __name__ == '__main__':
    fix_dataset_properly()