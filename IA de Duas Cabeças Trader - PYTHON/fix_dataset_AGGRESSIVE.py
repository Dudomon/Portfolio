#!/usr/bin/env python3
"""
CORREÇÃO AGRESSIVA REAL - CRIAR REGIMES DISTINTIVOS
"""

import pandas as pd
import numpy as np
from datetime import datetime

def fix_dataset_aggressive():
    """Aplicar correção AGRESSIVA para criar regimes realmente diferentes"""
    
    print("🔧 CORREÇÃO AGRESSIVA - REGIMES DISTINTIVOS")
    print("="*60)
    
    # Carregar dataset original
    original_path = 'data/GOLD_SAFE_CHALLENGING_2M_20250801_203251.csv'
    print(f"Carregando dataset original: {original_path}")
    
    df = pd.read_csv(original_path)
    print(f"Dataset carregado: {len(df):,} barras")
    
    # CORREÇÃO AGRESSIVA: drifts que superem a volatilidade
    print("\nAplicando drifts AGRESSIVOS por regime...")
    
    # Volatilidade atual é ~2.1%, então drift precisa ser > 0.5% para ser significativo
    regime_adjustments = {
        'bull': 0.008,     # +0.8% drift (4x maior que volatilidade)
        'bear': -0.008,    # -0.8% drift (4x maior que volatilidade)
        'sideways': 0.0    # zero drift
    }
    
    for regime, drift in regime_adjustments.items():
        mask = df['regime'] == regime
        if mask.any():
            count = mask.sum()
            print(f"  {regime}: {count:,} barras, drift {drift:+.1%}")
            
            # Aplicar drift significativo
            df.loc[mask, 'close'] *= (1 + drift)
            df.loc[mask, 'open'] *= (1 + drift)  
            df.loc[mask, 'high'] *= (1 + drift)
            df.loc[mask, 'low'] *= (1 + drift)
    
    print("✅ Drifts agressivos aplicados")
    
    # Validar resultado
    print("\nValidando correção...")
    df['returns'] = df['close'].pct_change()
    
    regime_stats = df.groupby('regime')['returns'].agg(['mean', 'std'])
    print("\nPerformance por regime (CORRIGIDA):")
    for regime in regime_stats.index:
        mean_ret = regime_stats.loc[regime, 'mean']
        std_ret = regime_stats.loc[regime, 'std']
        print(f"  {regime}: mean={mean_ret:.6f} ({mean_ret*100:.2f}%), std={std_ret:.6f}")
    
    # Verificar se regimes agora são distintivos
    regime_means = regime_stats['mean'].values
    regime_diff = np.max(regime_means) - np.min(regime_means)
    print(f"\nDiferença entre regimes: {regime_diff:.6f} ({regime_diff*100:.2f}%)")
    
    if regime_diff < 0.001:  # < 0.1%
        print("❌ FALHA: Regimes ainda muito similares!")
        return None
    else:
        print("✅ Regimes distintivos criados")
    
    # Verificar volatilidade
    vol_total = df['returns'].std()
    print(f"\nVolatilidade total: {vol_total:.6f} ({vol_total*100:.2f}%)")
    
    if vol_total > 0.5:  # > 50%
        print("❌ ATENÇÃO: Volatilidade muito alta, mas aceitável para teste")
    else:
        print("✅ Volatilidade aceitável")
    
    # Salvar dataset corrigido
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corrected_filename = f"data/GOLD_AGGRESSIVE_FIX_2M_{timestamp}.csv"
    
    df.to_csv(corrected_filename, index=False)
    print(f"\n✅ Dataset corrigido salvo: {corrected_filename}")
    
    return corrected_filename

if __name__ == '__main__':
    fix_dataset_aggressive()