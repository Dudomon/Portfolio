#!/usr/bin/env python3
"""
Fix Volatility Scaling - Reduzir para nível razoável
Criar dataset V3 com volatilidade moderada que não quebra sistema
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_balanced_dataset_v3():
    """Cria dataset V3 com volatilidade balanceada"""
    
    print("🔧 CRIANDO DATASET V3 - VOLATILIDADE BALANCEADA")
    print("=" * 55)
    
    # Carregar dataset original
    df = pd.read_csv('data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"📊 Dataset original carregado: {len(df):,} barras")
    
    # Target: 3.5x volatilidade original (ao invés de 8.98x)
    target_multiplier = 3.5
    print(f"🎯 Target de volatilidade: {target_multiplier}x original")
    
    # Calcular parâmetros mais conservadores
    base_factor = 2.0   # Ao invés de 8.6
    max_factor = 4.0    # Ao invés de 14.3
    
    print(f"🔧 Parâmetros conservadores:")
    print(f"   Base factor: {base_factor}x")
    print(f"   Max factor: {max_factor}x")
    
    # Aplicar scaling moderado
    returns = df['close'].pct_change()
    vol_rolling = returns.rolling(100, min_periods=10).std()
    vol_rolling = vol_rolling.fillna(vol_rolling.mean())
    
    # Percentil de volatilidade
    vol_percentile = vol_rolling.rank(pct=True)
    
    # Scaling mais conservador
    scale_factor = base_factor + (max_factor - base_factor) * (1 - vol_percentile)
    enhanced_returns = returns * scale_factor
    
    # Limitar returns extremos para evitar quebra da normalização
    enhanced_returns = np.clip(enhanced_returns, -0.05, 0.05)  # Max ±5%
    
    # Reconstruir preços
    price_multiplier = (1 + enhanced_returns).cumprod()
    first_close = df['close'].iloc[0]
    df['close_enhanced'] = price_multiplier * first_close
    
    # Ajustar OHLC
    close_ratio = df['close_enhanced'] / df['close']
    close_ratio = close_ratio.fillna(1.0)
    
    df['open_enhanced'] = df['open'] * close_ratio
    df['high_enhanced'] = df['high'] * close_ratio
    df['low_enhanced'] = df['low'] * close_ratio
    
    # Fix OHLC consistency
    df['high_enhanced'] = np.maximum.reduce([
        df['high_enhanced'], df['open_enhanced'], df['close_enhanced']
    ])
    df['low_enhanced'] = np.minimum.reduce([
        df['low_enhanced'], df['open_enhanced'], df['close_enhanced']
    ])
    
    # Preparar dataset final
    df_final = df[['time', 'open_enhanced', 'high_enhanced', 'low_enhanced', 'close_enhanced']].copy()
    df_final.columns = ['time', 'open', 'high', 'low', 'close']
    
    # Manter outras colunas
    df_final['tick_volume'] = df['tick_volume']
    df_final['spread'] = df['spread']
    df_final['real_volume'] = df['real_volume']
    
    # Recalcular indicadores com cuidado
    df_final['returns'] = df_final['close'].pct_change()
    df_final['volatility_20'] = df_final['returns'].rolling(20).std()
    df_final['sma_20'] = df_final['close'].rolling(20).mean()
    df_final['sma_50'] = df_final['close'].rolling(50).mean()
    
    # RSI
    delta = df_final['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_final['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df_final['high'] - df_final['low']
    hc = abs(df_final['high'] - df_final['close'].shift())
    lc = abs(df_final['low'] - df_final['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_final['atr_14'] = tr.rolling(14).mean()
    
    # Bollinger position - com proteção contra infinitos
    bb_mid = df_final['sma_20']
    bb_std = df_final['close'].rolling(20).std()
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)
    
    # Proteger contra divisão por zero
    bb_range = bb_upper - bb_lower
    bb_range = bb_range.replace(0, np.nan)
    df_final['bb_position'] = (df_final['close'] - bb_lower) / bb_range
    
    # Limitar BB position a range razoável
    df_final['bb_position'] = np.clip(df_final['bb_position'], -2, 3)
    
    # Preencher NaNs e colunas faltantes
    df_final = df_final.fillna(method='ffill').fillna(method='bfill')
    df_final['trend_strength'] = 0.0
    df_final['stoch_k'] = 50.0
    df_final['volume_ratio'] = df_final['tick_volume']
    df_final['var_99'] = df_final['close'] * 0.95
    
    # Validar dataset
    print(f"\n✅ VALIDAÇÃO DATASET V3:")
    
    # Verificar ranges
    price_range = df_final['close'].max() - df_final['close'].min()
    price_std = df_final['close'].std()
    price_mean = df_final['close'].mean()
    coef_var = price_std / price_mean
    
    print(f"   Preços: [{df_final['close'].min():.0f}, {df_final['close'].max():.0f}]")
    print(f"   Coef. variação: {coef_var*100:.1f}%")
    
    # Verificar BB position
    bb_issues = df_final['bb_position'].isna().sum() + np.isinf(df_final['bb_position']).sum()
    print(f"   BB position issues: {bb_issues}")
    
    # Verificar volatilidade final
    vol_orig = df['close'].pct_change().std()
    vol_v3 = df_final['close'].pct_change().std()
    vol_ratio = vol_v3 / vol_orig
    
    print(f"   Volatilidade original: {vol_orig*100:.3f}%")
    print(f"   Volatilidade V3: {vol_v3*100:.3f}%")
    print(f"   Multiplicador: {vol_ratio:.2f}x")
    
    # Salvar dataset V3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/GC_YAHOO_ENHANCED_V3_BALANCED_{timestamp}.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n💾 DATASET V3 SALVO:")
    print(f"   Arquivo: {output_file}")
    print(f"   Barras: {len(df_final):,}")
    
    # Projeção de trades
    projected_trades = 3.5 * (vol_ratio / 1.91)  # Base na experiência anterior
    print(f"\n🎯 PROJEÇÃO:")
    print(f"   Trades/dia esperados: ~{projected_trades:.1f}")
    print(f"   Target: 8-12 trades/dia")
    
    if 8 <= projected_trades <= 12:
        print("✅ Projeção dentro do target!")
    else:
        print("⚠️ Projeção fora do target - pode precisar ajuste")
    
    return output_file, df_final, vol_ratio

def update_daytrader_to_v3(dataset_path):
    """Atualiza daytrader.py para usar dataset V3"""
    
    print(f"\n🔧 ATUALIZANDO DAYTRADER PARA V3:")
    
    try:
        with open('daytrader.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        updated = False
        
        for i, line in enumerate(lines):
            if 'dataset_path =' in line and 'GC_YAHOO' in line:
                old_line = line
                lines[i] = f"    dataset_path = '{dataset_path}'"
                print(f"   ✅ Linha atualizada:")
                print(f"      Antes: {old_line.strip()}")
                print(f"      Depois: {lines[i].strip()}")
                updated = True
                break
        
        if updated:
            with open('daytrader.py', 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"   ✅ DayTrader.py atualizado!")
        else:
            print(f"   ⚠️ Linha do dataset não encontrada")
        
        return updated
        
    except Exception as e:
        print(f"   ❌ Erro ao atualizar: {e}")
        return False

if __name__ == "__main__":
    try:
        print("🚨 CORRIGINDO PROBLEMA DE VOLATILIDADE EXCESSIVA")
        print("Criando dataset V3 com volatilidade balanceada...")
        
        # Criar dataset V3 balanceado
        output_file, df_v3, vol_ratio = create_balanced_dataset_v3()
        
        # Atualizar daytrader
        updated = update_daytrader_to_v3(output_file)
        
        print(f"\n🎉 DATASET V3 CRIADO COM SUCESSO!")
        print(f"   Arquivo: {output_file}")
        print(f"   Volatilidade: {vol_ratio:.2f}x original (vs 8.98x do V2)")
        print(f"   DayTrader: {'✅ Atualizado' if updated else '❌ Precisa atualização manual'}")
        
        print(f"\n🎯 BENEFÍCIOS ESPERADOS:")
        print(f"   - Normalização estável")
        print(f"   - BB position sem infinitos")
        print(f"   - SL/TP adequados")
        print(f"   - ~8-12 trades/dia")
        print(f"   - Convergência mantida")
        
    except Exception as e:
        print(f"❌ Erro na correção: {e}")
        import traceback
        traceback.print_exc()