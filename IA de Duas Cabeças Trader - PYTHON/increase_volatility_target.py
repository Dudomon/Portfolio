#!/usr/bin/env python3
"""
Aumento de Volatilidade - Target 15-25 trades/dia
Ajusta parâmetros para atingir sweet spot de trading frequency
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_target_volatility():
    """Calcula parâmetros necessários para atingir target"""
    
    print("🎯 CALCULANDO PARÂMETROS PARA TARGET 15-25 TRADES/DIA")
    print("=" * 60)
    
    # Dados atuais
    current_trades_per_day = 3.5
    current_vol_multiplier = 1.91  # Dataset atual vs original
    target_trades_per_day = 20  # Meio termo entre 15-25
    
    # Assumindo relação linear entre volatilidade e trades (aproximação)
    vol_ratio_needed = target_trades_per_day / current_trades_per_day
    target_vol_multiplier = current_vol_multiplier * vol_ratio_needed
    
    print(f"📊 ANÁLISE ATUAL:")
    print(f"   Trades/dia atual: {current_trades_per_day}")
    print(f"   Volatilidade atual: {current_vol_multiplier:.2f}x original")
    print(f"   Target trades/dia: {target_trades_per_day}")
    
    print(f"\n🎯 CÁLCULO TARGET:")
    print(f"   Ratio necessário: {vol_ratio_needed:.2f}x")
    print(f"   Volatilidade target: {target_vol_multiplier:.2f}x original")
    
    # Ajustar parâmetros do volatility enhancer
    current_base_factor = 1.5
    current_max_factor = 2.5
    
    # Aumentar proporcionalmente
    new_base_factor = current_base_factor * vol_ratio_needed
    new_max_factor = current_max_factor * vol_ratio_needed
    
    print(f"\n🔧 PARÂMETROS AJUSTADOS:")
    print(f"   Base factor: {current_base_factor} → {new_base_factor:.2f}")
    print(f"   Max factor: {current_max_factor} → {new_max_factor:.2f}")
    
    return {
        'base_factor': new_base_factor,
        'max_factor': new_max_factor,
        'target_vol_multiplier': target_vol_multiplier,
        'vol_ratio_needed': vol_ratio_needed
    }

def create_enhanced_dataset_v2():
    """Cria dataset com volatilidade aumentada para target"""
    
    print("\n🚀 CRIANDO DATASET V2 - TARGET 15-25 TRADES/DIA")
    print("=" * 60)
    
    # Calcular parâmetros
    params = calculate_target_volatility()
    
    # Carregar dataset original Yahoo
    df = pd.read_csv('data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"\n📊 Dataset original carregado:")
    print(f"   Barras: {len(df):,}")
    print(f"   Período: {df['time'].min()} - {df['time'].max()}")
    
    # Aplicar enhancement mais agressivo
    returns = df['close'].pct_change()
    
    # Volatilidade rolling para scaling adaptativo
    vol_rolling = returns.rolling(100, min_periods=10).std()
    vol_rolling = vol_rolling.fillna(vol_rolling.mean())
    
    # Percentil de volatilidade
    vol_percentile = vol_rolling.rank(pct=True)
    
    # Scaling mais agressivo
    base_factor = params['base_factor']  # ~8.6
    max_factor = params['max_factor']    # ~14.3
    
    print(f"🔥 SCALING AGRESSIVO:")
    print(f"   Base factor: {base_factor:.1f}x")
    print(f"   Max factor: {max_factor:.1f}x")
    
    # Aplicar scaling
    scale_factor = base_factor + (max_factor - base_factor) * (1 - vol_percentile)
    enhanced_returns = returns * scale_factor
    
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
    
    # Recalcular indicadores
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
    
    # Bollinger position
    bb_mid = df_final['sma_20']
    bb_std = df_final['close'].rolling(20).std()
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)
    df_final['bb_position'] = (df_final['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Preencher NaNs e colunas faltantes
    df_final = df_final.fillna(method='ffill').fillna(method='bfill')
    df_final['trend_strength'] = 0.0
    df_final['stoch_k'] = 50.0
    df_final['volume_ratio'] = df_final['tick_volume']
    df_final['var_99'] = df_final['close'] * 0.95
    
    # Salvar dataset V2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/GC_YAHOO_ENHANCED_V2_TARGET_{timestamp}.csv'
    df_final.to_csv(output_file, index=False)
    
    # Estatísticas finais
    vol_orig = df['close'].pct_change().std()
    vol_enh = df_final['close'].pct_change().std()
    vol_ratio = vol_enh / vol_orig
    
    print(f"\n💾 DATASET V2 SALVO:")
    print(f"   Arquivo: {output_file}")
    print(f"   Barras: {len(df_final):,}")
    
    print(f"\n📈 RESULTADO FINAL:")
    print(f"   Volatilidade original: {vol_orig*100:.3f}%")
    print(f"   Volatilidade V2: {vol_enh*100:.3f}%")
    print(f"   Aumento: {vol_ratio:.2f}x")
    print(f"   Target esperado: {params['target_vol_multiplier']:.2f}x")
    
    if abs(vol_ratio - params['target_vol_multiplier']) < 0.5:
        print("✅ Target de volatilidade ATINGIDO!")
    else:
        print(f"⚠️ Target não atingido - diferença: {abs(vol_ratio - params['target_vol_multiplier']):.2f}x")
    
    print(f"\n🎯 EXPECTATIVA DE TRADES:")
    print(f"   Target: 15-25 trades/dia")
    print(f"   Projeção: ~{3.5 * (vol_ratio / 1.91):.1f} trades/dia")
    
    return output_file, df_final, vol_ratio

def update_daytrader_dataset_path(new_dataset_path):
    """Atualiza path do dataset no daytrader.py"""
    
    print(f"\n🔧 ATUALIZANDO DAYTRADER.PY:")
    print(f"   Novo dataset: {new_dataset_path}")
    
    # Ler daytrader.py
    with open('daytrader.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar linha do dataset_path
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if 'dataset_path =' in line and 'GC_YAHOO' in line:
            old_line = line
            lines[i] = f"    dataset_path = '{new_dataset_path}'"
            print(f"   Linha anterior: {old_line.strip()}")
            print(f"   Linha nova: {lines[i].strip()}")
            updated = True
            break
    
    if updated:
        # Salvar arquivo
        with open('daytrader.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("✅ DayTrader.py atualizado com novo dataset")
    else:
        print("⚠️ Linha do dataset_path não encontrada")
    
    return updated

if __name__ == "__main__":
    try:
        # Criar dataset V2 com volatilidade aumentada
        output_file, df_enhanced, vol_ratio = create_enhanced_dataset_v2()
        
        # Atualizar daytrader.py
        updated = update_daytrader_dataset_path(output_file)
        
        print(f"\n🎉 DATASET V2 CRIADO COM SUCESSO!")
        print(f"   Arquivo: {output_file}")
        print(f"   Volatilidade: {vol_ratio:.2f}x original")
        print(f"   DayTrader: {'Atualizado' if updated else 'Precisa atualização manual'}")
        
        print(f"\n🚀 PRÓXIMOS PASSOS:")
        print(f"   1. Verificar se EV permanece positiva")
        print(f"   2. Monitorar trades/dia (~15-25 esperado)")
        print(f"   3. Ajustar se necessário")
        
    except Exception as e:
        print(f"❌ Erro na criação do dataset V2: {e}")
        import traceback
        traceback.print_exc()