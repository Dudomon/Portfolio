#!/usr/bin/env python3
"""
🔍 ANÁLISE VOLATILIDADE: Atual vs V3 Balanced vs Yahoo Original
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_volatility_metrics(df, name):
    """Calcula métricas de volatilidade para um dataset"""
    print(f"\n📊 ANÁLISE: {name}")
    print("=" * 60)
    
    # Calcular retornos percentuais
    df['returns'] = df['close'].pct_change().dropna()
    
    # Métricas básicas
    volatility = df['returns'].std() * 100  # Convert to %
    mean_return = df['returns'].mean() * 100
    daily_range = ((df['high'] - df['low']) / df['close'] * 100).mean()
    
    # Métricas avançadas
    sharp_moves = (abs(df['returns']) > df['returns'].std() * 2).sum()
    total_moves = len(df['returns'].dropna())
    sharp_ratio = sharp_moves / total_moves * 100
    
    # Distribuição de volatilidade
    volatility_quantiles = df['returns'].abs().quantile([0.5, 0.75, 0.9, 0.95, 0.99]).values * 100
    
    # Range intraday médio
    intraday_range = ((df['high'] - df['low']) / df['open'] * 100).mean()
    
    results = {
        'name': name,
        'total_bars': len(df),
        'volatility_pct': volatility,
        'mean_return_pct': mean_return,
        'daily_range_pct': daily_range,
        'sharp_moves': sharp_moves,
        'sharp_ratio_pct': sharp_ratio,
        'intraday_range_pct': intraday_range,
        'vol_50pct': volatility_quantiles[0],
        'vol_75pct': volatility_quantiles[1], 
        'vol_90pct': volatility_quantiles[2],
        'vol_95pct': volatility_quantiles[3],
        'vol_99pct': volatility_quantiles[4],
        'returns_data': df['returns'].dropna()
    }
    
    # Print results
    print(f"📈 Total de barras: {results['total_bars']:,}")
    print(f"📊 Volatilidade: {results['volatility_pct']:.3f}%")
    print(f"💰 Retorno médio: {results['mean_return_pct']:.4f}%")
    print(f"📏 Range diário médio: {results['daily_range_pct']:.2f}%")
    print(f"⚡ Movimentos bruscos (>2σ): {results['sharp_moves']:,} ({results['sharp_ratio_pct']:.2f}%)")
    print(f"🔄 Range intraday médio: {results['intraday_range_pct']:.2f}%")
    print(f"📊 Distribuição volatilidade:")
    print(f"   50%: {results['vol_50pct']:.3f}%")
    print(f"   75%: {results['vol_75pct']:.3f}%") 
    print(f"   90%: {results['vol_90pct']:.3f}%")
    print(f"   95%: {results['vol_95pct']:.3f}%")
    print(f"   99%: {results['vol_99pct']:.3f}%")
    
    return results

def main():
    print("🔍 ANÁLISE COMPARATIVA DE VOLATILIDADE")
    print("=" * 80)
    
    # Paths dos datasets
    current_dataset = "data/GOLD_SAFE_CHALLENGING_2M_20250801_203251.csv"
    v3_balanced_dataset = "data/GC_YAHOO_ENHANCED_V3_BALANCED_20250804_192226.csv"
    
    # Encontrar dataset Yahoo original mais recente
    yahoo_datasets = [
        "data/GC=F_YAHOO_DAILY_5MIN_20250711_041924.csv",
        "data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv"
    ]
    
    yahoo_dataset = None
    for path in yahoo_datasets:
        if Path(path).exists():
            yahoo_dataset = path
            break
    
    if not yahoo_dataset:
        print("❌ Dataset Yahoo original não encontrado!")
        return
    
    results = []
    
    # Analisar datasets
    datasets = [
        (current_dataset, "ATUAL (SAFE_CHALLENGING)"),
        (v3_balanced_dataset, "V3 BALANCED"), 
        (yahoo_dataset, "YAHOO ORIGINAL")
    ]
    
    for dataset_path, name in datasets:
        if not Path(dataset_path).exists():
            print(f"❌ Dataset não encontrado: {dataset_path}")
            continue
            
        try:
            print(f"\n🔄 Carregando {name}...")
            df = pd.read_csv(dataset_path)
            
            # Padronizar colunas
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            result = calculate_volatility_metrics(df, name)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Erro processando {name}: {e}")
    
    # Comparação final
    if len(results) >= 2:
        print("\n🎯 COMPARAÇÃO FINAL")
        print("=" * 80)
        
        # Criar tabela comparativa
        comparison_df = pd.DataFrame([{
            'Dataset': r['name'],
            'Barras': f"{r['total_bars']:,}",
            'Volatilidade': f"{r['volatility_pct']:.3f}%",
            'Range Diário': f"{r['daily_range_pct']:.2f}%", 
            'Movimentos Bruscos': f"{r['sharp_ratio_pct']:.2f}%",
            'Vol 95%': f"{r['vol_95pct']:.3f}%"
        } for r in results])
        
        print(comparison_df.to_string(index=False))
        
        # Ratios comparativos (usando Yahoo como baseline)
        yahoo_result = next((r for r in results if 'YAHOO' in r['name']), None)
        if yahoo_result:
            print(f"\n📊 RATIOS vs YAHOO ORIGINAL:")
            print("-" * 40)
            
            for result in results:
                if result['name'] == yahoo_result['name']:
                    continue
                    
                vol_ratio = result['volatility_pct'] / yahoo_result['volatility_pct']
                range_ratio = result['daily_range_pct'] / yahoo_result['daily_range_pct']
                sharp_ratio = result['sharp_ratio_pct'] / yahoo_result['sharp_ratio_pct']
                
                print(f"\n🔸 {result['name']}:")
                print(f"   Volatilidade: {vol_ratio:.2f}x Yahoo")
                print(f"   Range Diário: {range_ratio:.2f}x Yahoo") 
                print(f"   Movimentos Bruscos: {sharp_ratio:.2f}x Yahoo")
                
                if vol_ratio > 2.0:
                    print("   ⚠️  MUITO VOLÁTIL - Pode causar convergência prematura")
                elif vol_ratio < 0.5:
                    print("   ⚠️  POUCO VOLÁTIL - Pode ser muito fácil")
                else:
                    print("   ✅ VOLATILIDADE BALANCEADA")

if __name__ == "__main__":
    main()