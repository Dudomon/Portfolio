#!/usr/bin/env python3
"""
🔍 TESTE DE QUALIDADE DATASET - Análise Completa
Verifica NANs, preços estáticos, outliers e qualidade geral
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def teste_qualidade_dataset(filepath):
    """Teste completo de qualidade do dataset"""
    print("🔍 TESTE DE QUALIDADE DATASET")
    print("="*60)
    
    # Carregar dataset
    print(f"📂 Carregando: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset carregado: {len(df):,} barras")
    
    # 1. TESTE DE NANs
    print("\n🚨 TESTE 1: NANs e Valores Ausentes")
    print("-"*40)
    
    nan_counts = df.isnull().sum()
    total_nans = nan_counts.sum()
    
    if total_nans == 0:
        print("✅ SEM NANs: Dataset limpo!")
    else:
        print(f"❌ ENCONTRADOS {total_nans:,} NANs:")
        nan_cols = nan_counts[nan_counts > 0]
        for col, count in nan_cols.items():
            pct = (count / len(df)) * 100
            print(f"   {col}: {count:,} ({pct:.2f}%)")
    
    # 2. TESTE DE PREÇOS ESTÁTICOS
    print("\n📊 TESTE 2: Preços Estáticos")
    print("-"*40)
    
    price_cols = ['open', 'high', 'low', 'close']
    
    for col in price_cols:
        if col in df.columns:
            # Verificar sequências estáticas (mesmo preço por >10 barras)
            static_sequences = []
            current_price = None
            current_count = 0
            
            for i, price in enumerate(df[col]):
                if price == current_price:
                    current_count += 1
                else:
                    if current_count > 10:  # >10 barras com mesmo preço
                        static_sequences.append({
                            'price': current_price,
                            'count': current_count,
                            'start_idx': i - current_count,
                            'end_idx': i - 1
                        })
                    current_price = price
                    current_count = 1
            
            # Verificar última sequência
            if current_count > 10:
                static_sequences.append({
                    'price': current_price,
                    'count': current_count,
                    'start_idx': len(df) - current_count,
                    'end_idx': len(df) - 1
                })
            
            if static_sequences:
                print(f"⚠️ {col}: {len(static_sequences)} sequências estáticas encontradas")
                for seq in static_sequences[:3]:  # Mostrar primeiras 3
                    print(f"   Preço {seq['price']} por {seq['count']} barras (idx {seq['start_idx']}-{seq['end_idx']})")
            else:
                print(f"✅ {col}: Sem sequências estáticas")
    
    # 3. TESTE DE OHLC CONSISTENCY
    print("\n🎯 TESTE 3: Consistência OHLC")
    print("-"*40)
    
    ohlc_violations = 0
    violations_details = []
    
    for i in range(len(df)):
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
        
        violations = []
        if h < max(o, c):
            violations.append(f"High {h} < max(O:{o}, C:{c})")
        if l > min(o, c):
            violations.append(f"Low {l} > min(O:{o}, C:{c})")
        if h < l:
            violations.append(f"High {h} < Low {l}")
            
        if violations:
            ohlc_violations += 1
            if len(violations_details) < 5:  # Mostrar primeiros 5
                violations_details.append(f"Linha {i}: {', '.join(violations)}")
    
    if ohlc_violations == 0:
        print("✅ OHLC: Todas as relações estão corretas")
    else:
        pct = (ohlc_violations / len(df)) * 100
        print(f"❌ OHLC: {ohlc_violations:,} violações ({pct:.3f}%)")
        for detail in violations_details:
            print(f"   {detail}")
    
    # 4. TESTE DE OUTLIERS
    print("\n📈 TESTE 4: Outliers de Preço")
    print("-"*40)
    
    close_prices = df['close'].dropna()
    q1 = close_prices.quantile(0.25)
    q3 = close_prices.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = close_prices[(close_prices < lower_bound) | (close_prices > upper_bound)]
    
    if len(outliers) == 0:
        print("✅ OUTLIERS: Nenhum outlier extremo encontrado")
    else:
        pct = (len(outliers) / len(close_prices)) * 100
        print(f"⚠️ OUTLIERS: {len(outliers):,} outliers ({pct:.3f}%)")
        print(f"   Range normal: ${lower_bound:.2f} - ${upper_bound:.2f}")
        print(f"   Outliers: ${outliers.min():.2f} - ${outliers.max():.2f}")
    
    # 5. TESTE DE CONTINUIDADE TEMPORAL
    print("\n⏰ TESTE 5: Continuidade Temporal")
    print("-"*40)
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calcular gaps
    time_diffs = df['time'].diff()
    expected_diff = pd.Timedelta(minutes=1)
    
    # Gaps maiores que 4 minutos
    large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=4)]
    
    if len(large_gaps) == 0:
        print("✅ TEMPO: Continuidade perfeita (gaps <= 4 min)")
    else:
        print(f"⚠️ TEMPO: {len(large_gaps):,} gaps grandes (>4 min)")
        gap_sizes = large_gaps.dt.total_seconds() / 3600  # Converter para horas
        print(f"   Gap médio: {gap_sizes.mean():.1f}h")
        print(f"   Gap máximo: {gap_sizes.max():.1f}h")
    
    # 6. ESTATÍSTICAS GERAIS
    print("\n📊 TESTE 6: Estatísticas Gerais")
    print("-"*40)
    
    print(f"Período: {df['time'].min()} → {df['time'].max()}")
    print(f"Total de dias: {(df['time'].max() - df['time'].min()).days:,}")
    print(f"Preço inicial: ${df['close'].iloc[0]:.2f}")
    print(f"Preço final: ${df['close'].iloc[-1]:.2f}")
    
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"Retorno total: {total_return:+.2f}%")
    
    # Volatilidade diária estimada
    daily_returns = df.groupby(df['time'].dt.date)['close'].last().pct_change().dropna()
    daily_vol = daily_returns.std() * np.sqrt(252) * 100
    print(f"Volatilidade anual estimada: {daily_vol:.1f}%")
    
    # Volume
    if 'tick_volume' in df.columns:
        avg_volume = df['tick_volume'].mean()
        print(f"Volume médio: {avg_volume:.0f}")
    
    # 7. TESTE DE FORMATO DE PREÇOS
    print("\n💰 TESTE 7: Formato de Preços (XXXX.XX)")
    print("-"*40)
    
    # Verificar se todos os preços têm máximo 2 casas decimais
    for col in price_cols:
        if col in df.columns:
            # Contar casas decimais
            decimals = df[col].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            max_decimals = decimals.max()
            
            if max_decimals <= 2:
                print(f"✅ {col}: Formato correto (máx {max_decimals} decimais)")
            else:
                wrong_format = (decimals > 2).sum()
                pct = (wrong_format / len(df)) * 100
                print(f"❌ {col}: {wrong_format:,} valores com >2 decimais ({pct:.3f}%)")
    
    print("\n" + "="*60)
    print("🏁 TESTE DE QUALIDADE CONCLUÍDO")
    
    # Resumo final
    issues = []
    if total_nans > 0:
        issues.append(f"{total_nans:,} NANs")
    if ohlc_violations > 0:
        issues.append(f"{ohlc_violations:,} violações OHLC")
    if len(outliers) > 0:
        issues.append(f"{len(outliers):,} outliers")
    if len(large_gaps) > 0:
        issues.append(f"{len(large_gaps):,} gaps temporais")
    
    if not issues:
        print("🎉 QUALIDADE EXCELENTE: Dataset aprovado!")
    else:
        print(f"⚠️ ISSUES ENCONTRADAS: {', '.join(issues)}")
    
    return df

if __name__ == "__main__":
    filepath = "data/GC=F_PREMIUM_3Y_1MIN_20250911_192559.csv"
    df = teste_qualidade_dataset(filepath)