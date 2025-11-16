#!/usr/bin/env python3
"""
🔍 TESTE DE QUALIDADE RÁPIDO - Análise Essencial
Verifica NANs, preços estáticos e formato XXXX.XX
"""

import pandas as pd
import numpy as np

def teste_qualidade_rapido():
    """Teste rápido de qualidade essencial"""
    print("🔍 TESTE QUALIDADE RÁPIDO")
    print("="*50)
    
    filepath = "data/GC=F_HYBRID_V2_3Y_1MIN_20250911_200306.csv"
    
    # Ler apenas primeiras linhas para análise rápida
    print("📂 Lendo amostra do dataset...")
    df_sample = pd.read_csv(filepath, nrows=10000)  # Primeiras 10k linhas
    print(f"✅ Amostra carregada: {len(df_sample):,} linhas")
    
    # 1. TESTE NANs
    print("\n🚨 TESTE NANs:")
    total_nans = df_sample.isnull().sum().sum()
    if total_nans == 0:
        print("✅ SEM NANs na amostra")
    else:
        print(f"❌ {total_nans:,} NANs encontrados")
        nan_cols = df_sample.isnull().sum()
        for col, count in nan_cols[nan_cols > 0].items():
            print(f"   {col}: {count}")
    
    # 2. TESTE FORMATO PREÇOS
    print("\n💰 TESTE FORMATO XXXX.XX:")
    price_cols = ['open', 'high', 'low', 'close']
    
    for col in price_cols:
        if col in df_sample.columns:
            # Verificar sample de preços
            sample_prices = df_sample[col].head(100).tolist()
            
            # Contar decimais
            decimal_counts = []
            for price in sample_prices:
                if pd.notna(price):
                    price_str = str(float(price))
                    if '.' in price_str:
                        decimal_part = price_str.split('.')[1].rstrip('0')
                        decimal_counts.append(len(decimal_part))
                    else:
                        decimal_counts.append(0)
            
            if decimal_counts:
                max_decimals = max(decimal_counts)
                avg_decimals = np.mean(decimal_counts)
                
                if max_decimals <= 2:
                    print(f"✅ {col}: Formato OK (máx {max_decimals} decimais)")
                else:
                    print(f"❌ {col}: Formato ERRADO (máx {max_decimals} decimais)")
                
                # Mostrar exemplos
                examples = [f"{p:.4f}" for p in sample_prices[:5] if pd.notna(p)]
                print(f"   Exemplos: {', '.join(examples)}")
    
    # 3. TESTE PREÇOS ESTÁTICOS SIMPLES
    print("\n📊 TESTE PREÇOS ESTÁTICOS:")
    
    for col in price_cols:
        if col in df_sample.columns:
            # Verificar quantos preços únicos
            unique_prices = df_sample[col].nunique()
            total_rows = len(df_sample[col].dropna())
            diversity_pct = (unique_prices / total_rows) * 100
            
            if diversity_pct > 90:
                print(f"✅ {col}: Boa diversidade ({diversity_pct:.1f}%)")
            elif diversity_pct > 70:
                print(f"⚠️ {col}: Diversidade OK ({diversity_pct:.1f}%)")
            else:
                print(f"❌ {col}: Baixa diversidade ({diversity_pct:.1f}%)")
    
    # 4. TESTE VALORES RANGE
    print("\n📈 TESTE RANGE DE PREÇOS:")
    
    for col in price_cols:
        if col in df_sample.columns:
            prices = df_sample[col].dropna()
            min_price = prices.min()
            max_price = prices.max()
            range_pct = ((max_price - min_price) / min_price) * 100
            
            print(f"{col}: ${min_price:.2f} - ${max_price:.2f} (variação: {range_pct:.2f}%)")
    
    # 5. TESTE BÁSICO OHLC
    print("\n🎯 TESTE OHLC BÁSICO:")
    
    violations = 0
    for i in range(min(1000, len(df_sample))):  # Testar primeiras 1000 linhas
        row = df_sample.iloc[i]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        
        if pd.notna(o) and pd.notna(h) and pd.notna(l) and pd.notna(c):
            if h < max(o, c) or l > min(o, c) or h < l:
                violations += 1
    
    if violations == 0:
        print("✅ OHLC: Sem violações na amostra")
    else:
        print(f"❌ OHLC: {violations} violações na amostra")
    
    print("\n" + "="*50)
    
    # RESUMO FINAL
    issues = []
    if total_nans > 0:
        issues.append("NANs detectados")
    if violations > 0:
        issues.append("Violações OHLC")
    
    # Verificar formato geral
    format_ok = True
    for col in price_cols:
        if col in df_sample.columns:
            sample_price = df_sample[col].dropna().iloc[0]
            decimals = len(str(float(sample_price)).split('.')[-1].rstrip('0'))
            if decimals > 2:
                format_ok = False
                break
    
    if not format_ok:
        issues.append("Formato de preços incorreto")
    
    if not issues:
        print("🎉 DATASET APROVADO: Qualidade boa na amostra!")
    else:
        print(f"⚠️ ISSUES ENCONTRADOS: {', '.join(issues)}")
    
    return df_sample

if __name__ == "__main__":
    df = teste_qualidade_rapido()