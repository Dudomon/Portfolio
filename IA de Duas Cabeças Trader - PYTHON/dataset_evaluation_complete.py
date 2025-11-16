#!/usr/bin/env python3
"""
📊 AVALIAÇÃO COMPLETA DO DATASET SILUS
=====================================

Análise detalhada do dataset atual vs novo dataset MT5 de 25 semanas
"""

import sys
sys.path.append("D:/Projeto")

import pandas as pd
import numpy as np
import os
from datetime import datetime

def evaluate_silus_dataset():
    """📊 AVALIAÇÃO COMPLETA DO DATASET USADO PELO SILUS"""

    print("📊 AVALIAÇÃO COMPLETA DO DATASET SILUS")
    print("=" * 80)

    # 1. IDENTIFICAR E CARREGAR DATASET ATUAL DO SILUS
    current_dataset_path = 'data/GC=F_REALISTIC_V4_20250911_235945.csv'
    new_dataset_path = 'data/GOLD_1M_MT5_GOLD_25WEEKS_20250923_190721.pkl'

    print(f"🎯 DATASET ATUAL SILUS: {current_dataset_path}")
    print(f"🆕 DATASET NOVO MT5: {new_dataset_path}")
    print("")

    # Verificar se arquivos existem
    if not os.path.exists(current_dataset_path):
        print(f"❌ Dataset atual não encontrado: {current_dataset_path}")
        return

    if not os.path.exists(new_dataset_path):
        print(f"❌ Dataset novo não encontrado: {new_dataset_path}")
        return

    try:
        # Carregar dataset atual (CSV)
        print("📂 Carregando dataset atual do SILUS...")
        df_current = pd.read_csv(current_dataset_path)

        # Processar timestamp (seguindo lógica do SILUS)
        df_current['timestamp'] = pd.to_datetime(df_current['time'])
        df_current.set_index('timestamp', inplace=True)
        df_current.drop('time', axis=1, inplace=True)

        # Renomear colunas para compatibilidade
        df_current = df_current.rename(columns={
            'open': 'open_1m',
            'high': 'high_1m',
            'low': 'low_1m',
            'close': 'close_1m',
            'tick_volume': 'volume_1m'
        })

        print(f"✅ Dataset atual carregado: {len(df_current):,} barras")

        # Carregar dataset novo (PKL)
        print("📂 Carregando dataset novo MT5...")
        df_new = pd.read_pickle(new_dataset_path)
        print(f"✅ Dataset novo carregado: {len(df_new):,} barras")

    except Exception as e:
        print(f"❌ Erro ao carregar datasets: {e}")
        return

    # 2. ANÁLISE COMPARATIVA DETALHADA
    print("\n" + "=" * 60)
    print("📊 ANÁLISE COMPARATIVA DETALHADA")
    print("=" * 60)

    # Informações básicas
    print(f"\n📈 INFORMAÇÕES BÁSICAS:")
    print(f"   Dataset Atual SILUS:")
    print(f"     - Barras: {len(df_current):,}")
    print(f"     - Período: {df_current.index.min()} até {df_current.index.max()}")
    print(f"     - Duração: {(df_current.index.max() - df_current.index.min()).days} dias")
    print(f"     - Colunas: {list(df_current.columns)}")

    print(f"\n   Dataset Novo MT5:")
    print(f"     - Barras: {len(df_new):,}")
    print(f"     - Período: {df_new['timestamp'].min()} até {df_new['timestamp'].max()}")
    print(f"     - Duração: {(df_new['timestamp'].max() - df_new['timestamp'].min()).days} dias")
    print(f"     - Colunas: {list(df_new.columns)}")

    # 3. ANÁLISE DE QUALIDADE DOS DADOS
    print(f"\n📊 ANÁLISE DE QUALIDADE DOS DADOS:")

    # Dataset atual
    print(f"\n🔍 DATASET ATUAL SILUS:")
    missing_current = df_current.isnull().sum()
    duplicates_current = df_current.index.duplicated().sum()

    print(f"   Missing values por coluna:")
    for col, missing in missing_current.items():
        if missing > 0:
            print(f"     {col}: {missing:,} ({missing/len(df_current)*100:.2f}%)")

    print(f"   Timestamps duplicados: {duplicates_current}")

    # Estatísticas básicas do close
    if 'close_1m' in df_current.columns:
        close_stats_current = df_current['close_1m'].describe()
        print(f"   Close statistics:")
        print(f"     Min: ${close_stats_current['min']:.2f}")
        print(f"     Max: ${close_stats_current['max']:.2f}")
        print(f"     Mean: ${close_stats_current['mean']:.2f}")
        print(f"     Std: ${close_stats_current['std']:.2f}")

        # Gaps grandes
        if len(df_current) > 1:
            price_changes = df_current['close_1m'].pct_change().abs()
            large_gaps = (price_changes > 0.05).sum()  # Mudanças > 5%
            print(f"   Gaps grandes (>5%): {large_gaps}")

    # Dataset novo
    print(f"\n🔍 DATASET NOVO MT5:")
    missing_new = df_new.isnull().sum()
    duplicates_new = df_new['timestamp'].duplicated().sum()

    print(f"   Missing values por coluna:")
    for col, missing in missing_new.items():
        if missing > 0:
            print(f"     {col}: {missing:,} ({missing/len(df_new)*100:.2f}%)")

    print(f"   Timestamps duplicados: {duplicates_new}")

    # Estatísticas básicas do close
    if 'close_1m' in df_new.columns:
        close_stats_new = df_new['close_1m'].describe()
        print(f"   Close statistics:")
        print(f"     Min: ${close_stats_new['min']:.2f}")
        print(f"     Max: ${close_stats_new['max']:.2f}")
        print(f"     Mean: ${close_stats_new['mean']:.2f}")
        print(f"     Std: ${close_stats_new['std']:.2f}")

        # Gaps grandes
        if len(df_new) > 1:
            price_changes = df_new['close_1m'].pct_change().abs()
            large_gaps = (price_changes > 0.05).sum()  # Mudanças > 5%
            print(f"   Gaps grandes (>5%): {large_gaps}")

    # 4. ANÁLISE DE VOLUME E LIQUIDEZ
    print(f"\n💧 ANÁLISE DE VOLUME E LIQUIDEZ:")

    if 'volume_1m' in df_current.columns:
        volume_current = df_current['volume_1m']
        print(f"\n📊 DATASET ATUAL:")
        print(f"   Volume médio: {volume_current.mean():,.0f}")
        print(f"   Volume mediano: {volume_current.median():,.0f}")
        print(f"   Volume zero/baixo (<100): {(volume_current < 100).sum():,} barras")

    if 'volume_1m' in df_new.columns:
        volume_new = df_new['volume_1m']
        print(f"\n📊 DATASET NOVO:")
        print(f"   Volume médio: {volume_new.mean():,.0f}")
        print(f"   Volume mediano: {volume_new.median():,.0f}")
        print(f"   Volume zero/baixo (<100): {(volume_new < 100).sum():,} barras")

    # 5. ANÁLISE DE INDICADORES TÉCNICOS
    print(f"\n📈 ANÁLISE DE INDICADORES TÉCNICOS:")

    # Dataset atual
    technical_cols_current = [col for col in df_current.columns if any(ind in col.lower() for ind in ['rsi', 'sma', 'ema', 'bb', 'atr', 'stoch'])]
    print(f"\n🔍 DATASET ATUAL:")
    print(f"   Indicadores disponíveis: {len(technical_cols_current)}")
    if technical_cols_current:
        print(f"   Colunas: {technical_cols_current[:10]}")  # Primeiros 10

        # Verificar se indicadores têm valores válidos
        for col in technical_cols_current[:5]:  # Verificar primeiros 5
            valid_values = df_current[col].notna().sum()
            print(f"     {col}: {valid_values:,}/{len(df_current):,} valores válidos ({valid_values/len(df_current)*100:.1f}%)")

    # Dataset novo
    technical_cols_new = [col for col in df_new.columns if any(ind in col.lower() for ind in ['rsi', 'sma', 'ema', 'bb', 'atr', 'stoch'])]
    print(f"\n🔍 DATASET NOVO:")
    print(f"   Indicadores disponíveis: {len(technical_cols_new)}")
    if technical_cols_new:
        print(f"   Colunas: {technical_cols_new[:10]}")  # Primeiros 10

        # Verificar se indicadores têm valores válidos
        for col in technical_cols_new[:5]:  # Verificar primeiros 5
            valid_values = df_new[col].notna().sum()
            print(f"     {col}: {valid_values:,}/{len(df_new):,} valores válidos ({valid_values/len(df_new)*100:.1f}%)")

    # 6. ANÁLISE DE PERIODICIDADE E GAPS
    print(f"\n⏰ ANÁLISE DE PERIODICIDADE E GAPS:")

    # Dataset atual
    if len(df_current) > 1:
        time_diffs_current = df_current.index.to_series().diff().dropna()
        mode_interval_current = time_diffs_current.mode()[0] if len(time_diffs_current.mode()) > 0 else None
        gaps_current = (time_diffs_current > pd.Timedelta(minutes=2)).sum()

        print(f"\n📊 DATASET ATUAL:")
        print(f"   Intervalo modal: {mode_interval_current}")
        print(f"   Gaps temporais (>2min): {gaps_current}")
        print(f"   Cobertura horária: {time_diffs_current.describe()}")

    # Dataset novo
    if len(df_new) > 1:
        df_new_sorted = df_new.sort_values('timestamp')
        time_diffs_new = df_new_sorted['timestamp'].diff().dropna()
        mode_interval_new = time_diffs_new.mode()[0] if len(time_diffs_new.mode()) > 0 else None
        gaps_new = (time_diffs_new > pd.Timedelta(minutes=2)).sum()

        print(f"\n📊 DATASET NOVO:")
        print(f"   Intervalo modal: {mode_interval_new}")
        print(f"   Gaps temporais (>2min): {gaps_new}")
        print(f"   Cobertura horária: {time_diffs_new.describe()}")

    # 7. RECOMENDAÇÕES
    print(f"\n" + "=" * 60)
    print("🎯 RECOMENDAÇÕES E CONCLUSÕES")
    print("=" * 60)

    # Comparar qualidade
    print(f"\n📋 COMPARATIVO DE QUALIDADE:")

    # Duração
    days_current = (df_current.index.max() - df_current.index.min()).days
    days_new = (df_new['timestamp'].max() - df_new['timestamp'].min()).days

    print(f"   Duração: Atual={days_current} dias vs Novo={days_new} dias")
    if days_new > days_current:
        print(f"     ✅ Dataset novo tem mais histórico")
    else:
        print(f"     ⚠️ Dataset atual tem mais histórico")

    # Volume de dados
    print(f"   Volume de dados: Atual={len(df_current):,} vs Novo={len(df_new):,}")
    if len(df_new) > len(df_current):
        print(f"     ✅ Dataset novo tem mais barras")
    else:
        print(f"     ⚠️ Dataset atual tem mais barras")

    # Atualidade
    max_date_current = df_current.index.max()
    max_date_new = df_new['timestamp'].max()

    print(f"   Atualidade: Atual={max_date_current} vs Novo={max_date_new}")
    if max_date_new > max_date_current:
        print(f"     ✅ Dataset novo é mais recente")
    else:
        print(f"     ⚠️ Dataset atual é mais recente")

    # Indicadores técnicos
    print(f"   Indicadores: Atual={len(technical_cols_current)} vs Novo={len(technical_cols_new)}")
    if len(technical_cols_new) > len(technical_cols_current):
        print(f"     ✅ Dataset novo tem mais indicadores")
    elif len(technical_cols_current) > len(technical_cols_new):
        print(f"     ⚠️ Dataset atual tem mais indicadores")
    else:
        print(f"     = Datasets têm quantidade similar de indicadores")

    print(f"\n🚀 RECOMENDAÇÃO FINAL:")

    # Score simples
    score_new = 0
    score_current = 0

    if days_new > days_current: score_new += 1
    else: score_current += 1

    if len(df_new) > len(df_current): score_new += 1
    else: score_current += 1

    if max_date_new > max_date_current: score_new += 1
    else: score_current += 1

    if len(technical_cols_new) >= len(technical_cols_current): score_new += 1
    else: score_current += 1

    print(f"   Score comparativo: Novo={score_new}/4 vs Atual={score_current}/4")

    if score_new > score_current:
        print(f"   ✅ RECOMENDADO: Migrar para dataset novo MT5")
        print(f"   📝 Vantagens: Mais recente, dados diretos MT5, sem interpolação")
    elif score_current > score_new:
        print(f"   ⚠️ MANTER: Dataset atual é superior")
        print(f"   📝 Motivo: Mais histórico ou indicadores")
    else:
        print(f"   🤔 EMPATE: Ambos datasets têm prós e contras")
        print(f"   📝 Considerar: Testar ambos para comparar performance")

    print(f"\n✅ AVALIAÇÃO COMPLETA FINALIZADA!")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_silus_dataset()