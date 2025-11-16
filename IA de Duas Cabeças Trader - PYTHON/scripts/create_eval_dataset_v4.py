#!/usr/bin/env python3
"""
🚀 CRIADOR DE DATASET OTIMIZADO PARA AVALIAÇÃO - V4 BASE
======================================================

OBJETIVO: Criar dataset pequeno e rápido baseado no V4 existente
- Usar dataset V4 como base (dados confiáveis)
- Extrair últimas 50k barras (vs 216k atuais)
- Features pré-computadas 
- Formato idêntico ao dataset V4
- ~5x mais rápido para testes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def load_v4_dataset():
    """
    📊 Carregar dataset V4 original
    """
    print("📊 Carregando dataset V4 original...")
    
    dataset_path = 'D:/Projeto/data/GC=F_REALISTIC_V4_20250911_235945.csv'
    
    try:
        data = pd.read_csv(dataset_path)
        print(f"✅ Dataset V4 carregado: {len(data)} barras")
        print(f"   Período: {data['time'].min()} a {data['time'].max()}")
        print(f"   Colunas: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Erro ao carregar dataset V4: {e}")
        return None

def optimize_v4_for_eval(data, target_size=50000):
    """
    🎯 Otimizar dataset V4 para avaliação rápida
    """
    print(f"🎯 Otimizando dataset V4: {len(data)} → {target_size} barras")
    
    # Pegar dados mais recentes (final do dataset)
    optimized = data.tail(target_size).reset_index(drop=True)
    
    # Converter time para datetime se string
    if 'time' in optimized.columns:
        if optimized['time'].dtype == 'object':
            optimized['time'] = pd.to_datetime(optimized['time'])
    
    print(f"✅ Dataset otimizado: {len(optimized)} barras")
    print(f"   Período otimizado: {optimized['time'].min()} a {optimized['time'].max()}")
    
    return optimized

def add_missing_1m_columns(data):
    """
    🔧 Garantir que todas as colunas _1m estejam presentes
    """
    print("🔧 Verificando colunas _1m...")
    
    # Mapeamento para garantir formato 1m
    required_mappings = {
        'open': 'open_1m',
        'high': 'high_1m', 
        'low': 'low_1m',
        'close': 'close_1m',
        'tick_volume': 'tick_volume_1m'
    }
    
    # Aplicar mapeamentos se necessário
    for old_col, new_col in required_mappings.items():
        if old_col in data.columns and new_col not in data.columns:
            data[new_col] = data[old_col]
            print(f"  ✅ Criado: {old_col} → {new_col}")
    
    # Verificar features técnicas mínimas _1m
    required_features_1m = [
        'returns_1m', 'volatility_20_1m', 'sma_20_1m', 'sma_50_1m', 
        'rsi_14_1m', 'stoch_k_1m', 'bb_position_1m', 'trend_strength_1m'
    ]
    
    missing_features = [f for f in required_features_1m if f not in data.columns]
    
    if missing_features:
        print(f"  ⚠️ Features ausentes: {missing_features}")
        
        # Criar features básicas se ausentes
        if 'returns_1m' in missing_features and 'close_1m' in data.columns:
            data['returns_1m'] = data['close_1m'].pct_change().fillna(0)
            print("  ✅ Criado: returns_1m")
        
        if 'volatility_20_1m' in missing_features and 'close_1m' in data.columns:
            data['volatility_20_1m'] = data['close_1m'].rolling(20).std().fillna(0.001)
            print("  ✅ Criado: volatility_20_1m")
            
        if 'sma_20_1m' in missing_features and 'close_1m' in data.columns:
            data['sma_20_1m'] = data['close_1m'].rolling(20).mean().fillna(data['close_1m'])
            print("  ✅ Criado: sma_20_1m")
        
        # Features com valores padrão para velocidade
        for feature in missing_features:
            if feature not in data.columns:
                if 'rsi' in feature:
                    data[feature] = 50.0
                elif 'stoch' in feature:
                    data[feature] = 50.0  
                elif 'bb_position' in feature:
                    data[feature] = 0.5
                else:
                    data[feature] = 0.001
                print(f"  ✅ Criado com valor padrão: {feature}")
    
    return data

def validate_eval_dataset(data):
    """
    ✅ Validar dataset otimizado para compatibilidade
    """
    print("✅ Validando dataset otimizado...")
    
    # Verificar tamanho mínimo
    if len(data) < 20000:
        print(f"❌ Dataset muito pequeno: {len(data)} barras (mínimo 20k)")
        return False
    
    # Verificar colunas essenciais
    essential_columns = ['time', 'close_1m', 'open_1m', 'high_1m', 'low_1m']
    missing_essential = [col for col in essential_columns if col not in data.columns]
    
    if missing_essential:
        print(f"❌ Colunas essenciais ausentes: {missing_essential}")
        return False
    
    # Verificar dados válidos
    null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
    if null_percentage > 10:
        print(f"❌ Muitos valores nulos: {null_percentage:.1f}% (máximo 10%)")
        return False
    
    # Verificar sequência temporal
    if 'time' in data.columns:
        time_diffs = pd.to_datetime(data['time']).diff().dt.total_seconds()
        invalid_times = time_diffs[time_diffs <= 0].count()
        if invalid_times > len(data) * 0.01:  # Máximo 1% de problemas temporais
            print(f"❌ Sequência temporal inválida: {invalid_times} problemas")
            return False
    
    print(f"✅ Dataset válido para avaliação:")
    print(f"   📊 Barras: {len(data):,}")
    print(f"   📈 Colunas: {len(data.columns)}")
    print(f"   🔍 Valores nulos: {null_percentage:.2f}%")
    print(f"   ⏰ Período: {data['time'].min()} a {data['time'].max()}")
    
    return True

def save_optimized_dataset(data, filename=None):
    """
    💾 Salvar dataset otimizado
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GC=F_EVAL_OPTIMIZED_V4_{timestamp}.csv"
    
    filepath = f"D:/Projeto/data/{filename}"
    
    print(f"💾 Salvando dataset otimizado: {filename}")
    
    try:
        data.to_csv(filepath, index=False)
        
        file_size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"✅ Dataset salvo:")
        print(f"   📁 Arquivo: {filename}")
        print(f"   📊 Tamanho: {file_size_mb:.1f} MB")
        print(f"   📈 Barras: {len(data):,}")
        print(f"   🔍 Colunas: {len(data.columns)}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        return None

def calculate_optimization_stats(original_size, optimized_size):
    """
    📊 Calcular estatísticas de otimização
    """
    reduction = (1 - optimized_size / original_size) * 100
    speedup = original_size / optimized_size
    
    print("\n📊 ESTATÍSTICAS DE OTIMIZAÇÃO:")
    print(f"   📉 Redução de tamanho: {reduction:.1f}%")
    print(f"   ⚡ Speedup esperado: {speedup:.1f}x mais rápido")
    print(f"   💾 Dados originais: {original_size:,} barras")
    print(f"   🎯 Dados otimizados: {optimized_size:,} barras")

def main():
    """
    🚀 Função principal: Criar dataset de avaliação otimizado do V4
    """
    print("🚀 CRIANDO DATASET DE AVALIAÇÃO OTIMIZADO - V4 BASE")
    print("=" * 60)
    
    # 1. Carregar dataset V4
    data = load_v4_dataset()
    if data is None:
        return False
    
    original_size = len(data)
    
    # 2. Otimizar tamanho (pegar últimas 50k barras)
    target_size = 50000
    data = optimize_v4_for_eval(data, target_size=target_size)
    
    # 3. Garantir colunas _1m
    data = add_missing_1m_columns(data)
    
    # 4. Validar dataset
    if not validate_eval_dataset(data):
        print("❌ Dataset inválido após otimização")
        return False
    
    # 5. Salvar dataset otimizado
    filepath = save_optimized_dataset(data)
    if filepath is None:
        return False
    
    # 6. Mostrar estatísticas
    calculate_optimization_stats(original_size, len(data))
    
    print("\n" + "=" * 60)
    print("✅ DATASET DE AVALIAÇÃO OTIMIZADO CRIADO!")
    print(f"📁 Arquivo: {os.path.basename(filepath)}")
    print(f"📊 Dados: {len(data):,} barras")
    print(f"🕐 Período: {data['time'].min()} a {data['time'].max()}")
    print(f"⚡ Performance: ~{original_size/len(data):.1f}x mais rápido")
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Atualizar completo_1m_optimized.py para usar este dataset")
    print("2. Testar velocidade de avaliação")
    print("3. Validar métricas vs dataset V4 completo")
    print("4. Comparar consistência dos resultados")
    
    return True

if __name__ == "__main__":
    main()