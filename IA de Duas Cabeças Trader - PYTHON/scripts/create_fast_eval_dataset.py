#!/usr/bin/env python3
"""
🚀 CRIADOR DE DATASET RÁPIDO PARA TESTE
=====================================

OBJETIVO: Criar dataset FUNCIONAL e RÁPIDO baseado no V4
- Usar V4 como base (formato conhecido que funciona)
- Extrair últimas 50k barras SEM MODIFICAR COLUNAS
- Manter formato EXATO do V4 original
- ZERO alterações de formato para evitar erros
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    🚀 Criar dataset rápido IMEDIATAMENTE
    """
    print("🚀 CRIANDO DATASET RÁPIDO PARA TESTE IMEDIATO")
    print("=" * 50)
    
    # 1. Carregar V4 original
    print("📊 Carregando dataset V4 original...")
    dataset_path = 'D:/Projeto/data/GC=F_REALISTIC_V4_20250911_235945.csv'
    
    try:
        data = pd.read_csv(dataset_path)
        print(f"✅ V4 carregado: {len(data)} barras")
        
        # 2. Extrair últimas 50k barras SEM MODIFICAR NADA
        target_size = 50000
        if len(data) > target_size:
            data_fast = data.tail(target_size).reset_index(drop=True)
            print(f"✅ Extraídas últimas {target_size} barras")
        else:
            data_fast = data.copy()
            print(f"⚠️ Dataset menor que {target_size}, usando tudo: {len(data)} barras")
        
        # 3. Salvar SEM MODIFICAÇÕES
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GC=F_FAST_EVAL_{timestamp}.csv"
        filepath = f"D:/Projeto/data/{filename}"
        
        data_fast.to_csv(filepath, index=False)
        
        file_size_mb = os.path.getsize(filepath) / (1024*1024)
        original_size_mb = os.path.getsize(dataset_path) / (1024*1024)
        
        print(f"\n✅ DATASET RÁPIDO CRIADO:")
        print(f"   📁 Arquivo: {filename}")
        print(f"   📊 Barras: {len(data_fast):,}")
        print(f"   💾 Tamanho: {file_size_mb:.1f} MB (vs {original_size_mb:.1f} MB original)")
        print(f"   🔍 Colunas: {len(data_fast.columns)} (EXATAS do V4)")
        print(f"   ⚡ Redução: {(1-len(data_fast)/len(data))*100:.1f}%")
        print(f"   🚀 Speedup: {len(data)/len(data_fast):.1f}x mais rápido")
        
        # 4. Mostrar sample para validação
        print(f"\n📋 SAMPLE DO DATASET:")
        print(data_fast.head(2))
        print(f"\n📋 COLUNAS: {list(data_fast.columns)}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🎯 DATASET PRONTO PARA USO: {os.path.basename(result)}")
    else:
        print(f"\n❌ FALHA NA CRIAÇÃO DO DATASET")