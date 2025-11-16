#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validar que dataset corrigido resolve o problema de zeros extremos
"""

import pandas as pd
import numpy as np
import pickle
import os

def validar_dataset_corrigido():
    """Validar dataset corrigido"""
    
    # Carregar dataset corrigido
    dataset_path = "data_cache/GC=F_YAHOO_DAILY_CACHE_20250711_041924.pkl"
    backup_path = "data_cache/GC=F_YAHOO_DAILY_CACHE_20250711_041924_BACKUP.pkl"
    
    print("VALIDANDO DATASET CORRIGIDO...")
    
    try:
        # Carregar dataset atual (corrigido)
        with open(dataset_path, 'rb') as f:
            df_corrigido = pickle.load(f)
        
        print(f"OK Dataset corrigido carregado: {len(df_corrigido)} linhas, {len(df_corrigido.columns)} colunas")
        
        # Analisar zeros extremos
        print("\nANALISE DE ZEROS EXTREMOS:")
        total_zeros = 0
        problematic_columns = []
        
        for col in df_corrigido.columns:
            if df_corrigido[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                zeros_count = np.sum(np.abs(df_corrigido[col]) < 1e-8)
                zeros_percentage = (zeros_count / len(df_corrigido)) * 100
                
                if zeros_percentage > 0:
                    print(f"  {col}: {zeros_count}/{len(df_corrigido)} ({zeros_percentage:.1f}%) zeros extremos")
                    
                if zeros_percentage > 10:  # >10% zeros
                    problematic_columns.append(col)
                    
                total_zeros += zeros_count
        
        print(f"\nRESULTADO DA VALIDACAO:")
        print(f"   Total de zeros extremos: {total_zeros}")
        print(f"   Colunas com >10% zeros: {len(problematic_columns)}")
        if problematic_columns:
            print(f"   Colunas problematicas: {problematic_columns}")
        
        # Comparar com backup se existir
        if os.path.exists(backup_path):
            print(f"\nCOMPARANDO COM BACKUP ORIGINAL:")
            
            with open(backup_path, 'rb') as f:
                df_original = pickle.load(f)
            
            original_zeros = 0
            for col in df_original.columns:
                if df_original[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    zeros_count = np.sum(np.abs(df_original[col]) < 1e-8)
                    original_zeros += zeros_count
            
            print(f"   Zeros extremos ANTES: {original_zeros}")
            print(f"   Zeros extremos DEPOIS: {total_zeros}")
            print(f"   REDUCAO: {original_zeros - total_zeros} zeros extremos removidos")
            
            if total_zeros < original_zeros * 0.1:  # Redução de >90%
                print("   STATUS: CORRECAO MUITO BEM SUCEDIDA!")
                return True
            elif total_zeros < original_zeros * 0.5:  # Redução de >50%
                print("   STATUS: CORRECAO PARCIALMENTE BEM SUCEDIDA!")
                return True
            else:
                print("   STATUS: CORRECAO INSUFICIENTE!")
                return False
        else:
            print("   Backup nao encontrado para comparacao")
            if total_zeros < 10000:  # Critério absoluto
                print("   STATUS: DATASET PARECE OK!")
                return True
            else:
                print("   STATUS: MUITOS ZEROS AINDA PRESENTES!")
                return False
        
    except Exception as e:
        print(f"ERRO ao validar dataset: {e}")
        return False

def simular_processamento():
    """Simular processamento como seria feito no PPO"""
    
    print("\nSIMULANDO PROCESSAMENTO COMO NO PPO...")
    
    try:
        # Carregar dataset
        with open("data_cache/GC=F_YAHOO_DAILY_CACHE_20250711_041924.pkl", 'rb') as f:
            df = pickle.load(f)
        
        # Simular feature columns típicas do PPO
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 'bb_position',
            'trend_strength', 'atr_14', 'stoch_k', 'volume_ratio', 'var_99'
        ]
        
        # Criar processed_data como no PPO
        processed_data = df[feature_columns].values.astype(np.float32)
        
        # Simular window_size típico
        window_size = 45
        current_step = 1000
        
        # Simular extração de dados como no PPO
        obs_market = processed_data[current_step - window_size:current_step]
        
        # Analisar zeros extremos
        market_zeros = np.sum(np.abs(obs_market.flatten()) < 1e-8)
        total_elements = len(obs_market.flatten())
        zeros_percentage = (market_zeros / total_elements) * 100
        
        print(f"   Market data window: {obs_market.shape}")
        print(f"   Zeros extremos: {market_zeros}/{total_elements} ({zeros_percentage:.1f}%)")
        
        if zeros_percentage < 5:  # <5% zeros
            print("   STATUS: EXCELENTE - Poucos zeros extremos!")
            return True
        elif zeros_percentage < 20:  # <20% zeros
            print("   STATUS: BOM - Zeros extremos aceitaveis!")
            return True
        else:
            print("   STATUS: RUIM - Muitos zeros extremos ainda!")
            return False
            
    except Exception as e:
        print(f"ERRO na simulacao: {e}")
        return False

def main():
    """Função principal"""
    
    print("VALIDACAO DO DATASET CORRIGIDO")
    print("=" * 50)
    
    # 1. Validar dataset corrigido
    dataset_ok = validar_dataset_corrigido()
    
    # 2. Simular processamento PPO
    processing_ok = simular_processamento()
    
    # 3. Resultado final
    print(f"\nRESULTADO FINAL:")
    print(f"   Dataset corrigido: {'OK' if dataset_ok else 'PROBLEMAS'}")
    print(f"   Processamento PPO: {'OK' if processing_ok else 'PROBLEMAS'}")
    
    if dataset_ok and processing_ok:
        print("   CONCLUSAO: DATASET CORRIGIDO RESOLVE O PROBLEMA!")
        print("   RECOMENDACAO: Executar treinamento PPO novamente")
    else:
        print("   CONCLUSAO: DATASET AINDA TEM PROBLEMAS!")
        print("   RECOMENDACAO: Investigar mais profundamente")

if __name__ == "__main__":
    main()