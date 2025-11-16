#!/usr/bin/env python3
import pickle
import numpy as np

print("🔍 VERIFICANDO ENHANCED NORMALIZER...")

try:
    # Carregar o arquivo
    with open('enhanced_normalizer_final.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"📊 Tipo: {type(data)}")
    
    if hasattr(data, 'obs_rms'):
        print(f"📊 Obs RMS mean shape: {data.obs_rms.mean.shape}")
        print(f"📊 Obs RMS var shape: {data.obs_rms.var.shape}")
        print(f"📊 Obs count: {data.obs_rms.count}")
        print(f"📊 Ret count: {data.ret_rms.count}")
        
        # Verificar se tem 1320 observações
        if data.obs_rms.mean.shape[0] == 1320:
            print("✅ CORRETO: Enhanced normalizer tem 1320 observações!")
        else:
            print(f"❌ ERRADO: Enhanced normalizer tem {data.obs_rms.mean.shape[0]} observações, deveria ter 1320")
            
    else:
        print("❌ Arquivo não tem obs_rms")
        
except Exception as e:
    print(f"❌ Erro: {e}") 