#!/usr/bin/env python3
"""
🧪 Testar Legion V1.zip para verificar sua policy
"""

import sys
sys.path.append("D:/Projeto")

from sb3_contrib import RecurrentPPO
import os

def test_legion_v1():
    """Testar Legion V1.zip para verificar qual policy usa"""
    
    legion_paths = [
        "D:/Projeto/Modelo PPO Trader/Modelo daytrade/Legion V1.zip",
        "D:/Projeto/Modelo PPO Trader/Modelo daytrade/11dim/Legion V1.zip"
    ]
    
    for legion_path in legion_paths:
        if os.path.exists(legion_path):
            print(f"🔍 Testando: {legion_path}")
            
            try:
                # Carregar sem policy_kwargs específicos
                model = RecurrentPPO.load(legion_path, device='cpu')
                
                print(f"✅ Modelo carregado!")
                print(f"📊 Policy class: {type(model.policy).__name__}")
                print(f"📊 Observation space: {model.observation_space.shape}")
                print(f"📊 Action space: {model.action_space.shape}")
                
                # Verificar atributos específicos
                policy = model.policy
                
                # Verificar se tem atributos V8
                has_v8_attrs = hasattr(policy, 'v8_lstm_hidden')
                has_v7_attrs = hasattr(policy, 'v7_shared_lstm_hidden')
                
                print(f"🔍 Atributos V8: {has_v8_attrs}")
                print(f"🔍 Atributos V7: {has_v7_attrs}")
                
                if has_v8_attrs:
                    print("✅ Este é um modelo V8 Elegance!")
                    print(f"  - v8_lstm_hidden: {getattr(policy, 'v8_lstm_hidden', 'N/A')}")
                    print(f"  - v8_features_dim: {getattr(policy, 'v8_features_dim', 'N/A')}")
                elif has_v7_attrs:
                    print("✅ Este é um modelo V7 Intuition!")
                else:
                    print("⚠️ Tipo de modelo desconhecido")
                
                # Verificar features extractor
                if hasattr(policy, 'features_extractor'):
                    fe = policy.features_extractor
                    print(f"🔧 Features Extractor: {type(fe).__name__}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Erro ao carregar {legion_path}: {e}")
                print("-" * 50)
        else:
            print(f"❌ Arquivo não encontrado: {legion_path}")

if __name__ == "__main__":
    test_legion_v1()