#!/usr/bin/env python3
"""
🔍 Extrair informações do modelo V11 usando SB3
"""

import sys
sys.path.append("D:/Projeto")

from sb3_contrib import RecurrentPPO
import os

def extract_model_info(model_path):
    """Extrair informações do modelo sem carregar com policy_kwargs"""
    print(f"🔍 Extraindo info do modelo: {os.path.basename(model_path)}")
    
    try:
        # Carregar sem policy_kwargs específicos
        model = RecurrentPPO.load(model_path, device='cpu')
        
        print("✅ Modelo carregado com sucesso!")
        print(f"📊 Policy class: {type(model.policy).__name__}")
        print(f"📊 Observation space: {model.observation_space}")
        print(f"📊 Action space: {model.action_space}")
        
        # Tentar acessar atributos da policy
        if hasattr(model.policy, '__dict__'):
            print("\n🎯 Atributos da policy:")
            for key, value in model.policy.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    print(f"  {key}: {type(value).__name__}")
        
        # Verificar features extractor
        if hasattr(model.policy, 'features_extractor'):
            fe = model.policy.features_extractor
            print(f"\n🔧 Features Extractor: {type(fe).__name__}")
            if hasattr(fe, '__dict__'):
                for key, value in fe.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    model_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_phase4integration_7500000_steps_20250822_163304.zip"
    extract_model_info(model_path)