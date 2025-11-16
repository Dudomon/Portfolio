#!/usr/bin/env python3
"""
🔍 Inspecionar kwargs salvos no modelo V11
"""

import zipfile
import tempfile
import pickle
import os

def inspect_model_kwargs(model_path):
    """Extrair e mostrar os kwargs salvos no modelo"""
    print(f"🔍 Inspecionando modelo: {os.path.basename(model_path)}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extrair ZIP
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Procurar arquivo 'data'
            data_file = os.path.join(temp_dir, 'data')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                print("📋 Conteúdo do arquivo 'data':")
                for key, value in data.items():
                    if key == 'policy_kwargs':
                        print(f"  🎯 {key}:")
                        for pk_key, pk_value in value.items():
                            print(f"    '{pk_key}': {repr(pk_value)},")
                    else:
                        print(f"  {key}: {type(value).__name__}")
                
                return data.get('policy_kwargs', {})
            else:
                print("❌ Arquivo 'data' não encontrado")
                return {}
                
    except Exception as e:
        print(f"❌ Erro: {e}")
        return {}

if __name__ == "__main__":
    model_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_phase4integration_7500000_steps_20250822_163304.zip"
    kwargs = inspect_model_kwargs(model_path)
    
    if kwargs:
        print("\n🎯 Para usar estes kwargs no código:")
        print("def get_exact_model_kwargs():")
        print("    return {")
        for key, value in kwargs.items():
            print(f"        '{key}': {repr(value)},")
        print("    }")