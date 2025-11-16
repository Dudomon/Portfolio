#!/usr/bin/env python3
import os
import sys
import zipfile
import tempfile
import pickle
from sb3_contrib import RecurrentPPO

# Adicionar path para encontrar TwoHeadV7Simple
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from trading_framework.policies.two_head_v7_simple import TwoHeadV7Simple

def test_v7_loading():
    print("TESTE SIMPLES V7 LOADING...")
    
    # 1. Testar ZIP
    zip_path = os.path.join(os.getcwd(), "modelo daytrade", "Legion daytrade.zip")
    print(f"ZIP Path: {zip_path}")
    print(f"ZIP Exists: {os.path.exists(zip_path)}")
    
    if not os.path.exists(zip_path):
        print("ZIP nao encontrado!")
        return False
    
    # 2. Extrair ZIP
    temp_dir = os.path.join(tempfile.gettempdir(), "test_v7_extract")
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Extraindo para: {temp_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        print("Arquivos extraídos:")
        for file in zip_ref.namelist():
            print(f"  - {file}")
    
    # 3. Carregar modelo
    print("Carregando modelo...")
    try:
        model = RecurrentPPO.load(temp_dir)
        print(f"✅ Modelo carregado: {type(model).__name__}")
        print(f"Policy: {type(model.policy).__name__}")
        
        # Verificar se é V7
        if hasattr(model.policy, 'features_extractor'):
            print(f"Features Extractor: {type(model.policy.features_extractor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_v7_loading()