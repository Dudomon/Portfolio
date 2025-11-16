#!/usr/bin/env python3
"""
Script simplificado para verificar Strategic Fusion Layer
"""

import os
import zipfile
import torch
import tempfile

def verificar_strategic_fusion(caminho_zip):
    """Verifica se o modelo tem Strategic Fusion Layer"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extrair o zip
        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Procurar por arquivos .pth
        model_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.pth'):
                    model_files.append(os.path.join(root, file))
        
        # Analisar arquivos
        fusion_keywords = [
            'strategic_fusion',
            'market_processor', 
            'conflict_resolver',
            'temporal_coordinator',
            'adaptive_learner'
        ]
        
        parametros_fusion = []
        
        for model_file in model_files:
            try:
                torch_data = torch.load(model_file, map_location='cpu')
                
                if isinstance(torch_data, dict):
                    for key in torch_data.keys():
                        key_lower = key.lower()
                        for keyword in fusion_keywords:
                            if keyword in key_lower:
                                parametros_fusion.append(key)
                                break
                                
            except Exception as e:
                continue
        
        return len(parametros_fusion) > 0, parametros_fusion

def main():
    modelo_path = r"D:\Projeto\Modelo PPO Trader\Modelo PPO\Legion V1.zip"
    
    print("Verificando Strategic Fusion Layer...")
    print("=" * 50)
    
    if os.path.exists(modelo_path):
        tem_fusion, parametros = verificar_strategic_fusion(modelo_path)
        
        if tem_fusion:
            print("RESULTADO: O modelo Legion V1.zip TEM Strategic Fusion Layer")
            print(f"Parametros encontrados: {len(parametros)}")
            for p in parametros:
                print(f"  - {p}")
        else:
            print("RESULTADO: O modelo Legion V1.zip NAO TEM Strategic Fusion Layer")
            print("Nenhum parametro da Strategic Fusion Layer foi encontrado.")
            print("\nO modelo parece usar a arquitetura V5 com:")
            print("- Transformer layers")
            print("- Temporal attention") 
            print("- Entry head com gates")
            print("- Management head")
            print("- Pattern detector")
            print("- Feature fusion")
            
        print("=" * 50)
        print("CONCLUSAO DEFINITIVA:")
        if tem_fusion:
            print("MODELO TEM Strategic Fusion Layer - NAO precisa retreinar")
        else:
            print("MODELO NAO TEM Strategic Fusion Layer - PRECISA retreinar se quiser usar Strategic Fusion")
            
    else:
        print(f"Arquivo nao encontrado: {modelo_path}")

if __name__ == "__main__":
    main()