#!/usr/bin/env python3
"""
Script para verificar se o modelo Legion V1.zip foi treinado com Strategic Fusion Layer
"""

import os
import zipfile
import torch
import tempfile
import shutil
from pathlib import Path
import pickle

def extrair_e_analisar_modelo(caminho_zip):
    """
    Extrai e analisa o modelo para verificar a presença da Strategic Fusion Layer
    """
    print(f"Analisando modelo: {caminho_zip}")
    
    # Criar diretório temporário
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extraindo para: {temp_dir}")
        
        # Extrair o zip
        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Listar arquivos extraídos
        print("\nArquivos extraídos:")
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"  {file_path}")
        
        # Procurar por arquivos do modelo
        model_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.pkl', '.zip', '.pt', '.pth')):
                    model_files.append(os.path.join(root, file))
        
        print(f"\nArquivos de modelo encontrados: {len(model_files)}")
        
        strategic_fusion_encontrado = False
        parametros_fusion = []
        
        # Analisar cada arquivo de modelo
        for model_file in model_files:
            print(f"\nAnalisando: {model_file}")
            
            try:
                # Tentar carregar como arquivo pickle (stable-baselines3)
                with open(model_file, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        print(f"Arquivo pickle carregado com sucesso")
                        
                        # Analisar a estrutura do dados
                        if hasattr(data, 'policy'):
                            print("Modelo stable-baselines3 detectado")
                            policy = data.policy
                            
                            # Verificar state_dict da policy
                            if hasattr(policy, 'state_dict'):
                                state_dict = policy.state_dict()
                                print(f"State dict encontrado com {len(state_dict)} parâmetros")
                                
                                # Procurar por parâmetros da Strategic Fusion Layer
                                for param_name in state_dict.keys():
                                    print(f"  Parâmetro: {param_name}")
                                    
                                    # Verificar nomes relacionados à Strategic Fusion Layer
                                    fusion_keywords = [
                                        'strategic_fusion',
                                        'market_processor', 
                                        'conflict_resolver',
                                        'temporal_coordinator',
                                        'adaptive_learner',
                                        'fusion_layer',
                                        'strategic',
                                        'market_state',
                                        'conflict',
                                        'temporal'
                                    ]
                                    
                                    for keyword in fusion_keywords:
                                        if keyword in param_name.lower():
                                            strategic_fusion_encontrado = True
                                            parametros_fusion.append(param_name)
                                            break
                        
                        # Verificar se há outros atributos relacionados
                        if isinstance(data, dict):
                            print("Dados em formato dicionário")
                            for key in data.keys():
                                print(f"  Chave: {key}")
                                if any(keyword in key.lower() for keyword in ['strategic', 'fusion', 'market_processor']):
                                    strategic_fusion_encontrado = True
                                    parametros_fusion.append(key)
                                    
                    except Exception as e:
                        print(f"Erro ao carregar como pickle: {e}")
                        
                        # Tentar carregar como modelo PyTorch
                        try:
                            torch_data = torch.load(model_file, map_location='cpu')
                            print("Modelo PyTorch carregado com sucesso")
                            
                            if isinstance(torch_data, dict):
                                for key in torch_data.keys():
                                    print(f"  Chave PyTorch: {key}")
                                    if any(keyword in key.lower() for keyword in ['strategic', 'fusion', 'market_processor']):
                                        strategic_fusion_encontrado = True
                                        parametros_fusion.append(key)
                                        
                        except Exception as torch_e:
                            print(f"Erro ao carregar como PyTorch: {torch_e}")
                            
            except Exception as e:
                print(f"Erro geral ao analisar {model_file}: {e}")
        
        return strategic_fusion_encontrado, parametros_fusion

def main():
    """Função principal"""
    modelo_path = r"D:\Projeto\Modelo PPO Trader\Modelo PPO\Legion V1.zip"
    backup_path = r"D:\Projeto\Modelo PPO Trader\Modelo PPO\Legion V1-v6-sem fusio.zip"
    
    print("="*60)
    print("VERIFICAÇÃO DA STRATEGIC FUSION LAYER")
    print("="*60)
    
    # Verificar modelo principal
    if os.path.exists(modelo_path):
        fusion_encontrado, parametros = extrair_e_analisar_modelo(modelo_path)
        
        print("\n" + "="*60)
        print("RESULTADO DA ANÁLISE")
        print("="*60)
        
        if fusion_encontrado:
            print("✅ O modelo Legion V1.zip TEM Strategic Fusion Layer")
            print(f"\nParâmetros relacionados encontrados ({len(parametros)}):")
            for param in parametros:
                print(f"  - {param}")
        else:
            print("❌ O modelo Legion V1.zip NÃO TEM Strategic Fusion Layer")
            print("\nNenhum parâmetro relacionado à Strategic Fusion Layer foi encontrado.")
            
        print("\n" + "="*60)
        
        # Verificar modelo backup se existir
        if os.path.exists(backup_path):
            print("\nVerificando modelo backup para comparação...")
            backup_fusion, backup_params = extrair_e_analisar_modelo(backup_path)
            
            print(f"\nModelo backup TEM Strategic Fusion Layer: {backup_fusion}")
            if backup_params:
                print(f"Parâmetros backup ({len(backup_params)}):")
                for param in backup_params:
                    print(f"  - {param}")
    else:
        print(f"❌ Arquivo não encontrado: {modelo_path}")

if __name__ == "__main__":
    main()