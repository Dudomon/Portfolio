#!/usr/bin/env python3
"""
🔍 ANALISADOR DE OBSERVATION SPACE - Modelos Stable-Baselines3
Extrai informações detalhadas sobre o observation space esperado por um modelo
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None
import torch
import gym
from gym.spaces import Box

def analyze_model_observation_space(model_path):
    """
    Analisa o observation space esperado por um modelo treinado
    
    Args:
        model_path (str): Caminho para o arquivo .zip do modelo
        
    Returns:
        dict: Informações detalhadas sobre o observation space
    """
    try:
        print(f"🔍 Analisando modelo: {os.path.basename(model_path)}")
        print("=" * 80)
        
        # Tentar carregar modelo como RecurrentPPO primeiro (mais comum no projeto)
        try:
            if RecurrentPPO is not None:
                model = RecurrentPPO.load(model_path, device='cpu')
                model_type = "RecurrentPPO"
            else:
                raise ImportError("RecurrentPPO não disponível")
        except:
            try:
                model = PPO.load(model_path, device='cpu')
                model_type = "PPO"
            except Exception as e:
                print(f"❌ Erro ao carregar modelo: {e}")
                return None
        
        print(f"✅ Modelo carregado com sucesso: {model_type}")
        print()
        
        # Extrair informações do observation space
        obs_space = model.observation_space
        
        # Informações básicas
        info = {
            'model_type': model_type,
            'model_path': model_path,
            'observation_space_type': type(obs_space).__name__,
            'observation_space_shape': obs_space.shape,
            'observation_space_dtype': obs_space.dtype,
            'total_features': obs_space.shape[0] if hasattr(obs_space, 'shape') else None,
            'low_bound': obs_space.low if hasattr(obs_space, 'low') else None,
            'high_bound': obs_space.high if hasattr(obs_space, 'high') else None
        }
        
        print("📊 OBSERVATION SPACE ANALYSIS:")
        print(f"   Tipo: {info['observation_space_type']}")
        print(f"   Shape: {info['observation_space_shape']}")
        print(f"   Dtype: {info['observation_space_dtype']}")
        print(f"   Total Features: {info['total_features']}")
        
        if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
            print(f"   Bounds: [{obs_space.low[0]:.2f}, {obs_space.high[0]:.2f}]")
        
        print()
        
        # Tentar extrair informações do policy network
        try:
            policy = model.policy
            print("🧠 POLICY NETWORK ANALYSIS:")
            print(f"   Policy Type: {type(policy).__name__}")
            
            # Analisar features extractor se existir
            if hasattr(policy, 'features_extractor'):
                extractor = policy.features_extractor
                print(f"   Features Extractor: {type(extractor).__name__}")
                
                # Informações específicas do extractor
                if hasattr(extractor, 'features_dim'):
                    print(f"   Features Dimension: {extractor.features_dim}")
                
                # Verificar se tem informações sobre observation
                if hasattr(extractor, 'observation_space'):
                    ext_obs = extractor.observation_space
                    print(f"   Extractor Obs Space: {ext_obs.shape}")
                
                # Tentar obter parâmetros específicos do trading
                trading_params = {}
                for attr in ['max_positions', 'window_size', 'features_per_bar', 'seq_len']:
                    if hasattr(extractor, attr):
                        trading_params[attr] = getattr(extractor, attr)
                        print(f"   {attr}: {trading_params[attr]}")
                
                info['trading_params'] = trading_params
            
            print()
            
            # Analisar actor e critic networks
            if hasattr(policy, 'mlp_extractor'):
                print("🎭 ACTOR-CRITIC NETWORKS:")
                mlp = policy.mlp_extractor
                
                # Policy (Actor) network
                if hasattr(mlp, 'policy_net'):
                    policy_layers = []
                    for layer in mlp.policy_net:
                        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                            policy_layers.append(f"{layer.in_features}→{layer.out_features}")
                    print(f"   Actor Layers: {' → '.join(policy_layers)}")
                
                # Value (Critic) network  
                if hasattr(mlp, 'value_net'):
                    value_layers = []
                    for layer in mlp.value_net:
                        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                            value_layers.append(f"{layer.in_features}→{layer.out_features}")
                    print(f"   Critic Layers: {' → '.join(value_layers)}")
            
        except Exception as e:
            print(f"⚠️ Não foi possível analisar policy network: {e}")
        
        print()
        
        # Tentar fazer um forward pass de teste para verificar compatibilidade
        try:
            print("🧪 FORWARD PASS TEST:")
            # Criar observação de teste
            test_obs = obs_space.sample()
            print(f"   Test observation shape: {test_obs.shape}")
            
            # Fazer predição
            if model_type == "RecurrentPPO" and RecurrentPPO is not None:
                # RecurrentPPO precisa de lstm states
                lstm_states = None
                episode_starts = np.array([True])
                action, lstm_states = model.predict(test_obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            else:
                action, _ = model.predict(test_obs, deterministic=True)
            
            print(f"   ✅ Forward pass successful!")
            print(f"   Action shape: {action.shape}")
            print(f"   Action value: {action}")
            
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
        
        print()
        print("🎯 ANÁLISE COMPLETA!")
        print("=" * 80)
        
        return info
        
    except Exception as e:
        print(f"❌ Erro geral na análise: {e}")
        return None

def analyze_trading_model_features(obs_shape):
    """
    Tenta deduzir a estrutura de features baseada no shape da observação
    """
    total_features = obs_shape[0]
    
    print(f"🔬 DEDUÇÃO DA ESTRUTURA DE FEATURES ({total_features}D):")
    
    # Padrões conhecidos do projeto
    known_patterns = {
        450: {
            'description': 'V10Pure/SILUS format',
            'structure': '10 barras × 45 features',
            'breakdown': {
                'market_data': 16,
                'positions': 18,  # 2 posições × 9 features
                'intelligent': 11
            }
        },
        900: {
            'description': 'V7 Daytrader format', 
            'structure': '20 barras × 45 features',
            'breakdown': {
                'market_data': '16 per bar',
                'positions': '18 total (2 pos × 9)',
                'intelligent': '11 total'
            }
        }
    }
    
    if total_features in known_patterns:
        pattern = known_patterns[total_features]
        print(f"   ✅ Padrão conhecido: {pattern['description']}")
        print(f"   📏 Estrutura: {pattern['structure']}")
        
        for component, size in pattern['breakdown'].items():
            print(f"   - {component}: {size}")
    else:
        print(f"   ❓ Padrão desconhecido ({total_features}D)")
        
        # Tentar deduzir divisões comuns
        common_bars = [10, 15, 20, 25, 30]
        for bars in common_bars:
            if total_features % bars == 0:
                features_per_bar = total_features // bars
                print(f"   💡 Possível: {bars} barras × {features_per_bar} features")

def main():
    """Função principal"""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Usar Legion V1 como padrão
        model_path = "D:/Projeto/Modelo PPO Trader/Modelo daytrade/Legion V1.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    # Analisar o modelo
    info = analyze_model_observation_space(model_path)
    
    if info and info['total_features']:
        analyze_trading_model_features(info['observation_space_shape'])

if __name__ == "__main__":
    main()