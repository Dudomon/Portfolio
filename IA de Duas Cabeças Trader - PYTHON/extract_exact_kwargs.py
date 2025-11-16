#!/usr/bin/env python3
"""
🔍 Extrair kwargs exatos do modelo V11 para usar no carregamento
"""

import sys
sys.path.append("D:/Projeto")

from sb3_contrib import RecurrentPPO
import os
import torch

def extract_exact_kwargs(model_path):
    """Extrair kwargs exatos do modelo carregado"""
    print(f"🔍 Extraindo kwargs exatos: {os.path.basename(model_path)}")
    
    try:
        # Carregar sem policy_kwargs específicos
        model = RecurrentPPO.load(model_path, device='cpu')
        
        print("✅ Modelo carregado!")
        
        # Acessar atributos da policy para reconstruir kwargs
        policy = model.policy
        
        # Extrair kwargs V8/V11
        kwargs = {
            'v8_lstm_hidden': getattr(policy, 'v8_lstm_hidden', 256),
            'v8_features_dim': getattr(policy, 'v8_features_dim', 256),
            'v8_context_dim': getattr(policy, 'v8_context_dim', 64),
            'v8_memory_size': getattr(policy, 'v8_memory_size', 512),
            'critic_learning_rate': getattr(policy, 'critic_learning_rate', 1e-5),
        }
        
        # Kwargs padrão de SB3
        if hasattr(policy, 'features_extractor'):
            fe = policy.features_extractor
            kwargs['features_extractor_class'] = type(fe)
            kwargs['features_extractor_kwargs'] = getattr(policy, 'features_extractor_kwargs', {'features_dim': 256})
        
        # Outros atributos importantes
        kwargs.update({
            'net_arch': getattr(policy, 'net_arch', [256, 128]),
            'lstm_hidden_size': getattr(policy, 'lstm_kwargs', {}).get('hidden_size', 256),
            'n_lstm_layers': getattr(policy, 'lstm_kwargs', {}).get('num_layers', 1),
            'ortho_init': getattr(policy, 'ortho_init', True),
            'log_std_init': getattr(policy, 'log_std_init', -0.5),
            'full_std': True,  # Padrão
            'use_expln': False,  # Padrão
            'squash_output': False,  # Padrão
        })
        
        # Detectar activation function
        if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'policy_net'):
            for layer in policy.mlp_extractor.policy_net:
                if isinstance(layer, torch.nn.LeakyReLU):
                    kwargs['activation_fn'] = torch.nn.LeakyReLU
                    break
                elif isinstance(layer, torch.nn.ReLU):
                    kwargs['activation_fn'] = torch.nn.ReLU
                    break
                elif isinstance(layer, torch.nn.Tanh):
                    kwargs['activation_fn'] = torch.nn.Tanh
                    break
        
        if 'activation_fn' not in kwargs:
            kwargs['activation_fn'] = torch.nn.LeakyReLU  # Padrão
        
        print("\n🎯 Kwargs extraídos:")
        for key, value in kwargs.items():
            print(f"  '{key}': {repr(value)},")
        
        print(f"\n📋 Função para usar no código:")
        print("def get_exact_v11_kwargs():")
        print("    from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor")
        print("    import torch")
        print("    return {")
        for key, value in kwargs.items():
            if key == 'features_extractor_class':
                print(f"        '{key}': TradingTransformerFeatureExtractor,")
            elif key == 'activation_fn':
                print(f"        '{key}': torch.nn.LeakyReLU,")
            else:
                print(f"        '{key}': {repr(value)},")
        print("    }")
        
        return kwargs
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return {}

if __name__ == "__main__":
    model_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/SILUS_phase4integration_7500000_steps_20250822_163304.zip"
    extract_exact_kwargs(model_path)