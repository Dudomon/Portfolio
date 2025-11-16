#!/usr/bin/env python3
"""
🔧 FIX CRÍTICO: Inicialização da primeira camada do actor SAC
Resolver 62.1% zeros persistentes em actor.latent_pi.0.weight
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_and_fix_actor_layer(model, verbose=True):
    """
    Analisar e corrigir a primeira camada do actor SAC
    """
    if not hasattr(model, 'policy') or not hasattr(model.policy, 'actor'):
        if verbose:
            print("❌ Modelo não tem policy.actor")
        return False
    
    actor = model.policy.actor
    
    # Encontrar primeira camada
    first_layer = None
    first_layer_name = None
    
    for name, layer in actor.named_modules():
        if isinstance(layer, nn.Linear) and 'latent_pi.0' in name:
            first_layer = layer
            first_layer_name = name
            break
    
    if first_layer is None:
        # Tentar encontrar primeira camada Linear
        for name, layer in actor.named_modules():
            if isinstance(layer, nn.Linear):
                first_layer = layer
                first_layer_name = name
                break
    
    if first_layer is None:
        if verbose:
            print("❌ Primeira camada Linear não encontrada")
        return False
    
    # Analisar estado atual
    weight = first_layer.weight.data
    zeros_before = (weight == 0).float().mean().item() * 100
    
    if verbose:
        print(f"🔍 Analisando camada: {first_layer_name}")
        print(f"   Shape: {weight.shape}")
        print(f"   Zeros ANTES: {zeros_before:.1f}%")
        print(f"   Range: [{weight.min().item():.4f}, {weight.max().item():.4f}]")
        print(f"   Std: {weight.std().item():.4f}")
    
    # Se muitos zeros, aplicar nova inicialização
    if zeros_before > 50:
        if verbose:
            print(f"🚨 APLICANDO FIX: {zeros_before:.1f}% zeros é crítico!")
        
        # 🎯 APLICAR INICIALIZAÇÃO V11 (xavier_uniform_ + orthogonal)
        input_dim, output_dim = weight.shape[1], weight.shape[0]
        
        # V11 Style: Xavier Uniform (provado funcionar)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(first_layer.weight, gain=1.0)
            
            # V11 Style: Garantir distribuição saudável
            # Se ainda há zeros, aplicar noise mínimo
            zero_mask = (first_layer.weight.data == 0)
            if zero_mask.any():
                first_layer.weight.data[zero_mask] = torch.randn_like(
                    first_layer.weight.data[zero_mask]
                ) * 0.01
        
        # Reinicializar bias também se existir
        if first_layer.bias is not None:
            with torch.no_grad():
                first_layer.bias.data.zero_()
        
        # Verificar resultado
        zeros_after = (first_layer.weight.data == 0).float().mean().item() * 100
        
        if verbose:
            print(f"✅ FIX APLICADO:")
            print(f"   Zeros DEPOIS: {zeros_after:.1f}%")
            print(f"   Range NOVO: [{first_layer.weight.data.min().item():.4f}, {first_layer.weight.data.max().item():.4f}]")
            print(f"   Std NOVO: {first_layer.weight.data.std().item():.4f}")
        
        return True
    else:
        if verbose:
            print(f"✅ Camada OK: {zeros_before:.1f}% zeros aceitável")
        return False

def apply_fix_to_sac_model(model, verbose=True):
    """
    Aplicar fix de inicialização ao modelo SAC
    """
    if verbose:
        print("🔧 APLICANDO FIX DE INICIALIZAÇÃO SAC")
        print("=" * 50)
    
    fixed = analyze_and_fix_actor_layer(model, verbose=verbose)
    
    if fixed:
        # Resetar optimizers para aplicar nova inicialização
        if hasattr(model.policy, 'actor_optimizer'):
            # Recriar optimizer state para nova inicialização
            for param_group in model.policy.actor_optimizer.param_groups:
                for param in param_group['params']:
                    if param in model.policy.actor_optimizer.state:
                        del model.policy.actor_optimizer.state[param]
        
        if verbose:
            print("🔄 Optimizer state resetado para nova inicialização")
    
    return fixed

if __name__ == "__main__":
    print("🔧 Fix de Inicialização da Primeira Camada Actor SAC")
    print("   Para usar: import fix_actor_initialization; fix_actor_initialization.apply_fix_to_sac_model(model)")