#!/usr/bin/env python3
"""
🔍 DEBUG PROFUNDO - FONTE DOS ZEROS
Investigar a causa raiz dos 64.4% zeros no SAC
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import gc
import os

def analyze_zero_patterns():
    """
    Analisar padrões dos zeros para identificar fonte
    """
    print("🔍 ANÁLISE PROFUNDA DOS PADRÕES DE ZEROS")
    print("=" * 60)
    
    # Tentar carregar modelo diretamente do checkpoint mais recente
    model = None
    
    # Procurar checkpoint files
    checkpoint_files = []
    for filename in os.listdir('.'):
        if filename.endswith('.zip') and ('sac' in filename.lower() or 'model' in filename.lower() or 'v11' in filename.lower()):
            checkpoint_files.append(filename)
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_checkpoint = checkpoint_files[0]
        print(f"🔍 Tentando carregar: {latest_checkpoint}")
        
        try:
            from stable_baselines3 import SAC
            model = SAC.load(latest_checkpoint, verbose=0)
            print(f"✅ Modelo SAC carregado: {latest_checkpoint}")
        except Exception as e:
            print(f"❌ Erro ao carregar {latest_checkpoint}: {e}")
    
    # Fallback: procurar na memória
    if model is None:
        print("🔍 Procurando SAC na memória...")
        for obj in gc.get_objects():
            try:
                if hasattr(obj, 'policy') and hasattr(obj.policy, 'actor'):
                    if hasattr(obj.policy.actor, 'latent_pi'):
                        model = obj
                        print(f"✅ Modelo SAC encontrado na memória: {type(model)}")
                        break
            except:
                continue
    
    if model is None:
        print("❌ Nenhum modelo SAC encontrado")
        return False
    
    # Analisar primeira camada em detalhes
    first_layer = model.policy.actor.latent_pi[0]
    weight_data = first_layer.weight.data
    
    print(f"\n📊 ANÁLISE DETALHADA - PRIMEIRA CAMADA:")
    print(f"   Shape: {weight_data.shape}")
    print(f"   Device: {weight_data.device}")
    print(f"   Dtype: {weight_data.dtype}")
    
    # Estatísticas básicas
    weight_flat = weight_data.flatten()
    zeros_mask = (weight_flat == 0)
    non_zeros_mask = ~zeros_mask
    
    zeros_count = zeros_mask.sum().item()
    total_count = weight_flat.numel()
    zeros_pct = (zeros_count / total_count) * 100
    
    print(f"\n🔢 ESTATÍSTICAS ZEROS:")
    print(f"   Total weights: {total_count}")
    print(f"   Zeros: {zeros_count} ({zeros_pct:.1f}%)")
    print(f"   Non-zeros: {total_count - zeros_count}")
    
    if zeros_count > 0:
        print(f"\n🎯 ANÁLISE DOS ZEROS:")
        
        # Verificar distribuição dos zeros por neurônio
        zeros_per_neuron = (weight_data == 0).sum(dim=1)
        print(f"   Zeros por neurônio (primeiros 10): {zeros_per_neuron[:10].tolist()}")
        print(f"   Max zeros em um neurônio: {zeros_per_neuron.max().item()}/{weight_data.shape[1]}")
        print(f"   Min zeros em um neurônio: {zeros_per_neuron.min().item()}/{weight_data.shape[1]}")
        
        # Neurônios completamente zeros
        dead_neurons = (zeros_per_neuron == weight_data.shape[1]).sum().item()
        print(f"   Neurônios completamente mortos: {dead_neurons}/{weight_data.shape[0]}")
        
        # Verificar distribuição por entrada
        zeros_per_input = (weight_data == 0).sum(dim=0)
        dead_inputs = (zeros_per_input == weight_data.shape[0]).sum().item()
        print(f"   Inputs completamente ignorados: {dead_inputs}/{weight_data.shape[1]}")
    
    if non_zeros_mask.sum() > 0:
        non_zeros_data = weight_flat[non_zeros_mask]
        print(f"\n📈 ANÁLISE DOS NÃO-ZEROS:")
        print(f"   Min: {non_zeros_data.min().item():.6f}")
        print(f"   Max: {non_zeros_data.max().item():.6f}")
        print(f"   Mean: {non_zeros_data.mean().item():.6f}")
        print(f"   Std: {non_zeros_data.std().item():.6f}")
        print(f"   Median: {non_zeros_data.median().item():.6f}")
    
    # Verificar gradientes se disponíveis
    if first_layer.weight.grad is not None:
        grad_data = first_layer.weight.grad.data
        grad_flat = grad_data.flatten()
        
        grad_zeros = (grad_flat == 0).sum().item()
        grad_zeros_pct = (grad_zeros / grad_flat.numel()) * 100
        
        print(f"\n🔄 ANÁLISE DOS GRADIENTES:")
        print(f"   Gradiente zeros: {grad_zeros} ({grad_zeros_pct:.1f}%)")
        print(f"   Gradiente range: [{grad_flat.min().item():.6f}, {grad_flat.max().item():.6f}]")
        
        # Correlação zeros weight vs zeros gradient
        weight_zeros_mask = (weight_data == 0)
        grad_zeros_mask = (grad_data == 0)
        
        correlation = ((weight_zeros_mask & grad_zeros_mask).sum().float() / 
                      weight_zeros_mask.sum().float()).item()
        print(f"   Correlação peso-zero → grad-zero: {correlation:.3f}")
    
    # Verificar inicialização esperada
    print(f"\n🎯 TESTE DE INICIALIZAÇÃO:")
    
    # Criar layer idêntico com diferentes inicializações
    test_layer = nn.Linear(weight_data.shape[1], weight_data.shape[0], bias=False)
    
    print(f"   Testando diferentes inicializações...")
    
    # Xavier Uniform
    nn.init.xavier_uniform_(test_layer.weight)
    xavier_zeros = (test_layer.weight == 0).sum().item()
    xavier_pct = (xavier_zeros / test_layer.weight.numel()) * 100
    print(f"   Xavier Uniform: {xavier_zeros} zeros ({xavier_pct:.1f}%)")
    
    # Kaiming Uniform
    nn.init.kaiming_uniform_(test_layer.weight, nonlinearity='leaky_relu')
    kaiming_zeros = (test_layer.weight == 0).sum().item()
    kaiming_pct = (kaiming_zeros / test_layer.weight.numel()) * 100
    print(f"   Kaiming Uniform: {kaiming_zeros} zeros ({kaiming_pct:.1f}%)")
    
    # Normal distribution
    nn.init.normal_(test_layer.weight, mean=0, std=0.1)
    normal_zeros = (test_layer.weight == 0).sum().item()
    normal_pct = (normal_zeros / test_layer.weight.numel()) * 100
    print(f"   Normal (0, 0.1): {normal_zeros} zeros ({normal_pct:.1f}%)")
    
    print(f"\n🔍 DIAGNÓSTICO:")
    if zeros_pct > 50:
        print(f"   🚨 PROBLEMA CRÍTICO: {zeros_pct:.1f}% zeros é anormal")
        print(f"   🔍 Inicializações normais têm ~0% zeros")
        print(f"   🎯 Possíveis causas:")
        print(f"      1. Weights sendo zerados durante treinamento")
        print(f"      2. Gradient clipping extremo")
        print(f"      3. Learning rate inadequado")
        print(f"      4. Regularização L1/L2 muito forte")
        print(f"      5. Dead ReLU problem")
        print(f"      6. Optimizer bug")
    else:
        print(f"   ✅ {zeros_pct:.1f}% zeros está normal")
    
    return True

def check_optimizer_state(model):
    """
    Verificar estado do optimizer para problemas
    """
    print(f"\n🔧 ANÁLISE DO OPTIMIZER:")
    
    if hasattr(model.policy, 'actor_optimizer'):
        optimizer = model.policy.actor_optimizer
        print(f"   Tipo: {type(optimizer)}")
        
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"   Group {i}:")
            print(f"      LR: {param_group['lr']}")
            print(f"      Weight decay: {param_group.get('weight_decay', 'N/A')}")
            print(f"      Params: {len(param_group['params'])}")
    
    # Verificar estado específico do primeiro layer
    first_layer = model.policy.actor.latent_pi[0]
    param_id = id(first_layer.weight)
    
    if hasattr(model.policy, 'actor_optimizer'):
        optimizer = model.policy.actor_optimizer
        if param_id in optimizer.state:
            state = optimizer.state[param_id]
            print(f"   Estado do primeiro layer:")
            for key, value in state.items():
                if torch.is_tensor(value):
                    print(f"      {key}: tensor {value.shape}")
                else:
                    print(f"      {key}: {value}")

if __name__ == "__main__":
    try:
        success = analyze_zero_patterns()
        if success:
            print("\n✅ ANÁLISE CONCLUÍDA - Verifique diagnóstico acima")
        else:
            print("\n❌ Não foi possível analisar - modelo não encontrado")
    except Exception as e:
        print(f"\n❌ ERRO na análise: {e}")
        import traceback
        traceback.print_exc()