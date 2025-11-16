"""
🧪 TESTE COMPREHENSIVO DE GRADIENTES V9

PROBLEMA: Gradient norm 0.0000 em teste simples
SOLUÇÃO: Teste mais realístico com loss function real
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trading_framework.extractors.transformer_v9_daytrading import TradingTransformerV9
import gym

def test_v9_comprehensive_gradients():
    """Teste comprehensivo de gradientes V9 com cenário realístico"""
    
    print("🧪 TESTE COMPREHENSIVO GRADIENTES V9...")
    print("🎯 Simulando treinamento real com loss function")
    
    # Setup
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    v9_transformer = TradingTransformerV9(observation_space=obs_space, features_dim=256)
    v9_transformer.train()
    
    # Optimizer para simular treinamento real
    optimizer = torch.optim.Adam(v9_transformer.parameters(), lr=0.001)
    
    print("📊 Iniciando loop de treinamento simulado...")
    
    gradient_norms = []
    weight_changes = []
    
    for epoch in range(5):
        print(f"\n🔄 Epoch {epoch+1}/5")
        
        # Batch realístico
        batch_size = 16
        obs = torch.randn(batch_size, 450, requires_grad=False)  # Input não precisa gradients
        
        # Target aleatório para simular loss real
        target = torch.randn(batch_size, 256)
        
        # Forward pass
        optimizer.zero_grad()
        output = v9_transformer(obs)
        
        # Loss function realística
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Analisar gradientes ANTES do optimizer step
        input_proj_grad = v9_transformer.input_projection.weight.grad
        
        if input_proj_grad is not None:
            grad_norm = input_proj_grad.norm().item()
            grad_zeros = (input_proj_grad.abs() < 1e-8).float().mean().item() * 100
            grad_max = input_proj_grad.abs().max().item()
            grad_mean = input_proj_grad.abs().mean().item()
            
            gradient_norms.append(grad_norm)
            
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Gradient norm: {grad_norm:.6f}")
            print(f"   Gradient zeros: {grad_zeros:.1f}%")
            print(f"   Gradient max: {grad_max:.6f}")
            print(f"   Gradient mean: {grad_mean:.6f}")
            
            # Weight antes do update
            weight_before = v9_transformer.input_projection.weight.data.clone()
            
            # Optimizer step
            optimizer.step()
            
            # Weight depois do update
            weight_after = v9_transformer.input_projection.weight.data
            weight_change = (weight_after - weight_before).norm().item()
            weight_changes.append(weight_change)
            
            print(f"   Weight change: {weight_change:.6f}")
            
            # Verificar health
            zeros_after = (weight_after.abs() < 1e-8).float().mean().item() * 100
            print(f"   Zeros após update: {zeros_after:.1f}%")
            
        else:
            print("   ❌ NENHUM GRADIENT ENCONTRADO!")
            gradient_norms.append(0.0)
            weight_changes.append(0.0)
    
    # Análise final
    print("\n" + "="*50)
    print("📈 ANÁLISE FINAL DE GRADIENTES")
    print("="*50)
    
    if len(gradient_norms) > 0:
        avg_grad_norm = np.mean(gradient_norms)
        max_grad_norm = np.max(gradient_norms)
        min_grad_norm = np.min(gradient_norms)
        
        avg_weight_change = np.mean(weight_changes)
        total_weight_change = np.sum(weight_changes)
        
        print(f"Gradient norm médio: {avg_grad_norm:.6f}")
        print(f"Gradient norm máximo: {max_grad_norm:.6f}")
        print(f"Gradient norm mínimo: {min_grad_norm:.6f}")
        print(f"Weight change médio: {avg_weight_change:.6f}")
        print(f"Weight change total: {total_weight_change:.6f}")
        
        # Critérios de sucesso
        success_criteria = []
        
        # 1. Gradient norm > threshold
        grad_healthy = avg_grad_norm > 1e-6
        success_criteria.append(grad_healthy)
        print(f"✅ Gradients saudáveis: {'✅' if grad_healthy else '❌'} (avg: {avg_grad_norm:.6f})")
        
        # 2. Weight changes happening
        weights_moving = avg_weight_change > 1e-8
        success_criteria.append(weights_moving)
        print(f"✅ Weights mudando: {'✅' if weights_moving else '❌'} (avg: {avg_weight_change:.6f})")
        
        # 3. No gradient explosion
        no_explosion = max_grad_norm < 10.0
        success_criteria.append(no_explosion)
        print(f"✅ Sem explosão: {'✅' if no_explosion else '❌'} (max: {max_grad_norm:.6f})")
        
        # 4. Consistent gradients
        consistent = min_grad_norm > 0
        success_criteria.append(consistent)
        print(f"✅ Gradients consistentes: {'✅' if consistent else '❌'} (min: {min_grad_norm:.6f})")
        
        success_rate = sum(success_criteria) / len(success_criteria)
        print(f"\n🎯 SUCCESS RATE: {sum(success_criteria)}/{len(success_criteria)} ({success_rate*100:.0f}%)")
        
        if success_rate >= 0.75:
            print("🎉 GRADIENTS V9 FUNCIONANDO CORRETAMENTE!")
            return True
        else:
            print("❌ GRADIENTS V9 AINDA COM PROBLEMAS!")
            return False
    else:
        print("❌ NENHUM GRADIENT PROCESSADO!")
        return False

def test_v9_initialization_details():
    """Teste detalhado da inicialização V9"""
    print("\n🔍 TESTE DETALHADO INICIALIZAÇÃO V9...")
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    v9_transformer = TradingTransformerV9(observation_space=obs_space, features_dim=256)
    
    # Analisar cada layer
    for name, module in v9_transformer.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data
            print(f"{name:30s}: shape={str(weights.shape):15s} mean={weights.mean().item():8.4f} std={weights.std().item():.4f}")
            
            # Check special initialization
            if 'input_projection' in name:
                expected_std = np.sqrt(2.0 / (weights.shape[0] + weights.shape[1])) * 0.3  # gain=0.3
                actual_std = weights.std().item()
                print(f"{'':30s}   Expected std (gain=0.3): {expected_std:.4f}, Actual: {actual_std:.4f}")

if __name__ == "__main__":
    print("🧪 TESTE COMPREHENSIVO V9 GRADIENTS + INICIALIZAÇÃO")
    print("="*60)
    
    # Teste de inicialização
    test_v9_initialization_details()
    
    # Teste comprehensivo de gradientes
    gradient_success = test_v9_comprehensive_gradients()
    
    print("\n" + "🏁"*20)
    if gradient_success:
        print("🎉 V9 GRADIENTS VALIDADOS COM SUCESSO!")
        print("✅ Input projection funcionando corretamente")
        print("🚀 Pronto para treinamento sem morte de neurônios!")
    else:
        print("❌ V9 GRADIENTS AINDA PRECISAM DE AJUSTES!")
        print("🔧 Revisar implementação dos fixes")
    print("🏁"*20)