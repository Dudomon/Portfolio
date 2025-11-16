#!/usr/bin/env python3
"""
🔬 INVESTIGAÇÃO DOS PESOS - ACTION[1] SEMPRE ZERO
Verificar se os pesos da Action[1] estão zerados ou bloqueados
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

def investigar_pesos_action1():
    print("🔬 INVESTIGAÇÃO DOS PESOS - ACTION[1]")
    print("=" * 50)
    
    # Carregar modelo
    checkpoint_path = projeto_path / "trading_framework/training/checkpoints/DAYTRADER/checkpoint_phase2riskmanagement_650000_steps_20250805_201935.zip"
    
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(checkpoint_path)
        print(f"✅ Modelo carregado: {model.num_timesteps:,} steps")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    policy = model.policy
    
    # 1. Investigar action_net (Linear layer final)
    print(f"\n🔍 INVESTIGAÇÃO DO ACTION_NET")
    print("-" * 40)
    
    action_net = policy.action_net
    print(f"📊 Action Network: {type(action_net)}")
    print(f"📊 Input/Output shape: {action_net.in_features} → {action_net.out_features}")
    
    # Verificar pesos e bias
    weight = action_net.weight  # Shape: [11, 128] (11 actions, 128 input features)
    bias = action_net.bias      # Shape: [11] (11 actions)
    
    print(f"📊 Weight shape: {weight.shape}")
    print(f"📊 Bias shape: {bias.shape}")
    
    # 2. Análise específica da Action[1]
    print(f"\n🎯 ANÁLISE ESPECÍFICA DA ACTION[1] (Quantidade)")
    print("-" * 40)
    
    action1_weights = weight[1, :]  # Pesos da Action[1]
    action1_bias = bias[1]          # Bias da Action[1]
    
    print(f"💰 Action[1] - Pesos da Quantidade:")
    print(f"   Peso médio: {action1_weights.mean():.8f}")
    print(f"   Peso std: {action1_weights.std():.8f}")
    print(f"   Peso min/max: {action1_weights.min():.8f} / {action1_weights.max():.8f}")
    print(f"   Bias: {action1_bias:.8f}")
    
    # Verificar se está zerado
    if torch.all(action1_weights == 0):
        print(f"   🔴 CRÍTICO: TODOS OS PESOS DA ACTION[1] ESTÃO ZERADOS!")
    elif torch.std(action1_weights) < 1e-8:
        print(f"   🟡 PROBLEMA: Pesos quase constantes (std={torch.std(action1_weights):.2e})")
    else:
        print(f"   ✅ Pesos variam normalmente")
    
    # 3. Comparar com outras ações
    print(f"\n📊 COMPARAÇÃO COM OUTRAS AÇÕES")
    print("-" * 40)
    
    for i in range(min(11, weight.shape[0])):
        action_weights = weight[i, :]
        action_bias = bias[i]
        
        action_names = [
            "Order Type", "Quantity", "Flag1", "Flag2", "Flag3", 
            "SL1", "SL2", "SL3", "TP1", "TP2", "TP3"
        ]
        name = action_names[i] if i < len(action_names) else f"Action[{i}]"
        
        print(f"   {name:12s}: mean={action_weights.mean():.6f}, std={action_weights.std():.6f}, bias={action_bias:.6f}")
        
        # Detectar problemas
        if torch.all(action_weights == 0):
            print(f"                🔴 ZERADO!")
        elif torch.std(action_weights) < 1e-7:
            print(f"                🟡 QUASE CONSTANTE")
    
    # 4. Teste de forward pass manual
    print(f"\n🧪 TESTE DE FORWARD PASS MANUAL")
    print("-" * 40)
    
    try:
        # Criar input sintético
        input_features = torch.randn(1, 128)  # Batch size 1, 128 features
        
        # Forward pass manual
        output = torch.matmul(input_features, weight.T) + bias
        
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output valores:")
        for i in range(min(11, output.shape[1])):
            action_names = [
                "Order Type", "Quantity", "Flag1", "Flag2", "Flag3", 
                "SL1", "SL2", "SL3", "TP1", "TP2", "TP3"
            ]
            name = action_names[i] if i < len(action_names) else f"Action[{i}]"
            print(f"   {name:12s}: {output[0, i]:.6f}")
        
        # Verificar se Action[1] pode produzir valores não-zero
        print(f"\n🔍 TESTE DE RANGE DA ACTION[1]:")
        
        # Testar com inputs extremos
        extreme_inputs = [
            torch.ones(1, 128) * 10,    # Input muito positivo
            -torch.ones(1, 128) * 10,   # Input muito negativo
            torch.randn(1, 128) * 5     # Input aleatório forte
        ]
        
        for i, extreme_input in enumerate(extreme_inputs):
            extreme_output = torch.matmul(extreme_input, weight.T) + bias
            action1_output = extreme_output[0, 1]
            print(f"   Teste {i+1}: Action[1] = {action1_output:.6f}")
            
            # Aplicar activation (tanh para bound [-1,1], depois scale para [0,1])
            action1_activated = torch.tanh(action1_output)  # [-1, 1]
            action1_scaled = (action1_activated + 1) / 2    # [0, 1]
            print(f"            Após ativação: {action1_scaled:.6f}")
    
    except Exception as e:
        print(f"❌ Erro no teste manual: {e}")
    
    # 5. Verificar se há clipping ou masking
    print(f"\n🔍 INVESTIGAÇÃO DE CLIPPING/MASKING")
    print("-" * 40)
    
    # Verificar se há clipping no policy
    if hasattr(policy, 'action_dist'):
        print(f"✅ Action distribution encontrada: {type(policy.action_dist)}")
        
        # Se for DiagGaussian, verificar log_std
        if hasattr(policy.action_dist, 'log_std'):
            log_std = policy.action_dist.log_std
            print(f"   Log std shape: {log_std.shape}")
            print(f"   Log std values: {log_std}")
            
            # Verificar se Action[1] tem std muito baixo
            if len(log_std) > 1:
                action1_log_std = log_std[1]
                action1_std = torch.exp(action1_log_std)
                print(f"   Action[1] std: {action1_std:.8f}")
                
                if action1_std < 1e-6:
                    print(f"   🔴 PROBLEMA: STD DA ACTION[1] MUITO BAIXO!")
                else:
                    print(f"   ✅ STD normal")
    
    # 6. Análise de gradientes (se disponível)
    print(f"\n🔍 VERIFICAÇÃO DE REQUIRES_GRAD")
    print("-" * 40)
    
    print(f"   Weight requires_grad: {weight.requires_grad}")
    print(f"   Bias requires_grad: {bias.requires_grad}")
    
    if weight.grad is not None:
        action1_weight_grad = weight.grad[1, :]
        print(f"   Action[1] gradient mean: {action1_weight_grad.mean():.8f}")
        print(f"   Action[1] gradient std: {action1_weight_grad.std():.8f}")
        
        if torch.all(action1_weight_grad == 0):
            print(f"   🔴 GRADIENTES DA ACTION[1] ESTÃO ZERADOS!")
        else:
            print(f"   ✅ Gradientes existem")
    else:
        print(f"   ℹ️ Gradientes não disponíveis (normal em inference)")
    
    # 7. Conclusão da investigação
    print(f"\n🏆 CONCLUSÃO DA INVESTIGAÇÃO DOS PESOS")
    print("=" * 50)
    
    # Análise dos achados
    action1_weight_zero = torch.all(action1_weights == 0)
    action1_weight_const = torch.std(action1_weights) < 1e-7
    action1_bias_zero = abs(action1_bias) < 1e-8
    
    print(f"🎯 DIAGNÓSTICO DA ACTION[1]:")
    
    if action1_weight_zero:
        print(f"   🔴 CRÍTICO: Pesos completamente zerados")
        print(f"   💡 CAUSA: Inicialização ruim ou gradientes zerados durante treino")
    elif action1_weight_const:
        print(f"   🟡 PROBLEMA: Pesos quase constantes")
        print(f"   💡 CAUSA: Gradientes muito pequenos ou learning rate inadequado")
    else:
        print(f"   ✅ PESOS: Normais (mean={action1_weights.mean():.6f}, std={action1_weights.std():.6f})")
    
    if action1_bias_zero:
        print(f"   🔴 PROBLEMA: Bias zerado")
    else:
        print(f"   ✅ BIAS: Normal ({action1_bias:.6f})")
    
    print(f"\n💡 POSSÍVEIS SOLUÇÕES:")
    if action1_weight_zero or action1_weight_const:
        print(f"   1. 🔄 Re-treinar com learning rate maior para Action[1]")
        print(f"   2. 🎯 Verificar se reward function usa quantidade")
        print(f"   3. 🔧 Implementar weight initialization específica")
        print(f"   4. 📊 Adicionar regularização para forçar variação")
    else:
        print(f"   1. 🔍 Verificar activation function e bounds")
        print(f"   2. 🎮 Problema pode estar no action processing")
        print(f"   3. 📊 Verificar se environment usa Action[1]")

def main():
    investigar_pesos_action1()

if __name__ == "__main__":
    main()