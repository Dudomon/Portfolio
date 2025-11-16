#!/usr/bin/env python3
"""
🔍 DEBUGGING GRADIENT ZEROS V8Heritage - Investigação específica
Vamos identificar exatamente qual layer está com 30.9% zeros nos gradientes
"""

import sys
sys.path.append("D:/Projeto")

import torch
import torch.nn as nn
from trading_framework.policies.two_head_v8_heritage import TwoHeadV8Heritage, get_v8_heritage_policy_kwargs
from stable_baselines3.common.policies import BasePolicy
from sb3_contrib import RecurrentPPO

def debug_v8_gradients_live():
    """Debug gradientes em um modelo V8Heritage real (se disponível)"""
    print("🔍 DEBUGGING V8HERITAGE GRADIENT ZEROS")
    print("=" * 60)
    
    try:
        # Tentar carregar um checkpoint V8Heritage real
        import glob
        
        # Procurar por checkpoints V8Heritage
        checkpoint_patterns = [
            "D:/Projeto/Otimizacao/treino_principal/models/HeritageV8*.zip",
            "D:/Projeto/**/HeritageV8*.zip",
            "D:/Projeto/**/*V8Heritage*.zip"
        ]
        
        all_checkpoints = []
        for pattern in checkpoint_patterns:
            all_checkpoints.extend(glob.glob(pattern, recursive=True))
        
        if not all_checkpoints:
            print("⚠️ Nenhum checkpoint V8Heritage encontrado, usando V8 sintético")
            debug_synthetic_v8()
            return
        
        print(f"✅ Encontrado checkpoint V8Heritage: {all_checkpoints[0]}")
        
        # Carregar modelo
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RecurrentPPO.load(all_checkpoints[0], device=device)
        
        # Verificar se é V8Heritage
        policy_name = model.policy.__class__.__name__
        print(f"🧠 Policy carregada: {policy_name}")
        
        if "V8Heritage" not in policy_name:
            print("⚠️ Policy não é V8Heritage, usando debug sintético")
            debug_synthetic_v8()
            return
        
        # Analisar gradientes do modelo real
        print("\n🔍 ANALISANDO GRADIENTES DO MODELO V8HERITAGE REAL:")
        analyze_model_gradients(model.policy)
        
    except Exception as e:
        print(f"❌ Erro ao carregar V8Heritage real: {e}")
        print("⚠️ Fallback para debug sintético")
        debug_synthetic_v8()

def debug_synthetic_v8():
    """Debug com modelo V8Heritage sintético"""
    print("\n🧠 CRIANDO V8HERITAGE SINTÉTICO PARA DEBUG:")
    
    # Criar policy V8Heritage sintética
    try:
        kwargs = get_v8_heritage_policy_kwargs()
        
        # Criar observação sintética (2580 dimensões como no V8Heritage)
        observation_space = torch.randn(1, 2580)
        action_space_size = 8
        
        print("🎯 V8Heritage sintético criado para debug")
        
        # Simular treinamento para gerar gradientes
        simulate_training_step(observation_space)
        
    except Exception as e:
        print(f"❌ Erro ao criar V8Heritage sintético: {e}")

def simulate_training_step(obs):
    """Simular um step de treinamento para gerar gradientes"""
    print("\n⚡ SIMULANDO TRAINING STEP PARA GERAR GRADIENTES:")
    
    # Criar OptimizedV8DecisionMaker
    from trading_framework.policies.two_head_v8_heritage import OptimizedV8DecisionMaker
    
    decision_maker = OptimizedV8DecisionMaker(input_dim=256)
    decision_maker.train()
    
    # Input sintético
    batch_size = 32
    input_features = torch.randn(batch_size, 256, requires_grad=True)
    target_actions = torch.randn(batch_size, 8)  # Target actions para calcular loss
    
    # Forward pass
    optimizer = torch.optim.Adam(decision_maker.parameters(), lr=1e-4)
    
    for step in range(5):  # Simular 5 steps
        optimizer.zero_grad()
        
        # Forward
        output = decision_maker(input_features)
        
        # Loss simples
        loss = torch.nn.functional.mse_loss(output, target_actions)
        
        # Backward
        loss.backward()
        
        print(f"\nStep {step + 1} - Loss: {loss.item():.6f}")
        
        # Analisar gradientes após backward
        analyze_gradients(decision_maker, step + 1)
        
        # Update
        optimizer.step()

def analyze_gradients(model, step):
    """Analisar gradientes detalhadamente"""
    print(f"  🔍 Analisando gradientes (Step {step}):")
    
    total_params = 0
    total_zeros = 0
    critical_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            
            # Estatísticas do gradiente
            total_elements = grad.numel()
            zero_elements = (grad == 0).sum().item()
            near_zero_elements = (grad.abs() < 1e-8).sum().item()
            zero_pct = (zero_elements / total_elements) * 100
            near_zero_pct = (near_zero_elements / total_elements) * 100
            
            total_params += total_elements
            total_zeros += zero_elements
            
            # Mostrar detalhes para layers críticos
            if zero_pct > 10 or "entry_quality_head" in name:
                print(f"    🚨 {name}:")
                print(f"       Shape: {grad.shape}")
                print(f"       Zeros: {zero_elements}/{total_elements} ({zero_pct:.1f}%)")
                print(f"       Near-zeros: {near_zero_elements}/{total_elements} ({near_zero_pct:.1f}%)")
                print(f"       Range: [{grad.min():.2e}, {grad.max():.2e}]")
                print(f"       Mean: {grad.mean():.2e}, Std: {grad.std():.2e}")
                
                critical_layers.append((name, zero_pct))
                
                # Análise específica se é LayerNorm
                if "1." in name and ("LayerNorm" in str(type(model)) or grad.shape == torch.Size([256]) or grad.shape == torch.Size([128])):
                    print(f"       🔬 SUSPEITA LayerNorm: shape {grad.shape}")
    
    # Resumo geral
    overall_zero_pct = (total_zeros / total_params) * 100 if total_params > 0 else 0
    print(f"  📊 Resumo: {total_zeros}/{total_params} ({overall_zero_pct:.1f}%) zeros nos gradientes")
    
    # Alertas críticos
    for name, zero_pct in critical_layers:
        if zero_pct > 25:
            print(f"  🚨 CRÍTICO: {name} com {zero_pct:.1f}% zeros!")

def analyze_model_gradients(policy):
    """Analisar gradientes em um modelo real"""
    print("🔍 ANALISANDO GRADIENTES DO MODELO REAL:")
    
    # Listar todos os parâmetros com gradientes
    gradient_info = []
    
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            total_elements = grad.numel()
            zero_elements = (grad == 0).sum().item()
            zero_pct = (zero_elements / total_elements) * 100
            
            gradient_info.append((name, zero_pct, zero_elements, total_elements))
    
    # Ordenar por % de zeros
    gradient_info.sort(key=lambda x: x[1], reverse=True)
    
    print("🔥 TOP GRADIENTES COM MAIS ZEROS:")
    for i, (name, zero_pct, zeros, total) in enumerate(gradient_info[:10]):
        print(f"  {i+1:2d}. {name}")
        print(f"      🚨 {zeros:,}/{total:,} ({zero_pct:.1f}%) zeros")
        
        # Identificar se é o problema específico
        if "decision_maker.entry_quality_head.1.weight" in name or zero_pct > 30:
            print(f"      ⚠️ POSSÍVEL MATCH PARA O PROBLEMA!")

if __name__ == "__main__":
    debug_v8_gradients_live()