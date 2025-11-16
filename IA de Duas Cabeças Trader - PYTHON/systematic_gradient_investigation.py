#!/usr/bin/env python3
"""
INVESTIGACAO SISTEMATICA DE GRADIENTES ZERO
Analise completa e estruturada do problema
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

def systematic_gradient_analysis():
    """Investigacao sistematica completa dos gradientes zero"""
    
    print("=" * 80)
    print("INVESTIGACAO SISTEMATICA - GRADIENTES ZERO")
    print("=" * 80)
    
    # ==========================================
    # FASE 1: ANALISE DA ARQUITETURA EXATA
    # ==========================================
    print("\nFASE 1: REPLICANDO ARQUITETURA EXATA")
    print("-" * 50)
    
    # Replicar exatamente a primeira camada problematica
    input_dim = 129
    hidden_dim = 192
    batch_size = 32
    
    # Criar a camada problematica
    temporal_projection_layer1 = nn.Linear(input_dim, hidden_dim)
    gelu = nn.GELU()
    dropout = nn.Dropout(0.05)
    
    print(f"Layer 1: Linear({input_dim}, {hidden_dim})")
    print(f"Weight shape: {temporal_projection_layer1.weight.shape}")
    print(f"Total parameters: {temporal_projection_layer1.weight.numel()}")
    print(f"62.2% de {temporal_projection_layer1.weight.numel()} = {int(temporal_projection_layer1.weight.numel() * 0.622)} zeros")
    
    # ==========================================
    # FASE 2: TESTE DE DIFERENTES INPUTS
    # ==========================================
    print(f"\nFASE 2: TESTANDO DIFERENTES TIPOS DE INPUT")
    print("-" * 50)
    
    test_cases = [
        ("Normal Distribution", lambda: torch.randn(batch_size, input_dim)),
        ("Uniform Distribution", lambda: torch.rand(batch_size, input_dim) * 2 - 1),
        ("Normalized Input", lambda: torch.randn(batch_size, input_dim) * 0.1),
        ("Large Values", lambda: torch.randn(batch_size, input_dim) * 10),
        ("Small Values", lambda: torch.randn(batch_size, input_dim) * 0.01),
        ("Sparse Input (30% zeros)", lambda: torch.randn(batch_size, input_dim) * (torch.rand(batch_size, input_dim) > 0.3).float()),
        ("Very Sparse (70% zeros)", lambda: torch.randn(batch_size, input_dim) * (torch.rand(batch_size, input_dim) > 0.7).float()),
        ("Clipped Input [-2,2]", lambda: torch.clamp(torch.randn(batch_size, input_dim), -2, 2)),
        ("Clipped Input [-1,1]", lambda: torch.clamp(torch.randn(batch_size, input_dim), -1, 1)),
    ]
    
    results = []
    
    for test_name, input_generator in test_cases:
        # Reset layer
        temporal_projection_layer1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(temporal_projection_layer1.weight)
        nn.init.zeros_(temporal_projection_layer1.bias)
        
        # Generate input
        x = input_generator()
        x.requires_grad_(True)
        
        # Forward + backward
        temporal_projection_layer1.zero_grad()
        
        y = temporal_projection_layer1(x)
        y_gelu = gelu(y)
        y_drop = dropout(y_gelu)
        
        loss = y_drop.sum()
        loss.backward()
        
        # Analyze gradients
        weight_grad = temporal_projection_layer1.weight.grad
        bias_grad = temporal_projection_layer1.bias.grad
        
        weight_zeros = (weight_grad.abs() < 1e-8).float().mean().item()
        bias_zeros = (bias_grad.abs() < 1e-8).float().mean().item()
        
        # Input statistics
        input_mean = x.mean().item()
        input_std = x.std().item()
        input_zeros = (x.abs() < 1e-8).float().mean().item()
        
        result = {
            'test': test_name,
            'weight_zeros': weight_zeros * 100,
            'bias_zeros': bias_zeros * 100,
            'input_mean': input_mean,
            'input_std': input_std,
            'input_zeros': input_zeros * 100,
            'output_mean': y.mean().item(),
            'output_std': y.std().item()
        }
        results.append(result)
        
        print(f"{test_name:20s}: Weight {weight_zeros*100:5.1f}% zeros, Bias {bias_zeros*100:5.1f}% zeros, Input {input_zeros*100:4.1f}% zeros")
    
    # ==========================================
    # FASE 3: TESTE DE OTIMIZADORES
    # ==========================================
    print(f"\nFASE 3: TESTANDO DIFERENTES OTIMIZADORES")
    print("-" * 50)
    
    optimizers_to_test = [
        ("Adam lr=1e-4", lambda params: Adam(params, lr=1e-4)),
        ("Adam lr=1e-3", lambda params: Adam(params, lr=1e-3)),
        ("Adam lr=3e-4", lambda params: Adam(params, lr=3e-4)),
        ("SGD lr=1e-3", lambda params: torch.optim.SGD(params, lr=1e-3)),
        ("SGD lr=1e-2", lambda params: torch.optim.SGD(params, lr=1e-2)),
    ]
    
    for opt_name, opt_creator in optimizers_to_test:
        # Create fresh model
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 64)
        )
        
        # Initialize properly
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        optimizer = opt_creator(model.parameters())
        
        # Train for 100 steps
        accumulated_grad_zeros = []
        
        for step in range(100):
            optimizer.zero_grad()
            
            # Generate input
            x = torch.randn(batch_size, input_dim)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            # Check gradients before optimizer step
            first_layer = model[0]
            grad_zeros = (first_layer.weight.grad.abs() < 1e-8).float().mean().item()
            accumulated_grad_zeros.append(grad_zeros)
            
            optimizer.step()
        
        avg_grad_zeros = np.mean(accumulated_grad_zeros[-10:])  # Last 10 steps
        print(f"{opt_name:15s}: Avg grad zeros in last 10 steps: {avg_grad_zeros*100:5.1f}%")
    
    # ==========================================
    # FASE 4: TESTE DE GRADIENT CLIPPING
    # ==========================================
    print(f"\nFASE 4: TESTANDO GRADIENT CLIPPING")
    print("-" * 50)
    
    clip_values = [0.1, 0.5, 1.0, 2.0, 5.0, None]
    
    for clip_value in clip_values:
        model = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
        
        optimizer = Adam(model.parameters(), lr=1e-4)
        
        grad_zeros_over_time = []
        
        for step in range(50):
            optimizer.zero_grad()
            
            x = torch.randn(batch_size, input_dim)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            # Apply gradient clipping if specified
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            grad_zeros = (model.weight.grad.abs() < 1e-8).float().mean().item()
            grad_zeros_over_time.append(grad_zeros)
            
            optimizer.step()
        
        avg_zeros = np.mean(grad_zeros_over_time[-10:])
        clip_str = f"clip={clip_value}" if clip_value else "no_clip"
        print(f"{clip_str:10s}: Avg grad zeros: {avg_zeros*100:5.1f}%")
    
    # ==========================================
    # FASE 5: CONCLUSOES E ANALISE
    # ==========================================
    print(f"\nFASE 5: ANALISE DOS RESULTADOS")
    print("-" * 50)
    
    # Find worst case
    worst_case = max(results, key=lambda x: x['weight_zeros'])
    best_case = min(results, key=lambda x: x['weight_zeros'])
    
    print(f"PIOR CASO: {worst_case['test']}")
    print(f"  Weight zeros: {worst_case['weight_zeros']:.1f}%")
    print(f"  Input stats: mean={worst_case['input_mean']:.4f}, std={worst_case['input_std']:.4f}")
    print(f"  Input zeros: {worst_case['input_zeros']:.1f}%")
    
    print(f"\nMELHOR CASO: {best_case['test']}")
    print(f"  Weight zeros: {best_case['weight_zeros']:.1f}%")
    print(f"  Input stats: mean={best_case['input_mean']:.4f}, std={best_case['input_std']:.4f}")
    print(f"  Input zeros: {best_case['input_zeros']:.1f}%")
    
    # Pattern analysis
    high_zero_cases = [r for r in results if r['weight_zeros'] > 50]
    
    if high_zero_cases:
        print(f"\nCASOS COM >50% ZEROS:")
        for case in high_zero_cases:
            print(f"  {case['test']}: {case['weight_zeros']:.1f}% zeros")
    
    print(f"\nHIPOTESES FINAIS:")
    
    if worst_case['input_zeros'] > 50:
        print("1. HIPOTESE PRINCIPAL: Input muito esparso causa gradientes zero")
    elif worst_case['input_std'] < 0.1:
        print("1. HIPOTESE PRINCIPAL: Input com variancia muito baixa")
    elif worst_case['input_std'] > 5:
        print("1. HIPOTESE PRINCIPAL: Input com variancia muito alta causa saturacao")
    else:
        print("1. HIPOTESE PRINCIPAL: Problema nao esta no input")
    
    print("2. TESTE CRITICO: Rodar este script com dados REAIS do treinamento")
    print("3. VERIFICAR: Se 62.2% zeros ocorre desde o inicio ou desenvolve ao longo do tempo")

if __name__ == "__main__":
    systematic_gradient_analysis()