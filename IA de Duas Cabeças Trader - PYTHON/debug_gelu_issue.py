#!/usr/bin/env python3
"""
DEBUG GELU ISSUE - Verificar se GELU está causando os zeros
"""

import torch
import torch.nn as nn
import numpy as np

def test_gelu_dead_zones():
    """Testar se GELU está causando dead zones"""
    
    print("TESTANDO GELU DEAD ZONES")
    print("=" * 50)
    
    # Criar sequential como no código real
    input_dim = 129
    
    temporal_projection = nn.Sequential(
        nn.Linear(input_dim, 192),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(192, 128)
    )
    
    batch_size = 32
    
    print("TESTE 1: Input com distribuição normal")
    normal_input = torch.randn(batch_size, input_dim)
    
    # Forward pass
    x1 = temporal_projection[0](normal_input)  # First linear
    print(f"Após Linear 1: mean={x1.mean():.4f}, std={x1.std():.4f}, min={x1.min():.4f}, max={x1.max():.4f}")
    
    x2 = temporal_projection[1](x1)  # GELU
    print(f"Após GELU: mean={x2.mean():.4f}, std={x2.std():.4f}, min={x2.min():.4f}, max={x2.max():.4f}")
    gelu_zeros = (x2.abs() < 1e-8).float().mean()
    print(f"GELU zeros: {gelu_zeros*100:.1f}%")
    
    x3 = temporal_projection[2](x2)  # Dropout
    print(f"Após Dropout: mean={x3.mean():.4f}, std={x3.std():.4f}")
    dropout_zeros = (x3.abs() < 1e-8).float().mean()
    print(f"Dropout zeros: {dropout_zeros*100:.1f}%")
    
    # Test gradients
    x3.requires_grad_(True)
    loss = x3.sum()
    loss.backward(retain_graph=True)
    
    grad1_zeros = (temporal_projection[0].weight.grad.abs() < 1e-8).float().mean()
    print(f"Linear1 grad zeros: {grad1_zeros*100:.1f}%")
    
    print("\nTESTE 2: Input normalizado (como VecNormalize)")
    # Simular input já normalizado pelo VecNormalize
    normalized_input = torch.randn(batch_size, input_dim)
    normalized_input = torch.clamp(normalized_input, -2.0, 2.0)  # Clipped
    
    temporal_projection.zero_grad()
    
    y1 = temporal_projection[0](normalized_input)
    print(f"Após Linear 1 (norm): mean={y1.mean():.4f}, std={y1.std():.4f}, min={y1.min():.4f}, max={y1.max():.4f}")
    
    y2 = temporal_projection[1](y1)
    print(f"Após GELU (norm): mean={y2.mean():.4f}, std={y2.std():.4f}")
    gelu_zeros_norm = (y2.abs() < 1e-8).float().mean()
    print(f"GELU zeros (norm): {gelu_zeros_norm*100:.1f}%")
    
    y3 = temporal_projection[2](y2)
    dropout_zeros_norm = (y3.abs() < 1e-8).float().mean()
    print(f"Dropout zeros (norm): {dropout_zeros_norm*100:.1f}%")
    
    loss_norm = y3.sum()
    loss_norm.backward()
    
    grad1_zeros_norm = (temporal_projection[0].weight.grad.abs() < 1e-8).float().mean()
    print(f"Linear1 grad zeros (norm): {grad1_zeros_norm*100:.1f}%")
    
    print("\nTESTE 3: Input extremamente pequeno (underflow)")
    tiny_input = torch.randn(batch_size, input_dim) * 1e-6  # Valores muito pequenos
    
    temporal_projection.zero_grad()
    
    z1 = temporal_projection[0](tiny_input)
    print(f"Tiny input - Após Linear 1: mean={z1.mean():.8f}, std={z1.std():.8f}")
    
    z2 = temporal_projection[1](z1)
    gelu_zeros_tiny = (z2.abs() < 1e-8).float().mean()
    print(f"Tiny input - GELU zeros: {gelu_zeros_tiny*100:.1f}%")
    
    z3 = temporal_projection[2](z2)
    loss_tiny = z3.sum()
    loss_tiny.backward()
    
    grad1_zeros_tiny = (temporal_projection[0].weight.grad.abs() < 1e-8).float().mean()
    print(f"Tiny input - Linear1 grad zeros: {grad1_zeros_tiny*100:.1f}%")
    
    print(f"\nCONCLUSAO:")
    print(f"Input normal: {grad1_zeros*100:.1f}% grad zeros")
    print(f"Input normalizado: {grad1_zeros_norm*100:.1f}% grad zeros")
    print(f"Input tiny: {grad1_zeros_tiny*100:.1f}% grad zeros")
    
    if grad1_zeros_tiny > 0.5:
        print("PROBLEMA IDENTIFICADO: Numerical underflow!")
        print("Inputs muito pequenos -> Gradientes ~0")
    elif grad1_zeros_norm > 0.4:
        print("PROBLEMA IDENTIFICADO: Normalizacao excessiva!")
    else:
        print("Problema nao reproduzido neste teste")

if __name__ == "__main__":
    test_gelu_dead_zones()