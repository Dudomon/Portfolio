#!/usr/bin/env python3
"""
DEBUG TEMPORAL PROJECTION - Investigar 68% zeros nos gradientes
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_temporal_projection_issue():
    """Investigar a causa dos 68% zeros na temporal_projection.0.weight"""
    
    print("INVESTIGANDO TEMPORAL PROJECTION ZEROS")
    print("=" * 60)
    
    # Simular a arquitetura exata
    input_dim = 129
    output_dim = 192
    
    # Criar a primeira camada
    layer = nn.Linear(input_dim, output_dim)
    
    print(f"LAYER INFO:")
    print(f"   Weight shape: {layer.weight.shape}")
    print(f"   Total params: {layer.weight.numel()}")
    print(f"   68% de {layer.weight.numel()} = {int(layer.weight.numel() * 0.68)} zeros")
    
    # Testar diferentes cenários de input
    batch_size = 32
    
    print(f"\nTESTANDO CENARIOS DE INPUT:")
    
    # Cenário 1: Input normal
    normal_input = torch.randn(batch_size, input_dim)
    output1 = layer(normal_input)
    print(f"1. Input normal - Output mean: {output1.mean():.4f}, std: {output1.std():.4f}")
    
    # Cenário 2: Input com muitos zeros
    sparse_input = torch.randn(batch_size, input_dim)
    sparse_mask = torch.rand(batch_size, input_dim) > 0.7  # 30% de valores não-zero
    sparse_input = sparse_input * sparse_mask.float()
    output2 = layer(sparse_input)
    zero_ratio = (sparse_input.abs() < 1e-8).float().mean()
    print(f"2. Input sparse ({zero_ratio*100:.1f}% zeros) - Output mean: {output2.mean():.4f}, std: {output2.std():.4f}")
    
    # Cenário 3: Input quase tudo zero (como pode estar acontecendo)
    extreme_sparse_input = torch.randn(batch_size, input_dim)
    extreme_mask = torch.rand(batch_size, input_dim) > 0.95  # 5% de valores não-zero
    extreme_sparse_input = extreme_sparse_input * extreme_mask.float()
    output3 = layer(extreme_sparse_input)
    extreme_zero_ratio = (extreme_sparse_input.abs() < 1e-8).float().mean()
    print(f"3. Input extreme sparse ({extreme_zero_ratio*100:.1f}% zeros) - Output mean: {output3.mean():.4f}, std: {output3.std():.4f}")
    
    # Testar gradientes
    print(f"\nTESTANDO FLUXO DE GRADIENTES:")
    
    # Simular backprop com input normal
    normal_input.requires_grad_(True)
    layer.zero_grad()
    output1 = layer(normal_input)
    loss1 = output1.sum()
    loss1.backward()
    
    grad_zeros_normal = (layer.weight.grad.abs() < 1e-8).float().mean()
    print(f"1. Gradientes com input normal: {grad_zeros_normal*100:.1f}% zeros")
    
    # Simular backprop com input sparse
    sparse_input.requires_grad_(True)
    layer.zero_grad()
    output2 = layer(sparse_input)
    loss2 = output2.sum()
    loss2.backward()
    
    grad_zeros_sparse = (layer.weight.grad.abs() < 1e-8).float().mean()
    print(f"2. Gradientes com input sparse: {grad_zeros_sparse*100:.1f}% zeros")
    
    # Simular backprop com input extreme sparse
    extreme_sparse_input.requires_grad_(True)
    layer.zero_grad()
    output3 = layer(extreme_sparse_input)
    loss3 = output3.sum()
    loss3.backward()
    
    grad_zeros_extreme = (layer.weight.grad.abs() < 1e-8).float().mean()
    print(f"3. Gradientes com input extreme sparse: {grad_zeros_extreme*100:.1f}% zeros")
    
    print(f"\nANALISE:")
    if grad_zeros_extreme > 0.6:
        print("CAUSA PROVAVEL: Inputs muito esparsos (muitos zeros)")
        print("   Solucao: Verificar normalizacao/preprocessing dos dados")
    elif grad_zeros_sparse > 0.3:
        print("CAUSA POSSIVEL: Inputs moderadamente esparsos")
        print("   Solucao: Ajustar normalizacao ou arquitetura")
    else:  
        print("Gradientes normais com inputs regulares")
        print("   Problema pode estar em outro lugar")
    
    print(f"\nHIPOTESES PARA INVESTIGAR:")
    print("1. VecNormalize zerando features após normalização")
    print("2. Dropout excessivo durante treinamento")
    print("3. Dead neurons por saturação de ativação")
    print("4. Inputs sendo preprocessados incorretamente")
    print("5. Learning rate muito baixo causando gradientes ~0")

if __name__ == "__main__":
    analyze_temporal_projection_issue()