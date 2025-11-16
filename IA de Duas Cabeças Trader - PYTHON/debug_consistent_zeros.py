#!/usr/bin/env python3
"""
DEBUG CONSISTENT ZEROS - Investigar os 3.1% zeros consistentes
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_consistent_zeros():
    """Analisar o padrão de zeros consistentes"""
    
    print("INVESTIGANDO 3.1% ZEROS CONSISTENTES")
    print("=" * 50)
    
    # Simular dados como aparecem no log
    input_dim = 129
    seq_len = 20
    batch_size = 32
    
    # Criar dados simulando o padrão observado
    print("TESTE 1: Simular 3.1% zeros fixos")
    
    # Calcular quantos zeros seriam 3.1%
    total_elements = batch_size * input_dim
    zeros_count = int(total_elements * 0.031)  # 3.1%
    
    print(f"Total elements por timestep: {total_elements}")
    print(f"3.1% = {zeros_count} zeros")
    print(f"Isso significa {zeros_count}/{total_elements} = {zeros_count/total_elements*100:.1f}%")
    
    # Simular dados com exatamente 3.1% zeros fixos
    temporal_data = []
    
    for t in range(seq_len):
        # Criar data normal
        data = torch.randn(batch_size, input_dim)
        
        # Zerar exatamente 3.1% de forma fixa (mesmas posições)
        zero_mask = torch.zeros(batch_size, input_dim, dtype=torch.bool)
        
        # Zerar sempre as mesmas features (simular problema real)
        fixed_features = [0, 3, 5, 7]  # Features que podem estar sempre zeradas
        zero_mask[:, fixed_features] = True
        
        # Zerar elementos para atingir exatamente 3.1%
        remaining_zeros = zeros_count - zero_mask.sum().item()
        if remaining_zeros > 0:
            flat_indices = torch.randperm(batch_size * input_dim)[:remaining_zeros]
            for idx in flat_indices:
                row = idx // input_dim
                col = idx % input_dim
                if not zero_mask[row, col]:  # Só se não estiver já zerado
                    zero_mask[row, col] = True
        
        # Aplicar máscara
        data[zero_mask] = 0.0
        
        actual_zeros = (data.abs() < 1e-8).float().mean()
        print(f"t={t}: {actual_zeros*100:.1f}% zeros")
        
        temporal_data.append(data)
    
    print(f"\nTESTE 2: Impacto nos gradientes")
    
    # Criar temporal projection
    temporal_projection = nn.Sequential(
        nn.Linear(input_dim, 192),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(192, 128)
    )
    
    # Processar todos os timesteps
    all_losses = []
    temporal_projection.zero_grad()
    
    for t, data in enumerate(temporal_data):
        output = temporal_projection(data)
        loss = output.sum()
        all_losses.append(loss)
    
    # Backward pass
    total_loss = sum(all_losses)
    total_loss.backward()
    
    # Analisar gradientes
    first_layer_grad = temporal_projection[0].weight.grad
    grad_zeros = (first_layer_grad.abs() < 1e-8).float().mean()
    
    print(f"Gradientes zeros com 3.1% input zeros: {grad_zeros*100:.1f}%")
    
    print(f"\nTESTE 3: Investigar features sempre zeradas")
    
    # Analisar quais features são sempre zero
    stacked_data = torch.stack(temporal_data, dim=1)  # [batch, seq, features]
    
    # Verificar features que são sempre zero
    always_zero_features = []
    for feature_idx in range(input_dim):
        feature_data = stacked_data[:, :, feature_idx]  # [batch, seq]
        if (feature_data.abs() < 1e-8).all():
            always_zero_features.append(feature_idx)
    
    print(f"Features sempre zeradas: {always_zero_features}")
    print(f"Isso representa {len(always_zero_features)/input_dim*100:.1f}% das features")
    
    # Verificar features que são frequentemente zero
    frequent_zero_features = []
    for feature_idx in range(input_dim):
        feature_data = stacked_data[:, :, feature_idx]
        zero_ratio = (feature_data.abs() < 1e-8).float().mean()
        if zero_ratio > 0.5:  # Mais de 50% zeros
            frequent_zero_features.append((feature_idx, zero_ratio.item()))
    
    print(f"\nFeatures frequentemente zeradas (>50% zeros):")
    for feat_idx, ratio in frequent_zero_features:
        print(f"  Feature {feat_idx}: {ratio*100:.1f}% zeros")
    
    print(f"\nCONCLUSAO:")
    if grad_zeros > 0.6:
        print("PROBLEMA CONFIRMADO: 3.1% zeros consistentes causam 60%+ grad zeros")
        print("SOLUCAO: Investigar quais features estão sempre zeradas no preprocessamento")
    elif len(always_zero_features) > 0:
        print(f"PROBLEMA PARCIAL: {len(always_zero_features)} features sempre zeradas")
        print("SOLUCAO: Remover ou corrigir features mortas no preprocessamento")
    else:
        print("Padrao nao reproduzido - problema pode ser mais complexo")

if __name__ == "__main__":
    analyze_consistent_zeros()