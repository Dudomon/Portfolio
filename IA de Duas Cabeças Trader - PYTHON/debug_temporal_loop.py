#!/usr/bin/env python3
"""
DEBUG TEMPORAL LOOP - Investigar o loop temporal específico
"""

import torch
import torch.nn as nn
import numpy as np

def test_temporal_loop_issue():
    """Testar o loop temporal que processa 20 barras"""
    
    print("TESTANDO TEMPORAL LOOP - 20 BARRAS")
    print("=" * 50)
    
    # Replicar exatamente a estrutura do código
    input_dim = 129  # features por barra
    seq_len = 20     # 20 barras
    d_model = 128
    batch_size = 32
    
    # Criar temporal_projection como no código original
    temporal_projection = nn.Sequential(
        nn.Linear(input_dim, 192),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(192, d_model)
    )
    
    print(f"Setup: {batch_size} batch, {seq_len} seq_len, {input_dim} input_dim")
    
    # Simular observations como chegam do ambiente
    total_features = seq_len * input_dim  # 2580
    observations = torch.randn(batch_size, total_features)
    
    # Reshape como no código original
    temporal_obs = observations.view(batch_size, seq_len, input_dim)
    print(f"Temporal obs shape: {temporal_obs.shape}")
    
    # REPLICAR O LOOP EXATO DO CÓDIGO
    temporal_variations = []
    
    # Processar cada timestep como no código original
    for t in range(seq_len):
        bar_features = temporal_obs[:, t, :]  # [batch_size, input_dim]
        projected_features = temporal_projection(bar_features)
        temporal_variations.append(projected_features)
    
    # Stack como no código original
    temporal_seq = torch.stack(temporal_variations, dim=1)
    print(f"Final temporal_seq shape: {temporal_seq.shape}")
    
    # Simular loss e backprop
    loss = temporal_seq.sum()  # Simple loss
    loss.backward()
    
    # Analisar gradientes
    first_linear = temporal_projection[0]
    grad_zeros = (first_linear.weight.grad.abs() < 1e-8).float().mean()
    print(f"Gradientes zeros no loop temporal: {grad_zeros*100:.1f}%")
    
    # TESTE COMPARATIVO: Processar tudo de uma vez (sem loop)
    print(f"\nTESTE COMPARATIVO - SEM LOOP:")
    
    # Criar nova layer para comparar
    temporal_projection_batch = nn.Sequential(
        nn.Linear(input_dim, 192),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(192, d_model)
    )
    
    # Processar todas as barras de uma vez
    flat_temporal = temporal_obs.view(-1, input_dim)  # [batch*seq, input_dim]
    batch_processed = temporal_projection_batch(flat_temporal)
    batch_result = batch_processed.view(batch_size, seq_len, d_model)
    
    loss_batch = batch_result.sum()
    loss_batch.backward()
    
    grad_zeros_batch = (temporal_projection_batch[0].weight.grad.abs() < 1e-8).float().mean()
    print(f"Gradientes zeros (batch processing): {grad_zeros_batch*100:.1f}%")
    
    # TESTE 3: Verificar se o problema é acumulação de gradientes
    print(f"\nTESTE 3 - ACUMULACAO DE GRADIENTES:")
    
    temporal_projection_accum = nn.Sequential(
        nn.Linear(input_dim, 192),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(192, d_model)
    )
    
    # Simular múltiplas iterações (como no treinamento real)
    for iteration in range(100):  # Simular 100 steps
        # Zero gradients
        temporal_projection_accum.zero_grad()
        
        # Process timesteps
        temporal_variations_accum = []
        for t in range(seq_len):
            bar_features = temporal_obs[:, t, :]
            projected = temporal_projection_accum(bar_features)
            temporal_variations_accum.append(projected)
        
        result = torch.stack(temporal_variations_accum, dim=1)
        loss_accum = result.sum()
        loss_accum.backward()
        
        # Simulate optimizer step (clamp gradients)
        with torch.no_grad():
            for param in temporal_projection_accum.parameters():
                if param.grad is not None:
                    param.grad.clamp_(-1.0, 1.0)  # Gradient clipping
    
    # Check final gradients after many iterations
    grad_zeros_accum = (temporal_projection_accum[0].weight.grad.abs() < 1e-8).float().mean()
    print(f"Gradientes zeros (apos 100 iteracoes): {grad_zeros_accum*100:.1f}%")
    
    print(f"\nRESUMO:")
    print(f"Loop temporal: {grad_zeros*100:.1f}% zeros")
    print(f"Batch processing: {grad_zeros_batch*100:.1f}% zeros")
    print(f"Apos 100 iteracoes: {grad_zeros_accum*100:.1f}% zeros")
    
    if grad_zeros_accum > 0.6:
        print("PROBLEMA IDENTIFICADO: Degradacao ao longo do treinamento!")
        print("Pode ser gradient clipping ou numerical decay")
    elif grad_zeros > 0.5:
        print("PROBLEMA IDENTIFICADO: Loop temporal causa zeros!")
    else:
        print("Problema ainda nao identificado com estes testes")

if __name__ == "__main__":
    test_temporal_loop_issue()