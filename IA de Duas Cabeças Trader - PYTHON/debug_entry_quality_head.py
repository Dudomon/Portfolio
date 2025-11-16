#!/usr/bin/env python3
"""
🔍 DEBUGGING ENTRY QUALITY HEAD - Investigação específica dos zeros
"""

import sys
sys.path.append("D:/Projeto")

import torch
import torch.nn as nn
from trading_framework.policies.two_head_v8_heritage import OptimizedV8DecisionMaker

def investigate_entry_quality_head():
    print("🔍 INVESTIGANDO ENTRY QUALITY HEAD ZEROS")
    print("=" * 60)
    
    # Criar um OptimizedV8DecisionMaker
    decision_maker = OptimizedV8DecisionMaker(input_dim=256)
    
    print("📊 Estrutura do entry_quality_head:")
    for i, layer in enumerate(decision_maker.entry_quality_head):
        print(f"   {i}: {layer}")
        if hasattr(layer, 'weight'):
            weight = layer.weight
            total_params = weight.numel()
            zero_params = (weight == 0).sum().item()
            zero_pct = (zero_params / total_params) * 100
            print(f"      🎯 Weight shape: {weight.shape}")
            print(f"      🔢 Total params: {total_params:,}")
            print(f"      ⚠️ Zeros: {zero_params:,} ({zero_pct:.1f}%)")
            
            if zero_pct > 5:
                print(f"      🚨 CRÍTICO: {zero_pct:.1f}% zeros detectados!")
                
                # Investigar valores
                print(f"      📈 Min value: {weight.min().item():.6f}")
                print(f"      📈 Max value: {weight.max().item():.6f}")
                print(f"      📈 Mean value: {weight.mean().item():.6f}")
                print(f"      📈 Std value: {weight.std().item():.6f}")
        print()
    
    print("🔬 ANÁLISE ESPECÍFICA DO LAYER 1:")
    layer_1 = decision_maker.entry_quality_head[1]
    print(f"Layer 1 type: {type(layer_1)}")
    print(f"Layer 1: {layer_1}")
    
    if hasattr(layer_1, 'weight'):
        weight_1 = layer_1.weight
        print(f"✅ Layer 1 tem weight: {weight_1.shape}")
        total_1 = weight_1.numel()
        zeros_1 = (weight_1 == 0).sum().item()
        zero_pct_1 = (zeros_1 / total_1) * 100
        print(f"🚨 Layer 1 zeros: {zeros_1}/{total_1} ({zero_pct_1:.1f}%)")
        
        # Valores específicos
        print(f"Min: {weight_1.min():.6f}, Max: {weight_1.max():.6f}")
        print(f"Mean: {weight_1.mean():.6f}, Std: {weight_1.std():.6f}")
        
        # Analisar distribuição
        flat_weights = weight_1.flatten()
        exactly_zero = (flat_weights == 0.0).sum()
        near_zero = (flat_weights.abs() < 1e-6).sum()
        print(f"Exactly zero: {exactly_zero}, Near zero: {near_zero}")
        
    else:
        print("❌ Layer 1 não tem weight!")
    
    print("\n🔬 TESTANDO FORWARD PASS:")
    # Teste com input real
    batch_size = 32
    input_features = torch.randn(batch_size, 256)
    
    print(f"Input shape: {input_features.shape}")
    
    # Forward através de cada layer
    x = input_features
    for i, layer in enumerate(decision_maker.entry_quality_head):
        x_before = x.clone()
        x = layer(x)
        
        # Calcular estatísticas
        x_zeros = (x == 0).sum().item()
        x_total = x.numel()
        x_zero_pct = (x_zeros / x_total) * 100
        
        print(f"Layer {i} ({type(layer).__name__}):")
        print(f"  Output shape: {x.shape}")
        print(f"  Output zeros: {x_zeros}/{x_total} ({x_zero_pct:.1f}%)")
        print(f"  Output range: [{x.min():.4f}, {x.max():.4f}]")
        
        # Se layer tem gradientes
        if hasattr(layer, 'weight') and layer.weight is not None:
            if layer.weight.grad is not None:
                grad_zeros = (layer.weight.grad == 0).sum().item()
                grad_total = layer.weight.grad.numel()
                grad_zero_pct = (grad_zeros / grad_total) * 100
                print(f"  Gradient zeros: {grad_zeros}/{grad_total} ({grad_zero_pct:.1f}%)")
            else:
                print(f"  No gradients yet")
        print()
    
    print("🎯 ANÁLISE FINAL:")
    print(f"Final output shape: {x.shape}")
    final_zeros = (x == 0).sum().item()
    final_total = x.numel()
    final_zero_pct = (final_zeros / final_total) * 100
    print(f"Final output zeros: {final_zeros}/{final_total} ({final_zero_pct:.1f}%)")

if __name__ == "__main__":
    investigate_entry_quality_head()