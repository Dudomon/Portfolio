#!/usr/bin/env python3
"""
ANALISADOR DE DADOS REAIS DE GRADIENTES
Analisa os dados capturados do treinamento real
"""

import torch
import torch.nn as nn
import pickle
import glob
import numpy as np
import os

def analyze_real_gradient_data():
    """Analisa os dados reais capturados do treinamento"""
    
    print("=" * 80)
    print("ANALISE DE DADOS REAIS DE GRADIENTES")
    print("=" * 80)
    
    # Procurar arquivos de debug
    debug_files = glob.glob("gradient_debug_data/*.pkl")
    
    if not debug_files:
        print("NENHUM arquivo de debug encontrado!")
        print("Execute o treinamento primeiro para capturar dados reais.")
        return
    
    print(f"Encontrados {len(debug_files)} arquivos de debug")
    
    for debug_file in debug_files:
        print(f"\n=== ANALISANDO: {os.path.basename(debug_file)} ===")
        
        try:
            with open(debug_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'bar_features' in data:
                # Dados de features mortas
                analyze_dead_features_data(data)
            elif 'weight_grad' in data:
                # Dados de gradientes interceptados
                analyze_intercepted_gradient_data(data)
            else:
                print("Tipo de dados desconhecido!")
                
        except Exception as e:
            print(f"Erro ao ler {debug_file}: {e}")


def analyze_dead_features_data(data):
    """Analisa dados de features mortas"""
    
    bar_features = data['bar_features']
    zero_features = data['always_zero_features']
    input_stats = data['input_stats']
    
    print(f"DADOS DE FEATURES MORTAS:")
    print(f"  Step: {data['step']}")
    print(f"  Shape: {bar_features.shape}")
    print(f"  Features sempre zero: {len(zero_features)}")
    print(f"  Input stats: {input_stats}")
    
    # Análise detalhada das features zero
    print(f"\nANALISE DAS FEATURES ZERO:")
    print(f"  Indices das primeiras 20 features mortas: {zero_features[:20]}")
    
    # Verificar se há padrão nos índices
    zero_indices = np.array(zero_features)
    if len(zero_indices) > 1:
        gaps = np.diff(zero_indices)
        print(f"  Gaps entre indices: min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.1f}")
        
        # Verificar se são múltiplos de algum número
        for divisor in [2, 3, 4, 5, 8, 10, 16, 20]:
            multiples = sum(1 for idx in zero_indices if idx % divisor == 0)
            if multiples > len(zero_indices) * 0.5:
                print(f"  PADRÃO DETECTADO: {multiples}/{len(zero_indices)} indices são múltiplos de {divisor}")
    
    # Testar impacto simulado
    print(f"\nTESTE DE IMPACTO SIMULADO:")
    test_model = nn.Linear(bar_features.shape[1], 192)
    nn.init.xavier_uniform_(test_model.weight)
    nn.init.zeros_(test_model.bias)
    
    # Forward com dados reais
    test_model.zero_grad()
    bar_features.requires_grad_(True)
    
    output = test_model(bar_features)
    loss = output.sum()
    loss.backward()
    
    grad_zeros = (test_model.weight.grad.abs() < 1e-8).float().mean().item()
    print(f"  Grad zeros com dados reais: {grad_zeros*100:.1f}%")
    
    # Comparar com dados sintéticos
    synthetic_input = torch.randn_like(bar_features)
    synthetic_input[:, zero_features] = 0  # Aplicar mesmo padrão de zeros
    
    test_model.zero_grad()
    synthetic_input.requires_grad_(True)
    
    output_synth = test_model(synthetic_input)
    loss_synth = output_synth.sum()
    loss_synth.backward()
    
    grad_zeros_synth = (test_model.weight.grad.abs() < 1e-8).float().mean().item()
    print(f"  Grad zeros com sintético (mesmo padrão): {grad_zeros_synth*100:.1f}%")
    
    # Teste sem features mortas
    clean_input = torch.randn_like(bar_features)
    
    test_model.zero_grad()
    clean_input.requires_grad_(True)
    
    output_clean = test_model(clean_input)
    loss_clean = output_clean.sum()
    loss_clean.backward()
    
    grad_zeros_clean = (test_model.weight.grad.abs() < 1e-8).float().mean().item()
    print(f"  Grad zeros com input limpo: {grad_zeros_clean*100:.1f}%")
    
    # Conclusão
    if grad_zeros > 0.5:
        print(f"\n❌ CONFIRMADO: Dados reais causam {grad_zeros*100:.1f}% grad zeros")
        if grad_zeros_synth > 0.4:
            print("   CAUSA: Features sempre zeradas")
        else:
            print("   CAUSA: Algo além das features zeradas")
    else:
        print(f"\n✅ Dados reais não reproduziram o problema")


def analyze_intercepted_gradient_data(data):
    """Analisa dados de gradientes interceptados"""
    
    print(f"DADOS DE GRADIENTES INTERCEPTADOS:")
    print(f"  Step: {data['step']}")
    print(f"  Timestamp: {data['timestamp']}")
    
    stats = data['stats']
    print(f"  Grad zeros: {stats['grad_zeros_percent']:.1f}%")
    print(f"  Grad stats: mean={stats['grad_mean']:.6f}, std={stats['grad_std']:.6f}")
    
    weight_grad = data['weight_grad']
    
    # Análise de padrões estruturais
    print(f"\nANALISE DE PADROES:")
    
    # Por linha (neurônios de saída)
    row_zeros = (weight_grad.abs() < 1e-8).float().mean(dim=1)
    print(f"  Neurônios com >50% grad zeros: {(row_zeros > 0.5).sum().item()}/{len(row_zeros)}")
    
    # Por coluna (features de entrada)
    col_zeros = (weight_grad.abs() < 1e-8).float().mean(dim=0)
    print(f"  Features com >50% grad zeros: {(col_zeros > 0.5).sum().item()}/{len(col_zeros)}")
    
    # Neurônios completamente mortos
    dead_neurons = (row_zeros > 0.99).sum().item()
    dead_features = (col_zeros > 0.99).sum().item()
    
    if dead_neurons > 0:
        print(f"  ❌ NEURÔNIOS MORTOS: {dead_neurons}")
        dead_indices = torch.where(row_zeros > 0.99)[0].tolist()
        print(f"     Indices: {dead_indices[:10]}")
    
    if dead_features > 0:
        print(f"  ❌ FEATURES MORTAS: {dead_features}")
        dead_feat_indices = torch.where(col_zeros > 0.99)[0].tolist()
        print(f"     Indices: {dead_feat_indices[:10]}")


if __name__ == "__main__":
    analyze_real_gradient_data()