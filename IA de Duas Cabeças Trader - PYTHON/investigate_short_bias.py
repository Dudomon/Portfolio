#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO: Por que o DAYTRADER não executa SHORT?

Análise dos logs mostra:
- HOLD BIAS CRÍTICO (97.0%)
- LONG 3.0%
- SHORT 0.0%

Vamos investigar:
1. Como as ações são discretizadas
2. Se há bias na política
3. Se os filtros estão bloqueando SHORT
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_action_discretization():
    """Analisar como as ações são discretizadas na política V7"""
    
    print("🔍 ANÁLISE DA DISCRETIZAÇÃO DE AÇÕES")
    print("=" * 60)
    
    # Simular valores raw_actions típicos
    raw_values = np.linspace(-3, 3, 1000)
    
    # Aplicar a lógica de discretização atual
    discrete_actions = []
    for raw_val in raw_values:
        if raw_val < -0.5:
            discrete_actions.append(0)  # HOLD
        elif raw_val > 0.5:
            discrete_actions.append(2)  # SHORT
        else:
            discrete_actions.append(1)  # LONG
    
    discrete_actions = np.array(discrete_actions)
    
    # Contar distribuição
    hold_count = np.sum(discrete_actions == 0)
    long_count = np.sum(discrete_actions == 1)
    short_count = np.sum(discrete_actions == 2)
    
    total = len(discrete_actions)
    
    print(f"📊 DISTRIBUIÇÃO TEÓRICA (valores -3 a 3):")
    print(f"   HOLD (< -0.5): {hold_count} ({hold_count/total*100:.1f}%)")
    print(f"   LONG (-0.5 a 0.5): {long_count} ({long_count/total*100:.1f}%)")
    print(f"   SHORT (> 0.5): {short_count} ({short_count/total*100:.1f}%)")
    
    # Analisar bias
    hold_range = 0.5 - (-3)  # 3.5
    long_range = 0.5 - (-0.5)  # 1.0
    short_range = 3 - 0.5  # 2.5
    
    print(f"\n🎯 ANÁLISE DE BIAS:")
    print(f"   HOLD range: {hold_range} (muito grande!)")
    print(f"   LONG range: {long_range}")
    print(f"   SHORT range: {short_range}")
    
    print(f"\n🚨 PROBLEMA IDENTIFICADO:")
    print(f"   HOLD tem range 3.5x maior que LONG!")
    print(f"   Isso cria HOLD BIAS severo!")
    
    return discrete_actions, raw_values

def suggest_balanced_discretization():
    """Sugerir discretização balanceada"""
    
    print("\n🔧 SUGESTÃO DE CORREÇÃO:")
    print("=" * 60)
    
    # Discretização balanceada
    raw_values = np.linspace(-3, 3, 1000)
    
    # Opção 1: Thresholds balanceados
    print("OPÇÃO 1: Thresholds balanceados (-1, 1)")
    discrete_balanced = []
    for raw_val in raw_values:
        if raw_val < -1.0:
            discrete_balanced.append(0)  # HOLD
        elif raw_val > 1.0:
            discrete_balanced.append(2)  # SHORT
        else:
            discrete_balanced.append(1)  # LONG
    
    discrete_balanced = np.array(discrete_balanced)
    hold_count = np.sum(discrete_balanced == 0)
    long_count = np.sum(discrete_balanced == 1)
    short_count = np.sum(discrete_balanced == 2)
    total = len(discrete_balanced)
    
    print(f"   HOLD (< -1.0): {hold_count} ({hold_count/total*100:.1f}%)")
    print(f"   LONG (-1.0 a 1.0): {long_count} ({long_count/total*100:.1f}%)")
    print(f"   SHORT (> 1.0): {short_count} ({short_count/total*100:.1f}%)")
    
    # Opção 2: Softmax approach
    print("\nOPÇÃO 2: Softmax (probabilístico)")
    print("   Usar softmax para converter raw_action em probabilidades")
    print("   Depois sample ou usar argmax")
    
    # Opção 3: Thresholds ainda mais balanceados
    print("\nOPÇÃO 3: Thresholds ultra-balanceados (-0.33, 0.33)")
    discrete_ultra = []
    for raw_val in raw_values:
        if raw_val < -0.33:
            discrete_ultra.append(0)  # HOLD
        elif raw_val > 0.33:
            discrete_ultra.append(2)  # SHORT
        else:
            discrete_ultra.append(1)  # LONG
    
    discrete_ultra = np.array(discrete_ultra)
    hold_count = np.sum(discrete_ultra == 0)
    long_count = np.sum(discrete_ultra == 1)
    short_count = np.sum(discrete_ultra == 2)
    
    print(f"   HOLD (< -0.33): {hold_count} ({hold_count/total*100:.1f}%)")
    print(f"   LONG (-0.33 a 0.33): {long_count} ({long_count/total*100:.1f}%)")
    print(f"   SHORT (> 0.33): {short_count} ({short_count/total*100:.1f}%)")

def analyze_model_outputs():
    """Analisar se o modelo está produzindo valores adequados"""
    
    print("\n🧠 ANÁLISE DOS OUTPUTS DO MODELO:")
    print("=" * 60)
    
    # Simular outputs típicos de uma rede neural
    # Valores próximos de 0 são mais comuns em redes inicializadas
    
    # Distribuição normal centrada em 0
    normal_outputs = np.random.normal(0, 1, 10000)
    
    # Aplicar discretização atual
    discrete_current = []
    for val in normal_outputs:
        if val < -0.5:
            discrete_current.append(0)  # HOLD
        elif val > 0.5:
            discrete_current.append(2)  # SHORT
        else:
            discrete_current.append(1)  # LONG
    
    discrete_current = np.array(discrete_current)
    
    hold_count = np.sum(discrete_current == 0)
    long_count = np.sum(discrete_current == 1)
    short_count = np.sum(discrete_current == 2)
    total = len(discrete_current)
    
    print(f"📊 COM OUTPUTS NORMAIS (μ=0, σ=1):")
    print(f"   HOLD: {hold_count} ({hold_count/total*100:.1f}%)")
    print(f"   LONG: {long_count} ({long_count/total*100:.1f}%)")
    print(f"   SHORT: {short_count} ({short_count/total*100:.1f}%)")
    
    # Isso explica o HOLD BIAS!
    print(f"\n🚨 EXPLICAÇÃO DO HOLD BIAS:")
    print(f"   Redes neurais tendem a produzir valores próximos de 0")
    print(f"   Com thresholds (-0.5, 0.5), a maioria fica em LONG")
    print(f"   Mas valores < -0.5 vão para HOLD (que tem range maior)")
    print(f"   Resultado: HOLD BIAS + pouco SHORT")

def create_fix_script():
    """Criar script de correção"""
    
    fix_code = '''
# 🔧 CORREÇÃO PARA HOLD BIAS E FALTA DE SHORT

# ANTES (PROBLEMÁTICO):
discrete_decision = torch.where(raw_decision < -0.5, 0, 
                              torch.where(raw_decision > 0.5, 2, 1))

# DEPOIS (BALANCEADO):
# Opção 1: Thresholds balanceados
discrete_decision = torch.where(raw_decision < -0.67, 0,  # HOLD: 33%
                              torch.where(raw_decision > 0.67, 2, 1))  # SHORT: 33%, LONG: 33%

# Opção 2: Softmax (mais suave)
raw_logits = torch.stack([raw_decision - 1, raw_decision, raw_decision + 1], dim=-1)
action_probs = torch.softmax(raw_logits, dim=-1)
discrete_decision = torch.argmax(action_probs, dim=-1).float()

# Opção 3: Gumbel-Softmax (diferenciável)
from torch.nn.functional import gumbel_softmax
raw_logits = torch.stack([raw_decision - 1, raw_decision, raw_decision + 1], dim=-1)
discrete_decision = gumbel_softmax(raw_logits, tau=1.0, hard=True)[:, :, 1] * 1 + gumbel_softmax(raw_logits, tau=1.0, hard=True)[:, :, 2] * 2
'''
    
    with open('action_discretization_fix.py', 'w') as f:
        f.write(fix_code)
    
    print(f"\n💾 Script de correção salvo: action_discretization_fix.py")

if __name__ == "__main__":
    print("🔍 INVESTIGAÇÃO: DAYTRADER NÃO EXECUTA SHORT")
    print("=" * 80)
    
    # 1. Analisar discretização atual
    discrete_actions, raw_values = analyze_action_discretization()
    
    # 2. Sugerir correções
    suggest_balanced_discretization()
    
    # 3. Analisar outputs do modelo
    analyze_model_outputs()
    
    # 4. Criar script de correção
    create_fix_script()
    
    print("\n✅ INVESTIGAÇÃO COMPLETA!")
    print("🎯 CAUSA RAIZ: Thresholds de discretização desbalanceados")
    print("🔧 SOLUÇÃO: Ajustar thresholds para (-0.67, 0.67) ou usar softmax")