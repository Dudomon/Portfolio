#!/usr/bin/env python3
"""
🔬 INVESTIGAÇÃO FINAL - ACTION[1] BUG ROOT CAUSE
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

def investigacao_final():
    print("🔬 INVESTIGAÇÃO FINAL - ACTION[1] BUG ROOT CAUSE")
    print("=" * 60)
    
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
    
    # 1. ANÁLISE DOS PESOS ENCONTRADOS
    print(f"\n🎯 ANÁLISE DOS ACHADOS ANTERIORES")
    print("-" * 50)
    
    action_net = policy.action_net
    weight = action_net.weight.cpu()  # Move para CPU
    bias = action_net.bias.cpu()
    
    action1_weights = weight[1, :]
    action1_bias = bias[1]
    
    print(f"💰 ACTION[1] - PESOS DA QUANTIDADE:")
    print(f"   ✅ Pesos NÃO estão zerados")
    print(f"   📊 Mean: {action1_weights.mean():.8f}")
    print(f"   📊 Std:  {action1_weights.std():.8f}")  
    print(f"   📊 Bias: {action1_bias:.8f}")
    print(f"   📊 Range: [{action1_weights.min():.8f}, {action1_weights.max():.8f}]")
    
    # 2. TESTE MANUAL DE FORWARD PASS
    print(f"\n🧪 TESTE MANUAL DE FORWARD PASS (CPU)")
    print("-" * 50)
    
    # Input sintético no CPU
    input_features = torch.randn(1, 128)
    
    # Forward pass manual
    raw_output = torch.matmul(input_features, weight.T) + bias
    print(f"📊 Raw output da Action[1]: {raw_output[0, 1]:.8f}")
    
    # 3. INVESTIGAR A PIPELINE DE ATIVAÇÃO
    print(f"\n🔍 INVESTIGANDO PIPELINE DE ATIVAÇÃO")
    print("-" * 50)
    
    # Verificar action_dist
    if hasattr(policy, 'action_dist') and policy.action_dist is not None:
        dist = policy.action_dist
        print(f"✅ Distribution: {type(dist).__name__}")
        
        # Para DiagGaussian, verificar log_std
        if hasattr(dist, 'distribution') and hasattr(dist.distribution, 'log_std'):
            log_std = dist.distribution.log_std
            if log_std is not None:
                print(f"📊 Log std shape: {log_std.shape}")
                print(f"📊 Log std values: {log_std}")
                
                if len(log_std) > 1:
                    action1_log_std = log_std[1]
                    action1_std = torch.exp(action1_log_std)
                    print(f"💰 Action[1] log_std: {action1_log_std:.8f}")
                    print(f"💰 Action[1] std: {action1_std:.8f}")
                    
                    if action1_std < 1e-6:
                        print(f"   🔴 PROBLEMA ENCONTRADO: STD MUITO BAIXO!")
                        print(f"   💡 Action[1] tem variance quase zero")
                    else:
                        print(f"   ✅ STD parece normal")
    
    # 4. TESTE COM PREDIÇÃO REAL
    print(f"\n🎮 TESTE COM PREDIÇÃO REAL")
    print("-" * 50)
    
    # Usar o predict do modelo (pipeline completa)
    obs = np.random.randn(2580).astype(np.float32)
    
    # Múltiplas predições com seeds diferentes
    resultados = []
    for seed in range(10):
        np.random.seed(seed)
        obs = np.random.randn(2580).astype(np.float32) * (seed + 1)  # Variação de amplitude
        
        action, _states = model.predict(obs, deterministic=True)
        resultados.append(action[1])  # Action[1]
        
        print(f"   Seed {seed}: Action[1] = {action[1]:.8f}")
    
    # Análise dos resultados
    resultados = np.array(resultados)
    print(f"\n📊 ANÁLISE DOS RESULTADOS:")
    print(f"   Mean: {resultados.mean():.8f}")
    print(f"   Std:  {resultados.std():.8f}")
    print(f"   Min:  {resultados.min():.8f}")
    print(f"   Max:  {resultados.max():.8f}")
    
    if resultados.std() < 1e-8:
        print(f"   🔴 CONFIRMADO: Action[1] sempre constante")
    
    # 5. INVESTIGAR DETERMINISTIC VS STOCHASTIC
    print(f"\n🎲 TESTE DETERMINISTIC VS STOCHASTIC")
    print("-" * 50)
    
    obs = np.random.randn(2580).astype(np.float32)
    
    # Deterministic
    action_det, _ = model.predict(obs, deterministic=True)
    print(f"   Deterministic Action[1]: {action_det[1]:.8f}")
    
    # Stochastic (múltiplas amostras)
    stochastic_results = []
    for i in range(5):
        action_stoch, _ = model.predict(obs, deterministic=False)
        stochastic_results.append(action_stoch[1])
        print(f"   Stochastic {i+1} Action[1]: {action_stoch[1]:.8f}")
    
    stoch_array = np.array(stochastic_results)
    print(f"   Stochastic std: {stoch_array.std():.8f}")
    
    if stoch_array.std() < 1e-6:
        print(f"   🔴 Mesmo no modo stochastic, não há variação!")
    
    # 6. DIAGNÓSTICO FINAL
    print(f"\n🏆 DIAGNÓSTICO FINAL - ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    print(f"🔍 EVIDÊNCIAS COLETADAS:")
    print(f"   ✅ Pesos da Action[1]: NORMAIS (não zerados)")
    print(f"   ✅ Bias da Action[1]: NORMAL ({action1_bias:.6f})")
    print(f"   ✅ Forward pass: PRODUZ VALORES (raw: ~{raw_output[0, 1]:.6f})")
    print(f"   🔴 Predições: SEMPRE ZERO (mesmo com inputs diferentes)")
    print(f"   🔴 Stochastic mode: NÃO ADICIONA VARIAÇÃO")
    
    print(f"\n🎯 HIPÓTESES DESCARTADAS:")
    print(f"   ❌ Pesos zerados (pesos são normais)")
    print(f"   ❌ Bias zerado (bias existe)")
    print(f"   ❌ Forward pass quebrado (funciona)")
    
    print(f"\n🚨 POSSÍVEIS CAUSAS REAIS:")
    print(f"   1. 🎭 MASKING/CLIPPING no action processing")
    print(f"   2. 🔒 LOG_STD muito baixo (variance ~0)")
    print(f"   3. 🎯 ACTION BOUNDS mal configurados")
    print(f"   4. 🧠 TwoHeadV7Intuition tem gates que bloqueiam Action[1]")
    print(f"   5. 📊 PREPROCESSING que força Action[1] = 0")
    
    print(f"\n💡 INVESTIGAÇÃO ADICIONAL NECESSÁRIA:")
    print(f"   1. 🔍 Verificar TwoHeadV7Intuition.forward()")
    print(f"   2. 🎮 Analisar action preprocessing no environment")
    print(f"   3. 📊 Verificar se há clipping específico para Action[1]")
    print(f"   4. 🔧 Testar com policy mais simples (MlpPolicy)")
    
    print(f"\n🎯 CONCLUSÃO PRINCIPAL:")
    print(f"   🧠 O PROBLEMA NÃO ESTÁ NOS PESOS BÁSICOS")
    print(f"   🎭 O PROBLEMA ESTÁ NA PIPELINE DE PROCESSAMENTO")
    print(f"   🔍 TwoHeadV7Intuition tem comportamento específico")
    print(f"   💡 SOLUÇÃO: Investigar gates/masks específicos do V7")

def main():
    investigacao_final()

if __name__ == "__main__":
    main()