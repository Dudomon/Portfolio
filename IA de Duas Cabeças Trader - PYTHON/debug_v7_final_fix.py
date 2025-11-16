#!/usr/bin/env python3
"""
🔬 DEBUG V7 FINAL - Root cause definitivo
DESCOBERTA: Action[1] sempre 0.000000 mas std normal (0.803109)
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

def debug_v7_final():
    print("🔬 DEBUG V7 FINAL - ROOT CAUSE DEFINITIVO")
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
    print(f"🧠 Policy: {type(policy).__name__}")
    
    # 1. DADOS JÁ CONHECIDOS
    print(f"\n📊 DADOS CONHECIDOS (do debug anterior):")
    print("-" * 50)
    print(f"✅ Actor head: 7 layers, input=320, output=11")
    print(f"✅ Action[1] weights: mean=-0.121, std=1.282, bias=0.652")
    print(f"✅ Action[1] log_std: -0.219 → std=0.803 (NORMAL!)")
    print(f"🔴 Action[1] output: SEMPRE 0.000000")
    print(f"📋 Distribution: DiagGaussianDistribution")
    
    # 2. TESTE CORRIGIDO (CUDA-aware)
    print(f"\n🧪 TESTE CORRIGIDO COM CUDA")
    print("-" * 50)
    
    try:
        # Descobrir device do modelo
        device = next(policy.parameters()).device
        print(f"📍 Device do modelo: {device}")
        
        # Input size para actor_head
        actor_input_size = 320  # Já sabemos do debug anterior
        
        # Testar inputs no device correto
        test_inputs = [
            torch.zeros(1, actor_input_size, device=device),
            torch.ones(1, actor_input_size, device=device) * 2.0,
            -torch.ones(1, actor_input_size, device=device) * 2.0,
            torch.randn(1, actor_input_size, device=device) * 0.5,
            torch.randn(1, actor_input_size, device=device) * 3.0
        ]
        
        test_names = ["Zeros", "Pos_Big", "Negative", "Rand_Small", "Rand_Big"]
        
        print(f"🔍 TESTANDO RAW OUTPUTS (CUDA):")
        raw_action1_values = []
        
        with torch.no_grad():
            for name, input_tensor in zip(test_names, test_inputs):
                raw_output = policy.actor_head(input_tensor)
                raw_action1 = float(raw_output[0, 1])
                sigmoid_result = float(torch.sigmoid(raw_output[0, 1]))
                
                raw_action1_values.append(raw_action1)
                print(f"   {name:10s}: raw[1]={raw_action1:+8.4f} → sigmoid={sigmoid_result:.6f}")
        
        # Análise crítica
        raw_array = np.array(raw_action1_values)
        print(f"\n📊 ANÁLISE CRÍTICA DOS RAW VALUES:")
        print(f"   Mean: {raw_array.mean():+.6f}")
        print(f"   Std:  {raw_array.std():.6f}")
        print(f"   Min:  {raw_array.min():+.6f}")
        print(f"   Max:  {raw_array.max():+.6f}")
        
        # DIAGNÓSTICO DEFINITIVO
        if raw_array.min() < -10:
            print(f"   🔴 SATURAÇÃO NEGATIVA: Valores < -10 fazem sigmoid ≈ 0")
            print(f"   💡 CAUSA: Actor head produz extremos negativos")
        elif raw_array.max() > raw_array.min() + 0.1:
            print(f"   ✅ RANGE NORMAL: Raw values variam ({raw_array.min():.2f} a {raw_array.max():.2f})")
        else:
            print(f"   🟡 RANGE LIMITADO: Pouca variação")
            
    except Exception as e:
        print(f"❌ Erro no teste CUDA: {e}")
    
    # 3. INVESTIGAÇÃO DA DiagGaussianDistribution
    print(f"\n🔍 INVESTIGAÇÃO DA DIAGGAUSSIANDISTRIBUTION")
    print("-" * 60)
    
    try:
        # Acessar a distribution diretamente
        action_dist = policy.action_dist
        print(f"📊 Distribution type: {type(action_dist)}")
        
        if hasattr(action_dist, 'distribution'):
            gauss_dist = action_dist.distribution
            print(f"📊 Gaussian distribution: {type(gauss_dist)}")
            
            # Verificar parâmetros
            if hasattr(gauss_dist, 'loc'):
                loc = gauss_dist.loc
                print(f"📊 Loc (mean): {loc}")
                if len(loc) > 1:
                    print(f"   Action[1] mean: {loc[1]:.6f}")
            
            if hasattr(gauss_dist, 'scale'):
                scale = gauss_dist.scale
                print(f"📊 Scale (std): {scale}")
                if len(scale) > 1:
                    print(f"   Action[1] std: {scale[1]:.6f}")
        
        # Testar sample direto
        print(f"\n🎲 TESTE DE SAMPLING DIRETO:")
        obs = np.random.randn(2580).astype(np.float32)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Features extraction
        with torch.no_grad():
            features = policy.extract_features(obs_tensor)
            print(f"   Features shape: {features.shape}")
            
            # Actor forward (se disponível)
            if hasattr(policy, 'forward_actor'):
                print(f"   Tentando forward_actor...")
                # Isso pode dar erro dependendo da implementação
        
    except Exception as e:
        print(f"❌ Erro na investigação da distribution: {e}")
    
    # 4. TESTE FINAL COM MULTIPLE SEEDS
    print(f"\n🎯 TESTE FINAL - MÚLTIPLOS SEEDS")
    print("-" * 50)
    
    results = []
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        np.random.seed(seed)
        obs = np.random.randn(2580).astype(np.float32) * 2.0
        
        # Teste determinístico
        action_det, _ = model.predict(obs, deterministic=True)
        
        # Teste estocástico
        action_stoch, _ = model.predict(obs, deterministic=False)
        
        results.append({
            'seed': seed,
            'det_action1': action_det[1],
            'stoch_action1': action_stoch[1],
            'det_action0': action_det[0],
            'stoch_action0': action_stoch[0]
        })
        
        print(f"   Seed {seed:3d}: Det={action_det[1]:.6f}, Stoch={action_stoch[1]:.6f}")
    
    # Análise dos resultados
    det_values = [r['det_action1'] for r in results]
    stoch_values = [r['stoch_action1'] for r in results]
    
    det_array = np.array(det_values)
    stoch_array = np.array(stoch_values)
    
    print(f"\n📊 ANÁLISE FINAL DOS RESULTADOS:")
    print(f"   Deterministic Action[1]:")
    print(f"     Values: {det_values}")
    print(f"     Std: {det_array.std():.8f}")
    
    print(f"   Stochastic Action[1]:")
    print(f"     Values: {[f'{v:.6f}' for v in stoch_values]}")
    print(f"     Std: {stoch_array.std():.8f}")
    
    # 5. ROOT CAUSE ANALYSIS
    print(f"\n🏆 ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    print(f"🔍 EVIDÊNCIAS COLETADAS:")
    print(f"   ✅ Pesos da Action[1]: NORMAIS (mean=-0.121, std=1.282)")
    print(f"   ✅ Bias da Action[1]: NORMAL (0.652)")
    print(f"   ✅ Log_std da Action[1]: NORMAL (-0.219 → std=0.803)")
    print(f"   ✅ Distribution: DiagGaussianDistribution (correto)")
    print(f"   🔴 Output: SEMPRE 0.000000 (determinístico E estocástico)")
    
    print(f"\n🎯 HIPÓTESES PRINCIPAIS:")
    
    # Hipótese baseada nos raw values
    if 'raw_array' in locals():
        if raw_array.min() < -5:
            print(f"   1. 🔴 RAW VALUES SATURADOS: Actor head produz valores < -5")
            print(f"      sigmoid({raw_array.min():.1f}) = {1/(1+np.exp(-raw_array.min())):.6f}")
            print(f"      💡 CAUSA: Inicialização ou treinamento inadequado")
            
        elif abs(raw_array.mean()) > 3:
            print(f"   1. 🟡 RAW VALUES EXTREMOS: Média={raw_array.mean():.2f}")
            print(f"      💡 CAUSA: Bias muito forte ou features inadequadas")
            
        else:
            print(f"   1. ✅ RAW VALUES: Parecem normais")
    
    print(f"   2. 🔍 POSSÍVEIS CAUSAS ALTERNATIVAS:")
    print(f"      - Clipping/masking específico para Action[1]")
    print(f"      - Preprocessing que força Action[1] = 0")  
    print(f"      - Bug na implementação do TwoHeadV7Intuition")
    print(f"      - Environment action space mal configurado")
    print(f"      - Normalization que zera Action[1]")
    
    print(f"\n💡 AÇÃO RECOMENDADA:")
    if 'raw_array' in locals() and raw_array.min() < -5:
        print(f"   🔧 AJUSTE IMEDIATO: Adicionar bias positivo (+5) à Action[1]")
        print(f"   🔄 LONGO PRAZO: Re-treinar com inicialização adequada")
    else:
        print(f"   🔍 INVESTIGAÇÃO PROFUNDA: Analisar TwoHeadV7Intuition.forward_actor")
        print(f"   🧪 TESTE COMPARATIVO: Testar com MlpPolicy simples")
    
    print(f"\n🎯 STATUS PARA USO EM PRODUÇÃO:")
    print(f"   ✅ Action[0] (direção): FUNCIONA")
    print(f"   🔴 Action[1] (quantidade): SEMPRE 0")
    print(f"   💡 SOLUÇÃO: Implementar position sizing manual")

if __name__ == "__main__":
    debug_v7_final()