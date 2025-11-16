#!/usr/bin/env python3
"""
🔬 DEBUG V7 DIRETO - Acesso direto aos raw_actions
Investigar por que Action[1] sempre retorna zero
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

def debug_v7_direct():
    print("🔬 DEBUG V7 DIRETO - Investigação Action[1]")
    print("=" * 50)
    
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
    
    # 1. ACESSAR DIRETAMENTE O ACTOR_HEAD
    print(f"\n🔍 ANÁLISE DIRETA DO ACTOR_HEAD")
    print("-" * 50)
    
    if hasattr(policy, 'actor_head'):
        actor_head = policy.actor_head
        print(f"✅ Actor head encontrado: {type(actor_head)}")
        
        # Verificar layers
        layers = list(actor_head.children())
        print(f"   Layers: {len(layers)}")
        for i, layer in enumerate(layers):
            print(f"     Layer {i}: {layer}")
        
        # Pegos da última layer (Linear)
        final_layer = layers[-1] if layers else None
        if final_layer and hasattr(final_layer, 'weight'):
            weight = final_layer.weight  # [11, input_dim]
            bias = final_layer.bias      # [11]
            
            print(f"\n💰 ANÁLISE DOS PESOS DA ACTION[1]:")
            action1_weight = weight[1, :]  # Pesos da Action[1]
            action1_bias = bias[1]         # Bias da Action[1]
            
            print(f"   Weight shape: {action1_weight.shape}")
            print(f"   Weight mean: {action1_weight.mean():.8f}")
            print(f"   Weight std:  {action1_weight.std():.8f}")
            print(f"   Weight min:  {action1_weight.min():.8f}")
            print(f"   Weight max:  {action1_weight.max():.8f}")
            print(f"   Bias:        {action1_bias:.8f}")
            
            # Verificar se há problema óbvio
            if action1_bias < -5.0:
                print(f"   🔴 BIAS MUITO NEGATIVO: {action1_bias:.3f} força sigmoid → 0")
            elif torch.all(action1_weight == 0):
                print(f"   🔴 PESOS ZERADOS: Todos os pesos são zero")
            elif action1_weight.std() < 1e-7:
                print(f"   🟡 PESOS CONSTANTES: Std muito baixo")
            else:
                print(f"   ✅ Pesos parecem normais")
    
    # 2. TESTE DIRETO COM INPUTS SINTÉTICOS
    print(f"\n🧪 TESTE DIRETO COM O ACTOR_HEAD")
    print("-" * 50)
    
    try:
        # Descobrir input size do actor_head
        actor_input_size = None
        if hasattr(policy, 'actor_head'):
            first_layer = list(policy.actor_head.children())[0]
            if hasattr(first_layer, 'in_features'):
                actor_input_size = first_layer.in_features
                print(f"📊 Actor head input size: {actor_input_size}")
        
        if actor_input_size:
            # Testar com diferentes inputs
            test_inputs = [
                torch.zeros(1, actor_input_size),           # Zero
                torch.ones(1, actor_input_size) * 0.5,     # Positivo pequeno
                torch.ones(1, actor_input_size) * 2.0,     # Positivo grande
                -torch.ones(1, actor_input_size) * 2.0,    # Negativo
                torch.randn(1, actor_input_size) * 0.1,    # Random pequeno
                torch.randn(1, actor_input_size) * 2.0     # Random grande
            ]
            
            test_names = ["Zeros", "Pos_Small", "Pos_Big", "Negative", "Rand_Small", "Rand_Big"]
            
            print(f"\n🔍 TESTANDO RAW OUTPUTS:")
            raw_action1_values = []
            
            for i, (name, input_tensor) in enumerate(zip(test_names, test_inputs)):
                with torch.no_grad():
                    raw_output = policy.actor_head(input_tensor)
                    raw_action1 = float(raw_output[0, 1])
                    sigmoid_result = float(torch.sigmoid(raw_output[0, 1]))
                    
                    raw_action1_values.append(raw_action1)
                    
                    print(f"   {name:10s}: raw[1]={raw_action1:+8.4f} → sigmoid={sigmoid_result:.6f}")
            
            # Análise dos resultados
            raw_array = np.array(raw_action1_values)
            
            print(f"\n📊 ANÁLISE DOS RAW VALUES:")
            print(f"   Mean: {raw_array.mean():+.6f}")
            print(f"   Std:  {raw_array.std():.6f}")
            print(f"   Min:  {raw_array.min():+.6f}")
            print(f"   Max:  {raw_array.max():+.6f}")
            
            # Diagnóstico
            if raw_array.max() < -5.0:
                print(f"   🔴 TODOS MUITO NEGATIVOS: sigmoid sempre ~0")
            elif raw_array.max() < -2.0:
                print(f"   🟡 TENDÊNCIA NEGATIVA: sigmoid < 0.12")
            elif raw_array.std() < 0.1:
                print(f"   🟡 BAIXA VARIAÇÃO: Range limitado")
            else:
                print(f"   ✅ Valores normais")
    
    except Exception as e:
        print(f"❌ Erro no teste direto: {e}")
    
    # 3. COMPARAR COM PREDIÇÃO REAL DO MODELO
    print(f"\n🎯 COMPARAÇÃO COM PREDIÇÃO REAL")
    print("-" * 50)
    
    # Teste com observação real
    obs = np.random.randn(2580).astype(np.float32)
    
    try:
        # Predição normal
        action, _states = model.predict(obs, deterministic=True)
        print(f"📊 Predição do modelo:")
        print(f"   Action[0]: {action[0]:.6f}")
        print(f"   Action[1]: {action[1]:.6f}")
        print(f"   Action[2]: {action[2]:.6f}")
        
        # Múltiplas predições para verificar consistência
        action1_values = []
        for i in range(10):
            obs_var = np.random.randn(2580).astype(np.float32) * (i + 1)
            action_var, _ = model.predict(obs_var, deterministic=True)
            action1_values.append(action_var[1])
        
        action1_array = np.array(action1_values)
        print(f"\n📊 Múltiplas predições Action[1]:")
        print(f"   Values: {[f'{v:.6f}' for v in action1_values]}")
        print(f"   Mean: {action1_array.mean():.6f}")
        print(f"   Std:  {action1_array.std():.6f}")
        
        if action1_array.std() < 1e-6:
            print(f"   🔴 CONFIRMADO: Action[1] sempre constante")
        else:
            print(f"   ✅ Action[1] varia normalmente")
    
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
    
    # 4. INVESTIGAR A PIPELINE COMPLETA
    print(f"\n🔍 INVESTIGAÇÃO DA PIPELINE COMPLETA")
    print("-" * 50)
    
    try:
        # Verificar se podemos acessar componentes internos
        if hasattr(policy, 'forward_actor'):
            print(f"✅ forward_actor disponível")
        
        if hasattr(policy, 'features_extractor'):
            print(f"✅ features_extractor disponível")
        
        if hasattr(policy, 'mlp_extractor'):
            print(f"✅ mlp_extractor disponível")
        
        if hasattr(policy, 'action_dist'):
            print(f"✅ action_dist disponível: {type(policy.action_dist)}")
        
        # Verificar se há log_std específico
        if hasattr(policy, 'log_std'):
            log_std = policy.log_std
            print(f"📊 Log std: {log_std}")
            if len(log_std) > 1:
                print(f"   Action[1] log_std: {log_std[1]:.6f}")
                print(f"   Action[1] std: {torch.exp(log_std[1]):.6f}")
    
    except Exception as e:
        print(f"❌ Erro na investigação da pipeline: {e}")
    
    # 5. CONCLUSÃO DO DEBUG
    print(f"\n🏆 CONCLUSÃO DO DEBUG DIRETO")
    print("=" * 50)
    
    print(f"🎯 DESCOBERTAS:")
    print(f"   1. Policy é TwoHeadV7Intuition")
    print(f"   2. Actor head tem estrutura normal")
    
    if 'raw_array' in locals():
        if raw_array.max() < -3:
            print(f"   3. 🔴 RAW VALUES muito negativos: {raw_array.mean():.3f}")
            print(f"      💡 CAUSA: Bias negativo ou weights inadequados")
            print(f"      🎯 EFEITO: sigmoid({raw_array.mean():.3f}) = {1/(1+np.exp(-raw_array.mean())):.6f}")
        else:
            print(f"   3. ✅ Raw values normais: {raw_array.mean():.3f}")
    
    if 'action1_array' in locals():
        if action1_array.std() < 1e-6:
            print(f"   4. 🔴 CONFIRMADO: Action[1] sempre {action1_array.mean():.6f}")
        else:
            print(f"   4. ✅ Action[1] varia: std={action1_array.std():.6f}")
    
    print(f"\n💡 PRÓXIMO PASSO:")
    if 'raw_array' in locals() and raw_array.max() < -3:
        print(f"   🔧 AJUSTAR BIAS: Adicionar +3 ao bias da Action[1]")
        print(f"   🔄 OU RE-TREINAR: Com inicialização melhor")
    else:
        print(f"   🔍 INVESTIGAR: Outros componentes da pipeline")

if __name__ == "__main__":
    debug_v7_direct()