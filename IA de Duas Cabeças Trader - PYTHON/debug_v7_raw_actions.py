#!/usr/bin/env python3
"""
🔬 DEBUG V7 RAW ACTIONS - Investigar raw_actions[:, 1] antes do sigmoid
Descobrir por que Action[1] sempre retorna zero
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

def debug_v7_raw_actions():
    print("🔬 DEBUG V7 RAW ACTIONS - Action[1] Investigation")
    print("=" * 60)
    
    # Carregar modelo
    checkpoint_path = projeto_path / "trading_framework/training/checkpoints/DAYTRADER/checkpoint_phase2riskmanagement_650000_steps_20250805_201935.zip"
    
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(checkpoint_path)
        print(f"✅ Modelo carregado: {model.num_timesteps:,} steps")
        print(f"🧠 Policy: {type(model.policy).__name__}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    policy = model.policy
    
    # Verificar se é TwoHeadV7Intuition
    if not hasattr(policy, 'forward_actor'):
        print(f"❌ Não é TwoHeadV7Intuition policy")
        return
    
    print(f"✅ TwoHeadV7Intuition policy confirmada")
    
    # 1. HOOK para capturar raw_actions
    print(f"\n🔍 INSTALANDO HOOK PARA CAPTURAR RAW_ACTIONS")
    print("-" * 50)
    
    captured_raw_actions = []
    
    def hook_raw_actions(module, input, output):
        """Hook para capturar raw_actions antes das ativações"""
        if hasattr(policy, '_debug_raw_actions'):
            captured_raw_actions.append(policy._debug_raw_actions.clone().detach())
    
    # Instalar hook no actor_head (última layer)
    if hasattr(policy, 'actor_head'):
        hook_handle = policy.actor_head.register_forward_hook(hook_raw_actions)
        print(f"✅ Hook instalado no actor_head")
    
    # 2. MONKEY PATCH para capturar raw_actions
    original_forward_actor = policy.forward_actor
    
    def debug_forward_actor(self, features, lstm_state, deterministic=True):
        # Chamar forward_actor original mas capturar raw_actions
        
        # Extrair contexto
        context = features
        
        # LSTM actor com attention
        if lstm_state is not None:
            lstm_features, new_lstm_state = self.actor_lstm(context.unsqueeze(0), lstm_state)
            lstm_features = lstm_features.squeeze(0)  # Remove sequence dim
        else:
            lstm_features, new_lstm_state = self.actor_lstm(context.unsqueeze(0))
            lstm_features = lstm_features.squeeze(0)
        
        # Multi-head attention
        attn_features, attn_weights = self.actor_attention(
            lstm_features.unsqueeze(0), lstm_features.unsqueeze(0), lstm_features.unsqueeze(0)
        )
        attn_features = attn_features.squeeze(0)
        
        # Combine LSTM e attention
        combined_features = torch.cat([lstm_features, attn_features], dim=-1)
        combined_features = self.actor_combine(combined_features)
        
        # Gerar raw actions (PRÉ-ATIVAÇÃO)
        raw_actions = self.actor_head(combined_features)
        
        # 🔬 CAPTURAR RAW_ACTIONS PARA DEBUG
        self._debug_raw_actions = raw_actions.clone().detach()
        
        # Continue com o processamento normal...
        actions = torch.zeros_like(raw_actions)
        
        # [0] decision: 0/1/2
        raw_decision = raw_actions[:, 0]
        discrete_decision = torch.where(raw_decision < -0.5, 0,
                                      torch.where(raw_decision > 0.5, 2, 1))
        actions[:, 0] = discrete_decision.float()
        
        # [1] entry_confidence: 0-1 - sigmoid  <-- AQUI É O PROBLEMA
        actions[:, 1] = torch.sigmoid(raw_actions[:, 1])
        
        # Resto das ações...
        actions[:, 2] = torch.tanh(raw_actions[:, 2])
        actions[:, 3] = torch.sigmoid(raw_actions[:, 3])
        actions[:, 4] = torch.tanh(raw_actions[:, 4])
        
        for i in range(5, 11):
            actions[:, i] = torch.tanh(raw_actions[:, i]) * 3.0
        
        return actions, new_lstm_state, {}
    
    # Aplicar monkey patch
    policy.forward_actor = debug_forward_actor.__get__(policy, type(policy))
    print(f"✅ Monkey patch aplicado ao forward_actor")
    
    # 3. TESTAR COM MÚLTIPLAS OBSERVAÇÕES
    print(f"\n🧪 TESTANDO COM MÚLTIPLAS OBSERVAÇÕES")
    print("-" * 50)
    
    raw_action1_values = []
    final_action1_values = []
    
    for test_id in range(20):
        # Observação variada
        if test_id < 5:
            obs = np.random.randn(2580).astype(np.float32) * 0.1  # Pequena
        elif test_id < 10:
            obs = np.random.randn(2580).astype(np.float32) * 2.0   # Grande
        elif test_id < 15:
            obs = np.ones(2580, dtype=np.float32) * (test_id - 10)  # Constante variada
        else:
            obs = np.zeros(2580, dtype=np.float32)  # Zero
        
        try:
            # Predição com captura de raw_actions
            captured_raw_actions.clear()
            action, _state = model.predict(obs, deterministic=True)
            
            # Extrair valores capturados
            if captured_raw_actions:
                raw_action1 = float(captured_raw_actions[-1][0, 1])  # [batch, action_dim]
                raw_action1_values.append(raw_action1)
                
                final_action1 = float(action[1])
                final_action1_values.append(final_action1)
                
                sigmoid_expected = float(torch.sigmoid(torch.tensor(raw_action1)))
                
                print(f"Test {test_id:2d}: raw[1]={raw_action1:+8.4f} → sigmoid={sigmoid_expected:.6f} → final={final_action1:.6f}")
            else:
                print(f"Test {test_id:2d}: ❌ Raw actions não capturadas")
        
        except Exception as e:
            print(f"Test {test_id:2d}: ❌ Erro: {e}")
    
    # 4. ANÁLISE DOS RESULTADOS
    print(f"\n📊 ANÁLISE DOS RAW_ACTIONS[1]")
    print("=" * 60)
    
    if raw_action1_values:
        raw_array = np.array(raw_action1_values)
        final_array = np.array(final_action1_values)
        
        print(f"🎯 RAW ACTIONS[:, 1] (PRÉ-SIGMOID):")
        print(f"   Count: {len(raw_array)}")
        print(f"   Mean: {raw_array.mean():+.6f}")
        print(f"   Std:  {raw_array.std():.6f}")
        print(f"   Min:  {raw_array.min():+.6f}")
        print(f"   Max:  {raw_array.max():+.6f}")
        
        print(f"\n🎯 FINAL ACTIONS[1] (PÓS-SIGMOID):")
        print(f"   Mean: {final_array.mean():.6f}")
        print(f"   Std:  {final_array.std():.6f}")
        print(f"   Min:  {final_array.min():.6f}")
        print(f"   Max:  {final_array.max():.6f}")
        
        # Diagnóstico específico
        print(f"\n🔍 DIAGNÓSTICO:")
        
        if raw_array.max() < -5.0:
            print(f"   🔴 PROBLEMA CRÍTICO: Todos raw_actions[1] < -5")
            print(f"   💡 CAUSA: sigmoid(x < -5) ≈ 0.007 (quase zero)")
            print(f"   🎯 SOLUÇÃO: Actor head produz valores muito negativos")
            
        elif raw_array.max() < -2.0:
            print(f"   🟡 PROBLEMA: Raw actions muito negativos")
            print(f"   💡 CAUSA: sigmoid(x < -2) < 0.12")
            print(f"   🎯 BIAS: Actor head tem bias negativo forte")
            
        elif raw_array.std() < 0.01:
            print(f"   🟡 PROBLEMA: Raw actions constantes")
            print(f"   💡 CAUSA: Actor head não varia output")
            print(f"   🎯 DEAD NEURONS: Possível morte de neurônios")
            
        else:
            print(f"   ✅ Raw actions normais, problema em outro lugar")
        
        # Verificar correlação
        expected_sigmoid = np.array([1.0 / (1.0 + np.exp(-x)) for x in raw_array])
        correlation = np.corrcoef(expected_sigmoid, final_array)[0, 1]
        print(f"   📊 Correlação sigmoid esperado vs final: {correlation:.6f}")
        
        if correlation < 0.99:
            print(f"   🔴 PROBLEMA: Sigmoid não está sendo aplicado corretamente")
        
        # 5. ANÁLISE DOS PESOS DO ACTOR HEAD
        print(f"\n🔍 ANÁLISE DOS PESOS DO ACTOR_HEAD")
        print("-" * 50)
        
        if hasattr(policy, 'actor_head'):
            # Última layer do actor_head
            final_layer = policy.actor_head[-1]  # Linear layer
            if hasattr(final_layer, 'weight') and hasattr(final_layer, 'bias'):
                weight = final_layer.weight  # [11, input_dim]
                bias = final_layer.bias      # [11]
                
                action1_weight = weight[1, :]  # Pesos da Action[1]
                action1_bias = bias[1]         # Bias da Action[1]
                
                print(f"💰 ACTOR HEAD - ACTION[1] WEIGHTS:")
                print(f"   Weight mean: {action1_weight.mean():.6f}")
                print(f"   Weight std:  {action1_weight.std():.6f}")
                print(f"   Weight min:  {action1_weight.min():.6f}")
                print(f"   Weight max:  {action1_weight.max():.6f}")
                print(f"   Bias:        {action1_bias:.6f}")
                
                # Diagnóstico dos pesos
                if action1_bias < -5.0:
                    print(f"   🔴 BIAS MUITO NEGATIVO: Força sigmoid → 0")
                elif abs(action1_weight.mean()) > 2.0:
                    print(f"   🟡 PESOS EXTREMOS: Podem causar saturação")
                elif action1_weight.std() < 0.01:
                    print(f"   🟡 PESOS CONSTANTES: Não há variação")
                else:
                    print(f"   ✅ Pesos parecem normais")
        
    else:
        print(f"❌ Nenhum raw_action capturado")
    
    # Cleanup
    try:
        hook_handle.remove()
    except:
        pass
    
    # 6. CONCLUSÃO
    print(f"\n🏆 CONCLUSÃO DO DEBUG V7")
    print("=" * 60)
    
    if raw_action1_values:
        mean_raw = np.mean(raw_action1_values)
        
        if mean_raw < -5:
            print(f"🔴 ROOT CAUSE ENCONTRADA: Actor head produz raw_actions[1] muito negativos")
            print(f"   Valor médio: {mean_raw:.3f}")
            print(f"   sigmoid({mean_raw:.3f}) = {1/(1+np.exp(-mean_raw)):.6f}")
            print(f"   💡 SOLUÇÃO: Re-treinar actor head ou ajustar bias")
            
        elif mean_raw < -2:
            print(f"🟡 PROBLEMA IDENTIFICADO: Raw actions negativos demais")
            print(f"   💡 AJUSTE: Bias positivo no actor head")
            
        else:
            print(f"❓ Raw actions normais, problema em outro componente")
    
    print(f"\n💡 PRÓXIMAS AÇÕES:")
    print(f"   1. Verificar inicialização do actor_head")
    print(f"   2. Analisar gradientes durante treinamento")
    print(f"   3. Considerar bias adjustment")
    print(f"   4. Testar com policy mais simples")

if __name__ == "__main__":
    debug_v7_raw_actions()