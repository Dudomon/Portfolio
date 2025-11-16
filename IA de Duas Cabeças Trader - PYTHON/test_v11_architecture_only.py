#!/usr/bin/env python3
"""
🧪 TESTE ARQUITETURA V11 HÍBRIDA - FOCO NA POLICY
Teste simplificado focado apenas na arquitetura LSTM+GRU
"""

import sys
import os
sys.path.append("D:/Projeto")

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import traceback

# Import da V11 Sigmoid
from trading_framework.policies.two_head_v11_sigmoid import (
    TwoHeadV11Sigmoid, 
    get_v8_elegance_kwargs
)

def test_v11_architecture():
    """🏗️ Teste direto da arquitetura V11 Híbrida"""
    print("🧪 TESTE ARQUITETURA V11 HÍBRIDA LSTM+GRU")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    try:
        # Simular observation_space e action_space
        from gym.spaces import Box
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(450,), dtype=np.float32)
        action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        print(f"📊 Observation Space: {observation_space.shape}")
        print(f"🎯 Action Space: {action_space.shape}")
        
        # Criar policy V11 diretamente
        kwargs = get_v8_elegance_kwargs()
        
        # Adicionar lr_schedule obrigatório
        def lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV11Sigmoid(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )
        
        policy = policy.to(device)
        print("✅ Policy V11 criada com sucesso")
        
        # Verificar componentes híbridos
        components = {
            'v8_shared_lstm': hasattr(policy, 'v8_shared_lstm'),
            'v11_shared_gru': hasattr(policy, 'v11_shared_gru'),
            'hybrid_fusion': hasattr(policy, 'hybrid_fusion'),
            'market_context': hasattr(policy, 'market_context'),
            'entry_head': hasattr(policy, 'entry_head'),
            'management_head': hasattr(policy, 'management_head'),
            'memory_bank': hasattr(policy, 'memory_bank'),
            'v8_critic': hasattr(policy, 'v8_critic')
        }
        
        print("\n🔍 COMPONENTES DETECTADOS:")
        for comp, present in components.items():
            status = "✅" if present else "❌"
            print(f"{status} {comp}")
        
        # Contar parâmetros
        if components['v8_shared_lstm']:
            lstm_params = sum(p.numel() for p in policy.v8_shared_lstm.parameters())
            print(f"\n🧠 LSTM: {lstm_params:,} parâmetros")
        
        if components['v11_shared_gru']:
            gru_params = sum(p.numel() for p in policy.v11_shared_gru.parameters())
            print(f"⚡ GRU: {gru_params:,} parâmetros")
            print("🔥 ARQUITETURA HÍBRIDA CONFIRMADA!")
        
        if components['hybrid_fusion']:
            fusion_params = sum(p.numel() for p in policy.hybrid_fusion.parameters())
            print(f"🔗 Fusão: {fusion_params:,} parâmetros")
        
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"📊 TOTAL: {total_params:,} parâmetros")
        
        # Teste de forward pass
        print("\n🔄 TESTE FORWARD PASS:")
        batch_size = 2
        obs = torch.randn(batch_size, 450, device=device)
        
        # Inicializar estados LSTM manualmente
        hidden_size = 256  # v8_lstm_hidden
        lstm_states = (
            torch.zeros(1, batch_size, hidden_size, device=device),  # hidden
            torch.zeros(1, batch_size, hidden_size, device=device)   # cell
        )
        episode_starts = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            # Forward actor
            actions, new_lstm_states, gate_info = policy.forward_actor(obs, lstm_states, episode_starts)
            print(f"✅ Actions shape: {actions.shape}")
            print(f"📊 Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
            
            # Forward critic
            values, _ = policy.forward_critic(obs, lstm_states, episode_starts)
            print(f"✅ Values shape: {values.shape}")
            print(f"📊 Values range: [{values.min():.3f}, {values.max():.3f}]")
        
        # Teste de gradientes
        print("\n📈 TESTE GRADIENTES:")
        policy.train()
        
        # Novo forward pass com gradientes habilitados
        obs_grad = torch.randn(batch_size, 450, device=device, requires_grad=True)
        actions_grad, _, _ = policy.forward_actor(obs_grad, lstm_states, episode_starts)
        values_grad, _ = policy.forward_critic(obs_grad, lstm_states, episode_starts)
        
        # Simular loss e backward
        fake_loss = actions_grad.mean() + values_grad.mean()
        fake_loss.backward()
        
        # Verificar gradientes
        grad_components = {}
        
        # LSTM gradientes
        for name, param in policy.v8_shared_lstm.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_components[f'lstm_{name}'] = grad_norm
        
        # GRU gradientes
        for name, param in policy.v11_shared_gru.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_components[f'gru_{name}'] = grad_norm
        
        # Fusion gradientes
        for i, layer in enumerate(policy.hybrid_fusion):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                grad_components[f'fusion_{i}'] = grad_norm
        
        healthy_grads = sum(1 for norm in grad_components.values() if norm > 1e-6)
        total_grads = len(grad_components)
        
        print(f"✅ Gradientes saudáveis: {healthy_grads}/{total_grads}")
        
        for name, norm in list(grad_components.items())[:5]:  # Mostrar apenas 5
            status = "✅" if norm > 1e-6 else "⚠️"
            print(f"{status} {name}: {norm:.2e}")
        
        # Limpar gradientes
        policy.zero_grad()
        
        # Resultado final
        if all(components.values()) and healthy_grads > total_grads * 0.7:
            print("\n🎉 TESTE APROVADO!")
            print("✅ V11 Híbrida LSTM+GRU funcionando perfeitamente")
            print("🚀 Pronta para pré-treino!")
            return True
        else:
            print("\n⚠️ TESTE PARCIAL")
            print("🔧 Alguns componentes precisam de revisão")
            return False
            
    except Exception as e:
        print(f"\n❌ TESTE FALHOU: {e}")
        traceback.print_exc()
        return False

def main():
    """🚀 Função principal"""
    success = test_v11_architecture()
    
    if success:
        print("\n" + "="*50)
        print("🎉 V11 HÍBRIDA APROVADA!")
        print("🔥 LSTM (longo prazo) + GRU (reatividade) + Fusão Neural")
        print("🚀 Sistema pronto para treinamento!")
        return 0
    else:
        print("\n" + "="*50)
        print("❌ V11 HÍBRIDA COM PROBLEMAS")
        print("🔧 Revisar implementação antes do treino")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n⏸️ Teste finalizado (código: {exit_code})")
    exit(exit_code)