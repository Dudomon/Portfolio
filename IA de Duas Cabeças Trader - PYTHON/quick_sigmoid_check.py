#!/usr/bin/env python3
"""
🔍 QUICK SIGMOID CHECK - Verificação rápida da situação atual
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

def quick_sigmoid_status():
    """Verificação rápida do status dos sigmoids"""
    
    print("🔍 QUICK SIGMOID CHECK - 6.2M CHECKPOINT")
    print("=" * 50)
    
    try:
        # Carregar modelo
        model = RecurrentPPO.load("D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase1fundamentalsextended_450000_steps_20250813_173043.zip", device='cuda')
        policy = model.policy
        print("✅ Modelo carregado")
        
        # Teste rápido - 50 predições
        print("\n📊 Testando 50 predições...")
        entry_qualities = []
        
        lstm_states = None
        for i in range(50):
            obs = np.random.normal(0, 1.0, (2580,)).astype(np.float32)
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            
            if len(action) >= 2:
                entry_qualities.append(float(action[1]))
        
        # Análise rápida
        eq_array = np.array(entry_qualities)
        near_zero = np.sum(eq_array < 0.1)
        near_one = np.sum(eq_array > 0.9)
        extremes_pct = (near_zero + near_one) / len(eq_array) * 100
        
        print(f"\n🎯 ENTRY QUALITY RESULTS:")
        print(f"   📊 Média: {np.mean(eq_array):.3f}")
        print(f"   📊 Desvio: {np.std(eq_array):.3f}")
        print(f"   🚨 Extremos: {extremes_pct:.1f}% (0s: {near_zero}, 1s: {near_one})")
        
        # Verificar arquitetura
        print(f"\n🔍 ARQUITETURA ATUAL:")
        
        # Contar sigmoids
        sigmoid_count = 0
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Sigmoid):
                sigmoid_count += 1
        
        print(f"   📊 Total de sigmoids: {sigmoid_count}")
        
        # Verificar se backbone usa tanh
        backbone = policy.unified_backbone
        if hasattr(backbone, 'actor_gate'):
            gate_layers = list(backbone.actor_gate.children())
            last_activation = gate_layers[-1].__class__.__name__
            print(f"   ✅ Backbone actor_gate: {last_activation}")
        
        if hasattr(backbone, 'critic_gate'):
            gate_layers = list(backbone.critic_gate.children())
            last_activation = gate_layers[-1].__class__.__name__ 
            print(f"   ✅ Backbone critic_gate: {last_activation}")
        
        # DIAGNÓSTICO RÁPIDO
        print(f"\n💡 DIAGNÓSTICO RÁPIDO:")
        
        if extremes_pct > 95:
            print(f"   🔥 SATURAÇÃO CRÍTICA: {extremes_pct:.1f}% extremos")
            print(f"   💡 Entry Quality ainda saturada apesar do fix tanh")
            print(f"   💡 Problema: Pesos herdados de checkpoint já saturado")
        elif extremes_pct > 80:
            print(f"   ⚠️ SATURAÇÃO ALTA: {extremes_pct:.1f}% extremos")
            print(f"   💡 Melhoria lenta - tanh fix funcionando gradualmente")
        else:
            print(f"   ✅ SATURAÇÃO CONTROLADA: {extremes_pct:.1f}% extremos")
            print(f"   💡 Tanh fix funcionando bem")
        
        # RECOMENDAÇÃO FINAL
        print(f"\n🎯 RECOMENDAÇÃO:")
        
        if sigmoid_count > 12 and extremes_pct > 95:
            print(f"   🔥 SITUAÇÃO CRÍTICA:")
            print(f"   • {sigmoid_count} sigmoids internos ainda presentes")
            print(f"   • {extremes_pct:.1f}% saturação na Entry Quality")
            print(f"   • Tanh fix aplicado mas pesos saturados persistem")
            print(f"   ")
            print(f"   💡 AÇÕES RECOMENDADAS:")
            print(f"   1. SUBSTITUIR todos os sigmoids internos por tanh")
            print(f"   2. REINICIAR treinamento do zero")
            print(f"   3. Ou: Reset completo dos pesos + LR muito baixo")
        else:
            print(f"   ✅ Situação controlável - continuar monitoramento")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    quick_sigmoid_status()