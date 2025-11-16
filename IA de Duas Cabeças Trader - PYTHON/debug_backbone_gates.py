#!/usr/bin/env python3
"""
🔍 DEBUG BACKBONE GATES - Investigar saturação dos gates críticos
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from datetime import datetime

CHECKPOINT_PATH = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase3noisehandlingfixed_4750000_steps_20250813_124443.zip"

def analyze_backbone_gates():
    """Analisar saturação específica dos backbone gates"""
    
    print("🧠 ANÁLISE BACKBONE GATES - RISCO SATURAÇÃO")
    print("=" * 60)
    
    try:
        # Carregar modelo
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        policy = model.policy
        
        if not hasattr(policy, 'unified_backbone'):
            print("❌ Unified backbone não encontrado")
            return
        
        backbone = policy.unified_backbone
        
        # Hook para capturar gates
        gate_activations = {}
        
        def create_gate_hook(gate_name):
            def hook(module, input, output):
                gate_activations[gate_name] = output.detach().cpu().numpy()
            return hook
        
        # Registrar hooks nos gates
        actor_hook = backbone.actor_gate.register_forward_hook(create_gate_hook('actor_gate'))
        critic_hook = backbone.critic_gate.register_forward_hook(create_gate_hook('critic_gate'))
        
        print("🔍 Coletando ativações dos gates...")
        
        # Executar múltiplas predições
        lstm_states = None
        for i in range(100):  # Menor quantidade para teste rápido
            obs = np.random.normal(0, 1.0, (2580,)).astype(np.float32)
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
        
        # Remover hooks
        actor_hook.remove()
        critic_hook.remove()
        
        # Analisar ativações
        print("\n📊 ANÁLISE DAS ATIVAÇÕES DOS GATES:")
        
        for gate_name, activations in gate_activations.items():
            if len(activations) > 0:
                # Flatten all activations
                flat_acts = activations.flatten()
                
                mean_act = np.mean(flat_acts)
                std_act = np.std(flat_acts)
                min_act = np.min(flat_acts)
                max_act = np.max(flat_acts)
                
                # Contar saturação
                near_zero = np.sum(flat_acts < 0.1)
                near_one = np.sum(flat_acts > 0.9)
                total = len(flat_acts)
                saturation_pct = (near_zero + near_one) / total * 100
                
                print(f"\n🔍 {gate_name.upper()}:")
                print(f"   📊 Stats: μ={mean_act:.3f} σ={std_act:.3f} range=[{min_act:.3f}, {max_act:.3f}]")
                print(f"   🚨 Saturação: {saturation_pct:.1f}% (zeros: {near_zero}, ones: {near_one})")
                
                # Classificar risco
                if saturation_pct > 80:
                    risk = "🔥 CRÍTICO"
                elif saturation_pct > 60:
                    risk = "⚠️ ALTO"
                elif saturation_pct > 40:
                    risk = "🟡 MODERADO"
                else:
                    risk = "✅ BAIXO"
                
                print(f"   🎯 Risco: {risk}")
        
        print(f"\n💡 RECOMENDAÇÃO:")
        total_gates_analyzed = len([g for g in gate_activations.keys() if len(gate_activations[g]) > 0])
        if total_gates_analyzed > 0:
            print(f"   📊 {total_gates_analyzed} gates backbone analisados")
            print(f"   ⏰ Em treinamento longo (10M+ steps), gates podem saturar")
            print(f"   🔧 Considerar aplicar fix tanh preventivamente")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    analyze_backbone_gates()