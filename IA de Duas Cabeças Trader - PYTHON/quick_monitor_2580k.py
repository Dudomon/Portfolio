#!/usr/bin/env python3
"""
🔍 MONITOR RÁPIDO - Checkpoint 2.58M Steps
Análise específica do checkpoint 2580000 para verificar saturação
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

# Checkpoint específico
CHECKPOINT_PATH = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/FINAL_phase1fundamentalsextended_2580000_steps_20250813_221317.zip"

def analyze_checkpoint_2580k():
    """Análise rápida do checkpoint 2.58M"""
    
    print("🔍 MONITOR SATURAÇÃO - Checkpoint 2.58M")
    print("=" * 60)
    
    try:
        # Carregar modelo
        print(f"📂 Carregando: {os.path.basename(CHECKPOINT_PATH)}")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        model.policy.set_training_mode(False)
        
        # Estatísticas básicas do modelo
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        
        print(f"📊 Total params: {total_params:,}")
        print(f"📊 Trainable params: {trainable_params:,}")
        
        # Análise de saturação
        print("\n🎯 ANÁLISE DE SATURAÇÃO:")
        
        # Test predictions
        lstm_states = None
        activations = []
        actions_taken = []
        
        for i in range(100):  # 100 samples rápidos
            obs = np.random.normal(0, 1.0, (2580,)).astype(np.float32)
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            actions_taken.append(action)
            
            if i % 20 == 0:
                print(f"   Sample {i}: action[0]={action[0]:.4f}")
        
        actions_array = np.array(actions_taken)
        
        # Análise das ações
        print(f"\n📈 ESTATÍSTICAS AÇÕES (100 samples):")
        print(f"   Entry decisions (action[0]):")
        print(f"     Mean: {actions_array[:, 0].mean():.4f}")
        print(f"     Std: {actions_array[:, 0].std():.4f}")
        print(f"     Min: {actions_array[:, 0].min():.4f}")
        print(f"     Max: {actions_array[:, 0].max():.4f}")
        
        # Verificar range das ações
        entry_range = actions_array[:, 0].max() - actions_array[:, 0].min()
        print(f"     Range: {entry_range:.4f}")
        
        if entry_range < 0.1:
            print("   ⚠️ BAIXA VARIABILIDADE - Possível saturação")
        elif entry_range < 0.5:
            print("   📊 VARIABILIDADE MODERADA")
        else:
            print("   ✅ BOA VARIABILIDADE")
        
        # Verificar concentração em extremos
        extreme_low = (actions_array[:, 0] < 0.1).sum()
        extreme_high = (actions_array[:, 0] > 0.9).sum()
        
        print(f"   Concentração extremos:")
        print(f"     < 0.1: {extreme_low}% ({extreme_low}/100)")
        print(f"     > 0.9: {extreme_high}% ({extreme_high}/100)")
        
        if extreme_low > 80 or extreme_high > 80:
            print("   🔥 ALTA CONCENTRAÇÃO EM EXTREMOS - Saturação detectada")
        elif extreme_low + extreme_high > 60:
            print("   ⚠️ CONCENTRAÇÃO MODERADA EM EXTREMOS")
        else:
            print("   ✅ DISTRIBUIÇÃO SAUDÁVEL")
        
        # Análise de layers específicos do V7
        print(f"\n🧠 ANÁLISE LAYERS V7:")
        
        # Tentar acessar layers específicos
        try:
            policy = model.policy
            print(f"   Policy type: {type(policy).__name__}")
            
            # Verificar se tem mlp_extractor
            if hasattr(policy, 'mlp_extractor'):
                extractor = policy.mlp_extractor
                print(f"   Extractor type: {type(extractor).__name__}")
                
                # Verificar layers
                if hasattr(extractor, 'shared_net'):
                    print(f"   Shared net layers: {len(extractor.shared_net)}")
                
            # Verificar action_net
            if hasattr(policy, 'action_net'):
                print(f"   Action net: {type(policy.action_net).__name__}")
        
        except Exception as e:
            print(f"   ⚠️ Erro acessando layers: {e}")
        
        print(f"\n✅ ANÁLISE CONCLUÍDA - Checkpoint 2.58M")
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_checkpoint_2580k()
    print(f"\nStatus: {'✅ Sucesso' if success else '❌ Falha'}")