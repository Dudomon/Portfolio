#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO: MELHORIAS PARA O CRÍTICO - EXPLAINED VARIANCE
Baseado nos problemas encontrados no reward system, investigar soluções
"""
import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch

def investigate_critic_improvements():
    """Investigar possíveis melhorias para o explained variance do crítico"""
    
    print("🔍 INVESTIGAÇÃO: MELHORIAS PARA O CRÍTICO")
    print("=" * 60)
    
    print("\n📊 PROBLEMAS IDENTIFICADOS NO REWARD SYSTEM:")
    print("   ❌ Correlação muito baixa (-0.01) entre reward e portfolio change")
    print("   ⚠️ Alta autocorrelação (0.96) - rewards muito relacionados")
    print("   ⚠️ 95% HOLD actions - modelo muito conservador")
    print("   ⚠️ Mean reward negativo (-0.42)")
    
    print("\n💡 HIPÓTESES PARA EXPLAINED VARIANCE RUIM:")
    print("=" * 60)
    
    print("\n🎯 HIPÓTESE 1: REWARD SYSTEM INCONSISTENTE")
    print("-" * 50)
    print("   🔍 Problema: Rewards não refletem performance real")
    print("   💊 Solução: Reformular reward para ser mais aligned")
    print("   📝 Implementação:")
    print("      - Usar mais componente de PnL real vs artificial")
    print("      - Reduzir peso de penalties abstratas")
    print("      - Aumentar reward por trades bem-sucedidos")
    
    print("\n🎯 HIPÓTESE 2: LEARNING RATE DO CRÍTICO MUITO BAIXO") 
    print("-" * 50)
    print("   🔍 Problema: Crítico aprende muito devagar")
    print("   💊 Solução: LR diferencial ainda mais agressivo")
    print("   📝 Implementação:")
    print("      - Critic LR: 5-8x maior que actor")
    print("      - Warm-up mais longo para crítico")
    print("      - Update frequency diferente")
    
    print("\n🎯 HIPÓTESE 3: ARQUITETURA DO CRÍTICO LIMITADA")
    print("-" * 50)
    print("   🔍 Problema: MLP pode não capturar complexidade temporal")
    print("   💊 Solução: Híbrido MLP + Attention ou LSTM leve")
    print("   📝 Implementação:")
    print("      - Adicionar camada de atenção ao crítico")
    print("      - LSTM shallow para crítico (1 layer)")
    print("      - Residual connections")
    
    print("\n🎯 HIPÓTESE 4: VALUE CLIPPING MUITO RESTRITIVO")
    print("-" * 50)
    print("   🔍 Problema: Gradientes do crítico sendo clipped demais")
    print("   💊 Solução: Relaxar clipping específico do crítico")
    print("   📝 Implementação:")
    print("      - Critic-specific clip range")
    print("      - Adaptive clipping baseado na variância")
    print("      - Gradient norm específico para crítico")
    
    print("\n🎯 HIPÓTESE 5: NORMALIZATION DOS REWARDS INADEQUADA")
    print("-" * 50)
    print("   🔍 Problema: VecNormalize pode estar mascarando signal")
    print("   💊 Solução: Normalization customizada ou desabilitada")
    print("   📝 Implementação:")
    print("      - Reward scaling manual")
    print("      - Whitening específico para rewards")
    print("      - Running statistics mais conservadoras")
    
    print("\n🎯 HIPÓTESE 6: BATCH SIZE INADEQUADO PARA CRÍTICO")
    print("-" * 50)
    print("   🔍 Problema: Batch muito pequeno para estimar value")
    print("   💊 Solução: Batch size maior ou mini-batches para crítico")
    print("   📝 Implementação:")
    print("      - Critic-specific batch size")
    print("      - Multiple critic updates per policy update")
    print("      - Experience replay buffer para crítico")
    
    # TESTE PRÁTICO: VERIFICAR GRADIENTES ATUAIS
    print("\n🔬 TESTE PRÁTICO: VERIFICAR ESTADO ATUAL")
    print("=" * 60)
    
    try:
        from sb3_contrib import RecurrentPPO
        
        # Carregar checkpoint atual
        checkpoint_path = "./Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase3noisehandlingfixed_4800000_steps_20250814_111420.zip"
        
        if os.path.exists(checkpoint_path):
            print(f"📂 Carregando: {os.path.basename(checkpoint_path)}")
            
            model = RecurrentPPO.load(checkpoint_path)
            policy = model.policy
            
            # Verificar learning rates atuais
            if hasattr(model, 'learning_rate'):
                current_lr = model.learning_rate
                print(f"   📊 Learning Rate atual: {current_lr}")
            
            # Verificar critic architecture
            if hasattr(policy, 'v7_critic_mlp'):
                critic = policy.v7_critic_mlp
                total_params = sum(p.numel() for p in critic.parameters())
                print(f"   🧠 Crítico MLP params: {total_params:,}")
                
                # Verificar estrutura
                print(f"   📐 Critic architecture: {critic}")
            
            # Verificar hyperparameters
            if hasattr(model, 'clip_range'):
                print(f"   ✂️ Clip range: {model.clip_range}")
            if hasattr(model, 'vf_coef'):
                print(f"   ⚖️ Value function coef: {model.vf_coef}")
            if hasattr(model, 'max_grad_norm'):
                print(f"   📏 Max grad norm: {model.max_grad_norm}")
                
        else:
            print("   ⚠️ Checkpoint não encontrado para análise")
            
    except Exception as e:
        print(f"   ❌ Erro na análise: {e}")
    
    # RECOMENDAÇÕES PRIORITÁRIAS
    print("\n🏆 RECOMENDAÇÕES PRIORITÁRIAS")
    print("=" * 60)
    
    recommendations = [
        {
            'priority': 1,
            'title': 'Fix Reward System Correlation',
            'description': 'Reformular reward para correlacionar melhor com performance real',
            'implementation': 'Aumentar peso de PnL real, reduzir penalties abstratas',
            'effort': 'MEDIUM'
        },
        {
            'priority': 2, 
            'title': 'Critic Learning Rate Boost',
            'description': 'Aumentar LR do crítico para 6-8x o do actor',
            'implementation': 'Dynamic LR Manager com ratios mais agressivos',
            'effort': 'LOW'
        },
        {
            'priority': 3,
            'title': 'Reward Normalization Review', 
            'description': 'Revisar se VecNormalize está prejudicando o signal',
            'implementation': 'Testar com reward scaling manual',
            'effort': 'LOW'
        },
        {
            'priority': 4,
            'title': 'Critic Architecture Enhancement',
            'description': 'Adicionar capacidade temporal limitada ao crítico',
            'implementation': 'Shallow LSTM ou attention layer',
            'effort': 'HIGH'
        },
        {
            'priority': 5,
            'title': 'Value Function Clipping Adjustment',
            'description': 'Relaxar clipping específico do crítico', 
            'implementation': 'Critic-specific hyperparameters',
            'effort': 'LOW'
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec['title']} [EFFORT: {rec['effort']}]")
        print(f"      📋 {rec['description']}")
        print(f"      🔧 {rec['implementation']}")
    
    print("\n💡 PRÓXIMOS PASSOS SUGERIDOS:")
    print("   1. Implementar fix #1 (reward correlation) e #2 (critic LR)")
    print("   2. Testar com reward normalization desabilitada (#3)")
    print("   3. Se ainda não resolver, considerar #4 (architecture)")
    print("   4. Monitorar explained variance a cada 500k steps")
    
    return recommendations

if __name__ == "__main__":
    recommendations = investigate_critic_improvements()
    print(f"\n✅ INVESTIGAÇÃO CONCLUÍDA - {len(recommendations)} recomendações identificadas")