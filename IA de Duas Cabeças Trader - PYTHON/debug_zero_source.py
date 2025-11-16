"""
🔍 DEBUG ZERO SOURCE - Descobrir ONDE os zeros começam
Sistema para rastrear exatamente qual componente inicia os zeros
"""

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

def analyze_zero_progression():
    """Analisa a progressão dos zeros no seu log"""
    
    print("🔍 ANÁLISE DA PROGRESSÃO DE ZEROS")
    print("=" * 60)
    
    # Dados do seu log
    zero_components = [
        ("features_extractor.temporal_projection.weight", 54.3),
        ("features_extractor.transformer_layer.self_attn.in_proj_bias", 33.3),
        ("lstm_actor.weight_ih_l0", 48.4),
        ("lstm_actor.weight_hh_l0", 61.3),
        ("lstm_actor.bias_ih_l0", 42.1),
        ("lstm_actor.bias_hh_l0", 42.1),
    ]
    
    print("📊 COMPONENTES COM ZEROS:")
    for component, percentage in zero_components:
        risk_level = "🚨 CRÍTICO" if percentage > 50 else "⚠️  ALTO" if percentage > 30 else "📊 NORMAL"
        print(f"  {risk_level} {component}: {percentage}%")
    
    print(f"\n🎯 ANÁLISE POR CAMADA:")
    
    # Transformer components
    transformer_zeros = [c for c in zero_components if "features_extractor" in c[0]]
    print(f"\n🔧 TRANSFORMER EXTRACTOR:")
    for component, percentage in transformer_zeros:
        print(f"  - {component.split('.')[-1]}: {percentage}%")
        
    # LSTM components  
    lstm_zeros = [c for c in zero_components if "lstm_actor" in c[0]]
    print(f"\n🧠 LSTM ACTOR:")
    for component, percentage in lstm_zeros:
        print(f"  - {component.split('.')[-1]}: {percentage}%")
    
    print(f"\n💡 HIPÓTESES SOBRE A ORIGEM:")
    
    # Análise das hipóteses
    max_transformer = max([p for c, p in transformer_zeros])
    max_lstm = max([p for c, p in lstm_zeros])
    
    if max_transformer > max_lstm:
        print(f"  🎯 HIPÓTESE 1: Transformer é a origem (max: {max_transformer}%)")
        print(f"    - temporal_projection está explodindo primeiro")
        print(f"    - Gradientes propagam para LSTM downstream")
        print(f"    - AÇÃO: Investigar transformer_extractor.py")
    else:
        print(f"  🎯 HIPÓTESE 2: LSTM é a origem (max: {max_lstm}%)")
        print(f"    - LSTM saturando por inputs extremos")
        print(f"    - Gradientes propagam para transformer upstream")
        print(f"    - AÇÃO: Investigar LSTM initialization ou inputs")
    
    if "weight_hh_l0" in [c[0].split('.')[-1] for c, p in lstm_zeros if p > 60]:
        print(f"  🚨 HIPÓTESE 3: LSTM recurrent weights problem")
        print(f"    - weight_hh_l0 (recurrent) > 60% zeros")
        print(f"    - Gradient vanishing/exploding em sequências")
        print(f"    - AÇÃO: Verificar sequence length e gradient clipping")
    
    print(f"\n🔍 PADRÃO DE DISTRIBUIÇÃO DE AÇÕES:")
    print(f"  - HOLD: 88.7% (MUITO ALTO - indica paralisia)")
    print(f"  - LONG: 7.5% (baixo)")  
    print(f"  - SHORT: 0.1% (quase zero)")
    print(f"  💡 ISSO SUGERE: Network está 'travando' nas ações seguras")
    
    print(f"\n🧠 ESTRATÉGIAS DE DEBUG:")
    print(f"  1. 🔧 REDUZIR transformer monitor frequencies mais ainda")
    print(f"  2. 🎯 VERIFICAR se gradient clipping está muito baixo")
    print(f"  3. 📉 TESTAR com reward system V2 temporariamente")
    print(f"  4. ⚡ AUMENTAR learning rate para compensar zeros")
    print(f"  5. 🔄 REINICIALIZAR pesos das camadas problemáticas")
    
    print(f"\n🎯 PRIORIDADE DE INVESTIGAÇÃO:")
    print(f"  1. features_extractor.temporal_projection.weight (54.3%)")
    print(f"  2. lstm_actor.weight_hh_l0 (61.3%)")
    print(f"  3. Interação entre transformer → LSTM")
    
    return {
        'transformer_max': max_transformer,
        'lstm_max': max_lstm,
        'primary_suspect': 'transformer' if max_transformer > max_lstm else 'lstm'
    }

def suggest_fixes():
    """Sugere correções específicas baseadas na análise"""
    
    print(f"\n🛠️ CORREÇÕES ESPECÍFICAS RECOMENDADAS:")
    print("=" * 60)
    
    print(f"\n1. 🎯 TEMPORAL_PROJECTION (54.3% zeros):")
    print(f"   - Reduzir xavier gain de 0.6 para 0.3")
    print(f"   - Adicionar gradient clipping específico")
    print(f"   - Verificar input normalization")
    
    print(f"\n2. 🧠 LSTM WEIGHTS (61.3% zeros):")
    print(f"   - Aumentar forget gate bias para 2.0")
    print(f"   - Verificar se sequence está muito longa")
    print(f"   - Considerar gradient clipping por camada")
    
    print(f"\n3. ⚡ LEARNING RATE:")
    print(f"   - Atual: 3e-05 pode estar MUITO baixo para recovery")
    print(f"   - Testar: 6e-05 temporariamente")
    print(f"   - Ou usar learning rate warm-up")
    
    print(f"\n4. 🔄 RESET STRATEGY:")
    print(f"   - Re-initialize só as camadas problemáticas")
    print(f"   - Manter pesos que estão funcionando")
    print(f"   - Gradient accumulation para estabilizar")

if __name__ == "__main__":
    results = analyze_zero_progression()
    suggest_fixes()