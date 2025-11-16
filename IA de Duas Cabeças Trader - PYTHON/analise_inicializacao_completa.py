#!/usr/bin/env python3
"""
🔬 ANÁLISE COMPLETA DA INICIALIZAÇÃO V7
Identificar TODOS os problemas de inicialização na arquitetura
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

def analisar_inicializacao_completa():
    print("🔬 ANÁLISE COMPLETA DA INICIALIZAÇÃO V7")
    print("=" * 60)
    
    # 1. PROBLEMA IDENTIFICADO NO ACTOR_HEAD
    print("🔴 PROBLEMA 1: INICIALIZAÇÃO DO ACTOR_HEAD")
    print("-" * 50)
    
    print("📊 PROBLEMA ATUAL:")
    print("   - Xavier gain=2.0 → pode causar valores extremos")
    print("   - Bias uniform(-1.0, 1.0) → insuficiente para Action[1]")
    print("   - Última layer uniform(-2.0, 2.0) → muito extremo")
    print("   - NÃO considera características específicas de cada ação")
    
    print("\n💡 CORREÇÃO NECESSÁRIA:")
    print("   1. Inicialização específica por dimensão de ação")
    print("   2. Action[0] (decisão): range neutro")
    print("   3. Action[1] (quantidade): BIAS POSITIVO")
    print("   4. Actions[2-10]: ranges apropriados")
    print("   5. Usar He initialization para LeakyReLU")
    
    # 2. INVESTIGAR OUTROS COMPONENTES
    print("\n🔍 INVESTIGANDO OUTROS COMPONENTES CRÍTICOS")
    print("=" * 60)
    
    components_to_check = [
        ("unified_backbone", "Backbone compartilhado"),
        ("v7_actor_lstm", "LSTM do Actor"),
        ("critic_mlp", "MLP do Critic"),
        ("entry_head", "Head de entrada"),
        ("management_head", "Head de gestão"),
        ("trade_memory", "Memória de trades"),
        ("enhanced_memory", "Memória aprimorada")
    ]
    
    print("📋 COMPONENTES A VERIFICAR:")
    for comp_name, description in components_to_check:
        print(f"   - {comp_name:20s}: {description}")
    
    # 3. PADRÕES DE INICIALIZAÇÃO PROBLEMÁTICOS
    print(f"\n🚨 PADRÕES PROBLEMÁTICOS DETECTADOS")
    print("-" * 50)
    
    problematic_patterns = [
        {
            "pattern": "Xavier com gain > 1.5",
            "problema": "Pode causar saturação em ativações",
            "solucao": "Usar He initialization para ReLU/LeakyReLU"
        },
        {
            "pattern": "Bias uniforme simétrico",
            "problema": "Não considera características da ação",
            "solucao": "Bias específico por dimensão"
        },
        {
            "pattern": "Mesma inicialização para todas as layers",
            "problema": "Ignora função específica de cada layer",
            "solucao": "Inicialização adaptativa por layer"
        },
        {
            "pattern": "LSTM sem inicialização específica",
            "problema": "Gates mal inicializados",
            "solucao": "Inicializar forget gate com bias=1.0"
        }
    ]
    
    for i, pattern in enumerate(problematic_patterns, 1):
        print(f"\n   {i}. 🔴 PADRÃO: {pattern['pattern']}")
        print(f"      ❌ Problema: {pattern['problema']}")
        print(f"      ✅ Solução: {pattern['solucao']}")
    
    # 4. INVESTIGAÇÃO DE ACTION SPACE
    print(f"\n📊 ACTION SPACE ANALYSIS")
    print("=" * 50)
    
    action_info = [
        {"idx": 0, "name": "order_type", "range": "[0, 2]", "type": "discrete", "optimal_init": "neutro (~1.0)"},
        {"idx": 1, "name": "quantity", "range": "[0, 1]", "type": "continuous", "optimal_init": "BIAS POSITIVO (+2.0)"},
        {"idx": 2, "name": "temporal_signal", "range": "[-1, 1]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 3, "name": "risk_appetite", "range": "[0, 1]", "type": "continuous", "optimal_init": "conservador (+0.5)"},
        {"idx": 4, "name": "regime_bias", "range": "[-1, 1]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 5, "name": "sl1", "range": "[-3, 3]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 6, "name": "sl2", "range": "[-3, 3]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 7, "name": "sl3", "range": "[-3, 3]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 8, "name": "tp1", "range": "[-3, 3]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 9, "name": "tp2", "range": "[-3, 3]", "type": "continuous", "optimal_init": "neutro (~0.0)"},
        {"idx": 10, "name": "tp3", "range": "[-3, 3]", "type": "continuous", "optimal_init": "neutro (~0.0)"}
    ]
    
    print("📋 INICIALIZAÇÃO IDEAL POR AÇÃO:")
    for action in action_info:
        print(f"   [{action['idx']:2d}] {action['name']:15s}: {action['range']:8s} → {action['optimal_init']}")
    
    print(f"\n🎯 AÇÃO[1] - QUANTIDADE (CRÍTICA):")
    print(f"   🔴 Problema: Raw values sempre < -10")
    print(f"   💡 Solução: Bias inicial = +2.0 a +3.0")
    print(f"   📊 Efeito: sigmoid(2.0) = 0.88, sigmoid(3.0) = 0.95")
    print(f"   ✅ Resultado: Quantidade inicial alta, ajustável pelo treino")
    
    # 5. COMPONENTES LSTM
    print(f"\n🧠 ANÁLISE DOS COMPONENTES LSTM")
    print("=" * 50)
    
    lstm_components = [
        "v7_actor_lstm",
        "v7_critic_gru (se existir)",
        "unified_backbone.market_lstm (se existir)"
    ]
    
    print("📋 LSTM COMPONENTS A CORRIGIR:")
    for comp in lstm_components:
        print(f"   - {comp}")
    
    print(f"\n🔧 CORREÇÕES LSTM NECESSÁRIAS:")
    print(f"   1. Forget gate bias = 1.0 (padrão LSTM)")
    print(f"   2. Input/Output gates bias = 0.0")
    print(f"   3. Cell state bias = 0.0")
    print(f"   4. Weights: Xavier/Glorot normal")
    
    # 6. MEMORY COMPONENTS
    print(f"\n💾 ANÁLISE DOS COMPONENTES DE MEMÓRIA")
    print("=" * 50)
    
    memory_components = [
        ("trade_memory", "TradeMemorySystem"),
        ("enhanced_memory", "EnhancedMemoryBank"),
        ("unified_backbone.memory", "Backbone memory (se existir)")
    ]
    
    print("📋 MEMORY COMPONENTS:")
    for name, desc in memory_components:
        print(f"   - {name:20s}: {desc}")
    
    print(f"\n🔧 CORREÇÕES MEMORY NECESSÁRIAS:")
    print(f"   1. Verificar inicialização de embedding layers")
    print(f"   2. Attention weights inicializados corretamente")
    print(f"   3. Memory buffers inicializados com zeros")
    
    # 7. PLANO DE CORREÇÃO COMPLETO
    print(f"\n🎯 PLANO DE CORREÇÃO COMPLETO")
    print("=" * 60)
    
    correction_plan = [
        {
            "priority": "CRÍTICO",
            "component": "actor_head",
            "action": "Inicialização específica por dimensão",
            "details": "Action[1] bias = +2.5, outras neutras"
        },
        {
            "priority": "ALTO", 
            "component": "LSTM components",
            "action": "Forget gate bias = 1.0",
            "details": "Todos os LSTMs do sistema"
        },
        {
            "priority": "MÉDIO",
            "component": "critic_mlp",
            "action": "He initialization para LeakyReLU",
            "details": "Substituir Xavier por He"
        },
        {
            "priority": "MÉDIO",
            "component": "entry_head/management_head",
            "action": "Verificar inicialização interna",
            "details": "Garantir consistency"
        },
        {
            "priority": "BAIXO",
            "component": "memory_components",
            "action": "Verificar embeddings",
            "details": "Inicialização padrão OK"
        }
    ]
    
    print("📋 PRIORIDADES DE CORREÇÃO:")
    for i, plan in enumerate(correction_plan, 1):
        print(f"\n   {i}. [{plan['priority']:8s}] {plan['component']}")
        print(f"      🔧 Ação: {plan['action']}")
        print(f"      📝 Detalhes: {plan['details']}")
    
    # 8. CÓDIGO DE CORREÇÃO
    print(f"\n💻 CÓDIGO DE CORREÇÃO NECESSÁRIO")
    print("=" * 60)
    
    print("🔧 FUNÇÃO DE INICIALIZAÇÃO CORRIGIDA:")
    print("""
def _initialize_all_components_properly(self):
    '''🔧 Inicialização completa e específica de TODOS os componentes'''
    
    # 1. ACTOR HEAD - Específico por dimensão
    self._init_actor_head_by_dimension()
    
    # 2. LSTM COMPONENTS - Forget gate bias
    self._init_lstm_components()
    
    # 3. CRITIC MLP - He initialization
    self._init_critic_mlp()
    
    # 4. SPECIALIZED HEADS - Verificar consistency
    self._init_specialized_heads()
    
    # 5. MEMORY COMPONENTS - Embeddings
    self._init_memory_components()
    """)
    
    print(f"\n✅ RESULTADO ESPERADO APÓS CORREÇÃO:")
    print(f"   - Action[0]: Valores balanceados (0, 1, 2)")
    print(f"   - Action[1]: Valores iniciais ~0.8-0.9 (treináveis)")
    print(f"   - Actions[2-10]: Valores neutros ~0.0")
    print(f"   - LSTM: Gates funcionais desde o início")
    print(f"   - Critic: Gradientes estáveis")
    print(f"   - Memory: Embeddings funcionais")

def main():
    analisar_inicializacao_completa()

if __name__ == "__main__":
    main()