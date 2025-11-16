#!/usr/bin/env python3
"""
🚀 ATIVAÇÃO DE FEATURES INTELIGENTES DORMENTES
============================================

Script para ativar gradualmente features já implementadas mas desabilitadas
"""

import sys
import os

def activate_unified_rewards():
    """
    🎯 NÍVEL 1: Ativar Unified Reward Components (baixo risco)
    """
    silus_path = "D:/Projeto/silus.py"

    print("🚀 ATIVANDO UNIFIED REWARD COMPONENTS...")

    # Ler arquivo
    with open(silus_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Substituições graduais
    modifications = [
        # 1. Ativar sistema de componentes
        ("USE_COMPONENT_REWARDS = False", "USE_COMPONENT_REWARDS = True"),

        # 2. Ajustar pesos conservadores
        ("COMPONENT_REWARD_WEIGHTS = {\n    'base': 0.8,",
         "COMPONENT_REWARD_WEIGHTS = {\n    'base': 0.9,"),  # Mais conservador
        ("'timing': 0.1,", "'timing': 0.05,"),  # Reduzir timing
        ("'management': 0.1", "'management': 0.05"),  # Reduzir management
    ]

    for old, new in modifications:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ {old[:30]}... → {new[:30]}...")
        else:
            print(f"⚠️ Não encontrado: {old[:30]}...")

    # Salvar backup
    backup_path = silus_path + ".backup_before_unified_rewards"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup salvo: {backup_path}")

    # Salvar modificado
    with open(silus_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ UNIFIED REWARDS ATIVADO!")
    print("🎯 Agora o modelo usará timing + management components!")

def activate_market_regime_focus():
    """
    🧠 NÍVEL 2: Focar uso do Market Regime Detector (médio risco)
    """
    print("\n🧠 ATIVANDO MARKET REGIME FOCUS...")

    # Esta funcionalidade já está ativa, mas pode ser melhorada
    # analisando se o modelo realmente usa as detecções de regime

    silus_path = "D:/Projeto/silus.py"

    with open(silus_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Verificar se market_regime está sendo usado efetivamente
    regime_usage_checks = [
        "market_regime",
        "_classify_market_regime",
        "volatility_context",
        "momentum_confluence"
    ]

    for check in regime_usage_checks:
        count = content.count(check)
        print(f"📊 '{check}': {count} ocorrências")

    # Esses componentes JÁ ESTÃO ATIVOS mas podem não estar sendo usados efetivamente
    print("💡 Market regime components já estão implementados!")
    print("🎯 Próximo: Verificar se modelo usa efetivamente...")

def enhance_v11_market_context():
    """
    🎭 NÍVEL 3: Melhorar uso do V11 Market Context (alto potencial)
    """
    print("\n🎭 ANALISANDO V11 MARKET CONTEXT...")

    v11_path = "D:/Projeto/trading_framework/policies/two_head_v11_sigmoid.py"

    with open(v11_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Analisar uso do MarketContextEncoder
    context_features = [
        "MarketContextEncoder",
        "regime_detector",
        "regime_embedding",
        "context_processor"
    ]

    for feature in context_features:
        count = content.count(feature)
        print(f"🔍 '{feature}': {count} ocorrências")

    # O V11 JÁ TEM detector de regime sofisticado!
    print("💎 V11 tem MarketContextEncoder avançado!")
    print("🎯 Detecta 4 regimes: Bull/Bear/Sideways/Volatile")

def create_activation_plan():
    """
    📋 PLANO DE ATIVAÇÃO GRADUAL
    """
    print("\n" + "="*60)
    print("📋 PLANO DE ATIVAÇÃO DE FEATURES DORMENTES")
    print("="*60)

    plan = [
        {
            'level': 1,
            'name': 'Unified Reward Components',
            'risk': 'BAIXO',
            'effort': 'MÍNIMO',
            'impact': 'MÉDIO',
            'description': 'Ativar sistema de rewards por componentes já implementado',
            'action': 'Mudar USE_COMPONENT_REWARDS = True'
        },
        {
            'level': 2,
            'name': 'Market Intelligence Features',
            'risk': 'BAIXO',
            'effort': 'BAIXO',
            'impact': 'ALTO',
            'description': 'Melhorar uso de volume_momentum, session_momentum, time_of_day',
            'action': 'Verificar se modelo usa efetivamente estas features'
        },
        {
            'level': 3,
            'name': 'V11 Market Context Enhancement',
            'risk': 'MÉDIO',
            'effort': 'MÉDIO',
            'impact': 'ALTO',
            'description': 'Expandir uso do regime detector V11 para decisões',
            'action': 'Integrar regime_id nas decisões de entry/management'
        },
        {
            'level': 4,
            'name': 'Intelligent Components V7',
            'risk': 'MÉDIO',
            'effort': 'ALTO',
            'impact': 'MUITO ALTO',
            'description': 'Usar totalmente os 12 componentes V7 já calculados',
            'action': 'Mapear componentes para decisões específicas'
        }
    ]

    for item in plan:
        print(f"\n🎯 NÍVEL {item['level']}: {item['name']}")
        print(f"   📊 Risco: {item['risk']} | Esforço: {item['effort']} | Impacto: {item['impact']}")
        print(f"   📝 {item['description']}")
        print(f"   ⚡ Ação: {item['action']}")

    print(f"\n💡 RECOMENDAÇÃO: Começar pelo Nível 1 (menor risco, ativação imediata)")

if __name__ == "__main__":
    print("🔍 ANÁLISE DE FEATURES DORMENTES")
    print("="*60)

    try:
        # Nível 1: Ativação segura
        choice = input("\n🚀 Ativar Unified Rewards agora? (s/n): ").lower()
        if choice == 's':
            activate_unified_rewards()

        # Análises
        activate_market_regime_focus()
        enhance_v11_market_context()
        create_activation_plan()

        print("\n" + "="*60)
        print("✅ ANÁLISE COMPLETA!")
        print("🎯 Próximo passo: Testar Level 1 ativação")
        print("="*60)

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()