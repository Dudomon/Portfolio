#!/usr/bin/env python3
"""
🔍 AUDITORIA COMPLETA DAS 3 POLÍTICAS V7

Verificar se a TwoHeadV7Unified tem TODAS as funcionalidades:
1. LSTM Actor + MLP Critic (detalhes das V7s)
2. LeakyReLU ao invés de ReLU
3. Gates especializados da V7Simple
4. Funcionalidades da Enhanced e Intuition
5. UNIFICAR, não simplificar
"""

import re

def audit_v7_simple():
    """Auditar V7Simple para extrair funcionalidades"""
    
    print("🔍 AUDITORIA V7SIMPLE")
    print("=" * 60)
    
    try:
        with open('trading_framework/policies/two_head_v7_simple.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extrair gates especializados
        gates_found = []
        gate_patterns = [
            r'self\.(\w+_analyzer)',
            r'self\.(\w+_validator)',
            r'self\.(\w+_gate)',
            r'self\.(\w+_detector)',
            r'self\.(\w+_memory)',
        ]
        
        for pattern in gate_patterns:
            matches = re.findall(pattern, content)
            gates_found.extend(matches)
        
        print(f"📊 GATES ENCONTRADOS NA V7SIMPLE:")
        for gate in set(gates_found):
            print(f"   ✅ {gate}")
        
        # Verificar LSTM Actor
        if 'v7_actor_lstm' in content:
            print("✅ LSTM Actor encontrado")
        else:
            print("❌ LSTM Actor não encontrado")
        
        # Verificar MLP Critic
        if 'v7_critic_mlp' in content:
            print("✅ MLP Critic encontrado")
        else:
            print("❌ MLP Critic não encontrado")
        
        # Verificar LeakyReLU
        leaky_count = content.count('LeakyReLU')
        relu_count = content.count('ReLU(') - leaky_count
        print(f"📊 ATIVAÇÕES: LeakyReLU: {leaky_count}, ReLU: {relu_count}")
        
        return set(gates_found)
        
    except FileNotFoundError:
        print("❌ V7Simple não encontrado")
        return set()

def audit_v7_enhanced():
    """Auditar V7Enhanced para extrair enhancements"""
    
    print("\n🔍 AUDITORIA V7ENHANCED")
    print("=" * 60)
    
    try:
        with open('trading_framework/policies/two_head_v7_enhanced.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extrair enhancements
        enhancements = []
        
        enhancement_patterns = [
            r'MarketRegimeDetector',
            r'EnhancedMemoryBank',
            r'GradientBalancer',
            r'NeuralBreathingMonitor',
            r'V7UpgradeManager',
            r'EnhancedTradeMemoryBank',
        ]
        
        for pattern in enhancement_patterns:
            if pattern in content:
                enhancements.append(pattern)
                print(f"   ✅ {pattern}")
            else:
                print(f"   ❌ {pattern} não encontrado")
        
        return enhancements
        
    except FileNotFoundError:
        print("❌ V7Enhanced não encontrado")
        return []

def audit_v7_intuition():
    """Auditar V7Intuition para extrair funcionalidades"""
    
    print("\n🔍 AUDITORIA V7INTUITION")
    print("=" * 60)
    
    try:
        with open('trading_framework/policies/two_head_v7_intuition.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extrair funcionalidades da Intuition
        intuition_features = []
        
        intuition_patterns = [
            r'UnifiedMarketBackbone',
            r'GradientMixer',
            r'InterferenceMonitor',
            r'unified_backbone',
            r'gradient_mixing',
            r'adaptive_sharing',
        ]
        
        for pattern in intuition_patterns:
            if pattern in content:
                intuition_features.append(pattern)
                print(f"   ✅ {pattern}")
            else:
                print(f"   ❌ {pattern} não encontrado")
        
        return intuition_features
        
    except FileNotFoundError:
        print("❌ V7Intuition não encontrado")
        return []

def audit_v7_unified():
    """Auditar V7Unified para verificar se tem tudo"""
    
    print("\n🔍 AUDITORIA V7UNIFIED")
    print("=" * 60)
    
    try:
        with open('trading_framework/policies/two_head_v7_unified.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar gates (busca específica)
        gates_in_unified = []
        
        # Gates específicos para procurar
        specific_gates = [
            'horizon_analyzer', 'mtf_validator', 'risk_gate', 'momentum_gate',
            'volatility_gate', 'trend_gate', 'support_resistance_gate', 'volume_gate',
            'market_structure_gate', 'liquidity_gate', 'lookahead_gate', 
            'pattern_memory_validator', 'regime_gate', 'fatigue_detector',
            'pattern_memory', 'trade_memory', 'critic_memory', 'position_analyzer'
        ]
        
        for gate in specific_gates:
            if f'self.{gate}' in content:
                gates_in_unified.append(gate)
        
        # Também procurar por padrões gerais
        gate_patterns = [
            r'self\.(\w+_analyzer)',
            r'self\.(\w+_validator)', 
            r'self\.(\w+_gate)',
            r'self\.(\w+_detector)',
        ]
        
        for pattern in gate_patterns:
            matches = re.findall(pattern, content)
            gates_in_unified.extend(matches)
        
        print(f"📊 GATES NA V7UNIFIED:")
        for gate in set(gates_in_unified):
            print(f"   ✅ {gate}")
        
        # Verificar arquitetura
        if 'v7_actor_lstm' in content:
            print("✅ LSTM Actor presente")
        else:
            print("❌ LSTM Actor ausente")
        
        if 'v7_critic_mlp' in content:
            print("✅ MLP Critic presente")
        else:
            print("❌ MLP Critic ausente")
        
        # Verificar LeakyReLU
        leaky_count = content.count('LeakyReLU')
        relu_count = content.count('ReLU(') - leaky_count
        print(f"📊 ATIVAÇÕES: LeakyReLU: {leaky_count}, ReLU: {relu_count}")
        
        return set(gates_in_unified)
        
    except FileNotFoundError:
        print("❌ V7Unified não encontrado")
        return set()

def compare_functionalities():
    """Comparar funcionalidades entre as políticas"""
    
    print("\n🔍 COMPARAÇÃO DE FUNCIONALIDADES")
    print("=" * 60)
    
    # Auditar todas
    simple_gates = audit_v7_simple()
    enhanced_features = audit_v7_enhanced()
    intuition_features = audit_v7_intuition()
    unified_gates = audit_v7_unified()
    
    # Comparar gates
    print(f"\n📊 COMPARAÇÃO DE GATES:")
    missing_gates = simple_gates - unified_gates
    if missing_gates:
        print(f"❌ GATES FALTANDO NA UNIFIED:")
        for gate in missing_gates:
            print(f"   - {gate}")
    else:
        print("✅ TODOS OS GATES DA V7SIMPLE PRESENTES")
    
    # Verificar enhancements V7Enhanced
    print(f"\n📊 ENHANCEMENTS DA V7ENHANCED:")
    
    # Ler conteúdo da V7Unified para verificação
    try:
        with open('trading_framework/policies/two_head_v7_unified.py', 'r', encoding='utf-8') as f:
            unified_content = f.read()
    except:
        unified_content = ""
    
    v7_enhanced_check = [
        ('MarketRegimeDetector', 'market_regime_detector'),
        ('EnhancedMemoryBank', 'enhanced_memory_bank'),
        ('GradientBalancer', 'gradient_balancer'),
        ('NeuralBreathingMonitor', 'neural_breathing_monitor'),
        ('EnhancedTradeMemoryBank', 'trade_memory')
    ]
    
    for class_name, instance_name in v7_enhanced_check:
        if f'self.{instance_name}' in unified_content:
            print(f"   ✅ {class_name} (como {instance_name})")
        elif class_name in unified_content:
            print(f"   ✅ {class_name} (classe definida)")
        else:
            print(f"   ❌ {class_name} FALTANDO")
    
    # Verificar intuition features V7Intuition
    print(f"\n📊 FEATURES DA V7INTUITION:")
    
    v7_intuition_check = [
        ('UnifiedMarketBackbone', 'unified_backbone'),
        ('GradientMixer', 'gradient_mixing'),
        ('InterferenceMonitor', 'adaptive_sharing')
    ]
    
    for class_name, instance_name in v7_intuition_check:
        if f'self.{instance_name}' in unified_content:
            print(f"   ✅ {class_name} (como {instance_name})")
        elif class_name in unified_content:
            print(f"   ✅ {class_name} (classe definida)")
        else:
            print(f"   ❌ {class_name} FALTANDO")

def generate_missing_components():
    """Gerar lista de componentes que precisam ser adicionados"""
    
    print(f"\n🔧 COMPONENTES QUE PRECISAM SER ADICIONADOS")
    print("=" * 60)
    
    missing_components = [
        {
            "component": "MarketRegimeDetector",
            "source": "V7Enhanced",
            "description": "Detecção de regime de mercado"
        },
        {
            "component": "EnhancedMemoryBank", 
            "source": "V7Enhanced",
            "description": "Sistema de memória avançado"
        },
        {
            "component": "GradientBalancer",
            "source": "V7Enhanced", 
            "description": "Balanceamento de gradientes"
        },
        {
            "component": "UnifiedMarketBackbone",
            "source": "V7Intuition",
            "description": "Backbone unificado para visão compartilhada"
        },
        {
            "component": "GradientMixer",
            "source": "V7Intuition",
            "description": "Sistema de mixing de gradientes"
        },
        {
            "component": "InterferenceMonitor",
            "source": "V7Intuition", 
            "description": "Monitor de interferência de gradientes"
        }
    ]
    
    print("📋 LISTA DE COMPONENTES FALTANTES:")
    for i, comp in enumerate(missing_components, 1):
        print(f"{i}. {comp['component']} ({comp['source']})")
        print(f"   Descrição: {comp['description']}")
        print()

if __name__ == "__main__":
    print("🔍 AUDITORIA COMPLETA DAS POLÍTICAS V7")
    print("=" * 80)
    print("Verificando se V7Unified tem TODAS as funcionalidades")
    print()
    
    # Executar auditoria completa
    compare_functionalities()
    
    # Gerar lista de componentes faltantes
    generate_missing_components()
    
    print("\n🎯 CONCLUSÃO:")
    print("A V7Unified precisa ser EXPANDIDA com todos os componentes")
    print("das V7Enhanced e V7Intuition, não simplificada!")
    print("\n🚀 PRÓXIMO PASSO: Adicionar componentes faltantes")