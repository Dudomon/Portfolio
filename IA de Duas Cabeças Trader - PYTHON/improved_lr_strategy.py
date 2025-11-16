#!/usr/bin/env python3
"""
🚀 ESTRATÉGIA DE LR MELHORADA PARA LSTM
Baseada na análise do seu sistema atual + necessidades das LSTMs
"""

def analyze_current_lr_system():
    """🔍 Análise do sistema atual"""
    print("🔍 ANÁLISE DO SEU SISTEMA DE LR ATUAL")
    print("=" * 60)
    
    current_system = {
        'lr_fixo': {
            'valor': 2.678385767462569e-05,
            'pros': ['Estável', 'Testado', 'Funciona'],
            'contras': ['Não adapta', 'Pode ser lento para LSTM']
        },
        'adaptive_callback': {
            'frequencia': 2000,
            'range': '1e-6 a 1e-3',
            'pros': ['Monitora gradientes', 'Adapta automaticamente'],
            'contras': ['Só diminui LR', 'Lógica contraproducente', 'Conflita com LR fixo']
        }
    }
    
    print("📊 SISTEMA ATUAL:")
    for system, details in current_system.items():
        print(f"\n{system.upper()}:")
        if 'valor' in details:
            print(f"   Valor: {details['valor']}")
        if 'frequencia' in details:
            print(f"   Frequência: {details['frequencia']} steps")
        if 'range' in details:
            print(f"   Range: {details['range']}")
        
        print(f"   ✅ Prós: {', '.join(details['pros'])}")
        print(f"   ❌ Contras: {', '.join(details['contras'])}")
    
    return current_system

def recommend_lstm_lr_strategy():
    """💡 Estratégia recomendada para LSTM"""
    print(f"\n💡 ESTRATÉGIA RECOMENDADA PARA LSTM")
    print("=" * 60)
    
    strategies = [
        {
            'name': 'OPÇÃO 1: LR FIXO + WARMUP (RECOMENDADO)',
            'description': 'Manter LR fixo mas adicionar warmup para LSTM',
            'implementation': '''
def lr_schedule_with_warmup(progress):
    warmup_steps = 0.1  # 10% dos steps
    base_lr = 2.678385767462569e-05  # Seu LR otimizado
    
    if progress < warmup_steps:
        # Warmup: começar com LR baixo
        return base_lr * 0.1 * (progress / warmup_steps)
    else:
        # LR fixo após warmup
        return base_lr
            ''',
            'pros': ['Mantém estabilidade', 'Ajuda LSTM inicializar', 'Simples'],
            'contras': ['Não adapta durante treinamento'],
            'impact': 'Alto para LSTM',
            'risk': 'Baixo'
        },
        {
            'name': 'OPÇÃO 2: ADAPTIVE LR MELHORADO',
            'description': 'Corrigir lógica do adaptive LR para LSTM',
            'implementation': '''
def determine_new_lr_for_lstm(gradient_health, lstm_health):
    if lstm_health < 0.1:  # LSTM muito problemática
        return current_lr * 0.5  # Diminuir
    elif lstm_health < 0.2:  # LSTM moderadamente problemática  
        return current_lr * 0.8  # Diminuir pouco
    elif gradient_health > 0.8:  # Gradientes muito ativos
        return current_lr * 1.1  # AUMENTAR (diferente do atual!)
    else:
        return current_lr  # Manter
            ''',
            'pros': ['Adapta especificamente para LSTM', 'Pode aumentar LR'],
            'contras': ['Mais complexo', 'Pode instabilizar'],
            'impact': 'Alto',
            'risk': 'Médio'
        },
        {
            'name': 'OPÇÃO 3: LR DIFERENCIADO POR LAYER',
            'description': 'LR específico para LSTM vs outros layers',
            'implementation': '''
# Separar parâmetros
lstm_params = []
other_params = []

for name, param in model.policy.named_parameters():
    if 'lstm' in name.lower():
        lstm_params.append(param)
    else:
        other_params.append(param)

# Optimizer com LRs diferentes
optimizer = torch.optim.Adam([
    {'params': lstm_params, 'lr': 1e-4},     # LR menor para LSTM
    {'params': other_params, 'lr': 2.68e-5}  # LR normal
])
            ''',
            'pros': ['Controle fino', 'LSTM pode ter LR próprio'],
            'contras': ['Requer modificação do PPO', 'Complexo'],
            'impact': 'Muito Alto',
            'risk': 'Alto'
        },
        {
            'name': 'OPÇÃO 4: DESABILITAR ADAPTIVE + WARMUP',
            'description': 'Remover adaptive LR conflitante e usar só warmup',
            'implementation': '''
# 1. Comentar/remover adaptive_lr_callback do daytrader.py
# adaptive_lr_callback = create_adaptive_lr_callback(...)

# 2. Usar LR schedule com warmup
def lr_schedule_lstm_friendly(progress):
    base_lr = 2.678385767462569e-05
    warmup = 0.05  # 5% warmup
    
    if progress < warmup:
        return base_lr * 0.2 * (progress / warmup)  # Começar com 20%
    else:
        return base_lr  # LR fixo testado
            ''',
            'pros': ['Remove conflito', 'Simples', 'Mantém LR testado'],
            'contras': ['Não adapta durante treinamento'],
            'impact': 'Médio',
            'risk': 'Muito Baixo'
        }
    ]
    
    print("🎯 OPÇÕES DISPONÍVEIS:")
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   📝 {strategy['description']}")
        print(f"   🎯 Impacto: {strategy['impact']}")
        print(f"   ⚠️ Risco: {strategy['risk']}")
        print(f"   ✅ Prós: {', '.join(strategy['pros'])}")
        print(f"   ❌ Contras: {', '.join(strategy['contras'])}")
    
    return strategies

def generate_immediate_fix():
    """🚀 Correção imediata recomendada"""
    print(f"\n🚀 CORREÇÃO IMEDIATA RECOMENDADA")
    print("=" * 60)
    
    print("🎯 IMPLEMENTAR OPÇÃO 4: DESABILITAR ADAPTIVE + WARMUP")
    print("\n📝 PASSOS:")
    print("1. Comentar adaptive_lr_callback no daytrader.py")
    print("2. Implementar lr_schedule com warmup")
    print("3. Aplicar gradient clipping 0.5")
    print("4. Aplicar LSTM initialization")
    
    code_changes = {
        'step1': '''
# No daytrader.py, comentar estas linhas:
# adaptive_lr_callback = create_adaptive_lr_callback(
#     initial_lr=BEST_PARAMS["learning_rate"],
#     min_lr=1e-6,
#     max_lr=1e-3,
#     adaptation_freq=2000,
#     verbose=1
# )

# E remover da CallbackList:
combined_callback = CallbackList([
    robust_callback, 
    metrics_callback, 
    progress_callback, 
    gradient_callback,
    zero_debug_callback,
    # adaptive_lr_callback,  # COMENTAR ESTA LINHA
    lstm_rescue_callback
])
        ''',
        
        'step2': '''
# Substituir lr_schedule por:
def lr_schedule_lstm_warmup(progress):
    """LR schedule otimizado para LSTM com warmup"""
    base_lr = 2.678385767462569e-05  # Seu LR testado
    warmup_steps = 0.05  # 5% dos steps para warmup
    
    if progress < warmup_steps:
        # Warmup suave: começar com 20% do LR
        warmup_factor = 0.2 + 0.8 * (progress / warmup_steps)
        return base_lr * warmup_factor
    else:
        # LR fixo após warmup (testado e estável)
        return base_lr

# Usar na criação da policy:
policy = TwoHeadV6Intelligent48h(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lr_schedule_lstm_warmup,  # NOVA FUNÇÃO
    lstm_hidden_size=128
)
        '''
    }
    
    print("\n💻 CÓDIGO PARA APLICAR:")
    for step, code in code_changes.items():
        print(f"\n{step.upper()}:")
        print(code)
    
    return code_changes

if __name__ == "__main__":
    # Analisar sistema atual
    current = analyze_current_lr_system()
    
    # Recomendar estratégias
    strategies = recommend_lstm_lr_strategy()
    
    # Gerar correção imediata
    fix = generate_immediate_fix()
    
    print(f"\n" + "=" * 60)
    print("🎯 RESUMO EXECUTIVO")
    print("=" * 60)
    
    print("🔍 PROBLEMA IDENTIFICADO:")
    print("   ❌ Adaptive LR conflita com LR fixo")
    print("   ❌ Adaptive LR só diminui (nunca aumenta)")
    print("   ❌ LSTM precisa de warmup suave")
    
    print(f"\n💡 SOLUÇÃO RECOMENDADA:")
    print("   ✅ DESABILITAR Adaptive LR (conflitante)")
    print("   ✅ IMPLEMENTAR LR Warmup (5% dos steps)")
    print("   ✅ MANTER LR fixo testado (2.68e-5)")
    print("   ✅ APLICAR gradient clipping 0.5")
    
    print(f"\n🚀 RESULTADO ESPERADO:")
    print("   ✅ LSTM inicializa suavemente")
    print("   ✅ Sem conflitos de LR")
    print("   ✅ Gradientes LSTM: 15.43% → <5% zeros")
    print("   ✅ Sistema mais estável")