#!/usr/bin/env python3
"""
🎯 ANÁLISE DE THRESHOLD PARA RETREINO
Determinar valor ideal de composite threshold
"""

def analyze_threshold_for_retraining():
    """Analisar threshold ideal para retreino"""
    print("ANALISE DE THRESHOLD PARA RETREINO DO ZERO")
    print("="*60)
    
    print("CENARIO ATUAL (MODELO EXPERIENTE):")
    print("  - Threshold 0.5: 99.1% bloqueio por max_positions")
    print("  - Threshold 0.7: 99.1% bloqueio por max_positions") 
    print("  - Threshold 0.85: Ainda cria posicoes LONGs facilmente")
    print("  - Modelo aprendeu a 'burlar' filtros")
    
    print("\nMODELO NOVO (RETREINO DO ZERO):")
    print("  - Pesos aleatorios iniciais")
    print("  - Nao sabe como 'burlar' sistema")
    print("  - Precisa aprender a gerar scores altos")
    print("  - Mais sensivel a thresholds")
    
    print("\nPROGRESSAO RECOMENDADA:")
    
    thresholds = [
        (0.6, "FASE INICIAL", "Exploracao basica - aprende a fazer trades"),
        (0.7, "DESENVOLVIMENTO", "Refinamento - mais seletivo"),
        (0.75, "OTIMIZACAO", "Alta qualidade - foco em melhores trades"),
        (0.8, "ESPECIALIZACAO", "Elite - apenas trades excecionais")
    ]
    
    for threshold, fase, descricao in thresholds:
        taxa_esperada = calculate_expected_rate(threshold)
        trades_por_dia = taxa_esperada * 18  # Target: 18 trades/dia
        
        print(f"\n{fase} - Threshold {threshold}")
        print(f"  Taxa esperada: ~{taxa_esperada*100:.1f}%")
        print(f"  Trades/dia estimados: ~{trades_por_dia:.1f}")
        print(f"  Descricao: {descricao}")
        
        if threshold == 0.75:
            print(f"  RECOMENDADO PARA INICIO")
    
    print(f"\nRECOMENDACAO FINAL:")
    print(f"  INICIAR COM: 0.75 (75%)")
    print(f"  RAZOES:")
    print(f"    - Modelo novo sera mais sensivel")
    print(f"    - Forca aprendizado de qualidade")
    print(f"    - Evita vicio em 'max positions'")
    print(f"    - Permite ~4-6 trades/dia (saudavel)")
    print(f"    - Pode ajustar depois conforme evolucao")
    
    print(f"\nTRADE-OFFS:")
    print(f"  PROS threshold 0.75:")
    print(f"    - Menos spam de posicoes")
    print(f"    - Foca em trades de qualidade")
    print(f"    - Max_positions raramente atingido")
    print(f"    - Aprendizado mais limpo")
    
    print(f"\n  CONTRAS threshold 0.75:")
    print(f"    - Inicio mais lento (menos trades)")
    print(f"    - Pode ser muito restritivo inicialmente")
    print(f"    - Modelo pode demorar para 'destravar'")
    
    print(f"\nESTRATEGIA DE RETREINO:")
    print(f"  1. Iniciar com 0.75")
    print(f"  2. Monitorar taxa de trades nos primeiros 100k steps")
    print(f"  3. Se < 2 trades/dia: reduzir para 0.7")
    print(f"  4. Se > 8 trades/dia: aumentar para 0.8")
    print(f"  5. Deixar modelo estabilizar e evoluir")

def calculate_expected_rate(threshold):
    """Calcular taxa esperada baseada no threshold"""
    # Estimativa baseada em que scores seguem distribuição normal
    # e modelo novo terá scores mais baixos inicialmente
    
    if threshold <= 0.6:
        return 0.4  # 40% - muito permissivo
    elif threshold <= 0.7:
        return 0.25  # 25% - moderado
    elif threshold <= 0.75:
        return 0.15  # 15% - seletivo
    elif threshold <= 0.8:
        return 0.08  # 8% - muito seletivo
    else:
        return 0.03  # 3% - elite

if __name__ == "__main__":
    analyze_threshold_for_retraining()