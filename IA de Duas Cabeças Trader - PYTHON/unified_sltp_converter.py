#!/usr/bin/env python3
"""
🔧 SISTEMA UNIFICADO DE CONVERSÃO SL/TP
Substitui todas as duplicações de código por uma única função consistente
"""

# Configuração unificada (mesmo do silus.py)
UNIFIED_SLTP_CONFIG = {
    'sl_min_points': 2.0,     # SL mínimo: 2 pontos (daytrade)
    'sl_max_points': 8.0,     # SL máximo: 8 pontos (daytrade)
    'tp_min_points': 3.0,     # TP mínimo: 3 pontos (daytrade)
    'tp_max_points': 15.0,    # TP máximo: 15 pontos (daytrade)
    'sl_tp_step': 0.5,        # Variação: 0.5 pontos
}

def convert_model_adjustments_to_points(sl_adjust, tp_adjust, context="adjustment"):
    """
    🎯 FUNÇÃO UNIFICADA: Converte ajustes do modelo ±0.5 para pontos válidos

    Args:
        sl_adjust (float): Ajuste SL do modelo (±0.5)
        tp_adjust (float): Ajuste TP do modelo (±0.5)
        context (str): "creation" para novas posições, "adjustment" para ajustes

    Returns:
        dict: {
            'sl_points': float,      # Pontos SL (sempre positivo)
            'tp_points': float,      # Pontos TP (sempre positivo)
            'sl_change': float,      # Mudança em pontos (pode ser negativa)
            'tp_change': float,      # Mudança em pontos (pode ser negativa)
            'valid': bool            # Se os valores são válidos
        }
    """

    # Validar inputs
    if not isinstance(sl_adjust, (int, float)) or not isinstance(tp_adjust, (int, float)):
        raise ValueError("sl_adjust e tp_adjust devem ser numéricos")

    result = {
        'sl_points': 0.0,
        'tp_points': 0.0,
        'sl_change': 0.0,
        'tp_change': 0.0,
        'valid': False,
        'context': context
    }

    if context == "creation":
        # 🏗️ CRIAÇÃO DE POSIÇÕES: Converter ±0.5 para ranges válidos

        # SL: Mapear ±0.5 para range [2.0, 8.0]
        # -0.5 = SL mínimo (2.0), +0.5 = SL máximo (8.0), 0 = meio (5.0)
        sl_range = UNIFIED_SLTP_CONFIG['sl_max_points'] - UNIFIED_SLTP_CONFIG['sl_min_points']
        sl_center = UNIFIED_SLTP_CONFIG['sl_min_points'] + (sl_range / 2)
        result['sl_points'] = sl_center + (sl_adjust * sl_range / 2)

        # TP: Mapear ±0.5 para range [3.0, 15.0]
        # -0.5 = TP mínimo (3.0), +0.5 = TP máximo (15.0), 0 = meio (9.0)
        tp_range = UNIFIED_SLTP_CONFIG['tp_max_points'] - UNIFIED_SLTP_CONFIG['tp_min_points']
        tp_center = UNIFIED_SLTP_CONFIG['tp_min_points'] + (tp_range / 2)
        result['tp_points'] = tp_center + (tp_adjust * tp_range / 2)

        # Para criação, change = points (não há valor anterior)
        result['sl_change'] = result['sl_points']
        result['tp_change'] = result['tp_points']

    elif context == "adjustment":
        # 🔧 AJUSTE DE POSIÇÕES: Converter ±0.5 para mudanças diretas

        # Usar os valores ±0.5 diretamente como pontos de mudança
        # Isso é mais intuitivo: +0.5 = aumentar 0.5 pontos, -0.5 = diminuir 0.5 pontos
        result['sl_change'] = sl_adjust  # ±0.5 pontos direto
        result['tp_change'] = tp_adjust  # ±0.5 pontos direto

        # Para ajustes, points = change (será somado ao valor atual)
        result['sl_points'] = abs(result['sl_change'])
        result['tp_points'] = abs(result['tp_change'])

    else:
        raise ValueError(f"Context inválido: {context}. Use 'creation' ou 'adjustment'")

    # Aplicar limites de segurança
    result['sl_points'] = max(UNIFIED_SLTP_CONFIG['sl_min_points'],
                             min(result['sl_points'], UNIFIED_SLTP_CONFIG['sl_max_points']))
    result['tp_points'] = max(UNIFIED_SLTP_CONFIG['tp_min_points'],
                             min(result['tp_points'], UNIFIED_SLTP_CONFIG['tp_max_points']))

    # Arredondar para múltiplos de 0.5
    result['sl_points'] = round(result['sl_points'] * 2) / 2
    result['tp_points'] = round(result['tp_points'] * 2) / 2
    result['sl_change'] = round(result['sl_change'] * 2) / 2
    result['tp_change'] = round(result['tp_change'] * 2) / 2

    # Validar se está dentro dos limites
    result['valid'] = (
        UNIFIED_SLTP_CONFIG['sl_min_points'] <= result['sl_points'] <= UNIFIED_SLTP_CONFIG['sl_max_points'] and
        UNIFIED_SLTP_CONFIG['tp_min_points'] <= result['tp_points'] <= UNIFIED_SLTP_CONFIG['tp_max_points']
    )

    return result

def test_unified_converter():
    """🧪 Teste completo do conversor unificado"""

    print("🧪 [TESTE] Sistema Unificado de Conversão SL/TP")
    print("=" * 60)

    # Casos de teste
    test_cases = [
        (-0.5, 0.0, "creation", "SL Mínimo, TP Centro"),
        (0.5, 0.0, "creation", "SL Máximo, TP Centro"),
        (0.0, -0.5, "creation", "SL Centro, TP Mínimo"),
        (0.0, 0.5, "creation", "SL Centro, TP Máximo"),
        (0.0, 0.0, "creation", "Valores Centrais"),

        (-0.5, 0.0, "adjustment", "Diminuir SL 0.5pts"),
        (0.5, 0.0, "adjustment", "Aumentar SL 0.5pts"),
        (0.0, -0.5, "adjustment", "Diminuir TP 0.5pts"),
        (0.0, 0.5, "adjustment", "Aumentar TP 0.5pts"),
    ]

    for sl_adj, tp_adj, context, desc in test_cases:
        print(f"\n🔍 [TESTE] {desc}")
        print(f"   Input: sl_adjust={sl_adj}, tp_adjust={tp_adj}, context='{context}'")

        try:
            result = convert_model_adjustments_to_points(sl_adj, tp_adj, context)
            print(f"   ✅ SL: {result['sl_points']:.1f}pts (change: {result['sl_change']:+.1f})")
            print(f"   ✅ TP: {result['tp_points']:.1f}pts (change: {result['tp_change']:+.1f})")
            print(f"   ✅ Válido: {result['valid']}")
        except Exception as e:
            print(f"   ❌ ERRO: {e}")

    print("\n" + "=" * 60)
    print("✅ Teste do conversor unificado concluído!")

    # Comparar com sistemas antigos
    print("\n🔍 [COMPARAÇÃO] Sistemas Antigos vs Unificado")

    # Teste específico: tp_adjust = -0.5 (caso do log)
    sl_adj, tp_adj = 0.5, -0.5

    print(f"\n🤖 [MODELO] Produz: sl_adjust={sl_adj}, tp_adjust={tp_adj}")

    # Sistema antigo bugado
    old_tp_change = tp_adj * 5.0  # -0.5 * 5.0 = -2.5
    print(f"❌ [ANTIGO] tp_change = {tp_adj} * 5.0 = {old_tp_change} (MUITO GRANDE!)")

    # Sistema unificado
    new_result = convert_model_adjustments_to_points(sl_adj, tp_adj, "adjustment")
    print(f"✅ [NOVO] tp_change = {new_result['tp_change']} (CORRETO!)")

    print(f"\n💡 [MELHORIA] Redução de {abs(old_tp_change - new_result['tp_change']):.1f} pontos!")

if __name__ == "__main__":
    test_unified_converter()