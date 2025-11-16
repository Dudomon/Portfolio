#!/usr/bin/env python3
"""
🧪 TESTE: Validar balanceamento do Action Space
Verifica se distribuição de ações é equilibrada (33% cada)
"""

import numpy as np

def test_action_distribution(low, high, threshold_long, threshold_short, n_samples=100000):
    """Testa se distribuição é balanceada"""

    np.random.seed(42)
    actions = np.random.uniform(low, high, n_samples)

    hold_count = np.sum(actions < threshold_long)
    long_count = np.sum((actions >= threshold_long) & (actions < threshold_short))
    short_count = np.sum(actions >= threshold_short)

    print(f"\n{'='*70}")
    print(f"Action Space: [{low}, {high}]")
    print(f"Thresholds: HOLD < {threshold_long}, LONG < {threshold_short}, SHORT >= {threshold_short}")
    print(f"{'='*70}")
    print(f"HOLD:  {hold_count:,} ({100*hold_count/n_samples:.1f}%)")
    print(f"LONG:  {long_count:,} ({100*long_count/n_samples:.1f}%)")
    print(f"SHORT: {short_count:,} ({100*short_count/n_samples:.1f}%)")

    # Calcular ranges
    total_range = high - low
    hold_range = threshold_long - low
    long_range = threshold_short - threshold_long
    short_range = high - threshold_short

    print(f"\n📏 RANGES:")
    print(f"   HOLD:  [{low:.2f}, {threshold_long:.2f}] = {hold_range:.2f} ({100*hold_range/total_range:.1f}% do total)")
    print(f"   LONG:  [{threshold_long:.2f}, {threshold_short:.2f}] = {long_range:.2f} ({100*long_range/total_range:.1f}% do total)")
    print(f"   SHORT: [{threshold_short:.2f}, {high:.2f}] = {short_range:.2f} ({100*short_range/total_range:.1f}% do total)")

    # Verificar balanceamento
    expected = n_samples / 3
    tolerance = 0.02  # 2% de tolerância

    hold_error = abs(hold_count/expected - 1)
    long_error = abs(long_count/expected - 1)
    short_error = abs(short_count/expected - 1)

    print(f"\n📊 DESVIO DO ESPERADO (33.33%):")
    print(f"   HOLD:  {hold_error:+.2%}")
    print(f"   LONG:  {long_error:+.2%}")
    print(f"   SHORT: {short_error:+.2%}")

    balanced = (hold_error < tolerance and long_error < tolerance and short_error < tolerance)

    if balanced:
        print(f"\n✅ BALANCEADO! (tolerância ±{tolerance*100}%)")
    else:
        print(f"\n❌ DESBALANCEADO!")

        # Identificar quem está com viés
        max_error = max(hold_error, long_error, short_error)
        if hold_error == max_error:
            bias = "HOLD"
            factor = hold_count / long_count if long_count > 0 else 0
        elif short_error == max_error:
            bias = "SHORT"
            factor = short_count / long_count if long_count > 0 else 0
        else:
            bias = "LONG"
            factor = long_count / short_count if short_count > 0 else 0

        print(f"   🚨 VIÉS DETECTADO: {bias} ({factor:.2f}x mais frequente)")

    return balanced, {
        'hold': hold_count / n_samples,
        'long': long_count / n_samples,
        'short': short_count / n_samples
    }

def main():
    print("\n" + "="*70)
    print("🧪 TESTE DE BALANCEAMENTO DO ACTION SPACE")
    print("="*70)

    # Teste 1: Configuração atual (ERRADA)
    print("\n\n🔴 CONFIGURAÇÃO ATUAL (COM VIÉS):")
    print("   Usado em: cherry.py, Robot_cherry.py (versão atual)")
    balanced1, dist1 = test_action_distribution(0, 2, 0.33, 0.67)

    # Teste 2: Opção 1 - Balanceado
    print("\n\n🟢 OPÇÃO 1 - BALANCEADO ([-1, 1]):")
    print("   Solução definitiva, requer re-treino")
    balanced2, dist2 = test_action_distribution(-1, 1, -0.33, 0.33)

    # Teste 3: Opção 2 - Compatível
    print("\n\n🟢 OPÇÃO 2 - COMPATÍVEL ([0, 2]):")
    print("   Mantém modelos atuais, apenas muda thresholds")
    balanced3, dist3 = test_action_distribution(0, 2, 0.67, 1.33)

    # Resumo
    print("\n\n" + "="*70)
    print("📊 RESUMO COMPARATIVO")
    print("="*70)

    configs = [
        ("Atual (VIÉS)", balanced1, dist1),
        ("Opção 1 (Balanceado)", balanced2, dist2),
        ("Opção 2 (Compatível)", balanced3, dist3)
    ]

    print(f"\n{'Configuração':<25} {'HOLD':<12} {'LONG':<12} {'SHORT':<12} {'Status':<10}")
    print("-" * 70)

    for name, balanced, dist in configs:
        status = "✅ OK" if balanced else "❌ VIÉS"
        print(f"{name:<25} {dist['hold']*100:>5.1f}%       {dist['long']*100:>5.1f}%       {dist['short']*100:>5.1f}%       {status}")

    print("\n" + "="*70)
    print("💡 RECOMENDAÇÃO:")
    print("="*70)
    print("• Opção 1: Melhor solução técnica (requer re-treino)")
    print("• Opção 2: Solução imediata (compatível com checkpoints atuais)")
    print("\nVeja FIX_SHORT_BIAS_PLAN.md para detalhes de implementação")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
