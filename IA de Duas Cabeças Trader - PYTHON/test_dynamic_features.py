"""
🧪 TESTE: Validar que features NÃO são mais estáticas
Verifica que breakout_strength, trend_consistency, support_resistance e market_structure
têm valores VARIÁVEIS (não constantes) no dataset
"""

import sys
import numpy as np
import pandas as pd

# Importar ambiente Cherry
from cherry import TradingEnv, load_optimized_data_original

def test_dynamic_features():
    """Teste rápido: Features devem ter variância > 0"""
    print("=" * 70)
    print("🧪 TESTE: Validando features dinâmicas (não-estáticas)")
    print("=" * 70)

    # Carregar dataset
    print("\n[1/4] Carregando dataset...")
    df = load_optimized_data_original()
    print(f"✅ Dataset carregado: {len(df):,} barras")

    # Criar environment
    print("\n[2/4] Criando environment Cherry...")
    env = TradingEnv(df=df)

    # Verificar se features existem no dataset
    print("\n[3/4] Verificando features no dataset...")
    required_features = [
        'volume_momentum',
        'price_position',
        'breakout_strength',
        'trend_consistency',
        'support_resistance',
        'volatility_regime',
        'market_structure'
    ]

    missing_features = [f for f in required_features if f not in env.df.columns]
    if missing_features:
        print(f"❌ ERRO: Features ausentes: {missing_features}")
        return False

    print("✅ Todas as 7 features existem no dataset")

    # Testar se features são dinâmicas (não constantes)
    print("\n[4/4] Testando variância das features...")
    results = {}

    for feature in required_features:
        values = env.df[feature].values

        # Calcular estatísticas
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        unique_count = len(np.unique(values))

        # Feature é dinâmica se:
        # 1. Tem variância (std > 0.001)
        # 2. Tem múltiplos valores únicos (> 10)
        is_dynamic = std_val > 0.001 and unique_count > 10

        results[feature] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'unique': unique_count,
            'dynamic': is_dynamic
        }

        status = "✅ DINÂMICA" if is_dynamic else "❌ ESTÁTICA"
        print(f"  {feature:25s} {status}")
        print(f"    Mean: {mean_val:.4f} | Std: {std_val:.4f} | Unique: {unique_count:,}")

    # Resultado final
    print("\n" + "=" * 70)
    static_features = [f for f, r in results.items() if not r['dynamic']]

    if static_features:
        print(f"❌ FALHA: {len(static_features)} features ESTÁTICAS detectadas:")
        for f in static_features:
            print(f"   - {f}")
        return False
    else:
        print("✅ SUCESSO: Todas as 7 features são DINÂMICAS!")
        print("\n📊 Estatísticas detalhadas:")
        for feature, stats in results.items():
            print(f"\n  {feature}:")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Valores únicos: {stats['unique']:,}")
        return True

if __name__ == "__main__":
    try:
        success = test_dynamic_features()
        print("\n" + "=" * 70)
        if success:
            print("🎉 TESTE PASSOU: Features corrigidas com sucesso!")
            sys.exit(0)
        else:
            print("💥 TESTE FALHOU: Features ainda estáticas")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
