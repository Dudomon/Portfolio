#!/usr/bin/env python3
"""
🧪 TESTE SIMPLES DAS INTELLIGENT FEATURES
========================================
"""

import sys
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd

def test_intelligent_features_simple():
    """🧪 TESTE DIRETO DOS MÉTODOS DE INTELLIGENT FEATURES"""

    print("🧪 TESTANDO INTELLIGENT FEATURES DIRETAMENTE")
    print("=" * 60)

    try:
        # Importar classe
        from silus import TradingEnv

        # Criar DataFrame básico para teste
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'close_1m': 2000 + np.random.randn(1000) * 10,
            'high_1m': 2000 + np.random.randn(1000) * 10 + 5,
            'low_1m': 2000 + np.random.randn(1000) * 10 - 5,
            'volume_1m': np.random.randint(1000, 5000, 1000),
            'sma_20_1m': 2000 + np.random.randn(1000) * 5,
        })

        # Criar environment básico
        env = TradingEnv(df)
        env.reset()

        print(f"✅ Environment criado com {len(df)} barras")

        # Testar métodos diretamente
        print("\n🔍 TESTANDO MÉTODOS DE INTELLIGENT FEATURES...")

        # Testar método novo
        step = 100
        try:
            intel_features_new = env._generate_intelligent_components_for_step(step)
            print(f"✅ _generate_intelligent_components_for_step: {intel_features_new.shape}")
            print(f"   Range: [{intel_features_new.min():.4f}, {intel_features_new.max():.4f}]")
            print(f"   Unique values: {len(np.unique(np.round(intel_features_new, 4)))}")

            if len(np.unique(np.round(intel_features_new, 4))) > 2:
                print("   🟢 DINÂMICO!")
            else:
                print("   🔴 ESTÁTICO!")

        except Exception as e:
            print(f"❌ Erro no método novo: {e}")

        # Testar método fallback
        try:
            intel_features_fallback = env._generate_basic_v7_components(step)
            print(f"✅ _generate_basic_v7_components: {intel_features_fallback.shape}")
            print(f"   Range: [{intel_features_fallback.min():.4f}, {intel_features_fallback.max():.4f}]")
            print(f"   Unique values: {len(np.unique(np.round(intel_features_fallback, 4)))}")

            if len(np.unique(np.round(intel_features_fallback, 4))) > 2:
                print("   🟢 DINÂMICO!")
            else:
                print("   🔴 ESTÁTICO!")

        except Exception as e:
            print(f"❌ Erro no método fallback: {e}")

        # Testar geração de single bar features
        try:
            single_bar = env._get_single_bar_features(step)
            print(f"✅ _get_single_bar_features: {single_bar.shape}")

            # Extrair intelligent features da single bar (posições 43-80)
            if len(single_bar) >= 80:
                intel_from_bar = single_bar[43:80]  # 37 features
                print(f"   Intelligent features extraídas: {len(intel_from_bar)}")
                print(f"   Range: [{intel_from_bar.min():.4f}, {intel_from_bar.max():.4f}]")
                print(f"   Unique values: {len(np.unique(np.round(intel_from_bar, 4)))}")

                if len(np.unique(np.round(intel_from_bar, 4))) > 2:
                    print("   🟢 DINÂMICO!")
                else:
                    print("   🔴 ESTÁTICO!")
            else:
                print(f"   ⚠️ Single bar muito pequena: {len(single_bar)} features")

        except Exception as e:
            print(f"❌ Erro na single bar: {e}")

        # Testar múltiplos steps para ver variação
        print("\n📊 TESTANDO VARIAÇÃO ENTRE STEPS...")
        features_list = []

        for test_step in [50, 100, 150, 200, 250]:
            try:
                features = env._generate_basic_v7_components(test_step)
                features_list.append(features[:5])  # Primeiras 5 features
                print(f"Step {test_step}: {features[:5]}")
            except:
                print(f"Step {test_step}: ERRO")

        if len(features_list) >= 2:
            diff = np.abs(features_list[0] - features_list[1])
            print(f"\nDiferença entre steps: max={diff.max():.6f}, mean={diff.mean():.6f}")

            if diff.max() > 0.001:
                print("🟢 FEATURES VARIAM ENTRE STEPS - CORREÇÃO FUNCIONANDO!")
            else:
                print("🔴 FEATURES IGUAIS ENTRE STEPS - AINDA ESTÁTICAS!")

        print(f"\n✅ TESTE SIMPLES COMPLETO!")

    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intelligent_features_simple()