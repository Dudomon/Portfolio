#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO FINAL DOS PROBLEMAS RESTANTES
============================================

Focar nos valores 0.5 suspeitos para garantir retreino 100% limpo
"""

import sys
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd

def investigate_remaining_issues():
    """🔍 INVESTIGAÇÃO CIRÚRGICA DOS VALORES 0.5 RESTANTES"""

    print("🔍 INVESTIGAÇÃO FINAL - VALORES 0.5 SUSPEITOS")
    print("=" * 70)

    try:
        from silus import TradingEnv

        # Criar DataFrame para teste
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'close_1m': 2000 + np.cumsum(np.random.randn(1000) * 0.1),
            'high_1m': None,
            'low_1m': None,
            'volume_1m': np.random.randint(500, 8000, 1000),
        })

        df['high_1m'] = df['close_1m'] + np.random.uniform(0, 5, 1000)
        df['low_1m'] = df['close_1m'] - np.random.uniform(0, 5, 1000)

        env = TradingEnv(df)
        env.reset()

        print(f"✅ Environment criado")

        # ANÁLISE CIRÚRGICA: MAPEAR EXATAMENTE ONDE ESTÃO OS 0.5
        step = 500
        single_bar = env._get_single_bar_features(step)

        print(f"\n📊 MAPEAMENTO PRECISO DOS VALORES 0.5:")
        print(f"Single bar shape: {single_bar.shape}")

        # Encontrar todas as posições com valor 0.5
        positions_05 = np.where(np.abs(single_bar - 0.5) < 1e-6)[0]
        print(f"\n🔍 Posições com valor 0.5: {list(positions_05)}")

        # Mapear por categoria
        categories = {
            'market_data': (0, 16),
            'positions': (16, 43),
            'intelligent': (43, 80),
            'resto': (80, len(single_bar))
        }

        for cat_name, (start, end) in categories.items():
            cat_positions = [pos for pos in positions_05 if start <= pos < end]
            if cat_positions:
                relative_positions = [pos - start for pos in cat_positions]
                print(f"   {cat_name}: posições absolutas {cat_positions} (relativas: {relative_positions})")

        # TESTAR CADA MÉTODO INDIVIDUALMENTE
        print(f"\n" + "="*50)
        print("🛠️ TESTE INDIVIDUAL DOS MÉTODOS GERADORES")
        print("="*50)

        methods_to_test = [
            ('_generate_intelligent_components_for_step', 37),
            ('_generate_fast_microstructure_features', 14),
            ('_generate_fast_volatility_features', 5),
            ('_generate_fast_correlation_features', 4),
            ('_generate_fast_momentum_features', 6),
            ('_generate_fast_enhanced_features', 20)
        ]

        all_method_results = {}

        for method_name, expected_size in methods_to_test:
            if hasattr(env, method_name):
                try:
                    method = getattr(env, method_name)
                    result = method(step)

                    # Contar valores 0.5
                    count_05 = np.sum(np.abs(result - 0.5) < 1e-6)
                    positions_05_method = np.where(np.abs(result - 0.5) < 1e-6)[0]

                    print(f"\n🔧 {method_name}:")
                    print(f"   Shape: {result.shape} (esperado: {expected_size})")
                    print(f"   Valores 0.5: {count_05}/{len(result)}")
                    if count_05 > 0:
                        print(f"   Posições com 0.5: {list(positions_05_method)}")
                        print(f"   🚨 PROBLEMA: Método ainda gera valores 0.5!")
                        print(f"   Valores: {result}")

                    all_method_results[method_name] = result

                except Exception as e:
                    print(f"   ❌ ERRO: {e}")
            else:
                print(f"\n⚠️ Método {method_name} não encontrado")

        # CONSTRUIR A SINGLE BAR MANUALMENTE PARA RASTREAR ORIGEM
        print(f"\n" + "="*50)
        print("🔍 CONSTRUÇÃO MANUAL DA SINGLE BAR")
        print("="*50)

        print(f"\n📊 Reconstruindo single bar step por step...")

        try:
            # Market data (primeiras 16)
            if step < len(env.processed_data):
                market_data = env.processed_data[step:step+1, :16].flatten()
            else:
                market_data = np.full(16, 0.3, dtype=np.float32)

            print(f"1. Market data: {market_data.shape}, valores 0.5: {np.sum(np.abs(market_data - 0.5) < 1e-6)}")

            # Positions (27 features)
            positions_obs = np.full((env.max_positions, 9), 0.001, dtype=np.float32)
            # ... (aplicar lógica de posições)
            positions_flat = positions_obs.flatten()

            print(f"2. Positions: {positions_flat.shape}, valores 0.5: {np.sum(np.abs(positions_flat - 0.5) < 1e-6)}")

            # Intelligent features
            intelligent_features = all_method_results.get('_generate_intelligent_components_for_step', np.full(37, 0.1))
            print(f"3. Intelligent: {intelligent_features.shape}, valores 0.5: {np.sum(np.abs(intelligent_features - 0.5) < 1e-6)}")

            # Outras features
            for method_name in ['_generate_fast_microstructure_features', '_generate_fast_volatility_features',
                              '_generate_fast_correlation_features', '_generate_fast_momentum_features',
                              '_generate_fast_enhanced_features']:
                if method_name in all_method_results:
                    features = all_method_results[method_name]
                    count_05 = np.sum(np.abs(features - 0.5) < 1e-6)
                    print(f"4. {method_name.split('_')[-2]}: {features.shape}, valores 0.5: {count_05}")
                    if count_05 > 0:
                        print(f"   🚨 ESTE MÉTODO É CULPADO!")

        except Exception as e:
            print(f"❌ Erro na construção manual: {e}")

        # TESTE DE MÚLTIPLOS STEPS PARA VER CONSISTÊNCIA
        print(f"\n" + "="*50)
        print("📈 TESTE DE CONSISTÊNCIA TEMPORAL")
        print("="*50)

        problem_methods = []
        for step_test in [100, 300, 500, 700]:
            bar = env._get_single_bar_features(step_test)
            count_05 = np.sum(np.abs(bar - 0.5) < 1e-6)
            print(f"Step {step_test}: {count_05} valores 0.5")

            if count_05 > 5:  # Threshold suspeito
                positions = np.where(np.abs(bar - 0.5) < 1e-6)[0]
                print(f"   Posições problemáticas: {list(positions)}")

        print(f"\n✅ INVESTIGAÇÃO FINAL COMPLETA!")

    except Exception as e:
        print(f"❌ ERRO NA INVESTIGAÇÃO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_remaining_issues()