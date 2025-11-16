#!/usr/bin/env python3
"""
🕵️ INVESTIGAÇÃO COMPLETA DE FEATURES FALSAS/ESTÁTICAS
===================================================

Análise sistemática de todo o observation space para detectar:
1. Valores constantes/estáticos
2. Features com padding artificial
3. Dados sintéticos mascarando falta de dados reais
"""

import sys
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd

def investigate_all_features():
    """🕵️ INVESTIGAÇÃO SISTEMÁTICA DE TODAS AS FEATURES"""

    print("🕵️ INVESTIGAÇÃO COMPLETA DE FEATURES FALSAS/ESTÁTICAS")
    print("=" * 80)

    try:
        from silus import TradingEnv

        # Criar DataFrame realista
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2000, freq='1min'),
            'close_1m': 2000 + np.cumsum(np.random.randn(2000) * 0.1),  # Random walk realista
            'high_1m': None,  # Vamos ver como lida com dados ausentes
            'low_1m': None,
            'volume_1m': np.random.randint(500, 8000, 2000),
            'sma_20_1m': None,  # Dados técnicos ausentes
        })

        # Adicionar high/low baseado em close
        df['high_1m'] = df['close_1m'] + np.random.uniform(0, 5, 2000)
        df['low_1m'] = df['close_1m'] - np.random.uniform(0, 5, 2000)

        env = TradingEnv(df)
        env.reset()

        print(f"✅ Environment criado com {len(df)} barras")
        print(f"📊 Observation space: {env.observation_space.shape}")

        # ANÁLISE 1: DETECTAR FEATURES ESTÁTICAS POR CATEGORIA
        print("\n" + "="*60)
        print("📊 ANÁLISE 1: CATEGORIZAÇÃO DAS FEATURES")
        print("="*60)

        step = 500
        single_bar = env._get_single_bar_features(step)

        print(f"Single bar shape: {single_bar.shape} (esperado: 45)")

        # Categorizar features por posição conhecida
        categories = {
            'market_data': (0, 16),      # Primeiras 16 features
            'positions': (16, 43),       # 27 features de posições (3x9)
            'intelligent': (43, 80),     # 37 features inteligentes
            'microstructure': (80, 85),  # ~5 features (se existir)
            'volatility': (85, 90),      # ~5 features (se existir)
            'correlation': (90, 95),     # ~5 features (se existir)
            'momentum': (95, 100),       # ~5 features (se existir)
            'enhanced': (100, 105),      # ~5 features (se existir)
        }

        for cat_name, (start, end) in categories.items():
            if end <= len(single_bar):
                features = single_bar[start:end]
                unique_count = len(np.unique(np.round(features, 6)))

                print(f"\n🔍 {cat_name.upper()}:")
                print(f"   Posições: {start}-{end-1} ({end-start} features)")
                print(f"   Range: [{features.min():.6f}, {features.max():.6f}]")
                print(f"   Valores únicos: {unique_count}/{len(features)}")
                print(f"   Primeiros 5 valores: {features[:5]}")

                # Detectar problemas
                if unique_count <= 2:
                    print(f"   🚨 SUSPEITO: Muito poucos valores únicos!")
                elif np.all(np.abs(features - features[0]) < 1e-6):
                    print(f"   🚨 ESTÁTICO: Todos os valores iguais!")
                elif len(features) > 5 and np.std(features) < 1e-6:
                    print(f"   🚨 QUASI-ESTÁTICO: Desvio padrão muito baixo!")
                else:
                    print(f"   ✅ OK: Aparenta ser dinâmico")

        # ANÁLISE 2: TESTAR VARIAÇÃO TEMPORAL
        print("\n" + "="*60)
        print("📈 ANÁLISE 2: VARIAÇÃO TEMPORAL DAS FEATURES")
        print("="*60)

        steps_to_test = [100, 300, 500, 700, 900]
        all_bars = []

        for step in steps_to_test:
            try:
                bar = env._get_single_bar_features(step)
                all_bars.append(bar)
                print(f"Step {step}: OK ({len(bar)} features)")
            except Exception as e:
                print(f"Step {step}: ERRO - {e}")

        if len(all_bars) >= 2:
            print(f"\n🔍 ANÁLISE DE VARIAÇÃO TEMPORAL:")

            for cat_name, (start, end) in categories.items():
                if end <= len(all_bars[0]):
                    # Extrair categoria de todas as barras
                    cat_features = [bar[start:end] for bar in all_bars]

                    # Calcular variação máxima
                    if len(cat_features) >= 2:
                        max_diff = 0
                        for i in range(len(cat_features)-1):
                            diff = np.abs(cat_features[i] - cat_features[i+1])
                            max_diff = max(max_diff, np.max(diff))

                        print(f"\n📊 {cat_name.upper()}:")
                        print(f"   Variação máxima entre steps: {max_diff:.8f}")

                        if max_diff < 1e-8:
                            print(f"   🚨 COMPLETAMENTE ESTÁTICO!")
                        elif max_diff < 1e-6:
                            print(f"   ⚠️ QUASI-ESTÁTICO (variação mínima)")
                        elif max_diff < 0.001:
                            print(f"   📈 BAIXA VARIAÇÃO (suspeito)")
                        else:
                            print(f"   ✅ BOA VARIAÇÃO (dinâmico)")

        # ANÁLISE 3: DETECTAR PADRÕES DE PADDING/FALLBACK
        print("\n" + "="*60)
        print("🔧 ANÁLISE 3: DETECÇÃO DE PADDING E FALLBACKS")
        print("="*60)

        # Verificar valores específicos que indicam fallbacks
        suspicious_values = [0.001, 0.01, 0.1, 0.25, 0.35, 0.4, 0.5, 1.0]

        for bar_idx, bar in enumerate(all_bars[:3]):  # Primeiras 3 barras
            print(f"\n🔍 BARRA {steps_to_test[bar_idx]}:")

            for val in suspicious_values:
                count = np.sum(np.abs(bar - val) < 1e-6)
                if count > 0:
                    positions = np.where(np.abs(bar - val) < 1e-6)[0]
                    print(f"   Valor {val}: {count} ocorrências nas posições {positions[:10]}")

                    if count > 5:
                        print(f"     🚨 SUSPEITO: Muitas ocorrências do valor {val}!")

        # ANÁLISE 4: VERIFICAR MÉTODOS DE GERAÇÃO
        print("\n" + "="*60)
        print("🛠️ ANÁLISE 4: MÉTODOS DE GERAÇÃO DE FEATURES")
        print("="*60)

        step = 500

        # Testar métodos individuais
        methods_to_test = [
            '_generate_fast_microstructure_features',
            '_generate_fast_volatility_features',
            '_generate_fast_correlation_features',
            '_generate_fast_momentum_features',
            '_generate_fast_enhanced_features'
        ]

        for method_name in methods_to_test:
            if hasattr(env, method_name):
                try:
                    method = getattr(env, method_name)
                    result = method(step)

                    unique_count = len(np.unique(np.round(result, 6)))

                    print(f"\n🔧 {method_name}:")
                    print(f"   Shape: {result.shape}")
                    print(f"   Range: [{result.min():.6f}, {result.max():.6f}]")
                    print(f"   Valores únicos: {unique_count}/{len(result)}")
                    print(f"   Valores: {result}")

                    if unique_count <= 2:
                        print(f"   🚨 MÉTODO SUSPEITO: Gera poucos valores únicos!")
                    elif np.std(result) < 1e-6:
                        print(f"   🚨 MÉTODO ESTÁTICO: Desvio padrão muito baixo!")
                    else:
                        print(f"   ✅ MÉTODO OK")

                except Exception as e:
                    print(f"   ❌ ERRO no método {method_name}: {e}")
            else:
                print(f"   ⚠️ Método {method_name} não encontrado")

        print(f"\n" + "="*80)
        print("✅ INVESTIGAÇÃO COMPLETA!")
        print("="*80)

    except Exception as e:
        print(f"❌ ERRO NA INVESTIGAÇÃO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_all_features()