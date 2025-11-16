#!/usr/bin/env python3
"""
🧪 TESTE DA CORREÇÃO DAS INTELLIGENT FEATURES
=============================================

Teste para verificar se as intelligent features agora são dinâmicas
ao invés de valores estáticos 0.4
"""

import sys
sys.path.append("D:/Projeto")

import numpy as np
from silus import TradingEnv

def test_intelligent_features_fix():
    """🧪 TESTAR SE INTELLIGENT FEATURES AGORA SÃO DINÂMICAS"""

    print("🧪 TESTANDO CORREÇÃO DAS INTELLIGENT FEATURES")
    print("=" * 60)

    try:
        # Criar environment
        env = TradingEnv(
            df_path="gold_1min_20250220_20250920.csv",
            base_tf='1m',
            sl_points=20,
            tp_points=40,
            episode_length=1000
        )

        # Reset environment
        obs = env.reset()
        print(f"✅ Environment criado. Observation shape: {obs.shape}")

        # Testar múltiplas observações para ver se são dinâmicas
        print("\n🔍 TESTANDO DINAMISMO DAS FEATURES...")

        observations = []
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            observations.append(obs.copy())

            if done:
                obs = env.reset()

        # Analisar as intelligent features em cada observação
        print(f"\n📊 ANÁLISE DAS INTELLIGENT FEATURES:")
        print(f"Observation space shape: {obs.shape}")
        print(f"Total dimensões: {len(obs)}")

        # Extrair intelligent features de cada barra
        # Como são 450D = 10 barras × 45 features, intelligent features estão nas posições específicas
        # market_data (16) + positions (27) = 43, então intelligent features começam na posição 43

        for obs_idx, observation in enumerate(observations):
            print(f"\n🔍 Observação {obs_idx + 1}:")

            # Extrair intelligent features de cada barra (10 barras)
            for barra in range(10):
                start_idx = barra * 45 + 43  # 43 = market_data + positions
                end_idx = start_idx + 37     # 37 intelligent features

                if end_idx <= len(observation):
                    intelligent_features = observation[start_idx:end_idx]

                    # Verificar se são todos iguais (estáticos) ou dinâmicos
                    unique_values = np.unique(np.round(intelligent_features, 4))
                    is_static = len(unique_values) <= 2  # Máximo 2 valores únicos = provável estático

                    print(f"  Barra {barra}: {len(unique_values)} valores únicos, "
                          f"range [{intelligent_features.min():.4f}, {intelligent_features.max():.4f}]"
                          f" {'🔴 ESTÁTICO' if is_static else '🟢 DINÂMICO'}")

        # Verificar se há diferenças entre observações
        print(f"\n📈 COMPARAÇÃO ENTRE OBSERVAÇÕES:")
        if len(observations) >= 2:
            obs1_intel = observations[0][43:43+37]  # Primeira barra da primeira obs
            obs2_intel = observations[1][43:43+37]  # Primeira barra da segunda obs

            diff = np.abs(obs1_intel - obs2_intel)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"  Diferença máxima: {max_diff:.6f}")
            print(f"  Diferença média: {mean_diff:.6f}")

            if max_diff > 0.001:
                print("  🟢 FEATURES DINÂMICAS - Correção funcionando!")
            else:
                print("  🔴 FEATURES AINDA ESTÁTICAS - Correção falhou!")

        print(f"\n✅ TESTE COMPLETO!")

    except Exception as e:
        print(f"❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intelligent_features_fix()