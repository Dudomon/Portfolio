"""
🧪 TESTE: Detectar viés SHORT no modelo Oitavo 4.7M

Testa o modelo em cenários neutros para verificar se há preferência por SHORT
"""

import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Importar ambiente
sys.path.insert(0, 'D:/Projeto')
from cherry import TradingEnv

def test_short_bias_oitavo():
    """Testa viés SHORT no Oitavo 4.7M"""
    print("\n" + "=" * 70)
    print("🧪 TESTE DE VIÉS SHORT - OITAVO 4.7M")
    print("=" * 70)

    checkpoint_path = "D:/Projeto/Otimizacao/treino_principal/models/Oitavo/Oitavo_simpledirecttraining_4700000_steps_20251008_042753.zip"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint não encontrado: {checkpoint_path}")
        return False, 0.0, 0, 0

    print(f"\n📦 Carregando modelo: {checkpoint_path}")

    # Carregar dados históricos
    print("📊 Carregando dados históricos...")
    from cherry import load_1m_dataset
    df = load_1m_dataset()
    print(f"✅ Dados carregados: {len(df)} barras")

    # Criar ambiente
    env = TradingEnv(df=df, window_size=10)

    # Carregar modelo
    try:
        model = PPO.load(checkpoint_path, env=env)
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False, 0.0, 0, 0

    # Estatísticas
    total_actions = 0
    entry_long = 0
    entry_short = 0
    hold_actions = 0

    # 🔥 TESTE: 100 steps em mercado NEUTRO (sem tendência clara)
    print("\n📊 Executando 100 steps em mercado neutro...")

    obs = env.reset()

    for step in range(100):
        action, _states = model.predict(obs, deterministic=False)

        # Analisar ação (Legion V1 format: [entry_signal, position_type, risk_appetite, entry_quality])
        entry_signal = action[0]  # -1 a +1
        position_type = action[1]  # -1=SHORT, 0=NEUTRAL, +1=LONG

        total_actions += 1

        # Detectar tipo de ação
        if entry_signal > 0.5:  # Quer entrar
            if position_type > 0.3:  # LONG
                entry_long += 1
                if step % 20 == 0:
                    print(f"   Step {step}: 🟢 LONG (entry={entry_signal:.2f}, type={position_type:.2f})")
            elif position_type < -0.3:  # SHORT
                entry_short += 1
                if step % 20 == 0:
                    print(f"   Step {step}: 🔴 SHORT (entry={entry_signal:.2f}, type={position_type:.2f})")
            else:
                hold_actions += 1
        else:
            hold_actions += 1

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    # Análise de viés
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DO TESTE")
    print("=" * 70)

    print(f"\n📈 Total de ações analisadas: {total_actions}")
    print(f"   🟢 Entries LONG: {entry_long} ({entry_long/total_actions*100:.1f}%)")
    print(f"   🔴 Entries SHORT: {entry_short} ({entry_short/total_actions*100:.1f}%)")
    print(f"   ⏸️  HOLD: {hold_actions} ({hold_actions/total_actions*100:.1f}%)")

    # Calcular ratio
    if entry_short > 0:
        long_short_ratio = entry_long / entry_short
    else:
        long_short_ratio = float('inf') if entry_long > 0 else 1.0

    print(f"\n📊 LONG/SHORT Ratio: {long_short_ratio:.2f}")

    # Diagnóstico
    print("\n" + "=" * 70)
    print("🔍 DIAGNÓSTICO DE VIÉS")
    print("=" * 70)

    if long_short_ratio > 1.5:
        print("✅ SEM VIÉS SHORT - Modelo prefere LONG")
        print(f"   Ratio {long_short_ratio:.2f} indica preferência por LONG")
    elif long_short_ratio < 0.67:
        print("🚨 VIÉS SHORT DETECTADO!")
        print(f"   Ratio {long_short_ratio:.2f} indica preferência por SHORT")
        print("   ⚠️ Modelo está fazendo mais SHORTs que LONGs em mercado neutro")
    else:
        print("✅ BALANCEADO - Sem viés significativo")
        print(f"   Ratio {long_short_ratio:.2f} está próximo de 1.0 (ideal)")

    # Calcular chi-square test (se houver entries)
    total_entries = entry_long + entry_short
    if total_entries >= 10:
        expected = total_entries / 2
        chi_square = ((entry_long - expected)**2 / expected) + ((entry_short - expected)**2 / expected)
        print(f"\n📐 Chi-Square Test: {chi_square:.2f}")
        if chi_square > 3.84:  # p < 0.05
            print("   ⚠️ Diferença estatisticamente significativa (p < 0.05)")
        else:
            print("   ✅ Diferença NÃO significativa (p > 0.05)")

    print("\n" + "=" * 70)

    # Retornar resultado
    has_short_bias = long_short_ratio < 0.67
    return has_short_bias, long_short_ratio, entry_long, entry_short

if __name__ == "__main__":
    try:
        has_bias, ratio, longs, shorts = test_short_bias_oitavo()

        print("\n🎯 RESULTADO FINAL:")
        if has_bias:
            print(f"❌ VIÉS SHORT DETECTADO (Ratio: {ratio:.2f})")
            print(f"   LONG: {longs} | SHORT: {shorts}")
            sys.exit(1)
        else:
            print(f"✅ SEM VIÉS SHORT (Ratio: {ratio:.2f})")
            print(f"   LONG: {longs} | SHORT: {shorts}")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
