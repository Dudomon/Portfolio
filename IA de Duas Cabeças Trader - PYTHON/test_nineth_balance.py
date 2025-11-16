"""
🧪 TESTE: Verificar se modelo vende em quedas e compra em altas

Testa o modelo em diferentes regimes de mercado para verificar se ele opera adequadamente
"""

import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Importar ambiente
sys.path.insert(0, 'D:/Projeto')
from cherry import TradingEnv, load_1m_dataset

def test_nineth_balance():
    """Testa resposta do Nineth 3.95M em diferentes regimes de mercado"""
    print("\n" + "=" * 80)
    print("🧪 TESTE DE RESPOSTA A REGIMES DE MERCADO - NINETH 3.95M")
    print("=" * 80)

    checkpoint_path = "D:/Projeto/Otimizacao/treino_principal/models/Nineth/Nineth_simpledirecttraining_3950000_steps_20251009_034940.zip"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint não encontrado: {checkpoint_path}")
        return False, 0.0, 0, 0

    print(f"\n📦 Carregando modelo: {checkpoint_path}")

    # Carregar dados históricos
    print("📊 Carregando dados históricos...")
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
        return

    # Estatísticas por regime
    regimes = {
        'uptrend': {'long': 0, 'short': 0, 'hold': 0},
        'downtrend': {'long': 0, 'short': 0, 'hold': 0},
        'sideways': {'long': 0, 'short': 0, 'hold': 0},
        'strong_downtrend': {'long': 0, 'short': 0, 'hold': 0},
        'strong_uptrend': {'long': 0, 'short': 0, 'hold': 0}
    }

    # Estatísticas detalhadas de raw_decision
    raw_decisions = []
    short_candidates = []  # Guardar casos onde raw_decision < -0.33

    # 🔥 TESTE MELHORADO: 2000 steps para capturar comportamentos raros
    print("\n📊 Executando 2000 steps para identificar regimes de mercado...")
    print("   (Teste expandido para capturar comportamentos raros como SHORT)\n")

    obs = env.reset()

    for step in range(2000):
        action, _states = model.predict(obs, deterministic=False)

        # Analisar ação - USAR MAPEAMENTO CORRETO
        raw_decision = action[0]  # -1 a +1
        raw_decisions.append(raw_decision)

        # Mapear conforme cherry.py
        if raw_decision < -0.33:  # SHORT
            action_type = 'short'
            short_candidates.append({
                'step': step,
                'raw_decision': raw_decision,
                'confidence': action[1],
                'price_change': None  # Será preenchido abaixo
            })
        elif raw_decision < 0.33:  # HOLD
            action_type = 'hold'
        else:  # LONG
            action_type = 'long'

        # Identificar regime de mercado (usando últimos 20 closes)
        current_idx = env.current_step
        if current_idx >= 20:
            recent_closes = df.iloc[current_idx-20:current_idx]['close_1m'].values
            price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

            # Atualizar short_candidates com price_change
            if action_type == 'short' and short_candidates:
                short_candidates[-1]['price_change'] = price_change

            # Classificar regime com mais granularidade
            if price_change > 0.005:  # Alta forte > 0.5%
                regime = 'strong_uptrend'
            elif price_change > 0.002:  # Alta > 0.2%
                regime = 'uptrend'
            elif price_change < -0.005:  # Queda forte > 0.5%
                regime = 'strong_downtrend'
            elif price_change < -0.002:  # Queda > 0.2%
                regime = 'downtrend'
            else:
                regime = 'sideways'

            # Registrar ação
            regimes[regime][action_type] += 1

            # Log de SHORTs (sempre, pois são raros!)
            if action_type == 'short':
                print(f"   ⚠️  Step {step} [{regime.upper()}]: 🔴 SHORT! (raw={raw_decision:.3f}, conf={action[1]:.2f}, change={price_change*100:.2f}%)")
            elif step % 500 == 0:
                print(f"   Step {step} [{regime.upper()}]: {'🟢 LONG' if action_type == 'long' else '⏸️  HOLD'} (raw={raw_decision:.3f})")

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    # Análise de resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS POR REGIME DE MERCADO")
    print("=" * 80)

    for regime_name, stats in regimes.items():
        total = stats['long'] + stats['short'] + stats['hold']
        if total == 0:
            continue

        print(f"\n📈 {regime_name.upper()}")
        print(f"   Total de ações: {total}")
        print(f"   🟢 LONG:  {stats['long']:3d} ({stats['long']/total*100:5.1f}%)")
        print(f"   🔴 SHORT: {stats['short']:3d} ({stats['short']/total*100:5.1f}%)")
        print(f"   ⏸️  HOLD:  {stats['hold']:3d} ({stats['hold']/total*100:5.1f}%)")

        # Calcular ratio
        if stats['short'] > 0:
            ratio = stats['long'] / stats['short']
            print(f"   📊 L/S Ratio: {ratio:.2f}")
        else:
            print(f"   📊 L/S Ratio: ∞ (zero shorts)")

    # Diagnóstico final
    print("\n" + "=" * 80)
    print("🔍 DIAGNÓSTICO DE ADEQUAÇÃO AO MERCADO")
    print("=" * 80)

    # Verificar UPTREND
    up_total = regimes['uptrend']['long'] + regimes['uptrend']['short']
    if up_total > 0:
        up_long_pct = regimes['uptrend']['long'] / up_total * 100
        if up_long_pct > 70:
            print("\n✅ UPTREND: Modelo prefere LONG corretamente")
        elif up_long_pct < 30:
            print("\n❌ UPTREND: Modelo prefere SHORT incorretamente!")
        else:
            print("\n⚠️ UPTREND: Modelo não tem preferência clara")

    # Verificar DOWNTREND
    down_total = regimes['downtrend']['long'] + regimes['downtrend']['short']
    if down_total > 0:
        down_short_pct = regimes['downtrend']['short'] / down_total * 100
        if down_short_pct > 70:
            print("✅ DOWNTREND: Modelo prefere SHORT corretamente")
        elif down_short_pct < 30:
            print("❌ DOWNTREND: Modelo prefere LONG incorretamente!")
        else:
            print("⚠️ DOWNTREND: Modelo não tem preferência clara")

        # CRÍTICO: Se não vende em queda, há problema
        if regimes['downtrend']['short'] == 0 and down_total > 10:
            print("\n🚨 PROBLEMA CRÍTICO: Modelo NÃO vende em quedas!")
            print("   Isso indica que o modelo pode não estar aprendendo a operar SHORT")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        test_nineth_balance()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
