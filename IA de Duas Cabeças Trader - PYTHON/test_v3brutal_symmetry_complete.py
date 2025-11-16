"""
🧪 TESTE COMPLETO: Simetria V3Brutal - LONG vs SHORT

Simula cenários IDÊNTICOS para LONG e SHORT e compara rewards.
Se houver assimetria, encontraremos ONDE está o viés.
"""

import sys
import numpy as np
from unittest.mock import Mock
import pandas as pd

sys.path.insert(0, 'D:/Projeto')
from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

def create_mock_env_long(pnl_usd, sl_distance, tp_distance, entry_price=2000.0):
    """Cria mock env para LONG position"""
    env = Mock()

    # Portfolio
    env.initial_balance = 500.0
    env.portfolio_value = 500.0 + pnl_usd
    env.peak_portfolio_value = 500.0
    env.total_realized_pnl = pnl_usd
    env.total_unrealized_pnl = 0.0

    # Position LONG
    env.positions = [{
        'type': 'long',
        'entry_price': entry_price,
        'sl': entry_price - sl_distance,
        'tp': entry_price + tp_distance,
        'unrealized_pnl': pnl_usd,
        'duration': 10
    }]

    # DataFrame com trend
    env.df = pd.DataFrame({
        'trend_consistency': [0.6] * 20,  # Mercado neutro
        'returns_1m': [0.0] * 20,
        'close_1m': [entry_price] * 20,
        'support_resistance': [0.5] * 20,
        'breakout_strength': [0.5] * 20
    })
    env.current_step = 10

    env.trades = []
    env.current_price = entry_price

    return env

def create_mock_env_short(pnl_usd, sl_distance, tp_distance, entry_price=2000.0):
    """Cria mock env para SHORT position - CENÁRIO IDÊNTICO ao LONG"""
    env = Mock()

    # Portfolio IDÊNTICO
    env.initial_balance = 500.0
    env.portfolio_value = 500.0 + pnl_usd
    env.peak_portfolio_value = 500.0
    env.total_realized_pnl = pnl_usd
    env.total_unrealized_pnl = 0.0

    # Position SHORT - MESMO PnL, MESMAS distâncias SL/TP
    env.positions = [{
        'type': 'short',
        'entry_price': entry_price,
        'sl': entry_price + sl_distance,  # SHORT: SL acima
        'tp': entry_price - tp_distance,  # SHORT: TP abaixo
        'unrealized_pnl': pnl_usd,
        'duration': 10
    }]

    # DataFrame IDÊNTICO
    env.df = pd.DataFrame({
        'trend_consistency': [0.6] * 20,  # Mercado neutro IDÊNTICO
        'returns_1m': [0.0] * 20,
        'close_1m': [entry_price] * 20,
        'support_resistance': [0.5] * 20,
        'breakout_strength': [0.5] * 20
    })
    env.current_step = 10

    env.trades = []
    env.current_price = entry_price

    return env

def test_scenario(scenario_name, pnl_usd, sl_distance, tp_distance):
    """Testa um cenário específico comparando LONG vs SHORT"""
    print(f"\n{'='*80}")
    print(f"📊 CENÁRIO: {scenario_name}")
    print(f"{'='*80}")
    print(f"   PnL: ${pnl_usd:.2f}")
    print(f"   SL distance: {sl_distance:.1f}pt")
    print(f"   TP distance: {tp_distance:.1f}pt")

    # Criar envs
    env_long = create_mock_env_long(pnl_usd, sl_distance, tp_distance)
    env_short = create_mock_env_short(pnl_usd, sl_distance, tp_distance)

    # Calcular rewards
    reward_calc = BrutalMoneyReward(initial_balance=500.0)

    # Action dummy (4D)
    action = np.array([0.0, 0.5, 0.0, 0.0])  # Neutro

    # OLD STATE dummy
    old_state = {'portfolio': 500.0}

    # Calcular rewards
    reward_long, info_long, _ = reward_calc.calculate_reward_and_info(env_long, action, old_state)

    # Reset para SHORT
    reward_calc_short = BrutalMoneyReward(initial_balance=500.0)
    reward_short, info_short, _ = reward_calc_short.calculate_reward_and_info(env_short, action, old_state)

    # Comparar
    print(f"\n📊 RESULTADOS:")
    print(f"   🟢 LONG reward:  {reward_long:.6f}")
    print(f"   🔴 SHORT reward: {reward_short:.6f}")

    diff = reward_long - reward_short
    diff_pct = abs(diff) / (abs(reward_long) + 0.0001) * 100

    print(f"\n   📐 Diferença: {diff:.6f} ({diff_pct:.2f}%)")

    # Diagnóstico
    if abs(diff) < 0.001:
        print(f"   ✅ SIMÉTRICO (diferença < 0.001)")
        return True
    elif abs(diff_pct) < 5:
        print(f"   ⚠️ QUASE SIMÉTRICO (diferença < 5%)")
        return True
    else:
        print(f"   🚨 ASSIMETRIA DETECTADA! (diferença > 5%)")
        print(f"\n   🔍 DETALHES LONG:")
        for key, val in info_long.items():
            if isinstance(val, (int, float)):
                print(f"      {key}: {val:.6f}")
        print(f"\n   🔍 DETALHES SHORT:")
        for key, val in info_short.items():
            if isinstance(val, (int, float)):
                print(f"      {key}: {val:.6f}")
        return False

def main():
    print("\n" + "="*80)
    print("🧪 TESTE COMPLETO DE SIMETRIA V3BRUTAL - LONG vs SHORT")
    print("="*80)
    print("\nTestando cenários IDÊNTICOS para LONG e SHORT...")
    print("Se houver viés, reward será diferente em situações equivalentes.")

    all_symmetric = True

    # TESTE 1: Lucro pequeno
    symmetric = test_scenario(
        "Lucro pequeno (+$10)",
        pnl_usd=10.0,
        sl_distance=12.0,
        tp_distance=15.0
    )
    all_symmetric = all_symmetric and symmetric

    # TESTE 2: Prejuízo pequeno
    symmetric = test_scenario(
        "Prejuízo pequeno (-$10)",
        pnl_usd=-10.0,
        sl_distance=12.0,
        tp_distance=15.0
    )
    all_symmetric = all_symmetric and symmetric

    # TESTE 3: Lucro grande
    symmetric = test_scenario(
        "Lucro grande (+$50)",
        pnl_usd=50.0,
        sl_distance=10.0,
        tp_distance=18.0
    )
    all_symmetric = all_symmetric and symmetric

    # TESTE 4: Prejuízo grande
    symmetric = test_scenario(
        "Prejuízo grande (-$50)",
        pnl_usd=-50.0,
        sl_distance=10.0,
        tp_distance=18.0
    )
    all_symmetric = all_symmetric and symmetric

    # TESTE 5: SL apertado
    symmetric = test_scenario(
        "SL apertado (10pt)",
        pnl_usd=5.0,
        sl_distance=10.0,
        tp_distance=15.0
    )
    all_symmetric = all_symmetric and symmetric

    # TESTE 6: TP distante
    symmetric = test_scenario(
        "TP distante (18pt)",
        pnl_usd=5.0,
        sl_distance=12.0,
        tp_distance=18.0
    )
    all_symmetric = all_symmetric and symmetric

    # TESTE 7: Neutro (sem PnL)
    symmetric = test_scenario(
        "Neutro (0 PnL)",
        pnl_usd=0.0,
        sl_distance=12.0,
        tp_distance=15.0
    )
    all_symmetric = all_symmetric and symmetric

    # Resultado final
    print("\n" + "="*80)
    print("🎯 RESULTADO FINAL:")
    print("="*80)

    if all_symmetric:
        print("✅ V3BRUTAL É SIMÉTRICO")
        print("   Rewards LONG e SHORT são equivalentes em cenários idênticos")
        print("   Viés SHORT não está nos rewards!")
        return 0
    else:
        print("🚨 V3BRUTAL TEM ASSIMETRIA!")
        print("   Rewards LONG e SHORT são DIFERENTES em cenários idênticos")
        print("   VIÉS SHORT ESTÁ NOS REWARDS!")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
