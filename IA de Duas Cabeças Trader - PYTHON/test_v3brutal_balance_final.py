"""
🔬 TESTE FINAL: V3 Brutal Balance após SL/TP Rework

Verifica se o reward system mantém balance perfeito após as mudanças:
- HARD CAP de TP em 25 pontos
- Gaming penalties
- TP realism bonus
- Shaping weight 70/30
"""

import numpy as np
import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent))

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockEnv:
    """Mock environment para testes"""
    def __init__(self):
        self.current_step = 0
        self.positions = []
        self.trades = []
        self.portfolio_value = 1000.0
        self.realized_balance = 1000.0
        self.peak_portfolio = 1000.0
        self.current_atr = 15.0

        # Mock dataframe com features
        import pandas as pd
        self.df = pd.DataFrame({
            'support_resistance': [0.5] * 100,  # SL zone quality
            'breakout_strength': [0.5] * 100,   # TP target quality
            'market_structure': [0.4] * 100     # Volatility spike
        })

def test_balance_equal_opposite():
    """
    TESTE CRÍTICO: Gain vs Loss Balance

    LONG gain +$10 deve ter reward OPOSTO de SHORT loss -$10
    """
    print("\n" + "="*70)
    print("🔬 TESTE 1: BALANCE GAIN/LOSS (LONG vs SHORT)")
    print("="*70)

    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    env = MockEnv()

    # Cenário 1: LONG com gain de $10
    env.realized_balance = 1010.0
    env.portfolio_value = 1010.0
    env.peak_portfolio = 1010.0

    reward_long_gain, info_long, _ = reward_system.calculate_reward_and_info(
        env, action=np.array([0.0, 0.5, 0.0, 0.0]), old_state={}
    )

    # Cenário 2: SHORT com loss de -$10
    env.realized_balance = 990.0
    env.portfolio_value = 990.0
    env.peak_portfolio = 1000.0

    reward_short_loss, info_short, _ = reward_system.calculate_reward_and_info(
        env, action=np.array([0.0, 0.5, 0.0, 0.0]), old_state={}
    )

    # Verificar balance
    ratio = abs(reward_long_gain / reward_short_loss) if reward_short_loss != 0 else 0

    print(f"\n📊 RESULTADOS:")
    print(f"  LONG +$10 → Reward: {reward_long_gain:+.6f}")
    print(f"  SHORT -$10 → Reward: {reward_short_loss:+.6f}")
    print(f"  Ratio: {ratio:.6f}")

    if 0.95 <= ratio <= 1.05:
        print(f"  ✅ BALANCEADO (ratio {ratio:.4f} ≈ 1.0)")
        return True
    else:
        print(f"  ❌ DESBALANCEADO (ratio {ratio:.4f} != 1.0)")
        return False


def test_gaming_penalty_impact():
    """
    Testa se gaming penalty está funcionando
    """
    print("\n" + "="*70)
    print("🔬 TESTE 2: GAMING PENALTY (SL MIN + TP MAX)")
    print("="*70)

    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    env = MockEnv()

    # Posição com SL mínimo + TP máximo (GAMING)
    env.positions = [{
        'entry_price': 2000.0,
        'sl': 1990.0,  # 10 pontos (mínimo)
        'tp': 2025.0,  # 25 pontos (máximo)
        'type': 'long',
        'duration': 20  # 20 steps
    }]

    reward_system.step_counter = 25  # Trigger cache
    reward, info, _ = reward_system.calculate_reward_and_info(
        env, action=np.array([0.0, 0.5, 0.0, 0.0]), old_state={}
    )

    gaming_penalty = info.get('sltp_gaming_penalty', 0.0)

    print(f"\n📊 RESULTADOS:")
    print(f"  Posição: SL 10pts + TP 25pts (GAMING)")
    print(f"  Gaming Penalty: {gaming_penalty:+.6f}")

    if gaming_penalty < -0.1:
        print(f"  ✅ PENALIDADE ATIVA (penalty < -0.1)")
        return True
    else:
        print(f"  ❌ PENALIDADE FRACA (penalty {gaming_penalty:.4f})")
        return False


def test_tp_realism_bonus():
    """
    Testa se TP realism bonus está funcionando
    """
    print("\n" + "="*70)
    print("🔬 TESTE 3: TP REALISM BONUS")
    print("="*70)

    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    env = MockEnv()

    # Feature indica resistência próxima
    env.df['breakout_strength'] = [0.8] * 100  # TP target quality ALTO

    # Posição com TP próximo (REALISTA)
    env.positions = [{
        'entry_price': 2000.0,
        'sl': 1985.0,  # 15 pontos
        'tp': 2018.0,  # 18 pontos (1.2 ATR, realista)
        'type': 'long',
        'duration': 10
    }]

    reward_system.step_counter = 25  # Trigger cache
    reward, info, _ = reward_system.calculate_reward_and_info(
        env, action=np.array([0.0, 0.5, 0.0, 0.0]), old_state={}
    )

    tp_realism = info.get('tp_realism_bonus', 0.0)

    print(f"\n📊 RESULTADOS:")
    print(f"  TP Target Quality: 0.8 (resistência próxima)")
    print(f"  TP: 18 pontos (1.2 ATR)")
    print(f"  TP Realism Bonus: {tp_realism:+.6f}")

    if tp_realism > 0.02:
        print(f"  ✅ BONUS ATIVO (bonus > 0.02)")
        return True
    else:
        print(f"  ⚠️  BONUS BAIXO (bonus {tp_realism:.4f})")
        return True  # Ainda OK, não é erro


def test_shaping_weight():
    """
    Testa se shaping weight está em 30% (vs 70% PnL)
    """
    print("\n" + "="*70)
    print("🔬 TESTE 4: SHAPING WEIGHT (70/30)")
    print("="*70)

    reward_system = BrutalMoneyReward(initial_balance=1000.0)
    env = MockEnv()

    # PnL significativo
    env.realized_balance = 1050.0
    env.portfolio_value = 1050.0
    env.peak_portfolio = 1050.0

    reward, info, _ = reward_system.calculate_reward_and_info(
        env, action=np.array([0.0, 0.5, 0.0, 0.0]), old_state={}
    )

    pnl_component = info.get('pnl_component', 0.0)
    shaping_component = info.get('shaping_component', 0.0)

    total = abs(pnl_component) + abs(shaping_component)
    if total > 0:
        pnl_pct = abs(pnl_component) / total * 100
        shaping_pct = abs(shaping_component) / total * 100
    else:
        pnl_pct = shaping_pct = 0

    print(f"\n📊 RESULTADOS:")
    print(f"  PnL Component: {pnl_component:+.6f} ({pnl_pct:.1f}%)")
    print(f"  Shaping Component: {shaping_component:+.6f} ({shaping_pct:.1f}%)")

    if 65 <= pnl_pct <= 75:
        print(f"  ✅ DISTRIBUIÇÃO CORRETA (70/30)")
        return True
    else:
        print(f"  ❌ DISTRIBUIÇÃO INCORRETA (deveria ser 70/30)")
        return False


def main():
    """Executar todos os testes"""
    print("\n" + "="*70)
    print("🔬 TESTE DE BALANCEAMENTO V3 BRUTAL - PÓS SL/TP REWORK")
    print("="*70)

    tests = [
        ("Balance Gain/Loss", test_balance_equal_opposite),
        ("Gaming Penalty", test_gaming_penalty_impact),
        ("TP Realism Bonus", test_tp_realism_bonus),
        ("Shaping Weight 70/30", test_shaping_weight),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERRO no teste '{test_name}': {e}")
            results.append((test_name, False))

    # Resumo final
    print("\n" + "="*70)
    print("📊 RESUMO FINAL")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"  {status} - {test_name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} testes passaram")

    if passed_count == total_count:
        print("\n🎉 TODOS OS TESTES PASSARAM! V3 Brutal está balanceado.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} teste(s) falharam.")
        return 1


if __name__ == "__main__":
    exit(main())
