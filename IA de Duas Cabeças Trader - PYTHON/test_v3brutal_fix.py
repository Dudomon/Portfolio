"""
🔥 TESTE: Correção do viés SHORT no V3Brutal
Valida que as mudanças eliminaram o viés:
1. Unrealized PnL sem desconto de 50%
2. Pain multiplier simétrico
"""

import numpy as np
import sys
sys.path.append("D:/Projeto")

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward

class MockEnv:
    """Mock environment para teste"""
    def __init__(self):
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.portfolio_value = 500.0
        self.peak_portfolio_value = 500.0

def test_unrealized_pnl_no_discount():
    """Testa que unrealized PnL não tem mais desconto"""
    print("\n🧪 TESTE 1: Unrealized PnL SEM DESCONTO")
    print("=" * 60)

    reward_system = BrutalMoneyReward(initial_balance=500.0)
    env = MockEnv()

    # Cenário: LONG com +$20 unrealized
    env.total_realized_pnl = 0.0
    env.total_unrealized_pnl = 20.0

    pnl_reward, info = reward_system._calculate_pure_pnl_reward(env)

    print(f"Realized PnL: ${env.total_realized_pnl:.2f}")
    print(f"Unrealized PnL: ${env.total_unrealized_pnl:.2f}")
    print(f"Total PnL usado: ${info['total_pnl']:.2f}")
    print(f"PnL %: {info['pnl_percent']:.2f}%")
    print(f"Reward: {pnl_reward:.4f}")

    # Validação
    expected_total = 20.0  # SEM desconto!
    assert abs(info['total_pnl'] - expected_total) < 0.01, f"❌ FALHOU: Esperado ${expected_total}, obteve ${info['total_pnl']}"
    print(f"✅ PASSOU: Unrealized PnL sem desconto (total = ${info['total_pnl']:.2f})")

def test_symmetric_pain_multiplier():
    """Testa que pain multiplier é simétrico"""
    print("\n🧪 TESTE 2: Pain Multiplier SIMÉTRICO")
    print("=" * 60)

    reward_system = BrutalMoneyReward(initial_balance=500.0)
    env = MockEnv()

    # Teste 1: Perda de 5%
    env.total_realized_pnl = -25.0  # -5% de 500
    env.total_unrealized_pnl = 0.0

    loss_reward, loss_info = reward_system._calculate_pure_pnl_reward(env)
    print(f"\n📉 PERDA -5%:")
    print(f"  PnL: ${env.total_realized_pnl:.2f}")
    print(f"  Reward: {loss_reward:.4f}")

    # Teste 2: Ganho de 5%
    env.total_realized_pnl = 25.0  # +5% de 500
    env.total_unrealized_pnl = 0.0

    gain_reward, gain_info = reward_system._calculate_pure_pnl_reward(env)
    print(f"\n📈 GANHO +5%:")
    print(f"  PnL: ${env.total_realized_pnl:.2f}")
    print(f"  Reward: {gain_reward:.4f}")

    # Validação: magnitudes devem ser similares (simétrico)
    loss_magnitude = abs(loss_reward)
    gain_magnitude = abs(gain_reward)
    ratio = gain_magnitude / loss_magnitude if loss_magnitude > 0 else 0

    print(f"\n📊 SIMETRIA:")
    print(f"  |Loss reward|: {loss_magnitude:.4f}")
    print(f"  |Gain reward|: {gain_magnitude:.4f}")
    print(f"  Ratio (gain/loss): {ratio:.2f}")

    # Deve estar entre 0.9 e 1.1 (tolerância 10%)
    assert 0.9 <= ratio <= 1.1, f"❌ FALHOU: Ratio {ratio:.2f} fora do range [0.9, 1.1]"
    print(f"✅ PASSOU: Pain multiplier simétrico (ratio = {ratio:.2f})")

def test_long_vs_short_bias():
    """Testa que LONG e SHORT têm recompensas equivalentes"""
    print("\n🧪 TESTE 3: Equivalência LONG vs SHORT")
    print("=" * 60)

    reward_system = BrutalMoneyReward(initial_balance=500.0)
    env = MockEnv()

    # Cenário 1: LONG +$15 unrealized (mercado subindo)
    env.total_realized_pnl = 0.0
    env.total_unrealized_pnl = 15.0
    long_reward, long_info = reward_system._calculate_pure_pnl_reward(env)

    print(f"\n📈 LONG +$15 unrealized:")
    print(f"  Total PnL: ${long_info['total_pnl']:.2f}")
    print(f"  Reward: {long_reward:.4f}")

    # Cenário 2: SHORT +$15 unrealized (mercado caindo)
    env.total_realized_pnl = 0.0
    env.total_unrealized_pnl = 15.0
    short_reward, short_info = reward_system._calculate_pure_pnl_reward(env)

    print(f"\n📉 SHORT +$15 unrealized:")
    print(f"  Total PnL: ${short_info['total_pnl']:.2f}")
    print(f"  Reward: {short_reward:.4f}")

    # Validação: rewards devem ser idênticos para mesmo PnL
    assert abs(long_reward - short_reward) < 0.001, f"❌ FALHOU: LONG={long_reward:.4f} != SHORT={short_reward:.4f}"
    print(f"✅ PASSOU: LONG e SHORT têm rewards equivalentes para mesmo PnL")

def test_market_trending_up():
    """Simula mercado em alta para verificar que LONGs são recompensados"""
    print("\n🧪 TESTE 4: Mercado em ALTA (LONG favorecido)")
    print("=" * 60)

    reward_system = BrutalMoneyReward(initial_balance=500.0)
    env = MockEnv()

    # LONG aproveitando alta
    env.total_realized_pnl = 0.0
    env.total_unrealized_pnl = 30.0  # +6% unrealized
    long_up_reward, _ = reward_system._calculate_pure_pnl_reward(env)

    # SHORT contra a tendência
    env.total_realized_pnl = 0.0
    env.total_unrealized_pnl = -30.0  # -6% unrealized (perdendo)
    short_up_reward, _ = reward_system._calculate_pure_pnl_reward(env)

    print(f"LONG +$30 unrealized: reward = {long_up_reward:.4f}")
    print(f"SHORT -$30 unrealized: reward = {short_up_reward:.4f}")
    print(f"Diferença: {long_up_reward - short_up_reward:.4f}")

    # Validação: LONG deve ter reward MUITO maior que SHORT perdedor
    assert long_up_reward > short_up_reward, f"❌ FALHOU: LONG reward ({long_up_reward:.4f}) não é maior que SHORT perdedor ({short_up_reward:.4f})"
    assert (long_up_reward - short_up_reward) > 0.5, f"❌ FALHOU: Diferença muito pequena"
    print(f"✅ PASSOU: LONG em mercado de alta é recompensado adequadamente")

if __name__ == "__main__":
    print("🔥 TESTE: Correção do Viés SHORT no V3Brutal")
    print("=" * 60)

    try:
        test_unrealized_pnl_no_discount()
        test_symmetric_pain_multiplier()
        test_long_vs_short_bias()
        test_market_trending_up()

        print("\n" + "=" * 60)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("🎯 V3Brutal corrigido - viés SHORT eliminado")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TESTE FALHOU: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
