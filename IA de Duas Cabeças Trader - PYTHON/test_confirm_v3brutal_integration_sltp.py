#!/usr/bin/env python3
"""
Teste para confirmar se as heurísticas SL/TP do V3Brutal influenciam o total_reward.

Hipótese: o total_reward usa apenas PnL + "proportional shaping" (baseado em
portfolio/action), ignorando o shaping rico de SL/TP (tp_hit, near_miss, etc.).

Estratégia:
- Criar dois sistemas idênticos de reward e aquecer 4 passos sem trades.
- No passo 5, em um caso, injetar trade com close_reason='tp_hit' (distância realista),
  no outro caso, nenhum trade.
- Manter PnL, portfolio_value e action idênticos nos dois casos.
- Verificar: (a) _calculate_reward_shaping retorna maior valor no caso TP-hit; (b) total_reward
  retornado por calculate_reward_and_info no passo 5 é (quase) idêntico entre os casos.
"""

import numpy as np

from trading_framework.rewards.reward_daytrade_v3_brutal import BrutalMoneyReward


class MockEnv:
    def __init__(self, *,
                 realized_pnl: float = 0.0,
                 unrealized_pnl: float = 0.0,
                 portfolio_value: float = 1000.0,
                 peak_value: float = 1000.0,
                 trades=None,
                 positions=None,
                 current_step: int = 0,
                 last_action=None,
                 current_price: float = 0.0,
                 df=None,
                 training_progress: float = 0.0):
        self.total_realized_pnl = realized_pnl
        self.total_unrealized_pnl = unrealized_pnl
        self.portfolio_value = portfolio_value
        self.peak_portfolio_value = peak_value
        self.trades = trades or []
        self.positions = positions or []
        self.current_step = current_step
        self.last_action = np.array(last_action if last_action is not None else [0.2, 0.0, -0.2])
        self.current_price = current_price
        self.df = df
        self.training_progress = training_progress


def warmup_steps(rw: BrutalMoneyReward, steps: int = 4):
    # Aquecer passando envs neutros (sem trades), mantendo PnL estável
    for t in range(steps):
        env = MockEnv(realized_pnl=10.0, unrealized_pnl=0.0, portfolio_value=1000.0, peak_value=1000.0,
                      trades=[], positions=[], current_step=t)
        rw.calculate_reward_and_info(env, env.last_action, {})


def main():
    # Instâncias separadas para controlar passo 5 de forma comparável
    rw_no_tp = BrutalMoneyReward(initial_balance=1000.0)
    rw_with_tp = BrutalMoneyReward(initial_balance=1000.0)

    warmup_steps(rw_no_tp, 4)
    warmup_steps(rw_with_tp, 4)

    # Passo 5: mesmo PnL e action para ambos
    action = np.array([0.2, 0.0, -0.2])

    # Caso A: sem TP-hit
    env_no_tp = MockEnv(realized_pnl=10.0, unrealized_pnl=0.0,
                        portfolio_value=1000.0, peak_value=1000.0,
                        trades=[], positions=[], current_step=5)

    # Caso B: com TP-hit (distância ~14 pontos)
    trade_tp_hit = {
        'close_reason': 'tp_hit',
        'entry_price': 2000.0,
        'exit_price': 2014.0,
        'type': 'long'
    }
    env_with_tp = MockEnv(realized_pnl=10.0, unrealized_pnl=0.0,
                          portfolio_value=1000.0, peak_value=1000.0,
                          trades=[trade_tp_hit], positions=[], current_step=5)

    # 1) Medir shaping interno (direto) para demonstrar diferença
    shaping_no_tp, _ = rw_no_tp._calculate_reward_shaping(env_no_tp, action)
    shaping_with_tp, _ = rw_with_tp._calculate_reward_shaping(env_with_tp, action)

    # 2) Medir total_reward oficial (que deve ignorar a diferença acima)
    total_no_tp, info_no_tp, _ = rw_no_tp.calculate_reward_and_info(env_no_tp, action, {})
    total_with_tp, info_with_tp, _ = rw_with_tp.calculate_reward_and_info(env_with_tp, action, {})

    print("\n=== Confirmação de Integração SL/TP no Total Reward ===")
    print(f"Shaping interno sem TP:   {shaping_no_tp:.6f}")
    print(f"Shaping interno com TP:   {shaping_with_tp:.6f}  (deveria ser > sem TP)")
    print(f"Total reward sem TP:      {total_no_tp:.6f}")
    print(f"Total reward com TP:      {total_with_tp:.6f}  (esperado ~igual ao sem TP)")

    # Critério: se a diferença absoluta do total < 1e-6, confirmamos suspeita
    diff = abs(total_with_tp - total_no_tp)
    print(f"Diferença absoluta total: {diff:.8f}")

    if diff < 1e-6 and shaping_with_tp > shaping_no_tp:
        print("\n✅ Confirmado: heurísticas SL/TP alteram o shaping interno, mas NÃO entram no total_reward.")
    else:
        print("\n⚠️ Resultado inesperado: reveja integração. Valores acima mostram o comportamento real.")


if __name__ == "__main__":
    main()
