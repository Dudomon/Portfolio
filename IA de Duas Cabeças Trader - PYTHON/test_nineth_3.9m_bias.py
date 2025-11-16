"""APLICACAO: Detectar vies SHORT no modelo Nineth 3.95M"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Garantir que o projeto esteja no path de import
sys.path.insert(0, 'D:/Projeto')
from cherry import TradingEnv, ACTION_THRESHOLD_SHORT, ACTION_THRESHOLD_LONG

CONFIDENCE_THRESHOLD = 0.8  # Mesmo filtro do ambiente cherry.TradingEnv


def classify_intent(raw_decision: float) -> str:
    """Classifica a intencao da acao usando os thresholds do ambiente de treino."""
    if raw_decision <= ACTION_THRESHOLD_SHORT:
        return "short"
    if raw_decision >= ACTION_THRESHOLD_LONG:
        return "long"
    return "hold"


def format_percent(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def test_short_bias_nineth(
    steps: int = 100,
    deterministic: bool = False,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
):
    """Executa o teste de vies SHORT reproduzindo a logica do ambiente cherry."""
    print("=" * 70)
    print("TESTE DE VIES SHORT - NINETH 3.95M")
    print("=" * 70)

    checkpoint_path = (
        "D:/Projeto/Otimizacao/treino_principal/models/Nineth/"
        "Nineth_simpledirecttraining_3950000_steps_20251009_034940.zip"
    )

    if not os.path.exists(checkpoint_path):
        print(f"ERRO: Checkpoint nao encontrado: {checkpoint_path}")
        return False, 0.0, 0, 0

    print(f"Carregando modelo: {checkpoint_path}")

    from cherry import load_1m_dataset

    print("Carregando dataset 1M neutro...")
    df = load_1m_dataset()
    print(f"Dataset carregado com {len(df)} barras")

    env = TradingEnv(df=df, window_size=10)

    try:
        model = PPO.load(checkpoint_path, env=env)
        print("Modelo carregado com sucesso")
    except Exception as exc:
        print(f"ERRO ao carregar modelo: {exc}")
        return False, 0.0, 0, 0

    total_actions = 0
    intent_counts = {"long": 0, "short": 0, "hold": 0}
    qualified_counts = {"long": 0, "short": 0}
    executed_counts = {"long": 0, "short": 0}
    blocked_counts = {"long": 0, "short": 0}

    confidence_min = float("inf")
    confidence_max = float("-inf")
    entry_signal_min = float("inf")
    entry_signal_max = float("-inf")
    mgmt_min = float("inf")
    mgmt_max = float("-inf")

    print(f"Executando {steps} steps (deterministic={deterministic})...")
    obs = env.reset()
    log_interval = max(1, steps // 12)

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        raw_decision = float(action[0])
        entry_confidence = float(action[1])
        mgmt_1 = float(action[2])
        mgmt_2 = float(action[3])

        confidence_min = min(confidence_min, entry_confidence)
        confidence_max = max(confidence_max, entry_confidence)
        entry_signal_min = min(entry_signal_min, raw_decision)
        entry_signal_max = max(entry_signal_max, raw_decision)
        mgmt_min = min(mgmt_min, mgmt_1, mgmt_2)
        mgmt_max = max(mgmt_max, mgmt_1, mgmt_2)

        total_actions += 1

        intent = classify_intent(raw_decision)
        intent_counts[intent] += 1

        qualified_intent = None
        if intent in ("long", "short") and entry_confidence >= confidence_threshold:
            qualified_counts[intent] += 1
            qualified_intent = intent

        if step % log_interval == 0:
            print(
                f" step {step:4d}: raw={raw_decision:+.2f} conf={entry_confidence:.2f} "
                f"intent={intent:<5} mgmt=({mgmt_1:+.2f},{mgmt_2:+.2f})"
            )

        before = len(env.positions)
        obs, reward, done, info = env.step(action)
        after = len(env.positions)

        executed = False
        if after > before:
            for pos in env.positions[before:]:
                pos_type = pos.get("type")
                if pos_type in executed_counts:
                    executed_counts[pos_type] += 1
                    executed = True

        if not executed and qualified_intent:
            max_positions = getattr(env, "max_positions", 0)
            slot_cooldowns = getattr(env, "slot_cooldowns", {})
            occupied_slots = {
                pos.get("position_id")
                for pos in env.positions
                if isinstance(pos.get("position_id"), int)
            }
            has_free_slot = any(
                (slot not in occupied_slots) and slot_cooldowns.get(slot, 0) == 0
                for slot in range(max_positions)
            )
            if not has_free_slot and qualified_intent in blocked_counts:
                blocked_counts[qualified_intent] += 1

        if done:
            obs = env.reset()

    print("\nRESULTADOS DO TESTE")
    print("-" * 70)
    print(f"Total de acoes: {total_actions}")
    print(
        f"Intencoes LONG:  {intent_counts['long']} "
        f"({format_percent(intent_counts['long'], total_actions)})"
    )
    print(
        f"Intencoes SHORT: {intent_counts['short']} "
        f"({format_percent(intent_counts['short'], total_actions)})"
    )
    print(
        f"Hold:            {intent_counts['hold']} "
        f"({format_percent(intent_counts['hold'], total_actions)})"
    )

    qualified_total = qualified_counts['long'] + qualified_counts['short']
    print("\nIntencoes qualificadas (conf >= {:.2f}):".format(confidence_threshold))
    print(
        f" LONG qualificados:  {qualified_counts['long']} "
        f"({format_percent(qualified_counts['long'], max(qualified_total, 1))})"
    )
    print(
        f" SHORT qualificados: {qualified_counts['short']} "
        f"({format_percent(qualified_counts['short'], max(qualified_total, 1))})"
    )

    executed_total = executed_counts['long'] + executed_counts['short']
    print("\nEntradas executadas (novas posicoes abertas):")
    print(
        f" LONG executados:  {executed_counts['long']} "
        f"({format_percent(executed_counts['long'], max(executed_total, 1))})"
    )
    print(
        f" SHORT executados: {executed_counts['short']} "
        f"({format_percent(executed_counts['short'], max(executed_total, 1))})"
    )

    if blocked_counts['long'] or blocked_counts['short']:
        print("\nBloqueios por slots/cooldown (intencoes qualificadas nao executadas):")
        print(f" LONG bloqueados:  {blocked_counts['long']}")
        print(f" SHORT bloqueados: {blocked_counts['short']}")

    print("\nIntervalos observados:")
    print(f" entry_signal: [{entry_signal_min:+.2f}, {entry_signal_max:+.2f}]")
    print(f" entry_confidence: [{confidence_min:.2f}, {confidence_max:.2f}]")
    print(f" management: [{mgmt_min:+.2f}, {mgmt_max:+.2f}]")

    if executed_counts['short'] > 0:
        long_short_ratio = executed_counts['long'] / executed_counts['short']
    else:
        long_short_ratio = float('inf') if executed_counts['long'] > 0 else 1.0

    print(f"\nLONG/SHORT ratio (execucao): {long_short_ratio:.2f}")

    if long_short_ratio > 1.5:
        print("Diagnostico: preferencia por LONG (sem vies short observado)")
        if qualified_counts['short'] > 0 and executed_counts['short'] == 0:
            print(
                "OBS: houve intencoes short com confianca >= filtro, "
                "mas nenhuma foi executada (slots/cooldown)."
            )
    elif long_short_ratio < 0.67:
        print("Diagnostico: vies SHORT detectado (mais shorts do que longs)")
    else:
        print("Diagnostico: distribuicao equilibrada")

    total_entries = executed_counts['long'] + executed_counts['short']
    if total_entries >= 10:
        expected = total_entries / 2
        chi_square = 0.0
        chi_square += (executed_counts['long'] - expected) ** 2 / expected
        chi_square += (executed_counts['short'] - expected) ** 2 / expected
        print(f"Chi-square: {chi_square:.2f}")
        if chi_square > 3.84:
            print("  Diferenca estatisticamente significativa (p < 0.05)")
        else:
            print("  Diferenca nao significativa (p > 0.05)")

    has_short_bias = long_short_ratio < 0.67
    return has_short_bias, long_short_ratio, executed_counts['long'], executed_counts['short']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de vies SHORT para o Nineth 3.95M")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Quantidade de steps a executar no ambiente neutro",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Usa politica deterministica ao inves de estocastica",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Threshold de confianca para considerar entrada executavel",
    )
    args = parser.parse_args()

    try:
        has_bias, ratio, longs, shorts = test_short_bias_nineth(
            steps=args.steps,
            deterministic=args.deterministic,
            confidence_threshold=args.confidence,
        )

        print("\nRESULTADO FINAL:")
        if has_bias:
            print(f" VIES SHORT DETECTADO (ratio={ratio:.2f})")
            print(f" LONG executados: {longs} | SHORT executados: {shorts}")
            sys.exit(1)
        else:
            print(f" Sem vies short (ratio={ratio:.2f})")
            print(f" LONG executados: {longs} | SHORT executados: {shorts}")
            sys.exit(0)
    except Exception as exc:
        print(f"ERRO durante o teste: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




