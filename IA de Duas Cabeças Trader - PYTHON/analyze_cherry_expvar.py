#!/usr/bin/env python3
"""
Análise específica do explained_variance no cherry.py
"""
import json
import sys

def analyze_cherry_expvar():
    print("🍒 ANÁLISE EXPLAINED_VARIANCE - CHERRY.PY")
    print("=" * 60)

    filename = "D:/Projeto/avaliacoes/training_20250925_155645_1092_c91f73d9.jsonl"

    exp_var_values = []
    value_losses = []
    steps = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    exp_var = data.get('explained_variance', None)
                    value_loss = data.get('value_loss', None)
                    step = data.get('step', 0)

                    if exp_var is not None:
                        exp_var_values.append(exp_var)
                        steps.append(step)
                        if value_loss is not None:
                            value_losses.append(value_loss)

                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Log do cherry não encontrado: {filename}")
        return

    if not exp_var_values:
        print("❌ Nenhum explained_variance encontrado no cherry!")
        return

    print(f"\n📊 ÚLTIMOS 15 VALORES CHERRY:")
    recent_data = list(zip(steps[-15:], exp_var_values[-15:]))

    negative_count = 0
    for step, exp_var in recent_data:
        if exp_var < 0:
            negative_count += 1
            status = f"❌ NEGATIVO: {exp_var:.4f}"
        elif exp_var > 0.5:
            status = f"✅ EXCELENTE: {exp_var:.4f}"
        elif exp_var > 0:
            status = f"🟡 POSITIVO: {exp_var:.4f}"
        else:
            status = f"⚪ ZERO: {exp_var:.4f}"

        print(f"   Step {step:>7}: {status}")

    # Estatísticas
    total_values = len(exp_var_values)
    positive_count = sum(1 for v in exp_var_values if v > 0)
    negative_count = sum(1 for v in exp_var_values if v < 0)
    zero_count = sum(1 for v in exp_var_values if v == 0)

    avg_exp_var = sum(exp_var_values) / len(exp_var_values)
    recent_avg = sum(exp_var_values[-10:]) / min(10, len(exp_var_values))

    print(f"\n📈 ESTATÍSTICAS CHERRY:")
    print(f"   Total registros: {total_values}")
    print(f"   Positivos: {positive_count} ({positive_count/total_values*100:.1f}%)")
    print(f"   Negativos: {negative_count} ({negative_count/total_values*100:.1f}%)")
    print(f"   Zeros: {zero_count} ({zero_count/total_values*100:.1f}%)")
    print(f"   Média geral: {avg_exp_var:.4f}")
    print(f"   Média últimos 10: {recent_avg:.4f}")

    # Diagnóstico específico
    print(f"\n🔍 DIAGNÓSTICO CHERRY:")
    if negative_count / total_values > 0.7:
        print("   ❌ PROBLEMA: Majoritariamente negativo - value function overfitting")
        print("   💡 CAUSA POSSÍVEL: Rewards muito voláteis ou inconsistentes")
    elif zero_count / total_values > 0.8:
        print("   ❌ PROBLEMA: Majoritariamente zero - value function não treina")
        print("   💡 CAUSA POSSÍVEL: Configurações PPO muito restritivas")
    elif recent_avg < -0.2:
        print("   ⚠️  ATENÇÃO: Média recente muito negativa")
        print("   💡 CAUSA POSSÍVEL: Value function perdendo capacidade preditiva")
    else:
        print("   ✅ Status relativamente normal")

    if value_losses:
        avg_loss = sum(value_losses[-10:]) / min(10, len(value_losses))
        print(f"   📉 Value loss média (últimos 10): {avg_loss:.4f}")

if __name__ == "__main__":
    analyze_cherry_expvar()