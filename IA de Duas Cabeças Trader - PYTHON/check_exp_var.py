#!/usr/bin/env python3
"""
Análise rápida do explained_variance nos logs de treinamento
"""
import json
import sys

def analyze_exp_var(filename):
    print(f"🔍 Analisando explained_variance em: {filename}")
    print("=" * 60)

    exp_var_values = []
    steps = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    exp_var = data.get('explained_variance', None)
                    step = data.get('step', 0)

                    if exp_var is not None:
                        exp_var_values.append(exp_var)
                        steps.append(step)

                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filename}")
        return

    if not exp_var_values:
        print("❌ Nenhum explained_variance encontrado!")
        return

    # Últimos 20 valores
    recent_values = list(zip(steps[-20:], exp_var_values[-20:]))

    print(f"\n📊 ÚLTIMOS 20 VALORES:")
    for step, exp_var in recent_values:
        status = "✅" if exp_var > 0 else "❌" if exp_var < -0.1 else "⚠️"
        print(f"   Step {step:>7}: {exp_var:>8.4f} {status}")

    # Estatísticas gerais
    total_values = len(exp_var_values)
    positive_count = sum(1 for v in exp_var_values if v > 0)
    negative_count = sum(1 for v in exp_var_values if v < 0)
    zero_count = sum(1 for v in exp_var_values if v == 0)

    avg_exp_var = sum(exp_var_values) / len(exp_var_values)
    recent_avg = sum(exp_var_values[-10:]) / min(10, len(exp_var_values))

    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"   Total de registros: {total_values}")
    print(f"   Positivos: {positive_count} ({positive_count/total_values*100:.1f}%)")
    print(f"   Negativos: {negative_count} ({negative_count/total_values*100:.1f}%)")
    print(f"   Zeros: {zero_count} ({zero_count/total_values*100:.1f}%)")
    print(f"   Média geral: {avg_exp_var:.4f}")
    print(f"   Média últimos 10: {recent_avg:.4f}")

    # Verificar se smoothing teve impacto
    if len(exp_var_values) > 50:
        old_avg = sum(exp_var_values[-50:-25]) / 25
        new_avg = sum(exp_var_values[-25:]) / 25

        print(f"\n🔄 IMPACTO DO SMOOTHING:")
        print(f"   Média antes (25 valores): {old_avg:.4f}")
        print(f"   Média depois (25 valores): {new_avg:.4f}")
        print(f"   Mudança: {new_avg - old_avg:+.4f}")

        if abs(new_avg - old_avg) < 0.05:
            print("   ⚠️  IMPACTO PEQUENO/IMPERCEPTÍVEL")
        elif new_avg > old_avg:
            print("   ✅ MELHORIA DETECTADA")
        else:
            print("   ❌ POSSÍVEL PIORA")

if __name__ == "__main__":
    analyze_exp_var("D:/Projeto/avaliacoes/training_20250925_105123_5776_490aa7db.jsonl")