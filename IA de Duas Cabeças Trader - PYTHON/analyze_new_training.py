#!/usr/bin/env python3
"""
Análise dos logs após mudanças no cherry.py - clip_fraction, kl e exp_var
"""
import json
import sys

def analyze_post_changes():
    print("🔍 ANÁLISE PÓS-MUDANÇAS CHERRY.PY")
    print("=" * 60)

    # Log mais recente
    filename = "D:/Projeto/avaliacoes/training_20250925_192821_14440_c9981196.jsonl"

    metrics = {
        'explained_variance': [],
        'value_loss': [],
        'clip_fraction': [],
        'approx_kl': [],
        'policy_loss': [],
        'step': []
    }

    try:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    for key in metrics.keys():
                        if key in data and data[key] is not None:
                            metrics[key].append(data[key])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Log não encontrado: {filename}")
        return

    if not metrics['step']:
        print("❌ Nenhum dado encontrado no log!")
        return

    print(f"\n📊 DADOS COLETADOS:")
    for key, values in metrics.items():
        if values:
            print(f"   {key}: {len(values)} valores")

    # Análise dos últimos valores
    print(f"\n🎯 ÚLTIMOS VALORES (10 entradas):")
    print("-" * 50)

    n_recent = min(10, len(metrics['step']))

    for i in range(-n_recent, 0):
        step = metrics['step'][i] if i < len(metrics['step']) else 'N/A'
        exp_var = metrics['explained_variance'][i] if i < len(metrics['explained_variance']) else 'N/A'
        value_loss = metrics['value_loss'][i] if i < len(metrics['value_loss']) else 'N/A'
        clip_frac = metrics['clip_fraction'][i] if i < len(metrics['clip_fraction']) else 'N/A'
        approx_kl = metrics['approx_kl'][i] if i < len(metrics['approx_kl']) else 'N/A'

        print(f"Step {step}:")
        print(f"  exp_var: {exp_var}")
        print(f"  value_loss: {value_loss}")
        print(f"  clip_frac: {clip_frac}")
        print(f"  approx_kl: {approx_kl}")
        print()

    # Estatísticas comparativas
    print(f"🔍 ANÁLISE COMPARATIVA:")
    print("-" * 50)

    if metrics['explained_variance']:
        exp_var_zeros = sum(1 for x in metrics['explained_variance'] if x == 0)
        exp_var_total = len(metrics['explained_variance'])
        zero_percent = (exp_var_zeros / exp_var_total) * 100
        print(f"📊 Explained Variance:")
        print(f"   Zeros: {exp_var_zeros}/{exp_var_total} ({zero_percent:.1f}%)")
        print(f"   Média: {sum(metrics['explained_variance'])/len(metrics['explained_variance']):.4f}")

    if metrics['clip_fraction']:
        avg_clip = sum(metrics['clip_fraction']) / len(metrics['clip_fraction'])
        print(f"📊 Clip Fraction:")
        print(f"   Média: {avg_clip:.4f}")
        if avg_clip > 0.3:
            print("   ❌ ALTO - Updates muito agressivos")
        elif avg_clip < 0.1:
            print("   ⚠️ BAIXO - Updates muito conservadores")
        else:
            print("   ✅ OK - Range adequado")

    if metrics['approx_kl']:
        avg_kl = sum(metrics['approx_kl']) / len(metrics['approx_kl'])
        max_kl = max(metrics['approx_kl'])
        print(f"📊 Approx KL:")
        print(f"   Média: {avg_kl:.4f}")
        print(f"   Máximo: {max_kl:.4f}")
        if avg_kl > 0.03:
            print("   ❌ ALTO - Provável early stopping")
        elif max_kl > 0.05:
            print("   ⚠️ PICOS ALTOS - Early stopping intermitente")
        else:
            print("   ✅ OK - Dentro do target_kl")

    if metrics['value_loss']:
        recent_value_loss = metrics['value_loss'][-5:] if len(metrics['value_loss']) >= 5 else metrics['value_loss']
        avg_recent_value_loss = sum(recent_value_loss) / len(recent_value_loss)
        print(f"📊 Value Loss (últimos 5):")
        print(f"   Média recente: {avg_recent_value_loss:.6f}")
        if avg_recent_value_loss < 0.001:
            print("   ❌ MUITO BAIXO - Value function não treina")
        elif avg_recent_value_loss > 0.1:
            print("   ⚠️ ALTO - Possível instabilidade")
        else:
            print("   ✅ OK - Value function ativo")

    print(f"\n🎯 DIAGNÓSTICO:")
    print("-" * 50)

    # Diagnóstico específico
    if metrics['explained_variance']:
        zero_percent = (sum(1 for x in metrics['explained_variance'] if x == 0) / len(metrics['explained_variance'])) * 100
        if zero_percent > 90:
            print("❌ PROBLEMA PERSISTE: 90%+ explained_variance = 0")
            print("   Mudanças não resolveram o problema do value function")
        elif zero_percent > 70:
            print("⚠️ MELHORIA PARCIAL: Ainda muitos zeros")
        else:
            print("✅ MELHORIA: Explained variance mais variável")

    if metrics['clip_fraction'] and metrics['approx_kl']:
        avg_clip = sum(metrics['clip_fraction']) / len(metrics['clip_fraction'])
        avg_kl = sum(metrics['approx_kl']) / len(metrics['approx_kl'])

        if avg_clip > 0.25:
            print("❌ CLIP FRACTION ALTO: LR/batch_size mudanças causaram updates agressivos")
        if avg_kl > 0.04:
            print("❌ KL ALTO: Early stopping frequente, value function ainda não treina")

    print("=" * 60)

if __name__ == "__main__":
    analyze_post_changes()