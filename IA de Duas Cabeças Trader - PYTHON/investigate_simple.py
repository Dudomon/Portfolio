#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO SIMPLES: Logs gerando valores incorretos?
"""

import json

def investigate_simple():
    print("🔍 INVESTIGAÇÃO SIMPLES - INTEGRIDADE DOS LOGS")
    print("=" * 60)

    files = [
        ("ANTES", "D:/Projeto/avaliacoes/training_20250925_155645_1092_c91f73d9.jsonl"),
        ("DEPOIS", "D:/Projeto/avaliacoes/training_20250925_182105_22376_6cd3fa56.jsonl")
    ]

    for label, filename in files:
        print(f"\n📊 {label}: {filename.split('/')[-1]}")
        print("-" * 40)

        try:
            training_entries = []

            with open(filename, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'explained_variance' in data:
                            training_entries.append(data)
                    except:
                        continue

            if not training_entries:
                print("❌ Nenhuma entrada de treinamento!")
                continue

            print(f"✅ {len(training_entries)} entradas encontradas")

            # Primeira e última entrada
            first = training_entries[0]
            last = training_entries[-1]

            print(f"\n🔍 PRIMEIRA ENTRADA (Step {first.get('step', 'N/A')}):")
            print(f"   exp_var: {first.get('explained_variance', 'N/A')}")
            print(f"   value_loss: {first.get('value_loss', 'N/A')}")
            print(f"   policy_loss: {first.get('policy_loss', 'N/A')}")
            print(f"   clip_frac: {first.get('clip_fraction', 'N/A')}")
            print(f"   approx_kl: {first.get('approx_kl', 'N/A')}")

            print(f"\n🔍 ÚLTIMA ENTRADA (Step {last.get('step', 'N/A')}):")
            print(f"   exp_var: {last.get('explained_variance', 'N/A')}")
            print(f"   value_loss: {last.get('value_loss', 'N/A')}")
            print(f"   policy_loss: {last.get('policy_loss', 'N/A')}")
            print(f"   clip_frac: {last.get('clip_fraction', 'N/A')}")
            print(f"   approx_kl: {last.get('approx_kl', 'N/A')}")

            # Contar zeros
            all_zero = 0
            for entry in training_entries:
                metrics = [
                    entry.get('explained_variance', 0),
                    entry.get('value_loss', 0),
                    entry.get('policy_loss', 0),
                    entry.get('clip_fraction', 0),
                    entry.get('approx_kl', 0)
                ]
                if all(m == 0 for m in metrics):
                    all_zero += 1

            zero_percent = (all_zero / len(training_entries)) * 100
            print(f"\n📊 ESTATÍSTICAS:")
            print(f"   Entradas com TODOS zeros: {all_zero}/{len(training_entries)} ({zero_percent:.1f}%)")

            if zero_percent > 80:
                print(f"   ❌ SUSPEITO: Muito alto!")
            elif zero_percent > 50:
                print(f"   ⚠️ ALTO: Possível problema")
            else:
                print(f"   ✅ NORMAL")

        except Exception as e:
            print(f"❌ Erro: {e}")

    print(f"\n🎯 PRÓXIMA INVESTIGAÇÃO:")
    print("1. Verificar se cherry.py está chamando model.learn()")
    print("2. Verificar se model está em mode .train() vs .eval()")
    print("3. Verificar se rewards têm variabilidade")

if __name__ == "__main__":
    investigate_simple()