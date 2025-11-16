#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO: Logs gerando valores incorretos?
Verificar se o problema está na coleta/logging das métricas
"""

import json
import os

def investigate_log_integrity():
    print("🔍 INVESTIGAÇÃO: INTEGRIDADE DOS LOGS")
    print("=" * 60)

    # Vamos comparar diferentes logs para ver padrões
    log_files = [
        "D:/Projeto/avaliacoes/training_20250925_155645_1092_c91f73d9.jsonl",  # Antes das mudanças
        "D:/Projeto/avaliacoes/training_20250925_182105_22376_6cd3fa56.jsonl"   # Depois das mudanças
    ]

    for i, filename in enumerate(log_files):
        print(f"\n📊 ANÁLISE LOG {i+1}: {os.path.basename(filename)}")
        print("-" * 50)

        if not os.path.exists(filename):
            print(f"❌ Arquivo não encontrado!")
            continue

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            total_lines = len(lines)
            training_lines = []

            for line_num, line in enumerate(lines, 1):
                try:
                    data = json.loads(line.strip())

                    # Verificar se é linha de treinamento
                    if 'explained_variance' in data:
                        training_lines.append((line_num, data))

                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON inválido na linha {line_num}: {e}")
                    continue
        except Exception as e:
            print(f"❌ Erro ao processar arquivo: {e}")
            continue

        print(f"📈 Total de linhas: {total_lines}")
        print(f"📈 Linhas de treinamento: {len(training_lines)}")

        if not training_lines:
            print("❌ NENHUMA linha de treinamento encontrada!")
            continue

        # Análise das primeiras e últimas 5 linhas de treinamento
        print(f"\n🔍 PRIMEIRAS 5 LINHAS DE TREINAMENTO:")
            for j, (line_num, data) in enumerate(training_lines[:5]):
                step = data.get('step', 'N/A')
                exp_var = data.get('explained_variance', 'N/A')
                policy_loss = data.get('policy_loss', 'N/A')
                value_loss = data.get('value_loss', 'N/A')
                clip_frac = data.get('clip_fraction', 'N/A')
                approx_kl = data.get('approx_kl', 'N/A')

                print(f"  Linha {line_num}: Step={step}")
                print(f"    exp_var={exp_var}, policy_loss={policy_loss}")
                print(f"    value_loss={value_loss}, clip_frac={clip_frac}, kl={approx_kl}")

            print(f"\n🔍 ÚLTIMAS 5 LINHAS DE TREINAMENTO:")
            for j, (line_num, data) in enumerate(training_lines[-5:]):
                step = data.get('step', 'N/A')
                exp_var = data.get('explained_variance', 'N/A')
                policy_loss = data.get('policy_loss', 'N/A')
                value_loss = data.get('value_loss', 'N/A')
                clip_frac = data.get('clip_fraction', 'N/A')
                approx_kl = data.get('approx_kl', 'N/A')

                print(f"  Linha {line_num}: Step={step}")
                print(f"    exp_var={exp_var}, policy_loss={policy_loss}")
                print(f"    value_loss={value_loss}, clip_frac={clip_frac}, kl={approx_kl}")

            # Detectar padrões suspeitos
            print(f"\n🚨 DETECÇÃO DE PADRÕES SUSPEITOS:")

            all_zeros_count = 0
            all_same_count = 0
            missing_fields_count = 0

            prev_values = None

            for line_num, data in training_lines:
                # Contar zeros absolutos
                key_metrics = ['explained_variance', 'policy_loss', 'value_loss', 'clip_fraction', 'approx_kl']
                current_values = [data.get(key, None) for key in key_metrics]

                # Zeros absolutos
                if all(val == 0 for val in current_values if val is not None):
                    all_zeros_count += 1

                # Valores idênticos consecutivos
                if prev_values is not None and current_values == prev_values:
                    all_same_count += 1

                # Campos ausentes
                missing = [key for key in key_metrics if key not in data or data[key] is None]
                if missing:
                    missing_fields_count += 1

                prev_values = current_values

            print(f"   🔴 Linhas com TODOS valores = 0: {all_zeros_count}/{len(training_lines)} ({all_zeros_count/len(training_lines)*100:.1f}%)")
            print(f"   🔴 Linhas idênticas consecutivas: {all_same_count}/{len(training_lines)} ({all_same_count/len(training_lines)*100:.1f}%)")
            print(f"   🔴 Linhas com campos ausentes: {missing_fields_count}/{len(training_lines)} ({missing_fields_count/len(training_lines)*100:.1f}%)")

            # Diagnóstico
            if all_zeros_count > len(training_lines) * 0.8:
                print(f"   ❌ SUSPEITO: 80%+ das linhas têm todos valores zerados")
            if all_same_count > len(training_lines) * 0.5:
                print(f"   ❌ SUSPEITO: 50%+ das linhas são idênticas consecutivas")
            if missing_fields_count > len(training_lines) * 0.1:
                print(f"   ❌ SUSPEITO: 10%+ das linhas têm campos ausentes")

    # Análise comparativa
    print(f"\n🔬 HIPÓTESES SOBRE O PROBLEMA:")
    print("-" * 50)
    print("1. 📝 PROBLEMA NO LOGGING:")
    print("   - Cherry.py pode estar logando valores default/zero")
    print("   - Callback de logging pode estar capturando métricas vazias")
    print("   - Timing issue: logging antes das métricas serem calculadas")

    print("\n2. 🧠 PROBLEMA NO PPO:")
    print("   - Model.learn() não está executando updates reais")
    print("   - Stable-Baselines3 pode estar com problema interno")
    print("   - Policy não está sendo atualizada")

    print("\n3. 🔄 PROBLEMA NO ENVIRONMENT:")
    print("   - Experiences não estão sendo coletadas corretamente")
    print("   - Rewards são constantes → sem gradiente → sem update")
    print("   - Observations são constantes → sem aprendizado")

    print("\n4. 💾 PROBLEMA DE CHECKPOINT:")
    print("   - Model carregado está congelado")
    print("   - Parâmetros não estão sendo atualizados")
    print("   - Gradientes bloqueados")

    print("\n🎯 PRÓXIMOS STEPS PARA DIAGNOSTICAR:")
    print("-" * 50)
    print("1. Verificar se cherry.py está realmente chamando model.learn()")
    print("2. Adicionar debug prints no momento da captura de métricas")
    print("3. Verificar se o model carregado permite updates (.train() vs .eval())")
    print("4. Verificar se rewards/observations têm variabilidade")
    print("5. Testar com model novo (sem checkpoint) para comparar")

if __name__ == "__main__":
    investigate_log_integrity()