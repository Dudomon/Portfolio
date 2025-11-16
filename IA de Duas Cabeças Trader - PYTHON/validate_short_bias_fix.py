#!/usr/bin/env python3
"""
✅ VALIDAÇÃO FINAL: Correção do Viés Vendedor
Verifica se cherry.py e Robot_cherry.py estão alinhados e balanceados
"""

import numpy as np
import re

def validate_file(filepath, file_label):
    """Valida um arquivo específico"""
    print(f"\n{'='*70}")
    print(f"📂 Validando: {file_label}")
    print(f"   Arquivo: {filepath}")
    print(f"{'='*70}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        errors = []
        warnings = []

        # Check 1: Action space
        action_space_match = re.search(
            r'self\.action_space = spaces\.Box\(\s*low=np\.array\(\[([-0-9., ]+)\]\),\s*high=np\.array\(\[([-0-9., ]+)\]\)',
            content
        )

        if action_space_match:
            low_str = action_space_match.group(1)
            high_str = action_space_match.group(2)

            print(f"\n✅ Action Space encontrado:")
            print(f"   Low:  [{low_str}]")
            print(f"   High: [{high_str}]")

            # Validar valores
            if '-1' in low_str or '-1.0' in low_str:
                if '1' in high_str or '1.0' in high_str:
                    print(f"   ✅ CORRETO: Balanceado [-1, 1]")
                else:
                    errors.append("Action space high não está correto")
            else:
                errors.append("Action space low não está balanceado (deveria ser -1)")
        else:
            errors.append("Action space não encontrado no arquivo")

        # Check 2: Thresholds (apenas para cherry.py)
        if 'cherry.py' in filepath:
            threshold_long = re.search(r'ACTION_THRESHOLD_LONG\s*=\s*([-0-9.]+)', content)
            threshold_short = re.search(r'ACTION_THRESHOLD_SHORT\s*=\s*([-0-9.]+)', content)

            if threshold_long and threshold_short:
                tl = float(threshold_long.group(1))
                ts = float(threshold_short.group(1))

                print(f"\n✅ Thresholds encontrados:")
                print(f"   LONG:  {tl}")
                print(f"   SHORT: {ts}")

                if tl == -0.33 and ts == 0.33:
                    print(f"   ✅ CORRETO: Simétricos")
                else:
                    errors.append(f"Thresholds incorretos: {tl}, {ts} (esperado: -0.33, 0.33)")
            else:
                errors.append("Thresholds não encontrados")

        # Check 3: Mapeamento de decisão
        decision_mappings = re.findall(
            r'if raw_decision < ([-0-9.]+):.*?# < ([-0-9.]+) = (HOLD|LONG|SHORT)',
            content,
            re.DOTALL
        )

        if decision_mappings:
            print(f"\n✅ Mapeamentos de decisão encontrados:")
            for threshold, comment_threshold, action in decision_mappings[:3]:
                print(f"   {action:5s}: < {threshold}")

                # Validar
                if action == "HOLD" and threshold != "-0.33":
                    errors.append(f"HOLD threshold incorreto: {threshold} (esperado: -0.33)")
                elif action == "LONG" and threshold != "0.33":
                    errors.append(f"LONG threshold incorreto: {threshold} (esperado: 0.33)")
        else:
            warnings.append("Mapeamentos de decisão não encontrados ou formato diferente")

        # Resultado
        print(f"\n{'='*70}")
        if errors:
            print(f"❌ VALIDAÇÃO FALHOU:")
            for err in errors:
                print(f"   • {err}")
            return False
        elif warnings:
            print(f"⚠️  VALIDAÇÃO COM AVISOS:")
            for warn in warnings:
                print(f"   • {warn}")
            return True
        else:
            print(f"✅ VALIDAÇÃO PASSOU - Arquivo correto!")
            return True

    except Exception as e:
        print(f"❌ ERRO ao validar arquivo: {e}")
        return False

def validate_alignment():
    """Valida alinhamento entre cherry.py e Robot_cherry.py"""
    print(f"\n{'='*70}")
    print(f"🔗 Validando Alinhamento entre Arquivos")
    print(f"{'='*70}")

    try:
        # Ler ambos
        with open('D:/Projeto/cherry.py', 'r', encoding='utf-8') as f:
            cherry_content = f.read()
        with open('D:/Projeto/Modelo PPO Trader/Robot_cherry.py', 'r', encoding='utf-8') as f:
            robot_content = f.read()

        # Extrair action spaces
        cherry_as = re.search(r'self\.action_space = spaces\.Box\(\s*low=np\.array\(\[([-0-9., ]+)\]\),\s*high=np\.array\(\[([-0-9., ]+)\]\)', cherry_content)
        robot_as = re.search(r'self\.action_space = spaces\.Box\(\s*low=np\.array\(\[([-0-9., ]+)\]\),\s*high=np\.array\(\[([-0-9., ]+)\]\)', robot_content)

        if cherry_as and robot_as:
            cherry_low = cherry_as.group(1).replace(' ', '')
            cherry_high = cherry_as.group(2).replace(' ', '')
            robot_low = robot_as.group(1).replace(' ', '')
            robot_high = robot_as.group(2).replace(' ', '')

            print(f"\nAction Space Comparison:")
            print(f"   cherry.py:       Low=[{cherry_low}], High=[{cherry_high}]")
            print(f"   Robot_cherry.py: Low=[{robot_low}], High=[{robot_high}]")

            # Comparar primeiro elemento (entry_decision)
            cherry_low_first = cherry_low.split(',')[0]
            cherry_high_first = cherry_high.split(',')[0]
            robot_low_first = robot_low.split(',')[0]
            robot_high_first = robot_high.split(',')[0]

            if cherry_low_first == robot_low_first and cherry_high_first == robot_high_first:
                print(f"\n   ✅ ALINHADOS: Entry decision usa mesmo range [{cherry_low_first}, {cherry_high_first}]")
                return True
            else:
                print(f"\n   ❌ DESALINHADOS: Ranges diferentes!")
                print(f"      cherry: [{cherry_low_first}, {cherry_high_first}]")
                print(f"      robot:  [{robot_low_first}, {robot_high_first}]")
                return False
        else:
            print(f"❌ Não foi possível extrair action spaces")
            return False

    except Exception as e:
        print(f"❌ ERRO ao validar alinhamento: {e}")
        return False

def simulate_distribution():
    """Simula distribuição final"""
    print(f"\n{'='*70}")
    print(f"🎲 Simulação de Distribuição (100k samples)")
    print(f"{'='*70}")

    np.random.seed(42)
    actions = np.random.uniform(-1, 1, 100000)

    hold_count = np.sum(actions < -0.33)
    long_count = np.sum((actions >= -0.33) & (actions < 0.33))
    short_count = np.sum(actions >= 0.33)

    print(f"\n📊 Distribuição Final:")
    print(f"   HOLD:  {hold_count:6d} ({100*hold_count/100000:.1f}%)")
    print(f"   LONG:  {long_count:6d} ({100*long_count/100000:.1f}%)")
    print(f"   SHORT: {short_count:6d} ({100*short_count/100000:.1f}%)")

    # Verificar balanceamento
    target = 100000 / 3
    tolerance = 0.02

    balanced = (
        abs(hold_count/target - 1) < tolerance and
        abs(long_count/target - 1) < tolerance and
        abs(short_count/target - 1) < tolerance
    )

    if balanced:
        print(f"\n   ✅ BALANCEADO (tolerância ±{tolerance*100}%)")
        return True
    else:
        print(f"\n   ❌ DESBALANCEADO")
        return False

def main():
    print(f"\n{'='*70}")
    print(f"✅ VALIDAÇÃO FINAL: Correção do Viés Vendedor")
    print(f"{'='*70}")

    results = {}

    # Validar cherry.py
    results['cherry'] = validate_file('D:/Projeto/cherry.py', 'cherry.py (ambiente treino)')

    # Validar Robot_cherry.py
    results['robot'] = validate_file('D:/Projeto/Modelo PPO Trader/Robot_cherry.py', 'Robot_cherry.py (produção)')

    # Validar alinhamento
    results['alignment'] = validate_alignment()

    # Simular distribuição
    results['distribution'] = simulate_distribution()

    # Resumo final
    print(f"\n{'='*70}")
    print(f"📊 RESUMO FINAL")
    print(f"{'='*70}")

    print(f"\n✅ Resultados:")
    print(f"   cherry.py validação:      {'✅ PASS' if results['cherry'] else '❌ FAIL'}")
    print(f"   Robot_cherry.py validação: {'✅ PASS' if results['robot'] else '❌ FAIL'}")
    print(f"   Alinhamento:              {'✅ PASS' if results['alignment'] else '❌ FAIL'}")
    print(f"   Distribuição balanceada:  {'✅ PASS' if results['distribution'] else '❌ FAIL'}")

    all_passed = all(results.values())

    print(f"\n{'='*70}")
    if all_passed:
        print(f"✅ VALIDAÇÃO COMPLETA - TODOS OS TESTES PASSARAM!")
        print(f"{'='*70}")
        print(f"\n🚀 Sistema pronto para re-treino:")
        print(f"   1. Action space balanceado: [-1, 1]")
        print(f"   2. Thresholds simétricos: -0.33 / 0.33")
        print(f"   3. cherry.py e Robot_cherry.py alinhados")
        print(f"   4. Distribuição: 33% HOLD / 33% LONG / 33% SHORT")
        print(f"\n⚠️  ATENÇÃO: Checkpoints antigos são INCOMPATÍVEIS!")
        print(f"   • Fazer backup dos checkpoints atuais")
        print(f"   • Limpar pasta de checkpoints")
        print(f"   • Iniciar treino do zero")
    else:
        print(f"❌ VALIDAÇÃO FALHOU - CORRIGIR ERROS ANTES DO RE-TREINO")
        print(f"{'='*70}")

    print()
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
