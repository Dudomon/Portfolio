"""
🔧 FIX ACTION SPACE SHORT BIAS

Corrige o mapeamento do action space para alinhar com dinâmica de redes neurais:
- LONG no extremo POSITIVO (>= 0.33)
- SHORT no extremo NEGATIVO (< -0.33)
- HOLD no CENTRO ([-0.33, 0.33])

Isso garante que decisões de alta confidence (extremos) sejam tanto LONG quanto SHORT.
"""

import re

def fix_action_space_mapping(file_path):
    """Aplica fix no action space mapping"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Atualizar comentários e constantes no topo
    old_header = r"""# 🎯 FIX SHORT BIAS: THRESHOLDS BALANCEADOS PARA DISTRIBUIÇÃO EQUILIBRADA
# Garante consistência na interpretação de ações em todo o código
# Com sigmoid \[0,1\]: HOLD\[0,0\.33\] LONG\[0\.33,0\.67\] SHORT\[0\.67,1\.0\] = ~33% cada
ACTION_THRESHOLD_LONG = -0\.33   # raw_decision < -0\.33 = HOLD \(33% do range\)
ACTION_THRESHOLD_SHORT = 0\.33   # raw_decision < 0\.33 = LONG, >= 0\.33 = SHORT \(33%\)"""

    new_header = """# 🎯 FIX SHORT BIAS V2: EXTREMOS=AÇÕES, CENTRO=HOLD
# Design alinhado com dinâmica de redes neurais gaussianas:
# - Alta confidence → extremos → LONG (positivo) ou SHORT (negativo)
# - Baixa confidence → centro → HOLD (incerteza)
# Distribuição: SHORT[-1,-0.33]=33.5% | HOLD[-0.33,0.33]=33% | LONG[0.33,1]=33.5%
ACTION_THRESHOLD_SHORT = -0.33  # raw_decision < -0.33 = SHORT (extremo negativo)
ACTION_THRESHOLD_LONG = 0.33    # raw_decision >= 0.33 = LONG (extremo positivo)"""

    content = re.sub(old_header, new_header, content)

    # 2. Atualizar todos os blocos de interpretação do action space
    # Padrão antigo:
    # if raw_decision < ACTION_THRESHOLD_LONG:
    #     entry_decision = 0  # HOLD
    # elif raw_decision < ACTION_THRESHOLD_SHORT:
    #     entry_decision = 1  # LONG
    # else:
    #     entry_decision = 2  # SHORT

    old_pattern = r"""if raw_decision < ACTION_THRESHOLD_LONG:
(\s+)entry_decision = 0  # HOLD
(\s+)elif raw_decision < ACTION_THRESHOLD_SHORT:
(\s+)entry_decision = 1  # LONG
(\s+)else:
(\s+)entry_decision = 2  # SHORT"""

    new_pattern = r"""if raw_decision < ACTION_THRESHOLD_SHORT:
\1entry_decision = 2  # SHORT (extremo negativo)
\2elif raw_decision < ACTION_THRESHOLD_LONG:
\3entry_decision = 0  # HOLD (centro)
\4else:
\5entry_decision = 1  # LONG (extremo positivo)"""

    content = re.sub(old_pattern, new_pattern, content)

    # 3. Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Action space mapping corrigido em {file_path}")
    return True

if __name__ == "__main__":
    import sys

    # Aplicar fix no cherry.py
    file_path = "D:/Projeto/cherry.py"

    print("🔧 Aplicando FIX SHORT BIAS V2...")
    print("=" * 80)

    success = fix_action_space_mapping(file_path)

    if success:
        print("\n✅ CORREÇÃO APLICADA COM SUCESSO!")
        print("\n📋 Mudanças aplicadas:")
        print("   1. Constantes: ACTION_THRESHOLD_SHORT = -0.33, ACTION_THRESHOLD_LONG = 0.33")
        print("   2. Mapeamento:")
        print("      - raw_decision < -0.33  → SHORT (extremo negativo)")
        print("      - -0.33 <= raw_decision < 0.33 → HOLD (centro)")
        print("      - raw_decision >= 0.33  → LONG (extremo positivo)")
        print("\n🎯 RESULTADO:")
        print("   - Alta confidence → extremos → LONG ou SHORT")
        print("   - Baixa confidence → centro → HOLD")
        print("   - Viés SHORT ELIMINADO!")
        sys.exit(0)
    else:
        print("\n❌ ERRO ao aplicar correção")
        sys.exit(1)
