"""
🔍 Script de Validação: Alinhamento Action Space Cherry.py ↔ Robot_cherry.py
"""

def validate_action_thresholds():
    """Valida se as constantes ACTION_THRESHOLD estão alinhadas"""

    print("="*80)
    print("🔍 VALIDAÇÃO: Alinhamento Action Space cherry.py ↔ Robot_cherry.py")
    print("="*80)

    # Ler cherry.py
    with open("D:\\Projeto\\cherry.py", "r", encoding="utf-8") as f:
        cherry_content = f.read()

    # Ler Robot_cherry.py
    with open("D:\\Projeto\\Modelo PPO Trader\\Robot_cherry.py", "r", encoding="utf-8") as f:
        robot_content = f.read()

    # Verificar constantes
    issues = []

    # 1. Verificar ACTION_THRESHOLD_LONG
    if "ACTION_THRESHOLD_LONG = -0.33" not in cherry_content:
        issues.append("❌ cherry.py: ACTION_THRESHOLD_LONG não encontrado ou valor incorreto")
    else:
        print("✅ cherry.py: ACTION_THRESHOLD_LONG = -0.33")

    if "ACTION_THRESHOLD_LONG = -0.33" not in robot_content:
        issues.append("❌ Robot_cherry.py: ACTION_THRESHOLD_LONG não encontrado ou valor incorreto")
    else:
        print("✅ Robot_cherry.py: ACTION_THRESHOLD_LONG = -0.33")

    # 2. Verificar ACTION_THRESHOLD_SHORT
    if "ACTION_THRESHOLD_SHORT = 0.33" not in cherry_content:
        issues.append("❌ cherry.py: ACTION_THRESHOLD_SHORT não encontrado ou valor incorreto")
    else:
        print("✅ cherry.py: ACTION_THRESHOLD_SHORT = 0.33")

    if "ACTION_THRESHOLD_SHORT = 0.33" not in robot_content:
        issues.append("❌ Robot_cherry.py: ACTION_THRESHOLD_SHORT não encontrado ou valor incorreto")
    else:
        print("✅ Robot_cherry.py: ACTION_THRESHOLD_SHORT = 0.33")

    # 3. Verificar uso consistente das constantes
    print("\n" + "="*80)
    print("🔍 VERIFICANDO USOS DAS CONSTANTES:")
    print("="*80)

    # cherry.py - contar usos
    cherry_uses = cherry_content.count("ACTION_THRESHOLD_LONG") + cherry_content.count("ACTION_THRESHOLD_SHORT")
    print(f"✅ cherry.py: {cherry_uses} usos das constantes ACTION_THRESHOLD")

    # Robot_cherry.py - contar usos
    robot_uses = robot_content.count("ACTION_THRESHOLD_LONG") + robot_content.count("ACTION_THRESHOLD_SHORT")
    print(f"✅ Robot_cherry.py: {robot_uses} usos das constantes ACTION_THRESHOLD")

    # 4. Verificar se ainda existem hardcoded thresholds
    print("\n" + "="*80)
    print("🔍 VERIFICANDO HARDCODED THRESHOLDS (NÃO DEVERIA EXISTIR):")
    print("="*80)

    # Padrões a procurar (excluindo comentários e definições)
    hardcoded_patterns = [
        ("< -0.33", "comparação hardcoded"),
        ("< 0.33", "comparação hardcoded"),
        (">= 0.33", "comparação hardcoded"),
        ("< 1.5", "threshold incorreto 1.5"),
        (">= 0.67", "threshold incorreto 0.67"),
    ]

    for pattern, desc in hardcoded_patterns:
        # Contar em cherry.py (excluir definições de constantes)
        cherry_lines = [line for line in cherry_content.split('\n')
                       if pattern in line and 'ACTION_THRESHOLD' not in line and not line.strip().startswith('#')]

        robot_lines = [line for line in robot_content.split('\n')
                      if pattern in line and 'ACTION_THRESHOLD' not in line and not line.strip().startswith('#')]

        if cherry_lines:
            print(f"⚠️  cherry.py: Encontrado '{pattern}' ({desc}) em {len(cherry_lines)} linha(s)")
            for line in cherry_lines[:3]:  # Mostrar apenas primeiras 3
                print(f"    → {line.strip()[:80]}")

        if robot_lines:
            print(f"⚠️  Robot_cherry.py: Encontrado '{pattern}' ({desc}) em {len(robot_lines)} linha(s)")
            for line in robot_lines[:3]:  # Mostrar apenas primeiras 3
                print(f"    → {line.strip()[:80]}")

    # 5. Verificar action space definition
    print("\n" + "="*80)
    print("🔍 VERIFICANDO ACTION SPACE DEFINITION:")
    print("="*80)

    # cherry.py action space
    if "self.action_space = spaces.Box(" in cherry_content:
        if "low=np.array([-1, 0, -1, -1])" in cherry_content:
            print("✅ cherry.py: Action space correto - Box([-1, 0, -1, -1], [1, 1, 1, 1])")
        else:
            issues.append("❌ cherry.py: Action space não está usando range [-1, 0, -1, -1]")

    # Robot_cherry.py - verificar comentários sobre action space
    if "ACTION_SPACE_SIZE = 4" in robot_content:
        print("✅ Robot_cherry.py: ACTION_SPACE_SIZE = 4")
    else:
        issues.append("❌ Robot_cherry.py: ACTION_SPACE_SIZE não está definido como 4")

    # RESUMO FINAL
    print("\n" + "="*80)
    print("📊 RESUMO DA VALIDAÇÃO:")
    print("="*80)

    if not issues:
        print("✅ SUCESSO! Todos os checks passaram!")
        print("✅ cherry.py e Robot_cherry.py estão COMPLETAMENTE ALINHADOS!")
        print("\n📋 Mapeamento de Ações (Action Space 4D):")
        print("   [0] entry_decision: [-1, 1]")
        print("       ├─ HOLD:  [-1.00, -0.33)")
        print("       ├─ LONG:  [-0.33,  0.33)")
        print("       └─ SHORT: [ 0.33,  1.00]")
        print("   [1] entry_confidence: [0, 1]")
        print("   [2] pos1_mgmt: [-1, 1]")
        print("   [3] pos2_mgmt: [-1, 1]")
        return True
    else:
        print(f"❌ FALHA! {len(issues)} problema(s) encontrado(s):")
        for issue in issues:
            print(f"   {issue}")
        return False

if __name__ == "__main__":
    success = validate_action_thresholds()
    exit(0 if success else 1)
