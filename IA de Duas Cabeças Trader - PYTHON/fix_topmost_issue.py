#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 CORREÇÃO CRÍTICA: Problema de -topmost causando travamento
O problema está no código que força a janela a ficar sempre no topo
"""

import os
import shutil
from datetime import datetime

def fix_topmost_issue():
    """Corrigir o problema específico do -topmost que causa travamento"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ Arquivo não encontrado: {robot_path}")
        return False
    
    # Fazer backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"Modelo PPO Trader/RobotV7_backup_topmost_{timestamp}.py"
    shutil.copy2(robot_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Ler arquivo
    with open(robot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes_applied = []
    
    # 1. Remover código problemático de -topmost
    problematic_code = """            # Pulsar topmost para trazer à frente
            self.root.lift()
            self.root.attributes('-topmost', True)
            # Pequeno atraso para garantir o raise, depois remove topmost
            self.root.after(250, lambda: self.root.attributes('-topmost', False))"""
    
    if problematic_code in content:
        safe_code = """            # Trazer à frente de forma segura (sem topmost)
            self.root.lift()
            self.root.focus_set()"""
        
        content = content.replace(problematic_code, safe_code)
        fixes_applied.append("Removido código -topmost problemático")
    
    # 2. Corrigir outras ocorrências de -topmost
    topmost_patterns = [
        ("self.root.attributes('-topmost', True)", "# self.root.attributes('-topmost', True)  # REMOVIDO - causa travamento"),
        ("self.root.attributes('-topmost', False)", "# self.root.attributes('-topmost', False)  # REMOVIDO - causa travamento"),
        ("attributes('-topmost'", "# attributes('-topmost'  # REMOVIDO - causa travamento")
    ]
    
    for old, new in topmost_patterns:
        if old in content:
            content = content.replace(old, new)
            fixes_applied.append(f"Removido: {old}")
    
    # 3. Simplificar _force_show_window
    force_show_pattern = """    def _force_show_window(self, initial=False):
        \"\"\"Garante que a janela esteja visível, não minimizada e dentro da tela.\"\"\"
        try:
            # Trazer de volta se minimizada ou escondida
            state = self.root.state()
            if state in ("iconic", "withdrawn"):
                self.root.deiconify()
                self.root.state('normal')
            # Trazer à frente de forma segura (sem topmost)
            self.root.lift()
            self.root.focus_set()
            # Garantir que está focada
            try:
                self.root.focus_force()"""
    
    if "_force_show_window" in content and "topmost" in content:
        # Substituir por versão simplificada
        simple_force_show = """    def _force_show_window(self, initial=False):
        \"\"\"Versão simplificada - garante visibilidade sem topmost\"\"\"
        try:
            # Apenas deiconificar se minimizada
            if self.root.state() == "iconic":
                self.root.deiconify()
            # Trazer à frente de forma simples
            self.root.lift()"""
        
        # Encontrar o método completo e substituir
        import re
        pattern = r"def _force_show_window\(self, initial=False\):.*?(?=\n    def|\n\nclass|\nclass|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content.replace(match.group(0), simple_force_show + "\n        except Exception:\n            pass")
            fixes_applied.append("Simplificado método _force_show_window")
    
    # 4. Remover heartbeat de visibilidade que pode estar causando problemas
    if "_smart_visibility_heartbeat" in content:
        content = content.replace(
            "self.safe_after(2000, self._smart_visibility_heartbeat)",
            "# self.safe_after(2000, self._smart_visibility_heartbeat)  # REMOVIDO - pode causar travamento"
        )
        fixes_applied.append("Removido heartbeat de visibilidade")
    
    # 5. Simplificar inicialização da janela
    if "_bind_visibility_events" in content:
        content = content.replace(
            "self._bind_visibility_events()",
            "# self._bind_visibility_events()  # REMOVIDO - simplificação"
        )
        fixes_applied.append("Removido bind de eventos de visibilidade")
    
    # Aplicar correções
    if fixes_applied:
        with open(robot_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Correções críticas aplicadas:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        return True
    else:
        print("ℹ️ Nenhuma correção necessária")
        return False

def create_simple_test():
    """Criar teste simples para verificar se a janela abre"""
    test_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE SIMPLES: Verificar se RobotV7 abre sem travar
"""

import subprocess
import sys
import time
import os

def simple_test():
    """Teste simples e direto"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 TESTE SIMPLES - RobotV7")
    print("=" * 30)
    print("⏳ Iniciando RobotV7...")
    print("⏳ Aguarde 5 segundos...")
    
    try:
        # Lançar RobotV7 em processo separado
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar 5 segundos
        time.sleep(5)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 iniciou com sucesso!")
            print("✅ Verifique se a janela está visível")
            print("\\n💡 Se a janela não aparecer:")
            print("   • Clique no ícone na barra de tarefas")
            print("   • Use Alt+Tab para alternar janelas")
            print("   • Verifique se não está atrás de outras janelas")
            
            input("\\nPressione Enter quando terminar o teste...")
            
            # Finalizar processo
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                print("✅ Processo finalizado")
            except:
                print("⚠️ Processo pode ainda estar rodando")
            
            return True
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    simple_test()
'''
    
    with open("simple_test_robotv7.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Teste simples criado: simple_test_robotv7.py")

def main():
    """Função principal"""
    print("🔧 CORREÇÃO CRÍTICA: Problema -topmost")
    print("=" * 50)
    print("🎯 PROBLEMA IDENTIFICADO:")
    print("   • Código -topmost está causando travamento da janela")
    print("   • Heartbeat de visibilidade pode estar interferindo")
    print("   • Eventos de visibilidade complexos demais")
    print()
    
    # Aplicar correção
    print("Aplicando correção crítica...")
    fixed = fix_topmost_issue()
    
    # Criar teste simples
    print("\\nCriando teste simples...")
    create_simple_test()
    
    print("\\n" + "=" * 50)
    print("🎯 CORREÇÃO CRÍTICA CONCLUÍDA!")
    
    if fixed:
        print("✅ Código problemático removido/corrigido")
        print("✅ Backup do arquivo original criado")
    
    print("✅ Teste simples criado")
    
    print("\\n📋 TESTE AGORA:")
    print("1. Execute: python simple_test_robotv7.py")
    print("2. Verifique se a janela aparece em 5 segundos")
    print("3. Se ainda travar, reinicie o computador")
    
    print("\\n🎯 A CORREÇÃO DEVE RESOLVER:")
    print("   ✅ Janela não mais travada na barra de tarefas")
    print("   ✅ Janela aparece normalmente")
    print("   ✅ Sem conflitos de -topmost")

if __name__ == "__main__":
    main()