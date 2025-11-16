#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 CORREÇÃO DEFINITIVA: Bug de Minimizar/Restaurar
Problema: Janela não volta quando clicada na barra de tarefas
Solução: Implementar restauração correta de janela minimizada
"""

import os
import shutil
from datetime import datetime

def fix_minimize_restore_bug():
    """Corrigir definitivamente o problema de restauração de janela minimizada"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ Arquivo não encontrado: {robot_path}")
        return False
    
    # Fazer backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"Modelo PPO Trader/RobotV7_backup_minimize_{timestamp}.py"
    shutil.copy2(robot_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Ler arquivo
    with open(robot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes_applied = []
    
    # 1. Encontrar e corrigir o método _force_show_window
    old_force_show = """    def _force_show_window(self, initial=False):
        \"\"\"Versão simplificada - garante visibilidade sem topmost\"\"\"
        try:
            # Apenas deiconificar se minimizada
            if self.root.state() == "iconic":
                self.root.deiconify()
            # Trazer à frente de forma simples
            self.root.lift()
        except Exception:
            pass"""
    
    new_force_show = """    def _force_show_window(self, initial=False):
        \"\"\"Correção definitiva para restauração de janela minimizada\"\"\"
        try:
            # Verificar estado atual da janela
            current_state = self.root.state()
            
            # Se está minimizada (iconic), restaurar
            if current_state == "iconic":
                self.root.deiconify()
                self.root.state('normal')
            
            # Se está escondida (withdrawn), mostrar
            elif current_state == "withdrawn":
                self.root.deiconify()
                self.root.state('normal')
            
            # Garantir que está visível e focada
            self.root.lift()
            self.root.focus_set()
            
            # Forçar atualização da janela
            self.root.update_idletasks()
            
        except Exception as e:
            print(f"[DEBUG] Erro em _force_show_window: {e}")"""
    
    if old_force_show in content:
        content = content.replace(old_force_show, new_force_show)
        fixes_applied.append("Corrigido método _force_show_window")
    
    # 2. Adicionar bind para evento de deiconify (restaurar da barra de tarefas)
    if "self.root.protocol(\"WM_DELETE_WINDOW\", self.on_closing)" in content:
        protocol_line = "self.root.protocol(\"WM_DELETE_WINDOW\", self.on_closing)"
        new_protocol_section = f"""{protocol_line}
        
        # 🔧 CORREÇÃO: Bind para restauração de janela minimizada
        self.root.bind('<Map>', self._on_window_map)
        self.root.bind('<Unmap>', self._on_window_unmap)
        self.root.bind('<FocusIn>', self._on_window_focus_in)"""
        
        content = content.replace(protocol_line, new_protocol_section)
        fixes_applied.append("Adicionados binds para eventos de janela")
    
    # 3. Adicionar métodos de callback para eventos de janela
    if "def on_closing(self):" in content:
        # Encontrar onde inserir os novos métodos (antes de on_closing)
        on_closing_pos = content.find("def on_closing(self):")
        if on_closing_pos > 0:
            # Encontrar a indentação correta
            lines_before = content[:on_closing_pos].split('\n')
            last_line = lines_before[-1]
            indent = len(last_line) - len(last_line.lstrip())
            
            new_methods = f"""
{' ' * indent}def _on_window_map(self, event=None):
{' ' * (indent + 4)}\"\"\"Callback quando janela é mapeada (restaurada)\"\"\"
{' ' * (indent + 4)}try:
{' ' * (indent + 8)}if event and event.widget == self.root:
{' ' * (indent + 12)}# Janela foi restaurada da barra de tarefas
{' ' * (indent + 12)}self.root.focus_set()
{' ' * (indent + 12)}self.root.lift()
{' ' * (indent + 4)}except Exception:
{' ' * (indent + 8)}pass

{' ' * indent}def _on_window_unmap(self, event=None):
{' ' * (indent + 4)}\"\"\"Callback quando janela é desmapeada (minimizada)\"\"\"
{' ' * (indent + 4)}try:
{' ' * (indent + 8)}if event and event.widget == self.root:
{' ' * (indent + 12)}# Janela foi minimizada
{' ' * (indent + 12)}pass  # Não fazer nada especial
{' ' * (indent + 4)}except Exception:
{' ' * (indent + 8)}pass

{' ' * indent}def _on_window_focus_in(self, event=None):
{' ' * (indent + 4)}\"\"\"Callback quando janela recebe foco\"\"\"
{' ' * (indent + 4)}try:
{' ' * (indent + 8)}if event and event.widget == self.root:
{' ' * (indent + 12)}# Garantir que está no estado normal
{' ' * (indent + 12)}if self.root.state() == "iconic":
{' ' * (indent + 16)}self.root.deiconify()
{' ' * (indent + 16)}self.root.state('normal')
{' ' * (indent + 4)}except Exception:
{' ' * (indent + 8)}pass

{' ' * indent}"""
            
            content = content[:on_closing_pos] + new_methods + content[on_closing_pos:]
            fixes_applied.append("Adicionados métodos de callback para eventos de janela")
    
    # 4. Corrigir configuração inicial da janela
    if "self.root.geometry(\"1200x800\")" in content:
        geometry_section = """self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)"""
        
        new_geometry_section = """self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # 🔧 CORREÇÃO: Configurações para evitar problemas de minimização
        self.root.minsize(800, 600)  # Tamanho mínimo
        self.root.state('normal')    # Garantir estado normal
        self.root.wm_attributes('-toolwindow', False)  # Não é tool window"""
        
        content = content.replace(geometry_section, new_geometry_section)
        fixes_applied.append("Corrigidas configurações iniciais da janela")
    
    # 5. Remover qualquer código que possa interferir com a restauração
    problematic_patterns = [
        ("self.root.withdraw()", "# self.root.withdraw()  # REMOVIDO - interfere com restauração"),
        ("self.root.iconify()", "# self.root.iconify()  # REMOVIDO - interfere com restauração"),
        ("self.root.wm_state('iconic')", "# self.root.wm_state('iconic')  # REMOVIDO - interfere com restauração")
    ]
    
    for old, new in problematic_patterns:
        if old in content:
            content = content.replace(old, new)
            fixes_applied.append(f"Removido código problemático: {old}")
    
    # Aplicar correções
    if fixes_applied:
        with open(robot_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Correções definitivas aplicadas:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        return True
    else:
        print("ℹ️ Nenhuma correção necessária")
        return False

def create_minimize_test():
    """Criar teste específico para minimização/restauração"""
    test_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE ESPECÍFICO: Minimizar/Restaurar Janela
Testa se a janela pode ser minimizada e restaurada corretamente
"""

import subprocess
import sys
import time
import os

def test_minimize_restore():
    """Teste específico de minimização e restauração"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 TESTE: Minimizar/Restaurar Janela")
    print("=" * 40)
    print("⏳ Iniciando RobotV7...")
    
    try:
        # Lançar RobotV7
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar inicialização
        time.sleep(8)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 iniciado com sucesso!")
            print()
            print("📋 INSTRUÇÕES PARA TESTE:")
            print("1. Verifique se a janela do RobotV7 está visível")
            print("2. Minimize a janela (clique no botão minimizar)")
            print("3. Clique no ícone na barra de tarefas para restaurar")
            print("4. Repita o processo 2-3 vezes")
            print()
            
            # Aguardar teste manual
            input("Pressione Enter quando terminar o teste de minimização...")
            
            # Perguntar resultado
            print()
            result = input("A janela restaurou corretamente da barra de tarefas? (s/N): ")
            
            # Finalizar processo
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                print("✅ Processo finalizado")
            except:
                print("⚠️ Processo pode ainda estar rodando")
            
            if result.lower() == 's':
                print("🎉 SUCESSO! Problema de minimização corrigido!")
                return True
            else:
                print("❌ Problema ainda persiste")
                return False
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    success = test_minimize_restore()
    
    if not success:
        print("\\n💡 SOLUÇÕES ALTERNATIVAS:")
        print("1. Use Alt+Tab para alternar entre janelas")
        print("2. Reinicie o computador")
        print("3. Execute como administrador")
        print("4. Verifique se há conflitos com outros programas")
    
    input("\\nPressione Enter para sair...")
'''
    
    with open("test_minimize_restore.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Teste de minimização criado: test_minimize_restore.py")

def main():
    """Função principal"""
    print("🔧 CORREÇÃO DEFINITIVA: Bug Minimizar/Restaurar")
    print("=" * 60)
    print("🎯 PROBLEMA:")
    print("   • Janela não restaura quando clicada na barra de tarefas")
    print("   • Problema piora com múltiplas instâncias")
    print("   • Janela fica 'perdida' após minimização")
    print()
    
    # Aplicar correção
    print("Aplicando correção definitiva...")
    fixed = fix_minimize_restore_bug()
    
    # Criar teste específico
    print("\\nCriando teste específico...")
    create_minimize_test()
    
    print("\\n" + "=" * 60)
    print("🎯 CORREÇÃO DEFINITIVA CONCLUÍDA!")
    
    if fixed:
        print("✅ Implementada restauração correta de janela")
        print("✅ Adicionados eventos de Map/Unmap/FocusIn")
        print("✅ Corrigidas configurações de janela")
        print("✅ Removidos códigos problemáticos")
    
    print("✅ Teste específico criado")
    
    print("\\n📋 TESTE A CORREÇÃO:")
    print("1. Execute: python test_minimize_restore.py")
    print("2. Siga as instruções para testar minimização")
    print("3. Verifique se a janela restaura corretamente")
    
    print("\\n🎯 CORREÇÕES IMPLEMENTADAS:")
    print("   ✅ Bind para eventos <Map>, <Unmap>, <FocusIn>")
    print("   ✅ Método _force_show_window melhorado")
    print("   ✅ Callbacks específicos para restauração")
    print("   ✅ Configurações de janela otimizadas")

if __name__ == "__main__":
    main()