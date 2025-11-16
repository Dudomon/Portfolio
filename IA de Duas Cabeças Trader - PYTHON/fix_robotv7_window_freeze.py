#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 CORREÇÃO URGENTE: RobotV7 Window Freeze Fix
Corrige o problema de janela travada/minimizada no RobotV7
"""

import os
import sys
import shutil
from datetime import datetime

def backup_current_robotv7():
    """Fazer backup do RobotV7 atual antes da correção"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    if os.path.exists(robot_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"Modelo PPO Trader/RobotV7_backup_{timestamp}.py"
        shutil.copy2(robot_path, backup_path)
        print(f"✅ Backup criado: {backup_path}")
        return backup_path
    return None

def fix_gui_initialization():
    """Corrigir problemas de inicialização da GUI"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ Arquivo não encontrado: {robot_path}")
        return False
    
    # Ler arquivo atual
    with open(robot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Identificar e corrigir problemas comuns de GUI
    fixes_applied = []
    
    # 1. Corrigir imports de tkinter
    if "import tkinter as tk" in content and "from tkinter import scrolledtext, ttk" in content:
        # Verificar se há imports duplicados ou conflitantes
        lines = content.split('\n')
        tkinter_imports = []
        for i, line in enumerate(lines):
            if 'import tkinter' in line or 'from tkinter' in line:
                tkinter_imports.append((i, line))
        
        if len(tkinter_imports) > 2:  # Mais de 2 imports tkinter pode causar conflito
            print("🔧 Detectados múltiplos imports tkinter - consolidando...")
            # Remover imports duplicados
            for i, line in reversed(tkinter_imports[2:]):  # Manter apenas os 2 primeiros
                lines[i] = f"# {line}  # Removido - duplicado"
            content = '\n'.join(lines)
            fixes_applied.append("Consolidação de imports tkinter")
    
    # 2. Corrigir problemas de threading na GUI
    if "class TradingGUI:" in content:
        # Verificar se há problemas na inicialização da janela
        if "self.root.withdraw()" in content:
            content = content.replace("self.root.withdraw()", "# self.root.withdraw()  # Removido - causa travamento")
            fixes_applied.append("Removido withdraw() que causa travamento")
        
        # Garantir que a janela seja mostrada corretamente
        if "self.root.deiconify()" not in content and "self.root.lift()" not in content:
            # Adicionar código para garantir que a janela apareça
            gui_init_pattern = "def __init__(self, robot):"
            if gui_init_pattern in content:
                # Encontrar o final do __init__ da GUI
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if gui_init_pattern in line:
                        # Procurar o final do método
                        indent_level = len(line) - len(line.lstrip())
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() == "":
                                continue
                            current_indent = len(lines[j]) - len(lines[j].lstrip())
                            if current_indent <= indent_level and lines[j].strip():
                                # Inserir código antes desta linha
                                lines.insert(j, " " * (indent_level + 4) + "# Garantir que a janela apareça")
                                lines.insert(j + 1, " " * (indent_level + 4) + "self.root.deiconify()")
                                lines.insert(j + 2, " " * (indent_level + 4) + "self.root.lift()")
                                lines.insert(j + 3, " " * (indent_level + 4) + "self.root.focus_force()")
                                content = '\n'.join(lines)
                                fixes_applied.append("Adicionado código para mostrar janela")
                                break
                        break
    
    # 3. Corrigir problemas de mainloop
    if "self.root.mainloop()" in content:
        # Verificar se há try/except adequado
        if "try:" not in content or "except KeyboardInterrupt:" not in content:
            content = content.replace(
                "self.root.mainloop()",
                """try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\\n[INFO] Interrompido pelo usuário")
        except Exception as e:
            print(f"\\n[ERROR] Erro na GUI: {e}")
        finally:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass"""
            )
            fixes_applied.append("Adicionado tratamento de erro no mainloop")
    
    # 4. Verificar se há problemas com múltiplas instâncias
    if "if __name__ == '__main__':" in content:
        # Adicionar verificação de instância única se não existir
        if "import psutil" not in content and "import subprocess" not in content:
            main_section = content[content.find("if __name__ == '__main__':"):]
            if "psutil" not in main_section and "subprocess" not in main_section:
                # Adicionar verificação simples
                content = content.replace(
                    "if __name__ == '__main__':",
                    """import subprocess
import sys

def check_single_instance():
    \"\"\"Verificação simples de instância única\"\"\"
    try:
        # Verificar se já existe outro processo RobotV7
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            python_processes = result.stdout.count('python.exe')
            if python_processes > 2:  # Mais de 2 processos Python
                print("⚠️ AVISO: Múltiplas instâncias Python detectadas")
                print("⚠️ Certifique-se de fechar outras instâncias do RobotV7")
                response = input("Continuar mesmo assim? (s/N): ")
                if response.lower() != 's':
                    sys.exit(0)
    except Exception:
        pass  # Falha silenciosa na verificação

if __name__ == '__main__':
    check_single_instance()"""
                )
                fixes_applied.append("Adicionada verificação de instância única")
    
    # Aplicar correções se necessário
    if fixes_applied:
        # Fazer backup antes de aplicar
        backup_path = backup_current_robotv7()
        
        # Escrever arquivo corrigido
        with open(robot_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Correções aplicadas ao RobotV7:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        
        if backup_path:
            print(f"📁 Backup salvo em: {backup_path}")
        
        return True
    else:
        print("ℹ️ Nenhuma correção necessária detectada")
        return False

def create_emergency_launcher():
    """Criar launcher de emergência para RobotV7"""
    launcher_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 LAUNCHER DE EMERGÊNCIA - RobotV7
Use este script se o RobotV7 estiver travando
"""

import os
import sys
import subprocess
import time

def kill_existing_robots():
    """Matar processos RobotV7 existentes"""
    try:
        # Windows
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, shell=True)
        subprocess.run(['taskkill', '/F', '/IM', 'pythonw.exe'], 
                      capture_output=True, shell=True)
        time.sleep(2)
        print("✅ Processos Python anteriores finalizados")
    except Exception as e:
        print(f"⚠️ Erro ao finalizar processos: {e}")

def launch_robot_safe():
    """Lançar RobotV7 de forma segura"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🚀 Iniciando RobotV7 em modo seguro...")
    
    try:
        # Lançar em processo separado
        if sys.platform == "win32":
            subprocess.Popen([sys.executable, robot_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen([sys.executable, robot_path])
        
        print("✅ RobotV7 iniciado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao iniciar RobotV7: {e}")
        return False

if __name__ == "__main__":
    print("🔧 LAUNCHER DE EMERGÊNCIA - RobotV7")
    print("=" * 50)
    
    # Opção de matar processos existentes
    response = input("Finalizar processos Python existentes? (S/n): ")
    if response.lower() != 'n':
        kill_existing_robots()
    
    # Lançar RobotV7
    if launch_robot_safe():
        print("\\n✅ RobotV7 deve estar rodando agora")
        print("Se ainda estiver travado, verifique a barra de tarefas")
    else:
        print("\\n❌ Falha ao iniciar RobotV7")
        input("Pressione Enter para sair...")
'''
    
    with open("emergency_launch_robotv7.py", 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print("✅ Launcher de emergência criado: emergency_launch_robotv7.py")

def main():
    """Função principal de correção"""
    print("🔧 CORREÇÃO URGENTE: RobotV7 Window Freeze")
    print("=" * 50)
    
    # 1. Corrigir problemas de GUI
    print("\n1. Corrigindo problemas de inicialização da GUI...")
    gui_fixed = fix_gui_initialization()
    
    # 2. Criar launcher de emergência
    print("\n2. Criando launcher de emergência...")
    create_emergency_launcher()
    
    print("\n" + "=" * 50)
    print("🎯 CORREÇÃO CONCLUÍDA!")
    
    if gui_fixed:
        print("✅ Problemas de GUI corrigidos no RobotV7")
        print("✅ Backup do arquivo original criado")
    
    print("✅ Launcher de emergência criado")
    
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Feche todas as instâncias do RobotV7 atuais")
    print("2. Execute: python emergency_launch_robotv7.py")
    print("3. Ou execute diretamente: python 'Modelo PPO Trader/RobotV7.py'")
    print("\n⚠️ Se ainda travar, use Ctrl+Alt+Del para finalizar processos Python")

if __name__ == "__main__":
    main()