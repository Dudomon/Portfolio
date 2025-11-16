#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 CORREÇÃO ESPECÍFICA: Window Restore Issue
Corrige especificamente o problema de janela não restaurar da barra de tarefas
"""

import os
import re

def fix_window_restore():
    """Corrigir problema específico de restauração de janela"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ Arquivo não encontrado: {robot_path}")
        return False
    
    # Ler arquivo
    with open(robot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes_applied = []
    
    # 1. Encontrar a classe TradingGUI e corrigir inicialização da janela
    if "class TradingGUI:" in content:
        # Procurar por problemas comuns que causam travamento
        
        # Problema 1: root.withdraw() sem deiconify correspondente
        if "self.root.withdraw()" in content and "self.root.deiconify()" not in content:
            content = content.replace(
                "self.root.withdraw()",
                "# self.root.withdraw()  # REMOVIDO - causava travamento"
            )
            fixes_applied.append("Removido withdraw() problemático")
        
        # Problema 2: Configurações de janela que podem causar travamento
        window_config_fixes = [
            ("self.root.state('withdrawn')", "# self.root.state('withdrawn')  # REMOVIDO"),
            ("self.root.iconify()", "# self.root.iconify()  # REMOVIDO"),
            ("self.root.wm_state('iconic')", "# self.root.wm_state('iconic')  # REMOVIDO")
        ]
        
        for old, new in window_config_fixes:
            if old in content:
                content = content.replace(old, new)
                fixes_applied.append(f"Removido comando problemático: {old}")
        
        # Problema 3: Adicionar configurações corretas de janela
        gui_init_pattern = r"(class TradingGUI:.*?def __init__\(self, robot\):.*?)(self\.root\.title\(.*?\))"
        match = re.search(gui_init_pattern, content, re.DOTALL)
        
        if match:
            # Inserir configurações corretas após title
            title_line = match.group(2)
            window_fixes = f"""{title_line}
        
        # 🔧 CORREÇÃO: Configurações de janela para evitar travamento
        self.root.attributes('-topmost', False)  # Não forçar sempre no topo
        self.root.lift()  # Trazer para frente
        self.root.focus_force()  # Forçar foco
        self.root.deiconify()  # Garantir que não está minimizada
        
        # Configurar comportamento de fechamento
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)"""
            
            content = content.replace(title_line, window_fixes)
            fixes_applied.append("Adicionadas configurações corretas de janela")
        
        # Problema 4: Adicionar método on_closing se não existir
        if "def on_closing(self):" not in content:
            # Encontrar o final da classe TradingGUI
            gui_class_pattern = r"(class TradingGUI:.*?)(\nclass|\n\ndef|\n\nif __name__)"
            match = re.search(gui_class_pattern, content, re.DOTALL)
            
            if match:
                gui_class_content = match.group(1)
                rest_content = content[match.end(1):]
                
                on_closing_method = """
    def on_closing(self):
        \"\"\"Método para fechar a aplicação corretamente\"\"\"
        try:
            if hasattr(self, 'robot') and self.robot:
                self.robot.stop_trading()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Erro ao fechar: {e}")
        finally:
            import sys
            sys.exit(0)
"""
                content = gui_class_content + on_closing_method + rest_content
                fixes_applied.append("Adicionado método on_closing")
    
    # 2. Corrigir problemas no main
    if "if __name__ == '__main__':" in content:
        main_section_start = content.find("if __name__ == '__main__':")
        main_section = content[main_section_start:]
        
        # Verificar se há tratamento adequado de exceções
        if "try:" not in main_section or "except" not in main_section:
            # Encontrar onde está o código principal
            if "gui = TradingGUI(robot)" in main_section:
                old_main = main_section
                new_main = main_section.replace(
                    "gui = TradingGUI(robot)",
                    """try:
        gui = TradingGUI(robot)
        print("✅ GUI inicializada com sucesso")
    except Exception as e:
        print(f"❌ Erro ao inicializar GUI: {e}")
        import traceback
        traceback.print_exc()
        input("Pressione Enter para sair...")
        sys.exit(1)"""
                )
                content = content.replace(old_main, new_main)
                fixes_applied.append("Adicionado tratamento de erro na inicialização")
    
    # Aplicar correções
    if fixes_applied:
        with open(robot_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Correções específicas aplicadas:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        return True
    else:
        print("ℹ️ Nenhuma correção adicional necessária")
        return False

def create_test_launcher():
    """Criar launcher de teste para verificar se a correção funcionou"""
    test_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE: Verificar se RobotV7 abre corretamente
"""

import subprocess
import sys
import time
import os

def test_robotv7():
    """Testar se RobotV7 abre sem travar"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 Testando RobotV7...")
    print("⏳ Aguarde 10 segundos para verificar se a janela abre...")
    
    try:
        # Lançar RobotV7
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar 10 segundos
        time.sleep(10)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 está rodando!")
            print("✅ Se você consegue ver a janela, a correção funcionou!")
            
            response = input("\\nA janela do RobotV7 está visível? (s/N): ")
            if response.lower() == 's':
                print("🎉 SUCESSO! Correção funcionou!")
                return True
            else:
                print("❌ Janela ainda não está visível")
                print("💡 Tente clicar no ícone na barra de tarefas")
                
                # Tentar finalizar o processo
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
                return False
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTE DE CORREÇÃO - RobotV7")
    print("=" * 40)
    
    success = test_robotv7()
    
    if not success:
        print("\\n💡 DICAS ADICIONAIS:")
        print("1. Verifique se há múltiplas instâncias rodando")
        print("2. Tente reiniciar o computador")
        print("3. Execute como administrador")
        print("4. Verifique antivírus/firewall")
    
    input("\\nPressione Enter para sair...")
'''
    
    with open("test_robotv7_fix.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Script de teste criado: test_robotv7_fix.py")

def main():
    """Função principal"""
    print("🔧 CORREÇÃO ESPECÍFICA: Window Restore Issue")
    print("=" * 50)
    
    # Aplicar correções específicas
    print("Aplicando correções específicas para restauração de janela...")
    fixed = fix_window_restore()
    
    # Criar script de teste
    print("\\nCriando script de teste...")
    create_test_launcher()
    
    print("\\n" + "=" * 50)
    print("🎯 CORREÇÃO ESPECÍFICA CONCLUÍDA!")
    
    if fixed:
        print("✅ Correções aplicadas ao RobotV7")
    
    print("✅ Script de teste criado")
    
    print("\\n📋 TESTE A CORREÇÃO:")
    print("1. Execute: python test_robotv7_fix.py")
    print("2. Ou execute diretamente o RobotV7")
    print("3. Verifique se a janela aparece corretamente")

if __name__ == "__main__":
    main()