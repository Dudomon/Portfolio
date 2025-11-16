#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 SOLUÇÃO RADICAL: GUI Simples e Confiável para RobotV7
Substituir a GUI complexa por uma versão simples que funciona
"""

import os
import shutil
from datetime import datetime

def create_simple_gui():
    """Criar uma GUI completamente nova e simples"""
    
    simple_gui_code = '''
# === GUI SIMPLES E CONFIÁVEL ===
class SimpleRobotGUI:
    """GUI simples e confiável - sem problemas de minimização"""
    
    def __init__(self, robot):
        self.robot = robot
        self.trading_active = False
        self.trading_thread = None
        self.stop_event = Event()
        
        # Criar janela principal
        self.root = tk.Tk()
        self.root.title("Legion AI Trader V7 - Simple GUI")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        # 🔧 CONFIGURAÇÕES ANTI-TRAVAMENTO
        self.root.resizable(True, True)
        self.root.minsize(600, 400)
        self.root.state('normal')
        
        # Não usar topmost, withdraw, iconify ou outros comandos problemáticos
        # Apenas configurações básicas e seguras
        
        self.setup_simple_gui()
        self.robot.log_widget = self.log_text
        
        # Protocolo de fechamento simples
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # Iniciar atualizações
        self.update_display()
    
    def setup_simple_gui(self):
        """Configurar interface simples"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(main_frame, text="🤖 Legion AI Trader V7", 
                              font=('Arial', 16, 'bold'), 
                              fg='#00ff00', bg='#2b2b2b')
        title_label.pack(pady=(0, 10))
        
        # Frame de controles
        control_frame = tk.Frame(main_frame, bg='#2b2b2b')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Botões principais
        self.model_button = tk.Button(control_frame, text="📁 SELECT MODEL", 
                                     command=self.select_model,
                                     bg='#4a4a4a', fg='white', font=('Arial', 10, 'bold'))
        self.model_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.start_button = tk.Button(control_frame, text="▶️ START TRADING", 
                                     command=self.start_trading,
                                     bg='#006600', fg='white', font=('Arial', 10, 'bold'))
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = tk.Button(control_frame, text="⏹️ STOP TRADING", 
                                    command=self.stop_trading,
                                    bg='#cc0000', fg='white', font=('Arial', 10, 'bold'))
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status
        self.status_label = tk.Label(control_frame, text="Status: Stopped", 
                                    fg='#ffff00', bg='#2b2b2b', font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT)
        
        # Frame de informações
        info_frame = tk.Frame(main_frame, bg='#2b2b2b')
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Informações básicas
        self.balance_label = tk.Label(info_frame, text="Balance: $500.00", 
                                     fg='#00ff00', bg='#2b2b2b', font=('Arial', 10))
        self.balance_label.pack(side=tk.LEFT)
        
        self.positions_label = tk.Label(info_frame, text="Positions: 0", 
                                       fg='#00ffff', bg='#2b2b2b', font=('Arial', 10))
        self.positions_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.model_label = tk.Label(info_frame, text="Model: Not loaded", 
                                   fg='#ff8800', bg='#2b2b2b', font=('Arial', 10))
        self.model_label.pack(side=tk.RIGHT)
        
        # Log de texto
        log_frame = tk.Frame(main_frame, bg='#2b2b2b')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(log_frame, text="📋 Trading Log:", 
                fg='white', bg='#2b2b2b', font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Área de log com scrollbar
        log_container = tk.Frame(log_frame, bg='#2b2b2b')
        log_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_container, 
                                                 height=20, 
                                                 bg='#1a1a1a', 
                                                 fg='#00ff00',
                                                 font=('Consolas', 9),
                                                 wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log inicial
        self.log_text.insert(tk.END, "🤖 Legion AI Trader V7 - Simple GUI\\n")
        self.log_text.insert(tk.END, "✅ GUI inicializada com sucesso\\n")
        self.log_text.insert(tk.END, "📋 Use SELECT MODEL para carregar um modelo\\n\\n")
    
    def select_model(self):
        """Selecionar modelo"""
        try:
            from tkinter import filedialog
            model_path = filedialog.askopenfilename(
                title="Selecionar Modelo",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
            )
            
            if model_path:
                self.log_text.insert(tk.END, f"📁 Modelo selecionado: {os.path.basename(model_path)}\\n")
                self.model_label.config(text=f"Model: {os.path.basename(model_path)}")
                
                # Carregar modelo no robot
                success = self.robot.load_model_from_path(model_path)
                if success:
                    self.log_text.insert(tk.END, "✅ Modelo carregado com sucesso\\n")
                    self.start_button.config(state=tk.NORMAL)
                else:
                    self.log_text.insert(tk.END, "❌ Erro ao carregar modelo\\n")
                
                self.log_text.see(tk.END)
                
        except Exception as e:
            self.log_text.insert(tk.END, f"❌ Erro ao selecionar modelo: {e}\\n")
            self.log_text.see(tk.END)
    
    def start_trading(self):
        """Iniciar trading"""
        if not self.trading_active and self.robot.model_loaded:
            self.trading_active = True
            self.stop_event.clear()
            
            self.trading_thread = Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            
            self.status_label.config(text="Status: Trading", fg='#00ff00')
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            self.log_text.insert(tk.END, "▶️ Trading iniciado\\n")
            self.log_text.see(tk.END)
    
    def stop_trading(self):
        """Parar trading"""
        if self.trading_active:
            self.trading_active = False
            self.stop_event.set()
            
            self.status_label.config(text="Status: Stopping...", fg='#ffff00')
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            self.log_text.insert(tk.END, "⏹️ Trading parado\\n")
            self.log_text.see(tk.END)
    
    def trading_loop(self):
        """Loop principal de trading"""
        try:
            while self.trading_active and not self.stop_event.is_set():
                if self.robot.model_loaded:
                    # Executar step do robot
                    self.robot.step()
                
                # Aguardar intervalo
                time.sleep(1)
                
        except Exception as e:
            self.log_text.insert(tk.END, f"❌ Erro no trading loop: {e}\\n")
            self.log_text.see(tk.END)
        finally:
            self.trading_active = False
            self.status_label.config(text="Status: Stopped", fg='#ffff00')
    
    def update_display(self):
        """Atualizar display"""
        try:
            # Atualizar informações
            if hasattr(self.robot, 'portfolio_value'):
                self.balance_label.config(text=f"Balance: ${self.robot.portfolio_value:.2f}")
            
            if hasattr(self.robot, 'current_positions'):
                self.positions_label.config(text=f"Positions: {self.robot.current_positions}")
            
        except Exception:
            pass
        
        # Reagendar atualização
        self.root.after(2000, self.update_display)
    
    def close_app(self):
        """Fechar aplicação"""
        try:
            self.stop_trading()
            time.sleep(1)
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
    
    def run(self):
        """Executar GUI"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Erro na GUI: {e}")

def main_simple_gui():
    """Função principal para GUI simples"""
    try:
        robot = TradingRobotV7()
        gui = SimpleRobotGUI(robot)
        gui.run()
    except Exception as e:
        print(f"Erro fatal: {e}")
        import traceback
        traceback.print_exc()
'''
    
    return simple_gui_code

def replace_gui_in_robotv7():
    """Substituir a GUI complexa pela simples no RobotV7"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ Arquivo não encontrado: {robot_path}")
        return False
    
    # Fazer backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"Modelo PPO Trader/RobotV7_backup_simple_{timestamp}.py"
    shutil.copy2(robot_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Ler arquivo
    with open(robot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Encontrar onde começa a classe TradingAppV7
    start_pos = content.find("class TradingAppV7:")
    if start_pos == -1:
        print("❌ Classe TradingAppV7 não encontrada")
        return False
    
    # Encontrar onde termina (próxima função main_gui ou final do arquivo)
    end_pos = content.find("def main_gui():", start_pos)
    if end_pos == -1:
        end_pos = len(content)
    
    # Substituir a GUI complexa pela simples
    simple_gui_code = create_simple_gui()
    
    new_content = (content[:start_pos] + 
                   simple_gui_code + 
                   "\n\ndef main_gui():\n" +
                   '    """Função principal para iniciar GUI simples"""\n' +
                   "    main_simple_gui()\n\n" +
                   content[end_pos:])
    
    # Escrever arquivo modificado
    with open(robot_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ GUI complexa substituída pela GUI simples")
    return True

def create_test_simple_gui():
    """Criar teste para a GUI simples"""
    test_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE: GUI Simples do RobotV7
Testar se a nova GUI simples funciona sem travamentos
"""

import subprocess
import sys
import time
import os

def test_simple_gui():
    """Testar GUI simples"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 TESTE: GUI Simples")
    print("=" * 30)
    print("⏳ Iniciando RobotV7 com GUI simples...")
    
    try:
        # Lançar RobotV7
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar inicialização
        time.sleep(5)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 com GUI simples iniciado!")
            print()
            print("📋 TESTE DE MINIMIZAÇÃO:")
            print("1. Verifique se a janela apareceu")
            print("2. Minimize a janela")
            print("3. Clique na barra de tarefas para restaurar")
            print("4. Repita várias vezes")
            print()
            
            # Aguardar teste
            input("Pressione Enter quando terminar o teste...")
            
            # Perguntar resultado
            result = input("\\nA GUI simples funcionou sem travamentos? (s/N): ")
            
            # Finalizar
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                print("✅ Processo finalizado")
            except:
                pass
            
            return result.lower() == 's'
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_gui()
    
    if success:
        print("🎉 SUCESSO! GUI simples funciona corretamente!")
    else:
        print("❌ Ainda há problemas com a GUI")
    
    input("\\nPressione Enter para sair...")
'''
    
    with open("test_simple_gui.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Teste da GUI simples criado: test_simple_gui.py")

def main():
    """Função principal"""
    print("🔧 SOLUÇÃO RADICAL: GUI Simples e Confiável")
    print("=" * 60)
    print("🎯 ESTRATÉGIA:")
    print("   • Substituir GUI complexa por versão simples")
    print("   • Remover todos os códigos problemáticos")
    print("   • Usar apenas comandos básicos e seguros do tkinter")
    print("   • Sem topmost, withdraw, iconify ou eventos complexos")
    print()
    
    # Substituir GUI
    print("Substituindo GUI complexa pela simples...")
    success = replace_gui_in_robotv7()
    
    if success:
        # Criar teste
        print("\\nCriando teste da GUI simples...")
        create_test_simple_gui()
        
        print("\\n" + "=" * 60)
        print("🎯 SOLUÇÃO RADICAL IMPLEMENTADA!")
        print("✅ GUI complexa removida")
        print("✅ GUI simples e confiável instalada")
        print("✅ Backup do arquivo original criado")
        print("✅ Teste criado")
        
        print("\\n📋 TESTE A NOVA GUI:")
        print("1. Execute: python test_simple_gui.py")
        print("2. Teste minimização/restauração várias vezes")
        print("3. Verifique se não trava mais")
        
        print("\\n🎯 CARACTERÍSTICAS DA GUI SIMPLES:")
        print("   ✅ Sem códigos problemáticos")
        print("   ✅ Interface limpa e funcional")
        print("   ✅ Botões básicos: SELECT MODEL, START, STOP")
        print("   ✅ Log de trading em tempo real")
        print("   ✅ Informações essenciais: Balance, Positions, Model")
    else:
        print("❌ Falha ao substituir GUI")

if __name__ == "__main__":
    main()