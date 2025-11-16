#!/usr/bin/env python3
"""
🤖 ROBOTV7 LEGION - LAUNCHER HÍBRIDO
====================================

Executável principal que:
1. Carrega GUI ORIGINAL
2. Gerencia dependências automaticamente  
3. Protege código fonte
4. Funciona standalone
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
import importlib.util

class RobotV7Launcher:
    """Launcher inteligente para RobotV7"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RobotV7 Legion - Launcher")
        self.root.geometry("500x300")
        self.root.configure(bg='#1e1e1e')
        
        # Configurar diretórios - usar sempre diretório atual
        self.launcher_dir = os.getcwd()
        self.robotlogin_path = os.path.join(self.launcher_dir, "__pycache__", "robotlogin.pyc")
        
        # Estado
        self.dependencies_checked = False
        self.robotlogin_loaded = False
        
        # Setup GUI
        self.setup_launcher_gui()
        
    def setup_launcher_gui(self):
        """Interface do launcher"""
        # Título
        title_label = tk.Label(
            self.root,
            text="🤖 ROBOTV7 LEGION",
            font=('Arial', 20, 'bold'),
            fg='#00ff41',
            bg='#1e1e1e'
        )
        title_label.pack(pady=30)
        
        subtitle_label = tk.Label(
            self.root,
            text="Professional Trading System",
            font=('Arial', 12),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        subtitle_label.pack(pady=10)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Initializing system...",
            font=('Arial', 10),
            fg='#ffaa00',
            bg='#1e1e1e'
        )
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=10)
        
        # Botões
        button_frame = tk.Frame(self.root, bg='#1e1e1e')
        button_frame.pack(pady=20)
        
        self.start_button = tk.Button(
            button_frame,
            text="🚀 START ROBOTV7",
            command=self.start_robotv7,
            font=('Arial', 12, 'bold'),
            bg='#00aa00',
            fg='#ffffff',
            width=15,
            state='disabled'
        )
        self.start_button.pack(side='left', padx=5)
        
        self.exit_button = tk.Button(
            button_frame,
            text="❌ EXIT",
            command=self.root.quit,
            font=('Arial', 12, 'bold'),
            bg='#aa0000',
            fg='#ffffff',
            width=10
        )
        self.exit_button.pack(side='right', padx=5)
        
        # Inicializar verificações em thread
        threading.Thread(target=self.initialize_system, daemon=True).start()
    
    def update_status(self, message, color='#ffaa00'):
        """Atualizar status na GUI"""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def initialize_system(self):
        """Inicializar sistema em background"""
        try:
            self.progress.start()
            
            # 1. Verificar arquivos essenciais
            self.update_status("Checking essential files...")
            time.sleep(1)
            
            if not os.path.exists(self.robotlogin_path):
                self.update_status("❌ robotlogin.pyc not found!", '#ff4444')
                messagebox.showerror("Error", f"robotlogin.pyc not found at: {self.robotlogin_path}")
                return
            
            # 2. Verificar dependências básicas
            self.update_status("Checking basic dependencies...")
            time.sleep(1)
            
            basic_deps = ['tkinter', 'threading', 'datetime']
            for dep in basic_deps:
                try:
                    __import__(dep)
                except ImportError:
                    self.update_status(f"❌ Missing basic dependency: {dep}", '#ff4444')
                    messagebox.showerror("Error", f"Missing basic dependency: {dep}")
                    return
            
            # 3. Verificar dependências avançadas (opcionais)
            self.update_status("Checking advanced dependencies...")
            time.sleep(1)
            
            advanced_deps = {
                'numpy': 'pip install numpy',
                'pandas': 'pip install pandas', 
                'requests': 'pip install requests',
                'cryptography': 'pip install cryptography'
            }
            
            missing_deps = []
            for dep, install_cmd in advanced_deps.items():
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append((dep, install_cmd))
            
            # 4. Instalar dependências em falta (se houver)
            if missing_deps:
                self.update_status("Installing missing dependencies...")
                for dep, cmd in missing_deps:
                    try:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                        self.update_status(f"✅ Installed {dep}")
                        time.sleep(0.5)
                    except subprocess.CalledProcessError:
                        self.update_status(f"⚠️ Failed to install {dep}", '#ff4444')
            
            # 5. Verificar robotlogin.py
            self.update_status("Validating RobotV7 system...")
            time.sleep(1)
            
            # Tentar import do robotlogin (sem executar)
            spec = importlib.util.spec_from_file_location("robotlogin", self.robotlogin_path)
            if spec is None:
                self.update_status("❌ Invalid robotlogin.py", '#ff4444')
                return
                
            # 6. Sistema pronto
            self.dependencies_checked = True
            self.robotlogin_loaded = True
            
            self.progress.stop()
            self.update_status("✅ System ready - Click START ROBOTV7", '#00ff00')
            self.start_button.config(state='normal')
            
        except Exception as e:
            self.progress.stop()
            self.update_status(f"❌ Initialization failed: {str(e)}", '#ff4444')
            messagebox.showerror("Error", f"System initialization failed:\n{str(e)}")
    
    def start_robotv7(self):
        """Iniciar RobotV7 com GUI original"""
        if not (self.dependencies_checked and self.robotlogin_loaded):
            messagebox.showerror("Error", "System not ready. Please wait for initialization.")
            return
        
        try:
            self.update_status("Starting RobotV7...")
            
            # Fechar launcher
            self.root.withdraw()  # Esconder janela
            
            # Executar robotlogin.py com GUI original
            process = subprocess.Popen(
                [sys.executable, self.robotlogin_path],
                cwd=self.launcher_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )
            
            # Aguardar um pouco para ver se inicia corretamente
            time.sleep(2)
            
            # Verificar se processo está rodando
            if process.poll() is None:  # Ainda rodando
                print("✅ RobotV7 started successfully with original GUI")
                self.root.quit()  # Fechar launcher
            else:
                # Processo terminou rapidamente (erro)
                self.root.deiconify()  # Mostrar launcher novamente
                self.update_status("❌ RobotV7 failed to start", '#ff4444')
                messagebox.showerror("Error", "RobotV7 failed to start. Check console for details.")
                
        except Exception as e:
            self.root.deiconify()  # Mostrar launcher novamente
            self.update_status(f"❌ Launch failed: {str(e)}", '#ff4444')
            messagebox.showerror("Error", f"Failed to start RobotV7:\n{str(e)}")
    
    def run(self):
        """Executar launcher"""
        print("RobotV7 Legion - Hybrid Launcher")
        print("=" * 40)
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nLauncher interrupted by user")
        except Exception as e:
            print(f"Launcher error: {e}")

def main():
    """Função principal"""
    launcher = RobotV7Launcher()
    launcher.run()

if __name__ == "__main__":
    main()