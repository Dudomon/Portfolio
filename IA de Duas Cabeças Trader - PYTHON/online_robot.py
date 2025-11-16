#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legion AI Trader - Robô Online com Google Drive
☁️ Sistema completo de trading com backend em nuvem
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import threading
import time

# Adicionar caminhos necessários
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Importar sistema de login simplificado
    from simple_login import SimpleAccessControl
    
    # Importar RobotV3
    sys.path.insert(0, os.path.join(current_dir, "Modelo PPO Trader"))
    from RobotV3 import TradingApp
    
    SYSTEM_OK = True
    print("✅ Sistema de login simplificado carregado")
        
except ImportError as e:
    print(f"❌ Erro ao importar dependências: {e}")
    SYSTEM_OK = False

class OnlineTradingApp:
    """Aplicação de trading online"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_main_interface()
        
        if SYSTEM_OK:
            self.access_control = SimpleAccessControl()
        else:
            self.access_control = None
            
        self.user_data = None
        self.robot_app = None
        self.is_authenticated = False
        self.is_robot_running = False
        
    def setup_main_interface(self):
        """Configura interface principal"""
        self.root.title("☁️ Legion AI Trader - Sistema Online")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a1a')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Cabeçalho
        header_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=3)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(header_frame, text="LEGION AI TRADER", 
                              font=('Arial', 28, 'bold'), fg='#00ff88', bg='#2d2d2d')
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(header_frame, text="🌐 Sistema de Trading Online", 
                                 font=('Arial', 16), fg='#ffffff', bg='#2d2d2d')
        subtitle_label.pack(pady=(0, 15))
        
        # Status do sistema
        status_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Status de sistema
        connection_frame = tk.Frame(status_frame, bg='#2d2d2d')
        connection_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(connection_frame, text="🔧 Sistema:", 
                font=('Arial', 12, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(side=tk.LEFT, padx=10)
        
        self.connection_status = tk.Label(connection_frame, 
                                        text="✅ FUNCIONANDO" if SYSTEM_OK else "❌ ERRO",
                                        font=('Arial', 12, 'bold'), 
                                        fg='#00ff88' if SYSTEM_OK else '#ff6666', 
                                        bg='#2d2d2d')
        self.connection_status.pack(side=tk.LEFT, padx=5)
        
        # Status de autenticação
        auth_frame = tk.Frame(status_frame, bg='#2d2d2d')
        auth_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(auth_frame, text="🔐 Autenticação:", 
                font=('Arial', 12, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(side=tk.LEFT, padx=10)
        
        self.auth_status = tk.Label(auth_frame, text="❌ NÃO AUTENTICADO",
                                   font=('Arial', 12, 'bold'), fg='#ff6666', bg='#2d2d2d')
        self.auth_status.pack(side=tk.LEFT, padx=5)
        
        # Botões de controle
        control_frame = tk.Frame(main_frame, bg='#1a1a1a')
        control_frame.pack(pady=20)
        
        # Botão de login
        self.login_btn = tk.Button(control_frame, text="🔐 FAZER LOGIN", 
                                  command=self.show_login,
                                  bg='#00ff88', fg='black',
                                  font=('Arial', 14, 'bold'), 
                                  width=25, height=2,
                                  state=tk.NORMAL if SYSTEM_OK else tk.DISABLED)
        self.login_btn.pack(pady=10)
        
        # Botão de iniciar robô
        self.start_btn = tk.Button(control_frame, text="🚀 INICIAR ROBÔ", 
                                  command=self.start_robot,
                                  bg='#ffaa00', fg='black',
                                  font=('Arial', 14, 'bold'), 
                                  width=25, height=2,
                                  state=tk.DISABLED)
        self.start_btn.pack(pady=5)
        
        # Botão de parar robô
        self.stop_btn = tk.Button(control_frame, text="🛑 PARAR ROBÔ", 
                                 command=self.stop_robot,
                                 bg='#ff6666', fg='white',
                                 font=('Arial', 14, 'bold'), 
                                 width=25, height=2,
                                 state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        # Botão de logout
        self.logout_btn = tk.Button(control_frame, text="🚪 LOGOUT", 
                                   command=self.logout,
                                   bg='#666666', fg='white',
                                   font=('Arial', 14, 'bold'), 
                                   width=25, height=2,
                                   state=tk.DISABLED)
        self.logout_btn.pack(pady=5)
        
        # Informações do usuário
        self.user_info_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        self.user_info_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(self.user_info_frame, text="👤 INFORMAÇÕES DO USUÁRIO", 
                font=('Arial', 14, 'bold'), fg='#00ff88', bg='#2d2d2d').pack(pady=10)
        
        self.user_info_text = tk.Label(self.user_info_frame, text="Nenhum usuário logado", 
                                      font=('Arial', 11), fg='#ffffff', bg='#2d2d2d',
                                      justify=tk.LEFT)
        self.user_info_text.pack(pady=(0, 15))
        
        # Log de atividades
        log_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        tk.Label(log_frame, text="📋 LOG DE ATIVIDADES", 
                font=('Arial', 14, 'bold'), fg='#00ff88', bg='#2d2d2d').pack(pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, bg='#1a1a1a', fg='#ffffff',
                               font=('Consolas', 10), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Verificar sistema
        self.check_system_status()
        
    def log_message(self, message: str, level: str = "info"):
        """Adiciona mensagem ao log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        print(formatted_msg)
        
    def check_system_status(self):
        """Verifica status do sistema"""
        if not SYSTEM_OK:
            self.log_message("❌ Sistema com problemas - verifique instalação", "error")
            self.log_message("💡 Execute: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client", "info")
            return
            
        self.log_message("✅ Sistema inicializado com sucesso", "success")
        
        # Verificar sessão salva
        try:
            if hasattr(self, 'access_control') and self.access_control and self.access_control.validate_current_session():
                self.user_data = self.access_control.current_session
                self.update_interface_authenticated()
                self.log_message("✅ Sessão restaurada automaticamente", "success")
            else:
                self.log_message("🔒 Nenhuma sessão válida encontrada", "info")
        except Exception as e:
            self.log_message(f"⚠️ Erro ao verificar sessão: {e}", "warning")
            
    def show_login(self):
        """Mostra janela de login"""
        if not SYSTEM_OK:
            messagebox.showerror("Erro", "Sistema não está funcionando corretamente!")
            return
            
        self.log_message("🔐 Abrindo sistema de login...", "info")
        
        # Desabilitar botão durante login
        self.login_btn.config(state=tk.DISABLED)
        
        def login_thread():
            try:
                user_data = self.access_control.require_login()
                self.root.after(0, lambda: self.handle_login_result(user_data))
            except Exception as e:
                self.root.after(0, lambda: self.handle_login_error(str(e)))
        
        threading.Thread(target=login_thread, daemon=True).start()
        
    def handle_login_result(self, user_data):
        """Processa resultado do login"""
        self.login_btn.config(state=tk.NORMAL)
        
        if user_data:
            self.user_data = user_data
            self.is_authenticated = True
            self.update_interface_authenticated()
            self.log_message(f"✅ Login realizado: {user_data['username']}", "success")
        else:
            self.log_message("❌ Login cancelado ou falhou", "error")
            
    def handle_login_error(self, error_msg):
        """Processa erro de login"""
        self.login_btn.config(state=tk.NORMAL)
        self.log_message(f"❌ Erro no login: {error_msg}", "error")
        messagebox.showerror("Erro de Login", f"Erro:\n{error_msg}")
        
    def update_interface_authenticated(self):
        """Atualiza interface após autenticação"""
        if not self.user_data:
            return
            
        # Atualizar status
        self.auth_status.config(text=f"✅ {self.user_data['username']}", fg='#00ff88')
        
        # Informações do usuário
        limits = self.access_control.get_user_limits()
        user_info = f"""
👤 Usuário: {self.user_data['username']}
🔑 Nível: {self.user_data['access_level']}
📊 Trades/Dia: {limits['max_daily_trades']}
📉 Drawdown Máx: {limits['max_drawdown_percent']}%
💰 Lote Máximo: {limits['max_lot_size']}
        """.strip()
        
        self.user_info_text.config(text=user_info)
        
        # Atualizar botões
        self.login_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)
        self.logout_btn.config(state=tk.NORMAL)
        
    def start_robot(self):
        """Inicia o robô"""
        if not self.is_authenticated:
            messagebox.showerror("Erro", "Faça login primeiro!")
            return
            
        self.log_message("🚀 Iniciando robô...", "info")
        
        def robot_thread():
            try:
                self.robot_app = TradingApp()
                self.is_robot_running = True
                self.root.after(0, self.update_robot_started)
                self.robot_app.run()
                
            except Exception as error:
                self.root.after(0, lambda: self.handle_robot_error(str(error)))
                
        threading.Thread(target=robot_thread, daemon=True).start()
        
    def update_robot_started(self):
        """Atualiza quando robô inicia"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log_message("✅ Robô iniciado!", "success")
        
    def handle_robot_error(self, error_msg):
        """Processa erro do robô"""
        self.is_robot_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_message(f"❌ Erro no robô: {error_msg}", "error")
        
    def stop_robot(self):
        """Para o robô"""
        self.log_message("🛑 Parando robô...", "warning")
        self.is_robot_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_message("✅ Robô parado!", "success")
            
    def logout(self):
        """Faz logout"""
        if self.is_robot_running:
            self.stop_robot()
            
        try:
            if self.access_control:
                self.access_control.logout()
                
            self.user_data = None
            self.is_authenticated = False
            
            # Resetar interface
            self.auth_status.config(text="❌ NÃO AUTENTICADO", fg='#ff6666')
            self.user_info_text.config(text="Nenhum usuário logado")
            
            self.login_btn.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.DISABLED)
            self.logout_btn.config(state=tk.DISABLED)
            
            self.log_message("🚪 Logout realizado!", "success")
            
        except Exception as e:
            self.log_message(f"⚠️ Erro no logout: {e}", "warning")
            
    def run(self):
        """Executa a aplicação"""
        self.log_message("🌐 Legion AI Trader iniciado!", "success")
        
        if not SYSTEM_OK:
            self.log_message("💡 Configure as dependências para usar o sistema", "info")
            
        self.root.mainloop()

def main():
    """Função principal"""
    print("=" * 60)
    print("☁️ LEGION AI TRADER - SISTEMA ONLINE")
    print("=" * 60)
    
    # Verificar credentials.json
    if not os.path.exists("credentials.json"):
        print("⚠️ AVISO: credentials.json não encontrado!")
        print("💡 Para usar Google Drive:")
        print("   1. Configure Google Cloud Console")
        print("   2. Baixe credentials.json")
        print("   3. Coloque na pasta do projeto")
        print()
        
    try:
        app = OnlineTradingApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Sistema encerrado")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main() 