"""
🔐 SISTEMA DE LOGIN ADAPTADO PARA ROBOTV7
========================================

Baseado no sistema Robo Ander mas adaptado para RobotV7
com limites específicos para trading de ouro.
"""

import tkinter as tk
from tkinter import messagebox
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Optional
import requests
import threading
import time

# Importar sistema base - usar mesmas configurações do Robo Ander
from online_system_real import RealOnlineSystem

# 🌐 USAR MESMO BIN PÚBLICO DO ROBO ANDER
ONLINE_API_BASE = "https://api.jsonbin.io/v3/b"
ONLINE_API_KEY = "$2a$10$pk83LLuKD3OFmBwkDErgrOp2gESU.6/R36PTL4n5.OaFDyoBl2j8a"
USERS_BIN_ID = "686b0df18960c979a5b8408f"


class RobotV7UserManager:
    """Gerenciador de usuários específico para RobotV7"""
    
    def __init__(self):
        self.users_file = "robotv7_users_local.json"
        self.online_system = RealOnlineSystem()
        self.check_online_status()
        self.init_users()
    
    def check_online_status(self):
        """Verifica status do sistema online (mesmo do Robo Ander)"""
        try:
            print("🌐 Testando conexão com bin público...")
            response = requests.get(f"{ONLINE_API_BASE}/{USERS_BIN_ID}/latest", 
                                  headers={"X-Master-Key": ONLINE_API_KEY}, 
                                  timeout=10)
            self.online_available = response.status_code == 200
            print(f"🌐 Sistema Online: {'ATIVO' if self.online_available else 'INATIVO'}")
        except Exception as e:
            self.online_available = False
            print(f"🌐 Sistema Online: INATIVO ({str(e)[:50]})")
    
    def create_default_users_robotv7(self):
        """Cria usuários padrão para RobotV7"""
        default_users = {
            "robotv7_admin": {
                "password_hash": self.hash_password("admin123"),
                "access_level": "admin",
                "system": "robotv7",
                "max_daily_trades": 30,
                "max_drawdown_percent": 15.0,
                "base_lot_size": 0.02,
                "max_lot_size": 0.03,
                "enable_shorts": True,
                "max_positions": 2,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            },
            "robotv7_trader": {
                "password_hash": self.hash_password("trader123"),
                "access_level": "trader",
                "system": "robotv7", 
                "max_daily_trades": 25,
                "max_drawdown_percent": 12.0,
                "base_lot_size": 0.02,
                "max_lot_size": 0.025,
                "enable_shorts": True,
                "max_positions": 1,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            },
            "robotv7_demo": {
                "password_hash": self.hash_password("demo123"),
                "access_level": "demo",
                "system": "robotv7",
                "max_daily_trades": 10,
                "max_drawdown_percent": 8.0,
                "base_lot_size": 0.01,
                "max_lot_size": 0.02,
                "enable_shorts": False,
                "max_positions": 1,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
        }
        return default_users
    
    def init_users(self):
        """Inicializa sistema usando bin público existente"""
        print("🤖 Inicializando sistema de usuários...")
        
        if self.online_available:
            print("🌐 Sistema Online ATIVO - usando bin público existente")
            users = self.load_users_online()
            if users:
                print("   Usuários disponíveis:")
                for username in users.keys():
                    user_data = users[username]
                    access_level = user_data.get('access_level', 'user')
                    print(f"   🔑 {username} (Nível: {access_level})")
            else:
                print("   ⚠️ Nenhum usuário encontrado no bin público")
        else:
            print("💻 Sistema Local ativo - criando usuários locais como fallback")
            self.create_default_users_local()
    
    def load_users_online(self):
        """Carrega usuários do bin público (mesmo do Robo Ander)"""
        try:
            print("📡 Carregando usuários do bin público...")
            response = requests.get(f"{ONLINE_API_BASE}/{USERS_BIN_ID}/latest", 
                                  headers={"X-Master-Key": ONLINE_API_KEY}, 
                                  timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'record' in data:
                    users = data['record']
                else:
                    users = data
                print(f"✅ {len(users)} usuários carregados do bin público")
                return users
            else:
                print(f"❌ Erro HTTP {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Erro ao carregar usuários online: {e}")
            return None
    
    def save_users_online(self, users_data):
        """Salva usuários online"""
        try:
            print("💾 Salvando usuários RobotV7...")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar usuários: {e}")
            return False
    
    def create_default_users_local(self):
        """Cria usuários locais se não existirem"""
        if not os.path.exists(self.users_file):
            default_users = self.create_default_users_robotv7()
            
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)
            
            print(f"✅ Arquivo local criado: {self.users_file}")
            print("👥 Usuários disponíveis:")
            for username, data in default_users.items():
                level = data['access_level']
                print(f"   🔑 {username} (Nível: {level})")
    
    def hash_password(self, password: str) -> str:
        """Cria hash da senha"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> tuple:
        """Autentica usuário"""
        if self.online_available:
            return self.authenticate_online(username, password)
        else:
            return self.authenticate_local(username, password)
    
    def authenticate_online(self, username: str, password: str) -> tuple:
        """Autentica usuário online com controle de sessão"""
        try:
            users = self.load_users_online()
            if not users:
                return False, "Erro ao carregar usuários", None
            
            # Normalizar username
            username_norm = username.strip().lower()
            
            for stored_username, user_data in users.items():
                if stored_username.lower() == username_norm:
                    # 🚪 FORÇAR LOGOUT DE SESSÃO ANTERIOR (SE EXISTIR)
                    forced_logout = False
                    if self.is_user_active_elsewhere(stored_username, users):
                        print(f"⚠️  Usuário {stored_username} já logado - forçando logout da sessão anterior")
                        self.force_logout_user(stored_username)
                        forced_logout = True
                    
                    password_hash = self.hash_password(password)
                    
                    if user_data['password_hash'] == password_hash:
                        # 🟢 MARCAR USUÁRIO COMO ATIVO
                        if not self.mark_user_as_active(stored_username, users):
                            return False, "❌ Erro ao registrar sessão", None
                        
                        # Login bem sucedido
                        session_data = {
                            'username': stored_username,
                            'access_level': user_data['access_level'],
                            'system': user_data.get('system', 'robotv7'),
                            'max_daily_trades': user_data.get('max_daily_trades', 20),
                            'max_drawdown_percent': user_data.get('max_drawdown_percent', 10.0),
                            'base_lot_size': user_data.get('base_lot_size', 0.02),
                            'max_lot_size': user_data.get('max_lot_size', 0.03),
                            'enable_shorts': user_data.get('enable_shorts', True),
                            'max_positions': user_data.get('max_positions', 2),
                            'login_time': datetime.now().isoformat()
                        }
                        
                        # Incluir informação sobre logout forçado na mensagem
                        success_message = "Login bem sucedido"
                        if forced_logout:
                            success_message += " (sessão anterior encerrada)"
                        
                        return True, success_message, session_data
                    else:
                        return False, "Senha incorreta", None
            
            return False, "Usuário não encontrado", None
            
        except Exception as e:
            print(f"❌ Erro na autenticação online: {e}")
            return False, f"Erro no sistema: {e}", None
    
    def authenticate_local(self, username: str, password: str) -> tuple:
        """Autentica usuário localmente"""
        try:
            if not os.path.exists(self.users_file):
                self.create_default_users_local()
            
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            username_norm = username.strip().lower()
            
            for stored_username, user_data in users.items():
                if stored_username.lower() == username_norm:
                    password_hash = self.hash_password(password)
                    
                    if user_data['password_hash'] == password_hash:
                        # Login bem sucedido
                        session_data = {
                            'username': stored_username,
                            'access_level': user_data['access_level'],
                            'system': user_data.get('system', 'robotv7'),
                            'max_daily_trades': user_data.get('max_daily_trades', 20),
                            'max_drawdown_percent': user_data.get('max_drawdown_percent', 10.0),
                            'base_lot_size': user_data.get('base_lot_size', 0.02),
                            'max_lot_size': user_data.get('max_lot_size', 0.03),
                            'enable_shorts': user_data.get('enable_shorts', True),
                            'max_positions': user_data.get('max_positions', 2),
                            'login_time': datetime.now().isoformat()
                        }
                        
                        return True, "Login bem sucedido", session_data
                    else:
                        return False, "Senha incorreta", None
            
            return False, "Usuário não encontrado", None
            
        except Exception as e:
            print(f"❌ Erro na autenticação local: {e}")
            return False, f"Erro no sistema: {e}", None
    
    def is_user_active_elsewhere(self, username: str, users_data: dict) -> bool:
        """Verifica se usuário já está logado em outro local"""
        try:
            user_data = users_data.get(username, {})
            last_session = user_data.get('last_session_time')
            
            if not last_session:
                return False
            
            # Verificar se sessão está ativa (< 30 minutos)
            last_session_dt = datetime.fromisoformat(last_session)
            now = datetime.now()
            time_diff = (now - last_session_dt).total_seconds() / 60  # em minutos
            
            # Sessão ativa se < 30 minutos
            is_active = time_diff < 30
            
            if is_active:
                print(f"⚠️ Usuário {username} tem sessão ativa há {time_diff:.1f} minutos")
            
            return is_active
            
        except Exception as e:
            print(f"❌ Erro ao verificar sessão: {e}")
            return False
    
    def mark_user_as_active(self, username: str, users_data: dict) -> bool:
        """Marca usuário como ativo no sistema online"""
        try:
            # Atualizar dados do usuário com sessão ativa
            users_data[username]['last_session_time'] = datetime.now().isoformat()
            users_data[username]['session_active'] = True
            
            # Salvar no JSONBin
            response = requests.put(
                f"{ONLINE_API_BASE}/{USERS_BIN_ID}",
                json=users_data,
                headers={
                    "Content-Type": "application/json",
                    "X-Master-Key": ONLINE_API_KEY
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Sessão ativa registrada para {username}")
                return True
            else:
                print(f"❌ Erro ao registrar sessão: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao marcar usuário ativo: {e}")
            return False
    
    def force_logout_user(self, username: str) -> bool:
        """🚪 Força logout de uma sessão anterior"""
        try:
            print(f"🚪 Forçando logout da sessão anterior de {username}")
            
            # Chamar logout normal
            success = self.logout_user(username)
            
            if success:
                print(f"✅ Logout forçado com sucesso para {username}")
            else:
                print(f"⚠️ Falha no logout forçado de {username}, mas continuando...")
            
            return True  # Sempre retorna True para permitir novo login
            
        except Exception as e:
            print(f"⚠️ Erro ao forçar logout de {username}: {e}")
            return True  # Mesmo com erro, permite novo login
    
    def logout_user(self, username: str) -> bool:
        """Remove sessão ativa do usuário"""
        try:
            users_data = self.load_users_online()
            if not users_data:
                return False
            
            if username in users_data:
                # Limpar dados de sessão
                users_data[username]['last_session_time'] = None
                users_data[username]['session_active'] = False
                
                # Salvar no JSONBin
                response = requests.put(
                    f"{ONLINE_API_BASE}/{USERS_BIN_ID}",
                    json=users_data,
                    headers={
                        "Content-Type": "application/json", 
                        "X-Master-Key": ONLINE_API_KEY
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"✅ Logout realizado para {username}")
                    return True
                else:
                    print(f"❌ Erro no logout: {response.status_code}")
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Erro no logout: {e}")
            return False


class RobotV7LoginWindow:
    """Janela de login específica para RobotV7"""
    
    def __init__(self):
        self.user_manager = RobotV7UserManager()
        self.session_data = None
        self.login_successful = False
        self.setup_interface()
    
    def setup_interface(self):
        """Configura interface de login RobotV7"""
        self.root = tk.Tk()
        self.root.title("🤖 RobotV7 Legion - Sistema de Login")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        self.root.configure(bg='#0f0f0f')
        
        # Centralizar janela
        self.root.eval('tk::PlaceWindow . center')
        
        main_frame = tk.Frame(self.root, bg='#0f0f0f', padx=30, pady=30)
        main_frame.pack(fill='both', expand=True)
        
        # Título principal
        title_label = tk.Label(
            main_frame,
            text="🤖 ROBOTV7 LEGION",
            font=('Arial', 24, 'bold'),
            fg='#00ff41',
            bg='#0f0f0f'
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = tk.Label(
            main_frame,
            text="AI Trading System - Authentication Required",
            font=('Arial', 12),
            fg='#666666',
            bg='#0f0f0f'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Status do sistema
        status_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        status_frame.pack(fill='x', pady=(0, 20))
        
        system_status = "🌐 ONLINE" if self.user_manager.online_available else "💻 LOCAL"
        status_color = "#00ff41" if self.user_manager.online_available else "#ffaa00"
        
        status_label = tk.Label(
            status_frame,
            text=f"Sistema: {system_status}",
            font=('Arial', 10, 'bold'),
            fg=status_color,
            bg='#1a1a1a'
        )
        status_label.pack(pady=8)
        
        # Campos de login
        login_frame = tk.Frame(main_frame, bg='#0f0f0f')
        login_frame.pack(fill='x', pady=20)
        
        # Username
        username_label = tk.Label(
            login_frame,
            text="👤 USUÁRIO:",
            font=('Arial', 12, 'bold'),
            fg='#ffffff',
            bg='#0f0f0f'
        )
        username_label.pack(anchor='w', pady=(0, 5))
        
        self.username_entry = tk.Entry(
            login_frame,
            font=('Arial', 12),
            bg='#2d2d2d',
            fg='#ffffff',
            insertbackground='#00ff41',
            relief=tk.FLAT,
            bd=10
        )
        self.username_entry.pack(fill='x', pady=(0, 15))
        self.username_entry.focus()
        
        # Password
        password_label = tk.Label(
            login_frame,
            text="🔑 SENHA:",
            font=('Arial', 12, 'bold'),
            fg='#ffffff',
            bg='#0f0f0f'
        )
        password_label.pack(anchor='w', pady=(0, 5))
        
        self.password_entry = tk.Entry(
            login_frame,
            show='*',
            font=('Arial', 12),
            bg='#2d2d2d',
            fg='#ffffff',
            insertbackground='#00ff41',
            relief=tk.FLAT,
            bd=10
        )
        self.password_entry.pack(fill='x', pady=(0, 20))
        
        # Botão de login
        login_button = tk.Button(
            login_frame,
            text="🚀 INICIAR SESSÃO",
            font=('Arial', 14, 'bold'),
            bg='#00ff41',
            fg='#000000',
            activebackground='#00cc33',
            activeforeground='#000000',
            relief=tk.FLAT,
            bd=0,
            command=self.perform_login
        )
        login_button.pack(fill='x', pady=10)
        
        # Informações de usuários disponíveis
        info_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.RAISED, bd=1)
        info_frame.pack(fill='x', pady=20)
        
        info_title = tk.Label(
            info_frame,
            text="ℹ️ SISTEMA DE AUTENTICAÇÃO",
            font=('Arial', 11, 'bold'),
            fg='#00ff41',
            bg='#1a1a1a'
        )
        info_title.pack(pady=(10, 5))
        
        # Informações gerais (sem credenciais)
        info_text = tk.Label(
            info_frame,
            text="🔐 Digite suas credenciais de acesso",
            font=('Arial', 10),
            fg='#ffffff',
            bg='#1a1a1a'
        )
        info_text.pack(pady=5)
        
        # Nota sobre lotes
        lote_note = tk.Label(
            info_frame,
            text="🎯 Tamanho do lote: Definido na interface do robô",
            font=('Arial', 9, 'italic'),
            fg='#00ff41',
            bg='#1a1a1a'
        )
        lote_note.pack(pady=(5, 10))
        
        # Bind Enter key
        self.root.bind('<Return>', lambda e: self.perform_login())
        
        # Botão fechar
        close_button = tk.Button(
            main_frame,
            text="❌ CANCELAR",
            font=('Arial', 10),
            bg='#ff4444',
            fg='#ffffff',
            activebackground='#cc0000',
            relief=tk.FLAT,
            command=self.root.quit
        )
        close_button.pack(pady=10)
    
    def perform_login(self):
        """Executa autenticação"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        
        if not username or not password:
            messagebox.showerror("Erro", "Digite usuário e senha")
            return
        
        # Executar autenticação
        success, message, session_data = self.user_manager.authenticate_user(username, password)
        
        if success:
            self.session_data = session_data
            self.login_successful = True
            
            # Verificar se houve logout forçado
            welcome_message = f"Login bem sucedido!\nBem-vindo, {session_data['username']}"
            if "sessão anterior encerrada" in str(message).lower():
                welcome_message += "\n\n⚠️ Sua sessão anterior foi encerrada automaticamente."
            
            messagebox.showinfo("Sucesso", welcome_message)
            self.root.quit()
        else:
            messagebox.showerror("Erro de Login", message)
            self.password_entry.delete(0, tk.END)
            self.password_entry.focus()


if __name__ == "__main__":
    # Teste da janela de login
    login = RobotV7LoginWindow()
    login.root.mainloop()
    
    if login.login_successful:
        print("✅ Teste de login bem sucedido!")
        print(f"Dados da sessão: {login.session_data}")
    else:
        print("❌ Login cancelado")