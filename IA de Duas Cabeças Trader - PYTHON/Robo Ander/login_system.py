#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Login para RobotV3 - Controle de Acesso Seguro
🔐 Autenticação, níveis de acesso e segurança para o robô de trading
"""

import tkinter as tk
from tkinter import messagebox, ttk
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
import threading
import sqlite3
from typing import Dict, Optional, Tuple

class UserManager:
    """Gerenciador de usuários com banco de dados SQLite"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
        self.create_default_admin()
    
    def init_database(self):
        """Inicializa o banco de dados SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de usuários
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                access_level TEXT NOT NULL,
                max_daily_trades INTEGER DEFAULT 50,
                max_drawdown_percent REAL DEFAULT 10.0,
                max_lot_size REAL DEFAULT 0.16,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP
            )
        ''')
        
        # Tabela de logs de acesso
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 0
            )
        ''')
        
        # Tabela de sessões ativas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_default_admin(self):
        """Cria usuário administrador padrão"""
        admin_username = "admin"
        admin_password = "admin123"  # Senha padrão - deve ser alterada
        
        if not self.user_exists(admin_username):
            self.create_user(
                username=admin_username,
                password=admin_password,
                access_level="admin",
                max_daily_trades=100,
                max_drawdown_percent=15.0,
                max_lot_size=0.5
            )
            print(f"✅ Usuário admin criado: {admin_username} / {admin_password}")
    
    def hash_password(self, password: str) -> str:
        """Cria hash seguro da senha"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, access_level: str = "user", 
                   max_daily_trades: int = 50, max_drawdown_percent: float = 10.0, 
                   max_lot_size: float = 0.16) -> bool:
        """Cria novo usuário"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, password_hash, access_level, max_daily_trades, 
                                 max_drawdown_percent, max_lot_size)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, access_level, max_daily_trades, 
                 max_drawdown_percent, max_lot_size))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False  # Usuário já existe
        except Exception as e:
            print(f"❌ Erro ao criar usuário: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "localhost") -> Tuple[bool, str, Optional[Dict]]:
        """Autentica usuário e retorna (sucesso, mensagem, dados_usuário)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar se usuário está bloqueado
            cursor.execute('''
                SELECT locked_until FROM users WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            if result and result[0]:
                locked_until = datetime.fromisoformat(result[0])
                if datetime.now() < locked_until:
                    remaining = locked_until - datetime.now()
                    return False, f"Conta bloqueada por {int(remaining.total_seconds()/60)} minutos", None
            
            # Buscar usuário
            cursor.execute('''
                SELECT password_hash, access_level, is_active, failed_attempts, 
                       max_daily_trades, max_drawdown_percent, max_lot_size
                FROM users WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            if not result:
                self.log_access(username, "login_failed", ip_address, False)
                return False, "Usuário não encontrado", None
            
            stored_hash, access_level, is_active, failed_attempts, max_daily_trades, max_drawdown_percent, max_lot_size = result
            
            if not is_active:
                self.log_access(username, "login_failed", ip_address, False)
                return False, "Conta desativada", None
            
            # Verificar senha
            if self.hash_password(password) == stored_hash:
                # Login bem-sucedido
                cursor.execute('''
                    UPDATE users SET last_login = ?, failed_attempts = 0, locked_until = NULL
                    WHERE username = ?
                ''', (datetime.now().isoformat(), username))
                
                # Criar sessão
                session_token = self.create_session(username, ip_address)
                
                user_data = {
                    "username": username,
                    "access_level": access_level,
                    "max_daily_trades": max_daily_trades,
                    "max_drawdown_percent": max_drawdown_percent,
                    "max_lot_size": max_lot_size,
                    "session_token": session_token
                }
                
                conn.commit()
                conn.close()
                
                self.log_access(username, "login_success", ip_address, True)
                return True, "Login realizado com sucesso", user_data
            else:
                # Senha incorreta
                failed_attempts += 1
                locked_until = None
                
                if failed_attempts >= 5:
                    locked_until = datetime.now() + timedelta(minutes=30)
                
                cursor.execute('''
                    UPDATE users SET failed_attempts = ?, locked_until = ?
                    WHERE username = ?
                ''', (failed_attempts, locked_until.isoformat() if locked_until else None, username))
                
                conn.commit()
                conn.close()
                
                self.log_access(username, "login_failed", ip_address, False)
                
                if locked_until:
                    return False, f"Conta bloqueada por 30 minutos após {failed_attempts} tentativas", None
                else:
                    return False, f"Senha incorreta. Tentativas restantes: {5 - failed_attempts}", None
                    
        except Exception as e:
            print(f"❌ Erro na autenticação: {e}")
            return False, "Erro interno do sistema", None
    
    def create_session(self, username: str, ip_address: str) -> str:
        """Cria nova sessão para o usuário"""
        import secrets
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=8)  # Sessão de 8 horas
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO active_sessions (username, session_token, expires_at, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (username, session_token, expires_at.isoformat(), ip_address))
        
        conn.commit()
        conn.close()
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Valida token de sessão e retorna dados do usuário"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT username, expires_at FROM active_sessions 
                WHERE session_token = ?
            ''', (session_token,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            username, expires_at = result
            expires_at = datetime.fromisoformat(expires_at)
            
            if datetime.now() > expires_at:
                # Sessão expirada
                cursor.execute('DELETE FROM active_sessions WHERE session_token = ?', (session_token,))
                conn.commit()
                conn.close()
                return None
            
            # Buscar dados do usuário
            cursor.execute('''
                SELECT access_level, max_daily_trades, max_drawdown_percent, max_lot_size
                FROM users WHERE username = ?
            ''', (username,))
            
            user_data = cursor.fetchone()
            conn.close()
            
            if user_data:
                return {
                    "username": username,
                    "access_level": user_data[0],
                    "max_daily_trades": user_data[1],
                    "max_drawdown_percent": user_data[2],
                    "max_lot_size": user_data[3]
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Erro ao validar sessão: {e}")
            return None
    
    def logout(self, session_token: str):
        """Remove sessão do usuário"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM active_sessions WHERE session_token = ?', (session_token,))
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Erro ao fazer logout: {e}")
    
    def user_exists(self, username: str) -> bool:
        """Verifica se usuário existe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def log_access(self, username: str, action: str, ip_address: str, success: bool):
        """Registra log de acesso"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO access_logs (username, action, ip_address, success)
                VALUES (?, ?, ?, ?)
            ''', (username, action, ip_address, success))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Erro ao registrar log: {e}")

class LoginWindow:
    """Interface gráfica de login"""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.current_user = None
        
        self.root = tk.Tk()
        self.setup_interface()
        
    def setup_interface(self):
        """Configura a interface de login"""
        self.root.title("🔐 Login - Legion AI Trader")
        self.root.geometry("400x500")
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(False, False)
        
        # Centralizar na tela
        self.root.eval('tk::PlaceWindow . center')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Logo/Título
        title_label = tk.Label(main_frame, text="LEGION AI TRADER", 
                              font=('Arial', 20, 'bold'), fg='#00ff88', bg='#1a1a1a')
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(main_frame, text="Sistema de Controle de Acesso", 
                                 font=('Arial', 12), fg='#ffffff', bg='#1a1a1a')
        subtitle_label.pack(pady=(0, 30))
        
        # Frame de login
        login_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        login_frame.pack(fill=tk.X, pady=10)
        
        # Usuário
        tk.Label(login_frame, text="👤 Usuário:", font=('Arial', 12, 'bold'),
                fg='#ffffff', bg='#2d2d2d').pack(pady=(20, 5))
        
        self.username_entry = tk.Entry(login_frame, font=('Arial', 12), width=25)
        self.username_entry.pack(pady=(0, 15))
        self.username_entry.focus()
        
        # Senha
        tk.Label(login_frame, text="🔒 Senha:", font=('Arial', 12, 'bold'),
                fg='#ffffff', bg='#2d2d2d').pack(pady=(0, 5))
        
        self.password_entry = tk.Entry(login_frame, font=('Arial', 12), width=25, show="*")
        self.password_entry.pack(pady=(0, 20))
        
        # Botão de login
        self.login_button = tk.Button(login_frame, text="🚀 ENTRAR", 
                                     command=self.attempt_login,
                                     bg='#00ff88', fg='black',
                                     font=('Arial', 12, 'bold'), width=20, height=2)
        self.login_button.pack(pady=(0, 20))
        
        # Status
        self.status_label = tk.Label(login_frame, text="", 
                                    font=('Arial', 10), fg='#ffaa00', bg='#2d2d2d')
        self.status_label.pack(pady=(0, 20))
        
        # Informações
        info_frame = tk.Frame(main_frame, bg='#1a1a1a')
        info_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(info_frame, text="💡 Dica: Use 'admin' / 'admin123' para primeiro acesso", 
                font=('Arial', 9), fg='#888888', bg='#1a1a1a').pack()
        
        # Bind Enter key
        self.root.bind('<Return>', lambda e: self.attempt_login())
        
    def attempt_login(self):
        """Tenta fazer login"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        if not username or not password:
            self.show_status("❌ Preencha usuário e senha", "error")
            return
        
        # Desabilitar botão durante autenticação
        self.login_button.config(state=tk.DISABLED, text="🔐 Autenticando...")
        self.root.update()
        
        # Autenticar em thread separada
        def auth_thread():
            success, message, user_data = self.user_manager.authenticate_user(username, password)
            
            # Voltar para thread principal
            self.root.after(0, lambda: self.handle_auth_result(success, message, user_data))
        
        threading.Thread(target=auth_thread, daemon=True).start()
    
    def handle_auth_result(self, success: bool, message: str, user_data: Optional[Dict]):
        """Processa resultado da autenticação"""
        self.login_button.config(state=tk.NORMAL, text="🚀 ENTRAR")
        
        if success:
            self.current_user = user_data
            self.show_status("✅ Login realizado com sucesso!", "success")
            
            # Fechar janela de login após 1 segundo
            self.root.after(1000, self.close_login)
        else:
            self.show_status(f"❌ {message}", "error")
            self.password_entry.delete(0, tk.END)
            self.password_entry.focus()
    
    def show_status(self, message: str, status_type: str = "info"):
        """Mostra mensagem de status"""
        colors = {
            "success": "#00ff88",
            "error": "#ff6666", 
            "warning": "#ffaa00",
            "info": "#ffffff"
        }
        
        self.status_label.config(text=message, fg=colors.get(status_type, "#ffffff"))
    
    def close_login(self):
        """Fecha janela de login"""
        self.root.destroy()
    
    def run(self):
        """Executa a janela de login"""
        self.root.mainloop()
        return self.current_user

class AccessControl:
    """Controle de acesso para o robô"""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.current_session = None
        self.session_token = None
    
    def require_login(self) -> Optional[Dict]:
        """Exige login e retorna dados do usuário"""
        login_window = LoginWindow()
        user_data = login_window.run()
        
        if user_data:
            self.current_session = user_data
            self.session_token = user_data.get("session_token")
            return user_data
        
        return None
    
    def validate_current_session(self) -> bool:
        """Valida sessão atual"""
        if not self.session_token:
            return False
        
        user_data = self.user_manager.validate_session(self.session_token)
        if user_data:
            self.current_session = user_data
            return True
        
        return False
    
    def logout(self):
        """Faz logout do usuário atual"""
        if self.session_token:
            self.user_manager.logout(self.session_token)
        
        self.current_session = None
        self.session_token = None
    
    def get_user_limits(self) -> Dict:
        """Retorna limites do usuário atual"""
        if not self.current_session:
            return {}
        
        return {
            "max_daily_trades": self.current_session.get("max_daily_trades", 50),
            "max_drawdown_percent": self.current_session.get("max_drawdown_percent", 10.0),
            "max_lot_size": self.current_session.get("max_lot_size", 0.16),
            "access_level": self.current_session.get("access_level", "user")
        }
    
    def can_perform_action(self, action: str) -> bool:
        """Verifica se usuário pode realizar ação"""
        if not self.current_session:
            return False
        
        access_level = self.current_session.get("access_level", "user")
        
        # Permissões por nível de acesso
        permissions = {
            "admin": ["all"],  # Admin pode tudo
            "trader": ["trade", "view", "settings"],
            "viewer": ["view"],  # Só pode visualizar
            "user": ["trade", "view"]
        }
        
        user_perms = permissions.get(access_level, [])
        return "all" in user_perms or action in user_perms

def main():
    """Função principal para teste"""
    print("🔐 SISTEMA DE LOGIN - LEGION AI TRADER")
    print("=" * 50)
    
    access_control = AccessControl()
    user_data = access_control.require_login()
    
    if user_data:
        print(f"✅ Login realizado: {user_data['username']}")
        print(f"📊 Nível de acesso: {user_data['access_level']}")
        print(f"🎯 Limites: {access_control.get_user_limits()}")
        
        # Aqui você pode iniciar o RobotV3 com os limites do usuário
        return user_data
    else:
        print("❌ Login cancelado ou falhou")
        return None

if __name__ == "__main__":
    main() 