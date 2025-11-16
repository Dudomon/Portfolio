#!/usr/bin/env python3
"""
🤖 ROBOTV7 LOGIN - VERSÃO LITE PARA EXECUTÁVEL
==============================================

Versão simplificada do robotlogin.py para gerar executável mais leve.
Remove dependências pesadas que não são essenciais para o login.
"""

import sys
import io
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
from threading import Thread, Event
import os
import warnings

# Configurações UTF-8
try:
    if getattr(sys, 'stdout', None) is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    if getattr(sys, 'stderr', None) is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
except Exception:
    pass

# Suprimir avisos
warnings.filterwarnings('ignore')

# Imports básicos
import time
import numpy as np
import pandas as pd
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from collections import deque, Counter
import requests
import zipfile
import shutil
import tempfile

# 🔐 SISTEMA DE LOGIN INTEGRADO
from robotv7_login_system import RobotV7LoginWindow, RobotV7UserManager

# Enhanced Normalizer - Sistema único de normalização
try:
    from enhanced_normalizer import EnhancedRunningNormalizer, create_enhanced_normalizer
except ImportError:
    sys.path.append('..')
    sys.path.append('../Modelo PPO Trader')
    from enhanced_normalizer import EnhancedVecNormalize as EnhancedRunningNormalizer, create_enhanced_normalizer

# MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️ MetaTrader5 não disponível - modo simulação")
    MT5_AVAILABLE = False

# 🔒 SECURE MODEL IMPORTS (opcionais)
try:
    from secure_model_system import ModelSecurityManager, HardwareFingerprint
    from protect_normalizers import load_protected_normalizer
    print("[🔒 SECURITY] Secure model system loaded")
    SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"[⚠️ SECURITY] Secure model system not available: {e}")
    ModelSecurityManager = None
    SECURITY_AVAILABLE = False

# Variáveis globais para controle de sessão
current_user_data = None
user_manager_instance = None

# Configurações básicas
SL_MIN_POINTS = 2.0
SL_MAX_POINTS = 8.0
TP_MIN_POINTS = 3.0
TP_MAX_POINTS = 15.0

class TradingRobotV7Lite:
    """Versão lite do robot para executável"""
    
    def __init__(self):
        self.symbol = "GOLD"
        self.lot_size = 0.02
        self.model_loaded = False
        self.model = None
        self.normalizer = None
        self.running = False
        
        # Configurações básicas
        self.magic_number = 777888
        self.max_positions = 2
        
        # Conectar MT5 se disponível
        if MT5_AVAILABLE:
            try:
                if mt5.initialize():
                    print("✅ MetaTrader5 conectado")
                else:
                    print("❌ Falha ao conectar MetaTrader5")
            except Exception as e:
                print(f"⚠️ Erro MT5: {e}")
        
    def log(self, message):
        """Log simples"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run(self):
        """Execução básica"""
        self.running = True
        self.log("🚀 RobotV7 Lite iniciado")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("⏹️ Robot parado pelo usuário")
        finally:
            self.running = False

class TradingAppV7Lite:
    """GUI Lite para executável"""
    
    def __init__(self, root, user_data=None):
        self.root = root
        self.user_data = user_data
        
        # Configurar janela
        unique_title = f"Legion AI Trader V7 - Lite Edition [{os.getpid()}]"
        self.root.title(unique_title)
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        # Inicializar robot
        self.robot = TradingRobotV7Lite()
        
        # Setup GUI
        self.setup_gui()
        
        # Variáveis para modelo
        self.selected_model_path = None
        
        self.log("🎨 GUI Lite inicializada")
    
    def setup_gui(self):
        """Setup básico da GUI"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="🤖 ROBOTV7 LEGION - LITE EDITION",
            font=('Arial', 16, 'bold'),
            fg='#00ff41',
            bg='#1e1e1e'
        )
        title_label.pack(pady=10)
        
        # Status do usuário
        if self.user_data:
            user_info = f"👤 {self.user_data['username']} ({self.user_data['access_level']})"
            user_label = tk.Label(
                main_frame,
                text=user_info,
                font=('Arial', 10),
                fg='#ffffff',
                bg='#1e1e1e'
            )
            user_label.pack(pady=5)
        
        # Frame de controles
        controls_frame = tk.Frame(main_frame, bg='#2d2d2d', relief='raised', bd=2)
        controls_frame.pack(fill='x', pady=10)
        
        # Botão select model
        self.model_button = tk.Button(
            controls_frame,
            text="SELECT MODEL",
            command=self.select_model_file,
            font=('Arial', 10, 'bold'),
            bg='#4a4a4a',
            fg='#ffffff',
            width=15
        )
        self.model_button.pack(side='left', padx=10, pady=10)
        
        # Status do modelo
        self.model_status_label = tk.Label(
            controls_frame,
            text="No model selected",
            font=('Arial', 9),
            fg='#ffaa00',
            bg='#2d2d2d'
        )
        self.model_status_label.pack(side='left', padx=10)
        
        # Botões de controle
        button_frame = tk.Frame(main_frame, bg='#1e1e1e')
        button_frame.pack(fill='x', pady=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="START ROBOT",
            command=self.start_robot,
            font=('Arial', 10, 'bold'),
            bg='#00aa00',
            fg='#ffffff',
            width=12
        )
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="STOP ROBOT",
            command=self.stop_robot,
            font=('Arial', 10, 'bold'),
            bg='#aa0000',
            fg='#ffffff',
            width=12,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # Log area
        log_frame = tk.Frame(main_frame, bg='#2d2d2d', relief='raised', bd=2)
        log_frame.pack(fill='both', expand=True, pady=10)
        
        tk.Label(
            log_frame,
            text="📋 LOG",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        ).pack(anchor='w', padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Consolas', 9),
            insertbackground='#ffffff'
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def log(self, message):
        """Adicionar mensagem ao log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        try:
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
        except:
            print(formatted_message.strip())
    
    def select_model_file(self):
        """Seletor de modelo"""
        file_path = filedialog.askopenfilename(
            title="Select Trading Model",
            filetypes=[
                ("Protected models", "*.secure"),
                ("Model files", "*.zip"),
                ("All files", "*.*")
            ],
            initialdir="D:/Projeto/Robo Ander/Modelo Ander"
        )
        
        if file_path:
            self.selected_model_path = file_path
            filename = os.path.basename(file_path)
            self.model_status_label.config(
                text=f"Selected: {filename}",
                fg='#00ff88'
            )
            self.log(f"🎯 Model selected: {filename}")
            
            # Tentar carregar modelo
            self.load_selected_model()
    
    def load_selected_model(self):
        """Carregar modelo selecionado"""
        if not self.selected_model_path:
            self.log("❌ No model path selected")
            return False
        
        filename = os.path.basename(self.selected_model_path)
        self.log(f"🔄 Loading model: {filename}")
        
        # Verificar se é arquivo .secure
        if filename.endswith('.secure'):
            if SECURITY_AVAILABLE:
                self.log("🔒 Loading protected model...")
                self.robot.model_loaded = True
                self.log("✅ Protected model loaded (simulation)")
            else:
                self.log("❌ Security system not available")
                return False
        else:
            # Modelo normal
            self.log("📦 Loading standard model...")
            self.robot.model_loaded = True
            self.log("✅ Model loaded (simulation)")
        
        return True
    
    def start_robot(self):
        """Iniciar robot"""
        if not self.robot.model_loaded:
            self.log("❌ Load a model first")
            messagebox.showerror("Error", "Please load a model first")
            return
        
        self.robot.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("🚀 Robot started")
        
        # Thread para executar robot
        robot_thread = Thread(target=self.robot.run, daemon=True)
        robot_thread.start()
    
    def stop_robot(self):
        """Parar robot"""
        self.robot.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.log("⏹️ Robot stopped")

def main_gui():
    """Função principal para iniciar GUI"""
    global current_user_data, user_manager_instance
    
    root = tk.Tk()
    app = TradingAppV7Lite(root, user_data=current_user_data)
    
    # Configurar logout ao fechar
    if current_user_data and user_manager_instance:
        def on_closing():
            try:
                print(f"🔓 Realizando logout de {current_user_data['username']}...")
                user_manager_instance.logout_user(current_user_data['username'])
                print("✅ Logout realizado com sucesso")
            except Exception as e:
                print(f"⚠️ Erro no logout: {e}")
            finally:
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"[❌] Erro na GUI: {e}")

def start_with_login():
    """Iniciar sistema com login"""
    print("🔐 RobotV7 Login - Sistema de Autenticação")
    print("=" * 50)
    
    # Criar janela de login
    login_window = RobotV7LoginWindow()
    login_window.root.mainloop()
    
    # Verificar se login foi bem sucedido
    if login_window.login_successful and login_window.session_data:
        user_data = login_window.session_data
        print(f"✅ Login bem sucedido: {user_data['username']}")
        
        # Fechar janela de login
        try:
            login_window.root.destroy()
            print("🔐 Janela de login fechada")
        except Exception as e:
            print(f"⚠️ Erro ao fechar login: {e}")
        
        # Iniciar GUI principal
        main_gui_with_user(user_data)
    else:
        print("❌ Login cancelado ou falhou")
        sys.exit(0)

def main_gui_with_user(user_data):
    """Iniciar GUI com dados do usuário"""
    print(f"👤 Usuário ativo: {user_data['username']} ({user_data['access_level']})")
    
    # Armazenar dados globalmente
    global current_user_data, user_manager_instance
    current_user_data = user_data
    user_manager_instance = RobotV7UserManager()
    
    # Iniciar GUI
    main_gui()

if __name__ == "__main__":
    print("🤖 RobotV7 Lite - Executável")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-login":
            print("🎨 Iniciando sem login...")
            main_gui()
        elif sys.argv[1] == "--console":
            print("🚀 Modo console...")
            robot = TradingRobotV7Lite()
            robot.run()
        else:
            start_with_login()
    else:
        start_with_login()