#!/usr/bin/env python3
"""
🤖 ROBOTV7 LOGIN - VERSÃO MINIMAL PARA EXECUTÁVEL
================================================

Versão ULTRA SIMPLIFICADA que funciona standalone:
- SEM trading framework pesado
- SEM torch/tensorflow  
- SÓ login + GUI básica
- FUNCIONA como .exe standalone
"""

import sys
import io
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
from threading import Thread, Event
import os
import warnings
import time
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from collections import deque
import requests
import zipfile
import tempfile

# Configurações UTF-8
try:
    if getattr(sys, 'stdout', None) is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    if getattr(sys, 'stderr', None) is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
except Exception:
    pass

warnings.filterwarnings('ignore')

# Imports locais OPCIONAIS
try:
    from robotv7_login_system import RobotV7LoginWindow, RobotV7UserManager
    LOGIN_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️ Sistema de login não disponível - modo demo")
    LOGIN_SYSTEM_AVAILABLE = False

# Variáveis globais
current_user_data = None
user_manager_instance = None

class MinimalTradingRobot:
    """Robot mínimo para executável"""
    
    def __init__(self):
        self.symbol = "GOLD"
        self.lot_size = 0.02
        self.running = False
        self.model_loaded = False
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def load_model(self, model_path):
        """Simular carregamento de modelo"""
        self.log(f"📦 Loading model: {os.path.basename(model_path)}")
        time.sleep(1)  # Simular carregamento
        self.model_loaded = True
        self.log("✅ Model loaded successfully (simulation)")
        return True
        
    def run(self):
        """Execução básica do robot"""
        self.running = True
        self.log("🚀 RobotV7 Minimal iniciado")
        
        try:
            step = 0
            while self.running and step < 100:  # Máximo 100 steps para demo
                time.sleep(0.5)
                step += 1
                if step % 10 == 0:
                    self.log(f"📊 Step {step}: Sistema funcionando...")
                    
        except KeyboardInterrupt:
            self.log("⏹️ Robot parado pelo usuário")
        finally:
            self.running = False
            self.log("🏁 Robot finalizado")

class MinimalTradingApp:
    """GUI mínima para executável"""
    
    def __init__(self, root, user_data=None):
        self.root = root
        self.user_data = user_data
        
        # Configurar janela
        unique_title = f"RobotV7 Legion - Minimal Edition [{os.getpid()}]"
        self.root.title(unique_title)
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        # Robot
        self.robot = MinimalTradingRobot()
        
        # Setup GUI
        self.setup_gui()
        
        self.selected_model_path = None
        self.log("🎨 GUI Minimal inicializada")
    
    def setup_gui(self):
        """Setup da interface"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="🤖 ROBOTV7 LEGION - MINIMAL EDITION",
            font=('Arial', 16, 'bold'),
            fg='#00ff41',
            bg='#1e1e1e'
        )
        title_label.pack(pady=10)
        
        # Info do usuário
        if self.user_data:
            user_info = f"👤 {self.user_data['username']} ({self.user_data['access_level']})"
        else:
            user_info = "👤 Demo Mode - No Login Required"
            
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
            text="📋 SYSTEM LOG",
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
        """Log para interface"""
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
            title="Select Model File",
            filetypes=[
                ("All model files", "*.secure;*.zip;*.pkl"),
                ("Secure models", "*.secure"),
                ("Model archives", "*.zip"),
                ("Pickle files", "*.pkl"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_model_path = file_path
            filename = os.path.basename(file_path)
            self.model_status_label.config(
                text=f"Selected: {filename}",
                fg='#00ff88'
            )
            self.log(f"🎯 Model selected: {filename}")
            
            # Carregar modelo
            Thread(target=self.load_model_thread, daemon=True).start()
    
    def load_model_thread(self):
        """Carregar modelo em thread separada"""
        try:
            success = self.robot.load_model(self.selected_model_path)
            if success:
                self.log("✅ Model ready for trading")
            else:
                self.log("❌ Failed to load model")
        except Exception as e:
            self.log(f"❌ Error loading model: {e}")
    
    def start_robot(self):
        """Iniciar robot"""
        if not self.robot.model_loaded:
            self.log("❌ Load a model first")
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("🚀 Starting robot...")
        
        # Thread para robot
        robot_thread = Thread(target=self.robot.run, daemon=True)
        robot_thread.start()
    
    def stop_robot(self):
        """Parar robot"""
        self.robot.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.log("⏹️ Robot stopped")

def main_gui_minimal():
    """GUI sem login"""
    root = tk.Tk()
    app = MinimalTradingApp(root, user_data=None)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"[❌] GUI Error: {e}")

def main_with_login():
    """GUI com login"""
    if not LOGIN_SYSTEM_AVAILABLE:
        print("⚠️ Login system not available - starting in demo mode")
        main_gui_minimal()
        return
    
    print("🔐 RobotV7 Login System")
    print("=" * 30)
    
    # Login
    login_window = RobotV7LoginWindow()
    login_window.root.mainloop()
    
    if login_window.login_successful and login_window.session_data:
        user_data = login_window.session_data
        print(f"✅ Login successful: {user_data['username']}")
        
        try:
            login_window.root.destroy()
        except:
            pass
        
        # GUI com usuário
        global current_user_data, user_manager_instance
        current_user_data = user_data
        user_manager_instance = RobotV7UserManager()
        
        root = tk.Tk()
        app = MinimalTradingApp(root, user_data=user_data)
        
        # Logout ao fechar
        def on_closing():
            try:
                if current_user_data and user_manager_instance:
                    user_manager_instance.logout_user(current_user_data['username'])
            except:
                pass
            finally:
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    else:
        print("❌ Login cancelled")

if __name__ == "__main__":
    print("🤖 RobotV7 Legion - Minimal Edition")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-login":
            main_gui_minimal()
        else:
            main_with_login()
    else:
        main_with_login()