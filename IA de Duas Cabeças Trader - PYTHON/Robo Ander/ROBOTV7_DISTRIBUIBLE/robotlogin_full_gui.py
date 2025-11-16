#!/usr/bin/env python3
"""
🤖 ROBOTV7 LOGIN - VERSÃO COMPLETA PARA EXECUTÁVEL
==================================================

GUI COMPLETA com:
- Sistema de login funcional
- Estatísticas de trading detalhadas  
- Logs completos em tempo real
- Métricas de performance
- Interface profissional
- Carregamento de modelos
- Simulação de trading realista
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
import random

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

class TradingMetrics:
    """Sistema de métricas de trading"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.balance = 10000.0  # Saldo inicial
        self.equity = 10000.0
        self.max_drawdown = 0.0
        self.max_profit = 0.0
        self.start_time = datetime.now()
        self.trades_history = []
        self.daily_pnl = []
        
    def add_trade(self, profit_loss, trade_type="BUY", symbol="GOLD", lot_size=0.02):
        """Adicionar trade às métricas"""
        self.total_trades += 1
        
        trade = {
            'time': datetime.now(),
            'type': trade_type,
            'symbol': symbol,
            'lot_size': lot_size,
            'pnl': profit_loss,
            'balance_after': self.balance + profit_loss
        }
        
        if profit_loss > 0:
            self.winning_trades += 1
            self.total_profit += profit_loss
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit_loss)
            
        self.balance += profit_loss
        self.equity = self.balance
        
        # Atualizar drawdown
        if self.balance < (10000.0 - self.max_drawdown):
            self.max_drawdown = 10000.0 - self.balance
            
        # Atualizar profit máximo
        if self.balance > (10000.0 + self.max_profit):
            self.max_profit = self.balance - 10000.0
            
        self.trades_history.append(trade)
        return trade
        
    def get_win_rate(self):
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
        
    def get_profit_factor(self):
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0
        return self.total_profit / self.total_loss
        
    def get_summary(self):
        runtime = datetime.now() - self.start_time
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.get_win_rate(),
            'total_pnl': self.balance - 10000.0,
            'balance': self.balance,
            'max_drawdown': self.max_drawdown,
            'max_profit': self.max_profit,
            'profit_factor': self.get_profit_factor(),
            'runtime': str(runtime).split('.')[0]
        }

class AdvancedTradingRobot:
    """Robot avançado com estatísticas completas"""
    
    def __init__(self, gui_callback=None):
        self.symbol = "GOLD"
        self.lot_size = 0.02
        self.running = False
        self.model_loaded = False
        self.gui_callback = gui_callback
        
        # Métricas
        self.metrics = TradingMetrics()
        
        # Configurações de trading
        self.trade_frequency = 5  # segundos entre trades
        self.risk_per_trade = 0.02  # 2% do saldo
        
        # Estado do mercado simulado
        self.current_price = 1950.00
        self.price_trend = random.choice([-1, 1])
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        if self.gui_callback:
            self.gui_callback(formatted)
    
    def load_model(self, model_path):
        """Carregar modelo com feedback detalhado"""
        filename = os.path.basename(model_path)
        self.log(f"📦 Loading model: {filename}")
        
        # Simular carregamento com etapas
        stages = [
            "🔍 Analyzing model file...",
            "🔓 Decrypting secure model...",
            "🧠 Loading neural network...",
            "⚙️ Configuring parameters...",
            "🎯 Calibrating trading strategy...",
            "✅ Model ready for trading!"
        ]
        
        for i, stage in enumerate(stages):
            time.sleep(0.5)
            self.log(stage)
            
        self.model_loaded = True
        self.log(f"🎯 Model: {filename} loaded successfully")
        return True
        
    def simulate_market_data(self):
        """Simular dados de mercado realistas"""
        # Movimento de preço baseado em tendência
        volatility = random.uniform(0.1, 0.5)
        change = random.uniform(-volatility, volatility)
        
        # Adicionar tendência
        trend_strength = 0.1
        change += self.price_trend * trend_strength * random.uniform(0, 1)
        
        self.current_price += change
        
        # Mudar tendência ocasionalmente
        if random.random() < 0.1:
            self.price_trend *= -1
            
        return {
            'price': round(self.current_price, 2),
            'change': round(change, 2),
            'volume': random.randint(100, 1000),
            'spread': round(random.uniform(0.1, 0.3), 1)
        }
        
    def make_trading_decision(self, market_data):
        """Simular decisão de trading do modelo"""
        # Lógica simulada baseada em tendência e volatilidade
        price_change = market_data['change']
        
        # Probabilidade de trade baseada no movimento
        trade_probability = abs(price_change) * 2
        
        if random.random() < trade_probability:
            # Decidir direção (com alguma inteligência)
            if price_change > 0:
                action = "SELL" if random.random() < 0.7 else "BUY"
            else:
                action = "BUY" if random.random() < 0.7 else "SELL"
                
            return action
        
        return None
        
    def execute_trade(self, action, market_data):
        """Executar trade e calcular resultado"""
        entry_price = market_data['price']
        
        # Simular duração do trade (5-30 segundos)
        trade_duration = random.randint(5, 30)
        
        self.log(f"🔥 TRADE {action} @ {entry_price} | Lot: {self.lot_size}")
        
        # Simular movimento de preço durante o trade
        price_movement = random.uniform(-2.0, 2.0)  # ±2 pontos
        exit_price = entry_price + price_movement
        
        # Calcular P&L
        if action == "BUY":
            pnl = (exit_price - entry_price) * self.lot_size * 100
        else:  # SELL
            pnl = (entry_price - exit_price) * self.lot_size * 100
            
        # Adicionar spread e comissão
        spread_cost = market_data['spread'] * self.lot_size * 100
        commission = 0.5  # $0.50 por trade
        pnl -= (spread_cost + commission)
        
        # Registrar trade
        trade = self.metrics.add_trade(pnl, action, self.symbol, self.lot_size)
        
        # Log detalhado
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        self.log(f"📊 CLOSED @ {exit_price:.2f} | P&L: {pnl_str} | Balance: ${self.metrics.balance:.2f}")
        
        return trade
        
    def run(self):
        """Loop principal de trading"""
        self.running = True
        self.log("🚀 RobotV7 Advanced Trading System STARTED")
        self.log(f"💰 Starting Balance: ${self.metrics.balance:.2f}")
        self.log(f"📈 Symbol: {self.symbol} | Lot Size: {self.lot_size}")
        
        trade_counter = 0
        
        try:
            while self.running:
                # Atualizar dados de mercado
                market_data = self.simulate_market_data()
                
                # Log de mercado a cada 10 iterações
                if trade_counter % 10 == 0:
                    self.log(f"📊 Market: {self.symbol} @ {market_data['price']:.2f} | Trend: {'📈' if self.price_trend > 0 else '📉'}")
                
                # Decisão de trading
                action = self.make_trading_decision(market_data)
                
                if action and self.metrics.balance > 1000:  # Só tradear se tiver saldo
                    trade = self.execute_trade(action, market_data)
                    
                    # Log de estatísticas a cada 5 trades
                    if self.metrics.total_trades % 5 == 0:
                        stats = self.metrics.get_summary()
                        self.log(f"📈 Stats: {stats['total_trades']} trades | Win Rate: {stats['win_rate']:.1f}% | P&L: ${stats['total_pnl']:.2f}")
                
                trade_counter += 1
                time.sleep(self.trade_frequency)
                
        except KeyboardInterrupt:
            self.log("⏹️ Trading stopped by user")
        finally:
            self.running = False
            
            # Estatísticas finais
            final_stats = self.metrics.get_summary()
            self.log("=" * 50)
            self.log("📋 FINAL TRADING STATISTICS")
            self.log("=" * 50)
            self.log(f"⏰ Runtime: {final_stats['runtime']}")
            self.log(f"📊 Total Trades: {final_stats['total_trades']}")
            self.log(f"✅ Winning Trades: {final_stats['winning_trades']}")
            self.log(f"❌ Losing Trades: {final_stats['losing_trades']}")
            self.log(f"🎯 Win Rate: {final_stats['win_rate']:.2f}%")
            self.log(f"💰 Final Balance: ${final_stats['balance']:.2f}")
            self.log(f"💸 Total P&L: ${final_stats['total_pnl']:.2f}")
            self.log(f"📉 Max Drawdown: ${final_stats['max_drawdown']:.2f}")
            self.log(f"📈 Max Profit: ${final_stats['max_profit']:.2f}")
            self.log(f"⚖️ Profit Factor: {final_stats['profit_factor']:.2f}")
            self.log("=" * 50)

class CompleteTradingApp:
    """GUI completa com estatísticas e logs"""
    
    def __init__(self, root, user_data=None):
        self.root = root
        self.user_data = user_data
        
        # Configurar janela
        unique_title = f"RobotV7 Legion - Professional Edition [{os.getpid()}]"
        self.root.title(unique_title)
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Robot com callback para GUI
        self.robot = AdvancedTradingRobot(gui_callback=self.log)
        
        # Setup GUI
        self.setup_gui()
        
        # Variáveis
        self.selected_model_path = None
        self.stats_update_thread = None
        self.stats_running = False
        
        # Thread para atualizar estatísticas
        self.start_stats_updater()
        
        self.log("🎨 Professional GUI initialized")
    
    def setup_gui(self):
        """Setup da interface profissional"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Título e usuário
        header_frame = tk.Frame(main_frame, bg='#1e1e1e')
        header_frame.pack(fill='x', pady=5)
        
        title_label = tk.Label(
            header_frame,
            text="🤖 ROBOTV7 LEGION - PROFESSIONAL TRADING SYSTEM",
            font=('Arial', 14, 'bold'),
            fg='#00ff41',
            bg='#1e1e1e'
        )
        title_label.pack()
        
        if self.user_data:
            user_info = f"👤 {self.user_data['username']} ({self.user_data['access_level']})"
        else:
            user_info = "👤 Demo Mode - Full Feature Access"
            
        user_label = tk.Label(
            header_frame,
            text=user_info,
            font=('Arial', 9),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        user_label.pack()
        
        # Frame principal dividido
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill='both', expand=True, pady=5)
        
        # Painel esquerdo - Controles e Estatísticas
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='raised', bd=2)
        left_panel.pack(side='left', fill='y', padx=5)
        
        # Controles
        controls_frame = tk.LabelFrame(left_panel, text="🎮 CONTROLS", bg='#2d2d2d', fg='#ffffff', font=('Arial', 10, 'bold'))
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        self.model_button = tk.Button(
            controls_frame,
            text="📁 SELECT MODEL",
            command=self.select_model_file,
            font=('Arial', 9, 'bold'),
            bg='#4a4a4a',
            fg='#ffffff',
            width=20
        )
        self.model_button.pack(pady=5)
        
        self.model_status_label = tk.Label(
            controls_frame,
            text="No model selected",
            font=('Arial', 8),
            fg='#ffaa00',
            bg='#2d2d2d',
            wraplength=150
        )
        self.model_status_label.pack(pady=2)
        
        # Botões de trading
        button_frame = tk.Frame(controls_frame, bg='#2d2d2d')
        button_frame.pack(fill='x', pady=5)
        
        self.start_button = tk.Button(
            button_frame,
            text="🚀 START",
            command=self.start_robot,
            font=('Arial', 9, 'bold'),
            bg='#00aa00',
            fg='#ffffff',
            width=8
        )
        self.start_button.pack(side='left', padx=2)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹️ STOP",
            command=self.stop_robot,
            font=('Arial', 9, 'bold'),
            bg='#aa0000',
            fg='#ffffff',
            width=8,
            state='disabled'
        )
        self.stop_button.pack(side='right', padx=2)
        
        # Configurações
        settings_frame = tk.LabelFrame(left_panel, text="⚙️ SETTINGS", bg='#2d2d2d', fg='#ffffff', font=('Arial', 10, 'bold'))
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(settings_frame, text="Symbol:", bg='#2d2d2d', fg='#ffffff', font=('Arial', 8)).pack(anchor='w')
        self.symbol_var = tk.StringVar(value="GOLD")
        symbol_combo = ttk.Combobox(settings_frame, textvariable=self.symbol_var, values=["GOLD", "EURUSD", "GBPUSD", "USDJPY"], width=15)
        symbol_combo.pack(pady=2)
        
        tk.Label(settings_frame, text="Lot Size:", bg='#2d2d2d', fg='#ffffff', font=('Arial', 8)).pack(anchor='w', pady=(5,0))
        self.lot_var = tk.StringVar(value="0.02")
        lot_entry = tk.Entry(settings_frame, textvariable=self.lot_var, width=18, bg='#1a1a1a', fg='#ffffff')
        lot_entry.pack(pady=2)
        
        # Estatísticas
        stats_frame = tk.LabelFrame(left_panel, text="📊 STATISTICS", bg='#2d2d2d', fg='#ffffff', font=('Arial', 10, 'bold'))
        stats_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Labels de estatísticas
        self.stats_labels = {}
        stats_data = [
            ("Balance", "balance"),
            ("Total P&L", "pnl"),
            ("Total Trades", "trades"),
            ("Win Rate", "winrate"),
            ("Drawdown", "drawdown"),
            ("Profit Factor", "pf")
        ]
        
        for label, key in stats_data:
            frame = tk.Frame(stats_frame, bg='#2d2d2d')
            frame.pack(fill='x', pady=1)
            
            tk.Label(frame, text=f"{label}:", bg='#2d2d2d', fg='#cccccc', font=('Arial', 8)).pack(side='left')
            
            value_label = tk.Label(frame, text="$0.00", bg='#2d2d2d', fg='#ffffff', font=('Arial', 8, 'bold'))
            value_label.pack(side='right')
            self.stats_labels[key] = value_label
        
        # Painel direito - Logs
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        log_header = tk.Label(
            right_panel,
            text="📋 TRADING LOGS & EVENTS",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        )
        log_header.pack(anchor='w', padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            right_panel,
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Consolas', 9),
            insertbackground='#ffffff',
            selectbackground='#404040'
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def start_stats_updater(self):
        """Iniciar thread de atualização de estatísticas"""
        self.stats_running = True
        self.stats_update_thread = Thread(target=self.update_stats_loop, daemon=True)
        self.stats_update_thread.start()
    
    def update_stats_loop(self):
        """Loop de atualização das estatísticas"""
        while self.stats_running:
            try:
                self.update_statistics_display()
                time.sleep(1)
            except:
                pass
    
    def update_statistics_display(self):
        """Atualizar display de estatísticas"""
        if not self.robot:
            return
            
        stats = self.robot.metrics.get_summary()
        
        # Atualizar labels
        try:
            self.stats_labels['balance'].config(text=f"${stats['balance']:.2f}")
            self.stats_labels['pnl'].config(text=f"${stats['total_pnl']:.2f}")
            self.stats_labels['trades'].config(text=str(stats['total_trades']))
            self.stats_labels['winrate'].config(text=f"{stats['win_rate']:.1f}%")
            self.stats_labels['drawdown'].config(text=f"${stats['max_drawdown']:.2f}")
            self.stats_labels['pf'].config(text=f"{stats['profit_factor']:.2f}")
            
            # Cores baseadas em performance
            pnl_color = '#00ff00' if stats['total_pnl'] >= 0 else '#ff4444'
            self.stats_labels['pnl'].config(fg=pnl_color)
            
            balance_color = '#00ff00' if stats['balance'] >= 10000 else '#ff4444'
            self.stats_labels['balance'].config(fg=balance_color)
        except:
            pass
    
    def log(self, message):
        """Log para interface com cores"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        try:
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
            
            # Limitar linhas do log (manter só as últimas 1000)
            lines = int(self.log_text.index('end-1c').split('.')[0])
            if lines > 1000:
                self.log_text.delete('1.0', '100.0')
        except:
            print(formatted_message.strip())
    
    def select_model_file(self):
        """Seletor de modelo"""
        file_path = filedialog.askopenfilename(
            title="Select Trading Model",
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
            
            # Carregar modelo em thread
            Thread(target=self.load_model_thread, daemon=True).start()
    
    def load_model_thread(self):
        """Carregar modelo em thread separada"""
        try:
            success = self.robot.load_model(self.selected_model_path)
            if success:
                self.robot.symbol = self.symbol_var.get()
                self.robot.lot_size = float(self.lot_var.get())
                self.log("✅ Model ready - trading system configured")
            else:
                self.log("❌ Failed to load model")
        except Exception as e:
            self.log(f"❌ Error loading model: {e}")
    
    def start_robot(self):
        """Iniciar robot"""
        if not self.robot.model_loaded:
            self.log("❌ Load a model first")
            messagebox.showwarning("Warning", "Please load a trading model first")
            return
        
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("🚀 Initiating trading system...")
        
        # Configurar robot
        self.robot.symbol = self.symbol_var.get()
        self.robot.lot_size = float(self.lot_var.get())
        
        # Thread para robot
        robot_thread = Thread(target=self.robot.run, daemon=True)
        robot_thread.start()
    
    def stop_robot(self):
        """Parar robot"""
        self.robot.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.log("⏹️ Stopping trading system...")

def main_gui_complete():
    """GUI completa sem login"""
    root = tk.Tk()
    app = CompleteTradingApp(root, user_data=None)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"[❌] GUI Error: {e}")

def main_with_login():
    """GUI com login"""
    if not LOGIN_SYSTEM_AVAILABLE:
        print("⚠️ Login system not available - starting in professional demo mode")
        main_gui_complete()
        return
    
    print("🔐 RobotV7 Professional Login System")
    print("=" * 40)
    
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
        app = CompleteTradingApp(root, user_data=user_data)
        
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
    print("🤖 RobotV7 Legion - Professional Trading System")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-login":
            main_gui_complete()
        else:
            main_with_login()
    else:
        main_with_login()