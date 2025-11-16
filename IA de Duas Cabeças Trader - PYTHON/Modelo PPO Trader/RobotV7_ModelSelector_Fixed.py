# -*- coding: utf-8 -*-
import sys
import io
import os
import time
import glob
import pickle

# Adicionar paths necessários
project_root = "D:/Projeto"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fix stdout para Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Fix para TensorBoard/JAX conflict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import scrolledtext, ttk
from threading import Thread, Event
import warnings
warnings.filterwarnings('ignore')

"""
⚔️ Legion AI Trader V7 - MODEL SELECTOR + VIRTUAL PORTFOLIO
🎮 SELETOR DE MODELO: Dropdown para escolher qualquer modelo
💰 PORTFOLIO VIRTUAL: $500 inicial, completamente isolado
🎯 MÁSCARA: Cada sessão opera independente com prefixo único
"""

try:
    # Imports necessários com tratamento de erro
    import numpy as np
    import pandas as pd
    import torch
    
    # Stable Baselines3 imports (pode dar erro com TensorBoard)
    try:
        from stable_baselines3 import PPO
        from sb3_contrib import RecurrentPPO
        SB3_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ Stable Baselines3 import error (continuando sem SB3): {e}")
        SB3_AVAILABLE = False
    
    # MT5 import
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    except ImportError:
        print("⚠️ MetaTrader5 não disponível (modo simulação)")
        MT5_AVAILABLE = False

except ImportError as e:
    print(f"❌ Erro crítico de import: {e}")
    sys.exit(1)

class VirtualPortfolio:
    """💰 Sistema de Portfolio Virtual Isolado"""
    
    def __init__(self, initial_balance=500.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades_history = []
        self.session_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def record_trade(self, pnl, trade_type="UNKNOWN"):
        """📈 Registrar trade e atualizar métricas"""
        self.trades_history.append({
            'pnl': pnl,
            'type': trade_type,
            'timestamp': time.time(),
            'balance_after': self.current_balance + pnl
        })
        
        self.current_balance += pnl
        self.session_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            
        # Atualizar peak e drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_stats(self):
        """📊 Obter estatísticas do portfolio"""
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        return {
            'balance': self.current_balance,
            'session_pnl': self.session_pnl,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'growth': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        }

class ModelSelectorRobot:
    """🎮 Robot com Seletor de Modelo e Portfolio Virtual"""
    
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        
        # 🎮 SELETOR DE MODELO
        self.available_models = self.scan_available_models()
        self.current_model_name = "Nenhum modelo carregado"
        self.model = None
        self.model_loaded = False
        
        # 💰 PORTFOLIO VIRTUAL
        self.virtual_portfolio = VirtualPortfolio(initial_balance=500.0)
        
        # 🎯 PREFIXO ÚNICO PARA ESTA SESSÃO
        self.session_id = f"SESS_{int(time.time())}"[-8:]
        self.MODEL_PREFIX = f"MS_{self.session_id}_"
        
        # 🎮 ESTADO
        self.trading_active = False
        self.trading_thread = None
        self.stop_event = Event()
        
        # 🎮 UI VARIABLES
        self.root = None
        self.model_var = None
        self.model_combo = None
        self.portfolio_labels = {}
        self.log_widget = None
        
        # 🔌 MT5 CONNECTION
        self.mt5_connected = False
        if MT5_AVAILABLE:
            self.init_mt5()
            
        self._log(f"🎮 ModelSelector initialized with session ID: {self.session_id}")
        self._log(f"💰 Virtual Portfolio: ${self.virtual_portfolio.initial_balance}")
        self._log(f"🔍 Found {len(self.available_models)} available models")
    
    def init_mt5(self):
        """🔌 Inicializar conexão MT5"""
        try:
            if not mt5.initialize():
                self._log(f"⚠️ MT5 initialization failed: {mt5.last_error()}")
                self.mt5_connected = False
                return
                
            if not mt5.symbol_select(self.symbol, True):
                self._log(f"⚠️ Symbol {self.symbol} not available")
                self.mt5_connected = False
                return
                
            self.mt5_connected = True
            self._log(f"✅ MT5 connected - Symbol: {self.symbol}")
            
        except Exception as e:
            self._log(f"❌ MT5 connection error: {e}")
            self.mt5_connected = False
    
    def scan_available_models(self):
        """🔍 Escanear todos os modelos disponíveis"""
        models = []
        
        # 1. Pasta "Modelos para testar"
        test_path = "D:/Projeto/Modelos para testar/"
        if os.path.exists(test_path):
            for file in sorted(os.listdir(test_path)):
                if file.endswith('.zip'):
                    models.append({
                        'name': file.replace('.zip', ''),
                        'path': os.path.join(test_path, file),
                        'type': 'Testados'
                    })
        
        # 2. Checkpoints SILUS principais  
        silus_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/"
        if os.path.exists(silus_path):
            key_steps = ['300000', '350000', '400000', '450000', '500000', '750000', '1000000']
            silus_files = sorted([f for f in os.listdir(silus_path) if f.endswith('.zip')])
            
            for file in silus_files:
                if any(step in file for step in key_steps):
                    for step in key_steps:
                        if step in file:
                            friendly_name = f"SILUS {int(step)//1000}K steps"
                            models.append({
                                'name': friendly_name,
                                'path': os.path.join(silus_path, file),
                                'type': 'SILUS'
                            })
                            break
        
        return models
    
    def _log(self, message):
        """📝 Log message"""
        timestamp = time.strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        # GUI logging se disponível
        if hasattr(self, 'log_widget') and self.log_widget:
            try:
                self.log_widget.insert(tk.END, log_message + "\\n")
                self.log_widget.see(tk.END)
            except:
                pass
    
    def setup_styles(self):
        """🎨 Configurar estilos TTK"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configurar cores escuras
        self.style.configure('TLabel', background='#2d2d2d', foreground='#ffffff')
        self.style.configure('TFrame', background='#2d2d2d')
        self.style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), 
                           background='#2d2d2d', foreground='#00ff88')
    
    def create_ui(self):
        """🎮 Criar interface do usuário"""
        self.root = tk.Tk()
        self.root.title("🎮 Legion AI Trader V7 - Model Selector")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        
        # Setup styles
        self.setup_styles()
        
        # Container principal
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        self.create_header(main_container)
        
        # Content area
        content_frame = tk.Frame(main_container, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel
        left_panel = tk.Frame(content_frame, bg='#1e1e1e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        self.create_stats_panel(left_panel)
        self.create_control_panel(left_panel)
        
        # Right panel - Logs
        right_panel = tk.Frame(content_frame, bg='#1e1e1e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_log_panel(right_panel)
    
    def create_header(self, parent):
        """🎯 Criar header"""
        header_frame = tk.Frame(parent, bg='#1e1e1e', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title
        title_frame = tk.Frame(header_frame, bg='#1e1e1e')
        title_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        title = ttk.Label(title_frame, text="LEGION AI TRADER", style='Title.TLabel')
        title.pack(anchor='w')
        
        subtitle = tk.Label(title_frame, text="Model Selector V7 - Virtual Portfolio $500", 
                           font=('Segoe UI', 10), fg='#888888', bg='#1e1e1e')
        subtitle.pack(anchor='w')
        
        # Status
        status_frame = tk.Frame(header_frame, bg='#1e1e1e')
        status_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        connection_text = "● CONNECTED" if self.mt5_connected else "○ DISCONNECTED"
        connection_color = '#00ff88' if self.mt5_connected else '#ff6b6b'
        
        self.connection_status = tk.Label(status_frame, text=connection_text, 
                                         font=('Segoe UI', 10, 'bold'), 
                                         fg=connection_color, bg='#1e1e1e')
        self.connection_status.pack(anchor='e', pady=(10, 0))
    
    def create_control_panel(self, parent):
        """🎮 Painel de controle"""
        control_frame = tk.LabelFrame(parent, text="🎮 CONTROLE DE MODELO", 
                                     font=('Segoe UI', 10, 'bold'),
                                     fg='#ffffff', bg='#2d2d2d', 
                                     highlightbackground='#444444')
        control_frame.pack(fill=tk.X, pady=(10, 0), padx=5)
        
        # Seletor de modelo
        model_frame = tk.Frame(control_frame, bg='#2d2d2d')
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(model_frame, text="🤖 Modelo:", font=('Segoe UI', 9), 
                fg='#ffffff', bg='#2d2d2d').pack(side=tk.LEFT, padx=(0,10))
        
        self.model_var = tk.StringVar()
        model_names = [model['name'] for model in self.available_models]
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                       values=model_names, state="readonly", width=35)
        self.model_combo.pack(side=tk.LEFT, padx=(0,10))
        
        if model_names:
            self.model_combo.set(model_names[0])
        
        load_btn = tk.Button(model_frame, text="🔄 Carregar", 
                            command=self.load_selected_model,
                            bg='#4CAF50', fg='white', font=('Segoe UI', 9))
        load_btn.pack(side=tk.LEFT)
        
        # Status do modelo
        status_frame = tk.Frame(control_frame, bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=10, pady=(0,5))
        
        self.model_status_label = tk.Label(status_frame, text="📊 Nenhum modelo carregado",
                                          font=('Segoe UI', 9), fg='#ff6b6b', bg='#2d2d2d')
        self.model_status_label.pack(side=tk.LEFT)
        
        # Botões de controle
        buttons_frame = tk.Frame(control_frame, bg='#2d2d2d')
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_button = tk.Button(buttons_frame, text="▶️ INICIAR TRADING",
                                     command=self.start_trading, state=tk.DISABLED,
                                     bg='#28a745', fg='white', font=('Segoe UI', 10, 'bold'))
        self.start_button.pack(side=tk.LEFT, padx=(0,10))
        
        self.stop_button = tk.Button(buttons_frame, text="⏹️ PARAR TRADING",
                                    command=self.stop_trading, state=tk.DISABLED,
                                    bg='#dc3545', fg='white', font=('Segoe UI', 10, 'bold'))
        self.stop_button.pack(side=tk.LEFT)
    
    def create_stats_panel(self, parent):
        """💰 Painel de estatísticas"""
        stats_frame = tk.LabelFrame(parent, text="💰 PORTFOLIO VIRTUAL", 
                                   font=('Segoe UI', 10, 'bold'),
                                   fg='#ffffff', bg='#2d2d2d',
                                   highlightbackground='#444444')
        stats_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Grid de estatísticas
        stats_grid = tk.Frame(stats_frame, bg='#2d2d2d')
        stats_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Balance
        balance_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        balance_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=2)
        
        tk.Label(balance_frame, text="💵 Balance:", font=('Segoe UI', 9),
                fg='#ffffff', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['balance'] = tk.Label(balance_frame, text="$500.00",
                                                   font=('Segoe UI', 11, 'bold'),
                                                   fg='#00ff88', bg='#2d2d2d')
        self.portfolio_labels['balance'].pack(side=tk.RIGHT)
        
        # PnL
        pnl_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        pnl_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=2)
        
        tk.Label(pnl_frame, text="📈 Session PnL:", font=('Segoe UI', 9),
                fg='#ffffff', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['pnl'] = tk.Label(pnl_frame, text="$0.00",
                                               font=('Segoe UI', 10, 'bold'),
                                               fg='#ffffff', bg='#2d2d2d')
        self.portfolio_labels['pnl'].pack(side=tk.RIGHT)
        
        # Trades e Win Rate
        trades_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        trades_frame.grid(row=2, column=0, sticky='ew', pady=2, padx=(0,5))
        
        tk.Label(trades_frame, text="📊 Trades:", font=('Segoe UI', 8),
                fg='#cccccc', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['trades'] = tk.Label(trades_frame, text="0",
                                                  font=('Segoe UI', 9),
                                                  fg='#ffffff', bg='#2d2d2d')
        self.portfolio_labels['trades'].pack(side=tk.RIGHT)
        
        winrate_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        winrate_frame.grid(row=2, column=1, sticky='ew', pady=2, padx=(5,0))
        
        tk.Label(winrate_frame, text="🎯 Win Rate:", font=('Segoe UI', 8),
                fg='#cccccc', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['winrate'] = tk.Label(winrate_frame, text="0%",
                                                   font=('Segoe UI', 9),
                                                   fg='#ffffff', bg='#2d2d2d')
        self.portfolio_labels['winrate'].pack(side=tk.RIGHT)
        
        # Growth e Drawdown
        growth_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        growth_frame.grid(row=3, column=0, sticky='ew', pady=2, padx=(0,5))
        
        tk.Label(growth_frame, text="📊 Growth:", font=('Segoe UI', 8),
                fg='#cccccc', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['growth'] = tk.Label(growth_frame, text="0.0%",
                                                  font=('Segoe UI', 9),
                                                  fg='#ffffff', bg='#2d2d2d')
        self.portfolio_labels['growth'].pack(side=tk.RIGHT)
        
        dd_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        dd_frame.grid(row=3, column=1, sticky='ew', pady=2, padx=(5,0))
        
        tk.Label(dd_frame, text="📉 Max DD:", font=('Segoe UI', 8),
                fg='#cccccc', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['drawdown'] = tk.Label(dd_frame, text="0.0%",
                                                    font=('Segoe UI', 9),
                                                    fg='#ffffff', bg='#2d2d2d')
        self.portfolio_labels['drawdown'].pack(side=tk.RIGHT)
        
        stats_grid.columnconfigure(0, weight=1)
        stats_grid.columnconfigure(1, weight=1)
    
    def create_log_panel(self, parent):
        """📝 Painel de logs"""
        log_frame = tk.LabelFrame(parent, text="📝 LOGS DO SISTEMA", 
                                 font=('Segoe UI', 10, 'bold'),
                                 fg='#ffffff', bg='#2d2d2d',
                                 highlightbackground='#444444')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # ScrolledText para logs
        self.log_widget = scrolledtext.ScrolledText(log_frame, 
                                                   bg='#1a1a1a', fg='#ffffff',
                                                   font=('Consolas', 9),
                                                   wrap=tk.WORD)
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log inicial
        self._log("🎮 Model Selector iniciado")
        self._log(f"💰 Portfolio Virtual: $500.00")
        self._log(f"🎯 Session ID: {self.session_id}")
    
    def load_selected_model(self):
        """🔄 Carregar modelo selecionado"""
        selected_name = self.model_var.get()
        if not selected_name:
            self._log("⚠️ Nenhum modelo selecionado")
            return
        
        if not SB3_AVAILABLE:
            self._log("❌ Stable Baselines3 não disponível")
            return
            
        # Encontrar modelo
        selected_model = None
        for model in self.available_models:
            if model['name'] == selected_name:
                selected_model = model
                break
        
        if not selected_model:
            self._log(f"❌ Modelo não encontrado: {selected_name}")
            return
        
        self._log(f"🔄 Carregando modelo: {selected_name}")
        
        try:
            # Carregar modelo com diferentes estratégias
            if selected_model['path'].endswith('.zip'):
                # Tentar diferentes tipos de modelo
                model_loaded = False
                
                # Estratégia 1: RecurrentPPO (para SILUS)
                if 'SILUS' in selected_model['name']:
                    try:
                        self.model = RecurrentPPO.load(selected_model['path'])
                        model_loaded = True
                        self._log(f"✅ Carregado como RecurrentPPO: {selected_name}")
                    except Exception as e1:
                        self._log(f"⚠️ RecurrentPPO falhou: {e1}")
                
                # Estratégia 2: PPO padrão
                if not model_loaded:
                    try:
                        self.model = PPO.load(selected_model['path'])
                        model_loaded = True
                        self._log(f"✅ Carregado como PPO: {selected_name}")
                    except Exception as e2:
                        self._log(f"⚠️ PPO padrão falhou: {e2}")
                
                if not model_loaded:
                    raise Exception("Falha ao carregar com PPO ou RecurrentPPO")
            
            self.model_loaded = True
            self.current_model_name = selected_name
            
            # Atualizar UI
            self.model_status_label.config(
                text=f"📊 Modelo Ativo: {selected_name}",
                fg='#00ff88'
            )
            self.start_button.config(state=tk.NORMAL)
            
            self._log(f"✅ Modelo carregado com sucesso: {selected_name}")
            
        except Exception as e:
            self._log(f"❌ Erro ao carregar modelo {selected_name}: {e}")
            self._log(f"💡 Dica: Verifique se o modelo é compatível com PPO/RecurrentPPO")
            self.model_status_label.config(
                text="📊 Erro ao carregar modelo",
                fg='#ff6b6b'
            )
    
    def start_trading(self):
        """▶️ Iniciar trading"""
        if not self.model_loaded:
            self._log("❌ Nenhum modelo carregado!")
            return
        
        self._log(f"▶️ Iniciando trading com {self.current_model_name}")
        
        # Atualizar UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.model_combo.config(state=tk.DISABLED)
        
        # Reset stop event
        self.stop_event.clear()
        
        # Iniciar thread de trading
        self.trading_active = True
        self.trading_thread = Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
    
    def stop_trading(self):
        """⏹️ Parar trading"""
        self._log("⏹️ Parando trading...")
        
        # Sinalizar parada
        self.trading_active = False
        self.stop_event.set()
        
        # Atualizar UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.model_combo.config(state="readonly")
        
        # Mostrar estatísticas
        self.show_final_stats()
    
    def trading_loop(self):
        """🔄 Loop principal de trading"""
        self._log("🔄 Trading loop iniciado")
        
        step_count = 0
        while self.trading_active and not self.stop_event.is_set():
            try:
                step_count += 1
                
                # Simular trading decision
                if step_count % 100 == 0:  # A cada 100 steps
                    # Simular trade aleatório para demonstração
                    import random
                    pnl = random.uniform(-10, 15)  # PnL aleatório
                    trade_type = "DEMO"
                    
                    self.virtual_portfolio.record_trade(pnl, trade_type)
                    self.update_portfolio_display()
                    
                    self._log(f"📊 Step {step_count}: Demo Trade | PnL: ${pnl:.2f}")
                
                # Aguardar
                time.sleep(1)
                
                # Break se stop solicitado
                if self.stop_event.wait(timeout=0.1):
                    break
                    
            except Exception as e:
                self._log(f"❌ Erro no trading loop: {e}")
                time.sleep(5)
        
        self._log("🔄 Trading loop finalizado")
    
    def update_portfolio_display(self):
        """📊 Atualizar display do portfolio"""
        try:
            stats = self.virtual_portfolio.get_stats()
            
            # Balance
            balance_color = '#00ff88' if stats['balance'] >= 500 else '#ff6b6b'
            self.portfolio_labels['balance'].config(
                text=f"${stats['balance']:.2f}",
                fg=balance_color
            )
            
            # PnL
            pnl_color = '#00ff88' if stats['session_pnl'] >= 0 else '#ff6b6b'
            pnl_sign = '+' if stats['session_pnl'] >= 0 else ''
            self.portfolio_labels['pnl'].config(
                text=f"{pnl_sign}${stats['session_pnl']:.2f}",
                fg=pnl_color
            )
            
            # Trades
            self.portfolio_labels['trades'].config(text=str(stats['total_trades']))
            
            # Win Rate
            winrate_color = '#00ff88' if stats['win_rate'] >= 50 else '#ffaa00'
            self.portfolio_labels['winrate'].config(
                text=f"{stats['win_rate']:.1f}%",
                fg=winrate_color
            )
            
            # Growth
            growth_color = '#00ff88' if stats['growth'] >= 0 else '#ff6b6b'
            growth_sign = '+' if stats['growth'] >= 0 else ''
            self.portfolio_labels['growth'].config(
                text=f"{growth_sign}{stats['growth']:.1f}%",
                fg=growth_color
            )
            
            # Drawdown
            dd_color = '#ff6b6b' if stats['max_drawdown'] > 10 else '#ffaa00'
            self.portfolio_labels['drawdown'].config(
                text=f"{stats['max_drawdown']:.1f}%",
                fg=dd_color
            )
            
        except Exception as e:
            self._log(f"❌ Erro ao atualizar display: {e}")
    
    def show_final_stats(self):
        """📊 Mostrar estatísticas finais"""
        stats = self.virtual_portfolio.get_stats()
        self._log("=" * 50)
        self._log("📊 ESTATÍSTICAS DA SESSÃO")
        self._log("=" * 50)
        self._log(f"💰 Balance Final: ${stats['balance']:.2f}")
        self._log(f"📈 PnL Total: ${stats['session_pnl']:.2f}")
        self._log(f"📊 Total Trades: {stats['total_trades']}")
        self._log(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
        self._log(f"📊 Growth: {stats['growth']:.1f}%")
        self._log(f"📉 Max Drawdown: {stats['max_drawdown']:.1f}%")
        self._log("=" * 50)
    
    def run(self):
        """🚀 Executar aplicação"""
        self.create_ui()
        
        def on_closing():
            if self.trading_active:
                self.stop_trading()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

def main():
    """🚀 Função principal"""
    try:
        print("🚀 Iniciando Legion AI Trader V7 - Model Selector")
        
        app = ModelSelectorRobot(symbol="XAUUSD")
        app.run()
        
    except KeyboardInterrupt:
        print("\\n🛑 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()