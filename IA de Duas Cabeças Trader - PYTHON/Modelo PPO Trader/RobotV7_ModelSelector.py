# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import tkinter as tk
from tkinter import scrolledtext, ttk
from threading import Thread, Event
"""
⚔️ Legion AI Trader V7 - MODEL SELECTOR + VIRTUAL PORTFOLIO
🎮 SELETOR DE MODELO: Dropdown para escolher qualquer modelo
💰 PORTFOLIO VIRTUAL: $500 inicial, completamente isolado
🎯 MÁSCARA: Cada sessão opera independente com prefixo único
"""

# Importar tudo do RobotV7 original
sys.path.append('D:\\Projeto\\Modelo PPO Trader')
from RobotV7 import *
import glob

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

class ModelSelectorRobot(TradingRobotV7):
    """🎮 RobotV7 com Seletor de Modelo e Portfolio Virtual"""
    
    def __init__(self, symbol="XAUUSD", initial_balance=1000.0):
        # Inicializar com modelo padrão
        super().__init__(symbol, initial_balance)
        
        # 🎮 SELETOR DE MODELO
        self.available_models = self.scan_available_models()
        self.current_model_name = "Nenhum modelo carregado"
        
        # 💰 PORTFOLIO VIRTUAL
        self.virtual_portfolio = VirtualPortfolio(initial_balance=500.0)
        
        # 🎯 PREFIXO ÚNICO PARA ESTA SESSÃO
        self.session_id = f"SESS_{int(time.time())}"[-8:]  # Últimos 8 dígitos
        self.MODEL_PREFIX = f"MS_{self.session_id}_"
        
        # 🎮 UI VARIABLES
        self.model_var = None
        self.model_combo = None
        self.portfolio_labels = {}
        
        self._log(f"🎮 ModelSelector initialized with session ID: {self.session_id}")
        self._log(f"💰 Virtual Portfolio: ${self.virtual_portfolio.initial_balance}")
    
    def scan_available_models(self):
        """🔍 Escanear todos os modelos disponíveis"""
        models = []
        
        # 1. Pasta "Modelos para testar"
        test_path = "D:/Projeto/Modelos para testar/"
        if os.path.exists(test_path):
            for file in sorted(os.listdir(test_path)):
                if file.endswith('.zip'):
                    models.append({
                        'name': file,
                        'path': os.path.join(test_path, file),
                        'type': 'Testados'
                    })
        
        # 2. Checkpoints SILUS principais  
        silus_path = "D:/Projeto/Otimizacao/treino_principal/models/SILUS/"
        if os.path.exists(silus_path):
            # Checkpoints importantes
            key_steps = ['300000', '350000', '400000', '450000', '500000', '750000', '1000000']
            silus_files = sorted([f for f in os.listdir(silus_path) if f.endswith('.zip')])
            
            for file in silus_files:
                if any(step in file for step in key_steps):
                    # Extrair steps para nome amigável
                    for step in key_steps:
                        if step in file:
                            friendly_name = f"SILUS {int(step)//1000}K steps"
                            models.append({
                                'name': friendly_name,
                                'path': os.path.join(silus_path, file),
                                'type': 'SILUS'
                            })
                            break
        
        self._log(f"🔍 Found {len(models)} available models")
        return models
    
    def create_control_panel(self, parent):
        """🎮 Painel de controle com seletor de modelo"""
        control_frame = tk.LabelFrame(parent, text="🎮 CONTROLE DE MODELO", 
                                     font=('Segoe UI', 10, 'bold'),
                                     fg='#ffffff', bg='#2d2d2d', 
                                     highlightbackground='#444444')
        control_frame.pack(fill=tk.X, pady=(10, 0), padx=5)
        
        # 🎯 SELETOR DE MODELO
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
            self.model_combo.set(model_names[0])  # Selecionar primeiro
        
        load_btn = tk.Button(model_frame, text="🔄 Carregar", 
                            command=self.load_selected_model,
                            bg='#4CAF50', fg='white', font=('Segoe UI', 9))
        load_btn.pack(side=tk.LEFT)
        
        # 📊 STATUS DO MODELO
        status_frame = tk.Frame(control_frame, bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=10, pady=(0,5))
        
        self.model_status_label = tk.Label(status_frame, text="📊 Nenhum modelo carregado",
                                          font=('Segoe UI', 9), fg='#ff6b6b', bg='#2d2d2d')
        self.model_status_label.pack(side=tk.LEFT)
        
        # 🎮 BOTÕES DE CONTROLE
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
        """💰 Painel de estatísticas do portfolio virtual"""
        stats_frame = tk.LabelFrame(parent, text="💰 PORTFOLIO VIRTUAL", 
                                   font=('Segoe UI', 10, 'bold'),
                                   fg='#ffffff', bg='#2d2d2d',
                                   highlightbackground='#444444')
        stats_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Grid de estatísticas
        stats_grid = tk.Frame(stats_frame, bg='#2d2d2d')
        stats_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Balance atual
        balance_frame = tk.Frame(stats_grid, bg='#2d2d2d')
        balance_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=2)
        
        tk.Label(balance_frame, text="💵 Balance:", font=('Segoe UI', 9),
                fg='#ffffff', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.portfolio_labels['balance'] = tk.Label(balance_frame, text="$500.00",
                                                   font=('Segoe UI', 11, 'bold'),
                                                   fg='#00ff88', bg='#2d2d2d')
        self.portfolio_labels['balance'].pack(side=tk.RIGHT)
        
        # Session PnL
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
    
    def load_selected_model(self):
        """🔄 Carregar o modelo selecionado"""
        selected_name = self.model_var.get()
        if not selected_name:
            self._log("⚠️ Nenhum modelo selecionado")
            return
            
        # Encontrar o modelo
        selected_model = None
        for model in self.available_models:
            if model['name'] == selected_name:
                selected_model = model
                break
        
        if not selected_model:
            self._log(f"❌ Modelo não encontrado: {selected_name}")
            return
        
        self._log(f"🔄 Carregando modelo: {selected_name}")
        self._log(f"📁 Path: {selected_model['path']}")
        
        # Parar trading atual se estiver rodando
        if hasattr(self, 'trading_active') and self.trading_active:
            self.stop_trading()
        
        # Tentar carregar o modelo
        try:
            success = self.auto_load_model(selected_model['path'])
            if success:
                self.current_model_name = selected_name
                self.model_status_label.config(
                    text=f"📊 Modelo Ativo: {selected_name}",
                    fg='#00ff88'
                )
                self.start_button.config(state=tk.NORMAL)
                self._log(f"✅ Modelo carregado com sucesso: {selected_name}")
            else:
                self._log(f"❌ Falha ao carregar modelo: {selected_name}")
                self.model_status_label.config(
                    text="📊 Erro ao carregar modelo",
                    fg='#ff6b6b'
                )
        except Exception as e:
            self._log(f"❌ Erro ao carregar modelo: {e}")
            self.model_status_label.config(
                text="📊 Erro ao carregar modelo",
                fg='#ff6b6b'
            )
    
    def _get_robot_positions(self):
        """🎯 Filtrar apenas posições desta sessão (MÁSCARA)"""
        all_positions = mt5.positions_get(symbol=self.symbol)
        if not all_positions:
            return []
        
        # Filtrar apenas posições com nosso prefixo
        my_positions = []
        for pos in all_positions:
            if hasattr(pos, 'comment') and pos.comment and pos.comment.startswith(self.MODEL_PREFIX):
                my_positions.append(pos)
        
        return my_positions
    
    def _execute_v7_decision(self, action_analysis):
        """🎯 Executar decisão com prefixo único"""
        if not action_analysis or not self.mt5_connected:
            return "MT5_NOT_CONNECTED"
        
        action_name = action_analysis.get('action_name', 'UNKNOWN')
        
        # Adicionar prefixo único ao comment
        original_comment = action_analysis.get('comment', '')
        unique_comment = f"{self.MODEL_PREFIX}{original_comment}_{int(time.time())}"
        action_analysis['comment'] = unique_comment
        
        # Executar decisão original
        result = super()._execute_v7_decision(action_analysis)
        
        # Se houve trade, registrar no portfolio virtual
        if "SUCCESS" in result and "ENTRY" in action_name:
            # Estimar PnL inicial (será atualizado quando fechar)
            estimated_cost = action_analysis.get('volume', 0.02) * 1000  # Estimativa
            self._log(f"💰 Virtual Portfolio: Novo trade aberto (estimado: ${estimated_cost:.2f})")
            
        return result
    
    def _handle_position_closed(self, position, current_price):
        """💰 Registrar trade fechado no portfolio virtual"""
        try:
            # Calcular PnL real
            if position.type == 0:  # LONG
                pnl = (current_price - position.price_open) * position.volume
            else:  # SHORT
                pnl = (position.price_open - current_price) * position.volume
            
            # Registrar no portfolio virtual
            trade_type = "LONG" if position.type == 0 else "SHORT"
            self.virtual_portfolio.record_trade(pnl, trade_type)
            
            # Atualizar UI
            self.update_portfolio_display()
            
            self._log(f"💰 Trade fechado: {trade_type} | PnL: ${pnl:.2f} | Balance: ${self.virtual_portfolio.current_balance:.2f}")
            
            # Chamar função original se existir
            if hasattr(super(), '_handle_position_closed'):
                super()._handle_position_closed(position, current_price)
                
        except Exception as e:
            self._log(f"❌ Erro ao processar trade fechado: {e}")
    
    def update_portfolio_display(self):
        """📊 Atualizar display do portfolio virtual"""
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
    
    def start_trading(self):
        """▶️ Iniciar trading com modelo selecionado"""
        if not self.model_loaded:
            self._log("❌ Nenhum modelo carregado!")
            return
            
        self._log(f"▶️ Iniciando trading com {self.current_model_name}")
        self._log(f"🎯 Session ID: {self.session_id}")
        self._log(f"💰 Portfolio Virtual: ${self.virtual_portfolio.current_balance:.2f}")
        
        # Atualizar botões
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.model_combo.config(state=tk.DISABLED)
        
        # Iniciar trading
        super().start_trading()
    
    def stop_trading(self):
        """⏹️ Parar trading"""
        self._log("⏹️ Parando trading...")
        
        # Atualizar botões
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.model_combo.config(state="readonly")
        
        # Parar trading
        super().stop_trading()
        
        # Mostrar estatísticas finais
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

def main():
    """🚀 Iniciar RobotV7 com Model Selector"""
    try:
        print("🚀 Iniciando Legion AI Trader V7 - Model Selector")
        print("🎮 Sistema de seleção de modelos ativo")
        print("💰 Portfolio virtual: $500 inicial")
        
        app = ModelSelectorRobot(symbol="XAUUSD", initial_balance=1000.0)
        app.run()
        
    except KeyboardInterrupt:
        print("\\n🛑 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()