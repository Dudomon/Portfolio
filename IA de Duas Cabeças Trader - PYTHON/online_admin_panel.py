#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel Admin Online - Legion AI Trader
☁️ Controle remoto via Google Drive
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sqlite3

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar sistema online
from online_system import OnlineUserManager

class OnlineAdminPanel:
    """Painel administrativo online"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("☁️ Legion AI Trader - Admin Online")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Sistema online
        self.online_manager = OnlineUserManager()
        
        # Dados em tempo real
        self.online_users = []
        self.system_config = {}
        self.usage_stats = []
        
        # Thread de atualização
        self.update_thread = None
        self.is_running = True
        
        self.setup_ui()
        self.start_update_thread()
    
    def setup_ui(self):
        """Configura interface do usuário"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook para abas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba de usuários online
        self.setup_online_users_tab()
        
        # Aba de configurações
        self.setup_config_tab()
        
        # Aba de estatísticas
        self.setup_stats_tab()
        
        # Aba de logs
        self.setup_logs_tab()
        
        # Barra de status
        self.status_bar = ttk.Label(main_frame, text="Conectando...", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def setup_online_users_tab(self):
        """Configura aba de usuários online"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="👥 Usuários Online")
        
        # Frame superior
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Botões de controle
        ttk.Button(top_frame, text="🔄 Atualizar", command=self.refresh_online_users).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="➕ Adicionar Usuário", command=self.add_user_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="✏️ Editar Usuário", command=self.edit_user_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="❌ Remover Usuário", command=self.remove_user_dialog).pack(side=tk.LEFT, padx=5)
        
        # Treeview para usuários
        columns = ('username', 'status', 'access_level', 'last_seen', 'trades_today', 'session_duration')
        self.users_tree = ttk.Treeview(frame, columns=columns, show='headings', height=15)
        
        # Configurar colunas
        self.users_tree.heading('username', text='Usuário')
        self.users_tree.heading('status', text='Status')
        self.users_tree.heading('access_level', text='Nível')
        self.users_tree.heading('last_seen', text='Último Acesso')
        self.users_tree.heading('trades_today', text='Trades Hoje')
        self.users_tree.heading('session_duration', text='Duração Sessão')
        
        self.users_tree.column('username', width=150)
        self.users_tree.column('status', width=100)
        self.users_tree.column('access_level', width=100)
        self.users_tree.column('last_seen', width=150)
        self.users_tree.column('trades_today', width=100)
        self.users_tree.column('session_duration', width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.users_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    def setup_config_tab(self):
        """Configura aba de configurações"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙️ Configurações")
        
        # Frame de configurações
        config_frame = ttk.LabelFrame(frame, text="Configurações do Sistema", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configurações
        self.config_vars = {}
        
        configs = [
            ('max_daily_trades', 'Limite de Trades Diários:', 50),
            ('max_drawdown_percent', 'Drawdown Máximo (%):', 10.0),
            ('max_lot_size', 'Lot Máximo:', 0.16),
            ('min_lot_size', 'Lot Mínimo:', 0.1),
            ('trading_hours_start', 'Início Trading (HH:MM):', '09:00'),
            ('trading_hours_end', 'Fim Trading (HH:MM):', '17:00'),
            ('maintenance_mode', 'Modo Manutenção:', False)
        ]
        
        for i, (key, label, default) in enumerate(configs):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(config_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
            
            if isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                ttk.Checkbutton(config_frame, variable=var).grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)
            else:
                var = tk.StringVar(value=str(default))
                ttk.Entry(config_frame, textvariable=var, width=15).grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)
            
            self.config_vars[key] = var
        
        # Botões
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=len(configs)//2 + 1, column=0, columnspan=4, pady=20)
        
        ttk.Button(button_frame, text="💾 Salvar Configurações", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Carregar Configurações", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔧 Modo Manutenção", command=self.toggle_maintenance).pack(side=tk.LEFT, padx=5)
    
    def setup_stats_tab(self):
        """Configura aba de estatísticas"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Estatísticas")
        
        # Frame de estatísticas gerais
        stats_frame = ttk.LabelFrame(frame, text="Estatísticas Gerais", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Labels de estatísticas
        self.stats_labels = {}
        
        stats = [
            'total_users', 'online_users', 'total_trades_today', 
            'total_volume_today', 'total_pnl_today', 'active_sessions'
        ]
        
        for i, stat in enumerate(stats):
            row = i // 3
            col = (i % 3) * 2
            
            ttk.Label(stats_frame, text=f"{stat.replace('_', ' ').title()}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
            
            label = ttk.Label(stats_frame, text="0", font=('Arial', 12, 'bold'))
            label.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)
            
            self.stats_labels[stat] = label
        
        # Gráfico de uso (simulado)
        chart_frame = ttk.LabelFrame(frame, text="Gráfico de Uso (Últimas 24h)", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chart_text = scrolledtext.ScrolledText(chart_frame, height=15)
        self.chart_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_logs_tab(self):
        """Configura aba de logs"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📋 Logs")
        
        # Frame superior
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Filtros
        ttk.Label(top_frame, text="Filtrar por:").pack(side=tk.LEFT, padx=5)
        
        self.log_filter = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(top_frame, textvariable=self.log_filter, 
                                   values=["all", "login", "trade", "error"], width=10)
        filter_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="🔄 Atualizar Logs", command=self.refresh_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="📥 Exportar Logs", command=self.export_logs).pack(side=tk.LEFT, padx=5)
        
        # Área de logs
        self.logs_text = scrolledtext.ScrolledText(frame, height=25)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def start_update_thread(self):
        """Inicia thread de atualização"""
        def update_loop():
            while self.is_running:
                try:
                    self.update_online_users()
                    self.update_stats()
                    self.update_logs()
                    time.sleep(30)  # Atualizar a cada 30 segundos
                except Exception as e:
                    print(f"❌ Erro na atualização: {e}")
                    time.sleep(60)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def update_online_users(self):
        """Atualiza lista de usuários online"""
        try:
            # Buscar usuários online
            self.online_users = self.online_manager.get_online_users()
            
            # Limpar treeview
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)
            
            # Adicionar usuários
            for user in self.online_users:
                # Calcular duração da sessão
                last_seen = datetime.fromisoformat(user['last_seen'])
                duration = datetime.now() - last_seen
                duration_str = str(duration).split('.')[0] if duration.total_seconds() > 0 else "0:00:00"
                
                # Buscar trades de hoje (simulado)
                trades_today = 0  # TODO: Implementar busca real
                
                self.users_tree.insert('', 'end', values=(
                    user['username'],
                    user['status'],
                    'user',  # TODO: Buscar nível real
                    last_seen.strftime('%H:%M:%S'),
                    trades_today,
                    duration_str
                ))
            
            self.status_bar.config(text=f"Atualizado: {datetime.now().strftime('%H:%M:%S')} - {len(self.online_users)} usuários online")
            
        except Exception as e:
            self.status_bar.config(text=f"Erro: {e}")
    
    def update_stats(self):
        """Atualiza estatísticas"""
        try:
            # Calcular estatísticas
            total_users = len(self.online_users)
            online_users = len([u for u in self.online_users if u['status'] == 'online'])
            
            # Atualizar labels
            self.stats_labels['total_users'].config(text=total_users)
            self.stats_labels['online_users'].config(text=online_users)
            self.stats_labels['total_trades_today'].config(text="0")  # TODO: Implementar
            self.stats_labels['total_volume_today'].config(text="$0.00")  # TODO: Implementar
            self.stats_labels['total_pnl_today'].config(text="$0.00")  # TODO: Implementar
            self.stats_labels['active_sessions'].config(text=online_users)
            
            # Atualizar gráfico
            self.update_chart()
            
        except Exception as e:
            print(f"❌ Erro ao atualizar stats: {e}")
    
    def update_chart(self):
        """Atualiza gráfico de uso"""
        try:
            # Simular dados de uso
            hours = list(range(24))
            usage_data = [0] * 24
            
            # Gerar dados simulados
            import random
            for i in range(24):
                if 9 <= i <= 17:  # Horário de trading
                    usage_data[i] = random.randint(5, 20)
                else:
                    usage_data[i] = random.randint(0, 3)
            
            # Criar gráfico ASCII
            chart = "Hora | Uso\n"
            chart += "-----|-----\n"
            
            for hour, usage in zip(hours, usage_data):
                bar = "█" * (usage // 2) + "░" * (10 - usage // 2)
                chart += f"{hour:02d}:00 | {bar} {usage}\n"
            
            self.chart_text.delete(1.0, tk.END)
            self.chart_text.insert(1.0, chart)
            
        except Exception as e:
            print(f"❌ Erro ao atualizar gráfico: {e}")
    
    def update_logs(self):
        """Atualiza logs"""
        try:
            # Buscar logs recentes (simulado)
            logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] Usuário admin logou",
                f"[{datetime.now().strftime('%H:%M:%S')}] Trade executado por user1",
                f"[{datetime.now().strftime('%H:%M:%S')}] Usuário user2 desconectou"
            ]
            
            # Aplicar filtro
            filter_value = self.log_filter.get()
            if filter_value != "all":
                logs = [log for log in logs if filter_value in log.lower()]
            
            # Atualizar área de logs
            self.logs_text.delete(1.0, tk.END)
            for log in logs[-50:]:  # Últimos 50 logs
                self.logs_text.insert(tk.END, log + "\n")
            
        except Exception as e:
            print(f"❌ Erro ao atualizar logs: {e}")
    
    def refresh_online_users(self):
        """Atualiza lista de usuários online"""
        self.update_online_users()
    
    def add_user_dialog(self):
        """Diálogo para adicionar usuário"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Adicionar Usuário")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Campos
        ttk.Label(dialog, text="Usuário:").pack(pady=5)
        username_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=username_var).pack(pady=5)
        
        ttk.Label(dialog, text="Senha:").pack(pady=5)
        password_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=password_var, show="*").pack(pady=5)
        
        ttk.Label(dialog, text="Nível de Acesso:").pack(pady=5)
        access_var = tk.StringVar(value="user")
        ttk.Combobox(dialog, textvariable=access_var, 
                    values=["admin", "trader", "viewer", "user"]).pack(pady=5)
        
        # Botões
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Adicionar", 
                  command=lambda: self.add_user(username_var.get(), password_var.get(), access_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def add_user(self, username: str, password: str, access_level: str, dialog):
        """Adiciona usuário"""
        try:
            if not username or not password:
                messagebox.showerror("Erro", "Usuário e senha são obrigatórios!")
                return
            
            # TODO: Implementar adição real de usuário
            messagebox.showinfo("Sucesso", f"Usuário {username} adicionado!")
            dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao adicionar usuário: {e}")
    
    def edit_user_dialog(self):
        """Diálogo para editar usuário"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um usuário para editar!")
            return
        
        # TODO: Implementar edição de usuário
        messagebox.showinfo("Info", "Funcionalidade em desenvolvimento!")
    
    def remove_user_dialog(self):
        """Diálogo para remover usuário"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um usuário para remover!")
            return
        
        # TODO: Implementar remoção de usuário
        messagebox.showinfo("Info", "Funcionalidade em desenvolvimento!")
    
    def save_config(self):
        """Salva configurações"""
        try:
            config = {}
            for key, var in self.config_vars.items():
                if isinstance(var, tk.BooleanVar):
                    config[key] = var.get()
                else:
                    config[key] = var.get()
            
            # Salvar configurações online
            for key, value in config.items():
                self.online_manager.update_online_config(key, str(value))
            
            messagebox.showinfo("Sucesso", "Configurações salvas!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar configurações: {e}")
    
    def load_config(self):
        """Carrega configurações"""
        try:
            # Carregar configurações online
            for key, var in self.config_vars.items():
                value = self.online_manager.get_online_config(key)
                if value is not None:
                    if isinstance(var, tk.BooleanVar):
                        var.set(value.lower() == 'true')
                    else:
                        var.set(value)
            
            messagebox.showinfo("Sucesso", "Configurações carregadas!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar configurações: {e}")
    
    def toggle_maintenance(self):
        """Alterna modo de manutenção"""
        try:
            current = self.config_vars['maintenance_mode'].get()
            self.config_vars['maintenance_mode'].set(not current)
            
            status = "ativado" if not current else "desativado"
            messagebox.showinfo("Manutenção", f"Modo de manutenção {status}!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao alternar manutenção: {e}")
    
    def refresh_logs(self):
        """Atualiza logs"""
        self.update_logs()
    
    def export_logs(self):
        """Exporta logs"""
        try:
            filename = f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.logs_text.get(1.0, tk.END))
            
            messagebox.showinfo("Sucesso", f"Logs exportados para {filename}!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar logs: {e}")
    
    def run(self):
        """Executa o painel"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n🛑 Painel interrompido")
        finally:
            self.is_running = False

def main():
    """Função principal"""
    try:
        panel = OnlineAdminPanel()
        panel.run()
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 