#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel de Administração de Usuários - Legion AI Trader
👥 Gerenciamento completo de usuários do sistema
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import os
from datetime import datetime
from simple_login import SimpleUserManager

class UserManagerPanel:
    """Painel de gerenciamento de usuários"""
    
    def __init__(self):
        self.user_manager = SimpleUserManager()
        
        self.root = tk.Tk()
        self.setup_interface()
        self.load_users()
    
    def setup_interface(self):
        """Configura interface"""
        self.root.title("👥 Gerenciador de Usuários - Legion AI Trader")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a1a')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(main_frame, text="👥 GERENCIADOR DE USUÁRIOS", 
                              font=('Arial', 20, 'bold'), fg='#00ff88', bg='#1a1a1a')
        title_label.pack(pady=(0, 20))
        
        # Frame de botões
        buttons_frame = tk.Frame(main_frame, bg='#1a1a1a')
        buttons_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Botões de ação
        tk.Button(buttons_frame, text="➕ Criar Usuário", command=self.create_user,
                 bg='#00ff88', fg='black', font=('Arial', 12, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(buttons_frame, text="✏️ Editar Usuário", command=self.edit_user,
                 bg='#ffaa00', fg='black', font=('Arial', 12, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(buttons_frame, text="🔑 Alterar Senha", command=self.change_password,
                 bg='#ff9900', fg='black', font=('Arial', 12, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(buttons_frame, text="🗑️ Remover Usuário", command=self.remove_user,
                 bg='#ff6666', fg='white', font=('Arial', 12, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(buttons_frame, text="🔄 Atualizar", command=self.load_users,
                 bg='#666666', fg='white', font=('Arial', 12, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        # Frame da tabela
        table_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título da tabela
        tk.Label(table_frame, text="📋 LISTA DE USUÁRIOS", 
                font=('Arial', 14, 'bold'), fg='#00ff88', bg='#2d2d2d').pack(pady=10)
        
        # Treeview
        columns = ('Usuario', 'Nivel', 'Trades/Dia', 'Drawdown %', 'Lot Min', 'Lot Max', 'Criado')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configurar colunas
        self.tree.heading('Usuario', text='👤 Usuário')
        self.tree.heading('Nivel', text='🔑 Nível')
        self.tree.heading('Trades/Dia', text='📊 Trades/Dia')
        self.tree.heading('Drawdown %', text='📉 Drawdown %')
        self.tree.heading('Lot Min', text='💰 Lot Min')
        self.tree.heading('Lot Max', text='💰 Lot Máx')
        self.tree.heading('Criado', text='📅 Criado')
        
        # Largura das colunas
        self.tree.column('Usuario', width=120)
        self.tree.column('Nivel', width=80)
        self.tree.column('Trades/Dia', width=100)
        self.tree.column('Drawdown %', width=100)
        self.tree.column('Lot Min', width=80)
        self.tree.column('Lot Max', width=80)
        self.tree.column('Criado', width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Status bar
        self.status_label = tk.Label(main_frame, text="Sistema iniciado", 
                                    font=('Arial', 10), fg='#ffffff', bg='#1a1a1a')
        self.status_label.pack(fill=tk.X, pady=(10, 0))
        
        # Bind duplo clique para editar
        self.tree.bind('<Double-1>', lambda e: self.edit_user())
    
    def load_users(self):
        """Carrega usuários na tabela"""
        try:
            # Limpar tabela
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Carregar usuários do arquivo
            if os.path.exists(self.user_manager.users_file):
                with open(self.user_manager.users_file, 'r') as f:
                    users = json.load(f)
                
                for username, data in users.items():
                    # Formatar data
                    created_date = data.get('created_at', '')
                    if created_date:
                        try:
                            dt = datetime.fromisoformat(created_date)
                            created_date = dt.strftime('%d/%m/%Y %H:%M')
                        except:
                            created_date = 'Desconhecido'
                    
                    # Inserir na tabela
                    self.tree.insert('', tk.END, values=(
                        username,
                        data.get('access_level', 'user'),
                        data.get('max_daily_trades', 50),
                        f"{data.get('max_drawdown_percent', 10.0)}%",
                        data.get('base_lot_size', 0.1),
                        data.get('max_lot_size', 0.16),
                        created_date
                    ))
                
                self.status_label.config(text=f"✅ {len(users)} usuários carregados")
            else:
                self.status_label.config(text="⚠️ Arquivo de usuários não encontrado")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar usuários:\n{str(e)}")
            self.status_label.config(text=f"❌ Erro: {str(e)}")
    
    def create_user(self):
        """Cria novo usuário"""
        dialog = UserEditDialog(self.root, "Criar Usuário")
        result = dialog.show()
        
        if result:
            username, user_data = result
            
            try:
                # Verificar se usuário já existe
                if os.path.exists(self.user_manager.users_file):
                    with open(self.user_manager.users_file, 'r') as f:
                        users = json.load(f)
                else:
                    users = {}
                
                if username in users:
                    messagebox.showerror("Erro", "Usuário já existe!")
                    return
                
                # Adicionar usuário
                users[username] = user_data
                
                # Salvar
                with open(self.user_manager.users_file, 'w') as f:
                    json.dump(users, f, indent=2)
                
                messagebox.showinfo("Sucesso", f"Usuário '{username}' criado com sucesso!")
                self.load_users()
                self.status_label.config(text=f"✅ Usuário '{username}' criado")
                    
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao criar usuário:\n{str(e)}")
                self.status_label.config(text=f"❌ Erro: {str(e)}")
    
    def edit_user(self):
        """Edita usuário selecionado"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um usuário primeiro!")
            return
        
        # Obter usuário selecionado
        item = self.tree.item(selection[0])
        username = item['values'][0]
        
        try:
            # Carregar dados do usuário
            with open(self.user_manager.users_file, 'r') as f:
                users = json.load(f)
            
            if username not in users:
                messagebox.showerror("Erro", "Usuário não encontrado!")
                return
            
            user_data = users[username]
            
            # Abrir dialog de edição
            dialog = UserEditDialog(self.root, f"Editar Usuário: {username}", username, user_data)
            result = dialog.show()
            
            if result:
                new_username, new_user_data = result
                
                # Se o nome mudou, remover o antigo
                if new_username != username:
                    del users[username]
                
                # Atualizar dados
                users[new_username] = new_user_data
                
                # Salvar
                with open(self.user_manager.users_file, 'w') as f:
                    json.dump(users, f, indent=2)
                
                messagebox.showinfo("Sucesso", f"Usuário '{new_username}' atualizado com sucesso!")
                self.load_users()
                self.status_label.config(text=f"✅ Usuário '{new_username}' atualizado")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao editar usuário:\n{str(e)}")
            self.status_label.config(text=f"❌ Erro: {str(e)}")
    
    def change_password(self):
        """Altera senha do usuário"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um usuário primeiro!")
            return
        
        # Obter usuário selecionado
        item = self.tree.item(selection[0])
        username = item['values'][0]
        
        # Dialog para nova senha
        new_password = simpledialog.askstring("Alterar Senha", 
                                             f"Nova senha para '{username}':",
                                             show='*')
        
        if new_password:
            try:
                # Carregar usuários
                with open(self.user_manager.users_file, 'r') as f:
                    users = json.load(f)
                
                # Alterar senha
                users[username]['password_hash'] = self.user_manager.hash_password(new_password)
                
                # Salvar
                with open(self.user_manager.users_file, 'w') as f:
                    json.dump(users, f, indent=2)
                
                messagebox.showinfo("Sucesso", f"Senha de '{username}' alterada com sucesso!")
                self.status_label.config(text=f"✅ Senha de '{username}' alterada")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao alterar senha:\n{str(e)}")
                self.status_label.config(text=f"❌ Erro: {str(e)}")
    
    def remove_user(self):
        """Remove usuário selecionado"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um usuário primeiro!")
            return
        
        # Obter usuário selecionado
        item = self.tree.item(selection[0])
        username = item['values'][0]
        
        # Confirmar remoção
        if messagebox.askyesno("Confirmar", f"Tem certeza que deseja remover o usuário '{username}'?"):
            try:
                # Carregar usuários
                with open(self.user_manager.users_file, 'r') as f:
                    users = json.load(f)
                
                # Remover usuário
                if username in users:
                    del users[username]
                    
                    # Salvar
                    with open(self.user_manager.users_file, 'w') as f:
                        json.dump(users, f, indent=2)
                    
                    messagebox.showinfo("Sucesso", f"Usuário '{username}' removido com sucesso!")
                    self.load_users()
                    self.status_label.config(text=f"✅ Usuário '{username}' removido")
                else:
                    messagebox.showerror("Erro", "Usuário não encontrado!")
                    
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao remover usuário:\n{str(e)}")
                self.status_label.config(text=f"❌ Erro: {str(e)}")
    
    def run(self):
        """Executa o painel"""
        self.root.mainloop()

class UserEditDialog:
    """Dialog para criar/editar usuário com todos os parâmetros"""
    
    def __init__(self, parent, title, username=None, user_data=None):
        self.parent = parent
        self.title = title
        self.username = username
        self.user_data = user_data or {}
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.setup_dialog()
    
    def setup_dialog(self):
        """Configura o dialog"""
        self.dialog.title(self.title)
        self.dialog.geometry("500x700")
        self.dialog.configure(bg='#2d2d2d')
        self.dialog.resizable(False, False)
        
        # Centralizar
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Frame principal
        main_frame = tk.Frame(self.dialog, bg='#2d2d2d', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = tk.Label(main_frame, text=self.title, 
                              font=('Arial', 16, 'bold'), fg='#00ff88', bg='#2d2d2d')
        title_label.pack(pady=(0, 20))
        
        # Frame dos campos
        fields_frame = tk.Frame(main_frame, bg='#2d2d2d')
        fields_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Campo Usuário
        tk.Label(fields_frame, text="👤 Nome do Usuário:", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
        
        self.username_entry = tk.Entry(fields_frame, font=('Arial', 11), bg='#1a1a1a', fg='#ffffff',
                                      insertbackground='#ffffff', relief='flat', bd=5)
        self.username_entry.pack(fill='x', pady=(0, 15))
        if self.username:
            self.username_entry.insert(0, self.username)
        
        # Campo Senha (só para novos usuários)
        if not self.username:
            tk.Label(fields_frame, text="🔐 Senha:", 
                    font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
            
            self.password_entry = tk.Entry(fields_frame, font=('Arial', 11), bg='#1a1a1a', fg='#ffffff',
                                          insertbackground='#ffffff', show='*', relief='flat', bd=5)
            self.password_entry.pack(fill='x', pady=(0, 15))
        
        # Campo Nível de Acesso
        tk.Label(fields_frame, text="🔑 Nível de Acesso:", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
        
        self.access_level_var = tk.StringVar(value=self.user_data.get('access_level', 'user'))
        access_frame = tk.Frame(fields_frame, bg='#2d2d2d')
        access_frame.pack(fill='x', pady=(0, 15))
        
        tk.Radiobutton(access_frame, text="👤 User", variable=self.access_level_var, value='user',
                      bg='#2d2d2d', fg='#ffffff', selectcolor='#1a1a1a', 
                      font=('Arial', 10)).pack(side='left', padx=10)
        tk.Radiobutton(access_frame, text="🔧 Admin", variable=self.access_level_var, value='admin',
                      bg='#2d2d2d', fg='#ffffff', selectcolor='#1a1a1a', 
                      font=('Arial', 10)).pack(side='left', padx=10)
        tk.Radiobutton(access_frame, text="📈 Trader", variable=self.access_level_var, value='trader',
                      bg='#2d2d2d', fg='#ffffff', selectcolor='#1a1a1a', 
                      font=('Arial', 10)).pack(side='left', padx=10)
        
        # Campo Trades por Dia
        tk.Label(fields_frame, text="📊 Máximo de Trades por Dia:", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
        
        self.trades_per_day_var = tk.IntVar(value=self.user_data.get('max_daily_trades', 25))
        trades_frame = tk.Frame(fields_frame, bg='#2d2d2d')
        trades_frame.pack(fill='x', pady=(0, 15))
        
        tk.Scale(trades_frame, from_=1, to=100, orient='horizontal', variable=self.trades_per_day_var,
                bg='#1a1a1a', fg='#ffffff', troughcolor='#333333', highlightthickness=0,
                length=300).pack(side='left')
        tk.Label(trades_frame, textvariable=self.trades_per_day_var, 
                font=('Arial', 11, 'bold'), fg='#00ff88', bg='#2d2d2d', width=5).pack(side='left', padx=10)
        
        # Campo Drawdown Máximo
        tk.Label(fields_frame, text="📉 Drawdown Máximo (%):", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
        
        self.drawdown_var = tk.DoubleVar(value=self.user_data.get('max_drawdown_percent', 10.0))
        drawdown_frame = tk.Frame(fields_frame, bg='#2d2d2d')
        drawdown_frame.pack(fill='x', pady=(0, 15))
        
        tk.Scale(drawdown_frame, from_=1.0, to=50.0, resolution=0.5, orient='horizontal', 
                variable=self.drawdown_var, bg='#1a1a1a', fg='#ffffff', troughcolor='#333333', 
                highlightthickness=0, length=300).pack(side='left')
        drawdown_label = tk.Label(drawdown_frame, text="", font=('Arial', 11, 'bold'), 
                                 fg='#ff6666', bg='#2d2d2d', width=8)
        drawdown_label.pack(side='left', padx=10)
        
        def update_drawdown_label():
            drawdown_label.config(text=f"{self.drawdown_var.get():.1f}%")
        
        self.drawdown_var.trace('w', lambda *args: update_drawdown_label())
        update_drawdown_label()
        
        # Campo Lote Mínimo
        tk.Label(fields_frame, text="💰 Lote Mínimo:", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
        
        self.base_lot_var = tk.DoubleVar(value=self.user_data.get('base_lot_size', 0.1))
        base_lot_frame = tk.Frame(fields_frame, bg='#2d2d2d')
        base_lot_frame.pack(fill='x', pady=(0, 15))
        
        tk.Scale(base_lot_frame, from_=0.01, to=1.0, resolution=0.01, orient='horizontal', 
                variable=self.base_lot_var, bg='#1a1a1a', fg='#ffffff', troughcolor='#333333', 
                highlightthickness=0, length=300).pack(side='left')
        base_lot_label = tk.Label(base_lot_frame, text="", font=('Arial', 11, 'bold'), 
                                 fg='#00ff88', bg='#2d2d2d', width=8)
        base_lot_label.pack(side='left', padx=10)
        
        def update_base_lot_label():
            base_lot_label.config(text=f"{self.base_lot_var.get():.2f}")
        
        self.base_lot_var.trace('w', lambda *args: update_base_lot_label())
        update_base_lot_label()
        
        # Campo Lote Máximo
        tk.Label(fields_frame, text="💰 Lote Máximo:", 
                font=('Arial', 11, 'bold'), fg='#ffffff', bg='#2d2d2d').pack(anchor='w', pady=(0, 5))
        
        self.max_lot_var = tk.DoubleVar(value=self.user_data.get('max_lot_size', 0.3))
        max_lot_frame = tk.Frame(fields_frame, bg='#2d2d2d')
        max_lot_frame.pack(fill='x', pady=(0, 15))
        
        tk.Scale(max_lot_frame, from_=0.01, to=2.0, resolution=0.01, orient='horizontal', 
                variable=self.max_lot_var, bg='#1a1a1a', fg='#ffffff', troughcolor='#333333', 
                highlightthickness=0, length=300).pack(side='left')
        max_lot_label = tk.Label(max_lot_frame, text="", font=('Arial', 11, 'bold'), 
                                fg='#ffaa00', bg='#2d2d2d', width=8)
        max_lot_label.pack(side='left', padx=10)
        
        def update_max_lot_label():
            max_lot_label.config(text=f"{self.max_lot_var.get():.2f}")
        
        self.max_lot_var.trace('w', lambda *args: update_max_lot_label())
        update_max_lot_label()
        
        # Botões (fixos na parte inferior)
        buttons_frame = tk.Frame(main_frame, bg='#2d2d2d')
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        tk.Button(buttons_frame, text="✅ SALVAR", command=self.save_user,
                 bg='#00ff88', fg='black', font=('Arial', 14, 'bold'),
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        tk.Button(buttons_frame, text="❌ CANCELAR", command=self.cancel,
                 bg='#ff6666', fg='white', font=('Arial', 14, 'bold'),
                 width=15, height=2).pack(side=tk.RIGHT, padx=10)
    
    def save_user(self):
        """Salva o usuário"""
        username = self.username_entry.get().strip()
        
        if not username:
            messagebox.showerror("Erro", "Nome do usuário é obrigatório!")
            return
        
        # Validar lotes
        if self.base_lot_var.get() > self.max_lot_var.get():
            messagebox.showerror("Erro", "Lote mínimo não pode ser maior que o lote máximo!")
            return
        
        # Preparar dados do usuário
        user_data = {
            'access_level': self.access_level_var.get(),
            'max_daily_trades': self.trades_per_day_var.get(),
            'max_drawdown_percent': self.drawdown_var.get(),
            'base_lot_size': self.base_lot_var.get(),
            'max_lot_size': self.max_lot_var.get(),
            'created_at': self.user_data.get('created_at', datetime.now().isoformat())
        }
        
        # Se é novo usuário, adicionar hash da senha
        if not self.username and hasattr(self, 'password_entry'):
            password = self.password_entry.get()
            if not password:
                messagebox.showerror("Erro", "Senha é obrigatória!")
                return
            
            from simple_login import SimpleUserManager
            user_manager = SimpleUserManager()
            user_data['password_hash'] = user_manager.hash_password(password)
        elif self.username:
            # Manter hash da senha existente
            user_data['password_hash'] = self.user_data.get('password_hash', '')
        
        self.result = (username, user_data)
        self.dialog.destroy()
    
    def cancel(self):
        """Cancela a operação"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Mostra o dialog e retorna o resultado"""
        self.dialog.wait_window()
        return self.result

def main():
    """Função principal"""
    print("👥 GERENCIADOR DE USUÁRIOS - LEGION AI TRADER")
    print("==================================================")
    
    try:
        panel = UserManagerPanel()
        panel.run()
    except Exception as e:
        print(f"❌ Erro ao iniciar painel: {e}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main() 