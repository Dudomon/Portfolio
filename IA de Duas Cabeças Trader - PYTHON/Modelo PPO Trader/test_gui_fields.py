#!/usr/bin/env python3
"""
🧪 TESTE DOS CAMPOS DA GUI
==========================

Verifica se os campos da GUI estão sendo inicializados corretamente.
"""

import tkinter as tk
from tkinter import messagebox

class TestGUIFields:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧪 Teste - Campos GUI RobotV7")
        self.root.geometry("400x200")
        self.root.configure(bg='#2d2d2d')
        
        # Criar frame principal
        main_frame = tk.Frame(self.root, bg='#2d2d2d')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Label de título
        title_label = tk.Label(main_frame, 
                              text="🔍 TESTE DE CAMPOS DA GUI",
                              font=('Segoe UI', 12, 'bold'),
                              fg='#ffd93d', bg='#2d2d2d')
        title_label.pack(pady=(0, 20))
        
        # Teste do campo de lote (igual ao da GUI principal)
        lot_label = tk.Label(main_frame,
                           text="💰 Tamanho do Lote (Base):",
                           font=('Segoe UI', 10, 'bold'),
                           fg='#ffd93d', bg='#2d2d2d')
        lot_label.pack(anchor='w')
        
        # Campo de lote idêntico à GUI principal
        lot_input_frame = tk.Frame(main_frame, bg='#2d2d2d')
        lot_input_frame.pack(fill='x', pady=(5, 0))
        
        # Lot size entry (IDÊNTICO ao robotlogin.py)
        self.lot_size_var = tk.StringVar(value="0.02")
        self.lot_size_entry = tk.Entry(lot_input_frame, textvariable=self.lot_size_var,
                                     font=('Segoe UI', 10), width=8,
                                     bg='#404040', fg='#ffffff', 
                                     relief='flat', bd=2,
                                     insertbackground='#ffffff')
        self.lot_size_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botão de teste
        test_button = tk.Button(lot_input_frame, text="TESTAR",
                               command=self.test_field,
                               font=('Segoe UI', 8, 'bold'),
                               bg='#00d4ff', fg='#000000',
                               activebackground='#0099cc',
                               relief='flat', bd=0, padx=10, pady=3)
        test_button.pack(side=tk.LEFT)
        
        # Resultado
        self.result_var = tk.StringVar(value="Status: Aguardando teste...")
        result_label = tk.Label(main_frame,
                               textvariable=self.result_var,
                               font=('Segoe UI', 9),
                               fg='#ffffff', bg='#2d2d2d')
        result_label.pack(pady=(20, 0))
        
        print("🧪 Teste GUI iniciado")
        print(f"   Valor inicial do campo: '{self.lot_size_var.get()}'")
        print(f"   Campo visível: {self.lot_size_entry.winfo_exists()}")
    
    def test_field(self):
        """Testa se o campo está funcionando"""
        try:
            current_value = self.lot_size_var.get()
            print(f"🔍 Valor atual do campo: '{current_value}'")
            
            # Tentar converter para float
            float_value = float(current_value)
            
            self.result_var.set(f"✅ Campo OK: {current_value} (float: {float_value})")
            
            messagebox.showinfo("✅ Teste Bem-Sucedido", 
                               f"Campo de lote funcionando corretamente!\n"
                               f"Valor: {current_value}\n"
                               f"Float: {float_value}")
            
        except Exception as e:
            error_msg = f"❌ Erro: {e}"
            self.result_var.set(error_msg)
            print(f"❌ Erro no teste: {e}")
            
            messagebox.showerror("❌ Erro no Teste", 
                                f"Problema detectado:\n{e}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("🧪 INICIANDO TESTE DOS CAMPOS DA GUI")
    print("=" * 50)
    
    test_app = TestGUIFields()
    test_app.run()
    
    print("🧪 Teste finalizado")