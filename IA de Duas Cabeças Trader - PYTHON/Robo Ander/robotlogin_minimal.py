#!/usr/bin/env python3
"""
🤖 ROBOTV7 LOGIN - VERSÃO MINIMAL PARA EXECUTÁVEL
===============================================

Versão ultra-simplificada apenas com o login essencial.
"""

import tkinter as tk
from tkinter import messagebox
import os
import sys
import subprocess

# Adicionar caminho atual
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Variável global para sistema de login
LOGIN_SUCCESS = False
USER_DATA = None

def show_splash():
    """Mostra splash screen"""
    splash = tk.Tk()
    splash.title("RobotV7 Legion")
    splash.geometry("400x200")
    splash.configure(bg='#0f0f0f')
    splash.resizable(False, False)
    
    # Centralizar
    splash.eval('tk::PlaceWindow . center')
    
    # Logo/Título
    title_label = tk.Label(
        splash,
        text="🤖 ROBOTV7 LEGION",
        font=('Arial', 20, 'bold'),
        fg='#00ff41',
        bg='#0f0f0f'
    )
    title_label.pack(pady=30)
    
    subtitle_label = tk.Label(
        splash,
        text="Professional Trading System",
        font=('Arial', 12),
        fg='#ffffff',
        bg='#0f0f0f'
    )
    subtitle_label.pack(pady=10)
    
    # Botão de login
    login_button = tk.Button(
        splash,
        text="🔐 LOGIN",
        font=('Arial', 12, 'bold'),
        bg='#4a4a4a',
        fg='#ffffff',
        width=15,
        height=2,
        command=lambda: launch_login_and_close(splash)
    )
    login_button.pack(pady=20)
    
    # Botão modo demo
    demo_button = tk.Button(
        splash,
        text="🎮 DEMO MODE",
        font=('Arial', 10),
        bg='#666666',
        fg='#ffffff',
        width=15,
        command=lambda: launch_demo_and_close(splash)
    )
    demo_button.pack(pady=5)
    
    splash.mainloop()

def launch_login_and_close(splash):
    """Lança login completo e fecha splash"""
    splash.destroy()
    
    # Tentar executar robotlogin completo
    try:
        # Verificar se arquivo existe
        robotlogin_path = os.path.join(os.path.dirname(__file__), "robotlogin.py")
        
        if os.path.exists(robotlogin_path):
            print("🚀 Lançando RobotLogin completo...")
            # Executar robotlogin.py
            subprocess.run([sys.executable, robotlogin_path])
        else:
            messagebox.showerror(
                "Erro", 
                "Arquivo robotlogin.py não encontrado!\n\n"
                "Certifique-se de que todos os arquivos estão na mesma pasta."
            )
            
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar RobotLogin:\n{e}")

def launch_demo_and_close(splash):
    """Lança modo demo e fecha splash"""
    splash.destroy()
    
    try:
        robotlogin_path = os.path.join(os.path.dirname(__file__), "robotlogin.py")
        
        if os.path.exists(robotlogin_path):
            print("🎮 Lançando modo demo...")
            # Executar com flag --no-login
            subprocess.run([sys.executable, robotlogin_path, "--no-login"])
        else:
            messagebox.showerror(
                "Erro", 
                "Arquivo robotlogin.py não encontrado!"
            )
            
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar modo demo:\n{e}")

def main():
    """Função principal - mostra splash"""
    print("🤖 RobotV7 Legion - Starting...")
    show_splash()

if __name__ == "__main__":
    main()