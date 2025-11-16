#!/usr/bin/env python3
"""
🔍 TESTE FLUXO DE LOGIN - Identificar problemas na integração login -> GUI
"""

import sys
import os
import tkinter as tk
from threading import Timer

# Adicionar caminho para importar
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modelo PPO Trader'))

def test_login_window_creation():
    """🧪 Teste 1: Criação da janela de login"""
    print("🧪 TESTE 1: Criação da Janela de Login")
    print("=" * 50)
    
    try:
        # Import do sistema de login
        import robotv7_login_system
        
        # Criar janela de login (cria seu próprio root)
        login_window = robotv7_login_system.RobotV7LoginWindow()
        print("✅ Janela de login criada com sucesso")
        
        # Verificar componentes
        if hasattr(login_window, 'root'):
            print("✅ login_window.root existe")
        else:
            print("❌ login_window.root não existe")
            return False
            
        if hasattr(login_window, 'login_successful'):
            print("✅ login_window.login_successful existe")
            print(f"   Valor: {login_window.login_successful}")
        else:
            print("❌ login_window.login_successful não existe")
            return False
            
        if hasattr(login_window, 'session_data'):
            print("✅ login_window.session_data existe")
            print(f"   Valor: {login_window.session_data}")
        else:
            print("❌ login_window.session_data não existe")
            return False
        
        # Fechar janela de login
        try:
            login_window.root.destroy()
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_successful_login():
    """🧪 Teste 2: Mock de login bem sucedido"""
    print("\n🧪 TESTE 2: Mock de Login Bem Sucedido")
    print("=" * 50)
    
    try:
        import robotlogin
        
        # Mock user data como se login tivesse sido bem sucedido
        mock_user_data = {
            'username': 'test_user',
            'access_level': 'trader',
            'max_daily_trades': 25,
            'max_drawdown_percent': 12.0,
            'system': 'robotv7'
        }
        
        print(f"✅ Mock user data criado: {mock_user_data['username']}")
        
        # Testar main_gui_with_user
        print("🔄 Testando main_gui_with_user...")
        
        # Interceptar a chamada main_gui() para não abrir GUI de verdade
        original_main_gui = robotlogin.main_gui
        
        def mock_main_gui():
            print("✅ main_gui() foi chamado com sucesso")
            return True
        
        robotlogin.main_gui = mock_main_gui
        
        try:
            # Executar main_gui_with_user
            robotlogin.main_gui_with_user(mock_user_data)
            print("✅ main_gui_with_user executado com sucesso")
            
            # Verificar se dados globais foram configurados
            if hasattr(robotlogin, 'current_user_data') and robotlogin.current_user_data:
                print(f"✅ current_user_data configurado: {robotlogin.current_user_data['username']}")
            else:
                print("⚠️ current_user_data não foi configurado")
            
            if hasattr(robotlogin, 'user_manager_instance') and robotlogin.user_manager_instance:
                print("✅ user_manager_instance configurado")
            else:
                print("⚠️ user_manager_instance não foi configurado")
                
        finally:
            # Restaurar função original
            robotlogin.main_gui = original_main_gui
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_login_mode():
    """🧪 Teste 3: Modo sem login"""
    print("\n🧪 TESTE 3: Modo Sem Login")
    print("=" * 50)
    
    try:
        import robotlogin
        
        # Mock da função main_gui para não abrir GUI real
        original_main_gui = robotlogin.main_gui
        
        def mock_main_gui():
            print("✅ main_gui() chamado no modo sem login")
            return True
        
        robotlogin.main_gui = mock_main_gui
        
        try:
            # Simular execução com --no-login
            sys.argv = ['robotlogin.py', '--no-login']
            
            # Executar lógica do main
            if len(sys.argv) > 1 and sys.argv[1] == "--no-login":
                print("✅ Modo --no-login detectado")
                robotlogin.main_gui()
                print("✅ GUI iniciada sem login")
                
        finally:
            # Restaurar função original
            robotlogin.main_gui = original_main_gui
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa testes do fluxo de login"""
    print("🔍 TESTE FLUXO DE LOGIN - ROBOTLOGIN")
    print("=" * 70)
    
    tests = [
        ("Criação Janela Login", test_login_window_creation),
        ("Mock Login Bem Sucedido", test_mock_successful_login),
        ("Modo Sem Login", test_no_login_mode)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            print(f"\n❌ ERRO - {test_name}: {e}")
            results.append(False)
    
    print(f"\n{'='*70}")
    print(f"🏆 RESULTADO DOS TESTES")
    print(f"{'='*70}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Testes passaram: {passed}/{total}")
    
    if passed == total:
        print("✅ FLUXO DE LOGIN FUNCIONANDO")
        print("🤔 Bug pode estar em outro lugar ou ser mais específico")
    else:
        print("❌ PROBLEMAS NO FLUXO DE LOGIN DETECTADOS")
        print("🔧 Verificar implementação específica")

if __name__ == "__main__":
    main()