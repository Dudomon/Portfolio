#!/usr/bin/env python3
"""
🔍 DEBUG - Teste específico para identificar bugs na GUI do robotlogin
"""

import sys
import os

# Adicionar caminho para importar
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modelo PPO Trader'))

def test_basic_gui_import():
    """🧪 Teste 1: Importação básica"""
    print("🧪 TESTE 1: Importação Básica")
    print("=" * 50)
    
    try:
        # Import sem executar
        import robotlogin
        print("✅ robotlogin.py importado com sucesso")
        
        # Verificar classes principais
        if hasattr(robotlogin, 'TradingAppV7'):
            print("✅ TradingAppV7 disponível")
        else:
            print("❌ TradingAppV7 não encontrada")
            return False
            
        if hasattr(robotlogin, 'RobotV7UserManager'):
            print("✅ RobotV7UserManager disponível")
        else:
            print("❌ RobotV7UserManager não encontrada")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Falha na importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def test_gui_creation():
    """🧪 Teste 2: Criação básica da GUI"""
    print("\n🧪 TESTE 2: Criação da GUI")
    print("=" * 50)
    
    try:
        import robotlogin
        import tkinter as tk
        
        # Criar root window
        root = tk.Tk()
        root.withdraw()  # Esconder janela
        
        # Tentar criar app
        app = robotlogin.TradingAppV7(root)
        print("✅ TradingAppV7 criado com sucesso")
        
        # Verificar componentes básicos
        if hasattr(app, 'lot_size_entry'):
            print("✅ lot_size_entry existe")
            print(f"   Estado: {app.lot_size_entry['state']}")
        else:
            print("❌ lot_size_entry não encontrado")
        
        if hasattr(app, 'lot_size_var'):
            print("✅ lot_size_var existe")
            print(f"   Valor: {app.lot_size_var.get()}")
        else:
            print("❌ lot_size_var não encontrado")
            
        if hasattr(app, 'apply_lot_size'):
            print("✅ apply_lot_size método existe")
        else:
            print("❌ apply_lot_size método não encontrado")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Erro na criação da GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lot_size_functionality():
    """🧪 Teste 3: Funcionalidade do lot size"""
    print("\n🧪 TESTE 3: Funcionalidade Lot Size")
    print("=" * 50)
    
    try:
        import robotlogin
        import tkinter as tk
        
        # Criar GUI
        root = tk.Tk()
        root.withdraw()
        app = robotlogin.TradingAppV7(root)
        
        # Testar valor inicial
        initial_value = app.lot_size_var.get()
        print(f"✅ Valor inicial: {initial_value}")
        
        # Testar mudança de valor
        app.lot_size_var.set("0.05")
        new_value = app.lot_size_var.get()
        print(f"✅ Novo valor: {new_value}")
        
        # Testar apply_lot_size (sem executar)
        print("✅ Método apply_lot_size disponível")
        
        # Verificar estado do campo
        state = app.lot_size_entry['state']
        print(f"✅ Estado do campo: {state}")
        
        if state != 'normal':
            print("⚠️ PROBLEMA: Campo não está no estado 'normal'")
            return False
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa testes para identificar bugs da GUI"""
    print("🔍 DEBUG - IDENTIFICAÇÃO DE BUGS NA GUI ROBOTLOGIN")
    print("=" * 70)
    
    tests = [
        ("Importação Básica", test_basic_gui_import),
        ("Criação da GUI", test_gui_creation),
        ("Funcionalidade Lot Size", test_lot_size_functionality)
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
    print(f"🏆 RESULTADO DOS TESTES DEBUG")
    print(f"{'='*70}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Testes passaram: {passed}/{total}")
    
    if passed == total:
        print("✅ NENHUM BUG ÓBVIO DETECTADO")
        print("🤔 Problema pode ser mais sutil ou de integração")
    else:
        print("❌ BUGS DETECTADOS")
        print("🔧 Verificar implementação")

if __name__ == "__main__":
    main()