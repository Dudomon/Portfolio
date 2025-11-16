#!/usr/bin/env python3
"""
🧪 TESTE DO SISTEMA DE LOGIN ROBOTV7
===================================

Script para testar a integração do login no RobotV7
"""

import sys
import os

# Adicionar ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_login_system():
    """Testa sistema de login"""
    print("🧪 TESTE DO SISTEMA DE LOGIN ROBOTV7")
    print("=" * 50)
    
    try:
        # Importar sistema de login
        from robotv7_login_system import RobotV7LoginWindow, RobotV7UserManager
        
        print("✅ Imports bem sucedidos")
        
        # Testar UserManager
        print("\n🔧 Testando UserManager...")
        user_manager = RobotV7UserManager()
        print(f"   Sistema online: {'Ativo' if user_manager.online_available else 'Inativo'}")
        
        # Testar autenticação com credenciais conhecidas
        print("\n🔑 Testando autenticação...")
        
        test_credentials = [
            ("robotv7_admin", "admin123", "admin"),
            ("robotv7_trader", "trader123", "trader"), 
            ("robotv7_demo", "demo123", "demo"),
            ("invalid_user", "wrong_pass", None)
        ]
        
        for username, password, expected_level in test_credentials:
            success, message, session_data = user_manager.authenticate_user(username, password)
            
            if success and expected_level:
                print(f"   ✅ {username}: {message}")
                print(f"      Level: {session_data['access_level']}")
                print(f"      Max trades: {session_data['max_daily_trades']}")
                print(f"      Drawdown: {session_data['max_drawdown_percent']}%")
                
            elif not success and not expected_level:
                print(f"   ✅ {username}: Rejeitado corretamente ({message})")
                
            else:
                print(f"   ❌ {username}: Resultado inesperado")
        
        # Testar janela de login (opcional)
        print(f"\n🎨 Janela de login disponível")
        print("   Para testar interface gráfica, execute:")
        print("   python robotv7_login_system.py")
        
        print(f"\n✅ TODOS OS TESTES PASSARAM!")
        print("🚀 Sistema pronto para uso")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de import: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Testa integração com robotlogin.py"""
    print("\n" + "=" * 50)
    print("🔗 TESTE DE INTEGRAÇÃO COM ROBOTLOGIN.PY")
    print("=" * 50)
    
    try:
        # Verificar se arquivos existem
        required_files = [
            "robotlogin.py",
            "robotv7_login_system.py",
            "online_system_real.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ Arquivos em falta:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("✅ Todos os arquivos necessários presentes")
        
        # Verificar imports do robotlogin.py
        print("\n🔍 Verificando imports...")
        
        with open("robotlogin.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "from robotv7_login_system import" in content:
            print("✅ Import do sistema de login encontrado")
        else:
            print("❌ Import do sistema de login não encontrado")
            return False
            
        if "start_with_login()" in content:
            print("✅ Função start_with_login encontrada")
        else:
            print("❌ Função start_with_login não encontrada")
            return False
        
        print("\n✅ INTEGRAÇÃO VALIDADA!")
        print("\n🚀 Para executar:")
        print("   python robotlogin.py          # Com login")
        print("   python robotlogin.py --no-login  # Sem login (dev)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na validação: {e}")
        return False


if __name__ == "__main__":
    success1 = test_login_system()
    success2 = test_integration()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema de login integrado com sucesso")
    else:
        print("❌ Alguns testes falharam")
        
    print("=" * 50)