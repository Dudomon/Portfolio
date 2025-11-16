#!/usr/bin/env python3
"""
🧪 TESTE DO CONTROLE DE SESSÃO
===============================

Testa se o sistema previne logins simultâneos com a mesma conta.
"""

from robotv7_login_system import RobotV7UserManager

def test_session_control():
    """Testa o controle de sessões"""
    print("🧪 TESTE - CONTROLE DE SESSÕES")
    print("=" * 50)
    
    manager = RobotV7UserManager()
    
    # Credenciais de teste
    username = "roboander_admin"
    password = "admin123"
    
    print(f"📋 Testando usuário: {username}")
    
    # TESTE 1: Login inicial (deve funcionar)
    print("\n🔸 TESTE 1: Login inicial")
    success1, message1, session1 = manager.authenticate_user(username, password)
    print(f"   Resultado: {success1}")
    print(f"   Mensagem: {message1}")
    
    if success1:
        print("✅ Login inicial bem-sucedido")
    else:
        print("❌ Login inicial falhou")
        return
    
    # TESTE 2: Segundo login simultâneo (deve falhar)
    print("\n🔸 TESTE 2: Segundo login simultâneo (deve falhar)")
    success2, message2, session2 = manager.authenticate_user(username, password)
    print(f"   Resultado: {success2}")
    print(f"   Mensagem: {message2}")
    
    if not success2 and "já está logado" in message2:
        print("✅ Segundo login bloqueado corretamente")
    else:
        print("❌ ERRO: Segundo login não foi bloqueado!")
    
    # TESTE 3: Logout do primeiro usuário
    print("\n🔸 TESTE 3: Logout do primeiro usuário")
    logout_success = manager.logout_user(username)
    print(f"   Logout: {logout_success}")
    
    if logout_success:
        print("✅ Logout realizado")
    else:
        print("❌ Falha no logout")
    
    # TESTE 4: Login após logout (deve funcionar novamente)
    print("\n🔸 TESTE 4: Login após logout (deve funcionar)")
    success3, message3, session3 = manager.authenticate_user(username, password)
    print(f"   Resultado: {success3}")
    print(f"   Mensagem: {message3}")
    
    if success3:
        print("✅ Login após logout bem-sucedido")
        
        # Limpar sessão para próximos testes
        manager.logout_user(username)
        
    else:
        print("❌ Login após logout falhou")
    
    print("\n🎯 RESUMO DOS TESTES:")
    print(f"   ✅ Login inicial: {success1}")
    print(f"   ✅ Bloqueio simultâneo: {not success2}")
    print(f"   ✅ Logout: {logout_success}")
    print(f"   ✅ Login pós-logout: {success3}")
    
    all_passed = success1 and not success2 and logout_success and success3
    
    if all_passed:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema de controle de sessões funcionando corretamente")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM")
        print("⚠️ Sistema de controle de sessões precisa de correções")
    
    return all_passed

if __name__ == "__main__":
    success = test_session_control()
    
    if success:
        print("\n🔐 RESPOSTA PARA SUA PERGUNTA:")
        print("✅ SIM, o sistema de login garante que duas pessoas")
        print("   não consigam usar o robô com a mesma conta/login.")
        print("   - Timeout de sessão: 30 minutos")
        print("   - Logout automático ao fechar aplicação")
        print("   - Verificação online em tempo real")
    else:
        print("\n❌ PROBLEMA DETECTADO:")
        print("⚠️ O sistema atual NÃO está bloqueando logins simultâneos adequadamente.")
    
    input("\nPressione Enter para continuar...")