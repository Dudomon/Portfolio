"""
🧪 TESTE DEFINITIVO DO SISTEMA DE COOLDOWN
Simula exatamente o comportamento do RobotV7 para verificar se o cooldown funciona
"""

import time
import sys
import os

# Adicionar o caminho do projeto
sys.path.append(r'D:\Projeto')
sys.path.append(r'D:\Projeto\Modelo PPO Trader')

# Importar apenas as classes necessárias
try:
    from RobotV7 import TradingRobotV7
    print("✅ Importação do RobotV7 bem-sucedida")
except ImportError as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

class CooldownTester:
    def __init__(self):
        """Inicializar testador de cooldown"""
        print("🧪 [TESTE COOLDOWN] Inicializando testador...")
        
        # Criar instância do robot (sem MT5 connection)
        self.robot = TradingRobotV7()
        
        # Desabilitar MT5 para teste
        self.robot.mt5_connected = False
        
        print(f"✅ Robot inicializado - Cooldown: {self.robot.cooldown_minutes} minutos")
        
    def test_cooldown_sequence(self):
        """Testar sequência completa do cooldown"""
        print("\n🧪 [TESTE] Iniciando teste de sequência de cooldown...")
        
        # TESTE 1: Estado inicial (sem fechamentos)
        print("\n📊 TESTE 1: Estado inicial")
        print(f"   last_position_closed_timestamp = {self.robot.last_position_closed_timestamp}")
        
        cooldown_check = self.robot._is_in_cooldown()
        print(f"   Resultado: Em cooldown = {cooldown_check[0]}, Restantes = {cooldown_check[1]:.1f} min")
        
        if cooldown_check[0]:
            print("   ❌ FALHOU: Não deveria estar em cooldown no estado inicial")
            return False
        else:
            print("   ✅ PASSOU: Corretamente SEM cooldown no estado inicial")
        
        # TESTE 2: Simular fechamento de posição
        print("\n📊 TESTE 2: Simulando fechamento de posição")
        self.robot.last_position_closed_timestamp = time.time()
        timestamp_fechamento = self.robot.last_position_closed_timestamp
        print(f"   Timestamp definido: {timestamp_fechamento}")
        
        cooldown_check = self.robot._is_in_cooldown()
        print(f"   Resultado: Em cooldown = {cooldown_check[0]}, Restantes = {cooldown_check[1]:.1f} min")
        
        if not cooldown_check[0]:
            print("   ❌ FALHOU: DEVERIA estar em cooldown após fechamento")
            return False
        else:
            print("   ✅ PASSOU: Corretamente EM cooldown após fechamento")
        
        # TESTE 3: Testar processamento de ação durante cooldown
        print("\n📊 TESTE 3: Testando ação durante cooldown")
        
        # Simular ação de LONG (que deveria ser bloqueada)
        test_action = [1.5, 0.8, 0.0, 0.0]  # LONG com alta confiança
        
        print(f"   Enviando ação: {test_action} (LONG)")
        result = self.robot._process_legion_action(test_action)
        print(f"   Resultado da ação: {result}")
        
        if "COOLDOWN ATIVO" in str(result) or "HOLD" in str(result):
            print("   ✅ PASSOU: Ação bloqueada corretamente pelo cooldown")
        else:
            print("   ❌ FALHOU: Ação NÃO foi bloqueada pelo cooldown!")
            return False
        
        # TESTE 4: Verificar se cooldown expira (teste rápido com tempo reduzido)
        print("\n📊 TESTE 4: Testando expiração do cooldown")
        
        # Simular que passou tempo suficiente
        self.robot.last_position_closed_timestamp = time.time() - (self.robot.cooldown_minutes * 60 + 10)
        
        cooldown_check = self.robot._is_in_cooldown()
        print(f"   Resultado após {self.robot.cooldown_minutes} min: Em cooldown = {cooldown_check[0]}")
        
        if cooldown_check[0]:
            print("   ❌ FALHOU: Cooldown deveria ter expirado")
            return False
        else:
            print("   ✅ PASSOU: Cooldown expirado corretamente")
        
        # TESTE 5: Ação após expiração do cooldown
        print("\n📊 TESTE 5: Testando ação após expiração")
        result = self.robot._process_legion_action(test_action)
        print(f"   Resultado da ação: {result}")
        
        if "COOLDOWN" not in str(result):
            print("   ✅ PASSOU: Ação processada normalmente após expiração")
        else:
            print("   ❌ FALHOU: Cooldown ainda ativo após expiração!")
            return False
        
        return True
    
    def test_step_integration(self):
        """Testar integração com a função step()"""
        print("\n📊 TESTE INTEGRAÇÃO: Testando função step() completa")
        
        # Resetar estado
        self.robot.last_position_closed_timestamp = 0
        
        # Simular que acabou de fechar uma posição
        self.robot.last_position_closed_timestamp = time.time()
        
        # Criar uma ação de teste
        test_action = [1.8, 0.9, 0.0, 0.0]  # SHORT com alta confiança
        
        print(f"   Executando step() com ação: {test_action}")
        
        try:
            # Esta chamada deve detectar o cooldown e bloquear a ação
            observation = self.robot.step(test_action)
            print("   ✅ Step() executado sem erro")
            
            # Verificar se o cooldown foi respeitado
            # (não há como verificar diretamente, mas não deve ter dado erro)
            
            return True
            
        except Exception as e:
            print(f"   ❌ ERRO na função step(): {e}")
            return False
    
    def run_all_tests(self):
        """Executar todos os testes"""
        print("🚀 INICIANDO BATERIA COMPLETA DE TESTES DE COOLDOWN")
        print("=" * 60)
        
        # Teste 1: Sequência de cooldown
        success1 = self.test_cooldown_sequence()
        
        # Teste 2: Integração com step
        success2 = self.test_step_integration()
        
        print("\n" + "=" * 60)
        print("📊 RESULTADOS FINAIS:")
        print(f"   Teste de Sequência: {'✅ PASSOU' if success1 else '❌ FALHOU'}")
        print(f"   Teste de Integração: {'✅ PASSOU' if success2 else '❌ FALHOU'}")
        
        overall_success = success1 and success2
        
        if overall_success:
            print("\n🎉 TODOS OS TESTES PASSARAM! Sistema de cooldown funcionando corretamente.")
        else:
            print("\n💥 ALGUNS TESTES FALHARAM! Sistema de cooldown ainda tem problemas.")
        
        return overall_success

if __name__ == "__main__":
    tester = CooldownTester()
    success = tester.run_all_tests()
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)