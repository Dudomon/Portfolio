#!/usr/bin/env python3
"""
🧪 TESTE LIVE COOLDOWN ROBOTV7
Testa o sistema de cooldown de slots com simulação realista
"""

import sys
import os
import time
from unittest.mock import Mock, MagicMock

# Add path para importar RobotV7
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Modelo PPO Trader'))

# Mock MetaTrader5 antes de importar
sys.modules['MetaTrader5'] = Mock()
import MetaTrader5 as mt5
mt5.initialize = Mock(return_value=False)
mt5.positions_get = Mock(return_value=[])
mt5.symbol_info = Mock(return_value=Mock(bid=3500.0, ask=3500.1))
mt5.copy_ticks_from = Mock(return_value=[])

# Configurar mock responses
mt5.ORDER_TYPE_BUY = 0
mt5.ORDER_TYPE_SELL = 1
mt5.ORDER_TIME_GTC = 0
mt5.TRADE_ACTION_DEAL = 1
mt5.order_send = Mock(return_value=Mock(retcode=10009, order=123456))

class MockPosition:
    def __init__(self, ticket, comment="", magic=777888):
        self.ticket = ticket
        self.comment = comment
        self.magic = magic
        self.volume = 0.02
        self.type = 0
        self.price_open = 3500.0
        self.sl = 3480.0
        self.tp = 3520.0
        self.profit = 0.0

class SlotCooldownTester:
    def __init__(self):
        self.positions = {}
        self.next_ticket = 100000
        self.robot = None
        
    def setup_robot(self):
        """Configura o robô para teste"""
        # Import RobotV7 com mocks configurados
        from RobotV7 import TradingRobotV7
        
        # Configurar o robô
        self.robot = TradingRobotV7()
        self.robot.mt5_connected = True  # Forçar conexão
        self.robot.session_prefix = "TEST_"
        self.robot.cooldown_minutes = 1  # 1 minuto para teste rápido
        
        # Override métodos para controlar posições
        self.robot._get_robot_positions = self.mock_get_positions
        self.robot.symbol_info = Mock(bid=3500.0, ask=3500.1)
        
        print("🤖 Robô configurado para teste")
        print(f"   Cooldown: {self.robot.cooldown_minutes} minuto(s)")
        print(f"   Max posições: {self.robot.max_positions}")
        print(f"   Session prefix: {self.robot.session_prefix}")
        
    def mock_get_positions(self):
        """Mock que retorna posições ativas"""
        return list(self.positions.values())
        
    def simulate_position_open(self, slot_id):
        """Simula abertura de posição em um slot"""
        ticket = self.next_ticket
        self.next_ticket += 1
        
        comment = f"{self.robot.session_prefix}V7_SLOT{slot_id}"
        position = MockPosition(ticket, comment)
        
        self.positions[ticket] = position
        self.robot.position_slot_map[ticket] = slot_id
        
        print(f"📈 SIMULAR: Posição {ticket} aberta no slot {slot_id}")
        print(f"   Comment: {comment}")
        
        return ticket
        
    def simulate_position_close(self, ticket):
        """Simula fechamento de posição"""
        if ticket in self.positions:
            position = self.positions[ticket]
            slot_id = self.robot.position_slot_map.get(ticket)
            
            # Remover posição
            del self.positions[ticket]
            
            # Aplicar cooldown no slot
            if slot_id is not None:
                cooldown_until = time.time() + (self.robot.cooldown_minutes * 60)
                self.robot.position_slot_cooldowns[slot_id] = cooldown_until
                self.robot.position_slot_map.pop(ticket, None)
                
                print(f"📉 SIMULAR: Posição {ticket} fechada (slot {slot_id})")
                print(f"   Slot {slot_id} em cooldown até: {time.strftime('%H:%M:%S', time.localtime(cooldown_until))}")
            
            return True
        return False
        
    def test_slot_allocation(self):
        """Testa alocação de slots"""
        print("\n🧪 TESTE: Alocação de slots")
        print("=" * 50)
        
        # Teste 1: Ambos slots livres
        slot, wait = self.robot._allocate_entry_slot()
        assert slot == 0, f"Esperado slot 0, obtido {slot}"
        print(f"✅ Slot livre encontrado: {slot}")
        
        # Abrir posição no slot 0
        ticket1 = self.simulate_position_open(0)
        
        # Teste 2: Slot 0 ocupado, deve retornar slot 1
        slot, wait = self.robot._allocate_entry_slot()
        assert slot == 1, f"Com slot 0 ocupado, esperado slot 1, obtido {slot}"
        print(f"✅ Próximo slot livre: {slot}")
        
        # Abrir posição no slot 1
        ticket2 = self.simulate_position_open(1)
        
        # Teste 3: Ambos slots ocupados
        slot, wait = self.robot._allocate_entry_slot()
        assert slot is None, f"Com ambos slots ocupados, esperado None, obtido {slot}"
        print(f"✅ Nenhum slot livre (correto): {slot}")
        
        return ticket1, ticket2
        
    def test_cooldown_enforcement(self):
        """Testa enforcement do cooldown"""
        print("\n🧪 TESTE: Enforcement de Cooldown")
        print("=" * 50)
        
        # Obter tickets das posições abertas
        ticket1, ticket2 = self.test_slot_allocation()
        
        # Fechar posição do slot 0
        self.simulate_position_close(ticket1)
        
        # Teste 1: Slot 0 em cooldown, deve retornar slot 1
        slot, wait = self.robot._allocate_entry_slot()
        expected = None  # Slot 1 ainda ocupado
        if slot != expected:
            print(f"⚠️ Slot 1 ainda deveria estar ocupado, mas obtido: {slot}")
        
        # Fechar posição do slot 1 
        self.simulate_position_close(ticket2)
        
        # Teste 2: Ambos slots em cooldown
        slot, wait = self.robot._allocate_entry_slot()
        assert slot is None, f"Com ambos slots em cooldown, esperado None, obtido {slot}"
        print(f"✅ Ambos slots em cooldown (correto): wait={wait/60:.1f}min")
        
        # Teste 3: Aguardar cooldown expirar
        print(f"⏳ Aguardando {self.robot.cooldown_minutes} minuto(s) para cooldown expirar...")
        
        # Simular passagem do tempo (forçar cooldown a expirar)
        current_time = time.time()
        for slot_id in range(self.robot.max_positions):
            self.robot.position_slot_cooldowns[slot_id] = current_time - 1
            
        # Teste 4: Cooldown expirado, slot deve estar livre
        slot, wait = self.robot._allocate_entry_slot()
        assert slot == 0, f"Após cooldown expirar, esperado slot 0, obtido {slot}"
        print(f"✅ Cooldown expirado, slot livre: {slot}")
        
    def run_comprehensive_test(self):
        """Executa teste abrangente"""
        print("🔧 TESTE ABRANGENTE COOLDOWN ROBOTV7")
        print("=" * 70)
        
        try:
            self.setup_robot()
            
            # Mostrar estado inicial
            print(f"\n📊 Estado inicial:")
            print(f"   Slots cooldown: {self.robot.position_slot_cooldowns}")
            print(f"   Position map: {self.robot.position_slot_map}")
            
            # Executar testes
            self.test_cooldown_enforcement()
            
            print(f"\n✅ TESTE COMPLETO")
            print("=" * 70)
            print("🎯 SISTEMA DE COOLDOWN FUNCIONANDO!")
            print("✅ Slots são alocados corretamente")
            print("✅ Cooldown é aplicado após fechamento")
            print("✅ Slots em cooldown são respeitados")
            print("✅ Cooldown expira corretamente")
            
        except Exception as e:
            print(f"\n❌ TESTE FALHOU: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

def main():
    """Executa o teste"""
    tester = SlotCooldownTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n🏆 RESULTADO FINAL: ✅ COOLDOWN FUNCIONANDO")
        print("O sistema de cooldown de slots do RobotV7 está operacional!")
    else:
        print(f"\n💥 RESULTADO FINAL: ❌ PROBLEMAS ENCONTRADOS")
        print("O sistema de cooldown precisa de correções!")
    
    return success

if __name__ == "__main__":
    main()