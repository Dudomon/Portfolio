#!/usr/bin/env python3
"""
🔥 TESTE COMPLETO DO SISTEMA DE COOLDOWN - Debug Intensivo
Testa todas as funções críticas do sistema de cooldown do RobotV7
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Modelo PPO Trader"))

import time
from datetime import datetime
import MetaTrader5 as mt5

# Mock da classe TradingRobotV7 para teste isolado
class MockTradingRobot:
    def __init__(self):
        print("🧪 [MOCK] Inicializando mock do TradingRobot...")
        self.max_positions = 2
        self.magic_number = 777888
        self.cooldown_minutes = 35
        self.symbol = "XAUUSDz"
        
        # Sistema de cooldown
        self.position_slot_cooldowns = {i: 0.0 for i in range(self.max_positions)}
        self.position_slot_map = {}  # ticket -> slot
        
        # Mock de posições para teste
        self.mock_positions = []
        
        print(f"✅ [MOCK] Configurado: {self.max_positions} slots, magic {self.magic_number}, cooldown {self.cooldown_minutes}min")
    
    def _log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def _extract_slot_from_comment(self, comment):
        """Extrair slot do comentário - implementação mock"""
        try:
            if "slot" in str(comment).lower():
                import re
                match = re.search(r'slot[:\s]*(\d+)', str(comment), re.IGNORECASE)
                if match:
                    return int(match.group(1))
        except:
            pass
        return None
    
    def _allocate_entry_slot(self):
        """🔍 Função original do RobotV7 para teste"""
        try:
            import time as _t
            self._reconcile_slot_map()
            used = set(self.position_slot_map.values())
            now = _t.time()
            min_remain = None
            
            self._log(f"🔍 [SLOT-ALLOCATION] Buscando slot livre...")
            self._log(f"🔒 [SLOTS-STATUS] Slots em uso: {sorted(used) if used else 'Nenhum'}")
            
            for s in range(self.max_positions):
                allow_time = self.position_slot_cooldowns.get(s, 0.0)
                remain = max(0.0, allow_time - now)
                status = "OCUPADO" if s in used else ("LIVRE" if now >= allow_time else f"COOLDOWN({remain/60:.1f}m)")
                
                self._log(f"📍 [SLOT-{s}] Status: {status} | Próximo uso permitido: {datetime.fromtimestamp(allow_time).strftime('%H:%M:%S') if allow_time > 0 else 'Imediato'}")
                
                if s in used:
                    continue
                    
                if now >= allow_time:
                    self._log(f"✅ [SLOT-SELECTED] Slot {s} selecionado - disponível para uso")
                    return s, 0.0
                else:
                    if min_remain is None or remain < min_remain:
                        min_remain = remain
                        
            if min_remain:
                self._log(f"⏱️ [SLOT-WAIT] Nenhum slot livre. Próximo disponível em {min_remain/60:.1f} minutos")
            else:
                self._log(f"🚫 [SLOT-FULL] Todos os slots ocupados")
                
            return None, (min_remain or 0.0)
        except Exception as e:
            self._log(f"❌ [SLOT-ERROR] Erro na alocação de slot: {e}")
            return None, 0.0
    
    def _reconcile_slot_map(self):
        """Mock da reconciliação - simplificado"""
        # Para teste, manter mapeamento manual
        pass
    
    def _get_robot_positions(self):
        """Mock das posições - versão corrigida"""
        try:
            # Mock: usar lista interna
            robot_positions = self.mock_positions
            
            # 🔥 FIX CRÍTICO: Remover posições cujos slots estão em cooldown
            active_positions = []
            import time as _time
            current_time = _time.time()
            
            for pos in robot_positions:
                # Encontrar slot da posição
                slot = self.position_slot_map.get(pos['ticket'], None)
                if slot is None:
                    # Tentar extrair do comentário
                    comment = pos.get('comment', '')
                    slot = self._extract_slot_from_comment(str(comment))
                
                if slot is not None:
                    # Verificar se slot não está em cooldown
                    cooldown_until = self.position_slot_cooldowns.get(int(slot), 0.0)
                    if current_time >= cooldown_until:
                        active_positions.append(pos)
                    else:
                        # Log posição em cooldown sendo ignorada
                        remain = (cooldown_until - current_time) / 60
                        self._log(f"🔒 [COOLDOWN-IGNORED] Posição #{pos['ticket']} no slot {slot} ignorada - cooldown restante: {remain:.1f}m")
                else:
                    # Posição sem slot identificado - incluir (fallback)
                    active_positions.append(pos)
            
            return active_positions
            
        except Exception as e:
            self._log(f"[❌ ROBOT_POS] Erro ao obter posições do robô: {e}")
            return []

def test_cooldown_system():
    """🧪 Teste completo do sistema de cooldown"""
    robot = MockTradingRobot()
    
    print("\n" + "="*80)
    print("🧪 TESTE 1: Sistema limpo - sem cooldowns")
    print("="*80)
    
    # Teste 1: Sistema limpo
    slot, wait = robot._allocate_entry_slot()
    robot._log(f"✅ TESTE 1 RESULTADO: slot={slot}, wait={wait}")
    
    print("\n" + "="*80)
    print("🧪 TESTE 2: Simulando posição no slot 0")
    print("="*80)
    
    # Teste 2: Ocupar slot 0
    robot.position_slot_map[12345] = 0  # ticket 12345 -> slot 0
    robot.mock_positions.append({'ticket': 12345, 'comment': 'slot:0'})
    slot, wait = robot._allocate_entry_slot()
    robot._log(f"✅ TESTE 2 RESULTADO: slot={slot}, wait={wait}")
    
    print("\n" + "="*80)
    print("🧪 TESTE 3: Ativando cooldown no slot 0 (35min)")
    print("="*80)
    
    # Teste 3: Cooldown no slot 0
    robot.position_slot_cooldowns[0] = time.time() + (35 * 60)  # 35 minutos
    robot.mock_positions = []  # Remover posição (simulando fechamento)
    del robot.position_slot_map[12345]  # Remover mapeamento
    
    robot._log(f"🔒 [TESTE] Slot 0 em cooldown até {datetime.fromtimestamp(robot.position_slot_cooldowns[0]).strftime('%H:%M:%S')}")
    
    slot, wait = robot._allocate_entry_slot()
    robot._log(f"✅ TESTE 3 RESULTADO: slot={slot}, wait={wait/60:.1f}min")
    
    print("\n" + "="*80)
    print("🧪 TESTE 4: Ocupar slot 1 também")
    print("="*80)
    
    # Teste 4: Ocupar slot 1
    robot.position_slot_map[67890] = 1  # ticket 67890 -> slot 1
    robot.mock_positions.append({'ticket': 67890, 'comment': 'slot:1'})
    slot, wait = robot._allocate_entry_slot()
    robot._log(f"✅ TESTE 4 RESULTADO: slot={slot}, wait={wait/60:.1f}min")
    
    print("\n" + "="*80)
    print("🧪 TESTE 5: Teste _get_robot_positions com cooldown")
    print("="*80)
    
    # Teste 5: Posições ativas vs em cooldown
    robot.mock_positions = [
        {'ticket': 11111, 'comment': 'slot:0'},  # Em cooldown
        {'ticket': 22222, 'comment': 'slot:1'}   # Ativa
    ]
    robot.position_slot_map = {11111: 0, 22222: 1}
    
    active_pos = robot._get_robot_positions()
    robot._log(f"✅ TESTE 5 RESULTADO: {len(active_pos)} posições ativas (deveria ser 1)")
    for pos in active_pos:
        robot._log(f"   - Posição #{pos['ticket']} ativa")
    
    print("\n" + "="*80)
    print("🧪 TESTE CONCLUÍDO")
    print("="*80)

if __name__ == "__main__":
    test_cooldown_system()