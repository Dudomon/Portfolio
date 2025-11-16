#!/usr/bin/env python3
"""
🔥 TESTE COMPLETO E RIGOROSO DO SISTEMA DE COOLDOWN ROBOTV7
Simula cenário real com mapeamento, abertura, fechamento e cooldown
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Modelo PPO Trader"))

import time
from datetime import datetime
import re

class MockMT5Position:
    def __init__(self, ticket, magic, comment, symbol="XAUUSDz"):
        self.ticket = ticket
        self.magic = magic  
        self.comment = comment
        self.symbol = symbol

class MockMT5Deal:
    def __init__(self, position_id, profit, comment):
        self.position_id = position_id
        self.profit = profit
        self.comment = comment

class MockTradingRobotV7:
    def __init__(self):
        print("🧪 [MOCK] Inicializando TradingRobotV7 completo...")
        self.max_positions = 2
        self.magic_number = 777888
        self.cooldown_minutes = 35
        self.symbol = "XAUUSDz"
        self.mt5_connected = True
        
        # Sistema de cooldown
        self.position_slot_cooldowns = {i: 0.0 for i in range(self.max_positions)}
        self.position_slot_map = {}  # ticket -> slot
        self.position_stats = {}
        
        # Mock de dados MT5
        self.mock_positions = []
        self.mock_deals = []
        
        print(f"✅ [MOCK] Configurado: {self.max_positions} slots, magic {self.magic_number}, cooldown {self.cooldown_minutes}min")
    
    def _log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def _extract_slot_from_comment(self, comment):
        """Extrair slot do comentário"""
        try:
            import re
            m = re.search(r"SLOT(\d+)", str(comment))
            if m:
                return int(m.group(1))
            m = re.search(r"V7S(\d+)", str(comment))
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None
    
    def _get_robot_positions(self):
        """Mock das posições - versão CORRIGIDA"""
        try:
            # Usar mock de posições do robô
            robot_positions = [pos for pos in self.mock_positions if pos.magic == self.magic_number]
            
            # 🔥 FIX: Remover posições cujos slots estão em cooldown
            active_positions = []
            import time as _time
            current_time = _time.time()
            
            for pos in robot_positions:
                # Encontrar slot da posição
                slot = self.position_slot_map.get(pos.ticket, None)
                if slot is None:
                    # Tentar extrair do comentário
                    comment = getattr(pos, 'comment', '')
                    slot = self._extract_slot_from_comment(str(comment))
                
                if slot is not None:
                    # Verificar se slot não está em cooldown
                    cooldown_until = self.position_slot_cooldowns.get(int(slot), 0.0)
                    if current_time >= cooldown_until:
                        active_positions.append(pos)
                    else:
                        # Log posição em cooldown sendo ignorada
                        remain = (cooldown_until - current_time) / 60
                        self._log(f"🔒 [COOLDOWN-IGNORED] Posição #{pos.ticket} no slot {slot} ignorada - cooldown restante: {remain:.1f}m")
                else:
                    # Posição sem slot identificado - incluir (fallback)
                    active_positions.append(pos)
            
            return active_positions
            
        except Exception as e:
            self._log(f"[❌ ROBOT_POS] Erro ao obter posições do robô: {e}")
            return []
    
    def _allocate_entry_slot(self):
        """Função original do RobotV7 para teste"""
        try:
            import time as _t
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
    
    def simulate_open_position(self, ticket, slot_id):
        """Simular abertura de posição"""
        comment = f"V7_SLOT{slot_id}"
        pos = MockMT5Position(ticket, self.magic_number, comment)
        self.mock_positions.append(pos)
        
        # Mapear imediatamente (simula o fix)
        self.position_slot_map[ticket] = slot_id
        self._log(f"🔗 [SLOT-MAP] Ticket #{ticket} → Slot {slot_id} (MAPEADO IMEDIATAMENTE)")
        self.position_stats[ticket] = {'open_price': 3630.0, 'volume': 0.02, 'type': 'LONG'}
    
    def simulate_close_position(self, ticket, profit):
        """Simular fechamento de posição com debug completo"""
        self._log(f"🔒 [COOLDOWN ATIVADO] Posição #{ticket} fechada - Cooldown de {self.cooldown_minutes} minutos iniciado")
        self._log(f"📊 [POSIÇÃO FECHADA] Ticket #{ticket} | P&L: ${profit:.2f}")
        
        # Simular deal de fechamento
        pos = next((p for p in self.mock_positions if p.ticket == ticket), None)
        if pos:
            close_deal = MockMT5Deal(ticket, profit, pos.comment)
            
            # 🔥 FIX: Processo completo de identificação de slot
            try:
                slot = self.position_slot_map.get(ticket, None)
                self._log(f"🔍 [CLOSE-DEBUG] Ticket #{ticket} - Slot no mapa: {slot}")
                
                if slot is None:
                    # Tentar extrair do comentário do deal/posição
                    cmt = getattr(close_deal, 'comment', '') or ''
                    slot = self._extract_slot_from_comment(str(cmt))
                    self._log(f"🔍 [CLOSE-DEBUG] Ticket #{ticket} - Comment: '{cmt}' → Slot extraído: {slot}")
                    
                if slot is None:
                    # ÚLTIMO RECURSO: Buscar nas posições
                    for pos in self.mock_positions:
                        if pos.magic == self.magic_number and pos.ticket == ticket:
                            pos_cmt = getattr(pos, 'comment', '') or ''
                            slot = self._extract_slot_from_comment(str(pos_cmt))
                            self._log(f"🔍 [CLOSE-DEBUG] Ticket #{ticket} - Posição Comment: '{pos_cmt}' → Slot: {slot}")
                            break
                            
                if slot is not None:
                    cooldown_until = time.time() + (self.cooldown_minutes * 60)
                    self.position_slot_cooldowns[int(slot)] = cooldown_until
                    # Remover mapeamento do ticket
                    self.position_slot_map.pop(ticket, None)
                    cooldown_until_str = datetime.fromtimestamp(cooldown_until).strftime('%H:%M:%S')
                    self._log(f"🔒 [COOLDOWN-SLOT] Slot {int(slot)} em cooldown por {self.cooldown_minutes} min até {cooldown_until_str}")
                    self._log(f"📊 [COOLDOWN-DETAIL] Ticket #{ticket} | Profit: {profit:.2f} | Slot liberado em: {cooldown_until_str}")
                else:
                    self._log(f"❌ [CLOSE-ERROR] Ticket #{ticket} - NÃO foi possível identificar o slot! Cooldown não ativado.")
            except Exception as e:
                self._log(f"❌ [CLOSE-ERROR] Erro ao processar cooldown slot: {e}")
            
            # Remover posição da lista
            self.mock_positions.remove(pos)
            del self.position_stats[ticket]

def test_complete_cooldown_cycle():
    """🧪 Teste do ciclo completo de cooldown"""
    robot = MockTradingRobotV7()
    
    print("\n" + "="*80)
    print("🧪 TESTE COMPLETO: Ciclo Abertura → Fechamento → Cooldown → Nova Tentativa")
    print("="*80)
    
    # PASSO 1: Abrir posição no slot 1
    print("\n--- PASSO 1: Abrir posição no slot 1 ---")
    slot, wait = robot._allocate_entry_slot()
    robot._log(f"✅ PASSO 1: Slot alocado: {slot}")
    
    if slot is not None:
        robot.simulate_open_position(12345, slot)
        robot._log(f"📊 Posições ativas: {len(robot._get_robot_positions())}")
    
    # PASSO 2: Tentar abrir segunda posição
    print("\n--- PASSO 2: Tentar abrir segunda posição ---")
    slot2, wait2 = robot._allocate_entry_slot()
    robot._log(f"✅ PASSO 2: Slot alocado: {slot2}")
    
    if slot2 is not None:
        robot.simulate_open_position(67890, slot2)
        robot._log(f"📊 Posições ativas: {len(robot._get_robot_positions())}")
    
    # PASSO 3: Fechar primeira posição (ativar cooldown slot 1)
    print("\n--- PASSO 3: Fechar posição 12345 (slot 1) ---")
    robot.simulate_close_position(12345, -5.0)
    robot._log(f"📊 Posições ativas após fechamento: {len(robot._get_robot_positions())}")
    
    # PASSO 4: Tentar abrir nova posição (deve rejeitar por cooldown)
    print("\n--- PASSO 4: Tentar nova posição (deve ser rejeitada) ---")
    slot3, wait3 = robot._allocate_entry_slot()
    robot._log(f"✅ PASSO 4: Resultado: slot={slot3}, wait={wait3/60:.1f}min")
    
    # PASSO 5: Verificar _get_robot_positions ignora cooldown corretamente
    print("\n--- PASSO 5: Verificar filtro de posições ---")
    active_pos = robot._get_robot_positions()
    robot._log(f"✅ PASSO 5: {len(active_pos)} posições ativas (deveria ser 1)")
    
    print("\n" + "="*80)
    print("🧪 TESTE CONCLUÍDO")
    print("="*80)

if __name__ == "__main__":
    test_complete_cooldown_cycle()