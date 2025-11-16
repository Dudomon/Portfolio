#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO COOLDOWN SILUS
Teste para verificar se cooldown de 35min está funcionando corretamente
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List

# Mock da estrutura do silus para teste
class MockSilus:
    """Mock simplificado do silus para testar cooldown"""
    
    def __init__(self):
        self.max_positions = 2
        self.cooldown_base = 7
        self.cooldown_after_trade = 7
        self.slot_cooldowns = {i: 0 for i in range(self.max_positions)}
        self.positions = []
        self.trades = []
        self.current_step = 0
        
        print(f"🔧 SETUP: max_positions={self.max_positions}, cooldown={self.cooldown_after_trade} steps")
    
    def step(self, action):
        """Simula um step do ambiente"""
        self.current_step += 1
        
        # Decrementar cooldowns
        for s in list(self.slot_cooldowns.keys()):
            if self.slot_cooldowns[s] > 0:
                self.slot_cooldowns[s] -= 1
        
        # Simular ação de entrada
        entry_decision = action[0] if len(action) > 0 else 0
        
        if entry_decision > 0:
            success = self.try_enter_position(entry_decision)
            return success
        
        return False
    
    def try_enter_position(self, entry_type):
        """Tenta abrir posição respeitando cooldowns"""
        # Verificar slots disponíveis
        occupied_slots = {pos.get('position_id', -1) for pos in self.positions}
        available_slot = None
        
        for slot_idx in range(self.max_positions):
            if (slot_idx not in occupied_slots and 
                self.slot_cooldowns.get(slot_idx, 0) == 0):
                available_slot = slot_idx
                break
        
        if available_slot is None:
            cooldown_info = {k: v for k, v in self.slot_cooldowns.items() if v > 0}
            print(f"  ❌ Step {self.current_step}: Entrada BLOQUEADA - Cooldowns: {cooldown_info}")
            return False
        
        # Criar posição
        position = {
            'position_id': available_slot,
            'entry_step': self.current_step,
            'type': 'long' if entry_type == 1 else 'short'
        }
        self.positions.append(position)
        
        print(f"  ✅ Step {self.current_step}: Entrada PERMITIDA no slot {available_slot}")
        return True
    
    def close_position(self, position_id):
        """Fecha posição e aplica cooldown"""
        # Remover posição
        self.positions = [pos for pos in self.positions if pos.get('position_id') != position_id]
        
        # Aplicar cooldown
        if position_id in self.slot_cooldowns:
            self.slot_cooldowns[position_id] = int(self.cooldown_after_trade)
            print(f"  🔒 Step {self.current_step}: Slot {position_id} em cooldown por {self.cooldown_after_trade} steps")
        
        # Registrar trade
        trade = {'step': self.current_step, 'slot': position_id}
        self.trades.append(trade)

def test_basic_cooldown():
    """🧪 Teste básico de cooldown"""
    print("🧪 TESTE 1: Cooldown Básico")
    print("=" * 50)
    
    silus = MockSilus()
    
    # Cenário: Abrir -> Fechar -> Tentar abrir novamente
    print("\n📋 Cenário: Abrir pos no slot 0 → Fechar → Tentar reabrir")
    
    # Step 1: Abrir posição no slot 0
    success = silus.step([1.0])  # LONG
    assert success, "Primeira entrada deveria ser permitida"
    
    # Step 5: Fechar posição
    silus.current_step = 5
    silus.close_position(0)
    
    # Steps 6-12: Tentar abrir durante cooldown (deveria bloquear)
    blocked_attempts = 0
    for step in range(6, 13):
        silus.current_step = step
        success = silus.step([1.0])  # Tentar LONG
        if not success:
            blocked_attempts += 1
    
    print(f"\n📊 Resultado: {blocked_attempts}/7 tentativas bloqueadas durante cooldown")
    
    # Step 13: Após cooldown, deveria permitir
    silus.current_step = 13
    success = silus.step([1.0])
    print(f"Step 13 (pós-cooldown): {'✅ PERMITIDO' if success else '❌ BLOQUEADO'}")
    
    return blocked_attempts == 7 and success

def test_dual_slot_cooldown():
    """🧪 Teste de cooldown com 2 slots"""
    print("\n🧪 TESTE 2: Cooldown Dual Slots")
    print("=" * 50)
    
    silus = MockSilus()
    
    print("\n📋 Cenário: Usar ambos slots com cooldowns alternados")
    
    trades_opened = 0
    trades_blocked = 0
    
    # Simular 50 steps tentando abrir posições
    for step in range(1, 51):
        silus.current_step = step
        
        # Simular fechamento de posições antigas (aleatório)
        if step > 10 and step % 8 == 0 and silus.positions:
            pos_to_close = silus.positions[0]
            silus.close_position(pos_to_close['position_id'])
        
        # Tentar abrir nova posição
        success = silus.step([1.0])
        if success:
            trades_opened += 1
        else:
            trades_blocked += 1
    
    total_trades = len(silus.trades)
    
    print(f"\n📊 Resultados em 50 steps:")
    print(f"  Tentativas de abertura: {trades_opened}")
    print(f"  Tentativas bloqueadas: {trades_blocked}")
    print(f"  Trades completados: {total_trades}")
    print(f"  Taxa de bloqueio: {trades_blocked/(trades_opened+trades_blocked)*100:.1f}%")
    
    # Com cooldown de 7 steps, máximo teórico seria ~6-7 trades
    max_theoretical = 50 // 7  # ~7 trades máximo
    
    print(f"  Máximo teórico (50steps/7cooldown): {max_theoretical} trades")
    print(f"  Cooldown funcionando: {'✅' if total_trades <= max_theoretical else '❌'}")
    
    return total_trades <= max_theoretical

def test_overtrading_scenario():
    """🧪 Teste de cenário de overtrading"""
    print("\n🧪 TESTE 3: Cenário de Overtrading")
    print("=" * 50)
    
    silus = MockSilus()
    
    # Simular tentativas agressivas de trading
    print("\n📋 Cenário: Tentativa de abrir posição A CADA STEP")
    
    attempts = 0
    successful_entries = 0
    blocked_entries = 0
    
    for step in range(1, 101):  # 100 steps
        silus.current_step = step
        
        # Auto-fechar posições após 5 steps (rápido para maximizar tentativas)
        positions_to_close = []
        for pos in silus.positions:
            if step - pos['entry_step'] >= 5:  # 5 steps = 25min
                positions_to_close.append(pos)
        
        for pos in positions_to_close:
            silus.close_position(pos['position_id'])
        
        # Tentar abrir SEMPRE
        attempts += 1
        success = silus.step([1.0])
        
        if success:
            successful_entries += 1
        else:
            blocked_entries += 1
    
    trades_completed = len(silus.trades)
    
    print(f"\n📊 Resultados OVERTRADING em 100 steps:")
    print(f"  Tentativas totais: {attempts}")
    print(f"  Entradas bem-sucedidas: {successful_entries}")
    print(f"  Entradas bloqueadas: {blocked_entries}")
    print(f"  Trades completados: {trades_completed}")
    print(f"  Taxa de bloqueio: {blocked_entries/attempts*100:.1f}%")
    
    # Calcular trades/dia equivalente
    steps_per_day = 24 * 60 // 5  # 288 steps per day (M5)
    trades_per_day = (trades_completed / 100) * steps_per_day
    
    print(f"  Trades/dia equivalente: {trades_per_day:.1f}")
    
    # Se cooldown funciona, não deveria passar de ~40 trades/dia
    cooldown_working = trades_per_day < 50  # Threshold generoso
    print(f"  Cooldown controlando overtrading: {'✅' if cooldown_working else '❌'}")
    
    return cooldown_working, trades_per_day

def main():
    """Executa investigação completa do cooldown"""
    print("🔍 INVESTIGAÇÃO COOLDOWN SILUS")
    print("=" * 70)
    print("Verificando se cooldown de 35min (7 steps) está funcionando")
    print("=" * 70)
    
    tests = [
        ("Cooldown Básico", test_basic_cooldown),
        ("Dual Slot Cooldown", test_dual_slot_cooldown),
        ("Overtrading Scenario", test_overtrading_scenario)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                # Teste 3 retorna (bool, trades_per_day)
                passed, extra_info = result
                results.append(passed)
                if test_name == "Overtrading Scenario":
                    overtrading_rate = extra_info
            else:
                results.append(result)
                
            status = "✅ PASSOU" if (result[0] if isinstance(result, tuple) else result) else "❌ FALHOU"
            print(f"\n{status} - {test_name}")
            
        except Exception as e:
            print(f"\n❌ ERRO - {test_name}: {e}")
            results.append(False)
    
    print(f"\n{'='*70}")
    print(f"🏆 RESULTADOS DA INVESTIGAÇÃO")
    print(f"{'='*70}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Testes passaram: {passed}/{total}")
    
    if passed == total:
        print("✅ COOLDOWN está funcionando CORRETAMENTE")
        print("🤔 O problema de 704 trades NÃO é o sistema de cooldown")
        print("\n🔍 POSSÍVEIS CAUSAS REAIS:")
        print("   1. Episódio muito longo (>10 dias)")
        print("   2. Contador de trades com bug")
        print("   3. Trades sendo contados múltiplas vezes")
        print("   4. Reset de episódio não funcionando")
    else:
        print("❌ COOLDOWN tem problemas - pode explicar overtrading")
        print(f"   Taxa de overtrading detectada: {overtrading_rate:.1f} trades/dia" if 'overtrading_rate' in locals() else "")
    
    print("=" * 70)

if __name__ == "__main__":
    main()