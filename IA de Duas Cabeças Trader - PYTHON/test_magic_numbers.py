#!/usr/bin/env python3
"""
🧪 TESTE: Verificar se magic numbers são únicos por instância
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Modelo PPO Trader"))

# Mock mínimo para testar magic numbers
class MockTradingRobot:
    def __init__(self, session_id=None):
        import os
        from datetime import datetime as _dt
        from uuid import uuid4 as _uuid4
        
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{_uuid4().hex[:8]}"
        
        # Magic number único por sessão
        import hashlib
        session_hash = int(hashlib.md5(str(self.session_id).encode()).hexdigest()[:6], 16)
        self.magic_number = 777000 + (session_hash % 888)
        
        print(f"Sessão: {self.session_id}")
        print(f"Magic Number: {self.magic_number}")

def test_magic_uniqueness():
    """🧪 Teste de unicidade dos magic numbers"""
    print("🧪 TESTE: Verificando unicidade dos magic numbers entre instâncias")
    print("="*60)
    
    robots = []
    magic_numbers = set()
    
    # Criar 5 instâncias simuladas
    for i in range(5):
        robot = MockTradingRobot(f"TEST_SESSION_{i}")
        robots.append(robot)
        magic_numbers.add(robot.magic_number)
        print()
    
    print("="*60)
    print(f"✅ RESULTADO: {len(robots)} instâncias criadas")
    print(f"✅ Magic Numbers únicos: {len(magic_numbers)}")
    
    if len(magic_numbers) == len(robots):
        print("🎉 SUCESSO: Todos os magic numbers são únicos!")
    else:
        print("❌ FALHA: Magic numbers duplicados detectados!")
        
    print(f"Range de magic numbers: {min(magic_numbers)} - {max(magic_numbers)}")

if __name__ == "__main__":
    test_magic_uniqueness()