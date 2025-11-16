"""
🚨 TESTE EMERGENCIAL - CONTROLE DE RISCO AGRESSIVO
Verifica se as penalidades de DD estão funcionando
"""

import sys
import numpy as np

sys.path.append(r'D:\Projeto')

from trading_framework.rewards.reward_system_simple import create_simple_reward_system

class MockEnv:
    def __init__(self, dd_pct=0.0):
        self.trades = []
        self.positions = []
        self.current_step = 100
        self.current_drawdown = dd_pct
        self.df = None

def test_dd_penalties():
    """Testar penalidades de drawdown"""
    print("🚨 [TESTE RISCO] Verificando penalidades de DD...")
    
    reward_system = create_simple_reward_system(500.0)
    
    dd_scenarios = [
        (2.0, "DD baixo (seguro)"),
        (8.0, "DD moderado"),
        (15.0, "DD alto"), 
        (25.0, "DD crítico"),
        (35.0, "DD catastrófico"),
        (45.0, "DD do inferno (atual)")
    ]
    
    print("   Cenário              | DD%   | Reward Total | Penalidade DD")
    print("   " + "-" * 65)
    
    action = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])  # HOLD
    old_state = {'trades_count': 0}
    
    for dd_pct, desc in dd_scenarios:
        env = MockEnv(dd_pct)
        reward, info, _ = reward_system.calculate_reward_and_info(env, action, old_state)
        
        # Tentar extrair componente de risco
        risk_component = 0.0
        for comp_name, comp_value in info['components'].items():
            if 'risk' in comp_name.lower() or 'drawdown' in comp_name.lower():
                risk_component += comp_value
        
        print(f"   {desc:19} | {dd_pct:4.1f}% | {reward:11.2f} | {risk_component:11.2f}")
    
    print(f"\n🎯 Com DD 45%, o reward deve ser MUITO negativo para forçar HOLD!")

if __name__ == "__main__":
    test_dd_penalties()