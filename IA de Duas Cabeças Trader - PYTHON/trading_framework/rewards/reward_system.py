"""
Sistema de Recompensas para Trading PPO - VERSÃO SIMPLIFICADA
"""

import numpy as np
from typing import Dict, Optional, Tuple

# Configuração básica
CLEAN_REWARD_CONFIG = {
    "target_trades_per_day": 18,
    "sl_range": (10, 50),
    "tp_range": (15, 80),
    "min_trade_duration": 8,
    "max_reward": 25.0,
    "min_reward": -25.0,
    "hold_tolerance": 9999,
}

class CleanRewardCalculator:
    """Sistema de rewards simplificado"""
    
    def __init__(self, initial_balance: float = 1000.0, config: Optional[Dict] = None):
        self.initial_balance = initial_balance
        self.config = config or CLEAN_REWARD_CONFIG
        self.recent_actions = []
        self.consecutive_holds = 0
        self.trades_today = 0
        self.last_trade_step = -999
        
    def reset(self):
        """Reset do sistema de rewards para novo episódio"""
        self.recent_actions = []
        self.consecutive_holds = 0
        self.trades_today = 0
        self.last_trade_step = -999
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """Sistema simples de rewards"""
        reward = 0.0
        info = {"components": {}, "reward_type": "clean_system"}
        
        # Reward básico por ação
        if len(action) > 0:
            entry_decision = int(action[0])
            if entry_decision > 0:
                reward += 0.1  # Bônus por ação de entrada
        
        # Limites
        reward = max(-10.0, min(10.0, reward))
        
        return reward, info, False

def create_reward_system(system_type: str = "clean", 
                        initial_balance: float = 1000.0,
                        config: Optional[Dict] = None):
    """Cria o sistema de rewards"""
    return CleanRewardCalculator(initial_balance, config)

# Exportar configuração
ADAPTIVE_ANTI_OVERTRADING_CONFIG = CLEAN_REWARD_CONFIG
